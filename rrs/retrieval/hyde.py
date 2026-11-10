"""
HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Instead of embedding the raw question, the LLM generates a hypothetical
answer passage and that passage is embedded for retrieval.  The intuition is
that the embedding of a fluent, on-topic passage sits closer to the relevant
corpus chunks than the embedding of a terse question.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels", arXiv 2212.10496, 2022.
"""

from typing import List, Dict, Optional

from .dense import DenseRetriever


HYDE_PROMPT_TEMPLATE = """\
Please write a short passage (3-5 sentences) that would directly answer the
following question.  Write only the passage — no introductory phrases.

Question: {question}

Passage:"""


class HyDERetriever:
    """
    HyDE-augmented dense retriever.

    Parameters
    ----------
    llm           : LLM client with a .complete(prompt) -> str method.
    encoder_name  : str  Sentence-transformer model.
    top_k         : int  Number of results to return.
    n_hypothetical: int  Number of hypothetical docs to generate and average.
    """

    def __init__(
        self,
        llm,
        encoder_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        n_hypothetical: int = 1,
    ):
        self.llm = llm
        self.top_k = top_k
        self.n_hypothetical = n_hypothetical
        self._dense = DenseRetriever(encoder_name=encoder_name, top_k=top_k)
        self._encoder_name = encoder_name

    def index(self, corpus: List[str], metadata: Optional[List[Dict]] = None) -> None:
        self._dense.index(corpus, metadata)

    def _hypothetical_doc(self, question: str) -> str:
        prompt = HYDE_PROMPT_TEMPLATE.format(question=question)
        return self.llm.complete(prompt, max_tokens=256)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Generate hypothetical doc(s), embed, and retrieve."""
        import numpy as np
        k = top_k or self.top_k

        enc = self._dense._get_encoder()
        hyp_docs = [self._hypothetical_doc(query) for _ in range(self.n_hypothetical)]

        # Average the hypothetical embeddings (robust to multiple generations)
        hyp_vecs = enc.encode(hyp_docs, show_progress_bar=False, normalize_embeddings=True)
        avg_vec = hyp_vecs.mean(axis=0)
        avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-10)

        # Override the query vector directly in the FAISS index
        import faiss
        avg_vec_f32 = avg_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self._dense._index.search(avg_vec_f32, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append({
                "text": self._dense._corpus[idx],
                "score": float(score),
                "rank": rank,
                "doc_idx": idx,
                "metadata": self._dense._metadata[idx],
                "retriever": "hyde",
                "hypothetical_doc": hyp_docs[0],
            })
        return results
