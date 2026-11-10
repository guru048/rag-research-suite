"""
Full RAG pipeline: index → retrieve → rerank → generate → evaluate.

Combines all rrs modules into a single end-to-end callable.
Supports swappable chunkers, retrievers, rerankers, and LLMs.
"""

from typing import List, Dict, Optional

from .chunking.fixed import FixedChunker
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import CrossEncoderReranker
from .prompts.chain_of_thought import ChainOfThoughtPrompt


ANSWER_PROMPT_TEMPLATE = """\
Answer the following question using ONLY the provided context.
If the context does not contain enough information, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Parameters
    ----------
    llm         : LLM client (get_llm() from rrs.llm).
    chunker     : text splitter instance (default: FixedChunker).
    retriever   : retriever instance (default: HybridRetriever).
    reranker    : optional reranker (default: CrossEncoderReranker; None to skip).
    top_k       : int  Final number of context chunks to pass to the LLM.
    use_cot     : bool  Whether to use Chain-of-Thought prompting.
    """

    def __init__(
        self,
        llm,
        chunker=None,
        retriever=None,
        reranker=None,
        top_k: int = 3,
        use_cot: bool = False,
    ):
        self.llm = llm
        self.chunker = chunker or FixedChunker(chunk_size=200, overlap=20)
        self.retriever = retriever or HybridRetriever(top_k=top_k * 4)
        self.reranker = reranker  # None disables reranking
        self.top_k = top_k
        self.use_cot = use_cot
        self._cot = ChainOfThoughtPrompt(zero_shot=True) if use_cot else None
        self._indexed = False

    # ------------------------------------------------------------------ #

    def index(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> int:
        """Chunk documents and build the retriever index."""
        all_chunks = []
        all_meta = []
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunker.split(doc)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                base_meta = (metadata[doc_idx] if metadata else {}).copy()
                base_meta.update({"source_doc": doc_idx, "chunk_idx": chunk_idx})
                all_meta.append(base_meta)

        self.retriever.index(all_chunks, all_meta)
        self._indexed = True
        self._chunk_count = len(all_chunks)
        return len(all_chunks)

    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve + optionally rerank top chunks for a query."""
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve()")
        candidates = self.retriever.retrieve(query)

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=self.top_k)
        else:
            candidates = candidates[: self.top_k]

        return candidates

    # ------------------------------------------------------------------ #

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """Generate an answer from retrieved context chunks."""
        context = "\n\n".join(context_chunks)
        if self.use_cot and self._cot:
            prompt = self._cot.build(query, context=context)
        else:
            prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=query)
        return self.llm.complete(prompt, max_tokens=512)

    # ------------------------------------------------------------------ #

    def run(self, query: str) -> Dict:
        """
        End-to-end pipeline: retrieve → generate.

        Returns
        -------
        dict with query, retrieved_chunks, context, answer, metadata.
        """
        retrieved = self.retrieve(query)
        chunks = [r["text"] for r in retrieved]
        answer = self.generate(query, chunks)
        return {
            "query": query,
            "retrieved_chunks": retrieved,
            "context": "\n\n".join(chunks),
            "answer": answer,
            "n_chunks_used": len(chunks),
        }

    # ------------------------------------------------------------------ #

    @property
    def chunk_count(self) -> int:
        return getattr(self, "_chunk_count", 0)
