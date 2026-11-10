"""
Reranking: cross-encoder reranking and Maximal Marginal Relevance (MMR).

CrossEncoderReranker
--------------------
Uses a sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2) to score
each (query, passage) pair.  Cross-encoders attend to both texts jointly and
are significantly more accurate than bi-encoders for relevance scoring,
at the cost of O(n) inference passes.

MMRReranker
-----------
Maximal Marginal Relevance (Carbonell & Goldstein, 1998) balances relevance
with diversity: at each step it picks the document that maximises

    MMR(d) = λ · sim(q, d) − (1−λ) · max_{d' ∈ S} sim(d, d')

where S is the already-selected set.  λ=1 gives pure relevance; λ=0 gives
pure diversity.
"""

from typing import List, Dict, Optional
import numpy as np


class CrossEncoderReranker:
    """
    Rerank a candidate list with a cross-encoder model.

    Parameters
    ----------
    model_name : str  HuggingFace CrossEncoder model (default: ms-marco-MiniLM-L-6-v2).
    top_k      : int  How many results to return after reranking.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = None, top_k: int = 5):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.top_k = top_k
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """
        Score each candidate against the query and return sorted list.

        Input `candidates` must have a 'text' key.
        """
        k = top_k or self.top_k
        model = self._get_model()
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (cand, score) in enumerate(ranked[:k]):
            entry = dict(cand)
            entry["cross_encoder_score"] = float(score)
            entry["rank"] = rank
            entry["retriever"] = "cross_encoder_reranked"
            results.append(entry)
        return results


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class MMRReranker:
    """
    Maximal Marginal Relevance reranking for result diversity.

    Parameters
    ----------
    encoder_name : str    Sentence-transformer for computing embeddings.
    lambda_param : float  Trade-off between relevance (1) and diversity (0).
    top_k        : int    Number of documents to select.
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        lambda_param: float = 0.5,
        top_k: int = 5,
    ):
        self.encoder_name = encoder_name
        self.lambda_param = lambda_param
        self.top_k = top_k
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def rerank(self, query: str, candidates: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        k = min(top_k or self.top_k, len(candidates))
        enc = self._get_encoder()

        texts = [c["text"] for c in candidates]
        all_texts = [query] + texts
        all_vecs = enc.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)
        q_vec = all_vecs[0]
        doc_vecs = all_vecs[1:]

        # Relevance scores: cosine(query, doc)
        rel_scores = np.array([_cosine_sim(q_vec, v) for v in doc_vecs])

        selected_indices: List[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(k):
            if not remaining:
                break
            if not selected_indices:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: rel_scores[i])
            else:
                # MMR score
                def mmr_score(i):
                    rel = self.lambda_param * rel_scores[i]
                    max_sim = max(
                        _cosine_sim(doc_vecs[i], doc_vecs[j]) for j in selected_indices
                    )
                    div = (1 - self.lambda_param) * max_sim
                    return rel - div

                best = max(remaining, key=mmr_score)
            selected_indices.append(best)
            remaining.remove(best)

        results = []
        for rank, idx in enumerate(selected_indices):
            entry = dict(candidates[idx])
            entry["mmr_relevance"] = float(rel_scores[idx])
            entry["rank"] = rank
            entry["retriever"] = "mmr_reranked"
            results.append(entry)
        return results
