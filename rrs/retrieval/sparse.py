"""
BM25 sparse retrieval (Robertson & Zaragoza, 2009).

Uses rank_bm25 (Okapi BM25Plus variant) for term-frequency weighted
retrieval.  No embeddings required — works out of the box and serves as the
sparse leg of the hybrid retriever.

pip install rank-bm25
"""

from typing import List, Dict, Optional
import re


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Retriever:
    """
    BM25 sparse retriever backed by rank_bm25.

    Parameters
    ----------
    top_k : int   Default number of results to return.
    b     : float BM25 length normalisation parameter (default 0.75).
    k1    : float BM25 term frequency saturation parameter (default 1.5).
    """

    def __init__(self, top_k: int = 5, b: float = 0.75, k1: float = 1.5):
        self.top_k = top_k
        self.b = b
        self.k1 = k1
        self._bm25 = None
        self._corpus: List[str] = []
        self._metadata: List[Dict] = []

    def index(self, corpus: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Tokenise corpus and build BM25 index."""
        from rank_bm25 import BM25Plus
        self._corpus = list(corpus)
        self._metadata = metadata or [{} for _ in corpus]
        tokenised = [_tokenize(doc) for doc in corpus]
        self._bm25 = BM25Plus(tokenised, b=self.b, k1=self.k1)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Return top-k results sorted by BM25 score."""
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve()")
        k = top_k or self.top_k
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for rank, idx in enumerate(ranked_indices):
            results.append({
                "text": self._corpus[idx],
                "score": float(scores[idx]),
                "rank": rank,
                "doc_idx": idx,
                "metadata": self._metadata[idx],
                "retriever": "bm25",
            })
        return results

    @property
    def vocab_size(self) -> int:
        if self._bm25 is None:
            return 0
        return len(self._bm25.idf)
