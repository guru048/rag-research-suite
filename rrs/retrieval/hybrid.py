"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines dense (FAISS) and sparse (BM25) result lists using RRF
(Cormack et al., 2009).  RRF is robust to score-scale differences because
it uses only rank positions, not raw scores:

    RRF(d) = Σ_r  1 / (k + r(d))

where r(d) is the rank of document d in result list r and k=60 is a
smoothing constant.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods", SIGIR 2009.
"""

from typing import List, Dict, Optional

from .dense import DenseRetriever
from .sparse import BM25Retriever


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
) -> List[Dict]:
    """
    Merge multiple ranked result lists with RRF.

    Parameters
    ----------
    result_lists : list of ranked result lists (each result has 'doc_idx').
    k            : RRF smoothing constant (default 60).

    Returns sorted list of dicts with fused 'rrf_score'.
    """
    scores: Dict[int, float] = {}
    doc_map: Dict[int, Dict] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            idx = doc["doc_idx"]
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            if idx not in doc_map:
                doc_map[idx] = doc

    fused = []
    for idx, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(doc_map[idx])
        entry["rrf_score"] = rrf_score
        entry["retriever"] = "hybrid_rrf"
        fused.append(entry)

    for rank, entry in enumerate(fused):
        entry["rank"] = rank
    return fused


class HybridRetriever:
    """
    Hybrid dense+sparse retriever using Reciprocal Rank Fusion.

    Parameters
    ----------
    encoder_name  : str  Sentence-transformer model for the dense leg.
    top_k         : int  Number of final results to return.
    dense_top_k   : int  Candidates fetched from each retriever before fusion.
    sparse_top_k  : int  (defaults to dense_top_k if not set)
    rrf_k         : int  RRF smoothing constant.
    """

    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        dense_top_k: int = 20,
        sparse_top_k: Optional[int] = None,
        rrf_k: int = 60,
    ):
        self.top_k = top_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k or dense_top_k
        self.rrf_k = rrf_k
        self.dense = DenseRetriever(encoder_name=encoder_name, top_k=dense_top_k)
        self.sparse = BM25Retriever(top_k=sparse_top_k or dense_top_k)

    def index(self, corpus: List[str], metadata: Optional[List[Dict]] = None) -> None:
        self.dense.index(corpus, metadata)
        self.sparse.index(corpus, metadata)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        k = top_k or self.top_k
        dense_results = self.dense.retrieve(query, top_k=self.dense_top_k)
        sparse_results = self.sparse.retrieve(query, top_k=self.sparse_top_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)
        return fused[:k]
