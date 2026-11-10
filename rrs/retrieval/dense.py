"""
Dense retrieval with FAISS and sentence-transformers.

Encodes the corpus with all-MiniLM-L6-v2 (384-dim) and stores vectors in a
flat L2 FAISS index.  At query time, encodes the question and returns the
top-k nearest neighbours by cosine similarity (vectors are L2-normalised so
L2 distance == 2 - 2·cos).

No LangChain — raw faiss-cpu + sentence-transformers.
"""

from typing import List, Dict, Optional
import numpy as np


class DenseRetriever:
    """
    FAISS-based dense retriever.

    Parameters
    ----------
    encoder_name : str   HuggingFace model name for the bi-encoder.
    top_k        : int   Default number of results to return.
    """

    def __init__(self, encoder_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
        self.encoder_name = encoder_name
        self.top_k = top_k
        self._encoder = None
        self._index = None
        self._corpus: List[str] = []
        self._metadata: List[Dict] = []

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def _encode(self, texts: List[str]) -> np.ndarray:
        enc = self._get_encoder()
        vecs = enc.encode(texts, show_progress_bar=False, batch_size=64, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def index(self, corpus: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Build FAISS index over corpus."""
        import faiss
        self._corpus = list(corpus)
        self._metadata = metadata or [{} for _ in corpus]
        vecs = self._encode(corpus)
        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # inner product == cosine for normed vecs
        self._index.add(vecs)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Return top-k results as list of dicts with keys:
        text, score, rank, doc_idx, metadata.
        """
        if self._index is None:
            raise RuntimeError("Call index() before retrieve()")
        k = top_k or self.top_k
        q_vec = self._encode([query])
        scores, indices = self._index.search(q_vec, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append({
                "text": self._corpus[idx],
                "score": float(score),
                "rank": rank,
                "doc_idx": idx,
                "metadata": self._metadata[idx],
                "retriever": "dense",
            })
        return results

    def save(self, path: str) -> None:
        import faiss, json, os
        faiss.write_index(self._index, path + ".faiss")
        with open(path + ".corpus.json", "w") as f:
            json.dump({"corpus": self._corpus, "metadata": self._metadata}, f)

    def load(self, path: str) -> None:
        import faiss, json
        self._index = faiss.read_index(path + ".faiss")
        with open(path + ".corpus.json") as f:
            d = json.load(f)
        self._corpus = d["corpus"]
        self._metadata = d["metadata"]
