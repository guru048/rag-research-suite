"""
Semantic chunking (Kamradt 2023).

Splits a document at points where consecutive sentence embeddings show a
large cosine distance — i.e., where the topic changes.  Produces chunks
that are semantically cohesive rather than arbitrarily sized.

Algorithm
---------
1. Split text into sentences.
2. Embed each sentence with a lightweight encoder.
3. Compute cosine distance between adjacent sentence embeddings.
4. Identify breakpoints where distance exceeds `threshold` × std above mean.
5. Merge sentences within each segment into one chunk.

Reference: Greg Kamradt, "5 Levels of Text Splitting", 2023.
"""

from typing import List
import re
import numpy as np


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a, b))


class SemanticChunker:
    """
    Split text at semantic boundaries detected via embedding distance spikes.

    Parameters
    ----------
    encoder        : sentence_transformers.SentenceTransformer (or compatible)
    threshold_z    : float   Breakpoint at mean + threshold_z * std of distances.
    min_sentences  : int     Minimum sentences per chunk to avoid micro-chunks.
    """

    def __init__(self, encoder=None, threshold_z: float = 1.0, min_sentences: int = 2):
        self.encoder = encoder
        self.threshold_z = threshold_z
        self.min_sentences = min_sentences

    def _lazy_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self.encoder

    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        """Naive sentence splitter (handles ., !, ?)."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def split(self, text: str) -> List[str]:
        sentences = self._sentence_split(text)
        if len(sentences) <= self.min_sentences:
            return [text] if text.strip() else []

        enc = self._lazy_encoder()
        embeddings = enc.encode(sentences, show_progress_bar=False, batch_size=64)

        # Cosine distances between consecutive sentence embeddings
        distances = [
            _cosine_distance(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        if not distances:
            return [text]

        d_arr = np.array(distances)
        threshold = d_arr.mean() + self.threshold_z * d_arr.std()

        # Build chunk boundaries
        chunks, current = [], []
        for i, sentence in enumerate(sentences):
            current.append(sentence)
            if i < len(distances) and distances[i] > threshold:
                if len(current) >= self.min_sentences:
                    chunks.append(" ".join(current))
                    current = []
        if current:
            chunks.append(" ".join(current))

        return chunks

    def split_documents(self, documents: List[str]) -> List[dict]:
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            for chunk_idx, chunk in enumerate(self.split(doc)):
                all_chunks.append({
                    "text": chunk,
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "chunker": "semantic",
                })
        return all_chunks
