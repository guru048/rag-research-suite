"""
Fixed-size chunking with configurable overlap.

Splits text into chunks of exactly `chunk_size` tokens (approximated as
words), with `overlap` tokens of context carried forward from the previous
chunk.  Simple, fast, and the standard baseline for RAG benchmarks.
"""

from typing import List


class FixedChunker:
    """
    Split text into fixed-size word-token chunks with overlap.

    Parameters
    ----------
    chunk_size : int
        Target chunk size in approximate word-tokens.
    overlap    : int
        Number of tokens to repeat at the start of the next chunk
        (maintains local context across boundaries).
    """

    def __init__(self, chunk_size: int = 256, overlap: int = 32):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        """Return list of chunk strings."""
        tokens = text.split()
        if not tokens:
            return []

        chunks, start = [], 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)
            # advance by (chunk_size - overlap) so the next chunk re-reads
            # the last `overlap` tokens of this one
            start += self.chunk_size - self.overlap

        return chunks

    def split_documents(self, documents: List[str]) -> List[dict]:
        """
        Split a list of document strings into chunks, preserving doc index.

        Returns list of dicts: {text, doc_idx, chunk_idx}
        """
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            for chunk_idx, chunk in enumerate(self.split(doc)):
                all_chunks.append({
                    "text": chunk,
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "chunker": "fixed",
                })
        return all_chunks
