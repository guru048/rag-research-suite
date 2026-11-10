"""
Recursive character-level chunking.

Mirrors the LangChain RecursiveCharacterTextSplitter approach but implemented
from scratch.  Tries separators in priority order (paragraph → sentence →
word → character) and recursively splits any chunk that still exceeds
`chunk_size`.  Produces semantically cleaner splits than fixed-size chunking
at the cost of slightly variable chunk lengths.

Reference: LangChain documentation, 2023.
"""

from typing import List


class RecursiveChunker:
    """
    Recursively split text using a priority hierarchy of separators.

    Parameters
    ----------
    chunk_size    : int   Target chunk size (characters).
    overlap       : int   Character overlap between consecutive chunks.
    separators    : list  Priority-ordered list of separator strings.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits back together up to chunk_size, with overlap."""
        chunks: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s)
            if current_len + s_len + (len(separator) if current_parts else 0) > self.chunk_size:
                if current_parts:
                    chunk = separator.join(current_parts)
                    chunks.append(chunk)
                    # keep tail for overlap
                    tail_len, tail_parts = 0, []
                    for p in reversed(current_parts):
                        if tail_len + len(p) > self.overlap:
                            break
                        tail_parts.insert(0, p)
                        tail_len += len(p) + len(separator)
                    current_parts = tail_parts
                    current_len = tail_len
            current_parts.append(s)
            current_len += s_len + (len(separator) if len(current_parts) > 1 else 0)

        if current_parts:
            chunks.append(separator.join(current_parts))
        return chunks

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first applicable separator."""
        if not text:
            return []

        # Find the first separator that actually splits this text
        separator = separators[-1]  # fall-back: single characters
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        raw_splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []
        result: List[str] = []

        for split in raw_splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    result.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                remaining_seps = separators[separators.index(separator) + 1:] if separator in separators[1:] else [""]
                result.extend(self._split(split, remaining_seps or [""]))

        if good_splits:
            result.extend(self._merge_splits(good_splits, separator))
        return result

    def split(self, text: str) -> List[str]:
        """Split a single document string into chunks."""
        return [c for c in self._split(text, self.separators) if c.strip()]

    def split_documents(self, documents: List[str]) -> List[dict]:
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            for chunk_idx, chunk in enumerate(self.split(doc)):
                all_chunks.append({
                    "text": chunk,
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "chunker": "recursive",
                })
        return all_chunks
