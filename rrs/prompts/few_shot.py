"""
Dynamic few-shot example selection via embedding similarity.

Retrieves the most semantically similar examples to the current query from
an example pool, rather than using a fixed static set.  This is especially
helpful for diverse QA tasks where different question types benefit from
different exemplars.

The approach is documented in several RAG and prompt engineering surveys
(e.g., Liu et al., 2022, "What Makes Good In-Context Examples for GPT-3?").
"""

from typing import List, Dict, Optional
import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class FewShotPromptBuilder:
    """
    Dynamically select few-shot examples by similarity to the input query.

    Parameters
    ----------
    examples     : list of dicts {question, answer, [context], [reasoning]}
    encoder_name : str   Sentence-transformer model for similarity lookup.
    n_shots      : int   Number of examples to include in each prompt.
    template     : str   Prompt template with {examples} and {question} placeholders.
    """

    DEFAULT_TEMPLATE = (
        "{examples}\n\nQuestion: {question}\n{context_block}Answer:"
    )

    def __init__(
        self,
        examples: List[Dict],
        encoder_name: str = "all-MiniLM-L6-v2",
        n_shots: int = 3,
        template: Optional[str] = None,
    ):
        self.examples = examples
        self.encoder_name = encoder_name
        self.n_shots = n_shots
        self.template = template or self.DEFAULT_TEMPLATE
        self._encoder = None
        self._example_vecs: Optional[np.ndarray] = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def _get_example_vecs(self) -> np.ndarray:
        if self._example_vecs is None:
            enc = self._get_encoder()
            questions = [e["question"] for e in self.examples]
            self._example_vecs = enc.encode(
                questions, show_progress_bar=False, normalize_embeddings=True
            )
        return self._example_vecs

    def select_examples(self, query: str, n: Optional[int] = None) -> List[Dict]:
        """Return the n most similar examples to the query."""
        k = n or self.n_shots
        enc = self._get_encoder()
        q_vec = enc.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
        ex_vecs = self._get_example_vecs()
        sims = [_cosine_sim(q_vec, ev) for ev in ex_vecs]
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [self.examples[i] for i in top_indices]

    @staticmethod
    def _format_example(ex: Dict) -> str:
        parts = [f"Question: {ex['question']}"]
        if ex.get("context"):
            parts.append(f"Context: {ex['context']}")
        if ex.get("reasoning"):
            parts.append(f"Reasoning: {ex['reasoning']}")
        parts.append(f"Answer: {ex['answer']}")
        return "\n".join(parts)

    def build(self, query: str, context: Optional[str] = None, n: Optional[int] = None) -> str:
        """Build the prompt with dynamically selected examples."""
        selected = self.select_examples(query, n)
        examples_str = "\n\n".join(self._format_example(e) for e in selected)
        ctx_block = f"Context: {context}\n" if context else ""
        return self.template.format(
            examples=examples_str,
            question=query,
            context_block=ctx_block,
        )

    def add_example(self, example: Dict) -> None:
        """Add a new example and invalidate the embedding cache."""
        self.examples.append(example)
        self._example_vecs = None  # force re-encode on next call
