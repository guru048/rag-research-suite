"""
RAGAS-style evaluator combining multiple RAG quality metrics.

Implements the four core metrics from Es et al. (arXiv 2309.15217):

  - faithfulness      : grounding of the answer in context (no hallucination)
  - answer_relevancy  : how directly the answer addresses the question
  - context_precision : fraction of retrieved chunks that are actually useful
  - context_recall    : fraction of ground-truth information covered by context

Context precision and recall are approximated with embedding similarity when
ground-truth answers are available.

Reference: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented
Generation", arXiv 2309.15217, 2023.
"""

from typing import List, Dict, Optional
import numpy as np

from .faithfulness import FaithfulnessScorer
from .relevancy import AnswerRelevancyScorer


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class RAGASEvaluator:
    """
    Evaluate a full RAG pipeline on a set of QA examples.

    Parameters
    ----------
    llm              : LLM client with .complete(prompt).
    encoder_name     : str  Sentence-transformer model for embedding-based metrics.
    precision_threshold : float  Sim threshold for a context chunk to count as relevant.
    recall_threshold    : float  Sim threshold for a GT sentence to be covered.
    """

    def __init__(
        self,
        llm,
        encoder_name: str = "all-MiniLM-L6-v2",
        precision_threshold: float = 0.55,
        recall_threshold: float = 0.55,
    ):
        self.llm = llm
        self.encoder_name = encoder_name
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold
        self._encoder = None
        self._faithfulness = FaithfulnessScorer(llm, encoder_name)
        self._relevancy = AnswerRelevancyScorer(llm, encoder_name)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def _context_precision(self, question: str, contexts: List[str]) -> float:
        """Fraction of retrieved chunks relevant to the question."""
        if not contexts:
            return 0.0
        enc = self._get_encoder()
        all_texts = [question] + contexts
        vecs = enc.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)
        q_vec = vecs[0]
        ctx_vecs = vecs[1:]
        relevant = sum(
            1 for cv in ctx_vecs if _cosine_sim(q_vec, cv) >= self.precision_threshold
        )
        return relevant / len(contexts)

    def _context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        """
        Fraction of GT sentences that can be attributed to the retrieved context.
        """
        import re
        gt_sentences = [s.strip() for s in re.split(r"[.!?]", ground_truth) if s.strip()]
        if not gt_sentences or not contexts:
            return 0.0
        enc = self._get_encoder()
        combined_context = " ".join(contexts)
        ctx_sentences = [s.strip() for s in re.split(r"[.!?]", combined_context) if s.strip()] or [combined_context]

        all_texts = gt_sentences + ctx_sentences
        vecs = enc.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)
        gt_vecs = vecs[: len(gt_sentences)]
        ctx_vecs = vecs[len(gt_sentences):]

        covered = 0
        for gv in gt_vecs:
            max_sim = max((_cosine_sim(gv, cv) for cv in ctx_vecs), default=0.0)
            if max_sim >= self.recall_threshold:
                covered += 1
        return covered / len(gt_sentences)

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a single QA sample.

        Returns dict with faithfulness, answer_relevancy, context_precision,
        and (if ground_truth is provided) context_recall.
        """
        faith_result = self._faithfulness.score(answer, " ".join(contexts))
        rel_result = self._relevancy.score(question, answer)
        ctx_precision = self._context_precision(question, contexts)

        result = {
            "faithfulness": faith_result["faithfulness"],
            "answer_relevancy": rel_result["answer_relevancy"],
            "context_precision": round(ctx_precision, 4),
            "faithfulness_detail": faith_result,
            "relevancy_detail": rel_result,
        }

        if ground_truth is not None:
            result["context_recall"] = round(
                self._context_recall(ground_truth, contexts), 4
            )
            ragas_score_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        else:
            ragas_score_keys = ["faithfulness", "answer_relevancy", "context_precision"]

        result["ragas_score"] = round(
            float(np.mean([result[k] for k in ragas_score_keys])), 4
        )
        return result

    def evaluate_dataset(
        self,
        samples: List[Dict],
    ) -> Dict:
        """
        Evaluate a list of samples.

        Each sample dict: {question, answer, contexts, ground_truth (optional)}.
        Returns per-sample results plus aggregate mean metrics.
        """
        per_sample = []
        for s in samples:
            result = self.evaluate_sample(
                question=s["question"],
                answer=s["answer"],
                contexts=s["contexts"],
                ground_truth=s.get("ground_truth"),
            )
            per_sample.append(result)

        metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "ragas_score"]
        if any("context_recall" in r for r in per_sample):
            metric_keys.append("context_recall")

        aggregate = {
            k: round(float(np.mean([r[k] for r in per_sample if k in r])), 4)
            for k in metric_keys
        }

        return {"aggregate": aggregate, "per_sample": per_sample}
