"""
Answer relevancy scoring for RAG systems.

Implements the RAGAS answer relevancy metric (Es et al., 2023):
1. Use an LLM to generate n questions that the given answer would address.
2. Embed those generated questions and the original question.
3. Answer relevancy = mean cosine similarity between the generated questions
   and the original question.

High score → the answer directly addresses the question.
Low score  → the answer is off-topic or vague.

Reference: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented
Generation", arXiv 2309.15217, 2023.
"""

import re
from typing import List, Dict, Optional
import numpy as np


QUESTION_GEN_PROMPT = """\
Given the following answer, generate {n} questions that this answer would
directly and completely address. Write each question on a new line starting
with "Q:".

Answer: {answer}

Questions:"""


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class AnswerRelevancyScorer:
    """
    Compute answer relevancy using reverse question generation.

    Parameters
    ----------
    llm          : LLM client with .complete(prompt) method.
    encoder_name : str  Sentence-transformer for embedding questions.
    n_questions  : int  Number of questions to generate (more = more stable score).
    """

    def __init__(self, llm, encoder_name: str = "all-MiniLM-L6-v2", n_questions: int = 3):
        self.llm = llm
        self.encoder_name = encoder_name
        self.n_questions = n_questions
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def _generate_questions(self, answer: str) -> List[str]:
        prompt = QUESTION_GEN_PROMPT.format(n=self.n_questions, answer=answer)
        raw = self.llm.complete(prompt, max_tokens=256)
        questions = re.findall(r"Q:\s*(.+)", raw, re.IGNORECASE)
        if not questions:
            # Fallback: use the whole response as one question
            questions = [raw.strip().split("\n")[0]]
        return questions[: self.n_questions]

    def score(self, question: str, answer: str) -> Dict:
        """
        Returns dict with:
          answer_relevancy  float  [0, 1] (mean cosine to original question)
          generated_questions list[str]
          similarities      list[float]
        """
        enc = self._get_encoder()
        gen_questions = self._generate_questions(answer)

        all_texts = [question] + gen_questions
        all_vecs = enc.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)
        q_vec = all_vecs[0]
        gen_vecs = all_vecs[1:]

        sims = [_cosine_sim(q_vec, gv) for gv in gen_vecs]
        relevancy = float(np.mean(sims)) if sims else 0.0

        return {
            "answer_relevancy": round(relevancy, 4),
            "generated_questions": gen_questions,
            "similarities": [round(s, 4) for s in sims],
        }
