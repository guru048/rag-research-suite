"""
Faithfulness scoring for RAG systems.

Measures whether the generated answer is grounded in the retrieved context
(no hallucination).  Implements the claim-based approach from RAGAS
(Es et al., 2023, arXiv 2309.15217):

1. Use an LLM to decompose the answer into atomic claims.
2. For each claim, check whether it is entailed by the context via embedding
   similarity between the claim and context sentences.
3. Faithfulness = |supported claims| / |total claims|.
"""

import re
from typing import List, Dict, Optional
import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


CLAIM_PROMPT = """\
Decompose the following answer into a list of atomic factual claims.
Write each claim on a new line starting with "CLAIM:".
Answer only with the claims, nothing else.

Answer: {answer}

Claims:"""


class FaithfulnessScorer:
    """
    Compute faithfulness score using claim extraction + entailment check.

    Parameters
    ----------
    llm           : LLM client with .complete(prompt) method.
    encoder_name  : str   Sentence-transformer for entailment similarity.
    sim_threshold : float Claims with max context similarity >= this are supported.
    """

    def __init__(self, llm, encoder_name: str = "all-MiniLM-L6-v2", sim_threshold: float = 0.60):
        self.llm = llm
        self.encoder_name = encoder_name
        self.sim_threshold = sim_threshold
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def _extract_claims(self, answer: str) -> List[str]:
        prompt = CLAIM_PROMPT.format(answer=answer)
        raw = self.llm.complete(prompt, max_tokens=256)
        claims = re.findall(r"CLAIM:\s*(.+)", raw, re.IGNORECASE)
        if not claims:
            # Fallback: treat each sentence as a claim
            claims = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
        return claims

    def score(self, answer: str, context: str) -> Dict:
        """
        Returns dict with:
          faithfulness   float  [0, 1]
          n_claims       int
          n_supported    int
          claims         list of {text, supported, max_sim}
        """
        claims = self._extract_claims(answer)
        if not claims:
            return {"faithfulness": 1.0, "n_claims": 0, "n_supported": 0, "claims": []}

        enc = self._get_encoder()

        # Split context into sentences for fine-grained support check
        ctx_sentences = [s.strip() for s in re.split(r"[.!?]", context) if s.strip()] or [context]

        all_texts = claims + ctx_sentences
        all_vecs = enc.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)
        claim_vecs = all_vecs[: len(claims)]
        ctx_vecs = all_vecs[len(claims):]

        claim_results = []
        n_supported = 0
        for i, (claim, c_vec) in enumerate(zip(claims, claim_vecs)):
            sims = [_cosine_sim(c_vec, ctx_v) for ctx_v in ctx_vecs]
            max_sim = max(sims) if sims else 0.0
            supported = max_sim >= self.sim_threshold
            if supported:
                n_supported += 1
            claim_results.append({
                "text": claim,
                "supported": supported,
                "max_sim": round(max_sim, 4),
            })

        faithfulness = n_supported / len(claims)
        return {
            "faithfulness": round(faithfulness, 4),
            "n_claims": len(claims),
            "n_supported": n_supported,
            "claims": claim_results,
        }
