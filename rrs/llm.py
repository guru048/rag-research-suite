"""
LLM client abstraction with OpenAI backend and a mock fallback.

The mock LLM returns deterministic canned responses so every demo script
runs without an API key.  Set OPENAI_API_KEY in the environment to switch
to the real model automatically.
"""

import os
import re
from typing import List, Dict, Optional


class MockLLM:
    """
    Deterministic LLM that returns template-based answers.
    Used when no OPENAI_API_KEY is set, so demos run offline.
    """

    def __init__(self, model: str = "mock-gpt"):
        self.model = model
        self._call_count = 0

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self._call_count += 1
        prompt_lower = prompt.lower()

        # SQL generation
        if ("sql" in prompt_lower or "query" in prompt_lower) and any(k in prompt_lower for k in ("count", "select", "table", "show", "list", "how many", "average", "total", "which")):
            if "count" in prompt_lower:
                return "SELECT COUNT(*) FROM documents;"
            if "where" in prompt_lower or "filter" in prompt_lower:
                return "SELECT * FROM documents WHERE category = 'AI' LIMIT 10;"
            return "SELECT id, title, content FROM documents LIMIT 10;"

        # Hypothetical document for HyDE
        if "hypothetical" in prompt_lower or "passage" in prompt_lower:
            # Extract the question part
            lines = prompt.strip().split("\n")
            q = next((l for l in lines if "?" in l), "this topic")
            return (
                f"A relevant passage about {q.strip().rstrip('?')} would discuss "
                "the core concepts, methodologies, and findings from recent research. "
                "Experts in this area have shown that effective approaches involve "
                "careful analysis of the available evidence and systematic evaluation "
                "of multiple hypotheses before drawing conclusions."
            )

        # Faithfulness / claim extraction
        if "claim" in prompt_lower or "fact" in prompt_lower:
            return "CLAIM: The document discusses retrieval-augmented generation.\nCLAIM: Dense retrieval uses vector embeddings.\nCLAIM: BM25 is a sparse retrieval method."

        # Answer generation / QA
        if "answer" in prompt_lower or "question" in prompt_lower:
            return (
                "Based on the provided context, the answer involves the key concepts "
                "described in the retrieved passages. The evidence supports a conclusion "
                "that combines multiple factors including the methodological approach, "
                "the experimental setup, and the observed outcomes."
            )

        # Chain-of-thought trigger
        if "step" in prompt_lower or "think" in prompt_lower or "reason" in prompt_lower:
            return (
                "Step 1: Identify the key entities and relationships in the question.\n"
                "Step 2: Retrieve relevant context from the knowledge base.\n"
                "Step 3: Synthesize the retrieved information with the question.\n"
                "Step 4: Formulate a coherent and accurate answer.\n"
                "Therefore, the answer is derived from systematic reasoning over the context."
            )

        # Default
        return (
            "The retrieved context provides relevant information to address this query. "
            "Based on the available evidence, the most accurate response incorporates "
            "the key facts and relationships identified in the source documents."
        )

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        # Flatten message history into a single prompt for the mock
        combined = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self.complete(combined, max_tokens=max_tokens, temperature=temperature)


class OpenAILLM:
    """
    Thin wrapper around the OpenAI Chat Completions API.
    Requires: pip install openai  and  OPENAI_API_KEY set.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


def get_llm(model: Optional[str] = None) -> "MockLLM | OpenAILLM":
    """
    Return the best available LLM.

    Uses OpenAI when OPENAI_API_KEY is set; falls back to MockLLM so the
    project runs end-to-end without any API credentials.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM(model=model or "gpt-3.5-turbo")
    return MockLLM(model=model or "mock-gpt")
