"""
Structured output parsing with Pydantic.

Wraps LLM calls so the output is validated against a Pydantic schema.
Implements a simple retry-with-correction loop: if parsing fails, the
error message is fed back to the LLM for a second attempt.

This pattern is widely used in production RAG systems to enforce reliable
JSON output for downstream processing.
"""

import json
import re
from typing import Type, TypeVar, Optional, Dict, Any

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Minimal stub so the module imports even without pydantic installed
    class BaseModel:
        pass
    class ValidationError(Exception):
        pass


T = TypeVar("T", bound="BaseModel")

JSON_INSTRUCTION = (
    "\nRespond with valid JSON only. Do not include markdown code fences, "
    "explanatory text, or any content outside the JSON object."
)

CORRECTION_PROMPT = """\
Your previous response could not be parsed as valid JSON.
Error: {error}

Please respond again with ONLY valid JSON matching this schema:
{schema}

Your corrected JSON response:"""


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract the first JSON object or array."""
    # Remove ```json ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find the first { ... } or [ ... ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start=start):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start: i + 1]
    return text


class StructuredOutputParser:
    """
    Parse LLM output into a validated Pydantic model.

    Parameters
    ----------
    schema_class  : Type[BaseModel]   Pydantic model to validate against.
    llm           : LLM client with .complete(prompt) method.
    max_retries   : int   Number of correction attempts on parse failure.
    """

    def __init__(self, schema_class: Type, llm, max_retries: int = 2):
        if not HAS_PYDANTIC:
            raise ImportError("pydantic is required for StructuredOutputParser. pip install pydantic")
        self.schema_class = schema_class
        self.llm = llm
        self.max_retries = max_retries

    def _schema_str(self) -> str:
        return json.dumps(self.schema_class.model_json_schema(), indent=2)

    def parse(self, prompt: str) -> Any:
        """
        Run the LLM with the prompt and return a validated schema instance.

        Raises
        ------
        ValueError  if parsing still fails after max_retries corrections.
        """
        full_prompt = prompt + JSON_INSTRUCTION
        raw = self.llm.complete(full_prompt, max_tokens=512)

        for attempt in range(self.max_retries + 1):
            try:
                json_str = _extract_json(raw)
                data = json.loads(json_str)
                return self.schema_class(**data)
            except (ValidationError, json.JSONDecodeError, TypeError) as err:
                if attempt == self.max_retries:
                    raise ValueError(
                        f"Failed to parse structured output after {self.max_retries + 1} attempts. "
                        f"Last error: {err}\nLast response: {raw}"
                    )
                correction_prompt = CORRECTION_PROMPT.format(
                    error=str(err), schema=self._schema_str()
                )
                raw = self.llm.complete(correction_prompt, max_tokens=512)

    def parse_raw(self, prompt: str) -> Dict:
        """Like parse() but returns the raw dict instead of a model instance."""
        instance = self.parse(prompt)
        if hasattr(instance, "model_dump"):
            return instance.model_dump()
        return dict(instance)
