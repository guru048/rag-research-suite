"""
Chain-of-Thought (CoT) prompt engineering.

Implements zero-shot CoT ("Let's think step by step", Wei et al. 2022) and
few-shot CoT with manually specified reasoning chains.

Reference: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in
Large Language Models", NeurIPS 2022.
"""

from typing import List, Dict, Optional


ZERO_SHOT_COT_SUFFIX = "\nLet's think step by step."

FEW_SHOT_COT_TEMPLATE = """\
{examples}

Question: {question}
{context_block}
Let's think step by step."""

CONTEXT_BLOCK = "Context:\n{context}\n"


class ChainOfThoughtPrompt:
    """
    Build Chain-of-Thought prompts for a question-answering task.

    Parameters
    ----------
    few_shot_examples : list of dicts {question, context, reasoning, answer}
    zero_shot         : bool  If True, use zero-shot CoT suffix only.
    system_prefix     : str   Optional system-level instruction prepended.
    """

    def __init__(
        self,
        few_shot_examples: Optional[List[Dict]] = None,
        zero_shot: bool = False,
        system_prefix: str = "",
    ):
        self.few_shot_examples = few_shot_examples or []
        self.zero_shot = zero_shot
        self.system_prefix = system_prefix

    def _format_example(self, ex: Dict) -> str:
        ctx = CONTEXT_BLOCK.format(context=ex["context"]) if ex.get("context") else ""
        return (
            f"Question: {ex['question']}\n"
            f"{ctx}"
            f"Reasoning: {ex['reasoning']}\n"
            f"Answer: {ex['answer']}"
        )

    def build(self, question: str, context: Optional[str] = None) -> str:
        """
        Build the full prompt string.

        Parameters
        ----------
        question : str   The user question.
        context  : str   Retrieved context (optional).
        """
        prefix = (self.system_prefix + "\n\n").lstrip("\n") if self.system_prefix else ""

        if self.zero_shot or not self.few_shot_examples:
            ctx_block = CONTEXT_BLOCK.format(context=context) if context else ""
            return (
                f"{prefix}"
                f"Question: {question}\n"
                f"{ctx_block}"
                f"{ZERO_SHOT_COT_SUFFIX.lstrip()}"
            )

        examples_str = "\n\n".join(self._format_example(e) for e in self.few_shot_examples)
        ctx_block = CONTEXT_BLOCK.format(context=context) if context else ""
        prompt = FEW_SHOT_COT_TEMPLATE.format(
            examples=examples_str,
            question=question,
            context_block=ctx_block,
        )
        return prefix + prompt

    def build_messages(self, question: str, context: Optional[str] = None) -> List[Dict]:
        """Return OpenAI-style messages list."""
        content = self.build(question, context)
        messages = []
        if self.system_prefix:
            messages.append({"role": "system", "content": self.system_prefix})
            # Remove system prefix from user content since it's in system role
            content = self.build(question, context).replace(self.system_prefix + "\n\n", "")
        messages.append({"role": "user", "content": content})
        return messages
