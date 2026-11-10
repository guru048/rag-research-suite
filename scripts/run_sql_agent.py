"""
Text-to-SQL agent demo on an annotation platform database.

Creates an in-memory SQLite DB with annotators, tasks, and annotations tables,
then runs 8 natural language queries through the TextToSQLAgent.
No API key needed (MockLLM generates plausible SQL templates).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from rrs.sql_agent.agent import TextToSQLAgent
from rrs.llm import get_llm


QUESTIONS = [
    "How many annotators are there in total?",
    "Show me all completed tasks with their reward amounts.",
    "Which annotator has completed the most tasks?",
    "What is the average confidence score across all annotations?",
    "List all pending tasks ordered by reward descending.",
    "How many tasks does each category have?",
    "Show all annotations made by expert annotators.",
    "What is the total reward paid for completed NLP tasks?",
]


def format_rows(rows, max_rows=5):
    if not rows:
        return "  (no rows returned)"
    lines = []
    keys = list(rows[0].keys())
    header = " | ".join(f"{k:<18}" for k in keys)
    lines.append(f"  {header}")
    lines.append(f"  {'-' * len(header)}")
    for row in rows[:max_rows]:
        line = " | ".join(f"{str(v):<18}" for v in row.values())
        lines.append(f"  {line}")
    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def main():
    print("RAG Research Suite — Text-to-SQL Agent Demo\n")

    llm = get_llm()
    agent = TextToSQLAgent(llm=llm, db_path=":memory:", max_retries=3)

    print("Seeding annotation platform database...")
    agent.seed_demo_db()

    schema = agent._get_schema()
    print(f"Schema loaded: {schema.count('Table:')} tables\n")
    print(f"LLM: {llm.model}  (set OPENAI_API_KEY for real SQL generation)\n")

    successes = 0
    for i, question in enumerate(QUESTIONS, 1):
        print(f"{'='*60}")
        print(f"Q{i}: {question}")
        result = agent.query(question)
        print(f"\n  SQL ({result['attempts']} attempt{'s' if result['attempts'] > 1 else ''}):")
        for line in result["sql"].splitlines():
            print(f"    {line}")

        if result["error"]:
            print(f"\n  ERROR: {result['error']}")
        else:
            successes += 1
            print(f"\n  Result ({len(result['rows'])} row{'s' if len(result['rows']) != 1 else ''}):")
            print(format_rows(result["rows"]))
        print()

    print(f"{'='*60}")
    print(f"  Completed: {successes}/{len(QUESTIONS)} queries executed successfully")
    print(f"  Self-correction loop: up to {agent.max_retries} retry attempts per query")
    print(f"\n  Note: MockLLM generates template SQL. Real accuracy requires")
    print(f"  OPENAI_API_KEY set for GPT-3.5-turbo or better.\n")


if __name__ == "__main__":
    main()
