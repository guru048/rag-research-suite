"""
Text-to-SQL agent with SQLite backend and self-correction loop.

Flow
----
1. Schema discovery: introspect the SQLite database to get table/column info.
2. SQL generation: prompt the LLM with the schema + user question.
3. Execution: run the generated SQL against SQLite.
4. Self-correction: if execution fails (syntax error, bad column), feed the
   error back to the LLM for up to `max_retries` correction passes.
5. Result formatting: return rows as list-of-dicts.

This pattern is used in real data annotation platforms where annotators or
reviewers query datasets in natural language without knowing SQL.
"""

import sqlite3
import json
import re
from typing import List, Dict, Optional, Any, Tuple


SCHEMA_PROMPT = """\
You are an expert SQL assistant. Given the following SQLite database schema,
write a valid SQL query that answers the user's question.

Database schema:
{schema}

User question: {question}

Rules:
- Write only the SQL query, nothing else.
- Use only table and column names that appear in the schema above.
- End the query with a semicolon.

SQL:"""

CORRECTION_PROMPT = """\
The following SQL query produced an error. Please fix it.

Question: {question}
Schema: {schema}

Broken SQL:
{sql}

Error: {error}

Write only the corrected SQL query:"""


def _extract_sql(text: str) -> str:
    """Strip markdown fences and extract the SQL statement."""
    text = re.sub(r"```(?:sql)?\s*", "", text).strip().rstrip("`").strip()
    # Remove trailing explanations after the semicolon
    match = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH).*?;", text, re.DOTALL | re.IGNORECASE)
    return match.group(0).strip() if match else text.strip()


class TextToSQLAgent:
    """
    Convert natural language questions to SQL and execute them.

    Parameters
    ----------
    llm         : LLM client with .complete(prompt) method.
    db_path     : str   Path to the SQLite database file.
                        Use ':memory:' for an in-process demo database.
    max_retries : int   Self-correction iterations on SQL error.
    """

    def __init__(self, llm, db_path: str = ":memory:", max_retries: int = 3):
        self.llm = llm
        self.db_path = db_path
        self.max_retries = max_retries
        self._schema_cache: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Schema introspection                                                 #
    # ------------------------------------------------------------------ #

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_schema(self) -> str:
        if self._schema_cache:
            return self._schema_cache
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            parts = []
            for table in tables:
                info = conn.execute(f"PRAGMA table_info({table})").fetchall()
                cols = ", ".join(f"{row[1]} {row[2]}" for row in info)
                # Include first 3 sample rows for context
                try:
                    rows = conn.execute(f"SELECT * FROM {table} LIMIT 3").fetchall()
                    sample = "; ".join(str(dict(r)) for r in rows)
                    parts.append(f"Table: {table} ({cols})\nSample rows: {sample}")
                except Exception:
                    parts.append(f"Table: {table} ({cols})")
            self._schema_cache = "\n\n".join(parts)
        finally:
            conn.close()
        return self._schema_cache

    # ------------------------------------------------------------------ #
    # Seed database (for demos without an external DB)                    #
    # ------------------------------------------------------------------ #

    def seed_demo_db(self) -> None:
        """
        Populate the connected SQLite DB with annotation task sample data.
        This lets the demo script run with no external data files.
        """
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS annotators (
                    id          INTEGER PRIMARY KEY,
                    name        TEXT NOT NULL,
                    skill_level TEXT NOT NULL,
                    tasks_done  INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id          INTEGER PRIMARY KEY,
                    title       TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    difficulty  TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    annotator_id INTEGER REFERENCES annotators(id),
                    created_at  TEXT NOT NULL,
                    reward_usd  REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS annotations (
                    id          INTEGER PRIMARY KEY,
                    task_id     INTEGER REFERENCES tasks(id),
                    annotator_id INTEGER REFERENCES annotators(id),
                    label       TEXT NOT NULL,
                    confidence  REAL NOT NULL,
                    created_at  TEXT NOT NULL
                );
                INSERT OR IGNORE INTO annotators VALUES
                    (1, 'Alice',   'expert',       142),
                    (2, 'Bob',     'intermediate', 87),
                    (3, 'Charlie', 'beginner',     23),
                    (4, 'Diana',   'expert',       198),
                    (5, 'Ethan',   'intermediate', 55);
                INSERT OR IGNORE INTO tasks VALUES
                    (1,  'Sentiment analysis – tweet',       'NLP',   'easy',   'completed', 1,  '2025-01-10', 0.50),
                    (2,  'Named entity tagging – news',      'NLP',   'medium', 'completed', 2,  '2025-01-11', 1.20),
                    (3,  'Image object bounding box',        'Vision','hard',   'completed', 4,  '2025-01-12', 2.50),
                    (4,  'Audio transcription review',       'Audio', 'medium', 'in_progress',1, '2025-01-13', 1.80),
                    (5,  'Text summarisation quality',       'NLP',   'hard',   'completed', 3,  '2025-01-14', 2.00),
                    (6,  'Image classification – defect',   'Vision','easy',   'completed', 2,  '2025-01-15', 0.75),
                    (7,  'Relation extraction – biomedical', 'NLP',   'hard',   'pending',   NULL,'2025-01-16', 3.00),
                    (8,  'Video clip labelling',             'Video', 'hard',   'completed', 4,  '2025-01-17', 4.00),
                    (9,  'Spam detection – email',           'NLP',   'easy',   'completed', 5,  '2025-01-18', 0.40),
                    (10, 'Semantic segmentation review',     'Vision','hard',   'pending',   NULL,'2025-01-19', 5.00);
                INSERT OR IGNORE INTO annotations VALUES
                    (1, 1, 1, 'positive',  0.95, '2025-01-10'),
                    (2, 2, 2, 'ORG:Apple', 0.88, '2025-01-11'),
                    (3, 3, 4, 'car',       0.92, '2025-01-12'),
                    (4, 5, 3, 'poor',      0.70, '2025-01-14'),
                    (5, 6, 2, 'defect',    0.85, '2025-01-15'),
                    (6, 8, 4, 'walking',   0.98, '2025-01-17'),
                    (7, 9, 5, 'spam',      0.80, '2025-01-18');
            """)
            conn.commit()
            self._schema_cache = None  # invalidate cache after seeding
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # SQL generation + execution                                          #
    # ------------------------------------------------------------------ #

    def _generate_sql(self, question: str, schema: str) -> str:
        prompt = SCHEMA_PROMPT.format(schema=schema, question=question)
        raw = self.llm.complete(prompt, max_tokens=256)
        return _extract_sql(raw)

    def _execute_sql(self, sql: str) -> Tuple[List[Dict], Optional[str]]:
        """Return (rows, error_message).  rows is [] on error."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return rows, None
        except Exception as e:
            return [], str(e)
        finally:
            conn.close()

    def query(self, question: str) -> Dict:
        """
        Run the full Text-to-SQL pipeline and return results.

        Returns
        -------
        dict with keys:
          question, sql, rows, error (None on success), attempts, schema_used
        """
        schema = self._get_schema()
        sql = self._generate_sql(question, schema)
        rows, error = self._execute_sql(sql)
        attempts = 1

        while error and attempts <= self.max_retries:
            correction_prompt = CORRECTION_PROMPT.format(
                question=question, schema=schema, sql=sql, error=error
            )
            raw_corrected = self.llm.complete(correction_prompt, max_tokens=256)
            sql = _extract_sql(raw_corrected)
            rows, error = self._execute_sql(sql)
            attempts += 1

        return {
            "question": question,
            "sql": sql,
            "rows": rows,
            "error": error,
            "attempts": attempts,
            "schema_used": schema,
        }
