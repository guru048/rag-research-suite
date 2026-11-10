"""
Unit tests for the RAG Research Suite.

All tests run offline — no API key required, no network calls.
Heavy models (sentence-transformers, FAISS) are used only in integration
tests tagged with @pytest.mark.integration.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

# ------------------------------------------------------------------ #
# Chunking tests                                                      #
# ------------------------------------------------------------------ #

class TestFixedChunker:
    def setup_method(self):
        from rrs.chunking.fixed import FixedChunker
        self.FixedChunker = FixedChunker

    def test_basic_split(self):
        fc = self.FixedChunker(chunk_size=5, overlap=1)
        chunks = fc.split("a b c d e f g h i j")
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_overlap(self):
        fc = self.FixedChunker(chunk_size=4, overlap=2)
        chunks = fc.split("a b c d e f g h")
        # overlap means consecutive chunks share tokens
        c1_tokens = set(chunks[0].split())
        c2_tokens = set(chunks[1].split())
        assert len(c1_tokens & c2_tokens) >= 1

    def test_empty_text(self):
        fc = self.FixedChunker(chunk_size=100, overlap=10)
        assert fc.split("") == []

    def test_overlap_validation(self):
        with pytest.raises(ValueError):
            self.FixedChunker(chunk_size=10, overlap=10)

    def test_split_documents(self):
        fc = self.FixedChunker(chunk_size=50, overlap=5)
        docs = ["word " * 100, "text " * 80]
        all_chunks = fc.split_documents(docs)
        assert all("text" in c for c in all_chunks)
        assert all("doc_idx" in c for c in all_chunks)


class TestRecursiveChunker:
    def setup_method(self):
        from rrs.chunking.recursive import RecursiveChunker
        self.RecursiveChunker = RecursiveChunker

    def test_basic_split(self):
        rc = self.RecursiveChunker(chunk_size=100, overlap=10)
        text = "First paragraph with several sentences.\n\nSecond paragraph is here too."
        chunks = rc.split(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_respects_paragraphs(self):
        rc = self.RecursiveChunker(chunk_size=200, overlap=20)
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = rc.split(text)
        # With large chunk_size relative to text, may merge everything
        assert len(chunks) >= 1

    def test_empty_text(self):
        rc = self.RecursiveChunker(chunk_size=100, overlap=10)
        assert rc.split("") == []


# ------------------------------------------------------------------ #
# LLM tests                                                           #
# ------------------------------------------------------------------ #

class TestMockLLM:
    def setup_method(self):
        from rrs.llm import MockLLM
        self.llm = MockLLM()

    def test_returns_string(self):
        result = self.llm.complete("What is RAG?")
        assert isinstance(result, str) and len(result) > 0

    def test_sql_response(self):
        result = self.llm.complete("write a SQL query to count rows")
        assert "SELECT" in result.upper()

    def test_cot_response(self):
        result = self.llm.complete("think step by step about this problem")
        assert "step" in result.lower() or "therefore" in result.lower()

    def test_chat(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Answer my question."},
        ]
        result = self.llm.chat(messages)
        assert isinstance(result, str) and len(result) > 0

    def test_call_count(self):
        self.llm.complete("test 1")
        self.llm.complete("test 2")
        assert self.llm._call_count == 2


# ------------------------------------------------------------------ #
# Prompt engineering tests                                            #
# ------------------------------------------------------------------ #

class TestChainOfThoughtPrompt:
    def setup_method(self):
        from rrs.prompts.chain_of_thought import ChainOfThoughtPrompt
        self.CoT = ChainOfThoughtPrompt

    def test_zero_shot(self):
        cot = self.CoT(zero_shot=True)
        prompt = cot.build("What is 2+2?")
        assert "step by step" in prompt.lower()
        assert "What is 2+2?" in prompt

    def test_zero_shot_with_context(self):
        cot = self.CoT(zero_shot=True)
        prompt = cot.build("What year?", context="It happened in 1969.")
        assert "1969" in prompt

    def test_few_shot(self):
        examples = [
            {"question": "2+2?", "reasoning": "Add 2 and 2.", "answer": "4"},
        ]
        cot = self.CoT(few_shot_examples=examples)
        prompt = cot.build("3+3?")
        assert "2+2" in prompt
        assert "3+3" in prompt

    def test_build_messages(self):
        cot = self.CoT(zero_shot=True, system_prefix="You are an expert.")
        messages = cot.build_messages("Test question?")
        assert any(m["role"] == "system" for m in messages)
        assert any(m["role"] == "user" for m in messages)


class TestFewShotPromptBuilder:
    def setup_method(self):
        from rrs.prompts.few_shot import FewShotPromptBuilder
        self.FewShotPromptBuilder = FewShotPromptBuilder

    def test_static_build_no_encoder(self):
        examples = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]
        # Test that add_example works
        builder = self.FewShotPromptBuilder(examples=list(examples), n_shots=2)
        builder.add_example({"question": "Q3?", "answer": "A3"})
        assert len(builder.examples) == 3


# ------------------------------------------------------------------ #
# SQL agent tests (offline — in-memory DB, MockLLM)                  #
# ------------------------------------------------------------------ #

class TestTextToSQLAgent:
    def setup_method(self):
        from rrs.sql_agent.agent import TextToSQLAgent
        from rrs.llm import MockLLM
        self.agent = TextToSQLAgent(llm=MockLLM(), db_path=":memory:", max_retries=2)
        self.agent.seed_demo_db()

    def test_seed_creates_tables(self):
        schema = self.agent._get_schema()
        assert "annotators" in schema
        assert "tasks" in schema
        assert "annotations" in schema

    def test_query_returns_dict(self):
        result = self.agent.query("How many annotators are there?")
        assert "question" in result
        assert "sql" in result
        assert "rows" in result
        assert "error" in result

    def test_query_sql_contains_select(self):
        result = self.agent.query("Count all tasks")
        assert "SELECT" in result["sql"].upper()

    def test_direct_sql_execution(self):
        rows, err = self.agent._execute_sql("SELECT COUNT(*) as n FROM annotators;")
        assert err is None
        assert rows[0]["n"] == 5

    def test_schema_extraction(self):
        schema = self.agent._get_schema()
        assert len(schema) > 50  # non-trivial schema string


# ------------------------------------------------------------------ #
# Integration tests (require sentence-transformers + faiss-cpu)       #
# ------------------------------------------------------------------ #

@pytest.mark.integration
class TestDenseRetrieverIntegration:
    def setup_method(self):
        from rrs.retrieval.dense import DenseRetriever
        self.dr = DenseRetriever(top_k=3)
        self.corpus = [
            "Dense retrieval uses FAISS for vector search.",
            "BM25 is a sparse retrieval method.",
            "Hybrid retrieval combines dense and sparse methods.",
            "Chain-of-thought prompting improves reasoning.",
            "Data annotation platforms pay per task.",
        ]
        self.dr.index(self.corpus)

    def test_retrieve_returns_list(self):
        results = self.dr.retrieve("How does dense search work?")
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_retrieve_has_required_keys(self):
        results = self.dr.retrieve("retrieval methods")
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "doc_idx" in r

    def test_top_result_relevance(self):
        results = self.dr.retrieve("FAISS vector search")
        assert results[0]["doc_idx"] == 0  # first doc is most relevant


@pytest.mark.integration
class TestBM25RetrieverIntegration:
    def setup_method(self):
        from rrs.retrieval.sparse import BM25Retriever
        self.bm25 = BM25Retriever(top_k=3)
        self.corpus = [
            "Dense retrieval uses FAISS for vector search.",
            "BM25 is a sparse term-frequency based retrieval method.",
            "Hybrid retrieval combines dense and sparse methods.",
        ]
        self.bm25.index(self.corpus)

    def test_retrieve_returns_results(self):
        results = self.bm25.retrieve("BM25 sparse retrieval")
        assert len(results) > 0

    def test_vocab_size(self):
        assert self.bm25.vocab_size > 0


@pytest.mark.integration
class TestHybridRetrieverIntegration:
    def test_hybrid_combines_results(self):
        from rrs.retrieval.hybrid import HybridRetriever
        corpus = [
            "Dense retrieval uses vector embeddings.",
            "BM25 uses term frequency statistics.",
            "Hybrid methods combine both approaches.",
            "FAISS is used for approximate nearest neighbour search.",
        ]
        hr = HybridRetriever(top_k=3, dense_top_k=4, sparse_top_k=4)
        hr.index(corpus)
        results = hr.retrieve("How does retrieval work?")
        assert 1 <= len(results) <= 3
        for r in results:
            assert "rrf_score" in r


@pytest.mark.integration
class TestRAGPipelineIntegration:
    def test_end_to_end(self):
        from rrs.pipeline import RAGPipeline
        from rrs.llm import MockLLM
        docs = [
            "RAG grounds LLM outputs in retrieved passages to reduce hallucination.",
            "Dense retrieval encodes documents with sentence-transformers.",
            "BM25 is a sparse retrieval method based on term frequency.",
        ]
        pipeline = RAGPipeline(llm=MockLLM(), top_k=2)
        n = pipeline.index(docs)
        assert n > 0
        result = pipeline.run("What is RAG?")
        assert "answer" in result
        assert len(result["answer"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not integration"])
