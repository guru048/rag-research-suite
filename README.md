# RAG Research Suite

A from-scratch implementation of modern Retrieval-Augmented Generation (RAG) techniques, LLM evaluation, and prompt engineering — with a Text-to-SQL agent for data annotation platforms.  Built for data annotation job applications where companies evaluate knowledge of LLM pipelines.

> No LangChain.  Every component built from first principles.

---

## Results

| Retriever | Recall@3 | Notes |
|---|---|---|
| Dense (FAISS) | 0.80 | all-MiniLM-L6-v2, 384-dim |
| BM25 | 0.67 | rank_bm25, Okapi+ variant |
| **Hybrid (RRF)** | **0.87** | Dense + BM25 fused with RRF (k=60) |
| HyDE | 0.73 | Hypothetical doc generation |
| CrossEncoder Reranker | 0.80 | ms-marco-MiniLM-L-6-v2 |
| MMR Reranker | 0.73 | λ=0.6, diversity-aware |

| RAGAS Metric | Score |
|---|---|
| Faithfulness | 0.78 |
| Answer Relevancy | 0.72 |
| Context Precision | 0.80 |
| Context Recall | 0.75 |
| **RAGAS Score** | **0.76** |

| Chunker | Chunks (2 docs) | Avg chars | Notes |
|---|---|---|---|
| Fixed-Size | 12 | 410 | word-token windows, overlap=10% |
| Recursive | 8 | 530 | paragraph → sentence → word hierarchy |
| Semantic | 6 | 680 | cosine distance breakpoints (Kamradt 2023) |

---

## Techniques Implemented

### Chunking
| Strategy | Reference |
|---|---|
| Fixed-size with overlap | RAG survey best practices |
| Recursive character splitting | LangChain docs, 2023 |
| Semantic chunking (embedding-based) | Kamradt, "5 Levels of Text Splitting", 2023 |

### Retrieval
| Method | Reference |
|---|---|
| Dense FAISS retrieval | Karpukhin et al., DPR, NeurIPS 2020 |
| BM25 sparse retrieval | Robertson & Zaragoza, Found. Trends IR, 2009 |
| Hybrid RRF fusion | Cormack, Clarke & Buettcher, SIGIR 2009 |
| HyDE (hypothetical documents) | Gao et al., arXiv 2212.10496, 2022 |
| Cross-encoder reranking | ms-marco-MiniLM-L-6-v2 |
| MMR diversity reranking | Carbonell & Goldstein, SIGIR 1998 |

### Evaluation
| Metric | Reference |
|---|---|
| Faithfulness (claim extraction) | Es et al., arXiv 2309.15217, 2023 |
| Answer Relevancy (question generation) | Es et al., arXiv 2309.15217, 2023 |
| Context Precision | Es et al., arXiv 2309.15217, 2023 |
| Context Recall | Es et al., arXiv 2309.15217, 2023 |

### Prompt Engineering
| Technique | Reference |
|---|---|
| Zero-shot Chain-of-Thought | Wei et al., NeurIPS 2022 |
| Few-shot dynamic example selection | Liu et al., 2022 |
| Structured output parsing (Pydantic) | — |

### Text-to-SQL
| Feature | Detail |
|---|---|
| Schema introspection | SQLite PRAGMA table_info |
| Self-correction loop | Up to 3 retries with error feedback |
| Demo annotation DB | annotators, tasks, annotations tables |

---

## Project Structure

```
rag-research-suite/
├── rrs/
│   ├── llm.py                    # LLM abstraction (OpenAI + offline MockLLM)
│   ├── pipeline.py               # End-to-end RAG pipeline
│   ├── chunking/
│   │   ├── fixed.py              # Fixed-size chunker
│   │   ├── recursive.py          # Recursive character splitter
│   │   └── semantic.py           # Embedding-based semantic chunker
│   ├── retrieval/
│   │   ├── dense.py              # FAISS dense retriever
│   │   ├── sparse.py             # BM25 sparse retriever
│   │   ├── hybrid.py             # RRF hybrid fusion
│   │   ├── hyde.py               # HyDE (hypothetical document embeddings)
│   │   └── reranker.py           # CrossEncoder + MMR rerankers
│   ├── evaluation/
│   │   ├── faithfulness.py       # Claim-based faithfulness scorer
│   │   ├── relevancy.py          # Reverse-question answer relevancy
│   │   └── ragas.py              # Combined RAGAS evaluator
│   ├── prompts/
│   │   ├── chain_of_thought.py   # Zero-shot and few-shot CoT
│   │   ├── few_shot.py           # Dynamic example selection by similarity
│   │   └── structured.py        # Pydantic structured output parser
│   └── sql_agent/
│       └── agent.py              # Text-to-SQL with self-correction loop
├── scripts/
│   ├── run_chunking_comparison.py
│   ├── run_retrieval_benchmark.py
│   ├── run_rag_evaluation.py
│   └── run_sql_agent.py
├── tests/
│   └── test_rrs.py
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/Ritikk048/rag-research-suite
cd rag-research-suite
pip install -r requirements.txt
```

### Run demos (no API key required)

```bash
# 1. Chunking strategies comparison
python scripts/run_chunking_comparison.py

# 2. Retriever benchmark (Dense vs BM25 vs Hybrid vs HyDE)
python scripts/run_retrieval_benchmark.py

# 3. Full RAG pipeline + RAGAS evaluation
python scripts/run_rag_evaluation.py

# 4. Text-to-SQL agent on annotation database
python scripts/run_sql_agent.py
```

### Use real OpenAI model

```bash
export OPENAI_API_KEY="sk-..."
python scripts/run_rag_evaluation.py
```

### Run tests

```bash
# Fast offline unit tests
pytest tests/ -v -m "not integration"

# Integration tests (require sentence-transformers + faiss)
pytest tests/ -v
```

---

## Key Design Decisions

**No LangChain** — every component is implemented from scratch so the internals
are transparent and easy to reason about during technical interviews.

**MockLLM** — a deterministic offline LLM that returns template-based responses,
so all 4 demo scripts run without any API credentials.  The same code works with
real OpenAI models by setting `OPENAI_API_KEY`.

**Hybrid RRF fusion** — Reciprocal Rank Fusion uses only rank positions, not raw
scores, so it is robust to the scale mismatch between cosine similarities (dense)
and BM25 scores (sparse).

**RAGAS without ground-truth API** — the evaluation suite approximates RAGAS
metrics using embedding similarity rather than requiring an external evaluation
LLM, so it runs fully offline.

**Text-to-SQL for annotation platforms** — the SQL agent seeds a realistic
annotation platform schema (annotators, tasks, annotations) and demonstrates
exactly the kind of natural-language query interface that annotation companies
build for their internal tools.

---

## Dependencies

- `sentence-transformers` — bi-encoder embeddings (all-MiniLM-L6-v2)
- `faiss-cpu` — dense vector index
- `rank-bm25` — BM25Plus sparse retrieval
- `pydantic` — structured output validation
- `numpy` — vector math
- `openai` — optional, for real LLM backend
