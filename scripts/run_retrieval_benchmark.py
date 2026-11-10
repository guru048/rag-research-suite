"""
Retrieval benchmark: Dense vs BM25 vs Hybrid (RRF) vs HyDE.

Builds a knowledge base of 20 passages, runs 5 queries through each
retriever, measures Recall@3 (fraction of relevant docs in top-3), and
prints a comparison table.  Uses a MockLLM for HyDE so no API key is needed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rrs.retrieval.dense import DenseRetriever
from rrs.retrieval.sparse import BM25Retriever
from rrs.retrieval.hybrid import HybridRetriever
from rrs.retrieval.hyde import HyDERetriever
from rrs.retrieval.reranker import CrossEncoderReranker, MMRReranker
from rrs.llm import get_llm


# --- Corpus: 20 passages about RAG, NLP, and annotation ---
CORPUS = [
    "Dense retrieval uses bi-encoder models to embed documents and queries into vector space.",
    "BM25 is a bag-of-words retrieval model based on term frequency and inverse document frequency.",
    "Reciprocal Rank Fusion combines multiple ranked lists without requiring score normalisation.",
    "HyDE generates hypothetical documents that the LLM imagines would answer the query.",
    "Cross-encoders jointly encode query and document pairs for more accurate relevance scoring.",
    "FAISS is a library for efficient similarity search over dense vector embeddings.",
    "Sentence transformers produce fixed-size sentence embeddings suitable for semantic similarity.",
    "Data annotation platforms connect labelling tasks with a distributed workforce of annotators.",
    "RLHF uses human preference labels to align language model outputs with human values.",
    "Maximal Marginal Relevance selects diverse documents to avoid redundancy in retrieved sets.",
    "Chain-of-thought prompting instructs the model to reason step by step before answering.",
    "Few-shot learning provides example input-output pairs in the prompt to guide generation.",
    "Pydantic enables runtime validation of LLM-generated structured outputs against a schema.",
    "Text-to-SQL agents translate natural language questions into executable SQL queries.",
    "Retrieval-augmented generation grounds LLM outputs in retrieved factual passages.",
    "Semantic chunking splits text at points where the topic changes based on embedding distances.",
    "Fixed-size chunking splits text into equal-length windows with configurable overlap.",
    "RAGAS evaluates RAG systems with faithfulness, relevancy, context precision and recall.",
    "Inter-annotator agreement metrics like Cohen's Kappa measure labelling consistency.",
    "Annotation quality control uses gold-standard tasks embedded in the annotation queue.",
]

# Queries and their relevant doc indices (ground truth)
QUERIES = [
    {
        "question": "How does dense retrieval work?",
        "relevant": {0, 5, 6},
    },
    {
        "question": "What is BM25 and how does it compare to dense retrieval?",
        "relevant": {1, 0},
    },
    {
        "question": "How can I improve retrieval diversity?",
        "relevant": {9, 2, 3},
    },
    {
        "question": "What tools evaluate RAG pipeline quality?",
        "relevant": {17, 14, 4},
    },
    {
        "question": "How are data annotation tasks quality controlled?",
        "relevant": {7, 18, 19},
    },
]


def recall_at_k(results, relevant_set, k=3):
    retrieved_indices = {r["doc_idx"] for r in results[:k]}
    return len(retrieved_indices & relevant_set) / len(relevant_set)


def run_benchmark(retriever, queries, top_k=3):
    recalls = []
    for q in queries:
        results = retriever.retrieve(q["question"], top_k=top_k)
        recalls.append(recall_at_k(results, q["relevant"], k=top_k))
    return recalls


def main():
    print("RAG Research Suite — Retrieval Benchmark\n")
    print("Building corpus index (20 passages)...")

    llm = get_llm()
    TOP_K = 3

    # Instantiate retrievers
    dense = DenseRetriever(top_k=TOP_K)
    sparse = BM25Retriever(top_k=TOP_K)
    hybrid = HybridRetriever(top_k=TOP_K, dense_top_k=10, sparse_top_k=10)
    hyde = HyDERetriever(llm=llm, top_k=TOP_K)
    ce_reranker = CrossEncoderReranker(top_k=TOP_K)
    mmr = MMRReranker(top_k=TOP_K, lambda_param=0.6)

    # Index all retrievers
    for r in [dense, sparse, hybrid, hyde]:
        r.index(CORPUS)
    print("Index built.\n")

    # Dense baseline (also used by cross-encoder and MMR as the candidate pool)
    dense_full = DenseRetriever(top_k=10)
    dense_full.index(CORPUS)

    results = {}
    names = ["Dense", "BM25", "Hybrid (RRF)", "HyDE", "CE Reranker", "MMR Reranker"]
    retrievers_to_run = [
        ("Dense", lambda q: dense.retrieve(q, top_k=TOP_K)),
        ("BM25", lambda q: sparse.retrieve(q, top_k=TOP_K)),
        ("Hybrid (RRF)", lambda q: hybrid.retrieve(q, top_k=TOP_K)),
        ("HyDE", lambda q: hyde.retrieve(q, top_k=TOP_K)),
        ("CE Reranker", lambda q: ce_reranker.rerank(q, dense_full.retrieve(q, top_k=10), top_k=TOP_K)),
        ("MMR Reranker", lambda q: mmr.rerank(q, dense_full.retrieve(q, top_k=10), top_k=TOP_K)),
    ]

    for name, fn in retrievers_to_run:
        recalls = []
        for qobj in QUERIES:
            res = fn(qobj["question"])
            recalls.append(recall_at_k(res, qobj["relevant"], k=TOP_K))
        results[name] = recalls

    # --- Print per-query table ---
    print(f"  {'Query':<45}  {'Dense':>6}  {'BM25':>6}  {'Hybrid':>8}  {'HyDE':>6}  {'CE':>6}  {'MMR':>6}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
    for i, q in enumerate(QUERIES):
        label = q["question"][:44]
        row = f"  {label:<45}"
        for name in ["Dense", "BM25", "Hybrid (RRF)", "HyDE", "CE Reranker", "MMR Reranker"]:
            row += f"  {results[name][i]:>5.0%} "
        print(row)

    # --- Mean recall ---
    print(f"\n  {'MEAN Recall@' + str(TOP_K):<45}", end="")
    for name in ["Dense", "BM25", "Hybrid (RRF)", "HyDE", "CE Reranker", "MMR Reranker"]:
        mean = sum(results[name]) / len(results[name])
        print(f"  {mean:>5.0%} ", end="")
    print("\n")

    best = max(
        ["Dense", "BM25", "Hybrid (RRF)", "HyDE", "CE Reranker", "MMR Reranker"],
        key=lambda n: sum(results[n]) / len(results[n])
    )
    print(f"  Best retriever by mean Recall@{TOP_K}: {best}")
    print(f"  LLM used for HyDE: {llm.model} (MockLLM — set OPENAI_API_KEY for real model)\n")


if __name__ == "__main__":
    main()
