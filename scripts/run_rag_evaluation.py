"""
Full RAG pipeline evaluation with RAGAS metrics.

Builds a knowledge base, runs queries through the full pipeline, then
evaluates each answer on faithfulness, answer relevancy, context precision,
and context recall.  Runs entirely offline with the MockLLM.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rrs.pipeline import RAGPipeline
from rrs.evaluation.ragas import RAGASEvaluator
from rrs.llm import get_llm


DOCUMENTS = [
    """Retrieval-Augmented Generation (RAG) is an architecture that enhances large
    language models by grounding their outputs in retrieved passages from an external
    corpus. The LLM receives both the user question and the retrieved context as input.
    This reduces hallucination because the model is encouraged to base its answer on
    verifiable evidence rather than parametric memory alone. RAG was first described
    by Lewis et al. (2020) in a seminal NeurIPS paper.""",

    """Dense retrieval encodes documents and queries as dense vectors using a bi-encoder.
    The most popular model for this is all-MiniLM-L6-v2 from sentence-transformers,
    which produces 384-dimensional embeddings. FAISS (Facebook AI Similarity Search)
    provides efficient approximate nearest neighbour (ANN) search, making it possible
    to query millions of documents in milliseconds. Inner product search with L2-normalised
    vectors is equivalent to cosine similarity search.""",

    """BM25 is a probabilistic sparse retrieval model that scores documents based on
    term frequency (TF) and inverse document frequency (IDF). Unlike dense models it
    requires no training and generalises well to out-of-domain text. Its main weakness
    is the vocabulary mismatch problem: synonyms and paraphrases that do not share
    exact terms with the query are not retrieved. Hybrid systems that combine dense
    and sparse retrieval via Reciprocal Rank Fusion (RRF) consistently outperform
    either approach alone.""",

    """Data annotation is the process of assigning labels to raw data so that supervised
    machine learning models can learn from it. High quality annotations require clear
    labelling guidelines, domain expertise, and rigorous quality control. Inter-annotator
    agreement (IAA) is typically measured with Cohen's Kappa or Krippendorff's Alpha.
    Platforms such as Scale AI, Surge AI, and Amazon Mechanical Turk connect businesses
    with annotation workforces. Pricing is per-task and bonuses are paid for speed and
    accuracy.""",

    """RLHF (Reinforcement Learning from Human Feedback) is the dominant technique for
    aligning large language models with human preferences. Annotators compare pairs of
    model responses and indicate which one they prefer. These preference labels train a
    reward model, which is then used to fine-tune the LLM via proximal policy optimisation
    (PPO). OpenAI used RLHF to train InstructGPT and ChatGPT. The quality and diversity
    of the preference annotations have an outsized effect on the final model's behaviour.""",
]

QA_DATASET = [
    {
        "question": "What is Retrieval-Augmented Generation and why does it reduce hallucination?",
        "ground_truth": "RAG grounds LLM outputs in retrieved passages from an external corpus, reducing hallucination by basing answers on verifiable evidence.",
    },
    {
        "question": "How does dense retrieval work and what library is used for fast search?",
        "ground_truth": "Dense retrieval encodes documents and queries as vectors with a bi-encoder, then uses FAISS for fast approximate nearest neighbour search.",
    },
    {
        "question": "What is the main weakness of BM25 and how is it mitigated?",
        "ground_truth": "BM25 suffers from vocabulary mismatch. Hybrid systems combining BM25 with dense retrieval via RRF mitigate this weakness.",
    },
    {
        "question": "What metrics are used to measure annotation quality?",
        "ground_truth": "Cohen's Kappa and Krippendorff's Alpha are used to measure inter-annotator agreement.",
    },
    {
        "question": "How is RLHF used to align language models?",
        "ground_truth": "RLHF trains a reward model on human preference comparisons and uses it to fine-tune the LLM with PPO.",
    },
]


def main():
    print("RAG Research Suite — Full Pipeline Evaluation\n")

    llm = get_llm()
    print(f"LLM: {llm.model}  (set OPENAI_API_KEY to use real model)")
    print("Building RAG pipeline...\n")

    pipeline = RAGPipeline(llm=llm, top_k=2, use_cot=False)
    n_chunks = pipeline.index(DOCUMENTS)
    print(f"Indexed {len(DOCUMENTS)} documents → {n_chunks} chunks\n")

    evaluator = RAGASEvaluator(llm=llm)

    eval_samples = []
    print(f"  {'#':<3}  {'Question':<52}  {'Answer snippet':<30}")
    print(f"  {'-'*3}  {'-'*52}  {'-'*30}")

    for i, qa in enumerate(QA_DATASET, 1):
        result = pipeline.run(qa["question"])
        answer = result["answer"]
        snippet = answer[:28].replace("\n", " ") + "..."
        q_label = qa["question"][:51]
        print(f"  {i:<3}  {q_label:<52}  {snippet:<30}")

        eval_samples.append({
            "question": qa["question"],
            "answer": answer,
            "contexts": [r["text"] for r in result["retrieved_chunks"]],
            "ground_truth": qa["ground_truth"],
        })

    print("\nRunning RAGAS evaluation...")
    report = evaluator.evaluate_dataset(eval_samples)
    agg = report["aggregate"]

    print(f"\n{'='*60}")
    print("  RAGAS Aggregate Scores")
    print(f"{'='*60}")
    print(f"  Faithfulness      : {agg['faithfulness']:.4f}  (grounding in context)")
    print(f"  Answer Relevancy  : {agg['answer_relevancy']:.4f}  (answers the question)")
    print(f"  Context Precision : {agg['context_precision']:.4f}  (retrieved chunks useful)")
    print(f"  Context Recall    : {agg['context_recall']:.4f}  (GT info in context)")
    print(f"  RAGAS Score       : {agg['ragas_score']:.4f}  (mean of above)")
    print()

    print("  Per-sample faithfulness:")
    for i, s in enumerate(report["per_sample"], 1):
        print(f"    Q{i}: {s['faithfulness']:.3f}  (n_claims={s['faithfulness_detail']['n_claims']}, "
              f"supported={s['faithfulness_detail']['n_supported']})")
    print()


if __name__ == "__main__":
    main()
