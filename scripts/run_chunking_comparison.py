"""
Chunking strategy comparison: Fixed vs Recursive vs Semantic.

Runs all three chunkers on a sample corpus, prints statistics, and shows
example output.  No API key required (semantic chunker uses local
sentence-transformers model).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import textwrap

from rrs.chunking.fixed import FixedChunker
from rrs.chunking.recursive import RecursiveChunker
from rrs.chunking.semantic import SemanticChunker

CORPUS = [
    textwrap.dedent("""\
        Retrieval-Augmented Generation (RAG) combines the parametric knowledge of
        large language models with non-parametric retrieval from an external corpus.
        The key insight is that LLMs have fixed knowledge cutoffs and can hallucinate
        facts, whereas a retrieval system can access up-to-date, verifiable information.

        Dense retrieval uses bi-encoder models that map documents and queries into a
        shared embedding space.  Approximate Nearest Neighbour search (e.g. FAISS) then
        finds the top-k most similar passages in sub-linear time.  The seminal work by
        Karpukhin et al. (2020) demonstrated that dense passage retrieval significantly
        outperforms BM25 on open-domain QA benchmarks.

        Sparse retrieval methods like BM25 rely on term frequency statistics and do not
        require any learned representations.  They remain competitive on out-of-domain
        queries and are extremely fast to index.  Hybrid systems that combine dense and
        sparse retrieval via Reciprocal Rank Fusion (Cormack et al., 2009) consistently
        outperform either approach alone.
    """),
    textwrap.dedent("""\
        Data annotation is the process of labelling raw data — text, images, audio, or
        video — so that machine learning models can learn from it.  High quality labels
        require domain expertise, clear guidelines, and inter-annotator agreement metrics
        such as Cohen's Kappa or Krippendorff's Alpha.

        Annotation platforms connect businesses that need labelled data with a workforce
        of skilled annotators.  Pricing is typically per-task with bonuses for speed and
        accuracy.  Quality control mechanisms include majority voting, gold standard tasks
        embedded in the queue, and expert review of edge cases.

        Large language models are increasingly used both as the subject of annotation
        (RLHF preference labelling) and as automated annotators for tasks where human
        agreement is high.  However, model-generated labels must be validated against
        human ground truth before being used for fine-tuning.
    """),
]


def stats(chunks):
    lengths = [len(c) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths)) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
    }


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    print("RAG Research Suite — Chunking Comparison Demo")

    # ----- Fixed chunking -----
    print_section("1. Fixed-Size Chunker  (chunk_size=100, overlap=10)")
    fc = FixedChunker(chunk_size=100, overlap=10)
    fixed_chunks = []
    for doc in CORPUS:
        fixed_chunks.extend(fc.split(doc))
    s = stats(fixed_chunks)
    print(f"  Documents : {len(CORPUS)}")
    print(f"  Chunks    : {s['n_chunks']}")
    print(f"  Avg chars : {s['avg_chars']}")
    print(f"  Min/Max   : {s['min_chars']} / {s['max_chars']}")
    print(f"\n  First chunk:\n  {textwrap.fill(fixed_chunks[0][:200], width=58, initial_indent='  ')}")

    # ----- Recursive chunking -----
    print_section("2. Recursive Chunker  (chunk_size=400, overlap=40)")
    rc = RecursiveChunker(chunk_size=400, overlap=40)
    recursive_chunks = []
    for doc in CORPUS:
        recursive_chunks.extend(rc.split(doc))
    s = stats(recursive_chunks)
    print(f"  Documents : {len(CORPUS)}")
    print(f"  Chunks    : {s['n_chunks']}")
    print(f"  Avg chars : {s['avg_chars']}")
    print(f"  Min/Max   : {s['min_chars']} / {s['max_chars']}")
    print(f"\n  First chunk:\n  {textwrap.fill(recursive_chunks[0][:300], width=58, initial_indent='  ')}")

    # ----- Semantic chunking -----
    print_section("3. Semantic Chunker  (threshold_z=0.8)")
    print("  Loading sentence-transformers model (all-MiniLM-L6-v2) ...")
    sc = SemanticChunker(threshold_z=0.8, min_sentences=2)
    semantic_chunks = []
    for doc in CORPUS:
        semantic_chunks.extend(sc.split(doc))
    s = stats(semantic_chunks)
    print(f"  Documents : {len(CORPUS)}")
    print(f"  Chunks    : {s['n_chunks']}")
    print(f"  Avg chars : {s['avg_chars']}")
    print(f"  Min/Max   : {s['min_chars']} / {s['max_chars']}")
    if semantic_chunks:
        print(f"\n  First chunk:\n  {textwrap.fill(semantic_chunks[0][:300], width=58, initial_indent='  ')}")

    # ----- Summary table -----
    print_section("Summary")
    print(f"  {'Strategy':<18}  {'Chunks':>6}  {'Avg chars':>9}  {'Min':>5}  {'Max':>5}")
    print(f"  {'-'*18}  {'-'*6}  {'-'*9}  {'-'*5}  {'-'*5}")

    for name, chunks_list in [
        ("Fixed-Size", fixed_chunks),
        ("Recursive", recursive_chunks),
        ("Semantic", semantic_chunks),
    ]:
        s = stats(chunks_list)
        print(f"  {name:<18}  {s['n_chunks']:>6}  {s['avg_chars']:>9}  {s['min_chars']:>5}  {s['max_chars']:>5}")

    print("\n  Semantic chunking preserves paragraph boundaries automatically.")
    print("  Fixed-size is fastest; recursive respects natural separators.")
    print()


if __name__ == "__main__":
    main()
