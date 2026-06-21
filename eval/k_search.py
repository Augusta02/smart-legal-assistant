"""
eval/k_search.py
----------------
Hyperparameter sweep over RETRIEVER_K — finds the best K for context retrieval
WITHOUT re-running the LLM pipeline. Only the retriever is called, so this
completes in seconds per K value.

What it tests:
  - Context Recall     : % of ground-truth key terms found in retrieved chunks
  - Context Precision  : % of retrieved chunks relevant to the question

Once the best K is found, run:
    python eval/eval.py --no-cache    # regenerate full pipeline cache with best K
    python eval/lightweight_eval.py   # score all 4 metrics

Usage:
    python eval/k_search.py
    python eval/k_search.py --k-values 4 6 8 10 12 14
    python eval/k_search.py --apply   # writes best K to rag_pipeline.py automatically
"""

import sys
import re
import json
import argparse
import numpy as np
import nltk
from pathlib import Path

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from rag_pipeline import setup_rag_system, get_embeddings, build_vector_store

# Paths 
QA_PATH        = Path(__file__).parent / "qa_pairs.json"
PIPELINE_FILE  = _ROOT / "rag_pipeline.py"

# Stopwords 
STOPWORDS = set(stopwords.words("english"))

#  Token helpers 

def tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}

#  Retrieval metrics 
def context_recall(ground_truth: str, contexts: list[str]) -> float:
    gt_terms  = tokenize(ground_truth)
    if not gt_terms:
        return 1.0
    ctx_terms = tokenize(" ".join(contexts))
    return len(gt_terms & ctx_terms) / len(gt_terms)


def context_precision(question: str, ground_truth: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    target_terms = tokenize(question) | tokenize(ground_truth)
    relevant = sum(
        1 for chunk in contexts
        if len(tokenize(chunk) & target_terms) >= 3
    )
    return relevant / len(contexts)


#  Sweep

def sweep(k_values: list[int], vector_store) -> list[dict]:
    qa_pairs = json.loads(QA_PATH.read_text())
    results  = []

    for k in k_values:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        recalls, precisions = [], []

        for item in qa_pairs:
            q   = item["question"]
            gt  = item["ground_truth"]
            docs = retriever.invoke(q)
            ctxs = [d.page_content for d in docs]
            recalls.append(context_recall(gt, ctxs))
            precisions.append(context_precision(q, gt, ctxs))

        mean_recall    = float(np.mean(recalls))
        mean_precision = float(np.mean(precisions))
        combined = 0.6 * mean_recall + 0.4 * mean_precision

        results.append({
            "k":         k,
            "recall":    mean_recall,
            "precision": mean_precision,
            "combined":  combined,
        })
        print(
            f"  K={k:>2}  "
            f"Recall={mean_recall:.3f}  "
            f"Precision={mean_precision:.3f}  "
            f"Combined={combined:.3f}"
        )

    return results


#  Apply best K to rag_pipeline.py 

def apply_best_k(best_k: int) -> None:
    text    = PIPELINE_FILE.read_text(encoding="utf-8")
    updated = re.sub(r"^RETRIEVER_K\s*=\s*\d+", f"RETRIEVER_K = {best_k}", text, flags=re.MULTILINE)
    if updated == text:
        print(f"  Could not find RETRIEVER_K in {PIPELINE_FILE.name} — update manually.")
        return
    PIPELINE_FILE.write_text(updated, encoding="utf-8")
    print(f"  rag_pipeline.py updated → RETRIEVER_K = {best_k}")


#  Main 

def main():
    parser = argparse.ArgumentParser(
        description="Sweep RETRIEVER_K to find the best retrieval setting."
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int,
        default=[2, 4, 6, 8, 10, 12],
        help="K values to test (default: 2 4 6 8 10 12)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the best K back to rag_pipeline.py automatically",
    )
    args = parser.parse_args()

    print("Loading vector store (no rebuild — using existing DB)...")
    embeddings   = get_embeddings()
    vector_store = build_vector_store(embeddings, force_rebuild=False)
    results = sweep(args.k_values, vector_store)

    best = max(results, key=lambda r: r["combined"])

    print("\n" + "=" * 55)
    print("  K Search Results")
    print("=" * 55)
    print(f"  {'K':>4}  {'Recall':>7}  {'Precision':>9}  {'Combined':>8}  {'':>4}")
    print("  " + "-" * 45)
    for r in results:
        star = " ◀ best" if r["k"] == best["k"] else ""
        print(
            f"  {r['k']:>4}  "
            f"{r['recall']:>7.3f}  "
            f"{r['precision']:>9.3f}  "
            f"{r['combined']:>8.3f}"
            f"{star}"
        )
    print("=" * 55)
    print(f"\n  Best K = {best['k']}  "
          f"(Recall={best['recall']:.3f}, Precision={best['precision']:.3f})\n")

    if args.apply:
        print("Applying best K to rag_pipeline.py...")
        apply_best_k(best["k"])
    else:
        print(f"           (or manually set RETRIEVER_K = {best['k']} in rag_pipeline.py)\n")


if __name__ == "__main__":
    main()
