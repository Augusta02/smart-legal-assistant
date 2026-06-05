"""
eval/eval.py
------------
RAGAS evaluation for the Smart Legal Assistant RAG pipeline.

What this measures:
  - Faithfulness       : Is the answer grounded in the retrieved context?
                         (catches hallucination)
  - Answer Relevancy   : Does the answer actually address the question?
  - Context Precision  : Are the retrieved chunks relevant to the question?
  - Context Recall     : Did retrieval surface the chunks needed to answer?

How to run:
    cd legal_smart_assistant
    python eval/eval.py                        # uses cache if available
    python eval/eval.py --no-cache             # re-runs the full pipeline
    python eval/eval.py --output eval/results.csv
    python eval/eval.py --rebuild              # force-rebuilds vector DB too

Cache behaviour:
    On first run, pipeline outputs (answers + retrieved chunks) are saved to
    eval/pipeline_cache.json. Every subsequent run loads from this file and
    jumps straight to RAGAS scoring — saving ~10-15 minutes.
    Use --no-cache to discard the cache and re-generate everything.
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
import csv
# Suppress RAGAS deprecation warnings for LangChain wrappers.
# These wrappers still work correctly for local models — RAGAS's suggested
# replacement (llm_factory) only supports cloud providers (OpenAI, etc.).
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_pipeline import setup_rag_system, get_embeddings, get_llm

QA_PATH     = Path(__file__).parent / "qa_pairs.json"
CACHE_PATH  = Path(__file__).parent / "pipeline_cache.json"
METRICS     = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]



def load_cache() -> list[dict] | None:
    """Return cached pipeline results if they exist and are valid, else None."""
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH) as f:
            content = f.read().strip()
        if not content:
            print("Cache file is empty — re-running pipeline.")
            CACHE_PATH.unlink()
            return None
        return json.loads(content)
    except json.JSONDecodeError:
        print("Cache file is corrupt — re-running pipeline.")
        CACHE_PATH.unlink()
        return None


def save_cache(results: list[dict]) -> None:
    """Persist pipeline outputs so we skip re-generation on the next run."""
    with open(CACHE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pipeline outputs cached to {CACHE_PATH.name}")


def cache_is_stale(cached: list[dict], qa_pairs: list[dict]) -> bool:
    """
    Return True if the cache no longer matches the current qa_pairs.
    Detects when you add/edit questions so stale answers aren't scored.
    """
    cached_questions  = {r["question"] for r in cached}
    current_questions = {q["question"] for q in qa_pairs}
    return cached_questions != current_questions



def load_qa_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_pipeline_on_questions(qa_pairs: list[dict], rag_chain, retriever) -> list[dict]:
    """
    For each Q&A pair:
      1. Retrieve the relevant chunks (contexts).
      2. Generate an answer via the RAG chain.
      3. Return everything RAGAS needs.
    """
    results = []

    for i, item in enumerate(qa_pairs, 1):
        question    = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(qa_pairs)}] {question[:70]}...")

        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        answer_chunks = []
        for chunk in rag_chain.stream({"question": question, "chat_history": ""}):
            answer_chunks.append(chunk)
        answer = "".join(answer_chunks)

        results.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })

    return results


# RAGAS 
def build_ragas_dataset(results: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })


# Output 

def print_results(scores: dict) -> None:
    print("\n" + "=" * 55)
    print("  RAGAS Evaluation Results")
    print("=" * 55)

    # Friendly labels for known keys; fall back to the raw key name if not recognized. Also show a simple bar visualization of the score.
    friendly = {
        "faithfulness":      "Faithfulness       (anti-hallucination)",
        "answer_relevancy":  "Answer Relevancy   (on-topic answers)",
        "context_precision": "Context Precision  (retrieval accuracy)",
        "context_recall":    "Context Recall     (retrieval coverage)",
    }

    for key, value in scores.items():
        label = friendly.get(key, key.replace("_", " ").title())
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None

        if numeric is not None and numeric == numeric:  # NaN != NaN
            filled = int(numeric * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {label}: {numeric:.3f}  [{bar}]")
        else:
            print(f"  {label}: N/A  (LLM scorer returned unparseable response)")

    print("\nScores range from 0.0 (worst) to 1.0 (best).")
    print("Target: > 0.7 across all metrics.\n")


def save_results_csv(scores: dict, output_path: str) -> None:
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "score"])
        writer.writeheader()
        for metric, score in scores.items():
            writer.writerow({
                "metric": metric,
                "score": round(score, 4) if isinstance(score, float) else score,
            })
    print(f"Results saved to {output_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for the Legal Assistant RAG pipeline."
    )
    parser.add_argument("--output",   type=str,        default=None,  help="Save scores to CSV")
    parser.add_argument("--rebuild",  action="store_true",            help="Force-rebuild the vector DB")
    parser.add_argument("--no-cache", action="store_true", dest="no_cache",
                        help="Ignore cache and re-run the full pipeline")
    args = parser.parse_args()

    # ── Step 1: Load or generate pipeline outputs ──────────────────────────────
    qa_pairs = load_qa_pairs(QA_PATH)
    cached   = None if args.no_cache else load_cache()

    if cached and cache_is_stale(cached, qa_pairs):
        print("Cache is stale (qa_pairs.json changed) — re-running pipeline...\n")
        cached = None

    if cached:
        print(f"Using cached pipeline outputs ({len(cached)} questions). "
              "Use --no-cache to re-run.\n")
        results = cached
    else:
        print("Loading RAG system...")
        rag_chain, _, retriever = setup_rag_system(force_rebuild=args.rebuild)
        print(f"\nRunning pipeline on {len(qa_pairs)} questions...\n")
        results = run_pipeline_on_questions(qa_pairs, rag_chain, retriever)
        save_cache(results)

    # Step 2: RAGAS scoring
    print("Setting up RAGAS scorers...")
    ragas_llm        = LangchainLLMWrapper(get_llm(temperature=0))
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    for metric in METRICS:
        metric.llm        = ragas_llm
        metric.embeddings = ragas_embeddings

    print("Running RAGAS evaluation (this may take a few minutes)...")
    dataset     = build_ragas_dataset(results)

    # RunConfig: increase timeout for local Ollama (default is too short for local models)
    # timeout=180 gives each scoring call 3 minutes; max_workers=1 runs sequentially
    # to avoid overwhelming a single local GPU/CPU.
    run_cfg = RunConfig(timeout=180, max_retries=2, max_workers=1)
    eval_result = evaluate(dataset=dataset, metrics=METRICS, run_config=run_cfg)

    df = eval_result.to_pandas()
    import pandas as pd
    # Only average columns that are actually numeric (i.e. real metric scores)
    metric_cols = [
        c for c in df.columns
        if c not in ("question", "answer", "contexts", "ground_truth")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        print("\nNo numeric scores returned — all jobs likely timed out.")
        print("Try running with a smaller model or increase timeout further.")
        return
    scores = df[metric_cols].mean().to_dict()

    print_results(scores)

    if args.output:
        save_results_csv(scores, args.output)


if __name__ == "__main__":
    main()
