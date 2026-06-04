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
    python eval/eval.py

    # Save results to a CSV:
    python eval/eval.py --output eval/results.csv

Requires:
    pip install ragas datasets
    Ollama must be running locally with llama3 pulled.
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

# Suppress RAGAS deprecation warnings for LangChain wrappers.
# These wrappers still work correctly for local models — RAGAS's suggested
# replacement (llm_factory) only supports cloud providers (OpenAI, etc.).
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# RAGAS imports 
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# Project imports 
# Add parent directory so we can import rag_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_pipeline import setup_rag_system, get_embeddings, get_llm

#  Config
QA_PATH = Path(__file__).parent / "qa_pairs.json"
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def load_qa_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_pipeline_on_questions(qa_pairs: list[dict], rag_chain, retriever) -> list[dict]:
    """
    For each Q&A pair:
      1. Retrieve the relevant chunks (contexts).
      2. Generate an answer using the RAG chain.
      3. Collect everything RAGAS needs.

    Returns a list of dicts with keys:
        question, answer, contexts, ground_truth
    """
    results = []

    for i, item in enumerate(qa_pairs, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(qa_pairs)}] {question[:70]}...")

        # Retrieve context chunks
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Generate answer (no chat history needed for eval)
        answer_chunks = []
        for chunk in rag_chain.stream({"question": question, "chat_history": ""}):
            answer_chunks.append(chunk)
        answer = "".join(answer_chunks)

        results.append({
            "question": question,
            "answer": answer,
            # list of strings — retrieved chunks
            "contexts": contexts,        
            "ground_truth": ground_truth,
        })

    return results


def build_ragas_dataset(results: list[dict]) -> Dataset:
    """Convert pipeline outputs into a HuggingFace Dataset for RAGAS."""
    return Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })


def print_results(scores: dict) -> None:
    print("\n" + "=" * 50)
    print("  RAGAS Evaluation Results")
    print("=" * 50)

    metric_names = {
        "faithfulness":      "Faithfulness       (anti-hallucination)",
        "answer_relevancy":  "Answer Relevancy   (on-topic answers)",
        "context_precision": "Context Precision  (retrieval accuracy)",
        "context_recall":    "Context Recall     (retrieval coverage)",
    }

    for key, label in metric_names.items():
        value = scores.get(key, "N/A")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None

        # filters out NaN (NaN != NaN)
        if numeric is not None and numeric == numeric:  
            filled = int(numeric * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  {label}: {numeric:.3f}  [{bar}]")
        else:
            print(f"  {label}: N/A (scorer returned NaN — LLM response was unparseable)")

    print("=" * 50)
    print("\nScores range from 0.0 (worst) to 1.0 (best).")
    print("Aim for > 0.7 across all metrics for a production-ready RAG system.\n")


def save_results_csv(scores: dict, output_path: str) -> None:
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "score"])
        writer.writeheader()
        for metric, score in scores.items():
            writer.writerow({"metric": metric, "score": round(score, 4) if isinstance(score, float) else score})
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the Legal Assistant RAG pipeline.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results as CSV")
    parser.add_argument("--rebuild", action="store_true", help="Force-rebuild the vector DB before evaluating")
    args = parser.parse_args()

    # Load pipeline
    print("Loading RAG system...")
    rag_chain, _, retriever = setup_rag_system(force_rebuild=args.rebuild)

    # Wrap LLM and embeddings for RAGAS
    # RAGAS needs to call the LLM and embeddings itself to score faithfulness etc.
    # We wrap our existing Ollama LLM and HuggingFace embeddings so RAGAS can use them.
    ragas_llm = LangchainLLMWrapper(get_llm(temperature=0))
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    for metric in METRICS:
        metric.llm = ragas_llm
        metric.embeddings = ragas_embeddings

    # Run pipeline on all questions 
    print(f"\nLoading {QA_PATH.name}...")
    qa_pairs = load_qa_pairs(QA_PATH)
    print(f"Running pipeline on {len(qa_pairs)} questions...\n")
    results = run_pipeline_on_questions(qa_pairs, rag_chain, retriever)

    # Build dataset and evaluate 
    print("\nRunning RAGAS evaluation (this may take a few minutes)...")
    dataset = build_ragas_dataset(results)
    eval_result = evaluate(dataset=dataset, metrics=METRICS)

    # Display results 
    scores = eval_result.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean().to_dict()

    print_results(scores)

    if args.output:
        save_results_csv(scores, args.output)


if __name__ == "__main__":
    main()
