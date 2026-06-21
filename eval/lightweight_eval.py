"""
eval/lightweight_eval.py
------------------------
Lightweight evaluator for the Legal Smart Assistant.

Replaces RAGAS — no LLM-judge, no structured JSON output required.
All metrics are computed using embeddings and token overlap.

Metrics (all 0.0 → 1.0, higher is better):
  Answer Relevancy   : cosine similarity between question and answer embeddings
  Context Recall     : % of ground-truth key terms found in retrieved context
  Context Precision  : % of retrieved chunks relevant to the question
  Faithfulness       : % of answer sentences grounded in retrieved context

Run:
    python eval/lightweight_eval.py                  # use cache if available
    python eval/lightweight_eval.py --no-cache       # re-retrieve + re-answer
    python eval/lightweight_eval.py --output eval/results.csv
    python eval/lightweight_eval.py --verbose        # per-question breakdown
"""

import sys
import re
import csv
import json
import argparse
import numpy as np
import nltk
from pathlib import Path
from sentence_transformers import SentenceTransformer

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Project root on path so rag_pipeline imports cleanly
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Paths
CACHE_PATH      = Path(__file__).parent / "pipeline_cache.json"
QA_PATH         = Path(__file__).parent / "qa_pairs.json"
EMBEDDING_MODEL = "BAAI/bge-small-en"
STOPWORDS       = set(stopwords.words("english"))


#Text helpers 

def tokenize(text: str) -> set[str]:
    """Lowercase words, strip stopwords and short tokens."""
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (rough but good enough for overlap checks)."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]


#  Metrics 

def score_answer_relevancy(question: str, answer: str, model: SentenceTransformer) -> float:
    """
    Cosine similarity between question embedding and answer embedding.
    High score → answer stays on-topic.
    """
    q_emb, a_emb = model.encode([question, answer], normalize_embeddings=True)
    return float(np.clip(np.dot(q_emb, a_emb), 0.0, 1.0))


def score_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    % of ground-truth key terms found anywhere in the retrieved context chunks.
    High score → retrieval covered what was needed to answer correctly.
    """
    gt_terms = tokenize(ground_truth)
    if not gt_terms:
        return 1.0
    ctx_terms = tokenize(" ".join(contexts))
    return len(gt_terms & ctx_terms) / len(gt_terms)


def score_context_precision(question: str, ground_truth: str, contexts: list[str]) -> float:
    """
    % of retrieved chunks that are relevant (share ≥ 3 content words with
    the question + ground truth combined).
    High score → retrieval was precise, not noisy.
    """
    if not contexts:
        return 0.0
    target_terms = tokenize(question) | tokenize(ground_truth)
    relevant = sum(
        1 for chunk in contexts
        if len(tokenize(chunk) & target_terms) >= 3
    )
    return relevant / len(contexts)


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    % of answer sentences that share ≥ 3 content words with the retrieved context.
    High score → answer is grounded in retrieved text, not hallucinated.
    """
    sents = split_sentences(answer)
    if not sents:
        return 1.0
    ctx_terms = tokenize(" ".join(contexts))
    grounded = sum(
        1 for s in sents
        if len(tokenize(s) & ctx_terms) >= 3
    )
    return grounded / len(sents)


# Pipeline runner (re-retrieve + re-answer without going through eval.py)

def build_cache(qa_path: Path, cache_path: Path) -> list[dict]:
    """Re-run retrieval and generation for all QA pairs, save to cache."""
    from rag_pipeline import setup_rag_system

    print("Loading RAG system (this may take a moment)...")
    rag_chain, _, retriever = setup_rag_system()

    qa_pairs = json.loads(qa_path.read_text())
    records  = []
    print(f"Running pipeline on {len(qa_pairs)} questions...\n")
    for i, item in enumerate(qa_pairs, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i:02d}/{len(qa_pairs)}] {q[:65]}...")
        contexts = [d.page_content for d in retriever.invoke(q)]
        answer   = rag_chain.invoke({"question": q, "chat_history": ""})
        records.append({"question": q, "answer": answer, "contexts": contexts, "ground_truth": gt})

    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nCache saved → {cache_path.name}\n")
    return records


# Per-record evaluation

def evaluate_record(record: dict, model: SentenceTransformer) -> dict[str, float]:
    q   = record["question"]
    a   = record["answer"]
    gt  = record["ground_truth"]
    ctx = record["contexts"]
    return {
        "answer_relevancy":  score_answer_relevancy(q, a, model),
        "context_recall":    score_context_recall(gt, ctx),
        "context_precision": score_context_precision(q, gt, ctx),
        "faithfulness":      score_faithfulness(a, ctx),
    }

# Output helpers

def print_results(scores: dict[str, float], per_question: list[dict] | None = None) -> None:
    labels = {
        "answer_relevancy":  "Answer Relevancy   (on-topic answers)   ",
        "context_recall":    "Context Recall     (retrieval coverage)  ",
        "context_precision": "Context Precision  (retrieval accuracy)  ",
        "faithfulness":      "Faithfulness       (anti-hallucination)  ",
    }

    print("\n" + "=" * 62)
    print("  Lightweight Evaluation Results")
    print("=" * 62)
    for key, score in scores.items():
        label  = labels.get(key, key)
        filled = int(score * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        flag   = "✓" if score >= 0.7 else "✗"
        print(f"  {flag} {label}: {score:.3f}  [{bar}]")
    print("=" * 62)
    print("  Scores: 0.0 (worst) → 1.0 (best).  Target: ≥ 0.70\n")

    if per_question:
        print("  Per-question breakdown:")
        print(f"  {'#':>2}  {'AR':>5}  {'CR':>5}  {'CP':>5}  {'FA':>5}  Question")
        print("  " + "-" * 70)
        for i, row in enumerate(per_question, 1):
            q_short = row["question"][:45].ljust(45)
            print(
                f"  {i:>2}  "
                f"{row['answer_relevancy']:.3f}  "
                f"{row['context_recall']:.3f}  "
                f"{row['context_precision']:.3f}  "
                f"{row['faithfulness']:.3f}  "
                f"{q_short}"
            )
        print()


def save_csv(scores: dict[str, float], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "score"])
        writer.writeheader()
        for k, v in scores.items():
            writer.writerow({"metric": k, "score": round(v, 4)})
    print(f"Results saved → {path}")


# Main 

def main():
    parser = argparse.ArgumentParser(
        description="Lightweight RAG evaluator — no LLM-judge required."
    )
    parser.add_argument("--output",   type=str, default=None, help="Save scores to CSV")
    parser.add_argument("--verbose",  action="store_true",    help="Show per-question breakdown")
    parser.add_argument("--no-cache", action="store_true", dest="no_cache",
                        help="Re-run retrieval and generation, ignoring existing cache")
    args = parser.parse_args()

    # Load or build pipeline outputs
    if args.no_cache or not CACHE_PATH.exists():
        if args.no_cache:
            print("--no-cache: rebuilding pipeline outputs...\n")
        else:
            print("No cache found — running pipeline...\n")
        records = build_cache(QA_PATH, CACHE_PATH)
    else:
        print("Loading pipeline cache...")
        records = json.loads(CACHE_PATH.read_text())
        print(f"  {len(records)} questions loaded.  (pass --no-cache to re-run)\n")

    # Load embedding model (used for Answer Relevancy only)
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Ready.\n")

    # Evaluate
    print(f"Evaluating {len(records)} questions...")
    per_question = []
    agg: dict[str, list[float]] = {
        "answer_relevancy": [],
        "context_recall": [],
        "context_precision": [],
        "faithfulness": [],
    }

    for i, record in enumerate(records, 1):
        row_scores = evaluate_record(record, model)
        per_question.append({"question": record["question"], **row_scores})
        for k, v in row_scores.items():
            agg[k].append(v)
        print(f"  [{i:02d}/{len(records)}] done — AR={row_scores['answer_relevancy']:.2f} "
              f"CR={row_scores['context_recall']:.2f} "
              f"CP={row_scores['context_precision']:.2f} "
              f"FA={row_scores['faithfulness']:.2f}")

    mean_scores = {k: float(np.mean(v)) for k, v in agg.items()}

    print_results(mean_scores, per_question if args.verbose else None)

    if args.output:
        save_csv(mean_scores, args.output)


if __name__ == "__main__":
    main()
