# Legal Smart Assistant

A RAG system that helps everyday Nigerians understand their legal rights — built to answer questions like "my landlord raised my rent with one month's notice, what can I do?" in plain language, grounded in the actual statute.

Built with LangChain, ChromaDB, HuggingFace embeddings, and a custom evaluation harness. Runs locally via Ollama or on the cloud via Groq — same codebase, zero config change.

Live demo: [smart-legal-assistant.streamlit.app](https://smartlegalassistant.streamlit.app/)

---

## The Problem

Access to legal information in Nigeria is a real barrier. Lawyers are expensive, statutes are dense, and most people don't know what their rights are under the tenancy law or constitution. This assistant makes the law readable: it retrieves the relevant sections and explains what they mean for your specific situation, with citations back to the source.

---

## Technical Highlights

**Best-of-N agentic retrieval** — rather than retrieving once and hoping for the best, the system generates two different query angles per question, retrieves context and generates a full answer for each angle independently, then uses a judge prompt to select the best answer. This avoids the dilution that comes from merging all retrieved chunks into one context window.

**Hybrid BM25 + semantic retrieval** — pure semantic search misses exact statutory terms ("Section 13", "Rent Tribunal", "reasonable enjoyment") because embeddings generalise meaning. Adding BM25 keyword matching alongside vector search catches exact-match terms, improving context recall on statute-specific language without sacrificing semantic coverage.

**Dual LLM provider** — `get_llm()` auto-detects `GROQ_API_KEY` and switches between Groq (cloud) and Ollama (local) with no code change. Groq uses `meta-llama/llama-4-scout-17b-16e-instruct` for 30K TPM headroom on the free tier; local uses `llama3.2:3b`.

**Corpus boundary enforcement** — the system prompt explicitly defines the document corpus and instructs the model not to cite sections, fines, or procedures from laws not in the vector store. This prevents the model from confidently hallucinating content from laws it has seen in training (e.g. VAPP 2015) but has not retrieved.

**Retrieval hyperparameter search without LLM calls** — `eval/k_search.py` sweeps `RETRIEVER_K` values across the ground-truth QA pairs using only the retriever, scoring recall and precision at each K value and writing the best K directly to `rag_pipeline.py`. No generation calls means the full sweep runs in seconds.

**Custom lightweight evaluator** — `eval/lightweight_eval.py` measures four RAG quality dimensions using embeddings and token overlap, with no LLM judge. This keeps evaluation fast, free, and deterministic. Results are cacheable so retrieval metrics and generation metrics can be iterated independently.

---

## Architecture

| Layer | Technology |
|---|---|
| LLM — cloud | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq |
| LLM — local | `llama3.2:3b` via Ollama |
| Retrieval | Hybrid BM25 + ChromaDB semantic, best-of-N agentic loop |
| Vector Store | ChromaDB — persisted locally |
| Embeddings | `BAAI/bge-small-en` via HuggingFace |
| Orchestration | LangChain LCEL (modular packages only — no monolithic `langchain`) |
| UI | Streamlit with conversation persistence and source citations |
| Evaluation | Custom lightweight harness — 4 metrics, no LLM-judge |

---

## Evaluation

Four metrics, all computed without an LLM judge. Evaluation runs against 15 hand-crafted ground-truth Q&A pairs covering both the Tenancy Law and Constitution.

| Metric | What it measures | Method |
|---|---|---|
| Answer Relevancy | Is the answer on-topic? | Cosine similarity between question and answer embeddings |
| Context Recall | Did retrieval surface the right chunks? | % of ground-truth key terms found in retrieved context |
| Context Precision | Were retrieved chunks relevant? | % of chunks sharing significant overlap with question + ground truth |
| Faithfulness | Is the answer grounded in the retrieved context? | % of answer sentences with token overlap against retrieved chunks |

**Latest results**

| Metric | Score | Target |
|---|---|---|
| Answer Relevancy | 0.911 | >= 0.70 |
| Context Precision | 0.787 | >= 0.70 |
| Faithfulness | 0.734 | >= 0.70 |
| Context Recall | 0.565 | >= 0.70 |

Context Recall is the active improvement target. Hybrid BM25 retrieval and K=30 have been applied; updated scores pending eval rerun.

**Run evaluation**

```bash
# Cached pipeline outputs (fast — skips generation):
python eval/lightweight_eval.py

# Re-run full pipeline then score:
python eval/lightweight_eval.py --no-cache

# Per-question breakdown:
python eval/lightweight_eval.py --verbose

# Save to CSV:
python eval/lightweight_eval.py --output eval/results.csv
```

**Retriever hyperparameter search**

```bash
# Sweep default K range [2, 4, 6, 8, 10, 12]:
python eval/k_search.py

# Custom range, apply best K to rag_pipeline.py automatically:
python eval/k_search.py --k-values 10 15 20 25 30 --apply
```

`k_search.py` tests retrieval only — no LLM calls. After applying a new K, regenerate the pipeline cache with `--no-cache`.

---

## Project Structure

```
legal_smart_assistant/
├── rag_pipeline.py          # Core: embeddings, hybrid retrieval, best-of-N chain, system prompt
├── app.py                   # Streamlit UI: chat, source citations, conversation sidebar
├── ingest.py                # CLI: force-rebuild the vector DB from source PDFs
├── requirements.txt
├── data/
│   ├── Tenancy Law 2011.pdf
│   └── Constitution-of-the-Federal-Republic-of-Nigeria-2023.pdf
├── conversations/           # Persisted chat threads (JSON, auto-created)
└── eval/
    ├── lightweight_eval.py  # 4-metric evaluator — no LLM judge
    ├── k_search.py          # Retriever K sweep — no LLM calls
    ├── qa_pairs.json        # 15 ground-truth Q&A pairs
    ├── pipeline_cache.json  # Cached pipeline outputs (auto-generated)
    └── results.csv          # Latest scores (generated by --output flag)
```

---

## Getting Started

**Local setup (Ollama)**

```bash
# 1. Pull the model
ollama pull llama3.2:3b

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add PDFs to data/ folder, then run
streamlit run app.py
```

The vector DB builds automatically on first run. Force a rebuild after changing chunk settings:

```bash
python ingest.py --rebuild
```

**Cloud / Groq**

Set `GROQ_API_KEY` as an environment variable or Streamlit secret. The app detects it automatically and switches to Groq — no other change needed.

---

## Corpus

| Document | Coverage |
|---|---|
| Lagos Tenancy Law 2011 | Notice periods, rent increases, advance rent limits, eviction procedures, Rent Tribunal |
| Constitution of the Federal Republic of Nigeria (2023) | Full text including Chapter IV fundamental rights |

The system prompt includes a hard corpus boundary: the model will not cite sections, penalties, or procedures from laws not in the vector store.

---

## Roadmap

**Expanded corpus** — Violence Against Persons (Prohibition) Act 2015, Labour Act, Land Use Act, other state tenancy laws (Abuja, Rivers, Oyo). Each addition directly extends what the system can answer without hallucinating.

**Reranker** — a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of hybrid retrieval to improve precision before generation, particularly for multi-part questions.

**Incremental ingestion** — add new documents without rebuilding the full vector DB. Versioned so the system can track legal changes over time.

**Legal research mode** — surfaces conflicting provisions across documents, traces how a law has changed across versions, produces structured summaries for drafting legal arguments.

**HuggingFace corpus** — publish the curated Nigerian legal corpus as `Augusta02/nigerian-legal-corpus`.

---

## Contributing

The most impactful contributions right now are corpus additions (PDFs of Nigerian statutes) and additional ground-truth Q&A pairs in `eval/qa_pairs.json`. The eval harness runs without an LLM so you can test retrieval changes in seconds.

Open an issue to discuss before opening a large PR.
