# Legal Smart Assistant (RAG)

A fully local Retrieval-Augmented Generation (RAG) system for querying Nigerian Tenancy Law and the Constitution of the Federal Republic of Nigeria. Built with LangChain, Ollama, and ChromaDB — no data leaves the machine.

---

## Architecture

| Layer | Technology |
|---|---|
| Orchestration | LangChain LCEL |
| LLM Inference | Ollama (`llama3`) |
| Vector Store | ChromaDB (persisted locally) |
| Embeddings | `BAAI/bge-small-en` via HuggingFace |
| UI | Streamlit (token streaming) |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) |

---

## Key Features

**Query Contextualization** — follow-up questions are rewritten into standalone queries before retrieval, so conversation history doesn't degrade search quality.

**Local-First Privacy** — inference, embeddings, and vector search all run on-device via Ollama. No API keys required, no data sent to the cloud.

**Streaming UI** — token-level streaming via Streamlit reduces perceived latency on long answers.

**Conversation Persistence** — each conversation thread is saved as a JSON file with auto-generated titles. Threads are resumable across sessions.

**RAGAS Evaluation Harness** — 15 hand-crafted ground-truth Q&A pairs and an automated eval script measuring retrieval and generation quality.

---

## Project Structure

```
legal_smart_assistant/
├── rag_pipeline.py        # Core RAG logic — single source of truth
│                          # (embeddings, vector store, retrieval chain)
├── app.py                 # Streamlit UI — imports from rag_pipeline
├── ingest.py              # CLI interface — imports from rag_pipeline
├── requirements.txt
├── data/
│   ├── Tenancy Law 2011.pdf
│   └── Constitution-of-the-Federal-Republic-of-Nigeria-2023.pdf
└── eval/
    ├── eval.py            # RAGAS evaluation script
    └── qa_pairs.json      # 15 ground-truth Q&A pairs
```

---

## Getting Started

**1. Prerequisites**

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- `llama3` model pulled: `ollama pull llama3`

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your PDF source documents**

Place the following files in the `data/` folder:
- `Tenancy Law 2011.pdf`
- `Constitution-of-the-Federal-Republic-of-Nigeria-2023.pdf`

**4. Run the app**

```bash
streamlit run app.py
```

The vector DB is built automatically on first run. To force a rebuild:

```bash
python ingest.py --rebuild
```

**5. CLI mode**

```bash
python ingest.py
```

---

## Evaluation

The `eval/` folder contains a RAGAS evaluation harness for measuring RAG quality across four metrics:

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in retrieved context? (anti-hallucination) |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Were the retrieved chunks relevant? |
| Context Recall | Did retrieval surface all necessary chunks? |

**Run evaluation:**

```bash
python eval/eval.py

# Save results to CSV
python eval/eval.py --output eval/results.csv

# Force-rebuild vector DB before evaluating
python eval/eval.py --rebuild
```

Target scores: **> 0.7** across all metrics for a production-ready RAG system.

---

## Corpus

The system is grounded in two primary legal documents:

- **Lagos Tenancy Law 2011** — governs landlord-tenant relationships, eviction procedures, rent increases, and dispute resolution in Lagos State.
- **Constitution of the Federal Republic of Nigeria (2023)** — full constitutional text including Chapter IV fundamental rights.

---

## Roadmap

- [ ] Publish corpus to HuggingFace as `Augusta02/nigerian-legal-corpus`
- [ ] Add a reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve retrieval precision
- [ ] Increase retrieval `k` from 3 to 5 and re-evaluate
- [ ] Add source citation display in the Streamlit UI (show which document/page each answer came from)
