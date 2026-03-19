# Legal Smart Assistant (RAG)

A Retrieval-Augmented Generation (RAG) system focused on Nigerian Tenancy Law and the Constitution.

## 🚀 Architecture
- **Orchestration:** LangChain (LCEL)
- **Inference Engine:** Ollama (Llama 3/3.2)
- **Vector Store:** ChromaDB
- **Embeddings:** BAAI/bge-small-en (Local HuggingFace)
- **Persistence:** SQLite for Chat History (Thread Management)

## 🛠️ Key Engineering Features
- **Query Contextualization:** Re-writes follow-up questions into standalone search queries to improve retrieval hits.
- **Local-First Privacy:** No data leaves the machine; inference and embeddings are handled locally via Ollama.
- **Streaming UI:** Implemented token streaming via Streamlit to reduce perceived latency.

## 📁 Project Structure
- `ingest.py`: ETL pipeline for PDF parsing and vector indexing.
- `app.py`: Streamlit frontend and RAG chain logic.
- `local_chroma_db/`: Persisted vector embeddings.