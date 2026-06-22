"""
rag_pipeline.py
---------------
Shared core for the Smart Legal Assistant.
Both ingest.py (CLI) and app.py (Streamlit UI) import from here.
This is the single source of truth for all RAG configuration.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pathlib import Path
import os
import shutil
import time

# Paths and constants
_ROOT = Path(__file__).resolve().parent
DB_PATH = str(_ROOT / "local_chroma_db")
MODEL = "llama3.2:3b"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL = "BAAI/bge-small-en"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 30

PDF_SOURCES = [
    str(_ROOT / "data" / "Tenancy Law 2011.pdf"),
    str(_ROOT / "data" / "Constitution-of-the-Federal-Republic-of-Nigeria-2023.pdf"),
]

SYSTEM_INSTRUCTIONS = """
You are a friendly legal guide helping everyday Nigerians understand their rights.
Speak plainly — like a knowledgeable friend, not a lawyer reading from a textbook.

KEY LEGAL FACTS from Lagos Tenancy Law 2011 — for more context:

NOTICE TO QUIT (Section 13) — notice required to END a tenancy:
- Yearly tenant → 6 months notice minimum 
- Monthly tenant → 1 month notice minimum 
- Quarterly tenant → 3 months notice 
- Half-yearly tenant → 3 months notice 
If a landlord's notice to quit is shorter than these, it is INVALID. Say so directly.

RENT INCREASES (Section 37):
- The law does NOT set a specific advance notice period just for a rent increase.
- BUT if the landlord's notice of increase is also a notice to quit (leave if you don't accept),
  then Section 13 notice periods apply — 6 months for a yearly tenant, 1 month for monthly.
- A tenant can apply to the Rent Tribunal/Court to declare any increase UNREASONABLE (Section 37).
- There is NO cap on how much rent can be increased — but the increase can be challenged.

ADVANCE RENT (Section 4):
- Landlord cannot demand more than 1 year advance from a yearly tenant, or 6 months from a monthly tenant.
- Section 4 is about advance rent limits ONLY — not about notice or increase amounts.

DISPUTES: Go to the Rent Tribunal — faster and cheaper than regular court.

CORPUS BOUNDARY — CRITICAL:
Your only source documents are the Lagos Tenancy Law 2011 and the 2023 Nigerian Constitution.
If asked about any other law (Violence Against Persons Act, Labour Act, Companies Act, Criminal Code, etc.):
- Say clearly: "I don't have that law in my documents."
- You may reference what the Constitution says about the topic generally (e.g. right to dignity under Section 34).
- Do NOT cite specific sections, fines, penalties, or procedures from laws not in your corpus.
- Do NOT provide information from external resources — you cannot verify them.

HOW TO ANSWER:
1. Answer every question asked separately. Three questions = three clear answers.
2. Do not show maths when amounts are mentioned but make sure calculations are done correctly. For example, if rent was ₦1.5M and increased to ₦4M, that's a ₦2.5M increase — which is a 167% increase. So you would say: "Your rent was increased by ₦2.5M, which is a 167% increase." Not just "Your rent was increased by ₦2.5M."
   e.g. ₦1.5M to ₦4M → ₦2.5M more → (2.5 ÷ 1.5) × 100 = 167% increase.
3. Be direct.
4. Close every response with a "What would you like to do next?" section:
   - If there is a clear next step (e.g. file at the Tribunal, send a letter), state it first in one sentence.
   - Then offer 2-3 short follow-up questions the person can click or respond to, tailored to what was just discussed.
   - Format them as a numbered list starting with "You might also want to know:".
   - Example: "You might also want to know:
     1. How do I file a complaint at the Rent Tribunal?
     2. Can my landlord evict me while I am disputing the increase?
     3. What counts as evidence if this goes to the Tribunal?"
5. Use retrieved context only for supporting section numbers — not to override the facts above.
6. If something is genuinely unclear, say so. Do not guess.
"""

# Shared utilities

def format_docs(docs):
    """Join retrieved document chunks into a single context string."""
    return "\n\n".join([doc.page_content for doc in docs])


def get_embeddings():
    """Return the shared HuggingFace embeddings instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm(temperature: float = 0.7):
    """
    Return the LLM instance.
    - If GROQ_API_KEY is set (Streamlit Cloud / any cloud deployment) → use Groq.
    - Otherwise → use local Ollama.
    """
    groq_key = os.environ.get("GROQ_API_KEY") or _get_streamlit_secret("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=groq_key)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=MODEL, temperature=temperature)


def _get_streamlit_secret(key: str) -> str | None:
    """Read a key from st.secrets without hard-importing streamlit."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


# Vector store 
def build_vector_store(embeddings, force_rebuild: bool = False):
    """
    Load the persisted ChromaDB if it exists, otherwise build it from PDFs.

    Args:
        embeddings: HuggingFaceEmbeddings instance.
        force_rebuild: If True, delete and rebuild the DB from scratch.

    Returns:
        A Chroma vector store instance.
    """
    db_exists = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 1

    if db_exists and not force_rebuild:
        return Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
        )

    # Rebuild from source PDFs
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    pages = []
    for path in PDF_SOURCES:
        pages += PyPDFLoader(path).load()

    if not pages:
        raise ValueError(
            f"No documents loaded — check that all PDF paths exist: {PDF_SOURCES}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.split_documents(pages)

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )


def get_retriever(vector_store):
    """
    Return an ensemble retriever combining BM25 (keyword) + ChromaDB (semantic).

    Why hybrid:
      Pure semantic search misses exact legal terms because embeddings generalise meaning. BM25 catches
      exact-match terms; the ensemble re-ranks results from both, improving recall
      on statute-specific language without sacrificing semantic coverage.
    """
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever

        # Re-chunk the source PDFs for BM25 — no embeddings needed, completes in seconds
        pages = []
        for path in PDF_SOURCES:
            if os.path.exists(path):
                pages += PyPDFLoader(path).load()

        if pages:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            docs = splitter.split_documents(pages)
            bm25_retriever = BM25Retriever.from_documents(docs, k=RETRIEVER_K)

            return EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.6, 0.4], 
            )
    except ImportError:
        pass  

    return semantic_retriever


# RAG chain

def build_rag_chain(llm, retriever):
    """
    Best-of-N agentic RAG chain.

    For each question the chain:
      1. Optionally rewrites the question into a standalone query (using chat history).
      2. Generates 3 different query angles from the rewritten question.
      3. Retrieves context and generates a full answer for each angle independently.
      4. A judge prompt picks the best answer among the 3 candidates.
      5. Returns the winning answer.

    Why best-of-N instead of merged context:
      Merging all retrieved chunks into one prompt dilutes the context and the
      model has to reconcile competing passages. Generating separate answers per
      angle and then judging produces a more focused, accurate response — each
      candidate uses only the context most relevant to its query framing.
    """

    # question rewriter 
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Given the chat history and the latest user question,
            rewrite it into a standalone question.
            Do NOT answer it. Only rewrite if necessary."""),
        ("human", "Chat history:\n{chat_history}\n\nQuestion:\n{question}"),
    ])
    _rewrite_chain = contextualize_prompt | llm | StrOutputParser()

    def _maybe_rewrite(inputs: dict) -> str:
        if not inputs.get("chat_history", "").strip():
            return inputs["question"]
        return _rewrite_chain.invoke(inputs)

    #  query angle generator 
    _query_gen_prompt = ChatPromptTemplate.from_template(
        """You are a Nigerian legal research assistant. Given a question, generate
        3 different search queries covering different angles needed to fully answer it.
        Each query should target a distinct aspect (e.g. rights, procedures, penalties).

        Return ONLY 3 queries, one per line, no numbering or labels.

        Question: {question}
        Queries:"""
    )
    _query_chain = _query_gen_prompt | llm | StrOutputParser()

    #  answer prompt 
    _answer_prompt = ChatPromptTemplate.from_template(
        f"""{SYSTEM_INSTRUCTIONS}

        Context: {{context}}
        History: {{chat_history}}
        Question: {{question}}

        Answer:"""
    )
    _answer_chain = _answer_prompt | llm | StrOutputParser()

    #  judge prompt 
    _judge_prompt = ChatPromptTemplate.from_template(
        """You are evaluating three candidate answers to a legal question about Nigerian law.
            Choose the single best answer based on:
            1. Legal accuracy — correctly identifies the applicable law and what it actually says
            2. Completeness — addresses every part of the question asked
            3. Clarity — explains things plainly without unnecessary jargon
            4. Practical guidance — tells the person what to actually do next

            Question: {question}

            {candidates}

            Return ONLY the full text of the best answer. Do not add any prefix like "Answer 2" or
            "The best answer is". Just return the answer itself."""
    )
    _judge_chain = _judge_prompt | llm | StrOutputParser()

    def _invoke_with_retry(chain, inputs: dict, retries: int = 3) -> str:
        """Invoke a chain with exponential backoff on rate-limit errors."""
        for attempt in range(retries):
            try:
                return chain.invoke(inputs)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)

    # Full chain
    def _best_of_n(inputs: dict) -> str:
        question     = inputs["question"]
        chat_history = inputs.get("chat_history", "")
        rewritten    = _maybe_rewrite(inputs)

        # Generate 2 query angles (reduced from 3 to stay)
        raw     = _invoke_with_retry(_query_chain, {"question": rewritten})
        queries = [q.strip() for q in raw.strip().splitlines() if q.strip()][:2]
        if not queries:
            queries = [rewritten]
        # Deduplicate while preserving order; always include the rewritten original
        seen, all_queries = set(), []
        for q in [rewritten] + queries:
            if q not in seen:
                seen.add(q)
                all_queries.append(q)
        all_queries = all_queries[:2]

        # Generate one answer per query angle
        candidates = []
        for q in all_queries:
            docs    = retriever.invoke(q)
            context = format_docs(docs)
            answer  = _invoke_with_retry(_answer_chain, {
                "context":      context,
                "question":     question,
                "chat_history": chat_history,
            })
            candidates.append(answer)

        # If only one candidate (generation failed), return it directly
        if len(candidates) == 1:
            return candidates[0]

        # Judge picks the best
        candidates_text = "\n\n---\n\n".join(
            f"Answer {i + 1}:\n{a}" for i, a in enumerate(candidates)
        )
        return _invoke_with_retry(_judge_chain, {
            "question":   question,
            "candidates": candidates_text,
        })

    return RunnableLambda(_best_of_n)

 

def setup_rag_system(force_rebuild: bool = False):
    """
    Build and return the complete RAG system.

    Args:
        force_rebuild: Pass True to wipe and rebuild the vector DB.

    Returns:
        Tuple of (rag_chain, llm, retriever).
        retriever is exposed so eval.py can inspect retrieved docs directly.
    """
    embeddings = get_embeddings()
    llm = get_llm()
    vector_store = build_vector_store(embeddings, force_rebuild=force_rebuild)
    retriever = get_retriever(vector_store)
    rag_chain = build_rag_chain(llm, retriever)

    return rag_chain, llm, retriever
