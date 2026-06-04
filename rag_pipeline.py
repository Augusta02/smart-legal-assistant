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
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import os
import shutil

#  Configuration 
DB_PATH = "./local_chroma_db"
MODEL = "llama3"
EMBEDDING_MODEL = "BAAI/bge-small-en"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 3

PDF_SOURCES = [
    "./data/Tenancy Law 2011.pdf",
    "./data/Constitution-of-the-Federal-Republic-of-Nigeria-2023.pdf",
]

SYSTEM_INSTRUCTIONS = """
You are a helpful legal assistant for answering questions about Nigerian law,
specifically tenancy law and the Constitution of the Federal Republic of Nigeria.
Use the following retrieved context to answer the question. If you don't know
the answer, say you don't know.
Answer concisely and accurately based on the context provided.
Use markdown for formatting, provide examples if relevant, and be clear.
"""

# Shared utilities

def format_docs(docs):
    """Join retrieved document chunks into a single context string."""
    return "\n\n".join([doc.page_content for doc in docs])


def get_embeddings():
    """Return the shared HuggingFace embeddings instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm(temperature: float = 0.7):
    """Return the shared ChatOllama LLM instance."""
    return ChatOllama(model=MODEL, temperature=temperature)


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
    """Return a retriever from the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})


# RAG chain 

def build_rag_chain(llm, retriever):
    """
    Build and return the full LCEL RAG chain.

    The chain:
      1. Rewrites the user question into a standalone query (using chat history).
      2. Retrieves relevant chunks from the vector store.
      3. Passes context + history + question to the answer prompt.
      4. Returns a streamed string response.

    Args:
        llm: ChatOllama instance.
        retriever: Chroma retriever instance.

    Returns:
        A runnable LCEL chain.
    """
    # Step 1 — question rewriter
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Given the chat history and the latest user question,
rewrite the question so it becomes a standalone question.
Do NOT answer the question. Only rewrite it if necessary."""),
        ("human", "Chat history:\n{chat_history}\n\nQuestion:\n{question}"),
    ])

    question_rewrite_chain = contextualize_prompt | llm | StrOutputParser()

    # Step 2 — answer prompt
    rag_prompt = ChatPromptTemplate.from_template(f"""
{SYSTEM_INSTRUCTIONS}

Context: {{context}}
History: {{chat_history}}
Question: {{question}}

Answer:
""")

    # Step 3 — full chain
    rag_chain = (
        RunnablePassthrough.assign(rewritten_question=question_rewrite_chain)
        | {
            "context": itemgetter("rewritten_question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

 

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
