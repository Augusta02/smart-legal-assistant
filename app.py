import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import os
import shutil
import datetime
import time
import json
import uuid
from pathlib import Path

st.set_page_config(page_title="Smart Legal Assistant", page_icon="⚖️")

# Constants
DB = "./local_chroma_db"
CONVERSATIONS_DIR = "./conversations"
MODEL = "llama3"
HISTORY_LENGTH = 5
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)

# Ensure conversations directory exists
Path(CONVERSATIONS_DIR).mkdir(exist_ok=True)

# Conversation management functions
def get_thread_id():
    """Get or create a thread ID for the current conversation."""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id

def save_conversation(thread_id, messages):
    """Save conversation to JSON file."""
    thread_path = Path(CONVERSATIONS_DIR) / f"{thread_id}.json"
    data = {
        "thread_id": thread_id,
        "created_at": st.session_state.get(f"created_{thread_id}", datetime.datetime.now().isoformat()),
        "updated_at": datetime.datetime.now().isoformat(),
        "messages": messages,
        "title": st.session_state.get(f"title_{thread_id}", "Untitled Conversation")
    }
    with open(thread_path, "w") as f:
        json.dump(data, f, indent=2)

def generate_title(first_user_message, llm):
    """Generate a concise one-line title based on the first user message."""
    try:
        title_prompt = ChatPromptTemplate.from_template(
            "Given this user question, generate a very short (max 6 words) title: {question}\n\nTitle:"
        )
        title_chain = title_prompt | llm | StrOutputParser()
        title = title_chain.invoke({"question": first_user_message})
        return title.strip()[:50]  
    except:
        return first_user_message[:50]  

def load_conversation(thread_id):
    """Load conversation from JSON file."""
    thread_path = Path(CONVERSATIONS_DIR) / f"{thread_id}.json"
    if thread_path.exists():
        with open(thread_path, "r") as f:
            data = json.load(f)
        return data.get("messages", [])
    return []

def list_conversations():
    """List all saved conversations."""
    conversations = []
    for file in Path(CONVERSATIONS_DIR).glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                conversations.append({
                    "thread_id": data["thread_id"],
                    "title": data.get("title", "Untitled"),
                    "updated_at": data.get("updated_at", ""),
                    "preview": data["messages"][-1]["content"][:50] if data["messages"] else "Empty"
                })
        except:
            continue
    return sorted(conversations, key=lambda x: x["updated_at"], reverse=True)

def delete_conversation(thread_id):
    """Delete a conversation."""
    thread_path = Path(CONVERSATIONS_DIR) / f"{thread_id}.json"
    if thread_path.exists():
        thread_path.unlink()


def create_new_conversation():
    """Create a new conversation thread."""
    new_thread_id = str(uuid.uuid4())
    st.session_state.thread_id = new_thread_id
    st.session_state.messages = []
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)
    st.rerun()

INSTRUCTIONS = """
You are a helpful legal assistant for answering questions about Nigerian law, specifically tenancy law and the Constitution of the Federal Republic of Nigeria.
Use the following retrieved context to answer the question. If you don't know the answer, say you don't know.
Answer concisely and accurately based on the context provided. If the context does not contain the answer, say you don't know.
Use markdown for formatting, provide examples if relevant, and be clear.
"""



SUGGESTIONS = {
    "⚖️ What is tenancy law in Lagos?": "Explain tenancy law in Lagos, Nigeria.",
    "📜 Rights of tenants": "What are the rights of tenants under Nigerian tenancy law?",
    "🏠 Landlord responsibilities": "What are the responsibilities of landlords in Nigeria?",
    "📋 How to evict a tenant": "What is the legal process for evicting a tenant in Lagos?",
    "🇳🇬 Constitution overview": "Give an overview of the Nigerian Constitution.",
    "📜 Rights of Nigerian Citizens": "What are my rights as a Nigerian Citizen?"
}

@st.cache_resource
def setup_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # llm = OllamaLLM(model=MODEL)
    llm = ChatOllama(model='phi3:mini', temperature=0.7)
    
    # Load documents if not already in DB
    if not (os.path.exists(DB) and len(os.listdir(DB)) > 1):
        if os.path.exists(DB):
            shutil.rmtree(DB)
        loader1 = PyPDFLoader("./data/Tenancy Law 2011.pdf")
        loader2 = PyPDFLoader("./data/Constitution-of-the-Federal-Republic-of-Nigeria-2023.pdf")
        pages = loader1.load() + loader2.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(pages)
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB
        )
    else:
        vector_store = Chroma(
            persist_directory=DB,
            embedding_function=embeddings
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """Given the chat history and the latest user question,
    rewrite the question so it becomes a standalone question.

    Do NOT answer the question.
    Only rewrite it if necessary."""
            ),
            ("human", "Chat history:\n{chat_history}\n\nQuestion:\n{question}")
        ]
    )

    question_rewrite_chain = (
        contextualize_q_prompt
        | llm
        | StrOutputParser()
    )

    rag_prompt = ChatPromptTemplate.from_template(f"""
    {INSTRUCTIONS}

    Context: {{context}}
    History: {{chat_history}}
    Question: {{question}}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

  

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

rag_chain = setup_rag_system()

def get_response(question, chat_history):
    history_str = "\n".join(chat_history[-HISTORY_LENGTH:])
    return rag_chain.stream({"question": question, "chat_history": history_str})

# UI
st.title("⚖️ Smart Legal Assistant")
st.caption("Ask questions about Nigerian law, tenancy law, and the Constitution.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar for thread management
with st.sidebar:
    st.header("🗣️ Conversations")
    
    col1 = st.columns(1)[0]
    with col1:
        if st.button("➕ New", use_container_width=True):
            create_new_conversation()
    
    st.divider()
    
    conversations = list_conversations()
    if conversations:
        st.subheader("Recent")
        for conv in conversations[:10]:
            col_title, col_del = st.columns([4, 1])
            with col_title:
                if st.button(f"{conv['title'][:30]}", use_container_width=True, key=f"load_{conv['thread_id']}"):
                    st.session_state.thread_id = conv["thread_id"]
                    st.session_state.messages = load_conversation(conv["thread_id"])
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{conv['thread_id']}"):
                    delete_conversation(conv["thread_id"])
                    st.rerun()

# Show suggestions if no messages
if not st.session_state.messages:
    selected_suggestion = st.pills(
        label="Examples",
        label_visibility="collapsed",
        options=SUGGESTIONS.keys(),
        key="selected_suggestion",
    )
    if selected_suggestion:
        user_message = SUGGESTIONS[selected_suggestion]
    else:
        user_message = st.chat_input("Ask a question about Nigerian law...")
else:
    user_message = st.chat_input("Ask a follow-up...")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_message:
    # Rate limiting
    question_timestamp = datetime.datetime.now()
    time_diff = question_timestamp - st.session_state.prev_question_timestamp
    if time_diff < MIN_TIME_BETWEEN_REQUESTS:
        time.sleep((MIN_TIME_BETWEEN_REQUESTS - time_diff).total_seconds())
    st.session_state.prev_question_timestamp = question_timestamp

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_message)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_gen = get_response(user_message,chat_history = [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
            response = st.write_stream(response_gen)

    # Add to history
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Generate title for first message
    if len(st.session_state.messages) == 2:  # First user-assistant pair
        llm = ChatOllama(model='phi3:mini', temperature=0.7)
        title = generate_title(user_message, llm)
        st.session_state[f"title_{get_thread_id()}"] = title
    
    # Auto-save conversation
    save_conversation(get_thread_id(), st.session_state.messages)