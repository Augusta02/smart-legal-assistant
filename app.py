"""
app.py
------
Streamlit UI for the Legal Research Assistant.
All RAG logic lives in rag_pipeline.py — this file handles only the UI.
"""

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import datetime
import time
import json
import uuid
from pathlib import Path
from rag_pipeline import setup_rag_system, MODEL


# Constants 
CONVERSATIONS_DIR = "./conversations"
HISTORY_LENGTH = 5
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)
Path(CONVERSATIONS_DIR).mkdir(exist_ok=True)
st.set_page_config(page_title="Legal Research Assistant")

#  Conversation helpers 

def get_thread_id():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id


def save_conversation(thread_id, messages):
    thread_path = Path(CONVERSATIONS_DIR) / f"{thread_id}.json"
    data = {
        "thread_id": thread_id,
        "created_at": st.session_state.get(f"created_{thread_id}", datetime.datetime.now().isoformat()),
        "updated_at": datetime.datetime.now().isoformat(),
        "messages": messages,
        "title": st.session_state.get(f"title_{thread_id}", "Untitled Conversation"),
    }
    with open(thread_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_title(first_user_message, llm):
    try:
        title_prompt = ChatPromptTemplate.from_template(
            "Given this user question, generate a very short (max 6 words) title: {question}\n\nTitle:"
        )
        title_chain = title_prompt | llm | StrOutputParser()
        title = title_chain.invoke({"question": first_user_message})
        return title.strip()[:60]
    except Exception as e:
        st.warning(f"Title generation failed: {e}")
        return first_user_message[:60]


def load_conversation(thread_id):
    thread_path = Path(CONVERSATIONS_DIR) / f"{thread_id}.json"
    if thread_path.exists():
        with open(thread_path, "r") as f:
            data = json.load(f)
        return data.get("messages", [])
    return []


def list_conversations():
    conversations = []
    for file in Path(CONVERSATIONS_DIR).glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
            conversations.append({
                "thread_id": data["thread_id"],
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at", ""),
                "preview": data["messages"][-1]["content"][:60] if data["messages"] else "Empty",
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(conversations, key=lambda x: x["updated_at"], reverse=True)


def delete_conversation(thread_id):
    thread_path = Path(CONVERSATIONS_DIR) / f"{thread_id}.json"
    if thread_path.exists():
        thread_path.unlink()


def create_new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)
    st.rerun()


#  Suggestions 

SUGGESTIONS = {
    "⚖️ What is tenancy law in Lagos?": "Explain tenancy law in Lagos, Nigeria.",
    "📜 Rights of tenants": "What are the rights of tenants under Nigerian tenancy law?",
    "🏠 Landlord responsibilities": "What are the responsibilities of landlords in Nigeria?",
    "📋 How to evict a tenant": "What is the Research process for evicting a tenant in Lagos?",
    "🇳🇬 Constitution overview": "Give an overview of the Nigerian Constitution.",
    "📜 Rights of Nigerian Citizens": "What are my rights as a Nigerian Citizen?",
}

# RAG system (cached) 

@st.cache_resource
def load_rag_system():
    """Load the RAG chain, LLM, and retriever once — cached for the app lifetime."""
    rag_chain, llm, retriever = setup_rag_system()
    return rag_chain, llm, retriever


rag_chain, llm, retriever = load_rag_system()


def get_response(question, chat_history):
    history_str = "\n".join(chat_history[-(HISTORY_LENGTH * 2):])
    # Best-of-N chain generates 3 candidates then judges — returns full string.
    # We fake-stream it word by word so the UI doesn't stall on a blank screen.
    result = rag_chain.invoke({"question": question, "chat_history": history_str})
    for word in result.split(" "):
        yield word + " "


def get_sources(question: str) -> list[dict]:
    """Retrieve source chunks for a question and return deduped citation metadata."""
    docs = retriever.invoke(question)
    seen, citations = set(), []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", None)
        # Strip path — keep just the filename stem, e.g. "Tenancy Law 2011"
        label  = Path(source).stem
        if page is not None:
            label = f"{label} — page {int(page) + 1}" 
        if label not in seen:
            seen.add(label)
            citations.append({"label": label, "snippet": doc.page_content[:200]})
    return citations


# UI 

st.title("Legal Research Assistant")
st.caption("Ask questions about 2023 Nigerian law and constitution and Lagos tenancy law. ")

# Session state defaults
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    st.header("Conversations")
    if st.button("New", use_container_width=True):
        create_new_conversation()
    st.divider()

    conversations = list_conversations()
    if conversations:
        st.subheader("Recent")
        for conv in conversations[:10]:
            col_title, col_del = st.columns([4, 1])
            with col_title:
                if st.button(conv["title"][:30], use_container_width=True, key=f"load_{conv['thread_id']}"):
                    st.session_state.thread_id = conv["thread_id"]
                    st.session_state.messages = load_conversation(conv["thread_id"])
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{conv['thread_id']}"):
                    delete_conversation(conv["thread_id"])
                    st.rerun()

# Input
if not st.session_state.messages:
    selected_suggestion = st.pills(
        label="Examples",
        label_visibility="collapsed",
        options=SUGGESTIONS.keys(),
        key="selected_suggestion",
    )
    user_message = SUGGESTIONS[selected_suggestion] if selected_suggestion else st.chat_input("Ask a question about Nigerian law...")
else:
    user_message = st.chat_input("Ask a follow-up...")

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new message
if user_message:
    # Rate limiting
    now = datetime.datetime.now()
    time_diff = now - st.session_state.prev_question_timestamp
    if time_diff < MIN_TIME_BETWEEN_REQUESTS:
        time.sleep((MIN_TIME_BETWEEN_REQUESTS - time_diff).total_seconds())
    st.session_state.prev_question_timestamp = now

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Legal Assistant Thinking..."):
            history = [
                f"{m['role'].capitalize()}: {m['content']}"
                for m in st.session_state.messages[-(HISTORY_LENGTH * 2):]
            ]
            response = st.write_stream(get_response(user_message, history))

        # Source citations
        sources = get_sources(user_message)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['label']}**")
                    st.caption(f"> {s['snippet']}…")

    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Generate title on first exchange
    if len(st.session_state.messages) == 2:
        title = generate_title(user_message, llm)
        st.session_state[f"title_{get_thread_id()}"] = title

    save_conversation(get_thread_id(), st.session_state.messages)
