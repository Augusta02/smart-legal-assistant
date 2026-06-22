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

st.markdown("""
<style>
/* Lock sidebar — no resize */
[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebarResizeHandle"] { display: none !important; }

/* Row alignment */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 4px !important;
    margin-bottom: 2px !important;
}

/* Title buttons: no box, left-aligned, theme-aware color */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    justify-content: flex-start !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: inherit !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 2px 4px !important;
    height: auto !important;
    min-height: unset !important;
    line-height: 1.5 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] > div {
    justify-content: flex-start !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] p {
    text-align: left !important;
    width: 100% !important;
    margin: 0 !important;
    font-weight: 700 !important;
    color: inherit !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: transparent !important;
    opacity: 0.7 !important;
}

/* Delete button — no box at all, plain × */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > *:last-child button,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > *:last-child button[kind="secondary"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-width: 0 !important;
    border-style: none !important;
    border-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
    color: inherit !important;
    opacity: 0.45 !important;
    font-size: 11px !important;
    padding: 0 2px !important;
    height: auto !important;
    min-height: unset !important;
    line-height: 1.5 !important;
    justify-content: center !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > *:last-child button:hover {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-width: 0 !important;
    opacity: 0.8 !important;
}
</style>
""", unsafe_allow_html=True)

#  Conversation helpers 

def get_thread_id():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id


def save_conversation(thread_id, messages):
    thread_path = Path(CONVERSATIONS_DIR) / f"{thread_id}.json"

    # (prevents overwriting a real title with "Untitled" after a conversation is loaded)
    title = st.session_state.get(f"title_{thread_id}")
    if not title and thread_path.exists():
        try:
            with open(thread_path, "r") as f:
                title = json.load(f).get("title")
        except Exception:
            pass
    title = title or "Untitled Conversation"

    data = {
        "thread_id": thread_id,
        "created_at": st.session_state.get(f"created_{thread_id}", datetime.datetime.now().isoformat()),
        "updated_at": datetime.datetime.now().isoformat(),
        "messages": messages,
        "title": title,
    }
    with open(thread_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_title(first_user_message, llm):
    """Generate a short conversation title with retry on rate-limit errors."""
    title_prompt = ChatPromptTemplate.from_template(
        "Generate a very short title (max 5 words, no quotes) for a conversation that starts with this question: {question}\n\nTitle:"
    )
    title_chain = title_prompt | llm | StrOutputParser()
    for attempt in range(3):
        try:
            title = title_chain.invoke({"question": first_user_message})
            return title.strip().strip('"').strip("'")[:60]
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    # Fallback: first 6 words of the user message
    words = first_user_message.split()
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "")


def load_conversation(thread_id):
    thread_path = Path(CONVERSATIONS_DIR) / f"{thread_id}.json"
    if thread_path.exists():
        with open(thread_path, "r") as f:
            data = json.load(f)
        # Restore title to session state so save_conversation doesn't lose it
        if "title" in data:
            st.session_state[f"title_{thread_id}"] = data["title"]
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
    # Best-of-N chain generates candidates then judges — returns full string.
    # We fake-stream it word by word so the UI doesn't stall on a blank screen.
    try:
        result = rag_chain.invoke({"question": question, "chat_history": history_str})
    except Exception as e:
        error_msg = f"**Error from LLM provider:** {type(e).__name__}: {e}"
        yield error_msg
        return
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
    st.markdown("### Conversations")
    if st.button("New conversation", use_container_width=True, type="primary"):
        create_new_conversation()

    conversations = list_conversations()
    if conversations:
        st.markdown("---")
        st.caption("RECENT")
        for conv in conversations[:10]:
            is_active = conv["thread_id"] == st.session_state.get("thread_id")
            title = conv["title"].strip()
            # Truncate cleanly at word boundary
            if len(title) > 28:
                title = title[:28].rsplit(" ", 1)[0] + "…"

            col_title, col_del = st.columns([8, 1])
            with col_title:
                if st.button(title, use_container_width=True, key=f"load_{conv['thread_id']}"):
                    st.session_state.thread_id = conv["thread_id"]
                    st.session_state.messages = load_conversation(conv["thread_id"])
                    st.rerun()
            with col_del:
                if st.button("✕", key=f"del_{conv['thread_id']}", help="Delete"):
                    delete_conversation(conv["thread_id"])
                    st.rerun()

# Input
# Chat input is always rendered so it never disappears on mobile during reruns
placeholder = "Ask a question about Nigerian law..." if not st.session_state.messages else "Ask a follow-up..."
chat_input = st.chat_input(placeholder)

# Show suggestion pills only on empty conversation
selected_suggestion = None
if not st.session_state.messages:
    selected_suggestion = st.pills(
        label="Examples",
        label_visibility="collapsed",
        options=SUGGESTIONS.keys(),
        key="selected_suggestion",
    )

user_message = chat_input or (SUGGESTIONS.get(selected_suggestion) if selected_suggestion else None)

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
                f"User: {m['content']}"
                for m in st.session_state.messages[-(HISTORY_LENGTH * 2):]
                if m["role"] == "user"
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
