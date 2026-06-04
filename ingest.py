"""
ingest.py
---------
CLI interface for the Smart Legal Assistant.
Run this directly to chat with the RAG system in the terminal.

Usage:
    python ingest.py
    python ingest.py --rebuild   # force-rebuild the vector DB
"""

import sys
from rag_pipeline import setup_rag_system

HISTORY_LENGTH = 5


def main():
    force_rebuild = "--rebuild" in sys.argv

    print("\nLoading RAG system...")
    rag_chain, _, _ = setup_rag_system(force_rebuild=force_rebuild)
    print("Ready. Type 'exit' to quit.\n")

    chat_history = []

    while True:
        user_input = input("Ask a question about Nigerian law: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        history_str = "\n".join(chat_history[-HISTORY_LENGTH:])

        print("\nThinking...\n")
        full_response = ""
        for chunk in rag_chain.stream({
            "question": user_input,
            "chat_history": history_str,
        }):
            print(chunk, end="", flush=True)
            full_response += chunk

        print("\n")

        chat_history.append(f"Human: {user_input}")
        chat_history.append(f"AI: {full_response}")


if __name__ == "__main__":
    main()
