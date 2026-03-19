from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os 
import shutil




DB = "./local_chroma_db"
chat_history = []
llm = OllamaLLM(model="llama3")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

loader = PyPDFLoader("./data/Tenancy Law 2011.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)


if os.path.exists(DB) and len(os.listdir(DB)) > 1: 
    vector_store = Chroma(
        persist_directory=DB, 
        embedding_function=embeddings
    )
else:
    if os.path.exists(DB):
        shutil.rmtree(DB)
    if len(docs) > 0:
        vector_store = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings,
            persist_directory=DB  
        )
    

# set up retrieval system

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


rag_prompt = ChatPromptTemplate.from_template(
   """
    You are a helpful legal assistant for answering questions about tenancy law in Lagos, Nigeria. 
    Use the following retrieved context to answer the question. If you don't know the answer, say you don't know.

    Context:{context}
    History: {chat_history}
    Question:{question}

    Aswer concisely and accurately based on the context provided. If the context does not contain the answer, say you don't know.
    """
)



def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    rag_prompt | llm | StrOutputParser()
)

print("\nLocal RAG system set up successfully! You can now ask questions about tenancy law.\n")
print("Type 'exit' to quit\n")




while True:
    user_input = input("Ask a question about tenancy law: ")

    if user_input.lower() == "exit":
        break

    relevant_docs = retriever.invoke(user_input)
    context_text = format_docs(relevant_docs)
    history_str = "\n".join(chat_history[-5:])

    print("Thinking...\n") 
    full_response = ""
    for chunk in rag_chain.stream({
        "context": context_text,
        "question": user_input,
        "chat_history": history_str
    }):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n")

    chat_history.append(f"Human: {user_input}")
    chat_history.append(f"AI: {full_response}")