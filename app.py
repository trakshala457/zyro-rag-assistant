# TODO: Build your Streamlit chatbot application

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")

# Sidebar config
with st.sidebar:
    st.header("Setup")
    groq_key   = st.text_input("Groq API Key", type="password")
    corpus_path = st.text_input("HR Docs Folder", value="C:\Users\hp\Documents\gen ai\zyro-dynamics-hr-corpus\docs")
    init_btn   = st.button("Initialize", use_container_width=True)
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

REFUSAL = "I can only answer questions about Zyro Dynamics HR policies."

@st.cache_resource(show_spinner="Building pipeline…")
def build_pipeline(api_key, path):
    docs   = PyPDFDirectoryLoader(path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    emb    = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ret    = FAISS.from_documents(chunks, emb).as_retriever(search_kwargs={"k": 4})
    llm    = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0)

    guard = ChatPromptTemplate.from_template(
        "Is this question about HR policies, leave, payroll, or benefits? "
        "Reply ONLY 'in_scope' or 'out_of_scope'.\nQuestion: {question}"
    ) | llm | StrOutputParser()

    rag = (
        {"context": ret | (lambda d: "\n\n".join(x.page_content for x in d)),
         "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(
            "Answer using only the context below. Be concise.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        | llm | StrOutputParser()
    )

    def ask(q):
        return rag.invoke(q) if "out" not in guard.invoke({"question": q}).lower() else REFUSAL

    return ask

# Initialize pipeline
if init_btn:
    if not groq_key:
        st.sidebar.error("Enter your Groq API key.")
    elif not os.path.isdir(corpus_path):
        st.sidebar.error("Folder not found.")
    else:
        st.session_state.pipeline = build_pipeline(groq_key, corpus_path)
        st.sidebar.success("Ready!")

# Chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask an HR question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.pipeline is None:
        answer = "⚠️ Please initialize the pipeline from the sidebar first."
    else:
        with st.spinner("Thinking…"):
            answer = st.session_state.pipeline(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)