import streamlit as st
import os
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA

def create_llm():
    return OllamaLLM(model="llama3.2")

def create_qa_chain(llm, chroma_index):
    retriever = chroma_index.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

# Initialize LLM
model = create_llm()

# List all PDF files in the current folder
pdf_folder = os.path.dirname(__file__)
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

st.title("PDF ChatBot with RAG")

if pdf_files:
    selected_pdf = st.selectbox("Select a PDF file to chat with:", pdf_files)
    if selected_pdf:
        pdf_path = os.path.join(pdf_folder, selected_pdf)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

    # Updated embedding model
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_db = Chroma.from_documents(splits, embeddings)

        qa_chain = create_qa_chain(model, vector_db)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.text_input("Ask a question about your PDF:")
        if question:
            result = qa_chain({"query": question})
            st.write("**Answer:**", result["result"])
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["result"]
            })

        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for chat in reversed(st.session_state.chat_history[-3:]):
                with st.expander(f"Q: {chat['question'][:50]}..."):
                    st.write("**Question:**", chat['question'])
                    st.write("**Answer:**", chat['answer'])
else:
    st.info("No PDF files found in the folder. Please add PDFs to get started.")
