import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

def _build_chroma_from_pdfs(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        docs = loader.load()
        all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(all_docs)

    # Using llama3.3 as the embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return Chroma.from_documents(splits, embeddings)

def pdf_folder_indexer():
    st.header("PDF Folder Indexer")
    pdf_folder = os.path.dirname(__file__)
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        st.warning("No PDF files found in the folder.")
        return
    st.write(f"Found {len(pdf_files)} PDF(s): {', '.join(pdf_files)}")
    with st.spinner("Indexing PDFs..."):
        chroma_index = _build_chroma_from_pdfs(pdf_folder)
        st.session_state.chroma_index = chroma_index
        st.success("✅ ChromaDB index ready. You can ask questions below.")

def chatbot():
    st.info("Use the Q&A box below. Answers come only from your documents.")
