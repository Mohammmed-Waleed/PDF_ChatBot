import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
import tempfile

@st.cache_resource
def create_llm():
    return OllamaLLM(model="llama3.2")

@st.cache_resource
def create_qa_chain(llm, _chroma_index):  # Note the underscore before chroma_index
    retriever = _chroma_index.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

@st.cache_resource
def build_chroma_from_uploaded_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        documents.extend(docs)
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    chroma_index = Chroma.from_documents(splits, embeddings)
    return chroma_index
