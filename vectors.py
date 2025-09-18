import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import os

st.title("Restaurant Reviews PDF Upload and Search")

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db_location = "./chroma_langchain_db"
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    ids = []
    for i, uploaded_file in enumerate(uploaded_files):
        # Save uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF content
        loader = PyPDFLoader(uploaded_file.name)
        pdf_docs = loader.load()
        
        # Assign unique IDs and add metadata
        for j, doc in enumerate(pdf_docs):
            doc.id = f"{uploaded_file.name}_{j}"
            documents.append(doc)
            ids.append(doc.id)
        
        # Optionally remove temporary file
        os.remove(uploaded_file.name)

    # Add documents to vector store
    vector_store.add_documents(documents=documents, ids=ids)
    st.success(f"Added {len(documents)} pages from {len(uploaded_files)} PDF files to the vector store.")

# Create retriever and search interface
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

query = st.text_input("Enter your search query:")

if query:
    results = retriever.get_relevant_documents(query)
    st.write("Top Results:")
    for result in results:
        st.write(result.page_content)
        st.write("Metadata:", result.metadata)
