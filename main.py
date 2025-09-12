import streamlit as st
from dashboard import pdf_folder_indexer, chatbot
from llm_handler import create_llm, create_qa_chain
from question_handler import handle_question_flow
from langchain_ollama import OllamaEmbeddings

def main():
    st.set_page_config(page_title="Doc Q&A", page_icon="📄", layout="wide")
    # Not needed for Ollama, but kept for UI consistency

    # Step 1: Index PDFs in folder -> builds Chroma index
    pdf_folder_indexer()

    # Step 2: If index exists, build QA chain ONCE and keep in session_state
    if "chroma_index" in st.session_state:
        if "qa_chain" not in st.session_state:
            llm = create_llm()
            st.session_state.qa_chain = create_qa_chain(llm, st.session_state.chroma_index)

    # Step 3: Use the QA chain for question flow
    qa_chain = st.session_state.get("qa_chain", None)
    handle_question_flow(qa_chain)

if __name__ == "__main__":
    main()
