import streamlit as st
from dashboard import pdf_folder_indexer, chatbot
from llm_handler import create_llm, create_qa_chain

def main():
    # Page configuration
    st.set_page_config(page_title="Doc Q&A", page_icon="📄", layout="wide")
    st.title("📄 Document Question & Answer App")

    # Sidebar for PDF upload and reset
    with st.sidebar:
        st.header("📂 Upload PDFs")
        pdf_folder_indexer()

        if st.button("🔄 Reset Session"):
            # Clear session state keys
            for key in ["chroma_index", "qa_chain", "chat_history"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.info("Session reset. Please re-upload PDFs to start.")

    # Initialize QA chain if PDF index exists and QA chain not yet created
    if "chroma_index" in st.session_state and "qa_chain" not in st.session_state:
        try:
            llm = create_llm()
            st.session_state.qa_chain = create_qa_chain(llm, st.session_state.chroma_index)
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {e}")

    # Display chatbot interface if ready
    if "qa_chain" in st.session_state:
        chatbot()
    else:
        st.info("Please upload and index PDFs to start asking questions.")


if __name__ == "__main__":
    main()
