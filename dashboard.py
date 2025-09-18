import streamlit as st
from llm_handler import build_chroma_from_uploaded_pdfs

def pdf_folder_indexer():
    st.subheader("Step 1: Upload PDF Files to Index")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Indexing PDFs..."):
            chroma_index = build_chroma_from_uploaded_pdfs(uploaded_files)
            st.session_state.chroma_index = chroma_index
            st.success("✅ ChromaDB index is ready. You can now ask questions.")
    else:
        st.info("Please upload PDF files to proceed.")


def chatbot():
    st.subheader("Step 2: Ask Questions")

    if "qa_chain" not in st.session_state:
        st.warning("Please index PDFs first using the upload section above.")
        return

    qa_chain = st.session_state.qa_chain

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Form ensures input is cleared on submit
    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Enter your question about your PDFs:")
        submitted = st.form_submit_button("Ask")

        if submitted and question:
            with st.spinner("Generating answer..."):
                result = qa_chain({"query": question})
                answer = result.get("result", "No answer found.")
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for chat in reversed(st.session_state.chat_history[-5:]):  # Show last 5 messages
            with st.expander(f"Q: {chat['question'][:50]}..."):
                st.write("**Question:**", chat["question"])
                st.write("**Answer:**", chat["answer"])
