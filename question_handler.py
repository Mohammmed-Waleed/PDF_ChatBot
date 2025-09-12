import streamlit as st
from langchain.schema import Document

def display_response(response):
    """Display the LLM response and its sources."""
    if response and "result" in response:
        st.markdown(f"**Answer:** {response['result']}")
        if response.get("source_documents"):
            with st.expander("Sources"):
                for doc in response["source_documents"]:
                    if isinstance(doc, Document):
                        st.write(doc.page_content[:300] + "...")
    else:
        st.warning("No answer returned from QA chain.")

def handle_question_flow(qa_chain):
    """Handle user question -> retrieve docs -> generate answer -> update chat history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask a question about your documents:")

    if question and qa_chain:
        with st.spinner("Searching your documents..."):
            docs = qa_chain.retriever.get_relevant_documents(question)

            if not docs:
                st.warning("❌ Not found in your documents.")
                st.session_state.chat_history.append(
                    {"question": question, "answer": "Not found"}
                )
                return st.session_state.chat_history

            response = qa_chain.invoke({"query": question})
            answer = response.get("result", "No answer returned")

            st.session_state.chat_history.append(
                {"question": question, "answer": answer}
            )

            display_response(response)

    return st.session_state.chat_history
