
**📄 PDF Chatbot with Streamlit & LangChain

This project is a document Q&A application where you can upload one or more PDFs and then ask natural language questions. The app indexes the content of the PDFs using embeddings, stores them in a Chroma vector database, and uses an LLM (via LangChain & Ollama) to generate answers.

🚀 Features

Upload PDFs and automatically index them.

Store and retrieve document embeddings using ChromaDB.

Ask questions in natural language and get context-aware answers.

Clean Streamlit interface for interactive Q&A.

Modular code split across 4 files for maintainability.

🛠️ How It Works
1. PDF Upload & Indexing

In dashboard.py, PDFs are uploaded through Streamlit’s UI.

Text is extracted using PyPDF2.

Text chunks are created and converted into embeddings using OllamaEmbeddings (or SentenceTransformers).

Embeddings are stored inside a Chroma vector store for retrieval.

2. LLM Setup

In llm_handler.py, the app initializes an Ollama LLM (or any model you configure).

A RetrievalQA chain is built, which connects the vector store with the LLM.

3. Chat Interface

In dashboard.py, a chatbot interface lets the user type questions.

The app retrieves the most relevant text chunks from Chroma and passes them to the LLM.

The LLM generates a final, context-aware answer displayed in the UI.

4. Session Management

main.py initializes the app and manages session state.

Ensures the QA chain is built only once per session to save resources.

Controls when the chatbot UI is shown.

📂 Project Structure
PDF-Chatbot/
│
├── main.py             # Entry point – manages Streamlit session flow
├── dashboard.py        # Streamlit UI for PDF upload & chatbot interface
├── llm_handler.py      # LLM setup, embeddings, and QA chain creation
├── utils.py            # Helper functions (e.g., PDF text extraction, text splitting)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

⚡️ Installation
# Clone the repository
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

▶️ Usage
streamlit run main.py


Upload your PDF(s).

Wait for indexing.

Ask questions in the chatbot UI.

Get AI-powered answers from your documents.

🧰 Requirements

See requirements.txt
 for all dependencies.

💡 Example Questions

“Summarize chapter 3 of the uploaded PDF.”

“What are the main contributions of this research paper?”

“Who are the authors mentioned in the document?”
