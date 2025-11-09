RAG – Lightweight Retrieval-Augmented Generation Pipeline
This is a modular, local-first RAG pipeline built with LangChain 1.x, Hugging Face models, and ChromaDB.
It retrieves knowledge from local documents or web pages, embeds them with bge-base-en-v1.5,
and generates context-aware answers using a foundation model such as Phi-3-mini-4k-instruct.
🚀 Features
🔍 Dual ingestion: web (via URL) or local text files.
🧩 Embeddings powered by BAAI/bge-base-en-v1.5.
🧠 Generation with microsoft/Phi-3-mini-4k-instruct (MPS-compatible on Mac).
💾 Persistent vector storage using ChromaDB.
⚙️ Fully local — no external APIs required.
🧱 Simple CLI interface to query your documents.
🧩 Project Structure
rag_pipeline/
│
├── config.py              # All model and path settings
├── ingest.py              # Loads & chunks local/web documents
├── embeddings_store.py    # Builds or loads Chroma vector store
├── llm_zephyr.py          # Loads foundation LLM (reads from config)
├── rag_pipeline.py        # LCEL-based RAG chain
├── run_rag.py             # CLI entry point
└── .gitignore
⚙️ Setup
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
▶️ Run the Pipeline
python run_rag.py
Then choose your data source:
[1] 🌐 Web (https://ishraklatif.github.io)
[2] 📁 Local documents (/data folder)
[3] 🔀 Both sources
Once the RAG chain builds, you can interact via CLI:
🧠 Question: Who is Ishrak?
🤖 Answer: [Model response...]
🧠 Configuration
Edit config.py to switch models or adjust parameters:
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"   # Foundation model
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"        # Encoder
HF_DEVICE = "mps"                                     # "mps" for Mac, "cuda" or "cpu" otherwise
📦 Vectorstore & Data
Documents are stored locally in /data
Embeddings persist in /chroma_db
Both are ignored via .gitignore
🧾 License
MIT — free to use, modify, and extend.