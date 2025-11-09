# config.py
import os

# Root folder where your documents live (txt, md, pdf, etc.)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ChromaDB
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Foundation LLM:
# LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-alpha"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Device setup for HF pipeline: "cuda"/"mps" if you have GPU, else "cpu"
# HF_DEVICE = "cuda"  
HF_DEVICE = "mps"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Number of documents to retrieve for each query
TOP_K = 4
