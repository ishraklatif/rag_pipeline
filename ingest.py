import os
from typing import List

from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# Add website URLs here
URLS = ["https://ishraklatif.github.io"]


def load_web_documents() -> List[Document]:
    """Fetch and parse content from the configured website URLs."""
    print("Loading website content...")
    loader = WebBaseLoader(URLS)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from web source.")
    return docs


def load_local_documents() -> List[Document]:
    """Load .txt files from your local data directory."""
    print("📁 Loading local text documents...")
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ DATA_DIR not found: {DATA_DIR}")
        return []

    loader = DirectoryLoader(
        DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} local document(s).")
    return docs


def load_raw_documents(include_web: bool = True, include_local: bool = True) -> List[Document]:
    """
    Combines both website and local file loading.
    Set flags to include/exclude sources.
    """
    docs: List[Document] = []

    if include_web:
        docs.extend(load_web_documents())

    if include_local:
        docs.extend(load_local_documents())

    if not docs:
        raise ValueError("No documents loaded from any source.")

    print(f"📚 Total combined documents: {len(docs)}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split combined docs into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks.")
    return chunks
