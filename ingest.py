import os
from typing import List

from langchain_community.document_loaders import (
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, URLS  # URLS now comes from config


def load_web_documents() -> List[Document]:
    """Fetch and parse content from the configured website URLs."""
    print("Loading website content...")
    loader = WebBaseLoader(URLS)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from web source.")
    return docs


def load_local_documents() -> List[Document]:
    """Load .txt, .md, and .pdf files from your local data directory."""
    print("Loading local documents...")
    if not os.path.exists(DATA_DIR):
        print(f"DATA_DIR not found: {DATA_DIR}")
        return []

    all_docs: List[Document] = []

    # --- plain text ---
    txt_loader = DirectoryLoader(
        DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    txt_docs = txt_loader.load()
    if txt_docs:
        print(f"  .txt  → {len(txt_docs)} document(s)")
    all_docs.extend(txt_docs)

    # --- markdown ---
    md_loader = DirectoryLoader(
        DATA_DIR, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, show_progress=True
    )
    md_docs = md_loader.load()
    if md_docs:
        print(f"  .md   → {len(md_docs)} document(s)")
    all_docs.extend(md_docs)

    # --- PDF (each file loaded individually so errors don't kill the whole run) ---
    pdf_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(DATA_DIR)
        for f in files if f.lower().endswith(".pdf")
    ]
    for pdf_path in pdf_paths:
        try:
            pdf_docs = PyPDFLoader(pdf_path).load()
            print(f"  .pdf  → {len(pdf_docs)} page(s) from {os.path.basename(pdf_path)}")
            all_docs.extend(pdf_docs)
        except Exception as e:
            print(f"  Could not load {os.path.basename(pdf_path)}: {e}")

    print(f"Loaded {len(all_docs)} local document(s) total.")
    return all_docs


def load_raw_documents(include_web: bool = True, include_local: bool = True) -> List[Document]:
    """Combines both website and local file loading."""
    docs: List[Document] = []

    if include_web:
        docs.extend(load_web_documents())

    if include_local:
        docs.extend(load_local_documents())

    if not docs:
        raise ValueError(
            "No documents loaded. Check that your /data folder exists and "
            "contains .txt, .md, or .pdf files."
        )

    print(f"Total combined documents: {len(docs)}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split combined docs into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks