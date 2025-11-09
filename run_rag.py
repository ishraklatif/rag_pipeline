import os
from rag_pipeline.llm import get_llm
from ingest import load_raw_documents, chunk_documents
from embeddings_store import build_or_load_vectorstore
from rag_pipeline import build_rag_chain, ask_rag


def main():
    print("\nWelcome to My RAG CLI\n")

    # ------------------------------------------------------------------ #
    # Choose data source
    # ------------------------------------------------------------------ #
    print("\nChoose your data source:")
    print("  [1] Web (https://ishraklatif.github.io)")
    print("  [2] Local documents (/data folder)")
    print("  [3] Both sources")
    choice = input("Enter 1, 2, or 3: ").strip()

    include_web = choice in ("1", "3")
    include_local = choice in ("2", "3")

    print("\n📡 Loading and chunking data...")
    docs = load_raw_documents(include_web=include_web, include_local=include_local)
    chunks = chunk_documents(docs)

    # ------------------------------------------------------------------ #
    # Vectorstore + retriever
    # ------------------------------------------------------------------ #
    print("Loading or building Chroma vectorstore...")
    vectorstore = build_or_load_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ------------------------------------------------------------------ #
    # Load LLM
    # ------------------------------------------------------------------ #
    print("⚡ Loading LLM...")
    llm = get_llm()

    # ------------------------------------------------------------------ #
    # Build RAG chain
    # ------------------------------------------------------------------ #
    print("Building RAG chain...")
    rag_chain = build_rag_chain(llm, retriever)

    # ------------------------------------------------------------------ #
    # Start interactive session
    # ------------------------------------------------------------------ #
    ask_rag(rag_chain)


if __name__ == "__main__":
    main()
