import os
from llm import get_llm
from ingest import load_raw_documents, chunk_documents
from embeddings_store import build_or_load_vectorstore
from rag_pipeline import build_rag_chain, build_comparison_chain, ask_hybrid


def main():
    print("\nWelcome to Hybrid RAG CLI\n")
    print("Modes:")
    print("  Standard RAG   — fast retrieval for factual questions")
    print("  Long Context   — full document injection for comparisons\n")

    # ------------------------------------------------------------------ #
    # Choose data source (unchanged from original)
    # ------------------------------------------------------------------ #
    print("Choose your data source:")
    print("  [1] Web (https://ishraklatif.github.io)")
    print("  [2] Local documents (/data folder)")
    print("  [3] Both sources")
    choice = input("Enter 1, 2, or 3: ").strip()

    include_web = choice in ("1", "3")
    include_local = choice in ("2", "3")

    # ------------------------------------------------------------------ #
    # Load raw documents (kept un-chunked for long-context mode)
    # ------------------------------------------------------------------ #
    print("\nLoading documents...")
    raw_docs = load_raw_documents(include_web=include_web, include_local=include_local)

    # ------------------------------------------------------------------ #
    # Chunk for RAG (standard path)
    # ------------------------------------------------------------------ #
    chunks = chunk_documents(raw_docs)

    # ------------------------------------------------------------------ #
    # Vectorstore + retriever (standard RAG path)
    # ------------------------------------------------------------------ #
    print("Loading or building Chroma vectorstore...")
    vectorstore = build_or_load_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ------------------------------------------------------------------ #
    # Load LLM (shared by both chains)
    # ------------------------------------------------------------------ #
    print("Loading LLM...")
    llm = get_llm(max_new_tokens=200)

    # ------------------------------------------------------------------ #
    # Build both chains
    # ------------------------------------------------------------------ #
    print("Building RAG chain...")
    rag_chain = build_rag_chain(llm, retriever)

    print("Building comparison chain...")
    comparison_chain = build_comparison_chain(llm)

    # ------------------------------------------------------------------ #
    # Start hybrid interactive session
    # ------------------------------------------------------------------ #
    ask_hybrid(rag_chain, comparison_chain, raw_docs)


if __name__ == "__main__":
    main()
