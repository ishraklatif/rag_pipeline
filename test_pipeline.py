"""
test_pipeline.py — end-to-end pipeline health check

Tests every stage in order:
  1. Config         — all required settings exist
  2. Ingest         — documents load and chunk correctly
  3. Embeddings     — vectorstore builds and retrieves
  4. Reranker       — cross-encoder loads and scores
  5. LLM            — model loads and generates text
  6. RAG chain      — full standard RAG query works
  7. Comparison     — long-context chain works
  8. Router         — keyword detection routes correctly

Run with:
  python test_pipeline.py
  python test_pipeline.py --no-llm   (skip LLM + chain tests, much faster)
"""

import sys
import time
import argparse
import traceback

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
PASS  = "  PASS"
FAIL  = "  FAIL"
SKIP  = "  SKIP"
SEP   = "-" * 55

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg):
    print(f"{PASS}  {msg}")

def fail(msg, err=None):
    print(f"{FAIL}  {msg}")
    if err:
        print(f"        {type(err).__name__}: {err}")

def skip(msg):
    print(f"{SKIP}  {msg}")

def elapsed(t):
    return f"{time.time() - t:.1f}s"


# --------------------------------------------------------------------------- #
# Test 1 — Config
# --------------------------------------------------------------------------- #
def test_config():
    section("1. Config")
    try:
        import config
        required = ["DATA_DIR", "CHROMA_DIR", "EMBEDDING_MODEL_NAME",
                    "LLM_MODEL_NAME", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "URLS"]
        missing = [k for k in required if not hasattr(config, k)]
        if missing:
            fail(f"Missing keys: {missing}")
            return False
        ok(f"LLM model:       {config.LLM_MODEL_NAME}")
        ok(f"Embedding model: {config.EMBEDDING_MODEL_NAME}")
        ok(f"Chunk size:      {config.CHUNK_SIZE} / overlap {config.CHUNK_OVERLAP}")
        ok(f"URLs:            {config.URLS}")
        ok(f"Data dir:        {config.DATA_DIR}")
        return True
    except Exception as e:
        fail("Could not import config.py", e)
        return False


# --------------------------------------------------------------------------- #
# Test 2 — Ingest
# --------------------------------------------------------------------------- #
def test_ingest():
    section("2. Ingest")
    try:
        from ingest import load_raw_documents, chunk_documents
        import config

        t = time.time()
        # Try local only — avoids network dependency in tests
        docs = load_raw_documents(include_web=False, include_local=True)
        ok(f"Loaded {len(docs)} raw document(s) from /data  ({elapsed(t)})")

        if not docs:
            fail("No documents found — add files to /data folder")
            return None, None

        # Check metadata exists
        for d in docs:
            if "source" not in d.metadata:
                fail(f"Document missing 'source' metadata: {d.metadata}")
                return None, None
        ok("All documents have 'source' metadata")

        t = time.time()
        chunks = chunk_documents(docs)
        ok(f"Chunked into {len(chunks)} chunks  ({elapsed(t)})")

        avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
        ok(f"Average chunk length: {avg_len} chars")

        if avg_len > config.CHUNK_SIZE * 1.2:
            fail(f"Chunks are too large (avg {avg_len} > limit {config.CHUNK_SIZE})")
            return None, None

        return docs, chunks

    except ValueError as e:
        fail("Ingest failed — no documents loaded", e)
        return None, None
    except Exception as e:
        fail("Ingest error", e)
        traceback.print_exc()
        return None, None


# --------------------------------------------------------------------------- #
# Test 3 — Embeddings + vectorstore
# --------------------------------------------------------------------------- #
def test_embeddings(chunks):
    section("3. Embeddings & vectorstore")
    if not chunks:
        skip("No chunks available — skipping")
        return None

    try:
        from embeddings_store import build_or_load_vectorstore

        t = time.time()
        vectorstore = build_or_load_vectorstore(chunks)
        ok(f"Vectorstore ready  ({elapsed(t)})")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        t = time.time()
        results = retriever.invoke("test query")
        ok(f"Retriever returned {len(results)} chunk(s)  ({elapsed(t)})")

        if not results:
            fail("Retriever returned 0 results — vectorstore may be empty")
            return None

        ok(f"First chunk preview: {results[0].page_content[:80].strip()}...")
        return retriever

    except Exception as e:
        fail("Embeddings error", e)
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------- #
# Test 4 — Reranker
# --------------------------------------------------------------------------- #
def test_reranker(retriever):
    section("4. Reranker")
    if not retriever:
        skip("No retriever — skipping")
        return None

    try:
        from rag_pipeline import load_reranker, rerank

        t = time.time()
        reranker = load_reranker()
        ok(f"Cross-encoder loaded  ({elapsed(t)})")

        docs = retriever.invoke("What is this document about?")
        t = time.time()
        reranked = rerank("What is this document about?", docs, reranker)
        ok(f"Reranked {len(docs)} → {len(reranked)} chunk(s)  ({elapsed(t)})")
        ok(f"Top chunk preview: {reranked[0].page_content[:80].strip()}...")
        return reranker

    except Exception as e:
        fail("Reranker error", e)
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------- #
# Test 5 — LLM
# --------------------------------------------------------------------------- #
def test_llm():
    section("5. LLM")
    try:
        from llm import get_llm

        t = time.time()
        llm = get_llm(max_new_tokens=50)  # small limit for speed
        ok(f"LLM loaded  ({elapsed(t)})")

        t = time.time()
        # Direct invocation — bypasses the full chain
        response = llm.invoke("Say hello in one sentence.")
        ok(f"LLM responded in {elapsed(t)}")
        ok(f"Response: {str(response).strip()[:120]}")

        if not response or len(str(response).strip()) < 3:
            fail("LLM returned empty or very short response")
            return None

        return llm

    except Exception as e:
        fail("LLM error", e)
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------- #
# Test 6 — RAG chain
# --------------------------------------------------------------------------- #
def test_rag_chain(llm, retriever, reranker):
    section("6. RAG chain")
    if not all([llm, retriever, reranker]):
        skip("Missing llm / retriever / reranker — skipping")
        return None

    try:
        from rag_pipeline import build_rag_chain

        rag_chain = build_rag_chain(llm, retriever, reranker)
        ok("RAG chain built")

        t = time.time()
        answer = rag_chain.invoke("What is this document about?")
        ok(f"RAG answer received in {elapsed(t)}")
        ok(f"Answer: {str(answer).strip()[:200]}")

        if not answer or len(str(answer).strip()) < 3:
            fail("RAG chain returned empty answer")
            return None

        return rag_chain

    except Exception as e:
        fail("RAG chain error", e)
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------- #
# Test 7 — Comparison (long-context) chain
# --------------------------------------------------------------------------- #
def test_comparison_chain(llm, docs):
    section("7. Long-context comparison chain")
    if not llm or not docs:
        skip("Missing llm / docs — skipping")
        return None

    try:
        from rag_pipeline import build_comparison_chain, load_full_documents

        comparison_chain = build_comparison_chain(llm)
        ok("Comparison chain built")

        full_text = load_full_documents(docs)
        ok(f"Full doc text: {len(full_text)} chars injected")

        t = time.time()
        answer = comparison_chain.invoke({
            "documents": full_text,
            "question": "Summarise the main topics in these documents."
        })
        ok(f"Comparison answer received in {elapsed(t)}")
        ok(f"Answer: {str(answer).strip()[:200]}")

        if not answer or len(str(answer).strip()) < 3:
            fail("Comparison chain returned empty answer")
            return None

        return comparison_chain

    except Exception as e:
        fail("Comparison chain error", e)
        traceback.print_exc()
        return None


# --------------------------------------------------------------------------- #
# Test 8 — Router
# --------------------------------------------------------------------------- #
def test_router():
    section("8. Query router")
    try:
        from rag_pipeline import is_comparison_query

        cases = [
            ("Who is Ishrak?",                          False),
            ("What skills does he have?",               False),
            ("Compare the CV and the web page",         True),
            ("What is missing from the CV?",            True),
            ("What is the difference between X and Y?", True),
            ("Summarise the document",                  False),
            ("versus the other source",                 True),
        ]

        all_passed = True
        for query, expected in cases:
            result = is_comparison_query(query)
            if result == expected:
                label = "LC " if expected else "RAG"
                ok(f"[{label}]  '{query}'")
            else:
                fail(f"Wrong route for: '{query}'  (got {'LC' if result else 'RAG'}, expected {'LC' if expected else 'RAG'})")
                all_passed = False

        return all_passed

    except Exception as e:
        fail("Router error", e)
        return False


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def print_summary(results):
    section("Summary")
    total  = len(results)
    passed = sum(1 for _, v in results if v)
    failed = total - passed

    for name, v in results:
        status = "PASS" if v else "FAIL/SKIP"
        print(f"  {status:<10} {name}")

    print(f"\n  {passed}/{total} stages passed")
    if failed:
        print(f"  {failed} stage(s) need attention — see details above")
    else:
        print("  Pipeline is healthy and ready to use")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM and chain tests (faster, no model download needed)")
    args = parser.parse_args()

    print("\nPipeline end-to-end test")
    print("========================")
    if args.no_llm:
        print("Mode: --no-llm (skipping LLM, RAG chain, comparison chain)")

    results = []

    # Always run
    cfg_ok             = test_config()
    docs, chunks       = test_ingest()       if cfg_ok        else (None, None)
    retriever          = test_embeddings(chunks)
    reranker           = test_reranker(retriever)
    router_ok          = test_router()

    results += [
        ("Config",    cfg_ok),
        ("Ingest",    bool(docs)),
        ("Embeddings",bool(retriever)),
        ("Reranker",  bool(reranker)),
        ("Router",    router_ok),
    ]

    # Skip if --no-llm
    if args.no_llm:
        skip("LLM load skipped (--no-llm)")
        skip("RAG chain skipped (--no-llm)")
        skip("Comparison chain skipped (--no-llm)")
        results += [
            ("LLM",             None),
            ("RAG chain",       None),
            ("Comparison chain",None),
        ]
    else:
        llm              = test_llm()
        rag_ok           = test_rag_chain(llm, retriever, reranker)
        cmp_ok           = test_comparison_chain(llm, docs)
        results += [
            ("LLM",             bool(llm)),
            ("RAG chain",       bool(rag_ok)),
            ("Comparison chain",bool(cmp_ok)),
        ]

    print_summary(results)