from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder


# --------------------------------------------------------------------------- #
# Mode constants
# --------------------------------------------------------------------------- #
MODE_AUTO = "auto"
MODE_RAG  = "rag"
MODE_LC   = "lc"

MODE_LABELS = {
    MODE_AUTO: "Auto (keyword routing)",
    MODE_RAG:  "Standard RAG (forced)",
    MODE_LC:   "Long Context (forced)",
}

COMMANDS = {
    "/rag":  MODE_RAG,
    "/lc":   MODE_LC,
    "/auto": MODE_AUTO,
}

HELP_TEXT = """
Commands you can type at any time:
  /rag   — lock into Standard RAG mode  (fast, chunk-based retrieval)
  /lc    — lock into Long Context mode  (full docs, global reasoning)
  /auto  — go back to automatic routing (default)
  /mode  — show current mode
  /help  — show this message
  exit   — quit
"""

# Character budget for long-context document injection.
# Phi-3-mini-128k: 80_000 is safe
MAX_CONTEXT_CHARS = 80_000

# Reranker: cross-encoder scores each (query, chunk) pair for true relevance.
# Chunks scoring below this threshold are dropped before sending to the LLM.
# Range is roughly -10 (irrelevant) to +10 (highly relevant).
# 0.0 is a good starting point — raise it to be stricter, lower it to be looser.
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_THRESHOLD = 0.0


# --------------------------------------------------------------------------- #
# Prompt: standard RAG
# --------------------------------------------------------------------------- #
RAG_PROMPT_TEMPLATE = """You are a helpful assistant.
Use ONLY the provided context to answer the question.
If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""


# --------------------------------------------------------------------------- #
# Prompt: long-context comparison
# --------------------------------------------------------------------------- #
COMPARISON_PROMPT_TEMPLATE = """You are a careful analytical assistant.
You are given the FULL content of two or more documents below.
Your job is to compare them thoroughly and answer the question.

Pay close attention to:
- What is present in one document but absent in another
- Contradictions or disagreements between documents
- Shared themes or conclusions

Documents:
{documents}

Question:
{question}

Provide a structured comparison in your answer:
"""


# --------------------------------------------------------------------------- #
# Reranker
# --------------------------------------------------------------------------- #
def load_reranker() -> CrossEncoder:
    """
    Loads the cross-encoder reranker model locally.
    Called once at startup and passed into build_rag_chain().
    """
    print(f"Loading reranker: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, docs: list, reranker: CrossEncoder, threshold: float = RERANKER_THRESHOLD) -> list:
    """
    Re-scores retrieved chunks against the query using the cross-encoder.

    Steps:
      1. Score each (query, chunk_text) pair — cross-encoder reads both together
      2. Attach scores to docs and sort descending
      3. Drop any chunk below the threshold (relevance filtering)
      4. Return at least 1 chunk even if all are below threshold (safety fallback)

    Args:
        query:     The user's question
        docs:      List of LangChain Document objects from the retriever
        reranker:  Loaded CrossEncoder model
        threshold: Minimum relevance score to keep a chunk

    Returns:
        Reranked (and optionally filtered) list of Documents
    """
    if not docs:
        return docs

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    filtered = [doc for score, doc in scored if score >= threshold]

    # Safety fallback: always return at least the top-1 chunk
    if not filtered:
        filtered = [scored[0][1]]

    return filtered


# --------------------------------------------------------------------------- #
# Combine docs into context string
# --------------------------------------------------------------------------- #
def _combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --------------------------------------------------------------------------- #
# Build the RAG chain (with reranker)
# --------------------------------------------------------------------------- #
def build_rag_chain(llm, retriever, reranker: CrossEncoder):
    """
    Builds the classic RAG chain with reranking:
      retriever → reranker → combine → prompt → LLM

    The reranker step is wrapped in RunnableLambda so it fits
    cleanly into the LCEL chain alongside the retriever.
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    def retrieve_and_rerank(question: str) -> str:
        docs = retriever.invoke(question)
        reranked = rerank(question, docs, reranker)
        return _combine_docs(reranked)

    rag_chain = (
        {"context": RunnableLambda(retrieve_and_rerank), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# --------------------------------------------------------------------------- #
# Build the long-context comparison chain (unchanged)
# --------------------------------------------------------------------------- #
def build_comparison_chain(llm):
    prompt = ChatPromptTemplate.from_template(COMPARISON_PROMPT_TEMPLATE)

    comparison_chain = (
        {"documents": lambda x: x["documents"], "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    return comparison_chain


# --------------------------------------------------------------------------- #
# Router: keyword-based auto-detection
# --------------------------------------------------------------------------- #
COMPARISON_KEYWORDS = [
    "compare", "comparison", "contrast", "difference", "differ",
    "missing", "absent", "omitted", "not in", "only in",
    "both", "neither", "which document", "across documents",
    "between", "versus", "vs",
]

def is_comparison_query(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in COMPARISON_KEYWORDS)


# --------------------------------------------------------------------------- #
# Helper: format + safely truncate documents for long-context prompt
# --------------------------------------------------------------------------- #
def load_full_documents(docs, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not docs:
        return ""

    per_doc_budget = max_chars // len(docs)
    sections = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"Document {i+1}")
        content = doc.page_content

        if len(content) > per_doc_budget:
            content = content[:per_doc_budget]
            truncated = True
        else:
            truncated = False

        header = f"--- Document {i+1}: {source}{' [truncated]' if truncated else ''} ---"
        sections.append(f"{header}\n{content}")

    full_text = "\n\n".join(sections)

    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n[... truncated to fit context window]"

    return full_text


# --------------------------------------------------------------------------- #
# Unified interactive loop with manual mode switching
# --------------------------------------------------------------------------- #
def ask_hybrid(rag_chain, comparison_chain, raw_docs):
    full_doc_text = load_full_documents(raw_docs)
    current_mode = MODE_AUTO

    print("\nHybrid RAG ready.")
    print(f"Current mode: {MODE_LABELS[current_mode]}")
    print("Type /help to see mode commands.\n")

    while True:
        q = input("Question: ").strip()

        if not q:
            continue

        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if q.lower() in COMMANDS:
            current_mode = COMMANDS[q.lower()]
            print(f"  Mode switched to: {MODE_LABELS[current_mode]}\n")
            continue

        if q.lower() == "/mode":
            print(f"  Current mode: {MODE_LABELS[current_mode]}\n")
            continue

        if q.lower() == "/help":
            print(HELP_TEXT)
            continue

        if current_mode == MODE_RAG:
            use_lc = False
        elif current_mode == MODE_LC:
            use_lc = True
        else:
            use_lc = is_comparison_query(q)

        try:
            if use_lc:
                print("  → [Long Context]")
                ans = comparison_chain.invoke({"documents": full_doc_text, "question": q})
            else:
                print("  → [Standard RAG + reranker]")
                ans = rag_chain.invoke(q)

            print(f"\nAnswer:\n{ans}\n")

        except Exception as e:
            print(f"Error: {e}")