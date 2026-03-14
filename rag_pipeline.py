from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
# Phi-3-mini-4k:    keep at 1_800  (safe for 4k window)
# Phi-3-mini-128k:  raise to 80_000 (safe for 128k window)
MAX_CONTEXT_CHARS = 80_000


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
# Combine retrieved chunks into context (for standard RAG)
# --------------------------------------------------------------------------- #
def _combine_docs(docs):
    """Join retrieved document chunks into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)


# --------------------------------------------------------------------------- #
# Build the standard LCEL-style RAG chain
# --------------------------------------------------------------------------- #
def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | _combine_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# --------------------------------------------------------------------------- #
# Build the long-context comparison chain
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
    """
    Formats documents as labelled sections and truncates the total text
    to max_chars to stay within the model's context window.

    Each document gets an equal share of the budget so no single document
    crowds out the others.
    """
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

    # Final safety cut on the combined total
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n[... truncated to fit context window]"

    return full_text


# --------------------------------------------------------------------------- #
# Unified interactive loop with manual mode switching
# --------------------------------------------------------------------------- #
def ask_hybrid(rag_chain, comparison_chain, raw_docs):
    """
    Interactive loop with three modes:
      auto  — keyword router decides per question (default)
      rag   — always use Standard RAG regardless of question
      lc    — always use Long Context regardless of question

    Switch modes at any time by typing /rag, /lc, or /auto.
    """
    full_doc_text = load_full_documents(raw_docs)
    current_mode = MODE_AUTO

    print("\nHybrid RAG ready.")
    print(f"Current mode: {MODE_LABELS[current_mode]}")
    print("Type /help to see mode commands.\n")

    while True:
        q = input("Question: ").strip()

        if not q:
            continue

        # ---- exit ----
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # ---- mode switch commands ----
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

        # ---- resolve which chain to use ----
        if current_mode == MODE_RAG:
            use_lc = False
        elif current_mode == MODE_LC:
            use_lc = True
        else:  # MODE_AUTO
            use_lc = is_comparison_query(q)

        # ---- run the chain ----
        try:
            if use_lc:
                print("  → [Long Context]")
                ans = comparison_chain.invoke({"documents": full_doc_text, "question": q})
            else:
                print("  → [Standard RAG]")
                ans = rag_chain.invoke(q)

            print(f"\nAnswer:\n{ans}\n")

        except Exception as e:
            print(f"Error: {e}")