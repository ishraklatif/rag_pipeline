from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------------------------------- #
# Prompt template
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
# Combine retrieved documents into context
# --------------------------------------------------------------------------- #
def _combine_docs(docs):
    """Join retrieved documents into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)


# --------------------------------------------------------------------------- #
# Build the LCEL-style RAG chain
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
# Helper for interactive question answering
# --------------------------------------------------------------------------- #
def ask_rag(chain):
    print("\nAsk me anything (type 'exit' to quit):")
    while True:
        q = input("\nQuestion: ")
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            ans = chain.invoke(q)
            print(f"\nAnswer:\n{ans}\n")
        except Exception as e:
            print(f"Error: {e}")
