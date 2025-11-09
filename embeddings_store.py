import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 

from config import EMBEDDING_MODEL_NAME, CHROMA_DIR



def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings encoder using bge-base-en-v1.5.
    This runs locally (no API calls) and normalizes embeddings for cosine similarity.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )


def build_or_load_vectorstore(docs: List[Document]) -> Chroma:
    """
    Builds the Chroma vectorstore if it doesn't exist, otherwise loads it.
    - Uses bge-base-en-v1.5 for embeddings.
    - Persists embeddings locally in 'chroma_db/' folder.
    """
    embeddings = get_embedding_model()

    # Load vector store if created before
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        print("Loading existing Chroma vectorstore...")
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        print("Building new Chroma vectorstore...")
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        vectordb.persist()
        print("Vectorstore built and saved!")

    return vectordb
