import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import EMBEDDING_MODEL_NAME, CHROMA_DIR


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )


def build_or_load_vectorstore(docs: List[Document]) -> Chroma:
    embeddings = get_embedding_model()

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
        # .persist() removed in langchain-chroma 0.1.2+ — auto-persists via persist_directory
        print("Vectorstore built and saved!")

    return vectordb