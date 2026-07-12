"""Hybrid (BM25 + semantic) retriever."""

import os
import pickle

from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from ingest import CHROMA_DIR, DOCS_PATH, EMBEDDING_MODEL

load_dotenv(override=True)

GROQ_MODEL = "openai/gpt-oss-20b"


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=GROQ_MODEL,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        max_tokens=1024,
    )


def build_bm25_retriever(k: int = 5) -> BM25Retriever:
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


def build_semantic_retriever(k: int = 5):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return chroma.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(k: int = 5) -> EnsembleRetriever:
    return EnsembleRetriever(
        retrievers=[build_bm25_retriever(k), build_semantic_retriever(k)], weights=[0.5, 0.5]
    )


def retrieve(retriever, query: str, k: int = 5) -> list[Document]:
    """Run the retriever and truncate the fused result back to the top-k notes.

    EnsembleRetriever fuses BM25 + semantic results but doesn't cap the total
    (two k=5 sub-retrievers can return up to 10 after fusion), so this slices
    back to k. Also dedupes by source path as a safety net.
    """
    seen = set()
    unique_docs = []
    for doc in retriever.invoke(query):
        source = doc.metadata["source"]
        if source not in seen:
            seen.add(source)
            unique_docs.append(doc)
    return unique_docs[:k]


if __name__ == "__main__":
    retriever = build_hybrid_retriever()
    for doc in retrieve(retriever, "What do I think about deep work?"):
        print(doc.metadata["source"])
