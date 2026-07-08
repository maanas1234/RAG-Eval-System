"""Hybrid (BM25 + semantic) multi-query retriever."""

import pickle

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import CHROMA_DIR, DOCS_PATH, EMBEDDING_MODEL


def build_hybrid_retriever(k: int = 5) -> EnsembleRetriever:
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    semantic = chroma.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.5, 0.5])


def build_multi_query_retriever(k: int = 5) -> MultiQueryRetriever:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return MultiQueryRetriever.from_llm(retriever=build_hybrid_retriever(k), llm=llm)


if __name__ == "__main__":
    retriever = build_multi_query_retriever()
    results = retriever.invoke("What do I think about deep work?")
    for doc in results:
        print(doc.metadata["source"])
