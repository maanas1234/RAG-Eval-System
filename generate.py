"""Answer synthesis: retrieved chunks + query -> Gemini answer."""

from langchain_core.prompts import ChatPromptTemplate

from retrieve import build_llm, build_multi_query_retriever

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the context doesn't contain the answer, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def answer(question: str, retriever=None) -> dict:
    retriever = retriever or build_multi_query_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = PROMPT | build_llm()
    response = chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "answer": response.content,
        "contexts": [doc.page_content for doc in docs],
        "sources": [doc.metadata["source"] for doc in docs],
    }


if __name__ == "__main__":
    result = answer("What do I think about deep work?")
    print(result["answer"])
    print("\nSources:", result["sources"])
