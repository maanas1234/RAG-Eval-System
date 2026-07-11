"""Interactive CLI chat over the RAG pipeline."""

from generate import answer
from retrieve import build_multi_query_retriever


def main():
    print("Building retriever...")
    retriever = build_multi_query_retriever()
    print("Ready. Type a question (or 'exit' to quit).\n")

    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = answer(question, retriever=retriever)
        print(f"\n{result['answer']}\n")
        print("Sources:", result["sources"], "\n")


if __name__ == "__main__":
    main()
