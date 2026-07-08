"""Eval harness: retrieval metrics (Precision@k, Recall@k, MRR) + RAGAS generation metrics."""

import json

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from generate import answer
from retrieve import build_multi_query_retriever

EVAL_SET_PATH = "eval_set.json"


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for s in retrieved if s in relevant) / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for s in retrieved if s in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, s in enumerate(retrieved, start=1):
        if s in relevant:
            return 1 / i
    return 0.0


def run_retrieval_eval(eval_set: list[dict], retriever) -> dict:
    precisions, recalls, rr = [], [], []
    for item in eval_set:
        docs = retriever.invoke(item["query"])
        retrieved_sources = [d.metadata["source"] for d in docs]
        relevant = set(item["relevant_sources"])

        precisions.append(precision_at_k(retrieved_sources, relevant))
        recalls.append(recall_at_k(retrieved_sources, relevant))
        rr.append(reciprocal_rank(retrieved_sources, relevant))

    n = len(eval_set)
    return {
        "precision@k": sum(precisions) / n,
        "recall@k": sum(recalls) / n,
        "mrr": sum(rr) / n,
    }


def run_ragas_eval(eval_set: list[dict]) -> dict:
    rows = []
    for item in eval_set:
        result = answer(item["query"])
        rows.append(
            {
                "question": item["query"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": item["ground_truth"],
            }
        )

    dataset = Dataset.from_list(rows)
    scores = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return scores


def main():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    retriever = build_multi_query_retriever()

    print("Retrieval metrics:")
    print(run_retrieval_eval(eval_set, retriever))

    print("\nRAGAS metrics:")
    print(run_ragas_eval(eval_set))


if __name__ == "__main__":
    main()
