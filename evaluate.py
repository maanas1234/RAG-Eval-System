"""Eval harness: retrieval metrics (Precision@k, Recall@k, MRR) + RAGAS generation metrics."""

import json
from datetime import datetime, timezone

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from langchain_huggingface import HuggingFaceEmbeddings

from generate import answer
from ingest import EMBEDDING_MODEL
from retrieve import build_hybrid_retriever, build_llm, retrieve

EVAL_SET_PATH = "eval_set.json"
RESULTS_PATH = "results.json"
RUN_CONFIG = {"retrieval": "hybrid (BM25 + semantic)", "k": 5, "llm": "openai/gpt-oss-20b"}


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
        docs = retrieve(retriever, item["query"])
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
        llm=build_llm(),
        embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
    )
    return scores


def save_run(retrieval_metrics: dict, ragas_metrics: dict) -> None:
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            runs = json.load(f)
    except FileNotFoundError:
        runs = []

    runs.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": RUN_CONFIG,
            "retrieval_metrics": retrieval_metrics,
            "ragas_metrics": ragas_metrics._repr_dict,
        }
    )
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    print(f"\nSaved run to {RESULTS_PATH}")


def main():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    retriever = build_hybrid_retriever()

    print("Retrieval metrics:")
    retrieval_metrics = run_retrieval_eval(eval_set, retriever)
    print(retrieval_metrics)

    print("\nRAGAS metrics:")
    ragas_metrics = run_ragas_eval(eval_set)
    print(ragas_metrics)

    save_run(retrieval_metrics, ragas_metrics)


if __name__ == "__main__":
    main()
