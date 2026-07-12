"""Eval harness: retrieval metrics (Precision@k, Recall@k, MRR) + RAGAS generation metrics."""

import argparse
import json
from datetime import datetime, timezone

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from langchain_huggingface import HuggingFaceEmbeddings

from generate import answer
from ingest import EMBEDDING_MODEL
from retrieve import build_bm25_retriever, build_hybrid_retriever, build_llm, build_semantic_retriever, retrieve

EVAL_SET_PATH = "eval_set.json"
RESULTS_PATH = "results.json"

RETRIEVER_BUILDERS = {
    "hybrid": (build_hybrid_retriever, "hybrid (BM25 + semantic)"),
    "bm25": (build_bm25_retriever, "BM25 only"),
    "semantic": (build_semantic_retriever, "semantic only"),
}


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


def run_retrieval_eval(eval_set: list[dict], retriever, k: int) -> dict:
    precisions, recalls, rr = [], [], []
    for item in eval_set:
        docs = retrieve(retriever, item["query"], k)
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


def run_ragas_eval(eval_set: list[dict], retriever, k: int) -> dict:
    rows = []
    for item in eval_set:
        result = answer(item["query"], retriever=retriever, k=k)
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


def save_run(config: dict, retrieval_metrics: dict, ragas_metrics: dict) -> None:
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            runs = json.load(f)
    except FileNotFoundError:
        runs = []

    runs.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "retrieval_metrics": retrieval_metrics,
            "ragas_metrics": ragas_metrics._repr_dict,
        }
    )
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    print(f"\nSaved run to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieval", choices=RETRIEVER_BUILDERS.keys(), default="hybrid")
    args = parser.parse_args()
    k = args.k

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    build_retriever, retrieval_label = RETRIEVER_BUILDERS[args.retrieval]
    retriever = build_retriever(k)

    print("Retrieval metrics:")
    retrieval_metrics = run_retrieval_eval(eval_set, retriever, k)
    print(retrieval_metrics)

    print("\nRAGAS metrics:")
    ragas_metrics = run_ragas_eval(eval_set, retriever, k)
    print(ragas_metrics)

    config = {"retrieval": retrieval_label, "k": k, "llm": "openai/gpt-oss-20b"}
    save_run(config, retrieval_metrics, ragas_metrics)


if __name__ == "__main__":
    main()
