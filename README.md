# RAG Eval System

A multi-query, hybrid-retrieval RAG system built to practice the part most RAG projects skip: **evaluation**.

Corpus: personal essays/notes (Obsidian vault). Retrieval: BM25 + local semantic embeddings, fused. Generation: Gemini. The point of this repo is the eval harness, not the pipeline.

## Why evals

Anyone can wire up a RAG pipeline. Knowing whether it's actually retrieving the right context and generating grounded answers — and being able to prove it with numbers — is the actual skill. This repo evaluates both stages:

- **Retrieval quality:** Precision@k, Recall@k, MRR against a hand-labeled query → relevant-chunk set
- **Generation quality:** RAGAS — faithfulness, answer relevancy, context precision, context recall

## Stack

- **Orchestration:** LangChain
- **Vector store:** Chroma (local)
- **Embeddings:** local (sentence-transformers)
- **Sparse retrieval:** BM25
- **LLM:** Gemini (multi-query generation + answer synthesis)
- **Eval:** RAGAS + custom retrieval metrics

See [architecture.md](architecture.md) for pipeline details and design decisions.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env    # add your GEMINI_API_KEY
```

## Usage

```bash
python ingest.py        # chunk notes + build Chroma index + BM25 index
python evaluate.py      # run eval set, report retrieval + RAGAS metrics
```

## Status

Work in progress. See [architecture.md](architecture.md) for open questions and current pipeline stage.
