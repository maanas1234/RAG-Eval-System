# RAG Eval System

A hybrid-retrieval RAG system built to practice the part most RAG projects skip: **evaluation**.

Corpus: personal essays/notes (Obsidian vault). Retrieval: BM25 + local semantic embeddings, fused. Generation: gpt-oss-20b via Groq. The point of this repo is the eval harness, not the pipeline.

## Results

Eval set: 33 hand-labeled queries over a 69-note personal essay corpus (14 single-note, 5 multi-note, 2 unanswerable "trap" queries). k=5, gpt-oss-20b as both generator and RAGAS judge.

| Config | Precision@5 | Recall@5 | MRR | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|---|
| Hybrid + multi-query | 0.182 | 0.803 | 0.598 | 0.705 | 0.719 | 0.662 | 0.833 |
| **Hybrid only (current)** | 0.182 | **0.818** | **0.610** | **0.788** | 0.727 | **0.699** | 0.789 |

Dropping multi-query retrieval matched or improved every retrieval and faithfulness metric while cutting LLM calls per query roughly in half (no query-rewriting step) — see [architecture.md](architecture.md#design-decisions) for why.

Full run history (every config change, timestamped) is in [results.json](results.json).

## Why evals

Anyone can wire up a RAG pipeline. Knowing whether it's actually retrieving the right context and generating grounded answers — and being able to prove it with numbers — is the actual skill. This repo evaluates both stages:

- **Retrieval quality:** Precision@k, Recall@k, MRR against a hand-labeled query → relevant-chunk set
- **Generation quality:** RAGAS — faithfulness, answer relevancy, context precision, context recall

## Stack

- **Orchestration:** LangChain
- **Vector store:** Chroma (local)
- **Embeddings:** local (sentence-transformers)
- **Sparse retrieval:** BM25
- **LLM:** gpt-oss-20b via Groq (answer synthesis + RAGAS judge model)
- **Eval:** RAGAS + custom retrieval metrics

See [architecture.md](architecture.md) for pipeline details and design decisions.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env    # add your GROQ_API_KEY
```

## Usage

```bash
python ingest.py        # chunk notes + build Chroma index + BM25 index
python evaluate.py      # run eval set, report retrieval + RAGAS metrics
```

## Status

Work in progress. See [architecture.md](architecture.md) for open questions and current pipeline stage.
