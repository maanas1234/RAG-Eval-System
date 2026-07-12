# RAG Eval System

A hybrid-retrieval RAG system built to practice the part most RAG projects skip: **evaluation**.

Corpus: personal essays/notes (Obsidian vault). Retrieval: BM25 + local semantic embeddings, fused. Generation: gpt-oss-20b via OpenRouter. The point of this repo is the eval harness, not the pipeline.

## Results

Eval set: 33 hand-labeled queries over a 69-note personal essay corpus (14 single-note, 5 multi-note, 2 unanswerable "trap" queries). k=5 unless noted, gpt-oss-20b as both generator and RAGAS judge.

**Multi-query ablation:**

| Config | Precision@5 | Recall@5 | MRR | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|---|
| Hybrid + multi-query | 0.182 | 0.803 | 0.598 | 0.705 | 0.719 | 0.662 | 0.833 |
| Hybrid only | 0.182 | 0.818 | 0.610 | 0.788 | 0.727 | 0.699 | 0.789 |

Dropping multi-query retrieval matched or improved every retrieval and faithfulness metric while cutting LLM calls per query roughly in half — see [architecture.md](architecture.md#design-decisions).

**Retrieval-method ablation** (BM25 vs. semantic vs. hybrid, k=5):

| Retrieval | Precision@5 | Recall@5 | MRR | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.127 | 0.561 | 0.515 | invalid¹ | — | — | — |
| **Semantic only** | **0.194** | **0.848** | **0.703** | 0.667² | 0.731 | **0.752** | 0.818 |
| Hybrid | 0.182 | 0.818 | 0.610 | **0.788** | 0.727 | 0.699 | 0.789 |

¹ RAGAS scores for this run are corrupted — hit Groq's 200k-tokens/day free-tier cap mid-eval, most judge calls failed.
² A few judge calls hit the max_tokens cap and were dropped; treat this single number as noisier than the rest.

**Semantic-only beats hybrid on every retrieval metric.** This corpus is personal reflective writing, not keyword-dense reference text — a query like "when is lying okay?" rarely shares exact vocabulary with the note that answers it, so BM25's lexical matching mostly misses while semantic embeddings still catch the meaning. The 50/50 BM25+semantic blend is diluting a strong semantic retriever with a weak BM25 one. Hybrid still wins on faithfulness, though that gap needs a clean rerun to confirm given the dropped samples above.

k=3/5/8 sweep and full run history (every config change, timestamped) are in [results.json](results.json).

## Why evals

Anyone can wire up a RAG pipeline. Knowing whether it's actually retrieving the right context and generating grounded answers — and being able to prove it with numbers — is the actual skill. This repo evaluates both stages:

- **Retrieval quality:** Precision@k, Recall@k, MRR against a hand-labeled query → relevant-chunk set
- **Generation quality:** RAGAS — faithfulness, answer relevancy, context precision, context recall

## Stack

- **Orchestration:** LangChain
- **Vector store:** Chroma (local)
- **Embeddings:** local (sentence-transformers)
- **Sparse retrieval:** BM25
- **LLM:** gpt-oss-20b via OpenRouter (answer synthesis + RAGAS judge model)
- **Eval:** RAGAS + custom retrieval metrics

See [architecture.md](architecture.md) for pipeline details and design decisions.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env    # add your OPENROUTER_API_KEY
```

## Usage

```bash
python ingest.py        # chunk notes + build Chroma index + BM25 index
python evaluate.py      # run eval set, report retrieval + RAGAS metrics
```

## Status

Work in progress. See [architecture.md](architecture.md) for open questions and current pipeline stage.
