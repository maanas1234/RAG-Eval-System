# RAG Eval System

A hybrid-retrieval RAG system built specifically to practice the part most RAG projects skip: **rigorous evaluation**. The pipeline (ingest → retrieve → generate) is intentionally simple — the actual point of this repo is the eval harness and the decisions it drove.

Corpus: 69 of my own personal essays/journal notes (Obsidian vault). Every architectural decision below — dropping multi-query retrieval, picking k=5, questioning whether hybrid search is even worth it — was made by running an ablation and reading the numbers, not by assumption. That decision trail is the deliverable.

## TL;DR results

| Question | Answer | Evidence |
|---|---|---|
| Does multi-query retrieval help? | **No** — dropped it | [Multi-query ablation](#1-multi-query-retrieval-ablation) |
| What's the best `k`? | **5** | [k sweep](#2-k-sweep-k35-8) |
| Is hybrid (BM25+semantic) better than semantic alone? | **No, semantic alone wins** | [Retrieval-method ablation](#3-retrieval-method-ablation-bm25-vs-semantic-vs-hybrid) |

## What this evaluates

A RAG answer can be bad for two independent reasons: retrieval never found the right source material, or generation had the right material and still produced something ungrounded. This repo scores both stages **separately**, so a bad result can actually be diagnosed:

- **Retrieval quality** — Precision@k, Recall@k, MRR — computed against `eval_set.json`, 33 queries I hand-labeled myself (not LLM-generated) against my own notes, so the ground truth reflects genuine knowledge of what's actually in the corpus. 14 single-note queries, 5 multi-note queries (need two notes to answer fully), 2 "trap" queries with no answer anywhere in the vault (tests whether generation admits it doesn't know vs. hallucinating).
- **Generation quality** — RAGAS — faithfulness, answer relevancy, context precision, context recall — an LLM judge scores the generated answer against the retrieved context and a hand-written ground-truth answer.

<details>
<summary>What each metric actually measures</summary>

- **Precision@k** — of the k notes retrieved, what fraction are actually relevant. Punishes noise.
- **Recall@k** — of all truly relevant notes, what fraction got retrieved. Punishes misses.
- **MRR** — how high the *first* relevant result ranks. A retriever that finds the right note at position 1 scores better than one that buries it at position 5, even with identical precision/recall.
- **Faithfulness** — what fraction of claims in the generated answer are actually supported by the retrieved context. Low score = hallucination.
- **Answer relevancy** — does the answer actually address the question asked.
- **Context precision** — are the relevant chunks ranked above irrelevant ones in what got fed to the LLM.
- **Context recall** — does the retrieved context contain everything needed to reconstruct the ground-truth answer.

</details>

## Results

All runs below: gpt-oss-20b (via OpenRouter) as both generator and RAGAS judge, k=5 unless noted. Full timestamped run log, including raw numbers for every config tested, is in [results.json](results.json).

### 1. Multi-query retrieval ablation

| Config | Precision@5 | Recall@5 | MRR | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|---|
| Hybrid + multi-query | 0.182 | 0.803 | 0.598 | 0.705 | 0.719 | 0.662 | 0.833 |
| **Hybrid only** | 0.182 | **0.818** | **0.610** | **0.788** | 0.727 | **0.699** | 0.789 |

Multi-query retrieval (LLM rewrites the query into several phrasings, unions results) matched or lost on every metric while roughly doubling LLM calls per query. Dropped — see [architecture.md § Design Decisions](architecture.md#design-decisions) for the bug this surfaced along the way.

### 2. k sweep (k=3/5/8)

| k | Precision@k | Recall@k | MRR |
|---|---|---|---|
| 3 | **0.283** | 0.758 | 0.606 |
| **5** | 0.182 | 0.818 | 0.610 |
| 8 | 0.121 | **0.864** | **0.635** |

Precision falls and recall rises monotonically with k — mechanically guaranteed, not just an empirical trend (every note in the top-3 is also in the top-8, so recall can only go up). k=3 starves generation of context (answer relevancy drops to 0.653 vs. ~0.73 at k=5/8, not shown above — see `results.json`); k=8 barely improves recall over k=5 while nearly halving precision. **k=5 is the best tradeoff.**

### 3. Retrieval-method ablation (BM25 vs. semantic vs. hybrid)

| Retrieval | Precision@5 | Recall@5 | MRR | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.127 | 0.561 | 0.515 | invalid¹ | — | — | — |
| **Semantic only** | **0.194** | **0.848** | **0.703** | 0.667² | 0.731 | **0.752** | 0.818 |
| Hybrid | 0.182 | 0.818 | 0.610 | **0.788** | 0.727 | 0.699 | 0.789 |

¹ RAGAS scores corrupted — hit a free-tier LLM provider's daily token cap mid-run, most judge calls failed. Retrieval metrics (computed independently of the LLM) are still valid.
² A few judge calls hit a token cap and were dropped; treat this single number as noisier than the rest.

**Semantic-only beats hybrid on every retrieval metric.** My notes are personal reflective writing, not keyword-dense reference text — a query like "when is lying okay?" rarely shares exact vocabulary with the note that answers it, so BM25's lexical matching mostly misses while semantic embeddings still catch the meaning. The 50/50 BM25+semantic blend dilutes a strong semantic retriever with a weak BM25 one on *this* corpus. See [below](#what-id-do-differently-for-a-technical-corpus) for why this flips for other kinds of documents.

## What I'd do differently for a technical corpus

This corpus is personal essays — free-form, paraphrase-heavy, no exact repeated terminology. That's exactly the setting where BM25 (lexical/keyword match) loses to semantic embeddings, because BM25 only wins when a query shares literal vocabulary with its answer.

**Technical documentation, API references, codebases, or legal/compliance text are the opposite setting**, and the retrieval strategy should change accordingly:

- **BM25 gets much stronger.** Exact identifiers — function names, error codes, config keys, class names, version numbers — are common in tech docs and rare across a corpus, which is exactly what BM25's IDF weighting rewards. A query like `"TypeError: NoneType has no attribute 'get'"` or `"how does useEffect cleanup work"` benefits enormously from lexical matching that semantic embeddings can under-rank (embeddings are tuned for meaning, not for treating a rare exact token as a near-guaranteed signal).
- **Hybrid is more likely to earn its complexity there** — because technical queries split into two real populations: precise lookups ("what does flag `--no-verify` do") where BM25 wins, and conceptual questions ("how do I skip a pre-commit hook") where semantic wins. A corpus with both needs both.
- **Don't assume the split, measure it.** That's the actual generalizable lesson from this project, not "hybrid is good" or "semantic is good" — it's that the right retrieval strategy is corpus-dependent, and the way to know is to run the same kind of ablation done here: hand-label a real eval set from the target corpus, test BM25-only/semantic-only/hybrid, and let the numbers decide the default and the ensemble weighting (a fixed 50/50 split is itself an assumption worth sweeping like `k` was here).
- **Chunking strategy matters more for tech docs.** This project chunks per-note because each note is already one coherent thought. A technical doc (e.g. a long API reference page covering 20 endpoints) is not — chunking should follow structure (per-heading, per-function, keep code blocks intact) rather than per-file, or retrieval will pull in a lot of irrelevant material from the same document.
- **Consider a reranking stage.** Tech docs often have many near-duplicate-looking candidates (similar function names across modules); a cross-encoder reranker on top of the retrieved candidates tends to help precision more there than in a small personal-essay corpus.
- **Watch for versioning.** Tech docs frequently have multiple valid-looking answers from different doc versions — pure text similarity can't tell v1 syntax from v2 syntax. Metadata filtering (by version/product) alongside retrieval becomes necessary, not optional.

## Architecture

```
Obsidian vault (69 .md notes, 5 folders)
  │
  ▼
ingest.py     strip frontmatter → chunk per-note → build Chroma index + pickle docs (for BM25)
  │
  ▼
retrieve.py   BM25 and/or semantic search (configurable) → fuse (if hybrid) → dedupe → top-k
  │
  ▼
generate.py   top-k notes + question → LLM (gpt-oss-20b) → answer
  │
  ▼
evaluate.py   score retrieval (precision/recall/MRR) + generation (RAGAS) → results.json
```

`chat.py` wraps the same pipeline into an interactive CLI.

See [architecture.md](architecture.md) for full stack details and the design-decision log (including a real bug found and fixed in LangChain's MultiQueryRetriever dedup logic).

## Stack

- **Orchestration:** LangChain
- **Vector store:** Chroma (local)
- **Embeddings:** local (sentence-transformers, `all-MiniLM-L6-v2`)
- **Sparse retrieval:** BM25
- **LLM:** gpt-oss-20b via OpenRouter (answer synthesis + RAGAS judge model)
- **Eval:** RAGAS + custom retrieval metrics

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env    # add your OPENROUTER_API_KEY
```

## Usage

```bash
python ingest.py                              # build the index (run once, or after editing the corpus)
python chat.py                                # interactive chat over your notes
python evaluate.py                            # run eval set with default config (hybrid, k=5)
python evaluate.py --retrieval semantic --k 8 # run any config
```

## Repo structure

| File | Purpose |
|---|---|
| `ingest.py` | Load, chunk, and index the corpus |
| `retrieve.py` | BM25 / semantic / hybrid retrievers + LLM client |
| `generate.py` | Answer synthesis |
| `chat.py` | Interactive CLI |
| `evaluate.py` | Eval harness — retrieval metrics + RAGAS, logs every run |
| `eval_set.json` | 33 hand-labeled queries with ground-truth sources and answers |
| `results.json` | Full run history (every config tested, timestamped) |
| `architecture.md` | Stack details, pipeline, and design-decision log |

## Status / open questions

- Pipeline still defaults to hybrid retrieval despite the finding above that semantic-only wins on this corpus — not yet switched, documented as an open question in [architecture.md](architecture.md#design-decisions).
- BM25-only RAGAS run is invalid (provider rate limit) and needs a clean rerun if that comparison matters later.
