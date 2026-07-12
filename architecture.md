# RAG Eval System — Architecture

## Goal
Hybrid-retrieval RAG system (BM25 + semantic), evaluated with retrieval metrics (Precision@k, Recall@k, MRR) and generation metrics (RAGAS).

## Corpus
Source: `Obsidian Vault` (Maanas's personal essays/notes)
Included folders: `Atomic Notes`, `Content Plans`, `Knowledge Base` (incl. subfolders), `Research Papers`, `Self`
Excluded: `.obsidian`, `Tags`, `Templates`, `Things to explore`
~69 `.md` files total.

## Stack
- **Orchestration:** LangChain
- **Vector store:** Chroma (local, persisted to `chroma_db/`)
- **Sparse retrieval:** BM25 (LangChain's BM25Retriever)
- **Dense retrieval:** local embeddings (`sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`)
- **Fusion:** EnsembleRetriever (50/50 BM25 + semantic), truncated back to top-k after fusion
- **LLM:** gpt-oss-20b via OpenRouter (`langchain-openai`, `ChatOpenAI` pointed at OpenRouter's base URL) — used for answer synthesis and as the RAGAS judge model
- **Generation eval:** RAGAS (faithfulness, answer relevancy, context precision, context recall)
- **Retrieval eval:** custom — Precision@k, Recall@k, MRR against hand-labeled query → relevant-chunk(s) mapping

## Pipeline
1. **Ingest** — load `.md` files from included folders, strip frontmatter, chunk per-note (whole note = one chunk).
2. **Index** — build dense vector index + BM25 index over chunks.
3. **Retrieve** — run BM25 + semantic retrievers, fuse (EnsembleRetriever), dedupe by source, truncate to top-k.
4. **Generate** — pass top-k chunks + original query to LLM for answer synthesis.
5. **Evaluate**
   - Retrieval: Precision@k, Recall@k, MRR — computed against a hand-labeled eval set (~20-30 queries with known relevant chunk IDs).
   - Generation: RAGAS metrics on (question, answer, contexts, ground_truth) tuples.

## Eval Set
Hand-labeled by Maanas (not synthetic) — ~20-30 queries over the essay corpus, each with:
- query text
- ground-truth relevant chunk ID(s)
- ground-truth answer (for RAGAS context_recall / answer correctness)

## Design Decisions
- **Dropped multi-query retrieval.** Originally used LangChain's MultiQueryRetriever (LLM rewrites the query into N variants, unions results). Removed because: (1) it multiplied LLM calls and latency per query for unproven benefit, and (2) its dedup logic compares full `Document` equality including an `id` field that BM25Retriever leaves `None` but Chroma sets — so the same note retrieved via different sub-retrievers across variants wasn't deduped, silently inflating retrieved-set size and producing invalid metrics (recall@k > 1.0) until caught and fixed. Current pipeline is single-query hybrid retrieval.
- **Open question: is hybrid retrieval actually earning its complexity here?** A BM25-vs-semantic-vs-hybrid ablation (see README Results) showed semantic-only beating hybrid on every retrieval metric — BM25 performs poorly on this corpus (personal reflective essays rarely share exact vocabulary with the queries that should match them), and the 50/50 blend dilutes a strong semantic retriever with a weak BM25 one. Pipeline still defaults to hybrid pending a decision on whether to simplify to semantic-only or reweight the ensemble.

## Results
Every `evaluate.py` run appends a record to `results.json` (timestamp, config, retrieval + RAGAS metrics) — see that file for full run history. Headline numbers are in the README.
