# RAG Eval System — Architecture

## Goal
Multi-query RAG system with hybrid retrieval (BM25 + semantic), evaluated with retrieval metrics (Precision@k, Recall@k, MRR) and generation metrics (RAGAS).

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
- **Multi-query:** LangChain's MultiQueryRetriever (LLM generates query variants, results fused)
- **Fusion:** EnsembleRetriever (50/50 BM25 + semantic)
- **LLM:** gpt-oss-20b via OpenRouter (`langchain-openai`, `ChatOpenAI` pointed at OpenRouter's base URL) — used for multi-query generation and answer synthesis
- **Generation eval:** RAGAS (faithfulness, answer relevancy, context precision, context recall)
- **Retrieval eval:** custom — Precision@k, Recall@k, MRR against hand-labeled query → relevant-chunk(s) mapping

## Pipeline
1. **Ingest** — load `.md` files from included folders, strip frontmatter, chunk per-note (whole note = one chunk).
2. **Index** — build dense vector index + BM25 index over chunks.
3. **Retrieve** — for a query: generate N query variants (multi-query), run each through BM25 + semantic retrievers, fuse and dedupe results into top-k.
4. **Generate** — pass top-k chunks + original query to LLM for answer synthesis.
5. **Evaluate**
   - Retrieval: Precision@k, Recall@k, MRR — computed against a hand-labeled eval set (~20-30 queries with known relevant chunk IDs).
   - Generation: RAGAS metrics on (question, answer, contexts, ground_truth) tuples.

## Eval Set
Hand-labeled by Maanas (not synthetic) — ~20-30 queries over the essay corpus, each with:
- query text
- ground-truth relevant chunk ID(s)
- ground-truth answer (for RAGAS context_recall / answer correctness)

## Open Questions
- chat CLI (`chat.py`) built, eval set still needs hand-labeling.
