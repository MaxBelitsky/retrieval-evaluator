# Changelog

## 0.1.0 — 2026-11-12

Initial public release.

- Evaluate HuggingFace, OpenAI, and Cohere embedding models on BEIR-format private or public datasets.
- Use deterministic exact-cosine retrieval in-process, or opt into temporary embedded Weaviate for semantic, BM25, and hybrid experiments.
- Compute hit rate (`R@K`), `Recall@K`, `R_all@K`, and `MRR@K`, including correct scoring for merged macro-chunks and multiple relevance judgments.
- Import existing judgments from CSV, JSONL, or Hugging Face datasets, or generate synthetic queries from chunks in JSONL, Weaviate, Qdrant, or a custom Python source.
- Inspect aggregate metrics and per-query rankings in the Streamlit dashboard.
- Reproduce logical runs with dataset fingerprints, effective configuration IDs, saved artifacts, and strict input validation.
- Use a lightweight core install with explicit extras for model providers, dataset generation, connectors, and dashboard support.
