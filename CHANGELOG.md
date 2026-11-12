# Changelog

## 0.1.0 — Unreleased

Initial public release.

- Evaluate HuggingFace, OpenAI, and Cohere embedding models on BEIR-format private or public datasets.
- Use deterministic exact-cosine retrieval in-process, or opt into temporary embedded Weaviate for semantic, BM25, and hybrid experiments.
- Compute hit rate (`R@K`), `Recall@K`, `R_all@K`, and `MRR@K`, including correct scoring for merged macro-chunks and multiple relevance judgments.
- Generate bootstrap datasets from JSONL, Weaviate, Qdrant, custom callable sources, or imported CSV/JSONL/Hugging Face datasets.
- Inspect aggregate metrics and per-query rankings in the Streamlit dashboard.
- Reproduce logical runs with dataset fingerprints, effective configuration IDs, saved artifacts, and strict input validation.
- Use a lightweight core install with explicit extras for model providers, dataset generation, connectors, and dashboard support.
