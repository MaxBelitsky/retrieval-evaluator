MINIMAL_CONFIG = """\
# retrieval-evaluator configuration
# Paths are relative to the directory where you run retrieval-evaluator.
# Full reference: https://github.com/MaxBelitsky/retrieval-evaluator/blob/main/config/retrieval-evaluator.example.yaml

settings:
  run_details_directory: run_details
  evaluation_results_file: eval_runs.csv
  evaluation_dataset_dir: eval_data
  raw_chunks_cache_dir: raw_data
  embedders_config_path: config/embedders.yml
  max_batch_size: 2048
  index_backend: numpy

llm:
  # Used only by build-dataset. Set the matching API key as an env var.
  # Examples: "openai:gpt-4o", "anthropic:claude-sonnet-4-6"
  model: "openai:gpt-4o"
  max_attempts: 5
  timeout_s: 60
  retry_base_delay_s: 1.0
  retry_max_delay_s: 30.0
  retry_jitter_s: 0.5
  retry_on_invalid_response: true

retrieval_evaluation:
  dataset:
    document_selection_embedder: sentence-transformers/all-MiniLM-L6-v2
    n_docs_per_language: 30
    query_generation_system_prompt: >
      You are given a passage of text. Your task is to write a question for the passage
      that is provided to you. The questions you generate will be used to evaluate a
      retrieval system, so make sure to generate a question that a real user might ask.
      In addition to that, return the language of the passage (include only 1 language).
      Respond in JSON format: {"question": question, "language": language_abbreviation}.
    user_prompt_template: >
      Write ONE realistic user question that can be answered completely by the passage
      below - and only by that passage:\\n\\n'{passage}'
    source:
      # Supported types: weaviate, qdrant, jsonl, callable
      type: jsonl
      jsonl:
        root_dir: raw_data/source_collections
        filename_template: "{collection}.jsonl"
    field_mapping:
      # Map your source field names to the canonical internal names.
      # Required: id, text, doc_id
      # Optional: chunk_id (ordering), language (per-language sampling)
      id: id
      text: content
      doc_id: document_id
      chunk_id: chunk_id
      language: language
    max_kmeans_iters: 100
    sleep_s: 1
    default_seed: 42
    chunk_limit: 200

  pipeline:
    micro_chunk_tokens: 512
    default_alphas:
      - 1.0
    default_k_values:
      - 1
      - 3
      - 5
      - 10
"""

STARTER_EMBEDDERS = """\
# Embedder definitions for retrieval-evaluator.
# Each top-level key is the experiment name shown in results.
# Full annotated reference: config/retrieval-evaluator.example.yaml

# HuggingFace (local) — no API key needed; install retrieval-evaluator[hf]
all-MiniLM-L6-v2:
  provider: hf
  model: sentence-transformers/all-MiniLM-L6-v2
  max_tokens: 256

# OpenAI — requires OPENAI_API_KEY and: pip install "retrieval-evaluator[openai]"
# text-embedding-3-small:
#   provider: openai
#   model: text-embedding-3-small
#   max_tokens: 8191

# Cohere — requires COHERE_API_KEY and: pip install "retrieval-evaluator[cohere]"
# embed-multilingual-v3:
#   provider: cohere
#   model: embed-multilingual-v3.0
#   max_tokens: 512
"""
