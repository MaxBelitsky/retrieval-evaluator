# Architecture

A guide for contributors. Covers the two main pipelines and the extension points for adding new chunk sources, retrieval backends, and metrics.

---

## Evaluation Pipeline

**Entry points:** `retrieval_evaluator/__init__.py:evaluate()` (Python API) and `retrieval-evaluator eval` (CLI).

**Flow:**

```
retrieval_evaluator/__init__.py:evaluate()
  └── evaluation/embedders_evaluation.py:evaluate_embedders()
        └── evaluate_embedder_on_collection()  (one call per embedder × collection × alpha)
              ├── dataset/beir.py:load_beir_dataset()           → corpus_df, queries_df, qrels
              ├── evaluation/chunk_merging.py:merge_chunks()    → (optional) merge micro-chunks
              ├── database/vector_index.py:get_vector_index()   → VectorIndex context manager
              │     ├── VectorIndex.batch_insert()              → index corpus vectors
              │     └── VectorIndex.query()                     → ranked doc IDs per query
              ├── evaluation/metrics.py:compute_ranking_metrics() → R@K, Recall@K, MRR@K
              └── write artifacts to run_details/<run_id>/
```

**Key files:**

| File | Responsibility |
|---|---|
| `retrieval_evaluator/evaluation/embedders_evaluation.py` | Orchestration: loops over embedders, collections, alpha values |
| `retrieval_evaluator/evaluation/embedder.py` | Wraps HuggingFace, OpenAI, and Cohere models; handles caching, batch sizing |
| `retrieval_evaluator/evaluation/metrics.py` | Computes R@K, Recall@K, R_all@K, MRR@K from ranked results |
| `retrieval_evaluator/evaluation/chunk_merging.py` | Merges micro-chunks to macro-chunks for long-context models |
| `retrieval_evaluator/database/vector_index.py` | `VectorIndex` ABC + `get_vector_index()` factory |
| `retrieval_evaluator/database/numpy_index.py` | `NumpyVectorIndex`: exact in-memory cosine retrieval, the default backend |
| `retrieval_evaluator/database/weaviate_db.py` | `WeaviateVectorIndex`: temporary embedded Weaviate, supports hybrid search |
| `retrieval_evaluator/dataset/beir.py` | Load/save BEIR format datasets |

---

## Dataset Generation Pipeline

**Entry points:** `retrieval_evaluator/__init__.py:build_dataset()` (Python API) and `retrieval-evaluator build-dataset` (CLI).

**Flow:**

```
retrieval_evaluator/__init__.py:build_dataset()
  └── dataset/dataset_builder.py:build_eval_sets()
        ├── dataset/chunk_sources.py:create_chunk_source()  → ChunkSource
        │     └── ChunkSource.load_chunks()                 → raw chunk dicts
        ├── dataset/document_selection.py                   → K-means diversity sampling
        ├── dataset/query_generation.py                     → LLM synthetic query per chunk
        └── dataset/beir.py                                 → write corpus.jsonl, queries.jsonl, qrels.tsv, meta.json
```

**Key files:**

| File | Responsibility |
|---|---|
| `retrieval_evaluator/dataset/dataset_builder.py` | Orchestrates the full generation pipeline |
| `retrieval_evaluator/dataset/chunk_sources.py` | `ChunkSource` protocol + built-in sources (Weaviate, Qdrant, JSONL, callable) |
| `retrieval_evaluator/dataset/document_selection.py` | K-means diversity-based document selection |
| `retrieval_evaluator/dataset/query_generation.py` | LLM query generation with retry/backoff |
| `retrieval_evaluator/dataset/beir.py` | BEIR format I/O |

---

## Extension Points

### Adding a New Chunk Source

A chunk source loads raw records from a data store. Add one when you want to pull chunks from a new system (e.g. Postgres, Pinecone).

1. Implement the `ChunkSource` protocol in `retrieval_evaluator/dataset/chunk_sources.py`:

```python
class MyChunkSource:
    def load_chunks(self, collection_name: str, *, force_reload: bool = False) -> list[dict]:
        # return a list of dicts; each dict must contain the fields named in field_mapping
        ...
```

2. Add a builder function in the same file:

```python
def _create_my_source(dataset_cfg: dict, source_cfg: dict) -> ChunkSource:
    my_cfg = source_cfg.get("my_source", {})
    return MyChunkSource(url=my_cfg.get("url", "http://localhost:5432"))
```

3. Register it in `_SOURCE_BUILDERS`:

```python
_SOURCE_BUILDERS = {
    "weaviate": _create_weaviate_source,
    "qdrant": _create_qdrant_source,
    "jsonl": _create_jsonl_source,
    "callable": _create_callable_source,
    "my_source": _create_my_source,   # add this line
}
```

4. Add corresponding config keys to `config/retrieval-evaluator.example.yaml` under `retrieval_evaluation.dataset.source`.

5. Add tests in `tests/dataset/` covering `load_chunks()` for your source.

### Adding a New Retrieval Backend

A retrieval backend is a vector index used during evaluation. Add one when you want to evaluate using a different vector store (e.g. Qdrant, Pinecone) as the retrieval engine.

1. Implement the `VectorIndex` ABC in a new file under `retrieval_evaluator/database/`:

```python
from retrieval_evaluator.database.vector_index import VectorIndex

class MyVectorIndex(VectorIndex):
    def __enter__(self) -> "MyVectorIndex":
        # set up the index
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # tear down / clean up
        pass

    def batch_insert(self, ids: list[str], vectors, texts: list[str]) -> None:
        # insert candidates; vectors is None for an alpha=0 lexical-only run
        ...

    def query(self, vector, k: int, alpha: float = 1.0, query: str | None = None) -> list[str]:
        # return up to k doc IDs ranked by relevance; vector is None for alpha=0
        # alpha=1.0 → pure semantic, alpha=0.0 → pure BM25, in-between → hybrid
        # if alpha != 1.0 and you don't support hybrid, raise ValueError
        ...

    def supports_hybrid(self) -> bool:
        return True  # or False if you only support semantic search
```

2. Register it in `retrieval_evaluator/database/vector_index.py`:

```python
def get_vector_index(backend: str = "numpy") -> VectorIndex:
    if backend == "numpy":
        from retrieval_evaluator.database.numpy_index import NumpyVectorIndex
        return NumpyVectorIndex()
    if backend == "weaviate":
        from retrieval_evaluator.database.weaviate_db import WeaviateVectorIndex
        return WeaviateVectorIndex()
    if backend == "my_backend":
        from retrieval_evaluator.database.my_index import MyVectorIndex
        return MyVectorIndex()
    raise ValueError(f"Unknown index backend: {backend!r}. Choose 'numpy', 'weaviate', or 'my_backend'.")
```

3. Wire in via config: set `settings.index_backend: my_backend` in `retrieval-evaluator.yaml`.

4. Add tests in `tests/database/` covering `batch_insert()`, `query()`, and the context manager lifecycle.

### Adding New Metrics

Metrics are computed in `retrieval_evaluator/evaluation/metrics.py:compute_ranking_metrics()`.

The function signature:

```python
def compute_ranking_metrics(
    rankings: dict[str, list[str]],   # {query_id: [ranked_doc_id, ...]}
    qrels: dict[str, dict[str, int]], # {query_id: {doc_id: relevance_score}}
    macro2micro: dict[str, list[str]],# macro-chunk → micro-chunk mapping (may be empty)
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    ...
```

New keys are written to the results CSV automatically. Update `retrieval_evaluator/dashboard/metrics.py` as well if the metric should appear in dashboard selectors and charts.

---

## Config Loading

Config is loaded lazily via `retrieval_evaluator/config.py:get_config()`. It searches for a YAML file at:
1. `config/retrieval-evaluator.yaml` (relative to the current working directory)
2. `~/.config/retrieval-evaluator/retrieval-evaluator.yaml`
3. `/etc/retrieval-evaluator/retrieval-evaluator.yaml`

CLI `--config` flag and `RETRIEVAL_EVALUATOR_CONFIG` env var override the search paths entirely. `get_config()` merges user config with built-in defaults and is safe to call multiple times — it caches the result.

Relative paths are resolved from the current working directory, including when the config is elsewhere. Parent directories are not searched. Use `get_config(reload=True)` if changing configuration during a Python session. Evaluation resolves its bundled example fallback separately from configuration, so dataset generation retains the configured output directory.
