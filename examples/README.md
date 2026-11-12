# Examples

Runnable scripts demonstrating common `retrieval-evaluator` use cases.
Run them from the project root with `uv run python examples/<script>.py` after installing the listed optional features.

| Script | What it demonstrates | Requirements |
|--------|----------------------|--------------|
| [01_quickstart.py](01_quickstart.py) | Evaluate the bundled EXAMPLE dataset with one embedder. | `[hf]` |
| [02_eval_local_jsonl.py](02_eval_local_jsonl.py) | Build a synthetic dataset from JSONL chunks, then evaluate it. | `[dataset,openai]`, `OPENAI_API_KEY` |
| [03_custom_source.py](03_custom_source.py) | Connect any data store via the `callable` chunk source. | base install |
| [05_programmatic_api.py](05_programmatic_api.py) | Use the Python API in a script or notebook. | `[hf]` |
| [06_ci_integration.py](06_ci_integration.py) | Fail closed when retrieval metrics miss configured thresholds. | chosen provider extra; `[weaviate]` for that backend |

## Sample data

`data/sample_chunks.jsonl` — 10 chunks across 3 documents (vector databases, embedding models, RAG).
