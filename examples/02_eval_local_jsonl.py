"""
02_eval_local_jsonl.py — Build and evaluate a temporary dataset from local JSONL chunks.

Expects a file at examples/data/sample_chunks.jsonl with fields:
  id, content, document_id

    pip install "retrieval-evaluator[dataset,openai]"
    export OPENAI_API_KEY="..."
    uv run python examples/02_eval_local_jsonl.py
"""
import json
import tempfile
from pathlib import Path

from retrieval_evaluator import build_dataset, evaluate

CHUNKS_FILE = Path("examples/data/sample_chunks.jsonl")


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = build_dataset(
            collection_name="sample",
            chunks=chunks,
            output_dir=tmp,
            field_mapping={"id": "id", "text": "content", "doc_id": "document_id"},
            llm_model="openai:gpt-4o",
            n_docs=3,
        )
        print(f"Temporary dataset written to {dataset_path}")

        results = evaluate(
            dataset_dir=Path(tmp),
            embedder_configs={
                "all-MiniLM-L6-v2": {
                    "provider": "hf",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "max_tokens": 256,
                }
            },
            collections=["sample"],
        )

    print(results[["embedder", "collection", "R@1", "R@5", "R@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
