"""
01_quickstart.py — Evaluate the bundled EXAMPLE dataset with a single embedder.

No config file required. Downloads the model on first run (~90 MB).

    pip install "retrieval-evaluator[hf]"
    uv run python examples/01_quickstart.py
"""
from retrieval_evaluator import evaluate


def main() -> None:
    results = evaluate(
        collections=["EXAMPLE"],
        embedder_configs={
            "all-MiniLM-L6-v2": {
                "provider": "hf",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "max_tokens": 256,
            }
        },
    )
    columns = ["embedder", "collection", "R@1", "R@5", "R@10", "MRR@10"]
    print(results[columns].to_string(index=False))


if __name__ == "__main__":
    main()
