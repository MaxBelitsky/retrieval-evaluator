"""
05_programmatic_api.py — Use the Python API in a script, notebook, or CI pipeline.

Shows how to evaluate multiple embedders, compare results, and save a dashboard run.
No config file or CLI required.

    pip install "retrieval-evaluator[hf]"
    uv run python examples/05_programmatic_api.py

Install retrieval-evaluator[dashboard] to inspect the saved run using the printed command.
"""
import shlex
from pathlib import Path

from retrieval_evaluator import evaluate

DATASET_DIR = Path("eval_data")
OUTPUT_DIR = Path("run_details/api-example")
if not DATASET_DIR.exists():
    # pip-installed users: use bundled data
    import retrieval_evaluator

    DATASET_DIR = Path(retrieval_evaluator.__file__).parent / "data"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate(
        dataset_dir=DATASET_DIR,
        collections=["EXAMPLE"],
        embedder_configs={
            "minilm": {
                "provider": "hf",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "max_tokens": 256,
            },
            # Uncomment to compare with a second model:
            # "bge-small": {
            #     "provider": "hf",
            #     "model": "BAAI/bge-small-en-v1.5",
            #     "max_tokens": 512,
            # },
        },
        k_values=[1, 5, 10],
        artifacts_dir=OUTPUT_DIR,
    )
    results_path = OUTPUT_DIR / "results.csv"
    results.to_csv(results_path, index=False)

    print("All results:")
    columns = ["embedder", "collection", "R@1", "R@5", "R@10", "MRR@10"]
    print(results[columns].to_string(index=False))

    print("\nBest R@5 per embedder:")
    best = results.groupby("embedder")["R@5"].max().reset_index()
    print(best.to_string(index=False))
    print("\nInspect this run:")
    print(shlex.join([
        "retrieval-evaluator", "dashboard",
        "--results-file", str(results_path),
        "--artifacts-dir", str(OUTPUT_DIR),
        "--dataset-dir", str(DATASET_DIR),
    ]))


if __name__ == "__main__":
    main()
