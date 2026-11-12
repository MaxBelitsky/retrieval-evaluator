"""
06_ci_integration.py — Fail CI when retrieval metrics fall below thresholds.

Required environment variables:
    EVAL_DATASET_DIR   BEIR dataset root (containing collection directories)
    EVAL_MODEL         embedding model identifier

Optional environment variables:
    EVAL_COLLECTIONS   comma-separated collections (default: all directories)
    EVAL_PROVIDER      hf, openai, or cohere (default: hf)
    EVAL_INDEX_BACKEND numpy or weaviate (default: numpy)
    THRESHOLD_R5       minimum mean R@5 (default: 0.50)
    THRESHOLD_R10      minimum mean R@10 (default: 0.60)

Exit code 0 means every threshold passed. Configuration/evaluation errors and
missing or non-finite metrics fail closed with a non-zero exit code.

Install the extra for EVAL_PROVIDER, for example:
    pip install "retrieval-evaluator[hf]"

If EVAL_INDEX_BACKEND=weaviate, install the weaviate extra as well.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from retrieval_evaluator import evaluate


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def metric_violations(results, thresholds: dict[str, float]) -> list[str]:
    """Return one explanation for each threshold that does not verifiably pass."""
    violations: list[str] = []
    for metric, threshold in thresholds.items():
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            violations.append(f"  {metric}: threshold must be finite and between 0 and 1")
            continue
        if metric not in results.columns:
            violations.append(f"  {metric}: metric is missing (threshold {threshold:.3f})")
            continue
        try:
            values = [float(value) for value in results[metric]]
        except (TypeError, ValueError):
            values = []
        if not values or any(not math.isfinite(value) for value in values):
            violations.append(f"  {metric}: metric is not finite (threshold {threshold:.3f})")
            continue
        actual = sum(values) / len(values)
        if actual < threshold:
            violations.append(f"  {metric}: {actual:.3f} < {threshold:.3f} (threshold)")
        else:
            print(f"PASS  {metric}: {actual:.3f} >= {threshold:.3f}")
    return violations


def main() -> int:
    try:
        dataset_dir = Path(_required_env("EVAL_DATASET_DIR")).expanduser()
        model = _required_env("EVAL_MODEL")
        if not dataset_dir.is_dir():
            raise ValueError(f"EVAL_DATASET_DIR is not a directory: {dataset_dir}")

        collections_value = os.environ.get("EVAL_COLLECTIONS", "")
        collections = [item.strip() for item in collections_value.split(",") if item.strip()]
        if not collections:
            collections = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
        if not collections:
            raise ValueError(f"No collection directories found under {dataset_dir}")

        provider = os.environ.get("EVAL_PROVIDER", "hf")
        backend = os.environ.get("EVAL_INDEX_BACKEND", "numpy")
        thresholds = {
            "R@5": float(os.environ.get("THRESHOLD_R5", "0.50")),
            "R@10": float(os.environ.get("THRESHOLD_R10", "0.60")),
        }
        invalid_thresholds = [
            metric
            for metric, threshold in thresholds.items()
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
        ]
        if invalid_thresholds:
            raise ValueError(
                "Thresholds must be finite values between 0 and 1: "
                + ", ".join(invalid_thresholds)
            )
        results = evaluate(
            dataset_dir=dataset_dir,
            collections=collections,
            embedder_configs={
                "candidate": {
                    "provider": provider,
                    "model": model,
                }
            },
            index_backend=backend,
        )
        if results.empty:
            raise RuntimeError("Evaluation returned no result rows")
    except Exception as exc:
        print(f"ERROR — retrieval quality gate could not run: {exc}")
        return 2

    violations = metric_violations(results, thresholds)
    if violations:
        print("\nFAIL — thresholds not met:")
        for violation in violations:
            print(violation)
        return 1

    print("\nAll thresholds met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
