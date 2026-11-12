from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT_DIR / "examples" / "06_ci_integration.py"


def _example_namespace() -> dict:
    return runpy.run_path(str(EXAMPLE_PATH), run_name="ci_example")


def test_ci_gate_fails_missing_and_non_finite_metrics():
    metric_violations = _example_namespace()["metric_violations"]
    results = pd.DataFrame({"R@5": [0.8, np.nan]})

    violations = metric_violations(results, {"R@5": 0.5, "R@10": 0.6})

    assert any("R@5: metric is not finite" in item for item in violations)
    assert any("R@10: metric is missing" in item for item in violations)


def test_ci_gate_rejects_non_finite_threshold():
    metric_violations = _example_namespace()["metric_violations"]

    violations = metric_violations(pd.DataFrame({"R@5": [1.0]}), {"R@5": float("nan")})

    assert violations == ["  R@5: threshold must be finite and between 0 and 1"]


def test_ci_gate_requires_explicit_dataset_and_model():
    env = os.environ.copy()
    env.pop("EVAL_DATASET_DIR", None)
    env.pop("EVAL_MODEL", None)
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "EVAL_DATASET_DIR is required" in result.stdout
