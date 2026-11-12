from __future__ import annotations

import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

import retrieval_evaluator

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _example_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "embedder": "mock",
                "collection": "EXAMPLE",
                "R@1": 0.5,
                "R@5": 0.75,
                "R@10": 1.0,
                "MRR@10": 0.6,
            }
        ]
    )


def test_every_example_compiles_and_is_listed_in_readme():
    paths = sorted(EXAMPLES.glob("*.py"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")

    assert {path.name for path in paths} == {
        "01_quickstart.py",
        "02_eval_local_jsonl.py",
        "03_custom_source.py",
        "05_programmatic_api.py",
        "06_ci_integration.py",
    }
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        assert path.name in readme
        assert path.name in examples_readme


def test_quickstart_example_runs_with_public_api(monkeypatch, capsys):
    evaluate = Mock(return_value=_example_results())
    monkeypatch.setattr(retrieval_evaluator, "evaluate", evaluate)

    runpy.run_path(str(EXAMPLES / "01_quickstart.py"), run_name="__main__")

    assert evaluate.call_args.kwargs["collections"] == ["EXAMPLE"]
    assert evaluate.call_args.kwargs["embedder_configs"]["all-MiniLM-L6-v2"]["provider"] == "hf"
    assert "mock" in capsys.readouterr().out


def test_local_jsonl_example_runs_with_public_api(monkeypatch, tmp_path, capsys):
    build_dataset = Mock(return_value=tmp_path / "sample")
    evaluate = Mock(return_value=_example_results())
    monkeypatch.setattr(retrieval_evaluator, "build_dataset", build_dataset)
    monkeypatch.setattr(retrieval_evaluator, "evaluate", evaluate)
    monkeypatch.chdir(ROOT)

    runpy.run_path(str(EXAMPLES / "02_eval_local_jsonl.py"), run_name="__main__")

    assert build_dataset.call_args.kwargs["collection_name"] == "sample"
    assert len(build_dataset.call_args.kwargs["chunks"]) == 10
    assert evaluate.call_args.kwargs["collections"] == ["sample"]
    assert "Temporary dataset written" in capsys.readouterr().out


def test_custom_source_example_runs(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(ROOT)

    runpy.run_path(str(EXAMPLES / "03_custom_source.py"), run_name="__main__")

    output = capsys.readouterr().out
    assert "Loaded 10 chunks" in output
    assert "examples.03_custom_source:load_from_my_db" in output
    config_text, command_text = output.split("add to config/retrieval-evaluator.yaml:\n", 1)[1].split(
        "Validate from the repository root:\n", 1
    )
    config_path = tmp_path / "source.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    # Run the printed module command using the interpreter already installed by the test environment.
    command = shlex.split(command_text)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    validation = subprocess.run(
        [sys.executable, *command[3:], "--config", str(config_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert "Source validation passed" in validation.stdout


def test_programmatic_api_example_runs_with_public_api(monkeypatch, capsys, tmp_path):
    evaluate = Mock(return_value=_example_results())
    monkeypatch.setattr(retrieval_evaluator, "evaluate", evaluate)
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(EXAMPLES / "05_programmatic_api.py"), run_name="__main__")

    assert evaluate.call_args.kwargs["collections"] == ["EXAMPLE"]
    assert "index_backend" not in evaluate.call_args.kwargs
    assert evaluate.call_args.kwargs["k_values"] == [1, 5, 10]
    assert evaluate.call_args.kwargs["artifacts_dir"] == Path("run_details/api-example")
    saved = pd.read_csv(tmp_path / "run_details/api-example/results.csv")
    pd.testing.assert_frame_equal(saved, _example_results())
    assert "Best R@5 per embedder" in capsys.readouterr().out
