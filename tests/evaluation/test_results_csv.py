from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from retrieval_evaluator.evaluation.embedders_evaluation import evaluate_embedders

MOCK_METRICS = {"R@1": 1.0, "MRR@1": 1.0, "chunks_indexed": 1}
MOCK_EMBEDDERS = {"mock-embedder": {"provider": "hf", "model": "mock", "max_tokens": 16}}


def run_mocked_eval(dataset_dir: Path, results_file: Path, artifacts_dir: Path, force_rerun: bool = False):
    with (
        patch("retrieval_evaluator.evaluation.embedders_evaluation.Embedder") as mock_embedder,
        patch("retrieval_evaluator.evaluation.embedders_evaluation.set_device") as mock_device,
        patch(
            "retrieval_evaluator.evaluation.embedders_evaluation.evaluate_embedder_on_collection",
            return_value=dict(MOCK_METRICS),
        ),
    ):
        mock_device.return_value = MagicMock(type="cpu")
        mock_embedder.return_value.max_tokens = 16
        mock_embedder.return_value.identity_config = dict(MOCK_EMBEDDERS["mock-embedder"])
        return evaluate_embedders(
            collections=["TESTCOL"],
            embedder_configs=MOCK_EMBEDDERS,
            dataset_dir=dataset_dir,
            results_file=results_file,
            artifacts_output_dir=artifacts_dir,
            force_rerun=force_rerun,
        )


def test_run_row_records_dataset_fingerprint(tmp_path, make_beir_collection):
    collection_dir = make_beir_collection(tmp_path / "eval_data")
    results_file = tmp_path / "results.csv"

    run_mocked_eval(tmp_path / "eval_data", results_file, tmp_path / "artifacts")

    df = pd.read_csv(results_file)
    assert "dataset_fingerprint" in df.columns
    from retrieval_evaluator.dataset.beir import derive_beir_meta

    assert df.loc[0, "dataset_fingerprint"] == derive_beir_meta(collection_dir)["dataset_fingerprint"]


def test_appending_to_legacy_csv_adds_fingerprint_column(tmp_path, make_beir_collection):
    make_beir_collection(tmp_path / "eval_data")
    results_file = tmp_path / "results.csv"
    legacy = pd.DataFrame(
        [
            {
                "run_idx": "legacy-run",
                "timestamp": "2026-01-01T00:00:00Z",
                "collection": "TESTCOL",
                "experiment_name": "old",
                "embedder": "old-model",
                "alpha": 1.0,
                "R@1": 0.9,
            }
        ]
    )
    legacy.to_csv(results_file, index=False)

    run_mocked_eval(tmp_path / "eval_data", results_file, tmp_path / "artifacts")

    df = pd.read_csv(results_file)
    assert len(df) == 2
    assert "dataset_fingerprint" in df.columns
    legacy_row = df[df["run_idx"] == "legacy-run"].iloc[0]
    assert pd.isna(legacy_row["dataset_fingerprint"])
    new_row = df[df["run_idx"] != "legacy-run"].iloc[0]
    assert isinstance(new_row["dataset_fingerprint"], str) and new_row["dataset_fingerprint"]


def test_evaluate_embedders_returns_new_run_rows(tmp_path, make_beir_collection):
    make_beir_collection(tmp_path / "eval_data")
    result = run_mocked_eval(tmp_path / "eval_data", tmp_path / "results.csv", tmp_path / "artifacts")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert {"embedder", "collection", "alpha", "R@1"} <= set(result.columns)


def test_missing_embedders_config_raises_helpful_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="retrieval-evaluator init"):
        evaluate_embedders(
            collections=["TESTCOL"],
            dataset_dir=tmp_path,
            results_file=tmp_path / "results.csv",
            artifacts_output_dir=tmp_path / "artifacts",
            embedders_config_path=tmp_path / "does-not-exist.yml",
        )


def test_sequential_runs_keep_timestamps_parseable(tmp_path, make_beir_collection):
    make_beir_collection(tmp_path / "eval_data")
    results_file = tmp_path / "results.csv"

    run_mocked_eval(tmp_path / "eval_data", results_file, tmp_path / "artifacts")
    # Second run must not reformat the first run's timestamp on rewrite.
    run_mocked_eval(tmp_path / "eval_data", results_file, tmp_path / "artifacts2", force_rerun=True)

    df = pd.read_csv(results_file, parse_dates=["timestamp"])
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), df["timestamp"].tolist()
