import json

import pandas as pd

from retrieval_evaluator.dashboard.data import (
    current_dataset_fingerprint,
    load_queries_jsonl,
    load_run_data,
    load_runs,
    mark_dataset_status,
)


def test_current_fingerprint_ignores_stale_meta_json(tmp_path, make_beir_collection):
    collection_dir = make_beir_collection(tmp_path)
    (collection_dir / "meta.json").write_text(json.dumps({"dataset_fingerprint": "abc123"}))
    fingerprint = current_dataset_fingerprint(collection_dir)
    assert fingerprint != "abc123"
    assert isinstance(fingerprint, str) and len(fingerprint) == 64


def test_current_fingerprint_derives_when_meta_missing(tmp_path, make_beir_collection):
    collection_dir = make_beir_collection(tmp_path)
    fp = current_dataset_fingerprint(collection_dir)
    assert isinstance(fp, str) and len(fp) == 64  # sha256 hex


def test_current_fingerprint_returns_none_for_missing_dataset(tmp_path):
    assert current_dataset_fingerprint(tmp_path / "nope") is None


def test_mark_dataset_status_classifies_rows():
    runs = pd.DataFrame(
        {
            "collection": ["col-a", "col-a", "col-b", "col-c"],
            "dataset_fingerprint": ["fp-current", "fp-old", "fp-b", None],
        }
    )
    marked = mark_dataset_status(runs, {"col-a": "fp-current", "col-b": None, "col-c": "fp-c"})
    assert marked["dataset_status"].tolist() == ["fresh", "stale", "unknown", "unknown"]


def test_mark_dataset_status_handles_missing_column_and_empty_df():
    runs = pd.DataFrame({"collection": ["col-a"]})
    marked = mark_dataset_status(runs, {"col-a": "fp"})
    assert marked["dataset_status"].tolist() == ["unknown"]

    empty = mark_dataset_status(pd.DataFrame(columns=["collection"]), {})
    assert "dataset_status" in empty.columns and empty.empty


def test_mark_dataset_status_uses_qrels_split():
    runs = pd.DataFrame(
        {
            "collection": ["col-a", "col-a"],
            "qrels_split": ["dev", "test"],
            "dataset_fingerprint": ["dev-fp", "test-fp"],
        }
    )
    marked = mark_dataset_status(
        runs,
        {("col-a", "dev"): "dev-fp", ("col-a", "test"): "changed"},
    )
    assert marked["dataset_status"].tolist() == ["fresh", "stale"]


def test_load_runs_preserves_numeric_looking_qrels_split(tmp_path):
    path = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "run_idx": "r1",
                "timestamp": "2026-01-01T00:00:00Z",
                "collection": "col-a",
                "embedder": "model",
                "qrels_split": "2024",
                "index_backend": "legacy-backend",
                "dataset_fingerprint": "012345",
                "R@1": 1.0,
            }
        ]
    ).to_csv(path, index=False)

    loaded = load_runs(path)

    assert loaded.loc[0, "qrels_split"] == "2024"
    assert loaded.loc[0, "dataset_fingerprint"] == "012345"


def test_dashboard_reads_new_runs_and_replaced_artifacts(tmp_path):
    results_path = tmp_path / "results.csv"
    original = pd.DataFrame([{
        "run_idx": "run1", "timestamp": "2026-01-01T00:00:00Z", "embedder": "first", "R@1": 0.0,
    }])
    original.to_csv(results_path, index=False)
    assert load_runs(results_path)["embedder"].tolist() == ["first"]
    updated = pd.concat([original, original.assign(run_idx="run2", embedder="second")])
    updated.to_csv(results_path, index=False)
    assert set(load_runs(results_path)["embedder"]) == {"first", "second"}

    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "rankings.json").write_text(json.dumps({"q1": []}))
    (run_dir / "qrels.json").write_text(json.dumps({"q1": {"c1": 1}}))
    (run_dir / "corpus_meta.jsonl").write_text('{"cid":"c1","text":"old passage"}\n')
    assert load_run_data(run_dir)[0] == {"q1": []}

    (run_dir / "rankings.json").write_text(json.dumps({"q1": ["c1"]}))
    (run_dir / "corpus_meta.jsonl").write_text('{"cid":"c1","text":"new passage"}\n')
    rankings, _, _, corpus = load_run_data(run_dir)
    assert rankings == {"q1": ["c1"]}
    assert corpus["c1"]["text"] == "new passage"

    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text('{"_id":"q1","text":"old question"}\n')
    assert load_queries_jsonl(queries_path) == {"q1": "old question"}
    queries_path.write_text('{"_id":"q1","text":"new question"}\n')
    assert load_queries_jsonl(queries_path) == {"q1": "new question"}
