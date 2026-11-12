import json
from unittest.mock import patch

import pandas as pd
import pytest

from retrieval_evaluator.evaluation.embedders_evaluation import _write_run_artifacts


def test_write_run_artifacts_snapshots_query_texts(tmp_path):
    corpus_eval = pd.DataFrame(
        [{"cid": "c1", "text": "hello world", "metadata": {"doc_id": "d1", "chunk_id": 1}}]
    )
    queries_df = pd.DataFrame([{"qid": "q1", "text": "what is hello?"}])
    run_dir = tmp_path / "run-1"

    _write_run_artifacts(
        run_dir,
        rankings={"q1": ["c1"]},
        macro2micro={},
        qrels={"q1": {"c1": 1}},
        corpus_eval=corpus_eval,
        queries_df=queries_df,
    )

    assert json.loads((run_dir / "queries.json").read_text()) == {"q1": "what is hello?"}
    assert json.loads((run_dir / "rankings.json").read_text()) == {"q1": ["c1"]}
    assert (run_dir / "qrels.json").exists()
    assert (run_dir / "corpus_meta.jsonl").read_text().strip()


def test_write_run_artifacts_preserves_existing_corpus_snapshot_on_failure(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    corpus_path = run_dir / "corpus_meta.jsonl"
    corpus_path.write_text("existing snapshot\n", encoding="utf-8")
    corpus_eval = pd.DataFrame(
        [{"cid": "c1", "text": "hello", "metadata": {"invalid": object()}}]
    )
    queries_df = pd.DataFrame([{"qid": "q1", "text": "hello?"}])

    with (
        patch("retrieval_evaluator.evaluation.embedders_evaluation.write_json"),
        pytest.raises(TypeError),
    ):
        _write_run_artifacts(
            run_dir,
            rankings={},
            macro2micro={},
            qrels={},
            corpus_eval=corpus_eval,
            queries_df=queries_df,
        )

    assert corpus_path.read_text(encoding="utf-8") == "existing snapshot\n"
