from pathlib import Path

import pytest

from retrieval_evaluator.config import reset_config


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path_factory, monkeypatch):
    """
    Point every test at an empty config file so results don't depend on whatever
    config/retrieval-evaluator.yaml happens to exist in the developer's working
    directory (e.g. after running `retrieval-evaluator init`). Tests exercise the
    built-in DEFAULT_CONFIG unless they set RETRIEVAL_EVALUATOR_CONFIG themselves.
    """
    empty_config = tmp_path_factory.mktemp("config") / "retrieval-evaluator.yaml"
    empty_config.write_text("", encoding="utf-8")
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_CONFIG", str(empty_config))
    reset_config()
    yield
    reset_config()


@pytest.fixture
def make_beir_collection():
    """Factory fixture: writes a minimal valid BEIR collection (1 chunk, 1 query)."""

    def _make(root: Path, name: str = "TESTCOL") -> Path:
        collection_dir = root / name
        collection_dir.mkdir(parents=True, exist_ok=True)
        (collection_dir / "corpus.jsonl").write_text(
            '{"_id": "c1", "title": "T", "text": "hello world", "metadata": {"doc_id": "d1", "chunk_id": 1}}\n',
            encoding="utf-8",
        )
        (collection_dir / "queries.jsonl").write_text(
            '{"_id": "q1", "text": "what is hello?"}\n', encoding="utf-8"
        )
        (collection_dir / "qrels.tsv").write_text(
            "query-id\t0\tcorpus-id\tscore\nq1\t0\tc1\t1\n", encoding="utf-8"
        )
        return collection_dir

    return _make
