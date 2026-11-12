from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

EVAL_DATA_DIR = Path("eval_data")
EXAMPLE_COLLECTION_DIR = EVAL_DATA_DIR / "EXAMPLE"


@pytest.fixture(autouse=True)
def _require_example_dataset():
    if not EXAMPLE_COLLECTION_DIR.exists():
        pytest.skip("eval_data/EXAMPLE not found — run from project root")


def _make_mock_st(dim=16):
    mock = MagicMock()
    mock.encode.side_effect = lambda texts, **kw: np.random.default_rng(0).random(
        (len(texts), dim)
    ).astype(np.float32)
    mock.tokenize.return_value = {"input_ids": np.zeros((1, 16), dtype=np.int64)}
    return mock


class _FakeIndex:
    def __init__(self):
        self._ids = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def batch_insert(self, ids, vectors, texts):
        self._ids = list(ids)

    def query(self, vector, k, alpha=1.0, query=None):
        return self._ids[:k]

    def supports_hybrid(self) -> bool:
        return True


def test_evaluate_returns_dataframe():
    import pandas as pd

    from retrieval_evaluator import evaluate

    cfg = {"provider": "hf", "model": "mock", "max_tokens": 256}

    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_st()), \
         patch("retrieval_evaluator.evaluation.embedders_evaluation.get_vector_index", return_value=_FakeIndex()):
        result = evaluate(
            dataset_dir=EVAL_DATA_DIR,
            embedder_configs={"mock-model": cfg},
            collections=["EXAMPLE"],
        )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "R@1" in result.columns
    assert "embedder" in result.columns
    assert result["embedder"].iloc[0] == "mock-model"


def test_evaluate_raises_without_embedder_configs():
    from retrieval_evaluator import evaluate

    with pytest.raises(ValueError, match="embedder_configs"):
        evaluate(dataset_dir=EVAL_DATA_DIR, embedder_configs=None)


def test_evaluate_auto_discovers_collections(tmp_path):
    import shutil

    import pandas as pd

    from retrieval_evaluator import evaluate

    shutil.copytree(EXAMPLE_COLLECTION_DIR, tmp_path / "EXAMPLE")
    cfg = {"provider": "hf", "model": "mock", "max_tokens": 256}

    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_st()), \
         patch("retrieval_evaluator.evaluation.embedders_evaluation.get_vector_index", return_value=_FakeIndex()):
        result = evaluate(
            dataset_dir=tmp_path,
            embedder_configs={"mock-model": cfg},
        )

    assert isinstance(result, pd.DataFrame)
    assert "EXAMPLE" in result["collection"].values


def test_build_dataset_writes_beir_files(tmp_path, monkeypatch):
    from retrieval_evaluator import build_dataset

    chunks = [
        {"id": f"c{i}", "content": f"Chunk text number {i}.", "document_id": f"doc{i // 3}"}
        for i in range(12)
    ]
    field_mapping = {"id": "id", "text": "content", "doc_id": "document_id"}

    llm_calls = []

    def fake_generate(df, **kwargs):
        queries = [f"What is chunk {i}?" for i in range(len(df))]
        qids = [f"q{i}" for i in range(len(df))]
        langs = [None] * len(df)
        llm_calls.append(len(df))
        return queries, qids, langs

    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_st()), \
         patch("retrieval_evaluator.dataset.dataset_builder.generate_queries_for_chunks", side_effect=fake_generate):
        out = build_dataset(
            collection_name="test-col",
            chunks=chunks,
            output_dir=tmp_path,
            field_mapping=field_mapping,
            n_docs=2,
        )

    assert out == tmp_path / "test-col"
    assert (out / "corpus.jsonl").exists()
    assert (out / "queries.jsonl").exists()
    assert (out / "qrels.tsv").exists()
    assert len(llm_calls) == 1


def test_build_dataset_rejects_collection_path_escape(tmp_path):
    from retrieval_evaluator import build_dataset

    with pytest.raises(ValueError, match="collection_name"):
        build_dataset(
            collection_name="../outside",
            chunks=[{"id": "c1"}],
            output_dir=tmp_path,
        )


def test_evaluate_persists_artifacts_when_artifacts_dir_given(tmp_path):
    import pandas as pd

    from retrieval_evaluator import evaluate

    cfg = {"provider": "hf", "model": "mock", "max_tokens": 256}
    artifacts = tmp_path / "artifacts"

    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_st()), \
         patch("retrieval_evaluator.evaluation.embedders_evaluation.get_vector_index", return_value=_FakeIndex()):
        result = evaluate(
            dataset_dir=EVAL_DATA_DIR,
            embedder_configs={"mock-model": cfg},
            collections=["EXAMPLE"],
            artifacts_dir=artifacts,
        )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    rankings_files = list(artifacts.rglob("rankings.json"))
    assert rankings_files, "rankings.json must be written to artifacts_dir"
