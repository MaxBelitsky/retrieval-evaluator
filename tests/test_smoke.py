"""
End-to-end smoke test on the bundled EXAMPLE dataset.
Uses a mocked SentenceTransformer to avoid downloading any model.
Run from the project root: uv run pytest tests/test_smoke.py
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

EXAMPLE_DIR = Path("eval_data/EXAMPLE")


@pytest.fixture(autouse=True)
def _require_example_dataset():
    if not EXAMPLE_DIR.exists():
        pytest.skip("eval_data/EXAMPLE not found — run from project root")


def _make_mock_st(dim=64):
    """Return a SentenceTransformer mock whose encode() produces deterministic vectors."""
    mock = MagicMock()
    mock.encode.side_effect = lambda texts, **kw: np.random.default_rng(0).random(
        (len(texts), dim)
    ).astype(np.float32)
    mock.tokenize.return_value = {"input_ids": np.zeros((1, 16), dtype=np.int64)}
    return mock


class _FakeVectorIndex:
    def __init__(self):
        self._ids = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def batch_insert(self, ids, vectors, texts):
        del vectors, texts
        self._ids = list(ids)

    def query(self, vector, k, alpha=1.0, query=None):
        del vector, alpha, query
        return self._ids[:k]

    def supports_hybrid(self) -> bool:
        return True


def test_smoke_eval_pipeline(tmp_path):
    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_st()):
        from retrieval_evaluator.evaluation.embedder import Embedder
        from retrieval_evaluator.evaluation.embedders_evaluation import evaluate_embedder_on_collection

        cfg = {"provider": "hf", "model": "mock", "max_tokens": 512}
        embedder = Embedder(cfg, device="cpu")

        with patch(
            "retrieval_evaluator.evaluation.embedders_evaluation.get_vector_index",
            return_value=_FakeVectorIndex(),
        ):
            metrics = evaluate_embedder_on_collection(
                collection_dir=EXAMPLE_DIR,
                embedder=embedder,
                batch_size=4,
                run_id="smoke",
                k_values=[1, 5, 10],
                alpha=1.0,
                artifacts_output_dir=tmp_path,
            )

    assert set(metrics) >= {"R@1", "R@5", "R@10", "MRR@10"}
    assert all(0.0 <= v <= 1.0 for k, v in metrics.items() if k != "chunks_indexed")
    assert (tmp_path / "smoke" / "rankings.json").exists()
