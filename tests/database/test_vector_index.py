import numpy as np
import pytest


def _make_vecs(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim)).astype("float32")


def test_numpy_insert_and_query_returns_correct_ids():
    from retrieval_evaluator.database.numpy_index import NumpyVectorIndex

    ids = ["a", "b", "c"]
    vecs = _make_vecs(3)
    with NumpyVectorIndex() as index:
        index.batch_insert(ids, vecs, ["text a", "text b", "text c"])
        results = index.query(vecs[0], k=1)

    assert results == ["a"]


def test_numpy_query_caps_k_and_breaks_ties_by_insertion_order():
    from retrieval_evaluator.database.numpy_index import NumpyVectorIndex

    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with NumpyVectorIndex() as index:
        index.batch_insert(["first", "second", "third"], vecs, ["t"] * 3)
        results = index.query([1.0, 0.0], k=10)

    assert results == ["first", "second", "third"]


def test_numpy_supports_multiple_insert_batches():
    from retrieval_evaluator.database.numpy_index import NumpyVectorIndex

    with NumpyVectorIndex() as index:
        index.batch_insert(["a"], [[1.0, 0.0]], ["a"])
        index.batch_insert(["b"], [[0.0, 1.0]], ["b"])
        assert index.query([0.0, 1.0], k=2) == ["b", "a"]


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[0.0, 0.0]], "zero-norm"),
        ([[np.nan, 1.0]], "finite"),
        ([], "two-dimensional"),
    ],
)
def test_numpy_rejects_invalid_vectors(vectors, message):
    from retrieval_evaluator.database.numpy_index import NumpyVectorIndex

    with NumpyVectorIndex() as index, pytest.raises(ValueError, match=message):
        index.batch_insert(["a"], vectors, ["a"])


def test_numpy_rejects_invalid_query_and_hybrid_search():
    from retrieval_evaluator.database.numpy_index import NumpyVectorIndex

    with NumpyVectorIndex() as index:
        index.batch_insert(["a"], [[1.0, 0.0]], ["a"])
        with pytest.raises(ValueError, match="hybrid"):
            index.query([1.0, 0.0], k=1, alpha=0.5)
        with pytest.raises(ValueError, match="dimension"):
            index.query([1.0, 0.0, 0.0], k=1)
        with pytest.raises(ValueError, match="zero-norm"):
            index.query([0.0, 0.0], k=1)


def test_numpy_teardown_clears_state():
    from retrieval_evaluator.database.numpy_index import NumpyVectorIndex

    index = NumpyVectorIndex()
    with index:
        index.batch_insert(["x"], [[1.0]], ["t"])

    assert index._vectors is None
    assert index._ids == []


def test_get_vector_index_returns_numpy_by_name_and_default():
    from retrieval_evaluator.database.numpy_index import NumpyVectorIndex
    from retrieval_evaluator.database.vector_index import get_vector_index

    assert isinstance(get_vector_index("numpy"), NumpyVectorIndex)
    assert isinstance(get_vector_index(), NumpyVectorIndex)


def test_get_vector_index_returns_weaviate():
    from unittest.mock import patch

    from retrieval_evaluator.database.vector_index import get_vector_index
    from retrieval_evaluator.database.weaviate_db import WeaviateVectorIndex

    with patch.object(WeaviateVectorIndex, "__init__", return_value=None):
        index = get_vector_index("weaviate")
    assert isinstance(index, WeaviateVectorIndex)


def test_get_vector_index_unknown_raises():
    from retrieval_evaluator.database.vector_index import get_vector_index

    with pytest.raises(ValueError, match="Unknown index backend"):
        get_vector_index("unknown")


def test_weaviate_collection_uses_non_deprecated_vector_config():
    from retrieval_evaluator.database.weaviate_db import _self_provided_vector_kwargs

    kwargs = _self_provided_vector_kwargs()
    assert list(kwargs.keys()) == ["vector_config"]


def test_weaviate_batch_insert_raises_on_failed_objects():
    from retrieval_evaluator.database.weaviate_db import WeaviateVectorIndex

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def add_object(self, **kwargs):
            del kwargs

    class FakeBatchManager:
        failed_objects = [type("Failure", (), {"message": "bad vector"})()]

        def dynamic(self):
            return FakeBatch()

    index = object.__new__(WeaviateVectorIndex)
    index._collection = type("Collection", (), {"batch": FakeBatchManager()})()

    with pytest.raises(RuntimeError, match="bad vector"):
        index.batch_insert(["c1"], _make_vecs(1), ["text"])


def test_weaviate_hybrid_search_only_scores_passage_content():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from retrieval_evaluator.database.weaviate_db import WeaviateVectorIndex

    query = MagicMock()
    query.hybrid.return_value = SimpleNamespace(
        objects=[SimpleNamespace(properties={"cid": "c1"})]
    )
    index = object.__new__(WeaviateVectorIndex)
    index._collection = SimpleNamespace(query=query)

    assert index.query(np.array([1.0], dtype=np.float32), 1, alpha=0.5, query="c1") == ["c1"]
    assert query.hybrid.call_args.kwargs["query_properties"] == ["content"]


def test_weaviate_bm25_does_not_require_vectors_and_only_scores_content():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from retrieval_evaluator.database.weaviate_db import WeaviateVectorIndex

    query = MagicMock()
    query.bm25.return_value = SimpleNamespace(
        objects=[SimpleNamespace(properties={"cid": "c1"})]
    )
    index = object.__new__(WeaviateVectorIndex)
    index._collection = SimpleNamespace(query=query)

    assert index.query(None, 1, alpha=0.0, query="password") == ["c1"]
    assert query.bm25.call_args.kwargs["query_properties"] == ["content"]
    query.hybrid.assert_not_called()
