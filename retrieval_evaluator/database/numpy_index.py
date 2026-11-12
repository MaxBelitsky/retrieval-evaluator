from __future__ import annotations

from typing import Sequence

import numpy as np

from retrieval_evaluator.database.vector_index import VectorIndex


def _normalized_rows(vectors, *, label: str) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{label} must not contain zero-norm vectors")
    return array / norms


class NumpyVectorIndex(VectorIndex):
    """In-memory exact cosine-similarity index for modest evaluation corpora."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []

    def __enter__(self) -> "NumpyVectorIndex":
        self._vectors = None
        self._ids = []
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._vectors = None
        self._ids = []

    def batch_insert(self, ids: Sequence[str], vectors, texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must contain the same number of items")
        normalized = _normalized_rows(vectors, label="vectors")
        if normalized.shape[0] != len(ids):
            raise ValueError("ids and vectors must contain the same number of items")
        if self._vectors is not None and normalized.shape[1] != self._vectors.shape[1]:
            raise ValueError(
                f"vector dimension {normalized.shape[1]} does not match the index dimension {self._vectors.shape[1]}"
            )
        self._vectors = normalized if self._vectors is None else np.vstack([self._vectors, normalized])
        self._ids.extend(str(item_id) for item_id in ids)

    def query(self, vector, k: int, alpha: float = 1.0, query: str | None = None) -> list[str]:
        del query
        if alpha != 1.0:
            raise ValueError(
                "NumpyVectorIndex does not support hybrid search (alpha != 1.0). "
                "Set index_backend: weaviate to use hybrid search."
            )
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        if self._vectors is None:
            raise RuntimeError("Cannot query an empty index")

        query_vector = _normalized_rows(np.asarray(vector).reshape(1, -1), label="query vector")[0]
        if query_vector.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"query dimension {query_vector.shape[0]} does not match the index dimension {self._vectors.shape[1]}"
            )

        scores = self._vectors @ query_vector
        # Stable insertion order is the explicit tie-breaker.
        ranked = np.lexsort((np.arange(len(scores)), -scores))[: min(k, len(scores))]
        return [self._ids[int(index)] for index in ranked]

    def supports_hybrid(self) -> bool:
        return False
