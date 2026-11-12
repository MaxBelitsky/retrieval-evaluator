from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class VectorIndex(ABC):
    @abstractmethod
    def __enter__(self) -> "VectorIndex": ...

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None: ...

    @abstractmethod
    def batch_insert(
        self, ids: Sequence[str], vectors: Any | None, texts: Sequence[str]
    ) -> None:
        """Insert candidates; vectors is None for a lexical-only evaluation."""
        ...

    @abstractmethod
    def query(
        self, vector: Any | None, k: int, alpha: float = 1.0, query: str | None = None
    ) -> list[str]: ...

    @abstractmethod
    def supports_hybrid(self) -> bool: ...


def get_vector_index(backend: str = "numpy") -> VectorIndex:
    if backend == "numpy":
        from retrieval_evaluator.database.numpy_index import NumpyVectorIndex

        return NumpyVectorIndex()
    if backend == "weaviate":
        try:
            from retrieval_evaluator.database.weaviate_db import WeaviateVectorIndex
        except ImportError as exc:
            raise ImportError(
                "Weaviate evaluation requires the weaviate extra. "
                'Install with: pip install "retrieval-evaluator[weaviate]"'
            ) from exc
        return WeaviateVectorIndex()
    raise ValueError(
        f"Unknown index backend: {backend!r}. Choose 'numpy' or 'weaviate'."
    )
