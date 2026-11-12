from typing import Any, Iterable

import numpy as np
import pandas as pd

from retrieval_evaluator.evaluation.embedder import Embedder
from retrieval_evaluator.utils.constants import EmbeddingTextType


def sort_chunks(
    vector_db_output: Iterable[dict[str, Any]],
    field_mapping: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Loads raw chunks into a DataFrame and renames schema-specific column names
    to canonical internal names using the provided field_mapping.
    """
    df = pd.DataFrame(vector_db_output)
    canonical_fields = {"id", "text", "doc_id", "chunk_id", "language"}
    mapping = field_mapping or {}
    rename = {v: k for k, v in mapping.items() if v and k in canonical_fields and v != k}
    if rename:
        df.rename(columns=rename, inplace=True)
    if "chunk_id" in df.columns:
        numeric_chunk_ids = pd.to_numeric(df["chunk_id"], errors="coerce")
        df = df.assign(
            _chunk_is_text=numeric_chunk_ids.isna(),
            _chunk_number=numeric_chunk_ids.fillna(0),
            _chunk_text=df["chunk_id"].astype(str),
        )
        sort_cols = [c for c in ["doc_id", "_chunk_is_text", "_chunk_number", "_chunk_text"] if c in df]
        df.sort_values(sort_cols, inplace=True, kind="stable")
        df.drop(columns=["_chunk_is_text", "_chunk_number", "_chunk_text"], inplace=True)
    elif "doc_id" in df.columns:
        df.sort_values("doc_id", inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)
    return df


def embed_chunks(df: pd.DataFrame, embedder: Embedder, batch_size: int | str = 64) -> np.ndarray:
    """
    Returns chunk embeddings for all rows in a dataframe.
    """
    texts = df["text"].tolist()
    return embedder.encode(texts, batch_size=batch_size, text_type=EmbeddingTextType.passage).astype(np.float32)


def pool_document_embeddings(df: pd.DataFrame, chunk_vecs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Pools (averages) chunk vectors per document.
    """
    df = df.copy()
    df["vec"] = list(chunk_vecs)
    # groupby sorts keys; derive doc_ids from the index so order matches pooled vectors.
    grouped = df.groupby("doc_id")["vec"].apply(
        lambda rows: np.mean(np.vstack(rows), axis=0)
    )
    doc_ids = grouped.index.tolist()
    pooled = grouped.tolist()
    return np.vstack(pooled).astype(np.float32), doc_ids


def select_diverse_documents(
    doc_vecs: np.ndarray,
    titles: list[str],
    n_documents: int,
    random_state: int = 42,
    top_km_iters: int = 100,
) -> set[str]:
    """
    Select one representative document nearest each KMeans cluster centroid.
    """
    if n_documents <= 0:
        raise ValueError("n_documents must be positive")
    if len(doc_vecs) != len(titles):
        raise ValueError("doc_vecs and titles must contain the same number of documents")
    if not titles:
        raise ValueError("Cannot select documents from an empty collection")
    if doc_vecs.ndim != 2 or not np.isfinite(doc_vecs).all():
        raise ValueError("doc_vecs must be a finite two-dimensional array")
    if len(titles) <= n_documents:
        return set(titles)

    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            "Diverse document selection requires the dataset extra. "
            'Install with: pip install "retrieval-evaluator[dataset]"'
        ) from exc

    # Cluster document embeddings
    n_clusters = min(n_documents, len(np.unique(doc_vecs, axis=0)))
    km = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        max_iter=top_km_iters,
        random_state=random_state,
    ).fit(doc_vecs)

    chosen_indices: list[int] = []
    for cluster_id in sorted(set(km.labels_)):
        indices = np.where(km.labels_ == cluster_id)[0]
        cluster_vecs = doc_vecs[indices]
        centroid = km.cluster_centers_[cluster_id]
        closest_index = indices[np.linalg.norm(cluster_vecs - centroid, axis=1).argmin()]
        chosen_indices.append(int(closest_index))

    # Degenerate/duplicate vectors can leave KMeans with fewer populated clusters.
    # Fill deterministically with the points farthest from those already selected.
    while len(chosen_indices) < n_documents:
        remaining = [index for index in range(len(titles)) if index not in chosen_indices]
        distances = [
            min(np.linalg.norm(doc_vecs[index] - doc_vecs[chosen]) for chosen in chosen_indices)
            for index in remaining
        ]
        chosen_indices.append(remaining[int(np.argmax(distances))])

    return {titles[index] for index in chosen_indices}
