from __future__ import annotations

import json
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

from retrieval_evaluator.config import get_evaluation_dataset_dir
from retrieval_evaluator.dataset.beir import ensure_beir_meta
from retrieval_evaluator.utils.paths import safe_child_path

RUN_ID_SCHEMA_VERSION = 2


def create_run_id(
    collection_name: str,
    embedder_cfg: dict,
    alpha: float,
    dataset_dir: str | Path | None = None,
    *,
    k_values: list[int] | tuple[int, ...] = (1, 3, 5, 10),
    index_backend: str = "numpy",
    micro_chunk_tokens: int = 512,
    qrels_split: str = "test",
) -> str:
    """
    Create a stable run identifier from the dataset fingerprint and eval settings.
    """
    eval_dir = get_evaluation_dataset_dir(dataset_dir)
    collection_dir = safe_child_path(eval_dir, collection_name, label="collection_name")
    meta = ensure_beir_meta(collection_dir, qrels_split=qrels_split)

    _raw_fp = meta.get("dataset_fingerprint") or meta.get("created_utc")
    if not _raw_fp:
        raise ValueError(
            f"Dataset at '{collection_dir}' has no 'dataset_fingerprint' or "
            "'created_utc' in meta.json — regenerate the dataset to fix this."
        )
    identity = {
        "schema_version": RUN_ID_SCHEMA_VERSION,
        "collection": collection_name,
        "dataset_fingerprint": str(_raw_fp),
        "embedder": embedder_cfg,
        "alpha": float(alpha),
        "k_values": sorted({int(k) for k in k_values}),
        "index_backend": index_backend,
        "micro_chunk_tokens": int(micro_chunk_tokens),
        "qrels_split": qrels_split,
    }
    canonical_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return str(uuid5(NAMESPACE_DNS, canonical_identity))
