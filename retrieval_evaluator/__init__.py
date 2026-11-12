from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

from retrieval_evaluator.utils.paths import safe_child_path


def evaluate(
    dataset_dir: str | Path | None = None,
    embedder_configs: dict[str, dict] | None = None,
    collections: list[str] | None = None,
    alphas: Sequence[float] = (1.0,),
    k_values: Sequence[int] = (1, 3, 5, 10),
    batch_size: int | Literal["auto"] = 64,
    artifacts_dir: str | Path | None = None,
    qrels_split: str = "test",
    index_backend: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate one or more embedder configs on a BEIR-format dataset directory.

    Returns a DataFrame with one row per (embedder, collection, alpha) combination.
    Columns include: embedder, collection, alpha, R@1, R@3, R@5, R@10, MRR@10, ...

    Args:
        dataset_dir: Path to a directory of BEIR-format collection subdirectories.
                     Defaults to the configured evaluation_dataset_dir.
        embedder_configs: Dict of {name: config_dict}. Required.
        collections: Which subdirectories to evaluate. Defaults to all subdirectories in dataset_dir.
        alphas: Hybrid search alpha values (1.0 = pure semantic, 0.0 = pure BM25).
        k_values: Retrieval cut-offs for metric computation.
        batch_size: Encoding batch size for HuggingFace models.
        artifacts_dir: Directory to write per-run artifacts (rankings, corpus meta, qrels).
                       When provided, artifacts are preserved after the call. To inspect
                       them in the dashboard, also save the returned DataFrame with
                       results.to_csv("results.csv", index=False), then launch
                       `retrieval-evaluator dashboard --results-file results.csv
                       --artifacts-dir <artifacts_dir> --dataset-dir <dataset_dir>`.
                       When omitted, artifacts are written to a temporary directory and
                       discarded. The API does not persist a results CSV automatically.
        qrels_split: BEIR qrels split to load from qrels/{split}.tsv when a root qrels.tsv
                     is not present.
        index_backend: Retrieval backend ("numpy" or "weaviate"). Defaults to config.
    """
    from retrieval_evaluator.config import get_evaluation_dataset_dir
    from retrieval_evaluator.evaluation.embedders_evaluation import evaluate_embedders

    if embedder_configs is None:
        raise ValueError("embedder_configs must be provided — pass a dict of {name: config_dict}")

    dataset_dir = get_evaluation_dataset_dir(dataset_dir)

    if collections is None:
        collections = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())

    with tempfile.TemporaryDirectory() as tmp:
        results_file = Path(tmp) / "results.csv"
        effective_artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir is not None else Path(tmp) / "artifacts"
        )
        if artifacts_dir is not None:
            effective_artifacts_dir.mkdir(parents=True, exist_ok=True)

        evaluate_embedders(
            collections=collections,
            embedder_configs=embedder_configs,
            alphas=alphas,
            batch_size=batch_size,
            k_values=k_values,
            dataset_dir=dataset_dir,
            results_file=results_file,
            artifacts_output_dir=effective_artifacts_dir,
            qrels_split=qrels_split,
            index_backend=index_backend,
        )

        if results_file.exists():
            return pd.read_csv(results_file)

    return pd.DataFrame()


def build_dataset(
    collection_name: str,
    chunks: list[dict],
    output_dir: str | Path = "eval_data",
    field_mapping: dict[str, str] | None = None,
    llm_model: str | None = None,
    n_docs: int = 30,
) -> Path:
    """
    Generate a BEIR-format evaluation dataset from an in-memory chunk list.

    Args:
        collection_name: Name for the output subdirectory.
        chunks: List of dicts. Must contain at least the fields named in field_mapping.
        output_dir: Root directory to write {collection_name}/ into.
        field_mapping: Maps canonical names (id, text, doc_id) to your chunk's field names.
                       Defaults to {"id": "id", "text": "content", "doc_id": "document_id"}.
        llm_model: LangChain model string for query generation (e.g. "openai:gpt-4o").
                   Defaults to the configured llm.model.
        n_docs: Number of diverse documents to sample per language.

    Returns:
        Path to the written dataset directory.
    """
    import copy

    from retrieval_evaluator.config import get_config
    from retrieval_evaluator.dataset.chunk_sources import InMemoryChunkSource
    from retrieval_evaluator.dataset.dataset_builder import build_eval_sets

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks must be a non-empty list of dictionaries")
    if any(not isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("chunks must contain only dictionaries")
    if n_docs <= 0:
        raise ValueError("n_docs must be positive")

    cfg = get_config()
    collection_path = safe_child_path(output_dir, collection_name, label="collection_name")
    dataset_cfg = copy.deepcopy(cfg["retrieval_evaluation"]["dataset"])
    settings = copy.deepcopy(cfg["settings"])
    settings["evaluation_dataset_dir"] = str(output_dir)
    dataset_cfg["n_docs_per_language"] = n_docs

    default_mapping = {"id": "id", "text": "content", "doc_id": "document_id"}
    effective_mapping = {**default_mapping, **(field_mapping or {})}
    dataset_cfg["field_mapping"] = {**dataset_cfg["field_mapping"], **effective_mapping}

    build_eval_sets(
        collections=[collection_name],
        chunk_source=InMemoryChunkSource(chunks),
        dataset_cfg=dataset_cfg,
        settings=settings,
        llm_model=llm_model or cfg["llm"]["model"],
    )
    return collection_path
