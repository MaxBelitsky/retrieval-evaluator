import datetime as dt
import json
import logging
import pathlib
import time
from itertools import product
from typing import Iterable, Literal

import pandas as pd
import yaml

from retrieval_evaluator.config import get_config, get_evaluation_dataset_dir
from retrieval_evaluator.database.vector_index import get_vector_index
from retrieval_evaluator.dataset.beir import ensure_beir_meta, load_beir_dataset
from retrieval_evaluator.evaluation.chunk_merging import merge_chunks
from retrieval_evaluator.evaluation.embedder import Embedder
from retrieval_evaluator.evaluation.metrics import compute_ranking_metrics
from retrieval_evaluator.evaluation.run_id import create_run_id
from retrieval_evaluator.utils.constants import EmbeddingTextType
from retrieval_evaluator.utils.device import set_device
from retrieval_evaluator.utils.json_io import atomic_text_writer, write_json
from retrieval_evaluator.utils.paths import safe_child_path

logger = logging.getLogger(__name__)


class EvaluationRunError(RuntimeError):
    def __init__(self, failures: list[dict[str, object]]):
        self.failures = failures
        preview = [
            f"{failure['embedder']} on {failure['collection']} (alpha={failure['alpha']}): {failure['error']}"
            for failure in failures[:3]
        ]
        preview_text = "; ".join(preview)
        if len(failures) > 3:
            preview_text += f"; and {len(failures) - 3} more"
        super().__init__(f"{len(failures)} evaluation run(s) failed: {preview_text}")


def _supports_chunk_merging(corpus_df: pd.DataFrame) -> bool:
    if "metadata" not in corpus_df.columns:
        return False
    for metadata in corpus_df["metadata"]:
        if not isinstance(metadata, dict):
            return False
        if metadata.get("doc_id") is None or metadata.get("chunk_id") is None:
            return False
    return True


def evaluate_embedder_on_collection(
    collection_dir: pathlib.Path,
    embedder: Embedder,
    batch_size: int | Literal["auto"],
    run_id: str,
    k_values: Iterable[int] = (1, 3, 5, 10),
    alpha: float = 1.0,
    artifacts_output_dir: pathlib.Path | None = None,
    micro_chunk_tokens: int | None = None,
    index_backend: str | None = None,
    qrels_split: str = "test",
) -> dict:
    """
    Returns `metrics` and saves artifacts: rankings, mapping, qrels, meta.
    """
    artifacts_output_dir = (
        pathlib.Path(artifacts_output_dir)
        if artifacts_output_dir is not None
        else pathlib.Path(get_config()["settings"]["run_details_directory"])
    )
    if micro_chunk_tokens is None:
        micro_chunk_tokens = get_config()["retrieval_evaluation"]["pipeline"]["micro_chunk_tokens"]
    micro_chunk_tokens = int(micro_chunk_tokens)
    if micro_chunk_tokens <= 0:
        raise ValueError("micro_chunk_tokens must be a positive integer")
    k_values = sorted({int(k) for k in k_values})
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain at least one positive integer")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    # Load the evaluation set in BEIR format
    corpus_df, queries_df, qrels = load_beir_dataset(collection_dir, qrels_split=qrels_split)

    # Merge chunks into bigger "macrochunks" if embedder supports larger context
    if (
        embedder.supports_tokenization
        and embedder.max_tokens > micro_chunk_tokens
        and _supports_chunk_merging(corpus_df)
    ):
        logger.info("Merging chunks")
        corpus_eval, macro2micro = merge_chunks(corpus_df, embedder)
    else:
        if embedder.max_tokens > micro_chunk_tokens and not embedder.supports_tokenization:
            logger.info("Skipping chunk merging (not supported for provider '%s')", embedder.cfg["provider"])
        elif embedder.max_tokens > micro_chunk_tokens:
            logger.warning(
                "Skipping chunk merging because corpus metadata does not include both doc_id and chunk_id"
            )
        corpus_eval = corpus_df.copy()
        macro2micro = {}

    backend = index_backend or get_config()["settings"].get("index_backend", "numpy")
    with get_vector_index(backend) as idx:
        if alpha != 1.0 and not idx.supports_hybrid():
            raise ValueError(
                f"index_backend '{backend}' does not support hybrid search. "
                "Set index_backend: weaviate in config to use alpha != 1.0."
            )

        passages = corpus_eval["text"].tolist()
        lexical_only = alpha == 0.0
        if lexical_only:
            p_vecs = None
        else:
            logger.info("\nEmbedding documents:")
            p_vecs = embedder.encode(passages, batch_size, text_type=EmbeddingTextType.passage)
        idx.batch_insert(corpus_eval["cid"].tolist(), p_vecs, texts=passages)

        rankings = {}
        queries = queries_df["text"].tolist()
        if lexical_only:
            q_vecs = [None] * len(queries)
        else:
            logger.info("\nEmbedding queries:")
            q_vecs = embedder.encode(queries, batch_size, text_type=EmbeddingTextType.query)
        for qid, q_vec, query in zip(queries_df["qid"], q_vecs, queries):
            ranked = list(idx.query(q_vec, max(k_values), alpha=alpha, query=query))
            if len(ranked) != len(set(ranked)):
                raise RuntimeError(f"index_backend '{backend}' returned duplicate IDs for query {qid!r}")
            unknown_ids = set(ranked) - set(corpus_eval["cid"])
            if unknown_ids:
                preview = ", ".join(sorted(str(item) for item in unknown_ids)[:5])
                raise RuntimeError(
                    f"index_backend '{backend}' returned IDs that were not indexed: {preview}"
                )
            rankings[qid] = ranked

    # Compute the metrics
    metrics = compute_ranking_metrics(rankings, qrels, macro2micro, k_values=k_values)
    metrics["chunks_indexed"] = len(corpus_eval)

    # Persist artifacts
    _write_run_artifacts(artifacts_output_dir / run_id, rankings, macro2micro, qrels, corpus_eval, queries_df)
    return metrics

def _write_run_artifacts(
    run_dir: pathlib.Path,
    rankings: dict,
    macro2micro: dict,
    qrels: dict,
    corpus_eval: pd.DataFrame,
    queries_df: pd.DataFrame,
) -> None:
    """Write self-contained run artifacts, including a snapshot of the query texts."""
    write_json(rankings, run_dir / "rankings.json")
    write_json(macro2micro, run_dir / "macro2micro.json")
    write_json(qrels, run_dir / "qrels.json")
    write_json(dict(zip(queries_df["qid"], queries_df["text"])), run_dir / "queries.json")
    meta_path = run_dir / "corpus_meta.jsonl"
    with atomic_text_writer(meta_path) as f:
        for row in corpus_eval.itertuples():
            f.write(json.dumps({"cid": row.cid, "text": row.text, "metadata": row.metadata}, ensure_ascii=False) + "\n")

    logger.info(f"Saved artifacts to {run_dir}")

def evaluate_embedders(
    collections: Iterable[str],
    embedders: Iterable[str] = (),
    force_rerun: bool = False,
    alphas: Iterable[float] = (1.0,),
    batch_size: int | Literal["auto"] = 64,
    k_values: Iterable[int] = (1, 3, 5, 10),
    embedder_configs: dict[str, dict] | None = None,
    dataset_dir: str | pathlib.Path | None = None,
    results_file: str | pathlib.Path | None = None,
    artifacts_output_dir: str | pathlib.Path | None = None,
    embedders_config_path: str | pathlib.Path | None = None,
    micro_chunk_tokens: int | None = None,
    qrels_split: str = "test",
    index_backend: str | None = None,
) -> pd.DataFrame:
    cfg = get_config()
    dataset_dir = get_evaluation_dataset_dir(dataset_dir)
    results_file = pathlib.Path(results_file or cfg["settings"]["evaluation_results_file"])
    artifacts_output_dir = pathlib.Path(
        artifacts_output_dir or cfg["settings"]["run_details_directory"]
    )
    embedders_config_path = pathlib.Path(
        embedders_config_path or cfg["settings"]["embedders_config_path"]
    )
    max_batch_size = int(cfg["settings"]["max_batch_size"])
    index_backend = index_backend or cfg["settings"].get("index_backend", "numpy")
    if micro_chunk_tokens is None:
        micro_chunk_tokens = cfg["retrieval_evaluation"]["pipeline"]["micro_chunk_tokens"]
    micro_chunk_tokens = int(micro_chunk_tokens)
    if micro_chunk_tokens <= 0:
        raise ValueError("micro_chunk_tokens must be a positive integer")
    k_values = sorted({int(k) for k in k_values})
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain at least one positive integer")
    alphas = list(alphas)
    if not alphas or any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("alphas must contain values between 0.0 and 1.0")
    collections = list(collections)
    if not collections:
        raise ValueError("At least one collection must be provided")

    # Load all embedder configs YAML
    if embedder_configs is None:
        if not embedders_config_path.exists():
            raise FileNotFoundError(
                f"Embedders config not found: {embedders_config_path}\n"
                "Run `retrieval-evaluator init` to create starter config files, "
                "or pass `--model <model-id>` for a config-free run."
            )
        with embedders_config_path.open() as f:
            default_configs = yaml.safe_load(f)
    else:
        default_configs = embedder_configs

    if not isinstance(default_configs, dict) or not default_configs:
        raise ValueError("Embedder configuration must contain at least one named embedder")

    # Select the provided embedder configs – all configs if no embedders are provided
    if embedders:
        embedder_configs = {}
        for embedder_name in embedders:
            if embedder_name not in default_configs:
                raise ValueError(f"No config defined for embedder '{embedder_name}'")
            embedder_configs[embedder_name] = default_configs[embedder_name]
    else:
        embedder_configs = default_configs

    existing_experiments = None
    if results_file.exists():
        existing_experiments = pd.read_csv(results_file)

    # Create and evaluate the embedders
    device = set_device()
    logger.info(f"The device is set to {device}")
    failures: list[dict[str, object]] = []
    all_results: list[dict] = []
    for embedder_name, embedder_cfg in embedder_configs.items():
        embedder = Embedder(embedder_cfg, device=device, max_batch_size=max_batch_size)
        logger.info(embedder)

        for collection_name, alpha in product(collections, alphas):
            run_idx = create_run_id(
                collection_name,
                embedder.identity_config,
                alpha,
                dataset_dir=dataset_dir,
                k_values=k_values,
                index_backend=index_backend,
                micro_chunk_tokens=micro_chunk_tokens,
                qrels_split=qrels_split,
            )
            logger.info(f"Evaluating run id: {run_idx}")

            if (
                existing_experiments is not None
                and "run_idx" in existing_experiments.columns
                and run_idx in existing_experiments["run_idx"].values
                and not force_rerun
            ):
                logger.info(f"Skipping collection {collection_name} and embedder {embedder_name}")
                continue

            collection_dir = safe_child_path(dataset_dir, collection_name, label="collection_name")
            start_time = time.perf_counter()
            try:
                metrics = evaluate_embedder_on_collection(
                    collection_dir,
                    embedder,
                    batch_size,
                    run_idx,
                    k_values,
                    alpha,
                    artifacts_output_dir=artifacts_output_dir,
                    micro_chunk_tokens=micro_chunk_tokens,
                    index_backend=index_backend,
                    qrels_split=qrels_split,
                )
            except Exception as e:
                logger.exception(
                    "Error evaluating %s on %s (alpha=%s)",
                    embedder_name,
                    collection_name,
                    alpha,
                )
                failures.append(
                    {
                        "embedder": embedder_name,
                        "collection": collection_name,
                        "alpha": alpha,
                        "error": str(e),
                    }
                )
                continue
            runtime = time.perf_counter() - start_time

            collection_meta = ensure_beir_meta(collection_dir, qrels_split=qrels_split)

            # Add an experiment name
            experiment_name = embedder_name + f"_{alpha}_{embedder.max_tokens}"

            # Save the results of the run
            run_result = {
                "run_idx": run_idx,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "collection": collection_name,
                "experiment_name": experiment_name,
                "embedder": embedder_name,
                "alpha": alpha,
                "context_window_used": embedder.max_tokens,
                "runtime_s": round(runtime, 1),
                "device": device.type,
                "batch_size": batch_size,
                "index_backend": index_backend,
                "qrels_split": qrels_split,
                "dataset_fingerprint": str(collection_meta.get("dataset_fingerprint") or ""),
                **metrics,
            }
            all_results.append(run_result)
            results_file.parent.mkdir(parents=True, exist_ok=True)

            new_row = pd.DataFrame([run_result])
            if existing_experiments is None:
                existing_experiments = new_row
            else:
                # A deterministic run ID represents one logical experiment.
                # Forced reruns replace that row and its artifacts instead of
                # creating indistinguishable history entries.
                if "run_idx" in existing_experiments.columns:
                    existing_experiments = existing_experiments[
                        existing_experiments["run_idx"] != run_idx
                    ]
                existing_experiments = pd.concat([existing_experiments, new_row], ignore_index=True)
            existing_experiments.to_csv(results_file, index=False)

    if failures:
        raise EvaluationRunError(failures)

    return pd.DataFrame(all_results)
