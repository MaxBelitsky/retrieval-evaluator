import copy
import logging
from pathlib import Path
from typing import Sequence

from retrieval_evaluator.config import get_config
from retrieval_evaluator.dataset.beir import write_beir_dataset
from retrieval_evaluator.dataset.chunk_sources import ChunkSource, create_chunk_source
from retrieval_evaluator.dataset.document_selection import (
    embed_chunks,
    pool_document_embeddings,
    select_diverse_documents,
    sort_chunks,
)
from retrieval_evaluator.dataset.query_generation import generate_queries_for_chunks
from retrieval_evaluator.evaluation.embedder import Embedder
from retrieval_evaluator.utils.device import set_device
from retrieval_evaluator.utils.paths import safe_child_path

logger = logging.getLogger(__name__)

def build_eval_set_for_collection(
    collection_name: str,
    embedder: Embedder,
    sleep_s: float = 1.0,
    seed: int = 42,
    max_km_iters: int = 100,
    chunk_limit: int = 200,
    force_reload: bool = False,
    chunk_source: ChunkSource | None = None,
    dataset_cfg: dict | None = None,
    settings: dict | None = None,
    llm_model: str | None = None,
):
    """
    Full pipeline for building a BEIR-format eval set for a single configured chunk source.
    """
    cfg = get_config()
    dataset_cfg = copy.deepcopy(dataset_cfg or cfg["retrieval_evaluation"]["dataset"])
    settings = copy.deepcopy(settings or cfg["settings"])
    field_mapping = dataset_cfg["field_mapping"]
    n_documents = dataset_cfg["n_docs_per_language"]
    eval_dataset_dir = Path(settings["evaluation_dataset_dir"])
    out_dir = safe_child_path(eval_dataset_dir, collection_name, label="collection_name")

    if chunk_source is None:
        chunk_source = create_chunk_source(dataset_cfg)

    # Load chunks from the configured source
    vector_db_output = chunk_source.load_chunks(collection_name, force_reload=force_reload)

    # Normalise column names to canonical (text, doc_id, language) via field_mapping
    logger.info(f"{collection_name}: embedding chunks")
    df = sort_chunks(vector_db_output, field_mapping=field_mapping)
    missing_fields = [field for field in ("id", "text", "doc_id") if field not in df.columns]
    if missing_fields:
        raise ValueError(
            f"{collection_name}: source is missing required mapped field(s): {', '.join(missing_fields)}"
        )
    if df.empty:
        raise ValueError(f"{collection_name}: source returned no chunks")

    # Apply the optional row filter before validating retained values. Excluded
    # row types (for example tables) need not satisfy the text-chunk schema.
    filter_col = field_mapping.get("chunk_filter_field")
    filter_val = field_mapping.get("chunk_filter_value")
    if filter_col and filter_col not in df.columns:
        raise ValueError(f"{collection_name}: configured filter field '{filter_col}' was not found")
    if filter_col and filter_val is not None:
        df = df[df[filter_col] == filter_val]
        if df.empty:
            raise ValueError(f"{collection_name}: no chunks remain after applying the configured filter")

    if df[["id", "text", "doc_id"]].isna().any().any():
        raise ValueError(f"{collection_name}: required mapped fields contain null values")
    if not df["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{collection_name}: mapped field 'text' must contain strings")
    for field in ("id", "text", "doc_id"):
        if (df[field].astype(str).str.strip() == "").any():
            raise ValueError(f"{collection_name}: mapped field '{field}' contains empty values")
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    if df["id"].duplicated().any():
        raise ValueError(f"{collection_name}: source contains duplicate chunk IDs")

    chunk_vecs = embed_chunks(df, embedder, batch_size="auto")

    # Pool chunk embeddings into document embeddings
    logger.info(f"{collection_name}: computing document embeddings")
    doc_vectors, doc_ids = pool_document_embeddings(df, chunk_vecs)

    # Select diverse documents — per-language if language column is present, otherwise globally
    chosen: set[str] = set()
    lang_col = "language" if "language" in df.columns else None

    if lang_col:
        language_groups: dict[str, list[int]] = {}
        for index, doc_id in enumerate(doc_ids):
            values = df.loc[df["doc_id"] == doc_id, lang_col].dropna()
            values = values[values.astype(str).str.strip() != ""]
            language = str(values.iloc[0]) if not values.empty else "[unspecified]"
            language_groups.setdefault(language, []).append(index)

        for lang, idxs in language_groups.items():
            picked = select_diverse_documents(
                doc_vectors[idxs],
                [doc_ids[i] for i in idxs],
                n_documents=n_documents,
                random_state=seed,
                top_km_iters=max_km_iters,
            )
            chosen |= picked
            logger.info(f"\t\t[{lang}] {len(picked)} docs selected")
    else:
        chosen = select_diverse_documents(
            doc_vectors,
            doc_ids,
            n_documents=n_documents,
            random_state=seed,
            top_km_iters=max_km_iters,
        )
        logger.info(f"\t\t{len(chosen)} docs selected")

    df = df[df["doc_id"].isin(chosen)]

    # Generate queries for each chunk
    logger.info(f"{collection_name}: generating queries")
    queries, qids, langs = generate_queries_for_chunks(
        df,
        sleep_s=sleep_s,
        chunk_limit=chunk_limit,
        system_prompt=dataset_cfg["query_generation_system_prompt"],
        user_prompt_template=dataset_cfg["user_prompt_template"],
        llm_model=llm_model,
    )

    df = df.copy()
    df["query"] = queries
    df["qid"] = qids
    if langs and any(lang is not None for lang in langs):
        source_languages = df["language"] if "language" in df else [None] * len(df)
        df["language"] = [
            generated if generated is not None else source
            for generated, source in zip(langs, source_languages)
        ]

    if not df["query"].notna().any():
        raise RuntimeError(f"{collection_name}: no queries were generated successfully")
    # Unqueried passages remain retrieval candidates, including failed or skipped generations.
    n_unqueried = int(df["query"].isna().sum())
    if n_unqueried:
        logger.info("%s: retaining %s chunks without generated queries", collection_name, n_unqueried)

    # Write the dataset to file system
    write_beir_dataset(df, out_dir, seed=seed)
    logger.info(f"{collection_name}: dataset written to {out_dir}")


def build_eval_sets(
    collections: Sequence[str],
    chunk_source: ChunkSource | None = None,
    dataset_cfg: dict | None = None,
    settings: dict | None = None,
    llm_model: str | None = None,
    **kwargs,
):
    cfg = get_config()
    dataset_cfg = copy.deepcopy(dataset_cfg or cfg["retrieval_evaluation"]["dataset"])
    settings = copy.deepcopy(settings or cfg["settings"])
    device = set_device()
    logger.info(f"Device: {device}")
    embedder_cfg = {
        "model": dataset_cfg["document_selection_embedder"],
        "provider": "hf",
    }
    embedder = Embedder(embedder_cfg, device, max_batch_size=settings["max_batch_size"])
    if chunk_source is None:
        chunk_source = create_chunk_source(dataset_cfg)
    for collection in collections:
        logger.info(f"Building dataset for {collection}")
        build_eval_set_for_collection(
            collection,
            embedder,
            chunk_source=chunk_source,
            dataset_cfg=dataset_cfg,
            settings=settings,
            llm_model=llm_model,
            **kwargs,
        )
