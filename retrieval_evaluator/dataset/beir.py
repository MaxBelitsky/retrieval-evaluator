import datetime as dt
import hashlib
import json
import logging
import pathlib
import re
from collections import defaultdict
from typing import Any, Dict, Tuple

import pandas as pd

from retrieval_evaluator.utils.json_io import atomic_text_writer, write_json

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ensure_unique_query_ids(df: pd.DataFrame) -> None:
    if "qid" not in df.columns:
        return

    qid_rows = df[df["qid"].notna()]
    if qid_rows.empty:
        return

    normalized = qid_rows["qid"].astype(str)
    empty = qid_rows[normalized.str.strip() == ""]
    if not empty.empty:
        raise ValueError("Query IDs must be non-empty strings")
    duplicates = qid_rows[normalized.duplicated(keep=False)]
    if duplicates.empty:
        return

    duplicate_ids = ", ".join(sorted({str(qid) for qid in duplicates["qid"].tolist()}))
    raise ValueError(f"Duplicate query IDs detected while writing BEIR data: {duplicate_ids}")


def _ensure_unique_corpus_ids(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        raise ValueError("Dataset is missing the required 'id' column")
    if df["id"].isna().any() or (df["id"].astype(str).str.strip() == "").any():
        raise ValueError("Corpus IDs must be non-empty strings")
    normalized = df["id"].astype(str)
    duplicates = df[normalized.duplicated(keep=False)]
    if not duplicates.empty:
        duplicate_ids = ", ".join(sorted({str(cid) for cid in duplicates["id"].tolist()}))
        raise ValueError(f"Duplicate corpus IDs detected while writing BEIR data: {duplicate_ids}")


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def write_beir_dataset(df: pd.DataFrame, out_dir: pathlib.Path, seed: int | None = None):
    """
    Write BEIR-format files for corpus, queries, and qrels from the DataFrame.
    """
    required_columns = {"id", "text", "query", "qid"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing_columns)}")
    if df.empty:
        raise ValueError("Cannot write an empty evaluation dataset")
    _ensure_unique_corpus_ids(df)
    _ensure_unique_query_ids(df)
    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        raise ValueError("Corpus text must be non-empty")
    query_rows = df[
        df["query"].notna()
        & df["qid"].notna()
        & (df["query"].astype(str).str.strip() != "")
    ]
    if query_rows.empty:
        raise ValueError("Cannot write an evaluation dataset without any generated queries")

    out_dir.mkdir(parents=True, exist_ok=True)
    c_path, q_path, r_path = (
        out_dir / "corpus.jsonl",
        out_dir / "queries.jsonl",
        out_dir / "qrels.tsv",
    )
    _OPTIONAL_META = ["doc_id", "language", "chunk_id", "page_num",
                      "content_source", "ingestion_date", "chunk_type"]
    meta_fields = [k for k in _OPTIONAL_META if k in df.columns]
    n_corpus = n_queries = 0
    with (
        atomic_text_writer(c_path) as c_f,
        atomic_text_writer(q_path) as q_f,
        atomic_text_writer(r_path) as r_f,
    ):
        for _, row in df.iterrows():
            corpus_id = str(row["id"])
            metadata = {k: _json_safe(row[k]) for k in meta_fields}
            c_f.write(
                json.dumps(
                    {"_id": corpus_id, "text": str(row["text"]), "metadata": metadata},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            n_corpus += 1
            if pd.notna(row["query"]) and pd.notna(row["qid"]) and str(row["query"]).strip():
                query_id = str(row["qid"])
                q_f.write(
                    json.dumps(
                        {"_id": query_id, "text": str(row["query"])},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                r_f.write(f"{query_id}\t0\t{corpus_id}\t1\n")
                n_queries += 1
    meta = derive_beir_meta(out_dir)
    meta.update({"created_utc": _utc_now(), "seed": seed})
    write_json(meta, out_dir / "meta.json", indent=2)
    logger.info(f"\t\t{out_dir}  ({n_corpus} chunks / {n_queries} queries)")


def write_beir_raw(
    corpus: list[dict],
    queries: list[dict],
    qrels: dict[str, dict[str, int]],
    output_dir: pathlib.Path,
) -> None:
    """
    Write BEIR-format files from pre-built dicts.

    corpus:  list of {"_id": str, "text": str, "metadata": dict}
    queries: list of {"_id": str, "text": str}
    qrels:   {query_id: {corpus_id: relevance_score}}
    """
    if not corpus:
        raise ValueError("Cannot write an empty corpus")
    if not queries:
        raise ValueError("Cannot write a dataset without queries")

    corpus_ids = _validate_jsonl_rows(corpus, id_key="_id", label="corpus")
    query_ids = _validate_jsonl_rows(queries, id_key="_id", label="queries")
    _validate_qrel_references(qrels, query_ids=query_ids, corpus_ids=corpus_ids)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        atomic_text_writer(output_dir / "corpus.jsonl") as corpus_file,
        atomic_text_writer(output_dir / "queries.jsonl") as queries_file,
        atomic_text_writer(output_dir / "qrels.tsv") as qrels_file,
    ):
        for row in corpus:
            corpus_file.write(
                json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            )
        for row in queries:
            queries_file.write(
                json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            )
        for qid, cid_scores in qrels.items():
            for cid, score in cid_scores.items():
                qrels_file.write(f"{str(qid)}\t0\t{str(cid)}\t{int(score)}\n")

    meta = derive_beir_meta(output_dir)
    meta.update({"created_utc": _utc_now(), "seed": 0})
    write_json(meta, output_dir / "meta.json", indent=2)


def _dataset_file_paths(
    collection_dir: pathlib.Path,
    qrels_split: str = "test",
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    if not qrels_split or pathlib.Path(qrels_split).name != qrels_split:
        raise ValueError("qrels_split must be a non-empty filename component")
    corpus_path = collection_dir / "corpus.jsonl"
    queries_path = collection_dir / "queries.jsonl"
    flat_qrels_path = collection_dir / "qrels.tsv"
    split_qrels_path = collection_dir / "qrels" / f"{qrels_split}.tsv"
    if qrels_split == "test" and flat_qrels_path.exists():
        qrels_path = flat_qrels_path
    else:
        qrels_path = split_qrels_path
    return corpus_path, queries_path, qrels_path


def _hash_dataset_files(paths: tuple[pathlib.Path, ...]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_corpus_row(row: dict) -> dict:
    metadata = dict(row.get("metadata") or {})
    passthrough_fields = {
        key: value
        for key, value in row.items()
        if key not in {"_id", "text", "metadata"}
    }
    for key, value in passthrough_fields.items():
        metadata.setdefault(key, value)

    normalized = {
        **passthrough_fields,
        "cid": str(row["_id"]),
        "text": row.get("text", ""),
        "metadata": metadata,
    }
    return normalized


def _normalize_query_row(row: dict) -> dict:
    passthrough_fields = {
        key: value for key, value in row.items() if key not in {"_id", "qid", "text"}
    }
    return {
        **passthrough_fields,
        "qid": str(row["_id"]),
        "text": row["text"],
    }


def _read_jsonl(path: pathlib.Path, *, label: str) -> list[dict]:
    rows: list[dict] = []
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required {label} file not found: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def _validate_jsonl_rows(rows: list[dict], *, id_key: str, label: str) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for index, row in enumerate(rows, 1):
        raw_id = row.get(id_key)
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(f"{label} row {index} is missing a non-empty '{id_key}'")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label} row {index} is missing non-empty text")
        row_id = str(raw_id)
        if row_id in seen:
            duplicates.add(row_id)
        seen.add(row_id)
    if duplicates:
        raise ValueError(f"Duplicate {label} IDs: {', '.join(sorted(duplicates))}")
    return seen


def _validate_qrel_references(
    qrels: Dict[str, Dict[str, int]],
    *,
    query_ids: set[str],
    corpus_ids: set[str],
) -> None:
    normalized_qrels: dict[str, dict[str, int]] = {}
    for raw_qid, raw_scores in qrels.items():
        if not isinstance(raw_scores, dict):
            raise ValueError(f"Qrels for query {raw_qid!r} must be a mapping")
        qid = str(raw_qid)
        normalized_qrels[qid] = {}
        for raw_cid, raw_score in raw_scores.items():
            try:
                score = int(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid relevance score for query {qid!r}, corpus {raw_cid!r}: {raw_score!r}"
                ) from exc
            normalized_qrels[qid][str(raw_cid)] = score

    all_queries = set(normalized_qrels)
    missing_queries = sorted(all_queries - query_ids)
    if missing_queries:
        raise ValueError(f"Qrels reference unknown query IDs: {', '.join(missing_queries[:10])}")
    all_corpus_ids = {cid for scores in normalized_qrels.values() for cid in scores}
    missing_corpus = sorted(all_corpus_ids - corpus_ids)
    if missing_corpus:
        raise ValueError(f"Qrels reference unknown corpus IDs: {', '.join(missing_corpus[:10])}")

    positive_qrels = {
        qid: {cid: score for cid, score in scores.items() if score > 0}
        for qid, scores in normalized_qrels.items()
    }
    positive_qrels = {qid: scores for qid, scores in positive_qrels.items() if scores}
    if not positive_qrels:
        raise ValueError("Dataset has no positive relevance judgments")



def _load_qrels(qrels_path: pathlib.Path) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(dict)
    try:
        handle = qrels_path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Qrels file not found: {qrels_path}. Use qrels.tsv or qrels/<split>.tsv."
        ) from exc
    with handle as f:
        saw_data_row = False
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) == 1:
                parts = re.split(r"\s+", line)
            normalized_parts = [part.strip().lower() for part in parts]
            is_three_column_header = (
                len(parts) == 3
                and normalized_parts[0] in {"query-id", "query_id", "qid"}
                and normalized_parts[1] in {"corpus-id", "corpus_id", "doc-id", "doc_id"}
                and normalized_parts[2] in {"score", "relevance", "rel"}
            )
            is_four_column_header = (
                len(parts) == 4
                and normalized_parts[0] in {"query-id", "query_id", "qid"}
                and normalized_parts[2] in {"corpus-id", "corpus_id", "doc-id", "doc_id"}
                and normalized_parts[3] in {"score", "relevance", "rel"}
            )
            if not saw_data_row and (is_three_column_header or is_four_column_header):
                continue
            saw_data_row = True

            if len(parts) == 4:
                qid, _, cid, rel = parts
            elif len(parts) == 3:
                qid, cid, rel = parts
            else:
                raise ValueError(
                    f"Expected 3 or 4 tab-separated columns in {qrels_path}, got {len(parts)}"
                )

            if cid in qrels[qid]:
                raise ValueError(f"Duplicate qrel for query '{qid}' and corpus ID '{cid}' at line {line_number}")
            try:
                qrels[qid][cid] = int(rel)
            except ValueError as exc:
                raise ValueError(f"Invalid relevance score in {qrels_path} at line {line_number}: {rel!r}") from exc
    return dict(qrels)


def derive_beir_meta(collection_dir: pathlib.Path, qrels_split: str = "test") -> dict:
    corpus_path, queries_path, qrels_path = _dataset_file_paths(collection_dir, qrels_split=qrels_split)

    corpus = _read_jsonl(corpus_path, label="corpus")
    queries = _read_jsonl(queries_path, label="queries")
    qrels = _load_qrels(qrels_path)
    corpus_ids = _validate_jsonl_rows(corpus, id_key="_id", label="corpus")
    query_ids = _validate_jsonl_rows(queries, id_key="_id", label="queries")
    _validate_qrel_references(qrels, query_ids=query_ids, corpus_ids=corpus_ids)

    doc_ids = set()
    for row in corpus:
        metadata = row.get("metadata") or {}
        if metadata.get("doc_id") is not None:
            doc_ids.add(str(metadata["doc_id"]))
        elif row.get("title") is not None:
            doc_ids.add(str(row["title"]))
        else:
            doc_ids.add(str(row["_id"]))

    latest_mtime = max(path.stat().st_mtime for path in (corpus_path, queries_path, qrels_path))
    created_utc = (
        dt.datetime.fromtimestamp(latest_mtime, tz=dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return {
        "created_utc": created_utc,
        "dataset_fingerprint": _hash_dataset_files((corpus_path, queries_path, qrels_path)),
        "n_documents": len(doc_ids),
        "n_chunks": len(corpus),
        "n_queries": len(queries),
        "seed": None,
    }


def ensure_beir_meta(collection_dir: pathlib.Path, qrels_split: str = "test") -> dict:
    """Return current dataset metadata without modifying the input dataset."""
    meta_path = collection_dir / "meta.json"
    derived = derive_beir_meta(collection_dir, qrels_split=qrels_split)

    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a JSON object in {meta_path}")
        merged = {**loaded, **derived}
        if loaded.get("created_utc"):
            merged["created_utc"] = loaded["created_utc"]
        if "seed" in loaded:
            merged["seed"] = loaded["seed"]
        return merged

    return derived


def load_beir_dataset(
    collection_dir: pathlib.Path,
    qrels_split: str = "test",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Loads BEIR-format collection (corpus.jsonl, queries.jsonl, qrels.tsv).
    Returns (corpus_df, queries_df, qrels dict).
    """
    corpus_path, queries_path, qrels_path = _dataset_file_paths(collection_dir, qrels_split=qrels_split)
    raw_corpus = _read_jsonl(corpus_path, label="corpus")
    queries = _read_jsonl(queries_path, label="queries")
    qrels = _load_qrels(qrels_path)
    if not raw_corpus:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    if not queries:
        raise ValueError(f"Queries are empty: {queries_path}")

    corpus_ids = _validate_jsonl_rows(raw_corpus, id_key="_id", label="corpus")
    query_ids = _validate_jsonl_rows(queries, id_key="_id", label="queries")
    _validate_qrel_references(qrels, query_ids=query_ids, corpus_ids=corpus_ids)

    positive_qids = {
        qid for qid, scores in qrels.items() if any(score > 0 for score in scores.values())
    }
    queries = [row for row in queries if str(row["_id"]) in positive_qids]
    qrels = {qid: scores for qid, scores in qrels.items() if qid in positive_qids}
    corpus = [_normalize_corpus_row(row) for row in raw_corpus]
    corpus_df = pd.DataFrame(corpus)
    queries_df = pd.DataFrame([_normalize_query_row(row) for row in queries])
    return corpus_df, queries_df, qrels
