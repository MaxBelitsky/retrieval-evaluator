from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_runs(path: Path) -> pd.DataFrame:
    non_numeric = {
        "run_idx",
        "timestamp",
        "collection",
        "experiment_name",
        "embedder",
        "device",
        "index_backend",
        "qrels_split",
        "dataset_fingerprint",
    }
    df = pd.read_csv(path, dtype={column: "string" for column in non_numeric})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    for col in [c for c in df.columns if c not in non_numeric]:
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            continue
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


def latest_runs(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    latest_ts = df.groupby(group_cols, dropna=False)["timestamp"].transform("max")
    return df[df["timestamp"] == latest_ts].copy().sort_values("timestamp", ascending=False)


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run_data(run_dir: Path):
    try:
        rankings = load_json(run_dir / "rankings.json")
        macro2micro_path = run_dir / "macro2micro.json"
        macro2micro = load_json(macro2micro_path) if macro2micro_path.exists() else {}
        qrels = load_json(run_dir / "qrels.json")
        corpus_meta = {
            doc.get("cid") or doc.get("_id"): doc
            for doc in load_jsonl(run_dir / "corpus_meta.jsonl")
            if doc.get("cid") or doc.get("_id")
        }
        return rankings, macro2micro, qrels, corpus_meta
    except Exception as exc:
        raise RuntimeError(f"Error loading run data from {run_dir}: {exc}") from exc


def load_queries_jsonl(queries_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["_id"]] = row.get("text", "")
    return mapping


def current_dataset_fingerprint(collection_dir: Path, qrels_split: str = "test") -> str | None:
    """Fingerprint the current dataset files exactly as evaluation does."""
    from retrieval_evaluator.dataset.beir import derive_beir_meta

    try:
        return derive_beir_meta(collection_dir, qrels_split=qrels_split)["dataset_fingerprint"]
    except Exception:
        return None


def mark_dataset_status(runs_df: pd.DataFrame, current_fingerprints: dict) -> pd.DataFrame:
    """Add a dataset_status column: fresh (fingerprint matches), stale (differs), unknown (unverifiable)."""
    df = runs_df.copy()
    if "dataset_fingerprint" not in df.columns:
        df["dataset_fingerprint"] = pd.NA
    if df.empty:
        df["dataset_status"] = pd.Series(dtype=str)
        return df

    def _status(row) -> str:
        split_value = row.get("qrels_split", "test")
        split = split_value if isinstance(split_value, str) and split_value else "test"
        current = current_fingerprints.get(
            (row["collection"], split),
            current_fingerprints.get(row["collection"]),
        )
        fingerprint = row["dataset_fingerprint"]
        if pd.isna(fingerprint) or not fingerprint or current is None:
            return "unknown"
        return "fresh" if str(fingerprint) == str(current) else "stale"

    df["dataset_status"] = df.apply(_status, axis=1)
    return df
