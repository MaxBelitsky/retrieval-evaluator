from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class DatasetImporter:
    """Convert CSV, JSONL, or Hugging Face datasets to BEIR format."""

    @staticmethod
    def _validate_fields(
        rows: list[dict[str, Any]],
        *,
        dataset_label: str,
        required_fields: dict[str, str],
    ) -> None:
        if not rows:
            return

        available = list(rows[0].keys())
        for argument_name, field_name in required_fields.items():
            if field_name not in rows[0]:
                raise ValueError(
                    f"{argument_name} '{field_name}' not found in {dataset_label}. "
                    f"Available fields: {available}"
                )

    def import_from_csv(
        self,
        corpus_path: Path,
        queries_path: Path,
        output_dir: Path,
        corpus_id_field: str,
        corpus_text_field: str,
        query_id_field: str,
        query_text_field: str,
        relevant_id_field: str,
        corpus_doc_id_field: str | None = None,
    ) -> Path:
        """Import from two CSV files (corpus + queries) and write BEIR format."""
        corpus_path = Path(corpus_path)
        queries_path = Path(queries_path)

        with corpus_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            corpus_rows = list(reader)

        self._validate_fields(
            corpus_rows,
            dataset_label="corpus CSV",
            required_fields={
                "corpus_id_field": corpus_id_field,
                "corpus_text_field": corpus_text_field,
            },
        )

        with queries_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            queries_rows = list(reader)

        self._validate_fields(
            queries_rows,
            dataset_label="queries CSV",
            required_fields={
                "query_id_field": query_id_field,
                "query_text_field": query_text_field,
                "relevant_id_field": relevant_id_field,
            },
        )

        return self._write(
            corpus_rows=corpus_rows,
            queries_rows=queries_rows,
            output_dir=Path(output_dir),
            corpus_id_field=corpus_id_field,
            corpus_text_field=corpus_text_field,
            corpus_doc_id_field=corpus_doc_id_field,
            query_id_field=query_id_field,
            query_text_field=query_text_field,
            relevant_id_field=relevant_id_field,
        )

    def import_from_jsonl(
        self,
        corpus_path: Path,
        queries_path: Path,
        output_dir: Path,
        corpus_id_field: str,
        corpus_text_field: str,
        query_id_field: str,
        query_text_field: str,
        relevant_id_field: str,
        corpus_doc_id_field: str | None = None,
    ) -> Path:
        """Import from two JSONL files (corpus + queries) and write BEIR format."""
        corpus_path = Path(corpus_path)
        queries_path = Path(queries_path)

        corpus_rows = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._validate_fields(
            corpus_rows,
            dataset_label="corpus JSONL",
            required_fields={
                "corpus_id_field": corpus_id_field,
                "corpus_text_field": corpus_text_field,
            },
        )
        queries_rows = [
            json.loads(line)
            for line in queries_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._validate_fields(
            queries_rows,
            dataset_label="queries JSONL",
            required_fields={
                "query_id_field": query_id_field,
                "query_text_field": query_text_field,
                "relevant_id_field": relevant_id_field,
            },
        )

        return self._write(
            corpus_rows=corpus_rows,
            queries_rows=queries_rows,
            output_dir=Path(output_dir),
            corpus_id_field=corpus_id_field,
            corpus_text_field=corpus_text_field,
            corpus_doc_id_field=corpus_doc_id_field,
            query_id_field=query_id_field,
            query_text_field=query_text_field,
            relevant_id_field=relevant_id_field,
        )

    def import_from_hf(
        self,
        dataset_name: str,
        output_dir: Path,
        context_field: str,
        question_field: str,
        split: str = "train",
        id_field: str | None = None,
    ) -> Path:
        """Import from a Hugging Face dataset. Requires: pip install retrieval-evaluator[datasets]"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace import requires the datasets package. "
                'Install with: pip install "retrieval-evaluator[datasets]"'
            )

        dataset = load_dataset(dataset_name, split=split)
        corpus_rows: list[dict[str, Any]] = []
        queries_rows: list[dict[str, Any]] = []
        seen_contexts: dict[str, str] = {}

        for i, row in enumerate(dataset):
            ctx = row[context_field]
            if ctx not in seen_contexts:
                cid = row[id_field] if id_field and id_field in row else f"c{len(seen_contexts)}"
                seen_contexts[ctx] = str(cid)
                corpus_rows.append({"cid": str(cid), "body": ctx, "parent": str(cid)})

            qid = f"q{i}"
            queries_rows.append({
                "qid": qid,
                "question": row[question_field],
                "answer_chunk": seen_contexts[ctx],
            })

        return self._write(
            corpus_rows=corpus_rows,
            queries_rows=queries_rows,
            output_dir=Path(output_dir),
            corpus_id_field="cid",
            corpus_text_field="body",
            corpus_doc_id_field=None,
            query_id_field="qid",
            query_text_field="question",
            relevant_id_field="answer_chunk",
        )

    def _write(
        self,
        corpus_rows: list[dict],
        queries_rows: list[dict],
        output_dir: Path,
        corpus_id_field: str,
        corpus_text_field: str,
        corpus_doc_id_field: str | None,
        query_id_field: str,
        query_text_field: str,
        relevant_id_field: str,
    ) -> Path:
        from retrieval_evaluator.dataset.beir import write_beir_raw

        corpus = []
        for row in corpus_rows:
            meta: dict[str, Any] = {}
            if corpus_doc_id_field and corpus_doc_id_field in row:
                meta["doc_id"] = row[corpus_doc_id_field]
            corpus.append({"_id": str(row[corpus_id_field]), "text": row[corpus_text_field], "metadata": meta})

        query_text_by_id: dict[str, Any] = {}
        for row in queries_rows:
            qid = str(row[query_id_field])
            query_text = row[query_text_field]
            if qid in query_text_by_id and query_text_by_id[qid] != query_text:
                raise ValueError(f"Query ID {qid!r} has conflicting query text")
            query_text_by_id.setdefault(qid, query_text)
        queries = [
            {"_id": qid, "text": query_text}
            for qid, query_text in query_text_by_id.items()
        ]

        qrels: dict[str, dict[str, int]] = {}
        for row in queries_rows:
            qid = str(row[query_id_field])
            cid = str(row[relevant_id_field])
            qrels.setdefault(qid, {})[cid] = 1

        write_beir_raw(corpus, queries, qrels, output_dir)
        return output_dir
