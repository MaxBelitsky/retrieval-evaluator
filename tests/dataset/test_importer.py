import csv
import json

import pytest

from retrieval_evaluator.dataset.importer import DatasetImporter


@pytest.fixture
def corpus_csv(tmp_path):
    path = tmp_path / "corpus.csv"
    rows = [
        {"cid": "c1", "body": "The quick brown fox.", "parent": "d1"},
        {"cid": "c2", "body": "Jumps over the lazy dog.", "parent": "d1"},
        {"cid": "c3", "body": "All that glitters is not gold.", "parent": "d2"},
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cid", "body", "parent"])
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def queries_csv(tmp_path):
    path = tmp_path / "queries.csv"
    rows = [
        {"qid": "q1", "question": "What does the fox do?", "answer_chunk": "c1"},
        {"qid": "q2", "question": "What is not gold?", "answer_chunk": "c3"},
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "question", "answer_chunk"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_import_csv_creates_beir_files(tmp_path, corpus_csv, queries_csv):
    importer = DatasetImporter()
    out = importer.import_from_csv(
        corpus_path=corpus_csv,
        queries_path=queries_csv,
        output_dir=tmp_path / "out",
        corpus_id_field="cid",
        corpus_text_field="body",
        query_id_field="qid",
        query_text_field="question",
        relevant_id_field="answer_chunk",
    )

    assert out == tmp_path / "out"
    assert (out / "corpus.jsonl").exists()
    assert (out / "queries.jsonl").exists()
    assert (out / "qrels.tsv").exists()
    assert (out / "meta.json").exists()


def test_import_csv_corpus_content(tmp_path, corpus_csv, queries_csv):
    importer = DatasetImporter()
    out = importer.import_from_csv(
        corpus_path=corpus_csv,
        queries_path=queries_csv,
        output_dir=tmp_path / "out",
        corpus_id_field="cid",
        corpus_text_field="body",
        query_id_field="qid",
        query_text_field="question",
        relevant_id_field="answer_chunk",
    )

    corpus_rows = [json.loads(line) for line in (out / "corpus.jsonl").read_text().splitlines() if line]
    assert len(corpus_rows) == 3
    assert corpus_rows[0]["_id"] == "c1"
    assert corpus_rows[0]["text"] == "The quick brown fox."


def test_import_csv_qrels_content(tmp_path, corpus_csv, queries_csv):
    importer = DatasetImporter()
    out = importer.import_from_csv(
        corpus_path=corpus_csv,
        queries_path=queries_csv,
        output_dir=tmp_path / "out",
        corpus_id_field="cid",
        corpus_text_field="body",
        query_id_field="qid",
        query_text_field="question",
        relevant_id_field="answer_chunk",
    )

    tsv = (out / "qrels.tsv").read_text()
    assert "q1\t0\tc1\t1" in tsv
    assert "q2\t0\tc3\t1" in tsv


def test_import_csv_supports_multiple_relevant_chunks_per_query(tmp_path, corpus_csv):
    queries_path = tmp_path / "queries.csv"
    queries_path.write_text(
        "qid,question,answer_chunk\n"
        "q1,What does the fox do?,c1\n"
        "q1,What does the fox do?,c2\n"
    )

    out = DatasetImporter().import_from_csv(
        corpus_path=corpus_csv,
        queries_path=queries_path,
        output_dir=tmp_path / "out",
        corpus_id_field="cid",
        corpus_text_field="body",
        query_id_field="qid",
        query_text_field="question",
        relevant_id_field="answer_chunk",
    )

    query_rows = [json.loads(line) for line in (out / "queries.jsonl").read_text().splitlines()]
    assert query_rows == [{"_id": "q1", "text": "What does the fox do?"}]
    assert (out / "qrels.tsv").read_text().splitlines() == ["q1\t0\tc1\t1", "q1\t0\tc2\t1"]


def test_import_rejects_conflicting_text_for_one_query_id(tmp_path, corpus_csv):
    queries_path = tmp_path / "queries.csv"
    queries_path.write_text(
        "qid,question,answer_chunk\n"
        "q1,What does the fox do?,c1\n"
        "q1,What does the dog do?,c2\n"
    )

    with pytest.raises(ValueError, match="conflicting query text"):
        DatasetImporter().import_from_csv(
            corpus_path=corpus_csv,
            queries_path=queries_path,
            output_dir=tmp_path / "out",
            corpus_id_field="cid",
            corpus_text_field="body",
            query_id_field="qid",
            query_text_field="question",
            relevant_id_field="answer_chunk",
        )


def test_import_jsonl(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps({"cid": "c1", "body": "Text.", "parent": "d1"}) + "\n"
        + json.dumps({"cid": "c2", "body": "More text.", "parent": "d1"}) + "\n"
    )
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        json.dumps({"qid": "q1", "question": "What?", "answer_chunk": "c1"}) + "\n"
    )

    importer = DatasetImporter()
    out = importer.import_from_jsonl(
        corpus_path=corpus_path,
        queries_path=queries_path,
        output_dir=tmp_path / "out",
        corpus_id_field="cid",
        corpus_text_field="body",
        query_id_field="qid",
        query_text_field="question",
        relevant_id_field="answer_chunk",
    )

    corpus_rows = [json.loads(line) for line in (out / "corpus.jsonl").read_text().splitlines() if line]
    assert len(corpus_rows) == 2
    assert (out / "qrels.tsv").exists()


def test_import_missing_corpus_field_raises(tmp_path, queries_csv):
    bad_corpus = tmp_path / "bad.csv"
    bad_corpus.write_text("wrong_field,body,parent\nc1,text,d1\n")

    importer = DatasetImporter()
    with pytest.raises(ValueError, match="cid"):
        importer.import_from_csv(
            corpus_path=bad_corpus,
            queries_path=queries_csv,
            output_dir=tmp_path / "out",
            corpus_id_field="cid",
            corpus_text_field="body",
            query_id_field="qid",
            query_text_field="question",
            relevant_id_field="answer_chunk",
        )


def test_import_missing_query_field_raises_for_csv(tmp_path, corpus_csv):
    bad_queries = tmp_path / "bad_queries.csv"
    bad_queries.write_text("qid,wrong_field\nq1,text\n")

    importer = DatasetImporter()
    with pytest.raises(ValueError, match="query_text_field 'question' not found in queries CSV"):
        importer.import_from_csv(
            corpus_path=corpus_csv,
            queries_path=bad_queries,
            output_dir=tmp_path / "out",
            corpus_id_field="cid",
            corpus_text_field="body",
            query_id_field="qid",
            query_text_field="question",
            relevant_id_field="answer_chunk",
        )


def test_import_missing_query_field_raises_for_jsonl(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(json.dumps({"cid": "c1", "body": "Text."}) + "\n")
    bad_queries_path = tmp_path / "queries.jsonl"
    bad_queries_path.write_text(json.dumps({"qid": "q1", "question": "What?"}) + "\n")

    importer = DatasetImporter()
    with pytest.raises(ValueError, match="relevant_id_field 'answer_chunk' not found in queries JSONL"):
        importer.import_from_jsonl(
            corpus_path=corpus_path,
            queries_path=bad_queries_path,
            output_dir=tmp_path / "out",
            corpus_id_field="cid",
            corpus_text_field="body",
            query_id_field="qid",
            query_text_field="question",
            relevant_id_field="answer_chunk",
        )
