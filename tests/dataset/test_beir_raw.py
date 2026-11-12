def test_write_beir_raw_creates_all_files(tmp_path):
    from retrieval_evaluator.dataset.beir import write_beir_raw

    corpus = [
        {"_id": "c1", "text": "The quick brown fox.", "metadata": {"doc_id": "d1"}},
        {"_id": "c2", "text": "Jumps over the lazy dog.", "metadata": {"doc_id": "d1"}},
    ]
    queries = [
        {"_id": "q1", "text": "What does the fox do?"},
    ]
    qrels = {"q1": {"c1": 1}}

    write_beir_raw(corpus, queries, qrels, tmp_path)

    assert (tmp_path / "corpus.jsonl").exists()
    assert (tmp_path / "queries.jsonl").exists()
    assert (tmp_path / "qrels.tsv").exists()
    assert (tmp_path / "meta.json").exists()

    rows = (tmp_path / "corpus.jsonl").read_text().strip().splitlines()
    assert len(rows) == 2

    tsv_content = (tmp_path / "qrels.tsv").read_text()
    assert "q1\t0\tc1\t1" in tsv_content


def test_write_beir_raw_roundtrips(tmp_path):
    from retrieval_evaluator.dataset.beir import load_beir_dataset, write_beir_raw

    corpus = [{"_id": "c1", "text": "Hello world.", "metadata": {"doc_id": "d1"}}]
    queries = [{"_id": "q1", "text": "What is hello world?"}]
    qrels = {"q1": {"c1": 1}}

    write_beir_raw(corpus, queries, qrels, tmp_path)
    corpus_df, queries_df, loaded_qrels = load_beir_dataset(tmp_path)

    assert corpus_df["cid"].iloc[0] == "c1"
    assert queries_df["text"].iloc[0] == "What is hello world?"
    assert loaded_qrels == {"q1": {"c1": 1}}


def test_write_beir_dataset_n_documents_counts_unique_parent_docs(tmp_path):
    import json

    import pandas as pd

    from retrieval_evaluator.dataset.beir import write_beir_dataset

    # 3 chunks from 2 parent documents
    df = pd.DataFrame([
        {"id": "c1", "text": "chunk 1", "doc_id": "doc-a", "query": "q1?", "qid": "q1"},
        {"id": "c2", "text": "chunk 2", "doc_id": "doc-a", "query": None,  "qid": None},
        {"id": "c3", "text": "chunk 3", "doc_id": "doc-b", "query": "q2?", "qid": "q2"},
    ])
    write_beir_dataset(df, tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())

    assert meta["n_documents"] == 2, "should count unique doc_ids, not chunks"
    assert meta["n_chunks"] == 3


def test_write_beir_raw_uses_consistent_meta_field_names(tmp_path):
    import json

    from retrieval_evaluator.dataset.beir import write_beir_raw

    write_beir_raw(
        corpus=[{"_id": "c1", "text": "text", "metadata": {}}],
        queries=[{"_id": "q1", "text": "question"}],
        qrels={"q1": {"c1": 1}},
        output_dir=tmp_path,
    )
    meta = json.loads((tmp_path / "meta.json").read_text())

    assert "created_utc" in meta, "must use created_utc, not created"
    assert "n_chunks" in meta, "must use n_chunks, not n_corpus"
    assert "created" not in meta
    assert "n_corpus" not in meta


def test_load_beir_dataset_supports_standard_split_qrels(tmp_path):
    from retrieval_evaluator.dataset.beir import load_beir_dataset

    (tmp_path / "corpus.jsonl").write_text('{"_id":"c1","text":"text"}\n')
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"question"}\n')
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\tc1\t1\n"
    )

    _, queries, qrels = load_beir_dataset(tmp_path)

    assert queries["qid"].tolist() == ["q1"]
    assert qrels == {"q1": {"c1": 1}}


def test_load_beir_dataset_uses_requested_non_default_split(tmp_path):
    from retrieval_evaluator.dataset.beir import load_beir_dataset

    (tmp_path / "corpus.jsonl").write_text('{"_id":"c1","text":"text"}\n')
    (tmp_path / "queries.jsonl").write_text(
        '{"_id":"q-dev","text":"dev question"}\n'
        '{"_id":"q-test","text":"test question"}\n'
    )
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "dev.tsv").write_text("q-dev\tc1\t1\n")
    (tmp_path / "qrels" / "test.tsv").write_text("q-test\tc1\t1\n")

    _, queries, qrels = load_beir_dataset(tmp_path, qrels_split="dev")

    assert queries["qid"].tolist() == ["q-dev"]
    assert qrels == {"q-dev": {"c1": 1}}


def test_load_beir_dataset_rejects_unknown_qrel_references(tmp_path):
    import pytest

    from retrieval_evaluator.dataset.beir import load_beir_dataset

    (tmp_path / "corpus.jsonl").write_text('{"_id":"c1","text":"text"}\n')
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"question"}\n')
    (tmp_path / "qrels.tsv").write_text("q1\tmissing\t1\n")

    with pytest.raises(ValueError, match="unknown corpus IDs"):
        load_beir_dataset(tmp_path)


def test_load_beir_dataset_does_not_mistake_qid_value_for_header(tmp_path):
    from retrieval_evaluator.dataset.beir import load_beir_dataset

    (tmp_path / "corpus.jsonl").write_text('{"_id":"c1","text":"text"}\n')
    (tmp_path / "queries.jsonl").write_text('{"_id":"qid","text":"question"}\n')
    (tmp_path / "qrels.tsv").write_text("qid\tc1\t1\n")

    _, queries, qrels = load_beir_dataset(tmp_path)

    assert queries["qid"].tolist() == ["qid"]
    assert qrels == {"qid": {"c1": 1}}


def test_write_beir_dataset_validates_before_creating_output(tmp_path):
    import pandas as pd
    import pytest

    from retrieval_evaluator.dataset.beir import write_beir_dataset

    output = tmp_path / "dataset"
    frame = pd.DataFrame(
        [{"id": "c1", "text": "text", "query": None, "qid": None}]
    )

    with pytest.raises(ValueError, match="without any generated queries"):
        write_beir_dataset(frame, output)

    assert not output.exists()


def test_write_beir_dataset_serializes_timestamp_metadata(tmp_path):
    import json

    import pandas as pd

    from retrieval_evaluator.dataset.beir import write_beir_dataset

    frame = pd.DataFrame(
        [
            {
                "id": "c1",
                "text": "text",
                "query": "question",
                "qid": "q1",
                "ingestion_date": pd.Timestamp("2026-09-03T10:15:00Z"),
            }
        ]
    )

    write_beir_dataset(frame, tmp_path)

    corpus = json.loads((tmp_path / "corpus.jsonl").read_text())
    assert corpus["metadata"]["ingestion_date"] == "2026-09-03T10:15:00+00:00"


def test_write_beir_dataset_preserves_existing_files_on_serialization_error(tmp_path):
    import pandas as pd
    import pytest

    from retrieval_evaluator.dataset.beir import write_beir_dataset

    original = "existing corpus\n"
    (tmp_path / "corpus.jsonl").write_text(original)
    frame = pd.DataFrame(
        [
            {
                "id": "c1",
                "text": "text",
                "query": "question",
                "qid": "q1",
                "content_source": object(),
            }
        ]
    )

    with pytest.raises(TypeError, match="not JSON serializable"):
        write_beir_dataset(frame, tmp_path)

    assert (tmp_path / "corpus.jsonl").read_text() == original
