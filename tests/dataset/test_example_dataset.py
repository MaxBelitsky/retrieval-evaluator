import json
from pathlib import Path

EXAMPLE_DIR = Path("eval_data/EXAMPLE")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_corpus_has_31_passages():
    corpus = _load_jsonl(EXAMPLE_DIR / "corpus.jsonl")
    assert len(corpus) == 31, f"expected 31 passages, got {len(corpus)}"


def test_corpus_micro_chunks_have_shared_doc_id():
    corpus = _load_jsonl(EXAMPLE_DIR / "corpus.jsonl")
    from collections import Counter
    doc_id_counts = Counter(row["metadata"]["doc_id"] for row in corpus)
    three_chunk_topics = [doc_id for doc_id, count in doc_id_counts.items() if count == 3]
    assert len(three_chunk_topics) == 8, (
        f"expected 8 micro-chunked topics (3 chunks each), got {len(three_chunk_topics)}: {three_chunk_topics}"
    )


def test_corpus_single_chunks_have_one_chunk():
    corpus = _load_jsonl(EXAMPLE_DIR / "corpus.jsonl")
    from collections import Counter
    doc_id_counts = Counter(row["metadata"]["doc_id"] for row in corpus)
    single_chunk_topics = [doc_id for doc_id, count in doc_id_counts.items() if count == 1]
    assert len(single_chunk_topics) == 7, (
        f"expected 7 single-chunk topics, got {len(single_chunk_topics)}: {single_chunk_topics}"
    )


def test_corpus_all_passages_are_english():
    corpus = _load_jsonl(EXAMPLE_DIR / "corpus.jsonl")
    non_english = [row["_id"] for row in corpus if row["metadata"].get("language") != "en"]
    assert not non_english, f"non-English corpus passages: {non_english}"


def test_corpus_all_passages_have_required_metadata():
    corpus = _load_jsonl(EXAMPLE_DIR / "corpus.jsonl")
    for row in corpus:
        assert "doc_id" in row["metadata"], f"{row['_id']} missing doc_id in metadata"
        assert "chunk_id" in row["metadata"], f"{row['_id']} missing chunk_id in metadata"
        assert "language" in row["metadata"], f"{row['_id']} missing language in metadata"


def test_queries_has_12_entries():
    queries = _load_jsonl(EXAMPLE_DIR / "queries.jsonl")
    assert len(queries) == 12, f"expected 12 queries, got {len(queries)}"


def test_queries_language_mix():
    queries = _load_jsonl(EXAMPLE_DIR / "queries.jsonl")
    texts = [q["text"] for q in queries]
    german = [t for t in texts if any(w in t for w in ["Wie", "Ich", "mein", "Unternehmen"])]
    spanish = [t for t in texts if any(w in t for w in ["¿Cómo", "¿Puedo", "añadir", "exporto", "descargo"])]
    assert len(german) == 3, f"expected 3 German queries, found {len(german)}: {german}"
    assert len(spanish) == 3, f"expected 3 Spanish queries, found {len(spanish)}: {spanish}"


def test_qrels_has_12_entries():
    qrels_path = EXAMPLE_DIR / "qrels.tsv"
    lines = [line for line in qrels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 12, f"expected 12 qrel lines, got {len(lines)}"


def test_qrels_all_corpus_ids_exist():
    corpus = _load_jsonl(EXAMPLE_DIR / "corpus.jsonl")
    corpus_ids = {row["_id"] for row in corpus}
    qrels_path = EXAMPLE_DIR / "qrels.tsv"
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        corpus_id = parts[2]
        assert corpus_id in corpus_ids, f"qrel references unknown corpus id: {corpus_id}"


def test_qrels_all_query_ids_exist():
    queries = _load_jsonl(EXAMPLE_DIR / "queries.jsonl")
    query_ids = {q["_id"] for q in queries}
    qrels_path = EXAMPLE_DIR / "qrels.tsv"
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        query_id = parts[0]
        assert query_id in query_ids, f"qrel references unknown query id: {query_id}"


def test_bundled_package_dataset_matches_repo_dataset():
    """pip users get retrieval_evaluator/data/EXAMPLE — it must never drift from eval_data/EXAMPLE."""
    bundled_dir = Path("retrieval_evaluator/data/EXAMPLE")
    for filename in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "meta.json"]:
        repo_file = EXAMPLE_DIR / filename
        bundled_file = bundled_dir / filename
        assert bundled_file.exists(), f"missing bundled file: {bundled_file}"
        assert bundled_file.read_text(encoding="utf-8") == repo_file.read_text(encoding="utf-8"), (
            f"{bundled_file} differs from {repo_file} — sync the bundled dataset"
        )
