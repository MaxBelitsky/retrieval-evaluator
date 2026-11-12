from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from retrieval_evaluator.dataset.beir import ensure_beir_meta, load_beir_dataset
from retrieval_evaluator.evaluation.chunk_merging import merge_chunks
from retrieval_evaluator.evaluation.metrics import compute_ranking_metrics
from retrieval_evaluator.evaluation.run_id import create_run_id


def test_merge_chunks_handles_single_doc():
    df = pd.DataFrame([
        {"cid": "a", "text": "one", "metadata": {"doc_id": "t1", "chunk_id": 0}},
        {"cid": "b", "text": "two", "metadata": {"doc_id": "t1", "chunk_id": 1}},
    ])

    class DummyEmbedder:
        max_tokens = 10

        def tokenize(self, text):
            # Return a numpy array with shape (1, n_tokens) to match real Embedder
            tokens = np.zeros((1, len(text.split())), dtype=np.int64)
            return tokens

    merged, mapping = merge_chunks(df, DummyEmbedder())
    assert isinstance(merged, pd.DataFrame)
    assert isinstance(mapping, dict)


class _WordCountEmbedder:
    """Counts whitespace tokens, mirroring the real Embedder's tokenize() shape."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def tokenize(self, text):
        return np.zeros((1, len(text.split())), dtype=np.int64)


def test_merge_chunks_orders_string_chunk_ids_numerically():
    # JSON exports often carry positions as strings; "10" must sort after "2".
    df = pd.DataFrame([
        {"cid": f"c{i}", "text": f"chunk{i} pad pad", "metadata": {"doc_id": "d1", "chunk_id": str(i)}}
        for i in [1, 2, 10]
    ])

    merged, _ = merge_chunks(df, _WordCountEmbedder(max_tokens=100))

    assert len(merged) == 1
    assert merged.iloc[0]["micro_ids"] == ["c1", "c2", "c10"]


def test_merge_chunks_macro_metadata_comes_from_first_buffered_chunk():
    # 'a' is 6 tokens, 'b' is 5: with max_tokens=8 the buffer flushes when 'b'
    # arrives, so each chunk becomes its own macro and must keep its own metadata.
    df = pd.DataFrame([
        {"cid": "a", "text": "one two three four five six", "metadata": {"doc_id": "d1", "chunk_id": 1}},
        {"cid": "b", "text": "seven eight nine ten eleven", "metadata": {"doc_id": "d1", "chunk_id": 2}},
    ])

    merged, _ = merge_chunks(df, _WordCountEmbedder(max_tokens=8))

    metadata_by_micros = {tuple(row["micro_ids"]): row["metadata"] for _, row in merged.iterrows()}
    assert metadata_by_micros[("a",)]["chunk_id"] == 1
    assert metadata_by_micros[("b",)]["chunk_id"] == 2


def test_embedder_cache_distinguishes_query_and_passage_prompts():
    mock = MagicMock()
    mock.encode.side_effect = [
        np.ones((1, 4), dtype=np.float32),
        np.full((1, 4), 2.0, dtype=np.float32),
    ]
    mock.tokenize.return_value = {"input_ids": np.zeros((1, 4), dtype=np.int64)}

    with patch("sentence_transformers.SentenceTransformer", return_value=mock):
        from retrieval_evaluator.evaluation.embedder import Embedder

        embedder = Embedder(
            {
                "provider": "hf",
                "model": "mock",
                "query_prompt": "query: ",
                "passage_prompt": "passage: ",
            },
            device="cpu",
        )

        query_vecs = embedder.encode(["same text"], batch_size=1, text_type="query")
        passage_vecs = embedder.encode(["same text"], batch_size=1, text_type="passage")
        cached_query_vecs = embedder.encode(["same text"], batch_size=1, text_type="query")

    assert mock.encode.call_count == 2
    assert np.array_equal(query_vecs, cached_query_vecs)
    assert not np.array_equal(query_vecs, passage_vecs)


def test_determine_batch_size_calls_forward_with_features_dict():
    class FakeSTModel:
        """Mimics sentence-transformers >= 3: forward takes the features dict positionally."""

        def __init__(self):
            self.forward_calls = 0

        def tokenize(self, texts):
            return {"input_ids": np.zeros((len(texts), 4), dtype=np.int64)}

        def forward(self, input, **kwargs):  # noqa: A002 — matches the ST signature
            assert isinstance(input, dict)
            self.forward_calls += 1
            return input

    fake = FakeSTModel()
    with patch("sentence_transformers.SentenceTransformer", return_value=fake):
        from retrieval_evaluator.evaluation.embedder import Embedder

        embedder = Embedder(
            {"provider": "hf", "model": "mock", "max_tokens": 16},
            device="cpu",
            max_batch_size=2,
        )

    batch_size = embedder.determine_batch_size(["short text", "another text"])

    assert batch_size >= 1
    assert fake.forward_calls > 0


def test_compute_ranking_metrics_hits():
    rankings = {"q1": ["a", "b", "c"], "q2": ["d", "e", "f"]}
    qrels = {"q1": {"b": 1}, "q2": {"z": 1}}
    metrics = compute_ranking_metrics(rankings, qrels, macro2micro={}, k_values=[1, 2, 3])
    assert set(metrics).issuperset({"R@1", "R@2", "R@3", "MRR@3"})


def test_compute_ranking_metrics_counts_missing_rankings_as_misses():
    metrics = compute_ranking_metrics(
        rankings={"q1": ["a"]},
        qrels={"q1": {"a": 1}, "q2": {"b": 1}},
        macro2micro={},
        k_values=[1],
    )

    assert metrics["R@1"] == 0.5
    assert metrics["Recall@1"] == 0.5
    assert metrics["MRR@1"] == 0.5


def test_compute_ranking_metrics_counts_merged_chunk_at_retrieved_rank():
    rankings = {"q1": ["macro_ab", "c"]}
    qrels = {"q1": {"a": 1, "b": 1}}
    macro2micro = {"macro_ab": ["a", "b"]}

    metrics = compute_ranking_metrics(rankings, qrels, macro2micro=macro2micro, k_values=[1, 2])

    assert metrics["R@1"] == 1.0
    assert metrics["Recall@1"] == 1.0
    assert metrics["R_all@1"] == 1.0
    assert metrics["MRR@2"] == 1.0


def test_compute_ranking_metrics_ignores_non_positive_qrels():
    rankings = {
        "q1": ["chunk-zero", "chunk-hit"],
        "q2": ["chunk-negative"],
    }
    qrels = {
        "q1": {"chunk-zero": 0, "chunk-hit": 1},
        "q2": {"chunk-negative": -1},
    }

    metrics = compute_ranking_metrics(rankings, qrels, macro2micro={}, k_values=[1, 2])

    assert metrics["R@1"] == 0.0
    assert metrics["R@2"] == 1.0
    assert metrics["Recall@1"] == 0.0
    assert metrics["Recall@2"] == 1.0
    assert metrics["MRR@2"] == 0.5


def test_compute_ranking_metrics_caps_mrr_at_advertised_cutoff():
    metrics = compute_ranking_metrics(
        rankings={"q1": ["miss-1", "miss-2", "relevant"]},
        qrels={"q1": {"relevant": 1}},
        macro2micro={},
        k_values=[1, 2],
    )

    assert metrics["MRR@2"] == 0.0


def test_load_beir_dataset(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id":"c1","text":"t","metadata":{"doc_id":"d","chunk_id":0}}\n'
    )
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"what"}\n')
    (tmp_path / "qrels.tsv").write_text("q1\t0\tc1\t1\n")
    c_df, q_df, qrels = load_beir_dataset(tmp_path)
    assert list(c_df["cid"]) == ["c1"]
    assert list(q_df["qid"]) == ["q1"]
    assert qrels == {"q1": {"c1": 1}}


def test_load_beir_dataset_derives_meta_without_mutating_dataset(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id":"c1","title":"Doc 1","text":"t"}\n'
    )
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"what"}\n')
    (tmp_path / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tc1\t1\n")

    c_df, q_df, qrels = load_beir_dataset(tmp_path)
    meta = ensure_beir_meta(tmp_path)

    assert not (tmp_path / "meta.json").exists()
    assert c_df.loc[0, "metadata"]["title"] == "Doc 1"
    assert list(q_df["qid"]) == ["q1"]
    assert qrels == {"q1": {"c1": 1}}
    assert meta["n_documents"] == 1
    assert meta["n_chunks"] == 1
    assert meta["n_queries"] == 1


def test_load_beir_dataset_accepts_space_delimited_qrels(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id":"c1","title":"Doc 1","text":"t"}\n'
    )
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"what"}\n')
    (tmp_path / "qrels.tsv").write_text("q1    0    c1    1\n")

    _, q_df, qrels = load_beir_dataset(tmp_path)

    assert list(q_df["qid"]) == ["q1"]
    assert qrels == {"q1": {"c1": 1}}


def test_create_run_id_does_not_generate_meta(tmp_path):
    collection_dir = tmp_path / "support_docs"
    collection_dir.mkdir()
    (collection_dir / "corpus.jsonl").write_text(
        '{"_id":"c1","title":"Doc 1","text":"t"}\n'
    )
    (collection_dir / "queries.jsonl").write_text('{"_id":"q1","text":"what"}\n')
    (collection_dir / "qrels.tsv").write_text("q1\tc1\t1\n")

    run_id = create_run_id(
        "support_docs",
        {"provider": "hf", "model": "mock", "max_tokens": 512},
        1.0,
        dataset_dir=tmp_path,
    )

    assert isinstance(run_id, str)
    assert not (collection_dir / "meta.json").exists()


def test_evaluate_embedder_on_collection_handles_standard_beir_without_metadata(tmp_path):
    collection_dir = tmp_path / "support_docs"
    collection_dir.mkdir()
    (collection_dir / "corpus.jsonl").write_text(
        '{"_id":"doc1","title":"Password Reset Guide","text":"Reset your password from the login page."}\n'
        '{"_id":"doc2","title":"Billing FAQ","text":"Update your payment method in billing settings."}\n'
    )
    (collection_dir / "queries.jsonl").write_text(
        '{"_id":"q1","text":"How do I reset my password?"}\n'
    )
    (collection_dir / "qrels.tsv").write_text("q1\tdoc1\t1\n")

    def _make_mock_st(dim=16):
        mock = MagicMock()
        mock.encode.side_effect = lambda texts, **kw: np.random.default_rng(0).random(
            (len(texts), dim)
        ).astype(np.float32)
        mock.tokenize.return_value = {"input_ids": np.zeros((1, 16), dtype=np.int64)}
        return mock

    class _FakeVectorIndex:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def batch_insert(self, ids, vectors, texts):
            del vectors, texts
            self._ids = list(ids)

        def query(self, vector, k, alpha=1.0, query=None):
            del vector, alpha, query
            return self._ids[:k]

        def supports_hybrid(self) -> bool:
            return True

    with patch("sentence_transformers.SentenceTransformer", return_value=_make_mock_st()):
        from retrieval_evaluator.evaluation.embedder import Embedder
        from retrieval_evaluator.evaluation.embedders_evaluation import evaluate_embedder_on_collection

        embedder = Embedder({"provider": "hf", "model": "mock", "max_tokens": 512}, device="cpu")

        with patch(
            "retrieval_evaluator.evaluation.embedders_evaluation.get_vector_index",
            return_value=_FakeVectorIndex(),
        ):
            metrics = evaluate_embedder_on_collection(
                collection_dir=collection_dir,
                embedder=embedder,
                batch_size=2,
                run_id="support-docs",
                k_values=[1, 5],
                alpha=1.0,
                artifacts_output_dir=tmp_path,
            )

    assert metrics["R@1"] == 1.0
    assert not (collection_dir / "meta.json").exists()
    assert (tmp_path / "support-docs" / "corpus_meta.jsonl").exists()


def test_bm25_evaluation_skips_passage_and_query_embeddings(tmp_path):
    collection_dir = tmp_path / "support_docs"
    collection_dir.mkdir()
    (collection_dir / "corpus.jsonl").write_text('{"_id":"c1","text":"reset password"}\n')
    (collection_dir / "queries.jsonl").write_text('{"_id":"q1","text":"password"}\n')
    (collection_dir / "qrels.tsv").write_text("q1\tc1\t1\n")

    embedder = MagicMock()
    embedder.cfg = {"provider": "openai"}
    embedder.max_tokens = 512
    embedder.supports_tokenization = False
    embedder.encode.side_effect = AssertionError("BM25 must not call the embedding provider")

    class FakeLexicalIndex:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def supports_hybrid(self):
            return True

        def batch_insert(self, ids, vectors, texts):
            assert ids == ["c1"]
            assert vectors is None
            assert texts == ["reset password"]

        def query(self, vector, k, alpha=1.0, query=None):
            assert vector is None
            assert k == 1
            assert alpha == 0.0
            assert query == "password"
            return ["c1"]

    from retrieval_evaluator.evaluation.embedders_evaluation import evaluate_embedder_on_collection

    with patch(
        "retrieval_evaluator.evaluation.embedders_evaluation.get_vector_index",
        return_value=FakeLexicalIndex(),
    ):
        metrics = evaluate_embedder_on_collection(
            collection_dir=collection_dir,
            embedder=embedder,
            batch_size=2,
            run_id="bm25",
            k_values=[1],
            alpha=0.0,
            artifacts_output_dir=tmp_path,
            micro_chunk_tokens=512,
            index_backend="weaviate",
        )

    assert metrics["R@1"] == 1.0
    embedder.encode.assert_not_called()


def test_encode_raises_for_unknown_text_type():
    mock = MagicMock()
    mock.encode.return_value = np.zeros((1, 4), dtype=np.float32)
    mock.tokenize.return_value = {"input_ids": np.zeros((1, 4), dtype=np.int64)}

    with patch("sentence_transformers.SentenceTransformer", return_value=mock):
        from retrieval_evaluator.evaluation.embedder import Embedder

        embedder = Embedder({"provider": "hf", "model": "mock"}, device="cpu")

        with pytest.raises(ValueError, match="Unknown text_type"):
            embedder.encode(["some text"], batch_size=1, text_type="not_a_real_type")


def test_evaluate_embedders_raises_if_any_run_fails(tmp_path):
    collection_dir = tmp_path / "support_docs"
    collection_dir.mkdir()
    (collection_dir / "corpus.jsonl").write_text('{"_id":"c1","text":"t"}\n')
    (collection_dir / "queries.jsonl").write_text('{"_id":"q1","text":"what"}\n')
    (collection_dir / "qrels.tsv").write_text("q1\tc1\t1\n")

    class DummyEmbedder:
        def __init__(self, cfg, device=None, max_batch_size=None):
            del device, max_batch_size
            self.cfg = cfg
            self.max_tokens = 512

        @property
        def identity_config(self):
            return {**self.cfg, "max_tokens": self.max_tokens}

    with patch("retrieval_evaluator.evaluation.embedders_evaluation.Embedder", DummyEmbedder):
        from retrieval_evaluator.evaluation.embedders_evaluation import EvaluationRunError, evaluate_embedders

        with patch(
            "retrieval_evaluator.evaluation.embedders_evaluation.evaluate_embedder_on_collection",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(EvaluationRunError, match="1 evaluation run\\(s\\) failed"):
                evaluate_embedders(
                    collections=["support_docs"],
                    embedder_configs={"mock-embedder": {"provider": "hf", "model": "mock"}},
                    dataset_dir=tmp_path,
                    results_file=tmp_path / "results.csv",
                    artifacts_output_dir=tmp_path / "artifacts",
                )


def test_create_run_id_is_stable_for_write_beir_raw_dataset(tmp_path):
    from retrieval_evaluator.dataset.beir import write_beir_raw
    from retrieval_evaluator.evaluation.run_id import create_run_id

    write_beir_raw(
        corpus=[{"_id": "c1", "text": "hello world", "metadata": {}}],
        queries=[{"_id": "q1", "text": "what?"}],
        qrels={"q1": {"c1": 1}},
        output_dir=tmp_path / "col",
    )
    cfg = {"provider": "hf", "model": "test-model"}
    id_1 = create_run_id("col", cfg, 1.0, dataset_dir=tmp_path)
    id_2 = create_run_id("col", cfg, 1.0, dataset_dir=tmp_path)

    assert id_1 == id_2, "run_id must be stable across calls for the same dataset"
    assert "None" not in id_1, "run_id must not contain the string 'None'"
