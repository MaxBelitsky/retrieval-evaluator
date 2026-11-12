import copy
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from retrieval_evaluator.config import DEFAULT_CONFIG
from retrieval_evaluator.database.weaviate_db import Weaviate
from retrieval_evaluator.dataset import dataset_builder as builder
from retrieval_evaluator.dataset import query_generation as query_generation_module
from retrieval_evaluator.dataset.beir import load_beir_dataset
from retrieval_evaluator.dataset.chunk_sources import (
    CallableChunkSource,
    InMemoryChunkSource,
    JsonlChunkSource,
    QdrantChunkSource,
    WeaviateChunkSource,
    create_chunk_source,
)
from retrieval_evaluator.dataset.document_selection import (
    embed_chunks,
    pool_document_embeddings,
    select_diverse_documents,
    sort_chunks,
)
from retrieval_evaluator.dataset.query_generation import call_llm, generate_queries_for_chunks, parse_llm_json_response


class DummyEmbedder:
    def encode(self, texts, batch_size=64, **kwargs):
        return np.array([[len(t), len(t) + 1] for t in texts], dtype=np.float32)


def test_group_chunks():
    # sort_chunks maps source-specific fields to canonical chunk fields.
    raw = [
        {"id": "c2", "title": "Doc1", "chunk_id": 2, "content": "foo"},
        {"id": "c1", "title": "Doc1", "chunk_id": 1, "content": "bar"},
        {"id": "c3", "title": "Doc2", "chunk_id": 0, "content": "baz"},
    ]
    df = sort_chunks(
        raw,
        field_mapping={
            "id": "id",
            "text": "content",
            "doc_id": "title",
            "chunk_id": "chunk_id",
            "language": "language",
        },
    )
    assert list(df["id"]) == ["c1", "c2", "c3"]
    assert list(df["doc_id"]) == ["Doc1", "Doc1", "Doc2"]
    assert list(df["chunk_id"]) == [1, 2, 0]


def test_embed_chunks():
    df = pd.DataFrame({"text": ["a", "bc"]})
    arr = embed_chunks(df, DummyEmbedder())
    assert arr.shape == (2, 2)
    assert (arr[1] == [2, 3]).all()


def test_pool_document_embeddings():
    df = pd.DataFrame({"doc_id": ["d1", "d1", "d2"], "text": ["x", "y", "z"]})
    chunk_vecs = np.array([[1, 1], [3, 3], [2, 2]], dtype=np.float32)
    pooled, doc_ids = pool_document_embeddings(df, chunk_vecs)
    assert pooled.shape == (2, 2)
    assert set(doc_ids) == {"d1", "d2"}
    assert np.allclose(pooled[0], [2, 2])


def test_select_diverse_documents_few():
    doc_vecs = np.array([[1, 2]], dtype=np.float32)
    selected = select_diverse_documents(doc_vecs, ["d"], n_documents=5, random_state=1)
    assert selected == {"d"}


def test_parse_llm_json_response_plain():
    js = '{"question": "foo?", "language": "en"}'
    res = parse_llm_json_response(js)
    assert res["question"] == "foo?"
    assert res["language"] == "en"


def test_parse_llm_json_response_fenced():
    text = '```json\n{"question": "bar?", "language": "de"}\n```'
    res = parse_llm_json_response(text)
    assert res["language"] == "de"


def test_generate_queries_for_chunks_keeps_dataframe_alignment_when_limited():
    df = pd.DataFrame(
        {
            "id": ["c1", "c2", "c3"],
            "text": ["first", "second", "third"],
        }
    )
    call_count = 0

    def fake_generate(_prompt, _system_prompt):
        nonlocal call_count
        call_count += 1
        return '{"question": "q?", "language": "en"}'

    queries, qids, langs = generate_queries_for_chunks(
        df,
        generate_fn=fake_generate,
        sleep_s=0,
        chunk_limit=2,
    )

    assert call_count == 2
    assert len(queries) == len(df)
    assert len(qids) == len(df)
    assert len(langs) == len(df)
    assert queries[:2] == ["q?", "q?"]
    assert queries[2] is None
    assert qids[2] is None
    assert langs[2] is None


def test_generate_queries_for_chunks_retries_rate_limits(monkeypatch):
    df = pd.DataFrame({"id": ["c1"], "text": ["first"]})
    calls = {"count": 0}
    delays = []

    class FakeRateLimitError(Exception):
        status_code = 429

    def fake_generate(_prompt, _system_prompt):
        calls["count"] += 1
        if calls["count"] < 3:
            raise FakeRateLimitError("Too many requests")
        return '{"question": "q?", "language": "en"}'

    from retrieval_evaluator.utils import retry as retry_module

    monkeypatch.setattr(query_generation_module.time, "sleep", delays.append)
    monkeypatch.setattr(retry_module.random, "uniform", lambda _a, _b: 0.25)

    queries, qids, langs = generate_queries_for_chunks(
        df,
        generate_fn=fake_generate,
        sleep_s=0,
        chunk_limit=1,
        max_attempts=3,
        retry_base_delay_s=1.0,
        retry_max_delay_s=30.0,
        retry_jitter_s=0.25,
    )

    assert calls["count"] == 3
    assert queries == ["q?"]
    assert qids[0] is not None
    assert langs == ["en"]
    assert delays == [1.25, 2.25]


def test_generate_queries_fails_fast_when_provider_package_missing():
    # A missing langchain provider package must abort the run on the first chunk
    # instead of logging one buried warning per chunk and failing at the end.
    df = pd.DataFrame({"id": ["c1", "c2"], "text": ["first", "second"]})
    calls = {"count": 0}

    def fake_generate(_prompt, _system_prompt):
        calls["count"] += 1
        raise ImportError("requires the langchain-openai package")

    with pytest.raises(ImportError, match="langchain-openai"):
        generate_queries_for_chunks(df, generate_fn=fake_generate, sleep_s=0, chunk_limit=2)
    assert calls["count"] == 1


def test_generate_queries_for_chunks_does_not_retry_non_retryable_errors():
    df = pd.DataFrame({"id": ["c1"], "text": ["first"]})
    calls = {"count": 0}

    class FakeBadRequestError(Exception):
        status_code = 400

    def fake_generate(_prompt, _system_prompt):
        calls["count"] += 1
        raise FakeBadRequestError("Bad request")

    queries, qids, langs = generate_queries_for_chunks(
        df,
        generate_fn=fake_generate,
        sleep_s=0,
        chunk_limit=1,
        max_attempts=3,
        retry_base_delay_s=0,
        retry_max_delay_s=0,
        retry_jitter_s=0,
    )

    assert calls["count"] == 1
    assert queries == [None]
    assert qids == [None]
    assert langs == [None]


def test_generate_queries_for_chunks_retries_invalid_model_output():
    df = pd.DataFrame({"id": ["c1"], "text": ["first"]})
    outputs = iter(
        [
            '{"language": "en"}',
            '{"question": "q?", "language": "en"}',
        ]
    )

    queries, qids, langs = generate_queries_for_chunks(
        df,
        generate_fn=lambda _prompt, _system_prompt: next(outputs),
        sleep_s=0,
        chunk_limit=1,
        max_attempts=2,
        retry_base_delay_s=0,
        retry_max_delay_s=0,
        retry_jitter_s=0,
    )

    assert queries == ["q?"]
    assert qids[0] is not None
    assert langs == ["en"]


def test_generate_queries_for_chunks_uses_row_identity_for_query_ids():
    df = pd.DataFrame(
        {
            "id": ["c1", "c2"],
            "text": ["duplicate text", "duplicate text"],
        }
    )

    queries, qids, langs = generate_queries_for_chunks(
        df,
        generate_fn=lambda _prompt, _system_prompt: '{"question": "q?", "language": "en"}',
        sleep_s=0,
        chunk_limit=2,
    )

    assert queries == ["q?", "q?"]
    assert langs == ["en", "en"]
    assert len(set(qids)) == 2


def test_call_llm_uses_provider_native_timeout(monkeypatch):
    captured = {}

    class APITimeoutError(Exception):
        pass

    class FakeLLM:
        def invoke(self, _messages):
            raise APITimeoutError("request timed out")

    def fake_get_llm(model=None, timeout_s=None):
        captured.update({"model": model, "timeout_s": timeout_s})
        return FakeLLM()

    monkeypatch.setattr(query_generation_module, "_get_llm", fake_get_llm)

    with pytest.raises(RuntimeError, match="failed after 1 attempts"):
        call_llm(
            "prompt",
            "system",
            max_retries=1,
            timeout_s=0.001,
            retry_base_delay_s=0,
            retry_max_delay_s=0,
            retry_jitter_s=0,
        )
    assert captured == {"model": None, "timeout_s": 0.001}


def test_get_llm_disables_provider_retries_and_sets_timeout(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_init_chat_model(model, **kwargs):
        captured.update({"model": model, **kwargs})
        return sentinel

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)
    query_generation_module._llm_cache.clear()

    result = query_generation_module._get_llm("openai:test-model", timeout_s=12.5)

    assert result is sentinel
    assert captured == {
        "model": "openai:test-model",
        "max_retries": 0,
        "timeout": 12.5,
    }


def test_generate_queries_retries_non_object_json():
    outputs = iter(["[]", '{"question": "valid?", "language": "en"}'])

    queries, _, languages = generate_queries_for_chunks(
        pd.DataFrame({"id": ["c1"], "text": ["first"]}),
        generate_fn=lambda _prompt, _system: next(outputs),
        sleep_s=0,
        max_attempts=2,
        retry_base_delay_s=0,
        retry_max_delay_s=0,
        retry_jitter_s=0,
    )

    assert queries == ["valid?"]
    assert languages == ["en"]


def test_create_chunk_source_jsonl(tmp_path):
    source_dir = tmp_path / "chunks"
    source_dir.mkdir()
    chunk_path = source_dir / "support_docs.jsonl"
    chunk_path.write_text(
        '{"id":"c1","document_id":"doc-1","position":0,"lang":"en","content":"Chunk one"}\n',
        encoding="utf-8",
    )

    dataset_cfg = {
        "source": {
            "type": "jsonl",
            "jsonl": {
                "root_dir": str(source_dir),
                "filename_template": "{collection}.jsonl",
            },
        }
    }

    source = create_chunk_source(dataset_cfg)

    assert isinstance(source, JsonlChunkSource)
    assert source.load_chunks("support_docs") == [
        {
            "id": "c1",
            "document_id": "doc-1",
            "position": 0,
            "lang": "en",
            "content": "Chunk one",
        }
    ]


def test_jsonl_chunk_source_reports_invalid_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "ok"}\nnot-json\n', encoding="utf-8")
    source = JsonlChunkSource(tmp_path)

    with pytest.raises(ValueError, match="line 2"):
        source.load_chunks("broken")


def test_create_chunk_source_weaviate_uses_single_configured_client(monkeypatch):
    created = {}

    class StubDatabase:
        def load_all_chunks(self, collection_name, properties, batch_size=100, last_id=None, force_reload=False):
            created["call"] = {
                "collection_name": collection_name,
                "properties": properties,
                "batch_size": batch_size,
                "last_id": last_id,
                "force_reload": force_reload,
            }
            return [{"id": "c1"}]

    def fake_from_config():
        created["from_config_calls"] = created.get("from_config_calls", 0) + 1
        return StubDatabase()

    monkeypatch.setattr("retrieval_evaluator.database.weaviate_db.Weaviate.from_config", fake_from_config)

    dataset_cfg = {
        "source": {
            "type": "weaviate",
            "weaviate": {
                "extract_properties": ["content", "title"],
                "batch_size": 42,
            },
        }
    }

    source = create_chunk_source(dataset_cfg)

    assert isinstance(source, WeaviateChunkSource)
    assert source.load_chunks("support_docs", force_reload=True) == [{"id": "c1"}]
    assert created["from_config_calls"] == 1
    assert created["call"] == {
        "collection_name": "support_docs",
        "properties": ["content", "title"],
        "batch_size": 42,
        "last_id": None,
        "force_reload": True,
    }


def test_create_chunk_source_qdrant_scrolls_and_maps_payload(monkeypatch, tmp_path):
    created = {"calls": []}

    class FakePoint:
        def __init__(self, point_id, payload):
            self.id = point_id
            self.payload = payload

    class FakeQdrantClient:
        def __init__(self, url, api_key=None, timeout=None):
            created["init"] = {
                "url": url,
                "api_key": api_key,
                "timeout": timeout,
            }

        def scroll(self, **kwargs):
            created["calls"].append(kwargs)
            if len(created["calls"]) == 1:
                return ([FakePoint("p1", {"content": "Chunk one", "title": "Doc1"})], "next-page")
            return ([FakePoint("p2", {"content": "Chunk two", "title": "Doc2"})], None)

    monkeypatch.setattr(
        "retrieval_evaluator.dataset.chunk_sources._load_qdrant_client_class",
        lambda: FakeQdrantClient,
    )
    monkeypatch.setattr(
        "retrieval_evaluator.config.config",
        {"settings": {"raw_chunks_cache_dir": str(tmp_path)}},
    )

    dataset_cfg = {
        "field_mapping": {
            "id": "id",
            "text": "content",
            "doc_id": "title",
            "chunk_id": "chunk_id",
            "language": "language",
            "chunk_filter_field": None,
            "chunk_filter_value": None,
        },
        "source": {
            "type": "qdrant",
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "secret",
                "batch_size": 1,
            },
        },
    }

    source = create_chunk_source(dataset_cfg)

    assert isinstance(source, QdrantChunkSource)
    assert source.load_chunks("support_docs", force_reload=True) == [
        {"id": "p1", "content": "Chunk one", "title": "Doc1"},
        {"id": "p2", "content": "Chunk two", "title": "Doc2"},
    ]
    assert created["init"] == {
        "url": "http://localhost:6333",
        "api_key": "secret",
        "timeout": None,
    }
    assert created["calls"] == [
        {
            "collection_name": "support_docs",
            "limit": 1,
            "offset": None,
            "with_payload": ["content", "title", "chunk_id", "language"],
            "with_vectors": False,
        },
        {
            "collection_name": "support_docs",
            "limit": 1,
            "offset": "next-page",
            "with_payload": ["content", "title", "chunk_id", "language"],
            "with_vectors": False,
        },
    ]


def test_qdrant_source_rejects_repeated_pagination_offset(tmp_path, monkeypatch):
    point = type("Point", (), {"id": "p1", "payload": {"content": "text"}})()
    client = MagicMock()
    client.scroll.return_value = ([point], "same-offset")
    monkeypatch.setattr(
        "retrieval_evaluator.dataset.chunk_sources.get_config",
        lambda: {"settings": {"raw_chunks_cache_dir": str(tmp_path)}},
    )
    source = QdrantChunkSource(url="http://localhost:6333", client=client)

    with pytest.raises(RuntimeError, match="repeated offset"):
        source.load_chunks("support_docs", force_reload=True)


def test_source_cache_names_include_unsanitized_collection_identity():
    from retrieval_evaluator.utils.cache import source_cache_filename

    first = source_cache_filename("qdrant", "a/b", {"url": "http://example"})
    second = source_cache_filename("qdrant", "a?b", {"url": "http://example"})

    assert first != second


def test_create_chunk_source_callable_function(tmp_path, monkeypatch):
    module_path = tmp_path / "sample_connector.py"
    module_path.write_text(
        """
def load_chunks(collection_name, force_reload=False, prefix=""):
    return [{"id": f"{prefix}{collection_name}", "force_reload": force_reload}]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    dataset_cfg = {
        "source": {
            "type": "callable",
            "callable": {
                "path": "sample_connector:load_chunks",
                "kwargs": {"prefix": "chunk-"},
            },
        }
    }

    source = create_chunk_source(dataset_cfg)

    assert isinstance(source, CallableChunkSource)
    assert source.load_chunks("support_docs", force_reload=True) == [
        {"id": "chunk-support_docs", "force_reload": True}
    ]


def test_create_chunk_source_callable_class(tmp_path, monkeypatch):
    module_path = tmp_path / "sample_class_connector.py"
    module_path.write_text(
        """
class MemoryChunkSource:
    def __init__(self, prefix=""):
        self.prefix = prefix

    def load_chunks(self, collection_name, *, force_reload=False):
        return [{"id": f"{self.prefix}{collection_name}", "force_reload": force_reload}]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    dataset_cfg = {
        "source": {
            "type": "callable",
            "callable": {
                "path": "sample_class_connector:MemoryChunkSource",
                "kwargs": {"prefix": "class-"},
            },
        }
    }

    source = create_chunk_source(dataset_cfg)

    assert isinstance(source, CallableChunkSource)
    assert source.load_chunks("support_docs", force_reload=False) == [
        {"id": "class-support_docs", "force_reload": False}
    ]


def test_build_eval_set_for_collection_uses_injected_chunk_source(monkeypatch, tmp_path):
    class StubChunkSource:
        def load_chunks(self, collection_name, *, force_reload=False):
            assert collection_name == "support_docs"
            assert force_reload is True
            return [
                {
                    "id": "c1",
                    "title": "Doc1",
                    "chunk_id": 0,
                    "language": "en",
                    "content": "Chunk one",
                }
            ]

    def fake_generate_queries(df, **kwargs):
        assert list(df["id"]) == ["c1"]
        return ["What is chunk one?"], ["q1"], ["en"]

    monkeypatch.setattr(builder, "generate_queries_for_chunks", fake_generate_queries)

    builder.build_eval_set_for_collection(
        "support_docs",
        DummyEmbedder(),
        chunk_source=StubChunkSource(),
        settings={"evaluation_dataset_dir": str(tmp_path)},
        sleep_s=0,
        force_reload=True,
    )

    out_dir = tmp_path / "support_docs"
    assert (out_dir / "corpus.jsonl").exists()
    assert (out_dir / "queries.jsonl").exists()
    assert (out_dir / "qrels.tsv").exists()

    corpus = [json.loads(line) for line in (out_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = [json.loads(line) for line in (out_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()]

    assert corpus[0]["_id"] == "c1"
    assert queries[0]["_id"] == "q1"


@pytest.mark.parametrize("mode, n_queries", [("limited", 1), ("failed", 2), ("all_failed", 0)])
def test_query_generation_does_not_remove_retrieval_candidates(monkeypatch, tmp_path, mode, n_queries):
    chunks = [
        {"id": "c1", "title": "Doc1", "content": "Chunk one", "language": "en"},
        {"id": "c2", "title": "Doc2", "content": "Chunk two", "language": "de"},
        {"id": "c3", "title": "Doc3", "content": "Chunk three", "language": "es"},
    ]

    def generate(prompt, *_args, **_kwargs):
        if mode == "all_failed" or "Chunk two" in prompt:
            raise ValueError("Cannot generate a question for this passage")
        language = "es" if "Chunk three" in prompt else "en"
        return json.dumps({"question": f"A question in {language}?", "language": language})

    monkeypatch.setattr(query_generation_module, "_get_llm", lambda **kwargs: None)
    monkeypatch.setattr(query_generation_module, "call_llm", generate)

    def build():
        builder.build_eval_set_for_collection(
            "support_docs",
            DummyEmbedder(),
            chunk_source=InMemoryChunkSource(chunks),
            settings={"evaluation_dataset_dir": str(tmp_path)},
            chunk_limit=1 if mode == "limited" else 3,
            sleep_s=0,
        )

    if not n_queries:
        with pytest.raises(RuntimeError, match="no queries were generated successfully"):
            build()
        assert not (tmp_path / "support_docs").exists()
        return

    build()
    corpus, queries, qrels = load_beir_dataset(tmp_path / "support_docs")
    assert set(corpus["cid"]) == {"c1", "c2", "c3"}
    assert {row.cid: row.metadata["language"] for row in corpus.itertuples()} == {
        "c1": "en", "c2": "de", "c3": "es",
    }
    assert len(queries) == len(qrels) == n_queries
    assert {cid for scores in qrels.values() for cid in scores} == (
        {"c1"} if mode == "limited" else {"c1", "c3"}
    )


def test_build_eval_set_filters_excluded_rows_before_value_validation(monkeypatch, tmp_path):
    class StubChunkSource:
        def load_chunks(self, collection_name, *, force_reload=False):
            del collection_name, force_reload
            return [
                {
                    "id": "c1",
                    "title": "Doc1",
                    "chunk_id": 0,
                    "content": "Useful text",
                    "chunk_type": "text",
                },
                {
                    "id": "c1",
                    "title": None,
                    "chunk_id": 1,
                    "content": None,
                    "chunk_type": "table",
                },
            ]

    monkeypatch.setattr(
        builder,
        "generate_queries_for_chunks",
        lambda df, **kwargs: (["A question?"], ["q1"], ["en"]),
    )
    dataset_cfg = copy.deepcopy(DEFAULT_CONFIG["retrieval_evaluation"]["dataset"])
    dataset_cfg["field_mapping"].update(
        {"chunk_filter_field": "chunk_type", "chunk_filter_value": "text"}
    )

    builder.build_eval_set_for_collection(
        "support_docs",
        DummyEmbedder(),
        chunk_source=StubChunkSource(),
        dataset_cfg=dataset_cfg,
        settings={"evaluation_dataset_dir": str(tmp_path)},
        sleep_s=0,
    )

    corpus_lines = (tmp_path / "support_docs" / "corpus.jsonl").read_text().splitlines()
    assert len(corpus_lines) == 1


def test_weaviate_load_all_chunks_raises_on_partial_fetch_failure(monkeypatch, tmp_path):
    database = object.__new__(Weaviate)
    database._url = "http://localhost:8080"
    database._key = None
    calls = {"count": 0}

    class FakeObject:
        def __init__(self, object_id, payload):
            self.uuid = object_id
            self.properties = payload

    def fake_get_batch_with_cursor(collection_name, batch_size, properties, cursor=None):
        del collection_name, batch_size, properties, cursor
        calls["count"] += 1
        if calls["count"] == 1:
            return [FakeObject("id-1", {"content": "one"})]
        raise RuntimeError("network error")

    monkeypatch.setattr(
        "retrieval_evaluator.database.weaviate_db.get_config",
        lambda: {"settings": {"raw_chunks_cache_dir": str(tmp_path)}},
    )
    database.get_batch_with_cursor = fake_get_batch_with_cursor

    with pytest.raises(RuntimeError, match="Weaviate chunk export failed"):
        database.load_all_chunks("support_docs", ["content"], batch_size=1, force_reload=True)


def test_weaviate_load_all_chunks_propagates_keyboard_interrupt(monkeypatch, tmp_path):
    database = object.__new__(Weaviate)
    database._url = "http://localhost:8080"
    database._key = None
    database.get_batch_with_cursor = MagicMock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(
        "retrieval_evaluator.database.weaviate_db.get_config",
        lambda: {"settings": {"raw_chunks_cache_dir": str(tmp_path)}},
    )

    with pytest.raises(KeyboardInterrupt):
        database.load_all_chunks("support_docs", ["content"], force_reload=True)


def test_in_memory_chunk_source_returns_chunks():
    from retrieval_evaluator.dataset.chunk_sources import InMemoryChunkSource

    chunks = [{"id": "c1", "text": "hello", "doc_id": "d1"}]
    source = InMemoryChunkSource(chunks)

    result = source.load_chunks("any_collection")
    assert result == chunks

    result2 = source.load_chunks("other_collection")
    assert result2 == chunks  # collection name is ignored


def test_in_memory_chunk_source_returns_copy():
    from retrieval_evaluator.dataset.chunk_sources import InMemoryChunkSource

    chunks = [{"id": "c1", "text": "hello", "doc_id": "d1", "metadata": {"page": 1}}]
    source = InMemoryChunkSource(chunks)

    result = source.load_chunks("col")
    result.append({"id": "extra"})
    result[0]["metadata"]["page"] = 2
    assert len(source.load_chunks("col")) == 1  # original unchanged
    assert source.load_chunks("col")[0]["metadata"]["page"] == 1
