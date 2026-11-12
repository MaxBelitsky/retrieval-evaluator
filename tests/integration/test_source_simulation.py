"""Source export contracts using a real local store or simulated HTTP responses."""

import json
import os
import socket
from contextlib import closing
from importlib.metadata import version
from uuid import NAMESPACE_DNS, uuid5

import httpx
import pytest

from retrieval_evaluator.config import get_config
from retrieval_evaluator.dataset.chunk_sources import QdrantChunkSource, WeaviateChunkSource, create_chunk_source


def test_qdrant_local_export_pagination_cache_and_reload(tmp_path):
    from qdrant_client import QdrantClient, models

    get_config()["settings"]["raw_chunks_cache_dir"] = str(tmp_path / "cache")
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection("support_docs", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        client.upsert("support_docs", points=[
            models.PointStruct(id=i, vector=[1.0, 0.0], payload={"content": f"Passage {i}", "title": "Doc", "private": "omit"})
            for i in range(3)
        ])
        source = QdrantChunkSource(
            url="http://qdrant.invalid", client=client, batch_size=1, payload_fields=["content", "title"],
        )
        exported = source.load_chunks("support_docs")
        assert {row["id"] for row in exported} == {"0", "1", "2"}
        assert all(set(row) == {"id", "content", "title"} for row in exported)
        client.set_payload("support_docs", payload={"content": "Updated"}, points=[0])
        assert source.load_chunks("support_docs") == exported
        refreshed = source.load_chunks("support_docs", force_reload=True)
        assert len(refreshed) == 3
        assert next(row for row in refreshed if row["id"] == "0")["content"] == "Updated"


@pytest.mark.parametrize("fail_second_page", [False, True])
def test_qdrant_remote_sdk_export_does_not_cache_partial_results(simulated_http, tmp_path, fail_second_page):
    offsets = []

    def respond(request):
        assert request.headers["api-key"] == "offline-test-key"
        if request.method == "GET" and request.url.path == "/":
            return httpx.Response(200, json={"title": "qdrant", "version": version("qdrant-client")})
        assert request.method == "POST"
        assert request.url.path == "/collections/support_docs/points/scroll"
        body = json.loads(request.content)
        assert body["limit"] == 1 and body["with_vector"] is False
        assert set(body["with_payload"]) == {"content", "title", "chunk_id", "language"}
        offset = body.get("offset")
        offsets.append(offset)
        if fail_second_page and offset == 1:
            return httpx.Response(503, json={"status": {"error": "simulated outage"}, "time": 0.0})
        i = offset or 0
        return httpx.Response(200, json={
            "result": {
                "points": [{"id": i, "payload": {"content": f"Passage {i}", "title": "Doc"}}],
                "next_page_offset": i + 1 if i < 2 else None,
            },
            "status": "ok", "time": 0.0,
        })

    simulated_http(respond)
    cfg = get_config()
    cfg["settings"]["raw_chunks_cache_dir"] = str(tmp_path / "cache")
    dataset_cfg = cfg["retrieval_evaluation"]["dataset"]
    dataset_cfg["source"]["type"] = "qdrant"
    dataset_cfg["source"]["qdrant"].update(url="https://qdrant.invalid", api_key="offline-test-key", batch_size=1)
    source = create_chunk_source(dataset_cfg)
    try:
        if fail_second_page:
            from qdrant_client.http.exceptions import UnexpectedResponse

            with pytest.raises(UnexpectedResponse):
                source.load_chunks("support_docs")
            assert offsets == [None, 1]
            assert not list((tmp_path / "cache").glob("*.json"))
        else:
            chunks = source.load_chunks("support_docs")
            assert [row["id"] for row in chunks] == ["0", "1", "2"]
            assert offsets == [None, 1, 2]
            assert source.load_chunks("support_docs") == chunks
            assert offsets == [None, 1, 2]  # The second export is served by the local cache.
    finally:
        if source._client is not None:
            source._client.close()


@pytest.mark.skipif(
    os.environ.get("RETRIEVAL_EVALUATOR_TEST_WEAVIATE") != "1",
    reason="Opt in to a temporary Weaviate service with RETRIEVAL_EVALUATOR_TEST_WEAVIATE=1",
)
def test_weaviate_source_export_against_temporary_service(tmp_path):
    import weaviate
    from weaviate.classes.config import DataType, Property
    from weaviate.embedded import EmbeddedOptions

    from retrieval_evaluator.database.weaviate_db import Weaviate, _self_provided_vector_kwargs

    # Reserve distinct unused ports before starting an isolated embedded service.
    with socket.socket() as http_socket, socket.socket() as grpc_socket:
        http_socket.bind(("127.0.0.1", 0))
        grpc_socket.bind(("127.0.0.1", 0))
        http_port = http_socket.getsockname()[1]
        grpc_port = grpc_socket.getsockname()[1]
    cfg = get_config()
    cfg["settings"]["raw_chunks_cache_dir"] = str(tmp_path / "cache")
    cfg["database"]["weaviate"].update(url=f"http://127.0.0.1:{http_port}", grpc_port=grpc_port)
    service = weaviate.WeaviateClient(embedded_options=EmbeddedOptions(
        persistence_data_path=str(tmp_path / "weaviate"), port=http_port, grpc_port=grpc_port,
        additional_env_vars={"LOG_LEVEL": "fatal", "ENABLE_MODULES": ""},
    ))
    database = None
    try:
        service.connect()
        collection = service.collections.create(
            "SupportDocs", properties=[Property(name="content", data_type=DataType.TEXT), Property(name="title", data_type=DataType.TEXT)],
            **_self_provided_vector_kwargs(),
        )
        ids = [str(uuid5(NAMESPACE_DNS, f"offline-passage-{i}")) for i in range(3)]
        for i, cid in enumerate(ids):
            collection.data.insert(uuid=cid, properties={"content": f"Passage {i}", "title": "Doc"})
        database = Weaviate.from_config()
        source = WeaviateChunkSource(extract_properties=["content", "title"], batch_size=1, database=database)
        exported = source.load_chunks("SupportDocs")
        assert {row["id"] for row in exported} == set(ids)
        assert all(set(row) == {"id", "content", "title"} for row in exported)
        collection.data.update(uuid=ids[0], properties={"content": "Updated"})
        assert source.load_chunks("SupportDocs") == exported
        refreshed = source.load_chunks("SupportDocs", force_reload=True)
        assert len(refreshed) == 3
        assert next(row for row in refreshed if row["id"] == ids[0])["content"] == "Updated"
    finally:
        if database is not None:
            database.client.close()
        service.close()
