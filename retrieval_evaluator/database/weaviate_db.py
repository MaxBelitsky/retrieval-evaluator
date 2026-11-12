from __future__ import annotations

import hashlib
import logging
import pathlib
import tempfile
import uuid as uuid_lib
import warnings
from typing import Sequence
from urllib.parse import urlparse

with warnings.catch_warnings():
    # authlib installs its own "always" filter for AuthlibDeprecationWarning at
    # import time, which would outrank any filter added before it. Import it
    # first so the ignore filter below ends up in front.
    try:
        from authlib.deprecate import AuthlibDeprecationWarning

        warnings.filterwarnings("ignore", category=AuthlibDeprecationWarning)
    except ImportError:
        pass
    warnings.filterwarnings("ignore", message=r".*authlib\.jose.*")
    import weaviate
    from weaviate.auth import AuthApiKey
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.query import HybridFusion, MetadataQuery
    from weaviate.embedded import EmbeddedOptions

from retrieval_evaluator.config import get_config
from retrieval_evaluator.database.vector_index import VectorIndex
from retrieval_evaluator.utils.cache import source_cache_filename
from retrieval_evaluator.utils.json_io import read_json_records, write_json

logger = logging.getLogger(__name__)

# Quiet the weaviate-client logger unless debug logging is enabled (covers
# Python-API users, where the CLI's setup_logger never runs).
_debug = logging.root.level <= logging.DEBUG
logging.getLogger("weaviate-client").setLevel(logging.DEBUG if _debug else logging.ERROR)


def _self_provided_vector_kwargs() -> dict:
    """collections.create() kwargs for self-provided vectors across weaviate-client versions."""
    # weaviate-client >= 4.16 deprecates vectorizer_config in favor of vector_config (Dep024).
    if hasattr(Configure, "Vectors"):
        return {"vector_config": Configure.Vectors.self_provided()}
    return {"vectorizer_config": Configure.Vectorizer.none()}


class Weaviate:
    def __init__(self, url: str, key: str | None):
        self._url = url
        self._key = key
        self._client = self._create_client(url, key)
        logger.info(f"Database {url} connection is established")

    @classmethod
    def from_config(cls) -> "Weaviate":
        weaviate_cfg = get_config()["database"]["weaviate"]
        return cls(weaviate_cfg["url"], weaviate_cfg.get("key") or None)

    @property
    def client(self):
        return self._client

    def is_ready(self) -> bool:
        return self._client.is_ready()

    def check_if_collection_exists(self, collection_name: str) -> bool:
        return self._client.collections.exists(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._client.collections.delete(collection_name)
        except Exception as e:
            logger.error(f"Could not delete collection '{collection_name}'.")
            raise e

    def _create_client(self, url: str, key: str | None = None) -> weaviate.WeaviateClient:
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            http_port = parsed.port or (443 if parsed.scheme == "https" else 8080)
            secure = parsed.scheme == "https"
            grpc_port = int(get_config()["database"]["weaviate"].get("grpc_port", 50051))
            auth = AuthApiKey(key) if key else None
            return weaviate.connect_to_custom(
                http_host=host,
                http_port=http_port,
                http_secure=secure,
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=secure,
                auth_credentials=auth,
            )
        except Exception as e:
            logger.error(f"Connection to Weaviate failed, url={url}")
            raise e

    def get_batch_with_cursor(
        self, collection_name: str, batch_size: int, properties: list[str], cursor=None
    ):
        collection = self._client.collections.get(collection_name)
        response = collection.query.fetch_objects(
            limit=batch_size,
            after=cursor,
            return_properties=properties,
        )
        return response.objects

    def load_all_chunks(
        self,
        collection_name: str,
        properties: list[str],
        batch_size: int = 100,
        last_id: str | None = None,
        force_reload: bool = False,
    ) -> list[dict]:
        """Load all chunks for a collection, with local JSON cache."""
        cache_dir = pathlib.Path(get_config()["settings"]["raw_chunks_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / source_cache_filename(
            "weaviate",
            collection_name,
            {
                "url": self._url,
                "credential": hashlib.sha256(self._key.encode("utf-8")).hexdigest()
                if self._key
                else None,
                "properties": sorted(properties),
                "last_id": last_id,
            },
        )

        if path.exists() and not force_reload:
            return read_json_records(path)

        all_data = []
        cursor = last_id
        seen_cursors: set[str] = set()
        try:
            while True:
                objects = self.get_batch_with_cursor(collection_name, batch_size, properties, cursor)
                if not objects:
                    break
                for obj in objects:
                    record = dict(obj.properties)
                    record["id"] = str(obj.uuid)
                    all_data.append(record)
                cursor = str(objects[-1].uuid)
                if cursor in seen_cursors:
                    raise RuntimeError(
                        f"Weaviate pagination repeated cursor {cursor!r} for '{collection_name}'"
                    )
                seen_cursors.add(cursor)
        except Exception as e:
            logger.error(
                "Failed to load chunks from Weaviate collection '%s' after %s records",
                collection_name,
                len(all_data),
            )
            raise RuntimeError(
                f"Weaviate chunk export failed for '{collection_name}' after {len(all_data)} records"
            ) from e

        write_json(all_data, path, indent=2)

        return all_data


class WeaviateVectorIndex(VectorIndex):
    """
    In-memory Weaviate index for retrieval evaluation.
    Use as a context manager: insert with batch_insert(), retrieve with query().
    """
    COLLECTION_NAME = "EvalChunk"

    def __init__(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._client = None
        self._collection = None

    def __enter__(self):
        import logging as _logging
        weaviate_log_level = "debug" if _logging.root.level <= _logging.DEBUG else "fatal"
        self._client = weaviate.WeaviateClient(
            embedded_options=EmbeddedOptions(
                persistence_data_path=self._tmpdir.name,
                additional_env_vars={"LOG_LEVEL": weaviate_log_level},
            )
        )
        self._client.connect()
        if self._client.collections.exists(self.COLLECTION_NAME):
            self._client.collections.delete(self.COLLECTION_NAME)
        self._client.collections.create(
            name=self.COLLECTION_NAME,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="cid", data_type=DataType.TEXT, index_searchable=False),
            ],
            **_self_provided_vector_kwargs(),
        )
        self._collection = self._client.collections.get(self.COLLECTION_NAME)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._client:
            self._client.close()
        self._tmpdir.cleanup()

    @staticmethod
    def _to_uuid(cid: str) -> str:
        return str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, cid))

    def batch_insert(self, ids: Sequence[str], vectors, texts: Sequence[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must contain the same number of items")
        if vectors is not None and len(vectors) != len(ids):
            raise ValueError("ids and vectors must contain the same number of items")
        vectors_to_insert = [None] * len(ids) if vectors is None else vectors
        with self._collection.batch.dynamic() as batch:
            for _id, vec, text in zip(ids, vectors_to_insert, texts):
                object_kwargs = {
                    "properties": {"content": text, "cid": _id},
                    "uuid": self._to_uuid(_id),
                }
                if vec is not None:
                    object_kwargs["vector"] = vec.astype("float32").tolist()
                batch.add_object(**object_kwargs)
        failed_objects = self._collection.batch.failed_objects
        if failed_objects:
            first_error = getattr(failed_objects[0], "message", str(failed_objects[0]))
            raise RuntimeError(
                f"Weaviate failed to index {len(failed_objects)} object(s): {first_error}"
            )

    def query(self, vector, k: int, alpha: float = 1.0, query: str | None = None) -> list[str]:
        if query and alpha == 0.0:
            response = self._collection.query.bm25(
                query=query,
                query_properties=["content"],
                limit=k,
            )
        elif query and alpha != 1.0:
            response = self._collection.query.hybrid(
                query=query,
                alpha=alpha,
                vector=vector.tolist(),
                query_properties=["content"],
                fusion_type=HybridFusion.RELATIVE_SCORE,
                limit=k,
                return_metadata=MetadataQuery(distance=True),
            )
        else:
            response = self._collection.query.near_vector(
                near_vector=vector.tolist(),
                limit=k,
                return_metadata=MetadataQuery(distance=True),
            )
        return [obj.properties["cid"] for obj in response.objects]

    def supports_hybrid(self) -> bool:
        return True
