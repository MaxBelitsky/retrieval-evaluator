from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from retrieval_evaluator.config import get_config
from retrieval_evaluator.utils.cache import source_cache_filename
from retrieval_evaluator.utils.json_io import read_json_records, write_json

logger = logging.getLogger(__name__)

@runtime_checkable
class ChunkSource(Protocol):
    def load_chunks(self, collection_name: str, *, force_reload: bool = False) -> list[dict[str, Any]]:
        """Load raw chunk records for a collection."""


def _fields_from_mapping(dataset_cfg: dict) -> list[str]:
    field_mapping = dataset_cfg.get("field_mapping", {})
    ordered_keys = ["text", "doc_id", "chunk_id", "language"]
    fields: list[str] = []

    for key in ordered_keys:
        field_name = field_mapping.get(key)
        if field_name and field_name not in fields:
            fields.append(field_name)

    filter_field = field_mapping.get("chunk_filter_field")
    if filter_field and filter_field not in fields:
        fields.append(filter_field)

    return fields


class WeaviateChunkSource:
    def __init__(
        self,
        extract_properties: Sequence[str],
        batch_size: int = 100,
        database=None,
    ):
        self._extract_properties = list(extract_properties)
        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError("Weaviate batch_size must be positive")
        self._database = database

    def _get_database(self):
        if self._database is None:
            try:
                from retrieval_evaluator.database.weaviate_db import Weaviate  # lazy — only needed for weaviate source
            except ImportError as exc:
                raise ImportError(
                    "Weaviate chunk sources require the weaviate extra. "
                    'Install with: pip install "retrieval-evaluator[weaviate]"'
                ) from exc
            self._database = Weaviate.from_config()
        if self._database is None:
            raise RuntimeError("Configured Weaviate client is not available")
        return self._database

    def load_chunks(self, collection_name: str, *, force_reload: bool = False) -> list[dict[str, Any]]:
        logger.info(f"{collection_name}: loading chunks from Weaviate")
        return self._get_database().load_all_chunks(
            collection_name,
            self._extract_properties,
            batch_size=self._batch_size,
            last_id=None,
            force_reload=force_reload,
        )


class JsonlChunkSource:
    def __init__(self, root_dir: Path, filename_template: str = "{collection}.jsonl"):
        self._root_dir = root_dir.expanduser()
        self._filename_template = filename_template

    def _path_for_collection(self, collection_name: str) -> Path:
        filename = self._filename_template.format(collection=collection_name)
        root = self._root_dir.resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Collection name resolves outside the JSONL source root: {collection_name!r}")
        return path

    def load_chunks(self, collection_name: str, *, force_reload: bool = False) -> list[dict[str, Any]]:
        del force_reload  # local file loading is always fresh
        path = self._path_for_collection(collection_name)
        if not path.exists():
            raise FileNotFoundError(f"JSONL chunk source not found for {collection_name}: {path}")

        logger.info(f"{collection_name}: loading chunks from {path}")
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in chunk source {path} at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object in chunk source {path} at line {line_number}"
                    )
                records.append(record)
        return records


class InMemoryChunkSource:
    """Wraps a pre-loaded list of chunk dicts. Used by the Python API."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self._chunks = copy.deepcopy(chunks)

    def load_chunks(self, collection_name: str, *, force_reload: bool = False) -> list[dict[str, Any]]:
        del collection_name, force_reload
        return copy.deepcopy(self._chunks)


def _load_qdrant_client_class():
    try:
        module = importlib.import_module("qdrant_client")
    except ImportError as exc:
        raise ImportError(
            "Qdrant chunk source requires the optional dependency 'qdrant-client'. "
            'Install with: pip install "retrieval-evaluator[qdrant]"'
        ) from exc

    try:
        return module.QdrantClient
    except AttributeError as exc:
        raise ImportError("Installed qdrant_client package does not expose QdrantClient") from exc


class QdrantChunkSource:
    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        batch_size: int = 100,
        payload_fields: Sequence[str] | None = None,
        client=None,
        timeout: int | None = None,
    ):
        self._url = url
        self._api_key = api_key if api_key else None
        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError("Qdrant batch_size must be positive")
        # True tells the Qdrant client to return all payload fields.
        self._payload_fields: list[str] | bool = (
            list(payload_fields) if payload_fields is not None else True
        )
        self._client = client
        self._timeout = timeout

    def _get_client(self):
        if self._client is None:
            qdrant_client_cls = _load_qdrant_client_class()
            self._client = qdrant_client_cls(
                url=self._url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def _cache_path(self, collection_name: str) -> Path:
        cache_dir = Path(get_config()["settings"]["raw_chunks_cache_dir"]).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / source_cache_filename(
            "qdrant",
            collection_name,
            {
                "url": self._url,
                "credential": hashlib.sha256(self._api_key.encode("utf-8")).hexdigest()
                if self._api_key
                else None,
                "payload_fields": self._payload_fields,
            },
        )

    def load_chunks(self, collection_name: str, *, force_reload: bool = False) -> list[dict[str, Any]]:
        cache_path = self._cache_path(collection_name)
        if cache_path.exists() and not force_reload:
            return read_json_records(cache_path)

        logger.info(f"{collection_name}: loading chunks from Qdrant")
        client = self._get_client()
        all_records: list[dict[str, Any]] = []
        next_offset = None
        seen_offsets: set[str] = set()

        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=self._batch_size,
                offset=next_offset,
                with_payload=self._payload_fields,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                record = dict(point.payload or {})
                record["id"] = str(point.id)
                all_records.append(record)

            if next_offset is None:
                break
            offset_key = repr(next_offset)
            if offset_key in seen_offsets:
                raise RuntimeError(
                    f"Qdrant pagination repeated offset {next_offset!r} for '{collection_name}'"
                )
            seen_offsets.add(offset_key)

        write_json(all_records, cache_path, indent=2)

        return all_records


class CallableChunkSource:
    """
    Resolve a chunk source from a Python import path.

    Supported targets:
    - class implementing `load_chunks(**...)`
    - object implementing `load_chunks(**...)`
    - plain function with signature similar to
      `fn(collection_name, force_reload=False, **kwargs) -> list[dict]`
    """

    def __init__(self, object_path: str, kwargs: dict[str, Any] | None = None):
        self._object_path = object_path
        self._kwargs = kwargs or {}
        self._resolved_target = None

    @staticmethod
    def _load_object(object_path: str):
        if ":" in object_path:
            module_name, object_name = object_path.split(":", 1)
        elif "." in object_path:
            module_name, object_name = object_path.rsplit(".", 1)
        else:
            raise ValueError(
                f"Invalid callable source path '{object_path}'. "
                "Use 'module.submodule:object' or 'module.submodule.object'."
            )

        module = importlib.import_module(module_name)
        try:
            return getattr(module, object_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Object '{object_name}' was not found in module '{module_name}'"
            ) from exc

    def _resolve_target(self):
        if self._resolved_target is not None:
            return self._resolved_target

        loaded = self._load_object(self._object_path)

        if inspect.isclass(loaded):
            loaded = loaded(**self._kwargs)
            if not isinstance(loaded, ChunkSource):
                raise TypeError(
                    f"Callable source class '{self._object_path}' did not produce a ChunkSource instance"
                )
            self._resolved_target = loaded
            return self._resolved_target

        if isinstance(loaded, ChunkSource):
            if self._kwargs:
                raise ValueError(
                    f"Callable source '{self._object_path}' resolved to an object instance; "
                    "remove 'kwargs' or point to a class/function instead."
                )
            self._resolved_target = loaded
            return self._resolved_target

        if callable(loaded):
            self._resolved_target = loaded
            return self._resolved_target

        raise TypeError(
            f"Callable source '{self._object_path}' must resolve to a ChunkSource, "
            "a ChunkSource class, or a callable loader function."
        )

    def load_chunks(self, collection_name: str, *, force_reload: bool = False) -> list[dict[str, Any]]:
        target = self._resolve_target()
        if isinstance(target, ChunkSource):
            return target.load_chunks(collection_name, force_reload=force_reload)

        logger.info(f"{collection_name}: loading chunks via callable source {self._object_path}")
        result = target(collection_name, force_reload=force_reload, **self._kwargs)
        if not isinstance(result, list):
            raise TypeError(
                f"Callable source '{self._object_path}' returned {type(result).__name__}, expected list"
            )
        return result


def _create_weaviate_source(dataset_cfg: dict, source_cfg: dict) -> ChunkSource:
    weaviate_cfg = source_cfg.get("weaviate", {})
    extract_properties = weaviate_cfg.get(
        "extract_properties",
        dataset_cfg.get("extract_properties", []),
    )
    batch_size = int(
        weaviate_cfg.get(
            "batch_size",
            get_config()["database"]["weaviate"].get("batch_size", 100),
        )
    )
    return WeaviateChunkSource(
        extract_properties=extract_properties,
        batch_size=batch_size,
    )


def _create_jsonl_source(dataset_cfg: dict, source_cfg: dict) -> ChunkSource:
    del dataset_cfg
    jsonl_cfg = source_cfg.get("jsonl", {})
    root_dir = Path(
        jsonl_cfg.get("root_dir", get_config()["settings"]["raw_chunks_cache_dir"])
    )
    filename_template = jsonl_cfg.get("filename_template", "{collection}.jsonl")
    return JsonlChunkSource(root_dir=root_dir, filename_template=filename_template)


def _create_qdrant_source(dataset_cfg: dict, source_cfg: dict) -> ChunkSource:
    qdrant_cfg = source_cfg.get("qdrant", {})
    payload_fields = qdrant_cfg.get("payload_fields")
    if payload_fields is None:
        payload_fields = dataset_cfg.get("extract_properties") or _fields_from_mapping(dataset_cfg)
    batch_size = int(qdrant_cfg.get("batch_size", 100))
    timeout = qdrant_cfg.get("timeout")

    return QdrantChunkSource(
        url=qdrant_cfg.get("url", "http://localhost:6333"),
        api_key=qdrant_cfg.get("api_key") or None,
        batch_size=batch_size,
        payload_fields=payload_fields,
        timeout=timeout,
    )


def _create_callable_source(dataset_cfg: dict, source_cfg: dict) -> ChunkSource:
    del dataset_cfg
    callable_cfg = source_cfg.get("callable", {})
    object_path = callable_cfg.get("path")
    if not object_path:
        raise ValueError("Callable chunk source requires retrieval_evaluation.dataset.source.callable.path")

    kwargs = callable_cfg.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise TypeError("Callable chunk source 'kwargs' must be a mapping")

    return CallableChunkSource(object_path=object_path, kwargs=kwargs)


_SOURCE_BUILDERS = {
    "weaviate": _create_weaviate_source,
    "jsonl": _create_jsonl_source,
    "qdrant": _create_qdrant_source,
    "callable": _create_callable_source,
}


def create_chunk_source(dataset_cfg: dict | None = None) -> ChunkSource:
    dataset_cfg = dataset_cfg or get_config()["retrieval_evaluation"]["dataset"]
    source_cfg = dataset_cfg.get("source", {})
    source_type = source_cfg.get("type", "jsonl")
    try:
        builder = _SOURCE_BUILDERS[source_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_SOURCE_BUILDERS))
        raise ValueError(
            f"Unsupported dataset source type '{source_type}'. "
            f"Expected one of: {supported}."
        ) from exc

    return builder(dataset_cfg, source_cfg)
