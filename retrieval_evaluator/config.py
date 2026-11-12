import copy
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict = {
    "settings": {
        "run_details_directory": "run_details",
        "evaluation_results_file": "eval_runs.csv",
        "evaluation_dataset_dir": "eval_data",
        "raw_chunks_cache_dir": "raw_data",
        "embedders_config_path": "config/embedders.yml",
        "max_batch_size": 2048,
        "index_backend": "numpy",
    },
    "llm": {
        "model": "openai:gpt-4o",
        "max_attempts": 5,
        "timeout_s": 60,
        "retry_base_delay_s": 1.0,
        "retry_max_delay_s": 30.0,
        "retry_jitter_s": 0.5,
        "retry_on_invalid_response": True,
    },
    "retrieval_evaluation": {
        "dataset": {
            "document_selection_embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "n_docs_per_language": 30,
            "query_generation_system_prompt": (
                "You are given a passage of text. Your task is to write a question for the passage "
                "that is provided to you. The questions you generate will be used to evaluate a "
                'retrieval system, so make sure to generate a question that a real user might ask. '
                'In addition to that, return the language of the passage (include only 1 language). '
                'Respond in JSON format: {"question": question, "language": language_abbreviation}.'
            ),
            "user_prompt_template": (
                "Write ONE realistic user question that can be answered completely by the passage\n"
                "below - and only by that passage:\n\n'{passage}'"
            ),
            "source": {
                "type": "jsonl",
                "weaviate": {
                    "extract_properties": ["chunk_id", "language", "title", "content"],
                },
                "qdrant": {
                    "url": "http://localhost:6333",
                    "api_key": "",
                    "batch_size": 100,
                    "payload_fields": None,
                },
                "jsonl": {
                    "root_dir": "raw_data/source_collections",
                    "filename_template": "{collection}.jsonl",
                },
                "callable": {
                    "path": "my_project.connectors:load_chunks",
                    "kwargs": {},
                },
            },
            "field_mapping": {
                "id": "id",
                "text": "content",
                "doc_id": "title",
                "chunk_id": "chunk_id",
                "language": "language",
                "chunk_filter_field": None,
                "chunk_filter_value": None,
            },
            "max_kmeans_iters": 100,
            "sleep_s": 1,
            "default_seed": 42,
            "chunk_limit": 200,
        },
        "pipeline": {
            "micro_chunk_tokens": 512,
            "default_alphas": [1.0],
            "default_k_values": [1, 3, 5, 10],
        },
    },
    "database": {
        "weaviate": {
            "url": "http://localhost:8080/",
            "key": "",
            "batch_size": 100,
        }
    },
}


def _search_paths() -> list[Path]:
    paths: list[Path] = []
    config_override = os.environ.get("RETRIEVAL_EVALUATOR_CONFIG")
    if config_override:
        paths.append(Path(config_override).expanduser())
    paths.extend(
        [
            Path("config/retrieval-evaluator.yaml"),
            Path.home() / ".config/retrieval-evaluator/retrieval-evaluator.yaml",
            Path("/etc/retrieval-evaluator/retrieval-evaluator.yaml"),
        ]
    )
    return paths


# Config subtrees whose keys are user-defined rather than part of the schema.
_FREE_FORM_PATHS = {
    ("retrieval_evaluation", "dataset", "source", "callable", "kwargs"),
}

# Valid optional keys that have no entry in DEFAULT_CONFIG because "missing"
# and "set to a default value" behave differently for them.
_EXTRA_KNOWN_PATHS = {
    ("retrieval_evaluation", "dataset", "source", "weaviate", "batch_size"),
    ("retrieval_evaluation", "dataset", "source", "qdrant", "timeout"),
    ("database", "weaviate", "grpc_port"),
}


def find_unknown_keys(user_cfg: dict, schema: dict = DEFAULT_CONFIG, _path: tuple = ()) -> list[str]:
    """Return dotted paths of keys in user_cfg that DEFAULT_CONFIG does not know about."""
    unknown: list[str] = []
    for key, value in user_cfg.items():
        path = (*_path, key)
        if key not in schema:
            if path not in _EXTRA_KNOWN_PATHS:
                unknown.append(".".join(path))
            continue
        if path in _FREE_FORM_PATHS:
            continue
        if isinstance(value, dict) and isinstance(schema[key], dict):
            unknown.extend(find_unknown_keys(value, schema[key], path))
    return unknown


def _warn_unknown_keys(user_cfg: dict, source_path: Path) -> None:
    unknown = find_unknown_keys(user_cfg)
    if unknown:
        logger.warning(
            "Unknown config key(s) in %s: %s. They are ignored — check for typos "
            "(see config/retrieval-evaluator.example.yaml for the full reference).",
            source_path,
            ", ".join(unknown),
        )


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested_value(cfg: dict, path: tuple[str, ...], value) -> None:
    cursor = cfg
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _apply_env_overrides(cfg: dict) -> dict:
    env_paths: dict[str, tuple[str, ...]] = {
        "RETRIEVAL_EVALUATOR_WEAVIATE_URL": ("database", "weaviate", "url"),
        "RETRIEVAL_EVALUATOR_WEAVIATE_KEY": ("database", "weaviate", "key"),
        "RETRIEVAL_EVALUATOR_QDRANT_URL": (
            "retrieval_evaluation",
            "dataset",
            "source",
            "qdrant",
            "url",
        ),
        "RETRIEVAL_EVALUATOR_QDRANT_API_KEY": (
            "retrieval_evaluation",
            "dataset",
            "source",
            "qdrant",
            "api_key",
        ),
    }
    for env_name, path in env_paths.items():
        value = os.environ.get(env_name)
        if value is not None:
            _set_nested_value(cfg, path, value)
    return cfg


def get_evaluation_dataset_dir(override: str | Path | None = None) -> Path:
    """Resolve an input dataset, leaving configured output paths unchanged."""
    if override is not None:
        return Path(override)
    dataset_dir = Path(get_config()["settings"]["evaluation_dataset_dir"])
    if dataset_dir == Path(DEFAULT_CONFIG["settings"]["evaluation_dataset_dir"]) and not dataset_dir.exists():
        bundled = Path(__file__).resolve().parent / "data"
        if bundled.exists():
            return bundled
    return dataset_dir


def _load() -> dict:
    global _loaded_config_path
    _loaded_config_path = None
    search_paths = _search_paths()
    explicit_path = os.environ.get("RETRIEVAL_EVALUATOR_CONFIG")
    if explicit_path and not Path(explicit_path).expanduser().is_file():
        raise FileNotFoundError(f"Configured RETRIEVAL_EVALUATOR_CONFIG file not found: {explicit_path}")
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for path in search_paths:
        if path.exists():
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
            _warn_unknown_keys(loaded, path)
            configured = _apply_env_overrides(_deep_merge(merged, loaded))
            _loaded_config_path = path.resolve()
            return configured
    return _apply_env_overrides(merged)


config: dict | None = None
_loaded_config_path: Path | None = None


def get_config(*, reload: bool = False) -> dict:
    global config
    if reload or config is None:
        config = _load()
    return config


def get_config_path() -> Path | None:
    """Return the file used by the loaded config, or None for built-in defaults."""
    get_config()
    return _loaded_config_path


def reset_config() -> None:
    global config, _loaded_config_path
    config = None
    _loaded_config_path = None
