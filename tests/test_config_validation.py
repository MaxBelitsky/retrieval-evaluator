"""
Config schema validation, and the contract that keeps the config templates in
sync with DEFAULT_CONFIG: the init template and the annotated example may choose
different starter *values*, but every key they use must exist in the schema.
"""

import logging
from pathlib import Path

import yaml

from retrieval_evaluator.config import find_unknown_keys, get_config, reset_config
from retrieval_evaluator.templates import MINIMAL_CONFIG

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_find_unknown_keys_flags_typos():
    cfg = {
        "settings": {"run_details_directoy": "x"},
        "llm": {"model": "openai:gpt-4o"},
    }
    assert find_unknown_keys(cfg) == ["settings.run_details_directoy"]


def test_find_unknown_keys_flags_top_level_typos():
    assert find_unknown_keys({"setings": {}}) == ["setings"]


def test_find_unknown_keys_allows_callable_kwargs():
    cfg = {
        "retrieval_evaluation": {
            "dataset": {"source": {"callable": {"kwargs": {"anything": 1, "nested": {"deep": 2}}}}}
        }
    }
    assert find_unknown_keys(cfg) == []


def test_find_unknown_keys_allows_optional_keys_without_defaults():
    cfg = {
        "retrieval_evaluation": {
            "dataset": {
                "source": {
                    "weaviate": {"batch_size": 50},
                    "qdrant": {"timeout": 30},
                }
            }
        },
        "database": {"weaviate": {"grpc_port": 50051}},
    }
    assert find_unknown_keys(cfg) == []


def test_loading_config_with_unknown_key_warns(tmp_path, monkeypatch, caplog):
    config_path = tmp_path / "retrieval-evaluator.yaml"
    config_path.write_text("settings:\n  evaluation_dataset_dirr: data\n", encoding="utf-8")
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_CONFIG", str(config_path))
    reset_config()

    with caplog.at_level(logging.WARNING, logger="retrieval_evaluator.config"):
        get_config(reload=True)

    assert "settings.evaluation_dataset_dirr" in caplog.text


def test_loading_valid_config_does_not_warn(tmp_path, monkeypatch, caplog):
    config_path = tmp_path / "retrieval-evaluator.yaml"
    config_path.write_text("settings:\n  evaluation_dataset_dir: data\n", encoding="utf-8")
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_CONFIG", str(config_path))
    reset_config()

    with caplog.at_level(logging.WARNING, logger="retrieval_evaluator.config"):
        get_config(reload=True)

    assert "Unknown config key" not in caplog.text


def test_init_template_matches_default_config_schema():
    parsed = yaml.safe_load(MINIMAL_CONFIG)
    assert find_unknown_keys(parsed) == [], (
        "retrieval_evaluator/templates.py MINIMAL_CONFIG uses keys that DEFAULT_CONFIG "
        "does not define — update retrieval_evaluator/config.py or the template"
    )


def test_example_config_matches_default_config_schema():
    example = ROOT_DIR / "config" / "retrieval-evaluator.example.yaml"
    parsed = yaml.safe_load(example.read_text(encoding="utf-8"))
    assert find_unknown_keys(parsed) == [], (
        "config/retrieval-evaluator.example.yaml uses keys that DEFAULT_CONFIG does not "
        "define — update retrieval_evaluator/config.py or the example"
    )
