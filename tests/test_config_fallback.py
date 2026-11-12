from pathlib import Path

import pytest

from retrieval_evaluator.config import get_evaluation_dataset_dir, reset_config


@pytest.fixture(autouse=True)
def _reset_cfg():
    reset_config()
    yield
    reset_config()


def test_dataset_dir_falls_back_to_bundled_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no local eval_data/ here

    from retrieval_evaluator.config import get_config
    cfg = get_config()

    assert cfg["settings"]["evaluation_dataset_dir"] == "eval_data"
    dataset_dir = get_evaluation_dataset_dir()
    assert dataset_dir.exists(), f"Expected bundled data dir to exist, got {dataset_dir}"
    assert (dataset_dir / "EXAMPLE").is_dir(), "Expected EXAMPLE dataset in bundled data"


def test_explicit_missing_input_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_evaluation_dataset_dir("eval_data") == Path("eval_data")


def test_default_config_uses_dependency_free_local_workflows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from retrieval_evaluator.config import get_config

    cfg = get_config()
    assert cfg["settings"]["index_backend"] == "numpy"
    assert cfg["retrieval_evaluation"]["dataset"]["source"]["type"] == "jsonl"
    assert cfg["database"]["weaviate"]["url"] == "http://localhost:8080/"


def test_dataset_dir_local_takes_precedence(tmp_path, monkeypatch):
    (tmp_path / "eval_data" / "EXAMPLE").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    from retrieval_evaluator.config import get_config
    cfg = get_config()

    dataset_dir = Path(cfg["settings"]["evaluation_dataset_dir"])
    assert not dataset_dir.is_absolute(), "Expected relative path (local eval_data/), got absolute bundled path"
    assert (tmp_path / dataset_dir).is_dir(), "Expected eval_data/ to exist under tmp_path"


def test_custom_missing_dataset_dir_does_not_fall_back(tmp_path, monkeypatch):
    config_path = tmp_path / "retrieval-evaluator.yaml"
    config_path.write_text("settings:\n  evaluation_dataset_dir: missing-custom-data\n")
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_CONFIG", str(config_path))
    monkeypatch.chdir(tmp_path)

    from retrieval_evaluator.config import get_config

    configured = Path(get_config()["settings"]["evaluation_dataset_dir"])
    assert configured == Path("missing-custom-data")


def test_missing_explicit_config_file_raises(tmp_path, monkeypatch):
    missing = tmp_path / "missing.yaml"
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_CONFIG", str(missing))

    from retrieval_evaluator.config import get_config

    with pytest.raises(FileNotFoundError, match="RETRIEVAL_EVALUATOR_CONFIG"):
        get_config(reload=True)
