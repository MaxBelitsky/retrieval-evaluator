from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
VALID_PROVIDERS = {"hf", "openai", "cohere"}


def _load_repo_embedders() -> dict:
    with (ROOT_DIR / "config" / "embedders.yml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_repo_embedders_yml_entries_are_valid():
    configs = _load_repo_embedders()
    assert isinstance(configs, dict) and configs, "embedders.yml must define at least one embedder"
    for name, cfg in configs.items():
        assert isinstance(cfg, dict), f"{name} must map to a config dict"
        assert cfg.get("provider") in VALID_PROVIDERS, f"{name} has invalid provider {cfg.get('provider')!r}"
        assert isinstance(cfg.get("model"), str) and cfg["model"], f"{name} is missing a model id"


def test_repo_embedders_yml_default_is_lightweight():
    configs = _load_repo_embedders()
    # The checked-in default must not make a fresh clone download multi-GB models.
    hf_models = {cfg["model"] for cfg in configs.values() if cfg.get("provider") == "hf"}
    assert hf_models <= {"sentence-transformers/all-MiniLM-L6-v2"}, (
        f"Unexpected heavy default models in config/embedders.yml: {hf_models}"
    )


def test_repo_embedders_yml_matches_starter_template():
    from retrieval_evaluator.templates import STARTER_EMBEDDERS

    assert (ROOT_DIR / "config" / "embedders.yml").read_text(encoding="utf-8") == STARTER_EMBEDDERS


def test_public_metadata_has_no_redundant_local_extra():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    metadata = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]

    assert "hf" in extras
    assert "local" not in extras
