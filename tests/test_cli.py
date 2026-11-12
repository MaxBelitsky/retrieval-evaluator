import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import retrieval_evaluator.config as config_module
from retrieval_evaluator import cli

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_eval_dashboard_hint_preserves_paths(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "custom config.yaml"
    config_path.write_text("settings: {}\n")
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_CONFIG", str(config_path))
    monkeypatch.setattr(config_module, "config", None)
    monkeypatch.setattr(
        "retrieval_evaluator.evaluation.embedders_evaluation.evaluate_embedders",
        lambda **kwargs: pd.DataFrame([
            {"embedder": "minilm", "collection": "EXAMPLE", "alpha": 1.0, "R@1": 1.0}
        ]),
    )
    options = [
        "--results-file", "my results/results.csv",
        "--artifacts-dir", "my results/artifacts",
        "--dataset-dir", "my datasets",
    ]
    assert cli.main(["eval", "--model", "minilm", *options]) == 0
    command = capsys.readouterr().out.split("Browse them with: ", 1)[1].strip()
    assert shlex.split(command) == [
        "retrieval-evaluator", "dashboard", "--config", str(config_path), *options,
    ]


def test_build_without_output_override_uses_local_directory(monkeypatch, tmp_path):
    import numpy as np

    from retrieval_evaluator.dataset.beir import load_beir_dataset

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RETRIEVAL_EVALUATOR_CONFIG", raising=False)
    monkeypatch.setattr(config_module, "_search_paths", lambda: [])
    monkeypatch.setattr(config_module, "config", None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "new-collection.jsonl").write_text(
        '{"id":"c1","content":"Reset your password in Settings.","document_id":"doc1"}\n'
    )
    # Keep generation offline while exercising the real CLI, source reader and dataset writer.
    monkeypatch.setattr("retrieval_evaluator.dataset.dataset_builder.Embedder", lambda *a, **kw: object())
    monkeypatch.setattr("retrieval_evaluator.dataset.dataset_builder.embed_chunks", lambda *a, **kw: np.ones((1, 2)))
    monkeypatch.setattr(
        "retrieval_evaluator.dataset.dataset_builder.generate_queries_for_chunks",
        lambda *a, **kw: (["How do I reset my password?"], ["q1"], ["en"]),
    )
    assert cli.main([
        "build-dataset", "--collections", "new-collection", "--jsonl-root", str(source),
        "--doc-id-field", "document_id",
    ]) == 0
    corpus, queries, qrels = load_beir_dataset(tmp_path / "eval_data" / "new-collection")
    assert corpus["cid"].tolist() == ["c1"]
    assert queries["qid"].tolist() == ["q1"]
    assert qrels == {"q1": {"c1": 1}}


def test_cli_module_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "retrieval_evaluator.cli", "--help"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "eval" in result.stdout
    assert "build-dataset" in result.stdout
    assert "dashboard" in result.stdout


def test_cli_no_args_prints_help(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "retrieval-evaluator" in out
    assert "build-dataset" in out


def test_cli_subcommand_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["eval", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--collections" in out
    assert "--alpha" in out


def test_build_dataset_help_discloses_dry_run_cache_writes(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["build-dataset", "--help"])

    assert excinfo.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "without calling the LLM or writing an evaluation dataset" in out
    assert "Remote sources may populate the configured local cache" in out


def test_cli_dashboard_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["dashboard", "--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--results-file" in out
    assert "--headless" in out


def test_cli_config_override_before_subcommand(monkeypatch, tmp_path):
    captured = {}

    def fake_run_eval(args):
        captured["args"] = args
        return 0

    config_path = tmp_path / "custom.yaml"
    config_path.write_text("settings: {}\n")
    original_config = os.environ.get("RETRIEVAL_EVALUATOR_CONFIG")

    monkeypatch.delenv("RETRIEVAL_EVALUATOR_CONFIG", raising=False)
    monkeypatch.setattr(cli, "_run_eval", fake_run_eval)

    try:
        result = cli.main(
            [
                "--config",
                str(config_path),
                "eval",
                "--collections",
                "EXAMPLE",
                "--alpha",
                "1.0",
                "0.5",
                "--k-values",
                "1",
                "5",
            ]
        )

        assert result == 0
        assert captured["args"].collections == ["EXAMPLE"]
        assert captured["args"].alpha == [[1.0, 0.5]]
        assert captured["args"].k_values == [1, 5]
        assert os.environ["RETRIEVAL_EVALUATOR_CONFIG"] == str(config_path)
    finally:
        if original_config is None:
            os.environ.pop("RETRIEVAL_EVALUATOR_CONFIG", None)
        else:
            os.environ["RETRIEVAL_EVALUATOR_CONFIG"] = original_config


def test_cli_config_override_after_subcommand(monkeypatch, tmp_path):
    captured = {}

    def fake_run_eval(args):
        captured["config"] = args.config
        return 0

    config_path = tmp_path / "custom.yaml"
    config_path.write_text("settings: {}\n")
    original_config = os.environ.get("RETRIEVAL_EVALUATOR_CONFIG")

    monkeypatch.delenv("RETRIEVAL_EVALUATOR_CONFIG", raising=False)
    monkeypatch.setattr(cli, "_run_eval", fake_run_eval)

    try:
        result = cli.main(["eval", "--config", str(config_path), "--collections", "EXAMPLE"])

        assert result == 0
        assert captured["config"] == str(config_path)
    finally:
        if original_config is None:
            os.environ.pop("RETRIEVAL_EVALUATOR_CONFIG", None)
        else:
            os.environ["RETRIEVAL_EVALUATOR_CONFIG"] = original_config


def test_cli_repeated_alpha_flags_are_flattened():
    args = cli.build_parser().parse_args(
        [
            "eval",
            "--alpha",
            "1.0",
            "--alpha",
            "0.5",
            "0.0",
        ]
    )

    assert args.alpha == [[1.0], [0.5, 0.0]]
    assert cli._flatten_option_groups(args.alpha) == [1.0, 0.5, 0.0]


def test_run_eval_uses_pipeline_defaults_from_config(monkeypatch):
    captured = {}

    def fake_evaluate_embedders(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(
        "retrieval_evaluator.evaluation.embedders_evaluation.evaluate_embedders",
        fake_evaluate_embedders,
    )
    monkeypatch.setattr(
        config_module,
        "config",
        {
            "settings": config_module.DEFAULT_CONFIG["settings"],
            "retrieval_evaluation": {
                "pipeline": {
                    "default_alphas": [0.7, 0.3],
                    "default_k_values": [1, 5, 10],
                }
            }
        },
    )

    args = cli.build_parser().parse_args(["eval", "--collections", "EXAMPLE"])
    result = cli._run_eval(args)

    assert result == 0
    assert captured["alphas"] == [0.7, 0.3]
    assert captured["k_values"] == [1, 5, 10]


def test_run_eval_supports_direct_model_config(monkeypatch):
    captured = {}

    def fake_evaluate_embedders(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(
        "retrieval_evaluator.evaluation.embedders_evaluation.evaluate_embedders",
        fake_evaluate_embedders,
    )
    monkeypatch.setattr(
        config_module,
        "config",
        {
            "settings": config_module.DEFAULT_CONFIG["settings"],
            "retrieval_evaluation": {
                "pipeline": {
                    "default_alphas": [1.0],
                    "default_k_values": [1, 3, 5, 10],
                }
            }
        },
    )

    args = cli.build_parser().parse_args(
        [
            "eval",
            "--collections",
            "EXAMPLE",
            "--model",
            "intfloat/multilingual-e5-base",
            "--dataset-dir",
            "eval_data",
            "--results-file",
            "results.csv",
            "--artifacts-dir",
            "artifacts",
            "--max-tokens",
            "1024",
            "--query-prompt",
            "query: ",
            "--passage-prompt",
            "passage: ",
            "--precision",
            "float32",
        ]
    )
    result = cli._run_eval(args)

    assert result == 0
    assert captured["embedder_configs"] == {
        "intfloat/multilingual-e5-base": {
            "provider": "hf",
            "model": "intfloat/multilingual-e5-base",
            "max_tokens": 1024,
            "query_prompt": "query: ",
            "passage_prompt": "passage: ",
            "precision": "float32",
        }
    }
    assert captured["dataset_dir"] == "eval_data"


def test_run_dashboard_launches_streamlit(monkeypatch):
    captured = {}

    def fake_find_spec(name):
        return object() if name in {"streamlit", "altair"} else None

    def fake_run(cmd, env, check):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["check"] = check

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    args = cli.build_parser().parse_args(
        [
            "dashboard",
            "--results-file",
            "runs.csv",
            "--artifacts-dir",
            "artifacts",
            "--dataset-dir",
            "eval_data",
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--headless",
        ]
    )
    result = cli._run_dashboard(args)

    assert result == 0
    assert captured["cmd"][:4] == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
    ]
    assert captured["cmd"][4].endswith("app.py")
    assert "--server.address" in captured["cmd"]
    assert "--server.port" in captured["cmd"]
    assert "--server.headless" in captured["cmd"]
    assert captured["env"]["RETRIEVAL_EVALUATOR_RESULTS_FILE"] == "runs.csv"
    assert captured["env"]["RETRIEVAL_EVALUATOR_RUN_DETAILS_DIR"] == "artifacts"
    assert captured["env"]["RETRIEVAL_EVALUATOR_DATASET_DIR"] == "eval_data"
    assert captured["check"] is False


def test_run_dashboard_ships_theme_flags(monkeypatch):
    # pip installs lack the repo's .streamlit/config.toml, so the intended
    # light theme must travel as streamlit CLI flags.
    captured = {}

    def fake_find_spec(name):
        return object() if name in {"streamlit", "altair"} else None

    def fake_run(cmd, env, check):
        captured["cmd"] = cmd

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    args = cli.build_parser().parse_args(["dashboard", "--headless"])
    assert cli._run_dashboard(args) == 0

    cmd = captured["cmd"]
    assert cmd[cmd.index("--server.address") + 1] == "127.0.0.1"
    for flag, value in [
        ("--theme.base", "light"),
        ("--theme.primaryColor", "black"),
        ("--theme.secondaryBackgroundColor", "#F8F9FB"),
        ("--client.toolbarMode", "minimal"),
        ("--ui.hideTopBar", "true"),
    ]:
        assert flag in cmd, f"missing {flag}"
        assert cmd[cmd.index(flag) + 1] == value


def test_run_dashboard_handles_keyboard_interrupt(monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli.subprocess, "run", interrupt)

    args = cli.build_parser().parse_args(["dashboard", "--headless"])

    assert cli._run_dashboard(args) == 130


def test_run_dashboard_requires_ui_dependencies(monkeypatch):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)

    args = cli.build_parser().parse_args(["dashboard"])
    with pytest.raises(SystemExit) as excinfo:
        cli._run_dashboard(args)

    assert "retrieval-evaluator[dashboard]" in str(excinfo.value)


def test_run_eval_rejects_model_and_embedders_together():
    args = cli.build_parser().parse_args(
        [
            "eval",
            "--collections",
            "EXAMPLE",
            "--model",
            "intfloat/multilingual-e5-base",
            "--embedders",
            "multilingual-e5-base",
        ]
    )

    with pytest.raises(SystemExit):
        cli._run_eval(args)


def test_run_build_dataset_uses_dataset_defaults_from_config(monkeypatch):
    captured = {}

    def fake_build_eval_sets(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "retrieval_evaluator.dataset.dataset_builder.build_eval_sets",
        fake_build_eval_sets,
    )

    # Provide a complete config with defaults merged in
    test_config = config_module._deep_merge(
        config_module.DEFAULT_CONFIG,
        {
            "retrieval_evaluation": {
                "dataset": {
                    "sleep_s": 0.25,
                    "default_seed": 7,
                    "max_kmeans_iters": 55,
                    "chunk_limit": 11,
                }
            }
        },
    )
    monkeypatch.setattr(
        config_module,
        "config",
        test_config,
    )

    args = cli.build_parser().parse_args(["build-dataset", "--collections", "support_docs"])
    result = cli._run_build_dataset(args)

    assert result == 0
    assert captured["collections"] == ["support_docs"]
    assert captured["sleep_s"] == 0.25
    assert captured["seed"] == 7
    assert captured["max_km_iters"] == 55
    assert captured["chunk_limit"] == 11
    assert captured["llm_model"] == "openai:gpt-4o"


def test_run_build_dataset_supports_direct_jsonl_configuration(monkeypatch):
    captured = {}

    def fake_build_eval_sets(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "retrieval_evaluator.dataset.dataset_builder.build_eval_sets",
        fake_build_eval_sets,
    )
    monkeypatch.setattr(
        config_module,
        "config",
        {
            "settings": {
                "evaluation_dataset_dir": "eval_data",
            },
            "llm": {
                "model": "openai:gpt-4o",
            },
            "retrieval_evaluation": {
                "dataset": {
                    "source": {
                        "type": "weaviate",
                        "jsonl": {
                            "root_dir": "raw_data/source_collections",
                            "filename_template": "{collection}.jsonl",
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
                    "document_selection_embedder": "sentence-transformers/all-MiniLM-L6-v2",
                    "n_docs_per_language": 30,
                    "sleep_s": 1,
                    "default_seed": 42,
                    "max_kmeans_iters": 100,
                    "chunk_limit": 200,
                }
            },
        },
    )

    args = cli.build_parser().parse_args(
        [
            "build-dataset",
            "--collections",
            "support_docs",
            "--output-dir",
            "custom_eval_data",
            "--source-type",
            "jsonl",
            "--jsonl-root",
            "raw_export",
            "--jsonl-filename-template",
            "{collection}.records.jsonl",
            "--id-field",
            "chunk_uuid",
            "--text-field",
            "body",
            "--doc-id-field",
            "document_id",
            "--chunk-id-field",
            "position",
            "--language-field",
            "lang",
            "--filter-field",
            "chunk_type",
            "--filter-value",
            "text",
            "--llm-model",
            "openai:gpt-4.1-mini",
            "--document-selection-embedder",
            "sentence-transformers/all-MiniLM-L12-v2",
            "--n-docs-per-language",
            "12",
        ]
    )

    result = cli._run_build_dataset(args)

    assert result == 0
    assert captured["collections"] == ["support_docs"]
    assert captured["settings"]["evaluation_dataset_dir"] == "custom_eval_data"
    assert captured["dataset_cfg"]["source"]["type"] == "jsonl"
    assert captured["dataset_cfg"]["source"]["jsonl"] == {
        "root_dir": "raw_export",
        "filename_template": "{collection}.records.jsonl",
    }
    assert captured["dataset_cfg"]["field_mapping"] == {
        "id": "chunk_uuid",
        "text": "body",
        "doc_id": "document_id",
        "chunk_id": "position",
        "language": "lang",
        "chunk_filter_field": "chunk_type",
        "chunk_filter_value": "text",
    }
    assert captured["dataset_cfg"]["document_selection_embedder"] == "sentence-transformers/all-MiniLM-L12-v2"
    assert captured["dataset_cfg"]["n_docs_per_language"] == 12
    assert captured["llm_model"] == "openai:gpt-4.1-mini"


def test_load_defaults_when_no_config_file(monkeypatch):
    monkeypatch.setattr(config_module, "_search_paths", lambda: [])

    loaded = config_module._load()

    assert loaded["settings"]["evaluation_dataset_dir"] == "eval_data"
    assert loaded["retrieval_evaluation"]["pipeline"]["default_alphas"] == [1.0]


def test_load_applies_connection_env_overrides(monkeypatch):
    monkeypatch.setattr(config_module, "_search_paths", lambda: [])
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_WEAVIATE_URL", "http://localhost:8081/")
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_WEAVIATE_KEY", "secret-weaviate-key")
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_QDRANT_URL", "http://localhost:6334")
    monkeypatch.setenv("RETRIEVAL_EVALUATOR_QDRANT_API_KEY", "secret-qdrant-key")

    loaded = config_module._load()

    assert loaded["database"]["weaviate"]["url"] == "http://localhost:8081/"
    assert loaded["database"]["weaviate"]["key"] == "secret-weaviate-key"
    assert loaded["retrieval_evaluation"]["dataset"]["source"]["qdrant"]["url"] == "http://localhost:6334"
    assert loaded["retrieval_evaluation"]["dataset"]["source"]["qdrant"]["api_key"] == "secret-qdrant-key"


def test_import_dataset_csv(tmp_path):
    import csv

    from retrieval_evaluator.cli import main

    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", newline="") as f:
        w = csv.DictWriter(f, ["cid", "body", "parent"])
        w.writeheader()
        w.writerow({"cid": "c1", "body": "Hello.", "parent": "d1"})

    queries = tmp_path / "queries.csv"
    with queries.open("w", newline="") as f:
        w = csv.DictWriter(f, ["qid", "question", "ans"])
        w.writeheader()
        w.writerow({"qid": "q1", "question": "What?", "ans": "c1"})

    out = tmp_path / "out"
    ret = main([
        "import-dataset",
        "--format", "csv",
        "--corpus", str(corpus),
        "--queries", str(queries),
        "--corpus-id-field", "cid",
        "--corpus-text-field", "body",
        "--query-id-field", "qid",
        "--query-text-field", "question",
        "--relevant-id-field", "ans",
        "--output", str(out),
    ])

    assert ret == 0
    assert (out / "corpus.jsonl").exists()
    assert (out / "qrels.tsv").exists()


def test_import_dataset_missing_format_fails(tmp_path):
    import pytest

    from retrieval_evaluator.cli import main

    with pytest.raises(SystemExit):
        main(["import-dataset", "--output", str(tmp_path)])


def test_cli_eval_prints_metrics_table(monkeypatch, capsys):
    df = pd.DataFrame(
        [{"embedder": "m", "collection": "c", "alpha": 1.0, "R@1": 0.5, "MRR@10": 0.6}]
    )
    monkeypatch.setattr(
        "retrieval_evaluator.evaluation.embedders_evaluation.evaluate_embedders",
        lambda **kwargs: df,
    )
    assert cli.main(["eval", "--collections", "c", "--model", "some/model"]) == 0
    out = capsys.readouterr().out
    assert "R@1" in out
    assert "0.500" in out


def test_cli_eval_reports_all_runs_cached(monkeypatch, capsys):
    monkeypatch.setattr(
        "retrieval_evaluator.evaluation.embedders_evaluation.evaluate_embedders",
        lambda **kwargs: pd.DataFrame(),
    )
    assert cli.main(["eval", "--collections", "c", "--model", "some/model"]) == 0
    assert "--force-rerun" in capsys.readouterr().out


def test_cli_eval_missing_embedders_config_exits_cleanly(monkeypatch):
    def _raise(**kwargs):
        raise FileNotFoundError("Embedders config not found. Run `retrieval-evaluator init`.")

    monkeypatch.setattr(
        "retrieval_evaluator.evaluation.embedders_evaluation.evaluate_embedders", _raise
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["eval", "--collections", "c"])
    assert "retrieval-evaluator init" in str(excinfo.value)


@pytest.mark.parametrize(
    "arguments",
    [
        ["eval", "--alpha", "1.1"],
        ["eval", "--max-tokens", "0"],
        ["eval", "--batch-size", "0"],
        ["build-dataset", "--collections", "c", "--sleep-s", "-1"],
        ["build-dataset", "--collections", "c", "--chunk-limit", "0"],
        ["dashboard", "--port", "70000"],
    ],
)
def test_cli_rejects_invalid_numeric_options(arguments):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


def test_cli_turns_expected_handler_errors_into_clean_exit(monkeypatch):
    def fail(_args):
        raise ValueError("bad source")

    monkeypatch.setattr(cli, "_run_check_source", fail)

    with pytest.raises(SystemExit, match="bad source"):
        cli.main(["check-source", "--collections", "c"])
