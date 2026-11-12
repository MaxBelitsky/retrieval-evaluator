from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from retrieval_evaluator.config import get_config, get_config_path, get_evaluation_dataset_dir
from retrieval_evaluator.utils.logger import setup_logger


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _batch_size(value: str) -> int | str:
    if value == "auto":
        return value
    return _positive_int(value)


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return parsed


def _config_parent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        help=(
            "Path to a config file. Overrides RETRIEVAL_EVALUATOR_CONFIG and the default "
            "config search paths."
        ),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    config_parent = _config_parent_parser()
    parser = argparse.ArgumentParser(
        prog="retrieval-evaluator",
        description="Evaluate embedding models (HuggingFace, OpenAI, Cohere) on your own RAG corpus.",
        parents=[config_parent],
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (includes verbose third-party logs).",
    )
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run retrieval evaluation for one or more collections and embedders.",
        parents=[config_parent],
    )
    eval_parser.add_argument(
        "--collections",
        "-c",
        nargs="+",
        default=["EXAMPLE"],
        help="Dataset collection names under evaluation_dataset_dir.",
    )
    eval_parser.add_argument(
        "--embedders",
        "-e",
        nargs="+",
        help="Embedder names from embedders.yml (default: all).",
    )
    eval_parser.add_argument(
        "--model",
        help=(
            "Model identifier for a single-run quickstart, interpreted per --provider "
            "(default: hf). When set, embedders.yml is not required."
        ),
    )
    eval_parser.add_argument(
        "--provider",
        default="hf",
        choices=["hf", "openai", "cohere"],
        help="Provider for --model (default: hf).",
    )
    eval_parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        help="Maximum input tokens for --model (default: model limit for HuggingFace; 512 otherwise).",
    )
    eval_parser.add_argument(
        "--query-prompt",
        help="Prompt prepended to queries for --model runs.",
    )
    eval_parser.add_argument(
        "--passage-prompt",
        help="Prompt prepended to passages for --model runs.",
    )
    eval_parser.add_argument(
        "--precision",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for --model runs.",
    )
    eval_parser.add_argument(
        "--dataset-dir",
        help="Override the evaluation dataset root.",
    )
    eval_parser.add_argument(
        "--qrels-split",
        default="test",
        help="BEIR qrels split under qrels/ when qrels.tsv is absent (default: test).",
    )
    eval_parser.add_argument(
        "--index-backend",
        choices=["numpy", "weaviate"],
        help="Override the retrieval index backend from config.",
    )
    eval_parser.add_argument(
        "--results-file",
        help="Override the results CSV path.",
    )
    eval_parser.add_argument(
        "--artifacts-dir",
        help="Override the per-run artifacts directory.",
    )
    eval_parser.add_argument(
        "--alpha",
        action="append",
        nargs="+",
        type=_probability,
        help=(
            "Hybrid search alpha values: 1.0=semantic, 0.0=BM25. "
            "Accepts one or more values per flag. "
            "Defaults to retrieval_evaluation.pipeline.default_alphas when omitted."
        ),
    )
    eval_parser.add_argument(
        "--batch-size",
        type=_batch_size,
        default=64,
        help="Encoding batch size or 'auto' (default: 64).",
    )
    eval_parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run even if experiment ID already exists.",
    )
    eval_parser.add_argument(
        "--k-values",
        nargs="+",
        type=_positive_int,
        help="Retrieval cut-offs. Defaults to retrieval_evaluation.pipeline.default_k_values when omitted.",
    )
    eval_parser.set_defaults(handler=_run_eval)

    dataset_parser = subparsers.add_parser(
        "build-dataset",
        help="Build a BEIR-format evaluation dataset from the configured chunk source.",
        parents=[config_parent],
    )
    dataset_parser.add_argument(
        "--collections",
        "-c",
        nargs="+",
        required=True,
        help="Collection names in the configured chunk source.",
    )
    dataset_parser.add_argument(
        "--output-dir",
        help="Override the output dataset root directory.",
    )
    dataset_parser.add_argument(
        "--source-type",
        choices=["jsonl", "weaviate", "qdrant", "callable"],
        help="Override the configured source type.",
    )
    dataset_parser.add_argument(
        "--jsonl-root",
        help="Root directory for `source-type=jsonl`.",
    )
    dataset_parser.add_argument(
        "--jsonl-filename-template",
        help="Filename template for `source-type=jsonl` (default: {collection}.jsonl).",
    )
    dataset_parser.add_argument(
        "--id-field",
        help="Source field mapped to the canonical `id` field.",
    )
    dataset_parser.add_argument(
        "--text-field",
        help="Source field mapped to the canonical `text` field.",
    )
    dataset_parser.add_argument(
        "--doc-id-field",
        help="Source field mapped to the canonical `doc_id` field.",
    )
    dataset_parser.add_argument(
        "--chunk-id-field",
        help="Source field mapped to the canonical `chunk_id` field.",
    )
    dataset_parser.add_argument(
        "--language-field",
        help="Source field mapped to the canonical `language` field.",
    )
    dataset_parser.add_argument(
        "--filter-field",
        help="Optional source field used for pre-filtering chunks.",
    )
    dataset_parser.add_argument(
        "--filter-value",
        help="Optional value kept when --filter-field is set.",
    )
    dataset_parser.add_argument(
        "--llm-model",
        help="Override the LLM model used for query generation.",
    )
    dataset_parser.add_argument(
        "--document-selection-embedder",
        help="Override the embedder used for diverse-document selection.",
    )
    dataset_parser.add_argument(
        "--n-docs-per-language",
        type=_positive_int,
        help="Override the number of selected documents per language.",
    )
    dataset_parser.add_argument(
        "--sleep-s",
        type=_nonnegative_float,
        default=None,
        help="Sleep between LLM calls in seconds. Defaults to retrieval_evaluation.dataset.sleep_s.",
    )
    dataset_parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=None,
        help="Random seed for document selection. Defaults to retrieval_evaluation.dataset.default_seed.",
    )
    dataset_parser.add_argument(
        "--max-km-iters",
        type=_positive_int,
        default=None,
        help="Max K-means iterations. Defaults to retrieval_evaluation.dataset.max_kmeans_iters.",
    )
    dataset_parser.add_argument(
        "--chunk-limit",
        type=_positive_int,
        default=None,
        help="Max chunks to generate queries for. Defaults to retrieval_evaluation.dataset.chunk_limit.",
    )
    dataset_parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Reload chunks from the configured source even if cached.",
    )
    dataset_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the chunk source and field mapping without calling the LLM or writing "
            "an evaluation dataset. Remote sources may populate the configured local cache."
        ),
    )
    dataset_parser.set_defaults(handler=_run_build_dataset)

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the retrieval dashboard for browsing evaluation runs.",
        parents=[config_parent],
    )
    dashboard_parser.add_argument(
        "--results-file",
        help="Override the results CSV path for the dashboard session.",
    )
    dashboard_parser.add_argument(
        "--artifacts-dir",
        help="Override the per-run artifacts directory for the dashboard session.",
    )
    dashboard_parser.add_argument(
        "--dataset-dir",
        help="Override the evaluation dataset root for the dashboard session.",
    )
    dashboard_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind the Streamlit server to this address (default: 127.0.0.1).",
    )
    dashboard_parser.add_argument(
        "--port",
        type=_port,
        help="Bind the Streamlit server to a specific port.",
    )
    dashboard_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the dashboard without opening a browser window.",
    )
    dashboard_parser.set_defaults(handler=_run_dashboard)

    init_parser = subparsers.add_parser(
        "init",
        help="Scaffold a minimal config/retrieval-evaluator.yaml in the current directory.",
        parents=[config_parent],
    )
    init_parser.add_argument(
        "--global",
        dest="global_config",
        action="store_true",
        help="Write to ~/.config/retrieval-evaluator/ instead of the current directory.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting.",
    )
    init_parser.set_defaults(handler=_run_init)

    check_source_parser = subparsers.add_parser(
        "check-source",
        help="Validate chunk source connectivity and field mapping before running build-dataset.",
        parents=[config_parent],
    )
    check_source_parser.add_argument(
        "--collections",
        "-c",
        nargs="+",
        required=True,
        help="Collection names to validate.",
    )
    check_source_parser.set_defaults(handler=_run_check_source)

    import_parser = subparsers.add_parser(
        "import-dataset",
        help="Convert CSV, JSONL, or HuggingFace datasets to BEIR format for evaluation.",
        parents=[config_parent],
    )
    import_parser.add_argument(
        "--format", "-f",
        required=True,
        choices=["csv", "jsonl", "hf"],
        help="Input format.",
    )
    import_parser.add_argument("--output", "-o", required=True, help="Output directory path.")
    import_parser.add_argument("--corpus", help="Path to corpus CSV or JSONL file.")
    import_parser.add_argument("--queries", help="Path to queries CSV or JSONL file.")
    import_parser.add_argument("--corpus-id-field", default="id", help="Field name for chunk IDs.")
    import_parser.add_argument("--corpus-text-field", default="text", help="Field name for chunk text.")
    import_parser.add_argument("--corpus-doc-id-field", default=None, help="Field name for parent document ID (optional).")
    import_parser.add_argument("--query-id-field", default="id", help="Field name for query IDs.")
    import_parser.add_argument("--query-text-field", default="text", help="Field name for query text.")
    import_parser.add_argument("--relevant-id-field", default="chunk_id", help="Field name for the relevant chunk ID in the queries file.")
    import_parser.add_argument("--dataset-name", help="HuggingFace dataset name (e.g. 'squad').")
    import_parser.add_argument("--split", default="train", help="HuggingFace dataset split.")
    import_parser.add_argument("--context-field", default="context", help="HF field containing passage text.")
    import_parser.add_argument("--question-field", default="question", help="HF field containing query text.")
    import_parser.add_argument("--id-field", default=None, help="HF field to use as corpus ID (auto-generated if omitted).")
    import_parser.set_defaults(handler=_run_import_dataset)

    return parser


def _apply_config_override(config_path: str | None) -> None:
    if config_path:
        os.environ["RETRIEVAL_EVALUATOR_CONFIG"] = config_path


def _flatten_option_groups(values: Sequence[Sequence[float]] | None) -> list[float] | None:
    if not values:
        return None
    return [value for group in values for value in group]


def _extract_config_override(argv: Sequence[str]) -> str | None:
    for idx, token in enumerate(argv):
        if token == "--config" and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return None


def _run_eval(args: argparse.Namespace) -> int:
    from retrieval_evaluator.evaluation.embedders_evaluation import evaluate_embedders

    merged_config = get_config()

    if args.model and args.embedders:
        raise SystemExit("`--model` cannot be used together with `--embedders`.")

    embedder_configs = None
    if args.model:
        inline_cfg = {
            "provider": args.provider,
            "model": args.model,
        }
        if args.max_tokens is not None:
            inline_cfg["max_tokens"] = args.max_tokens
        if args.query_prompt is not None:
            inline_cfg["query_prompt"] = args.query_prompt
        if args.passage_prompt is not None:
            inline_cfg["passage_prompt"] = args.passage_prompt
        if args.precision is not None:
            inline_cfg["precision"] = args.precision
        embedder_configs = {args.model: inline_cfg}

    settings = merged_config["settings"]
    config_path = get_config_path()
    dataset_path = get_evaluation_dataset_dir(args.dataset_dir).resolve()
    results_path = Path(args.results_file or settings["evaluation_results_file"]).resolve()
    artifacts_path = Path(args.artifacts_dir or settings["run_details_directory"]).resolve()
    bundled_note = " (bundled example data)" if dataset_path == Path(__file__).resolve().parent / "data" else ""
    print(f"Config:    {config_path or 'built-in defaults (no config file)'}")
    print(f"Dataset:   {dataset_path}{bundled_note}")
    print(f"Results:   {results_path}")
    print(f"Artifacts: {artifacts_path}", flush=True)

    try:
        results_df = evaluate_embedders(
            collections=args.collections,
            embedders=args.embedders or [],
            force_rerun=args.force_rerun,
            alphas=_flatten_option_groups(args.alpha)
            or merged_config["retrieval_evaluation"]["pipeline"]["default_alphas"],
            batch_size=args.batch_size,
            k_values=args.k_values
            or merged_config["retrieval_evaluation"]["pipeline"]["default_k_values"],
            embedder_configs=embedder_configs,
            dataset_dir=args.dataset_dir,
            results_file=args.results_file,
            artifacts_output_dir=args.artifacts_dir,
            qrels_split=args.qrels_split,
            index_backend=args.index_backend,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    if results_df.empty:
        print("No new runs — all requested combinations already have results. Use --force-rerun to re-evaluate.")
        return 0

    metric_cols = [c for c in results_df.columns if c.startswith(("R@", "MRR@"))]
    table_cols = ["embedder", "collection", "alpha", *metric_cols]
    print()
    print(results_df[table_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    dashboard_cmd = ["retrieval-evaluator", "dashboard"]
    config_path = os.environ.get("RETRIEVAL_EVALUATOR_CONFIG")
    if config_path:
        dashboard_cmd.extend(["--config", config_path])
    for option, value in (
        ("--results-file", args.results_file),
        ("--artifacts-dir", args.artifacts_dir),
        ("--dataset-dir", args.dataset_dir),
    ):
        if value:
            dashboard_cmd.extend([option, value])
    print(f"\nResults appended to {results_path}. Browse them with: {shlex.join(dashboard_cmd)}")
    return 0


def _run_build_dataset(args: argparse.Namespace) -> int:
    from retrieval_evaluator.dataset.dataset_builder import build_eval_sets

    merged_config = get_config()
    dataset_cfg = copy.deepcopy(merged_config["retrieval_evaluation"]["dataset"])
    settings = copy.deepcopy(merged_config["settings"])
    if args.output_dir:
        settings["evaluation_dataset_dir"] = args.output_dir

    if args.source_type:
        dataset_cfg["source"]["type"] = args.source_type
    if args.jsonl_root:
        dataset_cfg["source"].setdefault("jsonl", {})["root_dir"] = args.jsonl_root
        dataset_cfg["source"]["type"] = "jsonl"
    if args.jsonl_filename_template:
        dataset_cfg["source"].setdefault("jsonl", {})["filename_template"] = args.jsonl_filename_template
        dataset_cfg["source"]["type"] = "jsonl"

    field_mapping_overrides = {
        "id": args.id_field,
        "text": args.text_field,
        "doc_id": args.doc_id_field,
        "chunk_id": args.chunk_id_field,
        "language": args.language_field,
        "chunk_filter_field": args.filter_field,
        "chunk_filter_value": args.filter_value,
    }
    for key, value in field_mapping_overrides.items():
        if value is not None:
            dataset_cfg["field_mapping"][key] = value

    if args.document_selection_embedder:
        dataset_cfg["document_selection_embedder"] = args.document_selection_embedder
    if args.n_docs_per_language is not None:
        dataset_cfg["n_docs_per_language"] = args.n_docs_per_language

    if args.dry_run:
        from retrieval_evaluator.dataset.source_validation import validate_chunk_source
        all_passed = True
        for collection in args.collections:
            passed = validate_chunk_source(collection, dataset_cfg)
            if not passed:
                all_passed = False
        return 0 if all_passed else 1

    build_eval_sets(
        collections=args.collections,
        sleep_s=args.sleep_s if args.sleep_s is not None else dataset_cfg["sleep_s"],
        seed=args.seed if args.seed is not None else dataset_cfg["default_seed"],
        max_km_iters=(
            args.max_km_iters
            if args.max_km_iters is not None
            else dataset_cfg["max_kmeans_iters"]
        ),
        chunk_limit=args.chunk_limit if args.chunk_limit is not None else dataset_cfg["chunk_limit"],
        force_reload=args.force_reload,
        dataset_cfg=dataset_cfg,
        settings=settings,
        llm_model=args.llm_model or merged_config["llm"]["model"],
    )
    return 0


def _dashboard_entrypoint_path() -> Path:
    return Path(__file__).resolve().parent / "dashboard" / "app.py"


def _run_dashboard(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("streamlit") is None or importlib.util.find_spec("altair") is None:
        raise SystemExit(
            "Dashboard dependencies are not installed. "
            "Install with `pip install \"retrieval-evaluator[dashboard]\"` "
            "or `uv sync --extra dashboard`."
        )

    dashboard_path = _dashboard_entrypoint_path()
    if not dashboard_path.exists():
        raise SystemExit(f"Dashboard entrypoint not found: {dashboard_path}")

    env = os.environ.copy()
    path_overrides = {
        "RETRIEVAL_EVALUATOR_RESULTS_FILE": args.results_file,
        "RETRIEVAL_EVALUATOR_RUN_DETAILS_DIR": args.artifacts_dir,
        "RETRIEVAL_EVALUATOR_DATASET_DIR": args.dataset_dir,
    }
    for env_name, value in path_overrides.items():
        if value:
            env[env_name] = value

    # The dashboard's styling assumes a light theme. Ship it as CLI flags so
    # pip installs (which lack the repo's .streamlit/config.toml) look the same.
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--theme.base",
        "light",
        "--theme.primaryColor",
        "black",
        "--theme.secondaryBackgroundColor",
        "#F8F9FB",
        "--client.toolbarMode",
        "minimal",
        "--ui.hideTopBar",
        "true",
    ]
    cmd.extend(["--server.address", args.host])
    if args.port is not None:
        cmd.extend(["--server.port", str(args.port)])
    if args.headless:
        cmd.extend(["--server.headless", "true"])

    try:
        return subprocess.run(cmd, env=env, check=False).returncode
    except KeyboardInterrupt:
        return 130


def _run_init(args: argparse.Namespace) -> int:
    from retrieval_evaluator.templates import MINIMAL_CONFIG, STARTER_EMBEDDERS

    if args.global_config:
        config_dir = Path.home() / ".config" / "retrieval-evaluator"
    else:
        config_dir = Path("config")

    config_path = config_dir / "retrieval-evaluator.yaml"
    embedders_path = config_dir / "embedders.yml"
    config_content = MINIMAL_CONFIG
    if args.global_config:
        config_content = config_content.replace(
            "embedders_config_path: config/embedders.yml",
            f"embedders_config_path: {json.dumps(str(embedders_path))}",
        )

    config_dir.mkdir(parents=True, exist_ok=True)

    created = []
    skipped = []

    for path, content in [
        (config_path, config_content),
        (embedders_path, STARTER_EMBEDDERS),
    ]:
        if path.exists() and not args.force:
            skipped.append(str(path))
        else:
            path.write_text(content, encoding="utf-8")
            created.append(str(path))

    if created:
        print(f"Created: {', '.join(created)}")
    if skipped:
        print(f"Skipped (already exists): {', '.join(skipped)} — use --force to overwrite")
    if created:
        print("\nNext steps:")
        print(f"  1. Edit {config_path} — set source.type and field_mapping")
        print(f"  2. Edit {embedders_path} — define the models you want to compare")
        print("  3. Run: retrieval-evaluator check-source --collections YOUR_COLLECTION")
    return 0


def _run_import_dataset(args: argparse.Namespace) -> int:
    from retrieval_evaluator.dataset.importer import DatasetImporter

    importer = DatasetImporter()
    output = Path(args.output)

    if args.format == "csv":
        if not args.corpus or not args.queries:
            raise SystemExit("--corpus and --queries are required for --format csv")
        importer.import_from_csv(
            corpus_path=args.corpus,
            queries_path=args.queries,
            output_dir=output,
            corpus_id_field=args.corpus_id_field,
            corpus_text_field=args.corpus_text_field,
            query_id_field=args.query_id_field,
            query_text_field=args.query_text_field,
            relevant_id_field=args.relevant_id_field,
            corpus_doc_id_field=args.corpus_doc_id_field,
        )
    elif args.format == "jsonl":
        if not args.corpus or not args.queries:
            raise SystemExit("--corpus and --queries are required for --format jsonl")
        importer.import_from_jsonl(
            corpus_path=args.corpus,
            queries_path=args.queries,
            output_dir=output,
            corpus_id_field=args.corpus_id_field,
            corpus_text_field=args.corpus_text_field,
            query_id_field=args.query_id_field,
            query_text_field=args.query_text_field,
            relevant_id_field=args.relevant_id_field,
            corpus_doc_id_field=args.corpus_doc_id_field,
        )
    elif args.format == "hf":
        if not args.dataset_name:
            raise SystemExit("--dataset-name is required for --format hf")
        importer.import_from_hf(
            dataset_name=args.dataset_name,
            output_dir=output,
            context_field=args.context_field,
            question_field=args.question_field,
            split=args.split,
            id_field=args.id_field,
        )
    else:
        raise SystemExit(f"Unknown format: {args.format}")

    print(f"Dataset written to {output}")
    return 0


def _run_check_source(args: argparse.Namespace) -> int:
    from retrieval_evaluator.dataset.source_validation import validate_chunk_source

    merged_config = get_config()
    dataset_cfg = merged_config["retrieval_evaluation"]["dataset"]

    all_passed = True
    for collection in args.collections:
        passed = validate_chunk_source(collection, dataset_cfg)
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    config_override = _extract_config_override(argv_list)

    if not argv_list:
        parser.print_help()
        return 0

    args = parser.parse_args(argv_list)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    setup_logger(debug=getattr(args, "debug", False))
    _apply_config_override(args.config or config_override)
    try:
        return args.handler(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
