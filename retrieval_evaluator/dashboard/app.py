from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import streamlit as st

from retrieval_evaluator.config import get_config, get_evaluation_dataset_dir
from retrieval_evaluator.dashboard.data import current_dataset_fingerprint, load_runs, mark_dataset_status
from retrieval_evaluator.dashboard.inspector import inspector_tab
from retrieval_evaluator.dashboard.results import results_tab
from retrieval_evaluator.utils.paths import safe_child_path

st.set_page_config(
    page_title="retrieval-evaluator Dashboard",
    layout="wide",
)
alt.theme.enable("none")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
          padding-top: 1rem;
          padding-bottom: 2rem;
        }

        /* Larger tab buttons */
        [data-testid="stTab"] button {
          font-size: 1rem !important;
          padding: 0.5rem 1.5rem !important;
        }

        /* Pill-style cohort radio buttons */
        [data-testid="stRadio"] label {
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 20px;
          padding: 3px 12px;
          margin-right: 2px;
          cursor: pointer;
          font-size: 0.82rem;
          white-space: nowrap;
        }
        [data-testid="stRadio"] input[type="radio"] {
          display: none;
        }

        .doc-card {
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 6px;
          padding: 0.7rem 0.9rem;
          margin-bottom: 0.45rem;
        }

        .doc-card--hit {
          background: #f0fdf4;
          border-color: #166534;
        }

        .doc-rank {
          color: #94a3b8;
          font-size: 0.75rem;
          margin-bottom: 0.2rem;
        }

        .doc-body {
          color: #0f172a;
          line-height: 1.5;
          font-size: 0.88rem;
        }

        details > summary.doc-expand {
          list-style: none;
          cursor: pointer;
          color: #166534;
          font-size: 0.78rem;
          margin-top: 4px;
          user-select: none;
        }
        details > summary.doc-expand::-webkit-details-marker { display: none; }
        details > summary.doc-expand::before { content: "show more ↓"; }
        details[open] > summary.doc-expand::before { content: "show less ↑"; }

        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()

    settings = dict(get_config()["settings"])
    env_overrides = {
        "evaluation_results_file": os.environ.get("RETRIEVAL_EVALUATOR_RESULTS_FILE"),
        "run_details_directory": os.environ.get("RETRIEVAL_EVALUATOR_RUN_DETAILS_DIR"),
        "evaluation_dataset_dir": os.environ.get("RETRIEVAL_EVALUATOR_DATASET_DIR"),
    }
    for key, value in env_overrides.items():
        if value:
            settings[key] = value

    csv_path = Path(settings["evaluation_results_file"]).expanduser()
    details_dir = Path(settings["run_details_directory"]).expanduser()
    data_dir = get_evaluation_dataset_dir(env_overrides["evaluation_dataset_dir"]).expanduser()

    if not csv_path.exists():
        st.info(
            "**No evaluation results yet.**\n\n"
            "Run your first evaluation, then refresh this page:\n\n"
            "```\nretrieval-evaluator eval --collections EXAMPLE --model sentence-transformers/all-MiniLM-L6-v2\n```\n\n"
            f"(Looking for results at `{csv_path}` — override with `--results-file`.)"
        )
        st.stop()

    runs_df = load_runs(csv_path)

    fingerprints = {}
    fingerprint_columns = ["collection"]
    if "qrels_split" in runs_df:
        fingerprint_columns.append("qrels_split")
    for _, row in runs_df[fingerprint_columns].drop_duplicates().iterrows():
        collection = row["collection"]
        if not isinstance(collection, str) or not collection:
            continue
        split_value = row.get("qrels_split", "test")
        split = split_value if isinstance(split_value, str) and split_value else "test"
        try:
            collection_dir = safe_child_path(data_dir, collection, label="collection")
        except ValueError:
            fingerprints[(collection, split)] = None
        else:
            fingerprints[(collection, split)] = current_dataset_fingerprint(
                collection_dir,
                qrels_split=split,
            )
    runs_df = mark_dataset_status(runs_df, fingerprints)

    tab_results, tab_inspector = st.tabs(["Results", "Inspector"])
    with tab_results:
        results_tab(runs_df)
    with tab_inspector:
        inspector_tab(runs_df, details_dir, data_dir)


if __name__ == "__main__":
    main()
