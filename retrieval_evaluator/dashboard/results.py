from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from retrieval_evaluator.dashboard.data import latest_runs
from retrieval_evaluator.dashboard.metrics import (
    available_mrr_metric,
    available_r_metrics,
    format_metric,
    format_runtime,
    primary_r_metric,
)

CHART_COLORS = ["#2563eb", "#166534", "#dc2626", "#7c3aed", "#d97706", "#0891b2"]


def history_line_chart(df: pd.DataFrame, metric: str, color_field: str):
    return (
        alt.Chart(df, title=f"{metric} over time")
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y(f"{metric}:Q", title=metric, axis=alt.Axis(format="%")),
            color=alt.Color(
                f"{color_field}:N",
                scale=alt.Scale(range=CHART_COLORS),
                title=None,
                legend=alt.Legend(orient="bottom", columns=3, labelLimit=300),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip(f"{color_field}:N", title=color_field.replace("_", " ").title()),
                alt.Tooltip(f"{metric}:Q", title=metric, format=".3f"),
            ],
        )
        .properties(height=320)
    )


def _build_leaderboard(latest: pd.DataFrame, selected_collection: str) -> pd.DataFrame:
    metric_cols = available_r_metrics(latest)
    mrr_metric = available_mrr_metric(latest)
    if mrr_metric is not None:
        metric_cols.append(mrr_metric)
    if not metric_cols:
        return pd.DataFrame()

    id_cols = ["embedder"]
    for column in ("index_backend", "qrels_split", "alpha"):
        if column in latest.columns and latest[column].notna().any():
            id_cols.append(column)

    if selected_collection == "All":
        groups = latest.groupby(id_cols, dropna=False)
        agg = groups[metric_cols].mean()
        if "runtime_s" in latest.columns:
            agg["runtime_s"] = groups["runtime_s"].mean()
        agg["collections"] = latest.groupby(id_cols, dropna=False)["collection"].nunique()
        leaderboard = agg.reset_index()
    else:
        cols = id_cols + metric_cols
        if "runtime_s" in latest.columns:
            cols.append("runtime_s")
        leaderboard = latest[cols].copy()

    sort_metric = primary_r_metric(leaderboard)
    if sort_metric is not None:
        sort_cols = [sort_metric]
        ascending = [False]
        if "alpha" in leaderboard.columns:
            sort_cols.append("alpha")
            ascending.append(True)
        leaderboard = leaderboard.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return leaderboard.reset_index(drop=True)


def results_tab(runs_df: pd.DataFrame) -> None:
    if runs_df.empty:
        st.info("No evaluation results found.")
        return

    collections = sorted(runs_df["collection"].dropna().unique())
    recent_collections = runs_df.sort_values("timestamp", ascending=False)["collection"].dropna()
    most_recent = recent_collections.iloc[0] if not recent_collections.empty else None
    collection_options = ["All"] + list(collections)
    default_idx = (
        collection_options.index(most_recent)
        if most_recent in collection_options
        else min(1, len(collection_options) - 1)
    )
    col, _ = st.columns([1, 3])
    selected_collection = col.selectbox("Collection", collection_options, index=default_idx, key="results_collection")

    run_dimensions = [
        column
        for column in ("index_backend", "qrels_split", "alpha")
        if column in runs_df.columns and runs_df[column].notna().any()
    ]
    if selected_collection == "All":
        group_cols = ["embedder", "collection", *run_dimensions]
        latest = latest_runs(runs_df, group_cols)
    else:
        group_cols = ["embedder", *run_dimensions]
        latest = latest_runs(runs_df[runs_df["collection"] == selected_collection], group_cols)

    if latest.empty:
        st.info("No runs available.")
        return

    if "dataset_status" in latest.columns:
        n_stale = int((latest["dataset_status"] == "stale").sum())
        latest = latest[latest["dataset_status"] != "stale"]
        if n_stale:
            st.caption(
                f"⚠️ {n_stale} run(s) hidden — the dataset changed since they ran. "
                "Re-run `retrieval-evaluator eval` to refresh them."
            )
        if latest.empty:
            st.info("No runs match the current dataset version. Re-run `retrieval-evaluator eval`.")
            return

    primary_metric = primary_r_metric(latest)
    kpis: list[tuple[str, str]] = [("Embedders", str(latest["embedder"].nunique()))]
    if selected_collection == "All":
        kpis.append(("Collections", str(latest["collection"].nunique())))
    chips = "".join(
        f'<div style="background:#F8F9FB;border-radius:8px;padding:10px 22px;text-align:center">'
        f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.07em;color:#64748b;text-transform:uppercase;margin-bottom:3px">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:#0f172a;line-height:1">{value}</div>'
        f'</div>'
        for label, value in kpis
    )
    st.markdown(f'<div style="display:flex;gap:10px;margin-bottom:1rem">{chips}</div>', unsafe_allow_html=True)

    leaderboard = _build_leaderboard(latest, selected_collection)
    if not leaderboard.empty:
        display = leaderboard.copy()
        leaderboard_metric_cols = available_r_metrics(leaderboard)
        leaderboard_mrr_metric = available_mrr_metric(leaderboard)
        if leaderboard_mrr_metric is not None:
            leaderboard_metric_cols.append(leaderboard_mrr_metric)
        for col in leaderboard_metric_cols:
            display[col] = display[col].apply(format_metric)
        if "runtime_s" in display.columns:
            display["runtime_s"] = display["runtime_s"].apply(format_runtime)
        st.dataframe(display, width="stretch", hide_index=True)

    with st.expander("History"):
        if primary_metric is not None:
            hist_df = (
                runs_df if selected_collection == "All"
                else runs_df[runs_df["collection"] == selected_collection]
            ).copy()
            def _series_label(row: pd.Series) -> str:
                details: list[str] = []
                if "index_backend" in run_dimensions and pd.notna(row.get("index_backend")):
                    details.append(str(row["index_backend"]))
                if "qrels_split" in run_dimensions and pd.notna(row.get("qrels_split")):
                    details.append(str(row["qrels_split"]))
                if "alpha" in run_dimensions and pd.notna(row.get("alpha")):
                    details.append(f"α={float(row['alpha']):.1f}")
                suffix = f" ({', '.join(details)})" if details else ""
                return f"{row['embedder']}{suffix}"

            hist_df["_series"] = hist_df.apply(_series_label, axis=1)
            all_metrics = available_r_metrics(hist_df)
            mrr = available_mrr_metric(hist_df)
            if mrr:
                all_metrics.append(mrr)
            history_metric = st.selectbox(
                "Metric",
                all_metrics,
                index=all_metrics.index(primary_metric) if primary_metric in all_metrics else 0,
                key="history_metric",
            )
            st.altair_chart(history_line_chart(hist_df, history_metric, "_series"), width="stretch")
