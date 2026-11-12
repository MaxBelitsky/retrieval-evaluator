from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from retrieval_evaluator.dashboard.data import load_json, load_queries_jsonl, load_run_data
from retrieval_evaluator.dashboard.metrics import available_r_ks, truncate_text
from retrieval_evaluator.evaluation.metrics import candidate_micro_ids, expand_ranked_micro_ids, positive_qrel_ids
from retrieval_evaluator.utils.paths import safe_child_path


def r_at_k_labels(retrieved, gold_set, macro2micro, ks) -> dict[str, bool]:
    labels = {}
    for k in ks:
        top_k = set(expand_ranked_micro_ids(retrieved, macro2micro, limit=k))
        labels[f"R@{k}"] = bool(gold_set & top_k)
    return labels


def build_query_rows(rankings, qrels, macro2micro, qid_to_query, ks) -> pd.DataFrame:
    metric_cols = [f"R@{k}" for k in ks]
    base_columns = ["qid", "query", "cohort", "status", "first_hit_rank", *metric_cols, "sort_rank"]
    if not ks:
        return pd.DataFrame(columns=base_columns)

    first_k = ks[0]
    last_k = ks[-1]
    rows = []
    for qid, qrel_scores in qrels.items():
        gold_cids = positive_qrel_ids(qrel_scores)
        if not gold_cids:
            continue
        retrieved = rankings.get(qid, [])

        first_hit_rank = next(
            (
                rank
                for rank, candidate_id in enumerate(retrieved, 1)
                if gold_cids.intersection(candidate_micro_ids(candidate_id, macro2micro))
            ),
            None,
        )
        labels = r_at_k_labels(retrieved, gold_cids, macro2micro, ks=ks)
        if labels[f"R@{first_k}"]:
            status = f"Hit @{first_k}"
            cohort = "hit"
        elif first_k != last_k and labels[f"R@{last_k}"]:
            status = f"Recovered by @{last_k}"
            cohort = "recovery"
        else:
            status = f"Miss @{last_k}"
            cohort = "miss"
        rows.append(
            {
                "qid": qid,
                "query": qid_to_query.get(qid, f"[{qid}]"),
                "cohort": cohort,
                "status": status,
                "first_hit_rank": first_hit_rank,
                **labels,
            }
        )
    query_df = pd.DataFrame(rows, columns=base_columns[:-1])
    query_df["first_hit_rank"] = query_df["first_hit_rank"].astype(pd.Int64Dtype())
    query_df["sort_rank"] = query_df["first_hit_rank"].fillna(10_000)
    return query_df


def filter_query_rows(query_rows: pd.DataFrame, cohort: str, query_filter: str) -> pd.DataFrame:
    filtered = query_rows.copy()
    if cohort == "Failures":
        filtered = filtered[filtered["cohort"] == "miss"]
    elif cohort == "Recoveries":
        filtered = filtered[filtered["cohort"] == "recovery"]
    elif cohort == "Hits":
        filtered = filtered[filtered["cohort"] == "hit"]
    if query_filter.strip():
        filtered = filtered[filtered["query"].str.contains(query_filter, case=False, na=False)]
    return filtered.sort_values(["sort_rank", "query"])


def render_query_detail(
    qid: str,
    query_rows: pd.DataFrame,
    rankings,
    qrels,
    macro2micro,
    corpus_meta,
    qid_to_query,
    ks,
) -> None:
    indexed = query_rows.set_index("qid")
    if qid not in indexed.index:
        st.warning(f"Query '{qid}' not found in results.")
        return
    row = indexed.loc[qid]
    query_text = qid_to_query.get(qid, f"[{qid}]")
    gold_cids = positive_qrel_ids(qrels.get(qid, {}))

    st.subheader(query_text)

    chips = ""
    for metric in [f"R@{k}" for k in ks]:
        hit = bool(row[metric])
        if hit:
            chips += (
                f'<span style="background:#dcfce7;color:#166534;border-radius:4px;'
                f'padding:2px 8px;font-size:0.85rem;font-weight:600;margin-right:6px">'
                f'{metric} ✓</span>'
            )
        else:
            chips += (
                f'<span style="background:#fee2e2;color:#991b1b;border-radius:4px;'
                f'padding:2px 8px;font-size:0.85rem;font-weight:600;margin-right:6px">'
                f'{metric} ✗</span>'
            )
    first_hit = "not found" if pd.isna(row["first_hit_rank"]) else f"rank {int(row['first_hit_rank'])}"
    chips += (
        f'<span style="background:#f0f9ff;color:#075985;border-radius:4px;'
        f'padding:2px 8px;font-size:0.85rem;font-weight:600">'
        f'First relevant: {html.escape(first_hit)}</span>'
    )
    st.markdown(chips, unsafe_allow_html=True)

    st.caption("Gold chunk IDs")
    st.code(", ".join(sorted(gold_cids)) if gold_cids else "None", language=None)

    PREVIEW_LEN = 300
    top_k = max(ks)
    retrieved = rankings.get(qid, [])[:top_k]
    missing_gold = gold_cids - set(expand_ranked_micro_ids(retrieved, macro2micro))
    if missing_gold:
        st.caption(f"Relevant passages missing from the top {top_k}")
        for gold_id in sorted(missing_gold):
            candidate_id = gold_id
            if candidate_id not in corpus_meta:
                candidate_id = next(
                    (cid for cid, micro_ids in macro2micro.items() if gold_id in micro_ids),
                    gold_id,
                )
            with st.expander(gold_id):
                if candidate_id != gold_id:
                    st.caption(f"Text from merged candidate: {candidate_id}")
                st.text(corpus_meta.get(candidate_id, {}).get("text", "[No saved text available]"))

    st.caption(f"Top {top_k} retrieved documents")
    if not retrieved:
        st.info("The index returned no results for this query.")
    for rank, cid in enumerate(retrieved, 1):
        micro_ids = candidate_micro_ids(cid, macro2micro)
        meta = corpus_meta.get(cid, {})
        is_gold = bool(gold_cids.intersection(micro_ids))
        text = meta.get("text", "[No text available]")
        if len(text) <= PREVIEW_LEN:
            body_html = f'<div class="doc-body">{html.escape(text)}</div>'
        else:
            preview = html.escape(text[:PREVIEW_LEN])
            rest = html.escape(text[PREVIEW_LEN:])
            body_html = (
                f'<div class="doc-body">{preview}</div>'
                f'<details><summary class="doc-expand"></summary>'
                f'<div class="doc-body">{rest}</div></details>'
            )
        source_note = ""
        if len(micro_ids) > 1:
            source_note = f" &nbsp;·&nbsp; {len(micro_ids)} merged source chunks"
        st.markdown(
            f'<div class="doc-card {"doc-card--hit" if is_gold else ""}">'
            f'<div class="doc-rank">Rank {rank} &nbsp;·&nbsp; '
            f'<code>{html.escape(str(cid))}</code>{source_note}</div>'
            f'{body_html}</div>',
            unsafe_allow_html=True,
        )


def inspector_tab(runs_df: pd.DataFrame, details_dir: Path, data_dir: Path) -> None:
    if runs_df.empty:
        st.info("No runs available.")
        return

    sel_cols = st.columns([1, 2, 1.2, 0.8])

    collections = sorted(runs_df["collection"].dropna().unique())
    selected_collection = sel_cols[0].selectbox("Collection", collections, key="inspector_collection")

    collection_runs = runs_df[runs_df["collection"] == selected_collection]
    embedders = sorted(collection_runs["embedder"].dropna().unique())
    selected_embedder = sel_cols[1].selectbox("Embedder", embedders, key="inspector_embedder")

    embedder_runs = (
        collection_runs[collection_runs["embedder"] == selected_embedder]
        .sort_values("timestamp", ascending=False)
    )
    if "dataset_status" in embedder_runs.columns:
        embedder_runs = embedder_runs[embedder_runs["dataset_status"] != "stale"]
    if embedder_runs.empty:
        st.info("No runs match the current dataset version. Re-run `retrieval-evaluator eval` to inspect fresh runs.")
        return
    run_ids = embedder_runs["run_idx"].tolist()

    def _fmt_run(run_id: str) -> str:
        r = embedder_runs.loc[embedder_runs["run_idx"] == run_id].iloc[0]
        details = [r["timestamp"].strftime("%Y-%m-%d %H:%M")]
        if "index_backend" in embedder_runs.columns and pd.notna(r.get("index_backend")):
            details.append(str(r["index_backend"]))
        if "qrels_split" in embedder_runs.columns and pd.notna(r.get("qrels_split")):
            details.append(str(r["qrels_split"]))
        if "alpha" in embedder_runs.columns and pd.notna(r["alpha"]):
            details.append(f"α={float(r['alpha']):.1f}")
        return " · ".join(details)

    selected_run = sel_cols[2].selectbox("Run", run_ids, format_func=_fmt_run, key="inspector_run")
    selected_run_row = embedder_runs.loc[embedder_runs["run_idx"] == selected_run].iloc[0]
    ks = available_r_ks(selected_run_row)
    if not ks:
        st.error(f"Selected run does not contain any R@K metrics: {selected_run}")
        return

    run_dir = details_dir / str(selected_run)
    if not run_dir.exists():
        st.error(f"Run artifacts not found: {run_dir}")
        return

    run_queries_path = run_dir / "queries.json"
    if run_queries_path.exists():
        qid_to_query = load_json(run_queries_path)
    else:
        try:
            collection_dir = safe_child_path(data_dir, selected_collection, label="collection")
        except ValueError as exc:
            st.error(str(exc))
            return
        queries_path = collection_dir / "queries.jsonl"
        if not queries_path.exists():
            st.error(f"Queries file not found: {queries_path}")
            return
        qid_to_query = load_queries_jsonl(queries_path)
        if str(selected_run_row.get("dataset_status", "unknown")) != "fresh":
            st.warning(
                "This run predates query snapshots; query texts are read from the current "
                "dataset and may not match what was evaluated."
            )
    try:
        rankings, macro2micro, qrels, corpus_meta = load_run_data(run_dir)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    with st.spinner("Loading queries…"):
        query_rows = build_query_rows(rankings, qrels, macro2micro, qid_to_query, ks=ks)

    cohort = st.radio(
        "Cohort",
        ["All", "Failures", "Hits", "Recoveries"],
        horizontal=True,
        key="inspector_cohort",
        label_visibility="collapsed",
    )

    visible = filter_query_rows(query_rows, cohort, "")

    if visible.empty:
        st.warning("No queries match the current filters.")
        return

    left, right = st.columns([1, 1.6], gap="large")
    with left:
        query_lookup = visible.set_index("qid")
        selected_qid = st.selectbox(
            "Focus query",
            visible["qid"].tolist(),
            format_func=lambda qid: truncate_text(query_lookup.loc[qid, "query"], 70),
            key="inspector_focus_query",
        )

        rows_html = ""
        for _, row in visible.iterrows():
            q = html.escape(truncate_text(row["query"], 70))
            status = row["status"]
            rank = "—" if pd.isna(row["first_hit_rank"]) else f"#{int(row['first_hit_rank'])}"
            if row["cohort"] == "hit":
                dot_color = "#16a34a"
            elif row["cohort"] == "miss":
                dot_color = "#dc2626"
            else:
                dot_color = "#94a3b8"
            is_selected = row["qid"] == selected_qid
            bg = "#f8f9fb" if is_selected else "#ffffff"
            rows_html += (
                f'<div style="padding:7px 10px;border-bottom:1px solid #f1f5f9;'
                f'background:{bg};border-radius:6px;margin:4px">'
                f'<div style="font-size:0.82rem;font-weight:500;line-height:1.3;color:#0f172a">{q}</div>'
                f'<div style="font-size:0.72rem;color:#64748b;margin-top:2px;display:flex;align-items:center;gap:5px">'
                f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{dot_color};flex-shrink:0"></span>'
                f'{html.escape(status)} &nbsp;·&nbsp; {rank}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="border:1px solid #e2e8f0;border-radius:8px;overflow-y:auto;'
            f'max-height:480px;background:#fff;margin-top:6px;scrollbar-width:none">'
            f'<style>div::-webkit-scrollbar{{display:none}}</style>{rows_html}</div>',
            unsafe_allow_html=True,
        )
    with right:
        render_query_detail(
            selected_qid, query_rows, rankings, qrels, macro2micro, corpus_meta, qid_to_query, ks=ks
        )
