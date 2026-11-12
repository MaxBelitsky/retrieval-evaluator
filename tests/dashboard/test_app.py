import pandas as pd
import pytest

from retrieval_evaluator.dashboard.data import latest_runs
from retrieval_evaluator.dashboard.inspector import build_query_rows, r_at_k_labels
from retrieval_evaluator.dashboard.results import _build_leaderboard


@pytest.mark.parametrize(
    ("retrieved", "macro2micro", "corpus_meta", "missing"),
    [
        (["wrong"], {}, {"gold": {"text": "Expected answer text."}}, True),
        (["wrong"], {"merged": ["gold", "adjacent"]}, {"merged": {"text": "Expected answer text."}}, True),
        (["merged"], {"merged": ["gold", "adjacent"]}, {"merged": {"text": "Expected answer text."}}, False),
    ],
)
def test_inspector_shows_saved_text_for_missing_relevant_passages(retrieved, macro2micro, corpus_meta, missing):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(f'''
from retrieval_evaluator.dashboard.inspector import build_query_rows, render_query_detail
rankings = {{"q": {retrieved!r}}}
qrels = {{"q": {{"gold": 1}}}}
macro2micro = {macro2micro!r}
corpus_meta = {corpus_meta!r}
queries = {{"q": "What is the answer?"}}
rows = build_query_rows(rankings, qrels, macro2micro, queries, [1])
render_query_detail("q", rows, rankings, qrels, macro2micro, corpus_meta, queries, [1])
''').run()

    assert not app.exception
    assert bool(app.expander) == missing
    if missing:
        assert app.expander[0].label == "gold"
        assert app.text[0].value == "Expected answer text."
        if macro2micro:
            assert any("merged candidate: merged" in item.value for item in app.caption)


def test_r_at_k_labels_uses_requested_ks():
    labels = r_at_k_labels(["chunk-1"], {"chunk-1"}, {}, ks=(1, 3))
    assert set(labels.keys()) == {"R@1", "R@3"}
    assert labels["R@1"] is True
    assert labels["R@3"] is True


def _sample_latest() -> pd.DataFrame:
    return pd.DataFrame({
        "embedder": ["bge-m3", "bge-m3", "minilm"],
        "collection": ["col-a", "col-b", "col-a"],
        "R@1":   [0.50, 0.60, 0.40],
        "R@3":   [0.80, 0.85, 0.70],
        "MRR@3":[0.65, 0.72, 0.55],
        "runtime_s": [4.0, 4.5, 1.5],
    })


def test_build_leaderboard_specific_collection_columns():
    col_a = _sample_latest().query("collection == 'col-a'").copy()
    result = _build_leaderboard(col_a, "col-a")
    assert "embedder" in result.columns
    for col in ["R@1", "R@3", "MRR@3", "runtime_s"]:
        assert col in result.columns
    assert "R@5" not in result.columns
    assert "collections" not in result.columns
    assert len(result) == 2  # bge-m3 and minilm on col-a


def test_build_leaderboard_specific_sorted_by_highest_available_r_metric():
    col_a = _sample_latest().query("collection == 'col-a'").copy()
    result = _build_leaderboard(col_a, "col-a")
    r3 = result["R@3"].tolist()
    assert r3 == sorted(r3, reverse=True)


def test_build_leaderboard_all_adds_collections_column():
    result = _build_leaderboard(_sample_latest(), "All")
    assert "collections" in result.columns


def test_build_leaderboard_all_averages_metrics():
    result = _build_leaderboard(_sample_latest(), "All")
    bge = result[result["embedder"] == "bge-m3"].iloc[0]
    assert abs(bge["R@3"] - 0.825) < 1e-6  # mean(0.80, 0.85)
    assert bge["collections"] == 2


def test_build_leaderboard_all_sorted_by_highest_available_r_metric():
    result = _build_leaderboard(_sample_latest(), "All")
    r3 = result["R@3"].tolist()
    assert r3 == sorted(r3, reverse=True)


def test_build_leaderboard_returns_empty_df_when_no_metric_columns():
    df = pd.DataFrame({
        "embedder": ["bge-m3"],
        "collection": ["col-a"],
        "runtime_s": [4.0],
    })
    result = _build_leaderboard(df, "col-a")
    assert result.empty


def test_build_query_rows_uses_requested_ks_and_ignores_non_positive_qrels():
    rankings = {
        "q1": ["chunk-ignored", "chunk-hit"],
        "q2": ["chunk-negative"],
    }
    qrels = {
        "q1": {"chunk-ignored": 0, "chunk-hit": 1},
        "q2": {"chunk-negative": 0},
    }
    macro2micro: dict = {}
    qid_to_query = {
        "q1": "what is hybrid search",
        "q2": "should be skipped",
    }

    rows = build_query_rows(rankings, qrels, macro2micro, qid_to_query, ks=[1, 2])

    assert "R@1" in rows.columns
    assert "R@2" in rows.columns
    assert "R@5" not in rows.columns
    assert rows["qid"].tolist() == ["q1"]
    assert not bool(rows.loc[0, "R@1"])
    assert bool(rows.loc[0, "R@2"])
    assert rows.loc[0, "status"] == "Recovered by @2"


def test_build_query_rows_shows_missing_ranking_as_miss():
    rows = build_query_rows(
        rankings={},
        qrels={"q1": {"relevant": 1}},
        macro2micro={},
        qid_to_query={"q1": "missing ranking"},
        ks=[1, 3],
    )

    assert rows[["qid", "status", "cohort"]].to_dict("records") == [
        {"qid": "q1", "status": "Miss @3", "cohort": "miss"}
    ]
    assert pd.isna(rows.loc[0, "first_hit_rank"])


def test_build_leaderboard_breaks_metric_ties_by_alpha():
    latest = pd.DataFrame({
        "embedder": ["m", "m", "m"],
        "collection": ["c", "c", "c"],
        "alpha": [0.5, 1.0, 0.0],
        "R@1": [0.8, 0.8, 0.8],
    })
    result = _build_leaderboard(latest, "c")
    assert result["alpha"].tolist() == [0.0, 0.5, 1.0]


def test_latest_runs_keeps_legacy_rows_with_missing_new_dimensions():
    runs = pd.DataFrame(
        {
            "embedder": ["legacy", "new"],
            "index_backend": [None, "legacy-backend"],
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]),
        }
    )

    latest = latest_runs(runs, ["embedder", "index_backend"])

    assert set(latest["embedder"]) == {"legacy", "new"}


def test_build_leaderboard_keeps_legacy_rows_with_missing_new_dimensions():
    latest = pd.DataFrame(
        {
            "embedder": ["legacy", "new"],
            "collection": ["c", "c"],
            "index_backend": [None, "legacy-backend"],
            "R@1": [0.7, 0.8],
        }
    )

    result = _build_leaderboard(latest, "All")

    assert set(result["embedder"]) == {"legacy", "new"}
