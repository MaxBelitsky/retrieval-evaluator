from __future__ import annotations

from typing import Sequence


def candidate_micro_ids(candidate_id: str, macro2micro: dict[str, list[str]]) -> list[str]:
    """Return the underlying micro-chunk IDs for a retrieved candidate."""
    return macro2micro.get(candidate_id, [candidate_id])


def positive_qrel_ids(qrel_scores: dict[str, int] | None) -> set[str]:
    """Return only positively relevant chunk IDs from a qrels row."""
    return {
        candidate_id
        for candidate_id, score in (qrel_scores or {}).items()
        if score is not None and score > 0
    }


def expand_ranked_micro_ids(
    ranked: list[str],
    macro2micro: dict[str, list[str]],
    limit: int | None = None,
) -> list[str]:
    """
    Expand ranked retrieval results into unique micro-chunk IDs while preserving order.

    `limit` applies to the number of retrieved results, not to the expanded micro IDs.
    This matches retrieval semantics when a returned macro chunk contains multiple
    source micro chunks.
    """
    expanded = []
    seen = set()
    ranked_slice = ranked if limit is None else ranked[:limit]

    for candidate_id in ranked_slice:
        for micro_id in candidate_micro_ids(candidate_id, macro2micro):
            if micro_id not in seen:
                expanded.append(micro_id)
                seen.add(micro_id)

    return expanded


def compute_ranking_metrics(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    macro2micro: dict[str, list[str]],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Compute retrieval metrics for a set of query rankings.

    Metrics per k:
    - R@k:          fraction of queries where at least one relevant doc appears in top-k.
    - Recall@k:     average fraction of relevant docs retrieved in top-k (per query).
    - R_all@k:      fraction of queries where ALL relevant docs appear in top-k.
    - MRR@max(k):   mean reciprocal rank of the first relevant doc.

    Args:
        rankings:    {qid: [cid, ...]} ranked candidate IDs per query.
        qrels:       {qid: {cid: relevance_score}} per query; only scores > 0 count as relevant.
        macro2micro: {macro_cid: [micro_cid, ...]} chunk merge mapping (empty dict if no merging).
        k_values:    iterable of k cut-offs to evaluate.
    """
    hits = {k: 0 for k in k_values}
    recall = {k: [] for k in k_values}
    all_retrieved = {k: 0 for k in k_values}
    mrr = 0.0
    n = 0  # queries that have at least one positive qrel entry
    max_k = max(k_values)

    for qid, qrel_scores in qrels.items():
        positives = positive_qrel_ids(qrel_scores)
        if not positives:
            continue
        ranked = rankings.get(qid, [])
        n += 1

        for k in k_values:
            top_k = set(expand_ranked_micro_ids(ranked, macro2micro, limit=k))
            num_found = len(positives & top_k)
            recall[k].append(num_found / len(positives))
            if positives.issubset(top_k):
                all_retrieved[k] += 1

        # Classic R@k (hit at rank ≤ k) and MRR use retrieved-result ranks.
        found_rank = None
        for rank, candidate_id in enumerate(ranked[:max_k], 1):
            if positives.intersection(candidate_micro_ids(candidate_id, macro2micro)):
                found_rank = rank
                break
        for k in k_values:
            if found_rank is not None and found_rank <= k:
                hits[k] += 1
        if found_rank is not None:
            mrr += 1.0 / found_rank

    if n == 0:
        return (
            {f"R@{k}": 0.0 for k in k_values}
            | {f"Recall@{k}": 0.0 for k in k_values}
            | {f"R_all@{k}": 0.0 for k in k_values}
            | {f"MRR@{max_k}": 0.0}
        )

    out = {f"R@{k}": hits[k] / n for k in k_values}
    out.update({f"Recall@{k}": sum(recall[k]) / n for k in k_values})
    out.update({f"R_all@{k}": all_retrieved[k] / n for k in k_values})
    out[f"MRR@{max_k}"] = mrr / n
    return out
