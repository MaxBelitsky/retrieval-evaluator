from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import NAMESPACE_DNS, uuid5

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from retrieval_evaluator.evaluation.embedder import Embedder


def _chunk_order_key(metadata: dict) -> tuple:
    """Sort chunk_ids numerically when possible so string positions like "10" follow "2"."""
    chunk_id = metadata["chunk_id"]
    try:
        return (0, float(chunk_id))
    except (TypeError, ValueError):
        return (1, str(chunk_id))


def merge_chunks(corpus_df: pd.DataFrame, embedder: "Embedder") -> tuple[pd.DataFrame, dict]:
    """
    Merge micro-chunks into macro-chunks per doc for long-context evaluation.
    """
    merged_rows, macro2micro = [], {}

    doc_groups = corpus_df.groupby(corpus_df["metadata"].apply(lambda m: m["doc_id"]))
    for _, doc_df in doc_groups:
        doc_df = doc_df.sort_values(
            "metadata", key=lambda s: s.apply(_chunk_order_key)
        )
        buf_txt, buf_ids, buf_tokens, buf_meta = [], [], 0, None
        for row in doc_df.itertuples():
            token_ids = embedder.tokenize(row.text)
            toks = token_ids if isinstance(token_ids, np.ndarray) else token_ids.numpy()

            if buf_tokens + len(toks[0]) > embedder.max_tokens and buf_txt:
                macro_id = str(uuid5(NAMESPACE_DNS, "|".join(buf_ids)))
                merged_rows.append(
                    {
                        "cid": macro_id,
                        "text": " ".join(buf_txt),
                        "micro_ids": buf_ids.copy(),
                        "metadata": buf_meta,
                    }
                )
                macro2micro[macro_id] = buf_ids.copy()
                buf_txt, buf_ids, buf_tokens, buf_meta = [], [], 0, None

            if not buf_txt:
                buf_meta = row.metadata
            buf_txt.append(row.text)
            buf_ids.append(row.cid)
            buf_tokens += len(toks[0])

        if buf_txt:
            macro_id = buf_ids[0]
            merged_rows.append(
                {
                    "cid": macro_id,
                    "text": " ".join(buf_txt),
                    "micro_ids": buf_ids.copy(),
                    "metadata": buf_meta,
                }
            )
            macro2micro[macro_id] = buf_ids.copy()

    return pd.DataFrame(merged_rows), macro2micro
