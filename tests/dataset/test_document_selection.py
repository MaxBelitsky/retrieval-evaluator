import numpy as np
import pandas as pd

from retrieval_evaluator.dataset.document_selection import pool_document_embeddings, select_diverse_documents


def test_pool_document_embeddings_pairs_correct_ids_when_unsorted():
    # chunks arrive in d2-then-d1 order (not sorted by doc_id)
    df = pd.DataFrame([
        {"doc_id": "d2", "text": "chunk a"},
        {"doc_id": "d1", "text": "chunk b"},
        {"doc_id": "d2", "text": "chunk c"},
    ])
    # d2 has vecs [1,0] and [3,0] → mean [2,0]
    # d1 has vec  [0,5]           → mean [0,5]
    vecs = np.array([[1.0, 0.0], [0.0, 5.0], [3.0, 0.0]], dtype=np.float32)

    pooled, doc_ids = pool_document_embeddings(df, vecs)

    d1_idx = doc_ids.index("d1")
    d2_idx = doc_ids.index("d2")
    np.testing.assert_allclose(pooled[d1_idx], [0.0, 5.0], err_msg="d1 vector mismatch")
    np.testing.assert_allclose(pooled[d2_idx], [2.0, 0.0], err_msg="d2 vector mismatch")


def test_pool_document_embeddings_single_chunk_per_doc():
    df = pd.DataFrame([
        {"doc_id": "b", "text": "text"},
        {"doc_id": "a", "text": "text"},
    ])
    vecs = np.array([[9.0, 0.0], [0.0, 7.0]], dtype=np.float32)

    pooled, doc_ids = pool_document_embeddings(df, vecs)

    a_idx = doc_ids.index("a")
    b_idx = doc_ids.index("b")
    np.testing.assert_allclose(pooled[a_idx], [0.0, 7.0])
    np.testing.assert_allclose(pooled[b_idx], [9.0, 0.0])


def test_select_diverse_documents_handles_duplicate_vectors():
    vectors = np.zeros((3, 2), dtype=np.float32)

    selected = select_diverse_documents(vectors, ["a", "b", "c"], n_documents=2)

    assert len(selected) == 2
