import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_openai_mock(embedding_dim: int = 8):
    mock_openai = MagicMock()
    mock_embedding = [0.1] * embedding_dim
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=mock_embedding)]
    mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_response
    return mock_openai


def test_openai_encode_returns_float32_array(monkeypatch):
    mock_openai = _make_openai_mock(embedding_dim=8)
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    from retrieval_evaluator.evaluation.embedder import Embedder

    cfg = {"provider": "openai", "model": "text-embedding-3-small", "max_tokens": 8191}
    embedder = Embedder(cfg)
    result = embedder.encode(["Hello world"], batch_size=512)

    assert result.shape == (1, 8)
    assert result.dtype == np.float32


def test_openai_encode_uses_cache(monkeypatch):
    mock_openai = _make_openai_mock(embedding_dim=8)
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    from retrieval_evaluator.evaluation.embedder import Embedder

    cfg = {"provider": "openai", "model": "text-embedding-3-small", "max_tokens": 8191}
    embedder = Embedder(cfg)
    texts = ["Hello world"]

    embedder.encode(texts, batch_size=512)
    embedder.encode(texts, batch_size=512)

    assert mock_openai.OpenAI.return_value.embeddings.create.call_count == 1


def test_openai_missing_package_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)

    from retrieval_evaluator.evaluation.embedder import Embedder

    cfg = {"provider": "openai", "model": "text-embedding-3-small", "max_tokens": 8191}
    with pytest.raises(ImportError, match="openai"):
        Embedder(cfg)


def _make_cohere_mock(embedding_dim: int = 8):
    mock_cohere = MagicMock()
    mock_response = MagicMock()
    mock_response.embeddings.float_ = [[0.2] * embedding_dim]
    mock_cohere.ClientV2.return_value.embed.return_value = mock_response
    return mock_cohere


def test_cohere_encode_returns_float32_array(monkeypatch):
    mock_cohere = _make_cohere_mock(embedding_dim=8)
    monkeypatch.setitem(sys.modules, "cohere", mock_cohere)

    from retrieval_evaluator.evaluation.embedder import Embedder

    cfg = {"provider": "cohere", "model": "embed-multilingual-v3.0", "max_tokens": 512}
    embedder = Embedder(cfg)
    result = embedder.encode(["Hello world"], batch_size=96)

    assert result.shape == (1, 8)
    assert result.dtype == np.float32


def test_cohere_uses_search_document_for_passage(monkeypatch):
    from retrieval_evaluator.utils.constants import EmbeddingTextType
    mock_cohere = _make_cohere_mock(embedding_dim=8)
    monkeypatch.setitem(sys.modules, "cohere", mock_cohere)

    from retrieval_evaluator.evaluation.embedder import Embedder

    cfg = {"provider": "cohere", "model": "embed-multilingual-v3.0", "max_tokens": 512}
    embedder = Embedder(cfg)
    embedder.encode(["Some passage"], batch_size=96, text_type=EmbeddingTextType.passage)

    call_kwargs = mock_cohere.ClientV2.return_value.embed.call_args.kwargs
    assert call_kwargs["input_type"] == "search_document"


def test_cohere_uses_search_query_for_query(monkeypatch):
    from retrieval_evaluator.utils.constants import EmbeddingTextType
    mock_cohere = _make_cohere_mock(embedding_dim=8)
    monkeypatch.setitem(sys.modules, "cohere", mock_cohere)

    from retrieval_evaluator.evaluation.embedder import Embedder

    cfg = {"provider": "cohere", "model": "embed-multilingual-v3.0", "max_tokens": 512}
    embedder = Embedder(cfg)
    embedder.encode(["A query"], batch_size=96, text_type=EmbeddingTextType.query)

    call_kwargs = mock_cohere.ClientV2.return_value.embed.call_args.kwargs
    assert call_kwargs["input_type"] == "search_query"


def test_cohere_missing_package_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cohere", None)

    from retrieval_evaluator.evaluation.embedder import Embedder

    cfg = {"provider": "cohere", "model": "embed-multilingual-v3.0", "max_tokens": 512}
    with pytest.raises(ImportError, match="cohere"):
        Embedder(cfg)
