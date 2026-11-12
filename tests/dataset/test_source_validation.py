from unittest.mock import MagicMock, patch

from retrieval_evaluator.dataset.source_validation import validate_chunk_source

SAMPLE_CHUNKS = [
    {"chunk_id": "c1", "document_id": "doc1", "content": "The quick brown fox.", "lang": "en"},
    {"chunk_id": "c2", "document_id": "doc1", "content": "Jumps over the lazy dog.", "lang": "en"},
]

FIELD_MAPPING = {
    "id": "chunk_id",
    "text": "content",
    "doc_id": "document_id",
    "chunk_id": "chunk_id",
    "language": "lang",
}


def _make_dataset_cfg(field_mapping=None, source_type="jsonl"):
    return {
        "source": {"type": source_type, "jsonl": {"root_dir": ".", "filename_template": "{collection}.jsonl"}},
        "field_mapping": field_mapping or FIELD_MAPPING,
    }


def test_chunk_source_factory_defaults_to_jsonl(tmp_path):
    from retrieval_evaluator.dataset.chunk_sources import JsonlChunkSource, create_chunk_source

    source = create_chunk_source({"source": {"jsonl": {"root_dir": str(tmp_path)}}})

    assert isinstance(source, JsonlChunkSource)


def test_valid_source_returns_true(capsys):
    mock_source = MagicMock()
    mock_source.load_chunks.return_value = SAMPLE_CHUNKS

    with patch("retrieval_evaluator.dataset.source_validation.create_chunk_source", return_value=mock_source):
        result = validate_chunk_source("my_collection", _make_dataset_cfg())

    assert result is True
    captured = capsys.readouterr()
    assert "✓" in captured.out
    assert "required, found" in captured.out


def test_missing_required_field_returns_false(capsys):
    incomplete_chunks = [{"document_id": "doc1", "content": "text"}]
    mapping = {"id": "MISSING_FIELD", "text": "content", "doc_id": "document_id"}
    mock_source = MagicMock()
    mock_source.load_chunks.return_value = incomplete_chunks

    with patch("retrieval_evaluator.dataset.source_validation.create_chunk_source", return_value=mock_source):
        result = validate_chunk_source("my_collection", _make_dataset_cfg(field_mapping=mapping))

    assert result is False
    captured = capsys.readouterr()
    assert "MISSING" in captured.out


def test_optional_language_absent_is_not_an_error(capsys):
    chunks_no_lang = [{"chunk_id": "c1", "document_id": "doc1", "content": "text"}]
    mapping = {"id": "chunk_id", "text": "content", "doc_id": "document_id", "language": "lang"}
    mock_source = MagicMock()
    mock_source.load_chunks.return_value = chunks_no_lang

    with patch("retrieval_evaluator.dataset.source_validation.create_chunk_source", return_value=mock_source):
        result = validate_chunk_source("my_collection", _make_dataset_cfg(field_mapping=mapping))

    assert result is True
    captured = capsys.readouterr()
    assert "per-language sampling disabled" in captured.out


def test_empty_source_returns_false(capsys):
    mock_source = MagicMock()
    mock_source.load_chunks.return_value = []

    with patch("retrieval_evaluator.dataset.source_validation.create_chunk_source", return_value=mock_source):
        result = validate_chunk_source("my_collection", _make_dataset_cfg())

    assert result is False


def test_non_string_text_returns_false(capsys):
    mock_source = MagicMock()
    mock_source.load_chunks.return_value = [
        {"chunk_id": "c1", "document_id": "doc1", "content": 123}
    ]

    with patch("retrieval_evaluator.dataset.source_validation.create_chunk_source", return_value=mock_source):
        result = validate_chunk_source("my_collection", _make_dataset_cfg())

    assert result is False
    assert "must contain strings" in capsys.readouterr().out
