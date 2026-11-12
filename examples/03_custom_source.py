"""
03_custom_source.py — Connect any data store using the callable chunk source.

This pattern works with any database: PostgreSQL, Elasticsearch, a custom API, etc.
The loader function is registered in config and called by retrieval-evaluator at dataset
build time.

To use from the CLI:

    # In config/retrieval-evaluator.yaml:
    #   retrieval_evaluation:
    #     dataset:
    #       source:
    #         type: callable
    #         callable:
    #           path: "examples.03_custom_source:load_from_my_db"
    #           kwargs:
    #             table: "documents"
    #       field_mapping:
    #         id: id
    #         text: content
    #         doc_id: document_id
    #         chunk_id: chunk_id

    pip install retrieval-evaluator
    uv run python examples/03_custom_source.py

After saving the config above, validate it from the repository root:

    uv run python -m retrieval_evaluator.cli check-source --collections sample_chunks

The module invocation makes this checkout importable. Use the same invocation for
build-dataset, with its required extras and API key. For an installed connector
package, the retrieval-evaluator console command works directly.
"""


def load_from_my_db(collection_name: str, *, force_reload: bool = False, table: str = "documents") -> list[dict]:
    """
    Replace this implementation with your actual database query.

    Must return a list of dicts. Required fields (configurable via field_mapping):
      - id: unique chunk identifier
      - text (or your mapped name): the chunk text
      - doc_id (or your mapped name): the parent document identifier

    Optional fields:
      - chunk_id: integer position within the document (enables chunk merging)
      - language: ISO 639-1 code (enables per-language document sampling)
    """
    import json
    from pathlib import Path

    path = Path(f"examples/data/{collection_name}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"No JSONL file found for collection '{collection_name}' at {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    chunks = load_from_my_db("sample_chunks")
    print(f"Loaded {len(chunks)} chunks")
    print("First chunk:", chunks[0])
    print("\nField names present:", list(chunks[0].keys()))
    print(
        "\nTo use this source in a dataset build, add to config/retrieval-evaluator.yaml:\n"
        "  retrieval_evaluation:\n"
        "    dataset:\n"
        "      source:\n"
        "        type: callable\n"
        "        callable:\n"
        "          path: 'examples.03_custom_source:load_from_my_db'\n"
        "      field_mapping:\n"
        "        id: id\n"
        "        text: content\n"
        "        doc_id: document_id\n"
        "        chunk_id: chunk_id\n"
    )
    print("Validate from the repository root:")
    print("  uv run python -m retrieval_evaluator.cli check-source --collections sample_chunks")
