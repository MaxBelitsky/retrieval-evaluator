from retrieval_evaluator.config import get_config
from retrieval_evaluator.dataset.chunk_sources import create_chunk_source
from retrieval_evaluator.dataset.document_selection import sort_chunks

REQUIRED_FIELDS = ["id", "text", "doc_id"]
OPTIONAL_FIELDS = ["chunk_id", "language"]
OPTIONAL_NOTES = {
    "language": "per-language sampling disabled",
    "chunk_id": "chunk ordering disabled",
}
SAMPLE_SIZE = 10


def validate_chunk_source(collection_name: str, dataset_cfg: dict | None = None) -> bool:
    if dataset_cfg is None:
        dataset_cfg = get_config()["retrieval_evaluation"]["dataset"]
    field_mapping = dataset_cfg["field_mapping"]

    chunk_source = create_chunk_source(dataset_cfg)
    raw_chunks = chunk_source.load_chunks(collection_name)
    if not raw_chunks:
        print(f"✗ No chunks returned from source for collection '{collection_name}'")
        return False

    df = sort_chunks(raw_chunks, field_mapping=field_mapping)

    print(f"\nField mapping validation for '{collection_name}':")
    has_error = False
    for field in REQUIRED_FIELDS:
        source_name = field_mapping.get(field, field)
        if field in df.columns:
            print(f"  ✓ {field:<12} ← {source_name:<20} (required, found)")
        else:
            print(f"  ✗ {field:<12} ← {source_name:<20} (required, MISSING)")
            has_error = True

    for field in OPTIONAL_FIELDS:
        source_name = field_mapping.get(field, field)
        if field in df.columns:
            print(f"  ✓ {field:<12} ← {source_name:<20} (optional, found)")
        else:
            note = OPTIONAL_NOTES.get(field, "disabled")
            print(f"    {field:<12} ← {source_name:<20} (optional, not found — {note})")

    if has_error:
        missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
        print(f"\n✗ Missing required fields: {missing}")
        print("  Check your field_mapping in config/retrieval-evaluator.yaml.")
        return False

    filter_field = field_mapping.get("chunk_filter_field")
    filter_value = field_mapping.get("chunk_filter_value")
    if filter_field:
        if filter_field not in df.columns:
            print(f"\n✗ Configured filter field is missing: {filter_field}")
            return False
        if filter_value is not None:
            df = df[df[filter_field] == filter_value]
            if df.empty:
                print(f"\n✗ No chunks remain after filtering {filter_field}={filter_value!r}")
                return False

    invalid_fields = [
        field
        for field in REQUIRED_FIELDS
        if df[field].isna().any() or (df[field].astype(str).str.strip() == "").any()
    ]
    if invalid_fields:
        print(f"\n✗ Required fields contain null or empty values: {invalid_fields}")
        return False
    if not df["text"].map(lambda value: isinstance(value, str)).all():
        print("\n✗ Required text field must contain strings")
        return False
    if df["id"].astype(str).duplicated().any():
        print("\n✗ Chunk IDs must be unique after filtering")
        return False

    sample = df.head(SAMPLE_SIZE)
    print(f"\nSample rows ({min(3, len(sample))} of {len(df)}):")
    for _, row in sample.head(3).iterrows():
        text_preview = str(row["text"])[:80].replace("\n", " ")
        print(f"  id={str(row['id'])!r}  doc_id={str(row['doc_id'])!r}  text={text_preview!r}")

    print(f"\n✓ Source validation passed for '{collection_name}'")
    return True
