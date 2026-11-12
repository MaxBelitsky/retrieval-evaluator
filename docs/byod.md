# Bring Your Own Dataset

If you have real queries and relevance judgments, use them instead of generating synthetic queries. `retrieval-evaluator` accepts the common BEIR layout and validates IDs, text, and qrel references before embedding anything.

## Directory layout

The compact layout used by this project is:

```text
eval_data/
  my-collection/
    corpus.jsonl
    queries.jsonl
    qrels.tsv
    meta.json       # optional
```

Official BEIR split directories are also supported:

```text
eval_data/my-collection/qrels/test.tsv
eval_data/my-collection/qrels/dev.tsv
```

`eval` reads the `test` split by default. Select another with `--qrels-split dev` or `evaluate(..., qrels_split="dev")`.

## `corpus.jsonl`

Write one JSON object per line:

```json
{"_id":"chunk-1","text":"The actual chunk text.","metadata":{"doc_id":"document-1","chunk_id":0}}
```

- `_id` is required, non-empty, and unique after conversion to a string.
- `text` is required and must be a non-empty string.
- `metadata.doc_id` groups adjacent chunks from one document for optional macro-chunk evaluation.
- `metadata.chunk_id` controls within-document order; numeric strings such as `"10"` are sorted numerically.

`title` and other top-level BEIR corpus fields are preserved as metadata when the dataset is loaded.

## `queries.jsonl`

```json
{"_id":"q-1","text":"What does the user want to know?"}
```

Query IDs must be unique and query text must be a non-empty string.

## Relevance judgments

The flat four-column form is:

```text
q-1	0	chunk-1	1
q-1	0	chunk-2	1
q-2	0	chunk-3	1
```

The standard three-column form, commonly stored at `qrels/test.tsv`, is also accepted:

```text
query-id	corpus-id	score
q-1	chunk-1	1
q-2	chunk-3	1
```

Scores greater than zero are relevant; zero and negative judgments are ignored for scoring. Every qrel query and corpus ID must exist in the corresponding JSONL file, and duplicate `(query, corpus)` judgments are rejected.

## `meta.json`

Metadata is optional. Dataset fingerprint and counts are always derived from the current corpus, queries, and selected qrels bytes. Dataset writers add a file similar to:

```json
{
  "created_utc": "2026-09-03T10:00:00Z",
  "dataset_fingerprint": "...",
  "n_documents": 20,
  "n_chunks": 120,
  "n_queries": 45,
  "seed": 42
}
```

Evaluation does not rewrite an imported dataset or trust stale derived values in `meta.json`.

## Evaluate the dataset

```bash
retrieval-evaluator eval \
  --collections my-collection \
  --dataset-dir eval_data \
  --model intfloat/multilingual-e5-base \
  --query-prompt "query: " --passage-prompt "passage: "
```

Python API:

```python
from retrieval_evaluator import evaluate

results = evaluate(
    dataset_dir="eval_data",
    collections=["my-collection"],
    embedder_configs={
        "e5": {
            "provider": "hf",
            "model": "intfloat/multilingual-e5-base",
            "query_prompt": "query: ",
            "passage_prompt": "passage: ",
        }
    },
)
```

## Convert CSV or JSONL QA pairs

Use `import-dataset` when corpus and query records already identify the relevant chunk:

```bash
retrieval-evaluator import-dataset \
  --format csv \
  --corpus chunks.csv \
  --queries qa_pairs.csv \
  --corpus-id-field chunk_id \
  --corpus-text-field content \
  --corpus-doc-id-field document_id \
  --query-id-field question_id \
  --query-text-field question \
  --relevant-id-field chunk_id \
  --output eval_data/my-collection
```

Use `--format jsonl` with the same field options for JSONL inputs.
Repeat a query row with the same ID and text for each relevant chunk when a query
has multiple relevance judgments. Conflicting text for the same query ID is rejected.

## Import from the Hugging Face Hub

Install the optional importer:

```bash
pip install "retrieval-evaluator[datasets]"
```

For datasets with one context and question per row:

```bash
retrieval-evaluator import-dataset \
  --format hf \
  --dataset-name squad \
  --split validation \
  --context-field context \
  --question-field question \
  --output eval_data/squad-validation
```

This adapter treats identical context strings as the same corpus item and links each question to its context. Inspect the result before use if the source dataset has richer or graded relevance semantics.

## Download a public BEIR benchmark

The upstream `beir` package can download benchmark archives in the expected layout:

```bash
pip install beir
python -c "from beir import util; util.download_and_unzip('https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip', 'eval_data')"
retrieval-evaluator eval --collections fiqa --dataset-dir eval_data \
  --model intfloat/multilingual-e5-base \
  --query-prompt "query: " --passage-prompt "passage: "
```

Public benchmark scores are useful for a general sanity check; private-corpus judgments are usually more representative of a production RAG system's vocabulary, chunking, and user intent.
