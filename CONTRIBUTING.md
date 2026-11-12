# Contributing

Thanks for considering a contribution to `retrieval-evaluator`.

## Setup

Install the project with all optional features and dev dependencies:

```bash
uv sync --locked --all-extras --group dev
```

Useful local commands:

```bash
uv run retrieval-evaluator --help
uv run retrieval-evaluator eval --help
uv run retrieval-evaluator build-dataset --help
uv run retrieval-evaluator dashboard --help
uv run --locked ruff check retrieval_evaluator tests examples
uv run --locked pytest -p no:cacheprovider
```

`tests/integration` exercises real provider SDKs against simulated HTTP responses,
plus Qdrant's local store. These tests run in the normal suite without API keys,
model downloads, or paid calls. To also test source exports against a temporary
local Weaviate instance:

```bash
RETRIEVAL_EVALUATOR_TEST_WEAVIATE=1 uv run --locked pytest tests/integration
```

The Weaviate test requires local listening ports and may download its embedded
binary on first use. Its collection and cache use temporary storage.

## Development Guidelines

- Keep changes scoped and focused.
- Prefer improving the public UX over adding internal complexity.
- Add or update tests for behavior changes when practical.
- Update documentation when CLI behavior, config fields, or workflows change.
- Use environment variables for secrets; do not commit credentials in config examples.

## Pull Requests

Before opening a PR:

1. Run Ruff and the full test suite using the commands above.
2. Check whether the README, config reference, or contributor docs need updates.
3. If dependencies changed, run `uv lock` and include the updated `uv.lock`.
4. Keep the PR description concrete:
   - what changed
   - why it changed
   - how it was validated

Good PRs usually include:

- a clear user-facing motivation
- tests or an explanation of why tests were not added
- minimal unrelated cleanup

## Issues

- Use the bug template for reproducible failures and regressions.
- Use the feature template for product ideas, extension points, and workflow improvements.
- For questions, include the exact command, config, and error output when relevant.

## Project Priorities

This project is trying to be:

- useful on a real private corpus
- easy to try without a large setup cost
- explicit about evaluation limitations

Changes that improve those goals are usually a good fit.

## Architecture

Before making changes, read **[docs/architecture.md](docs/architecture.md)** — it covers the two pipelines (eval and dataset generation) and the exact extension points for adding new chunk sources, retrieval backends, and metrics.

## Roadmap

Open issues track planned improvements and good first contributions:
[github.com/MaxBelitsky/retrieval-evaluator/issues](https://github.com/MaxBelitsky/retrieval-evaluator/issues)
