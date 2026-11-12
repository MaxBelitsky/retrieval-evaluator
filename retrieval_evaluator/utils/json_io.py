import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[Any]:
    """Yield a UTF-8 temporary file and replace path only after a successful write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            yield handle
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(obj: Any, path: Path, *, indent: int | None = None) -> None:
    """Atomically write strict UTF-8 JSON so interrupted writes do not corrupt artifacts."""
    with atomic_text_writer(path) as handle:
        json.dump(obj, handle, ensure_ascii=False, allow_nan=False, indent=indent)
        handle.write("\n")


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read a cached JSON list and reject corrupt or unexpected cache contents."""
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON cache {path}: {exc.msg}") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected {path} to contain a JSON array of objects")
    return value
