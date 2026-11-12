from pathlib import Path


def safe_child_path(root: str | Path, child: str, *, label: str = "name") -> Path:
    """Join an untrusted child path without allowing it to escape its root."""
    if not isinstance(child, str) or not child.strip():
        raise ValueError(f"{label} must be a non-empty string")
    root_path = Path(root).expanduser()
    candidate = root_path / child
    resolved_root = root_path.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root) or resolved_candidate == resolved_root:
        raise ValueError(f"{label} resolves outside its configured root: {child!r}")
    return candidate
