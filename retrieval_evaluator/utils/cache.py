import hashlib
import json
import re
from typing import Any


def source_cache_filename(source: str, collection_name: str, identity: dict[str, Any]) -> str:
    """Build a filesystem-safe cache name scoped to a source connection and configuration."""
    safe_collection = re.sub(r"[^A-Za-z0-9_.-]+", "_", collection_name).strip("._") or "collection"
    digest = hashlib.sha256(
        json.dumps(
            {"source": source, "collection": collection_name, "identity": identity},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{safe_collection}_{source}_{digest}.json"
