# -*- coding: utf-8 -*-
"""SourceFacts cache: process-local plus content-hash disk.

Disk key = sha256(file bytes) + scanner_version + registry_version + architecture.
Store under ``<op>/.ascendc-codemap/<arch>/cache/source_facts/<key>.pkl``.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import threading
from pathlib import Path
from typing import Any

SCANNER_VERSION = "lexical-py-1"

_INDEX: dict[str, Any] = {}
_LOCK = threading.Lock()
_DISK_ENABLE = "UO_SOURCE_FACTS_CACHE"


def cache_get(key: str) -> Any | None:
    with _LOCK:
        return _INDEX.get(key)


def cache_put(key: str, value: Any) -> None:
    with _LOCK:
        _INDEX[key] = value


def cache_clear() -> None:
    with _LOCK:
        _INDEX.clear()


def disk_enabled() -> bool:
    raw = os.environ.get(_DISK_ENABLE, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def facts_cache_key(
    content_sha: str,
    *,
    scanner_version: str = SCANNER_VERSION,
    registry_version: str = "",
    architecture: str = "",
) -> str:
    payload = "\0".join(
        [
            str(content_sha or ""),
            str(scanner_version or SCANNER_VERSION),
            str(registry_version or ""),
            str(architecture or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def facts_cache_dir(op_root: str | Path | None, architecture: str) -> Path:
    override = os.environ.get("UO_CACHE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve() / "source_facts"
    root = Path(op_root or ".").expanduser().resolve()
    arch = str(architecture or os.environ.get("UO_ARCHITECTURE") or "default").strip() or "default"
    return root / ".ascendc-codemap" / arch / "cache" / "source_facts"


def disk_get(key: str, op_root: str | Path | None, architecture: str) -> Any | None:
    if not disk_enabled() or not key:
        return None
    path = facts_cache_dir(op_root, architecture) / f"{key}.pkl"
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:  # noqa: BLE001
        return None


def disk_put(key: str, value: Any, op_root: str | Path | None, architecture: str) -> None:
    if not disk_enabled() or not key:
        return
    folder = facts_cache_dir(op_root, architecture)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{key}.pkl"
        tmp = dest.with_suffix(".pkl.tmp")
        with tmp.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(dest)
    except Exception:  # noqa: BLE001
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass
