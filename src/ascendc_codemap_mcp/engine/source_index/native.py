# -*- coding: utf-8 -*-
"""Optional native lexical scanner. Python regex is the default.

Set ``UO_NATIVE_SCANNER=0`` to skip the import probe. A real Rust/tree-sitter
extension can expose ``scan_file(path, root, registry) -> SourceFacts`` as
``uo_native_scan.scan_file``; missing that module is not an error.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from ascendc_codemap_mcp.engine.source_index.model import SourceFacts

_LOADED: Callable[..., SourceFacts] | None | bool = False


def native_scanner() -> Callable[..., SourceFacts] | None:
    """Return a native scan callable, or None to use the Python scanner."""
    global _LOADED
    if _LOADED is not False:
        return _LOADED if callable(_LOADED) else None
    raw = os.environ.get("UO_NATIVE_SCANNER", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        _LOADED = None
        return None
    try:
        import uo_native_scan  # type: ignore[import-not-found]

        fn = getattr(uo_native_scan, "scan_file", None)
        _LOADED = fn if callable(fn) else None
    except Exception:  # noqa: BLE001
        _LOADED = None
    return _LOADED if callable(_LOADED) else None


def scan_file_or_none(path: Path, *, root: str, registry: set[str] | None) -> SourceFacts | None:
    fn = native_scanner()
    if fn is None:
        return None
    try:
        facts = fn(path, root=root, registry=registry)
    except Exception:  # noqa: BLE001
        return None
    return facts if isinstance(facts, SourceFacts) else None
