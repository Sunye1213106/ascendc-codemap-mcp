# -*- coding: utf-8 -*-
"""Process-local source text cache shared across CodeMap enrichment passes.

Every disk read is counted so ``performance.yaml`` can show files scanned more
than once. Callers should go through this module instead of ``Path.read_text``.
"""

from __future__ import annotations

import threading
from pathlib import Path

_TEXT: dict[str, str] = {}
_BY_BASENAME: dict[str, list[str]] = {}
_MASKED: dict[str, str] = {}
_MASKED_BY_ID: dict[int, str] = {}
_LOCK = threading.RLock()


def _key(path: str | Path) -> str:
    from ascendc_codemap_mcp.engine.paths import resolved

    return str(resolved(path))


def _remember(key: str) -> None:
    base = Path(key).name
    bucket = _BY_BASENAME.setdefault(base, [])
    if key not in bucket:
        bucket.append(key)


def read_text(path: str | Path) -> str:
    key = _key(path)
    with _LOCK:
        hit = _TEXT.get(key)
    if hit is not None:
        try:
            from ascendc_codemap_mcp.engine.perf import record_read

            record_read(key, 0, cache_hit=True)
        except Exception:  # noqa: BLE001
            pass
        return hit
    text = Path(key).read_text(encoding="utf-8", errors="replace")
    with _LOCK:
        existing = _TEXT.get(key)
        if existing is not None:
            try:
                from ascendc_codemap_mcp.engine.perf import record_read

                record_read(key, 0, cache_hit=True)
            except Exception:  # noqa: BLE001
                pass
            return existing
        _TEXT[key] = text
        _remember(key)
    try:
        from ascendc_codemap_mcp.engine.perf import record_read

        record_read(key, len(text.encode("utf-8", errors="replace")), cache_hit=False)
    except Exception:  # noqa: BLE001
        pass
    return text


def cached_snippet(path: str | Path, line: int) -> str:
    """Return one cached source line, or empty if the file was never read."""
    if int(line or 0) <= 0:
        return ""
    raw = str(path or "").replace("\\", "/")
    if not raw:
        return ""
    text = None
    needle = raw.lstrip("./")
    base = needle.rsplit("/", 1)[-1]
    keys = _BY_BASENAME.get(base) or _TEXT.keys()
    for key in keys:
        val = _TEXT.get(key)
        if val is None:
            continue
        norm = key.replace("\\", "/")
        if norm == raw or norm.endswith("/" + needle) or needle.endswith(norm.split("/")[-1]) and needle in norm:
            text = val
            break
    if text is None:
        return ""
    lines = text.splitlines()
    if int(line) > len(lines):
        return ""
    return lines[int(line) - 1].strip()[:400]


def mask_cached(text: str) -> str:
    """Mask comments/strings; cache by object identity of ``read_text`` hits."""
    key = id(text)
    with _LOCK:
        hit = _MASKED_BY_ID.get(key)
    if hit is not None:
        return hit
    from ascendc_codemap_mcp.engine.cpp_lex import mask_non_code

    masked = mask_non_code(text)
    with _LOCK:
        _MASKED_BY_ID[key] = masked
    return masked


def masked_text(path: str | Path) -> str:
    """Comment/string-masked source with the same line breaks as ``read_text``."""
    raw = read_text(path)
    key = _key(path)
    with _LOCK:
        hit = _MASKED.get(key)
    if hit is not None:
        return hit
    masked = mask_cached(raw)
    with _LOCK:
        existing = _MASKED.get(key)
        if existing is not None:
            return existing
        _MASKED[key] = masked
    return masked


def stats() -> dict[str, int]:
    return {"cached_files": len(_TEXT)}


def clear() -> None:
    with _LOCK:
        _TEXT.clear()
        _BY_BASENAME.clear()
        _MASKED.clear()
        _MASKED_BY_ID.clear()
