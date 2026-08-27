# -*- coding: utf-8 -*-
"""Neutralize Bisheng postfix attributes that vanilla clang cannot parse.

Bisheng writes ``__forceinline__ [host, aicore]`` / ``__forceinline__[aicore]``.
libclang treats the single brackets as a C++17 decomposition declaration and
then reports ``decomposition declaration template not supported`` plus a
cascade of unknown ``T`` / ``U`` in every TLA/CuTe header that uses
``HOST_DEVICE``.

The prelude can close well-known Catlass include guards, but operators also
``#define HOST_DEVICE`` in their own headers after the prelude. Those later
defines win. The only generic fix is to rewrite the bracket form out of the
operator tree before libclang sees it (``unsaved_files``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

_CPP_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".cuh", ".inc", ".cpp", ".cc", ".cxx", ".c"}
_SKIP_DIR_NAMES = {
    ".git",
    ".svn",
    ".ascendc-codemap",
    "__pycache__",
    "bin",
    "lib",
    "build",
    "output",
    "tests",
    "test",
}

# ``[host, aicore]``, ``[aicore]``, ``[host]``, ``[device]`` (any order / spacing).
_BISHENG_BRACKET_ATTR_RE = re.compile(
    r"\[\s*(?:host|device|aicore)(?:\s*,\s*(?:host|device|aicore))*\s*\]",
    re.IGNORECASE,
)

_UNSAVED_CACHE: dict[str, list[tuple[str, str]]] = {}


def strip_bisheng_bracket_attrs(text: str) -> str:
    """Remove Bisheng ``[host, aicore]``-style postfix attributes from source."""
    return _BISHENG_BRACKET_ATTR_RE.sub("", text or "")


def has_bisheng_bracket_attrs(text: str) -> bool:
    return bool(_BISHENG_BRACKET_ATTR_RE.search(text or ""))


def _skip_dir(name: str) -> bool:
    return name.lower() in _SKIP_DIR_NAMES or name.startswith(".")


def _path_spellings(path: Path) -> list[str]:
    """libclang must see the same spelling it will open on disk."""
    out: list[str] = []
    seen: set[str] = set()
    candidates = [path]
    try:
        candidates.append(path.resolve())
    except OSError:
        pass
    for cand in candidates:
        for spelling in (str(cand), cand.as_posix(), str(cand).replace("\\", "/")):
            if spelling and spelling not in seen:
                seen.add(spelling)
                out.append(spelling)
    return out


def collect_rewritten_sources(roots: Iterable[str | Path]) -> list[tuple[str, str]]:
    """Return ``(path, rewritten_text)`` for every tree file with bracket attrs."""
    pairs: list[tuple[str, str]] = []
    seen_files: set[str] = set()
    for raw in roots:
        root = Path(str(raw or ""))
        try:
            if not root.is_dir():
                continue
        except OSError:
            continue
        try:
            walker = os.walk(root)
        except OSError:
            continue
        for dirpath, dirnames, filenames in walker:
            dirnames[:] = [d for d in dirnames if not _skip_dir(d)]
            for fn in filenames:
                path = Path(dirpath) / fn
                if path.suffix.lower() not in _CPP_SUFFIXES:
                    continue
                key = str(path).replace("\\", "/").lower()
                if key in seen_files:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if not has_bisheng_bracket_attrs(text):
                    continue
                seen_files.add(key)
                rewritten = strip_bisheng_bracket_attrs(text)
                for spelling in _path_spellings(path):
                    pairs.append((spelling, rewritten))
    return pairs


def kernel_unsaved_files(op_dir: str | Path | None) -> list[tuple[str, str]]:
    """Cached ``unsaved_files`` entries for one operator checkout."""
    if not op_dir:
        return []
    try:
        key = str(Path(op_dir).resolve())
    except OSError:
        key = str(op_dir)
    cached = _UNSAVED_CACHE.get(key)
    if cached is not None:
        return cached
    roots = [Path(op_dir)]
    try:
        parent = Path(op_dir).resolve().parent
        common = parent / "common"
        if common.is_dir():
            roots.append(common)
    except OSError:
        pass
    pairs = collect_rewritten_sources(roots)
    _UNSAVED_CACHE[key] = pairs
    return pairs


def reset_unsaved_cache() -> None:
    _UNSAVED_CACHE.clear()


def parse_unsaved_kwargs(op_dir: str | Path | None, *, side: str = "kernel") -> dict:
    """Kwargs to splice into ``Index.parse`` for kernel TUs."""
    if str(side or "").strip().lower() != "kernel":
        return {}
    files = kernel_unsaved_files(op_dir)
    if not files:
        return {}
    return {"unsaved_files": files}
