# -*- coding: utf-8 -*-
"""CANN utility / proposal APIs loaded from installed headers.

CeilDiv, GetSortLen, ArithProgression (and the current Arange spelling) are
declared in interface headers, not the small YAML catalog. Same pattern as
``ascendc_vf``: scan CANN, do not special-case operators.
"""
from __future__ import annotations

import re
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path

_FN_RE = re.compile(
    r"__aicore__\s+inline\s+[^\n;{]{0,240}?\b([A-Za-z_]\w*)\s*\("
)
# Keep known spellings even if a given unpack omits the header.
_ALWAYS = frozenset({"CeilDiv", "GetSortLen", "ArithProgression", "Arange"})


def _header_roots(cann: Path) -> list[Path]:
    from ascendc_codemap_mcp.engine.paths import resolve_cann_relative

    rels = (
        "cann-asc-devkit/x86_64-linux/asc/include/basic_api",
        "cann-asc-devkit/x86_64-linux/asc/include/interface",
        "cann-asc-devkit/x86_64-linux/asc/include/adv_api",
        "cann-asc-devkit/x86_64-linux/asc/include/tiling",
        "cann-asc-devkit/x86_64-linux/ascendc/include/highlevel_api",
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for rel in rels:
        d = resolve_cann_relative(cann, rel)
        if d.is_dir() and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _scan_file(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    names = {n for n in _FN_RE.findall(text) if n and n[0].isupper()}
    for extra in _ALWAYS:
        if re.search(rf"\b{extra}\s*\(", text):
            names.add(extra)
    return names


_SCAN_PATTERNS = (
    "kernel_operator_*_intf.h",
    "*arithprogression*.h",
    "*proposal*.h",
    "*math*.h",
    "*utils*.h",
)


def _scan_dir(folder: Path) -> set[str]:
    """Names declared in the interesting headers under one CANN include root.

    Walk once and match in memory. Globbing per pattern re-walked the whole
    CANN include tree five times over to read the same handful of headers, and
    these trees are thousands of files deep. ``fnmatch`` is used rather than
    ``fnmatchcase`` so matching stays case-insensitive on Windows and sensitive
    elsewhere, which is what ``rglob`` did.
    """
    names: set[str] = set()
    for path in folder.rglob("*"):
        if not any(fnmatch(path.name, pattern) for pattern in _SCAN_PATTERNS):
            continue
        if path.is_file():
            names.update(_scan_file(path))
    return names


@lru_cache(maxsize=1)
def cann_util_api_names() -> frozenset[str]:
    from ascendc_codemap_mcp.engine.paths import cann_root

    names: set[str] = set(_ALWAYS)
    root = cann_root()
    if root is not None:
        for folder in _header_roots(root):
            names.update(_scan_dir(folder))
    return frozenset(n for n in names if n and n[0].isupper())


def is_cann_util_api(callee: str) -> bool:
    short = str(callee or "").split("::")[-1]
    if "<" in short:
        short = short.split("<", 1)[0]
    return short in cann_util_api_names()
