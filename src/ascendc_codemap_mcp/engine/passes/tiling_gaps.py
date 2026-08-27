# -*- coding: utf-8 -*-
"""Unresolved tiling I/O without minting OTHER dump entities.

Ambiguous field reads/writes stay on the caller (or CodeMap meta) so
``other_count`` can be 0. They are not locate_blocking field-owner holes.
"""
from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity

_SITE_KEY = "tiling_unresolved_sites"
_META_KEY = "tiling_unresolved"


def record_unresolved_tiling(
    codemap: CodeMap,
    owner: Entity | None,
    *,
    role: str,
    file: str,
    line: int,
    expression: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    site: dict[str, Any] = {
        "role": role,
        "file": file,
        "line": int(line or 0),
        "expression": str(expression or "")[:600],
    }
    if extra:
        site.update(extra)
    if owner is not None:
        sites = list((owner.attrs or {}).get(_SITE_KEY) or [])
        sites.append(site)
        owner.attrs[_SITE_KEY] = sites
        return
    meta = dict(codemap.meta.get(_META_KEY) or {})
    sites = list(meta.get("sites") or [])
    sites.append(site)
    meta["sites"] = sites
    codemap.meta[_META_KEY] = meta
