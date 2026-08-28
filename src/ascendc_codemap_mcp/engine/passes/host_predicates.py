# -*- coding: utf-8 -*-
"""Host multi-predicate assignments: ``flag = a && b && c`` as DERIVES edges.

Does not guess operator-specific field names. Identifiers come from the RHS
clang already recorded on the write event.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.source_layout import host_ir_keeps_file

_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
_SKIP = frozenset(
    {
        "true",
        "false",
        "bool",
        "static_cast",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int",
        "int32_t",
        "int64_t",
        "nullptr",
        "this",
        "sizeof",
    }
)
_KINDS = (
    EntityKind.INPUT,
    EntityKind.TILING_FIELD,
    EntityKind.TILING_KEY,
    EntityKind.FIELD,
    EntityKind.VARIABLE,
    EntityKind.COMPILE_VAR,
    EntityKind.MACRO,
)


def _unique(codemap: CodeMap, name: str) -> Entity | None:
    leaf = str(name or "").replace("->", ".").rsplit(".", 1)[-1]
    if not leaf or leaf in _SKIP:
        return None
    hits: dict[str, Entity] = {}
    for kind in _KINDS:
        for ent in codemap.by_name(leaf, kind=kind):
            hits[ent.id] = ent
        if leaf != name:
            for ent in codemap.by_name(name, kind=kind):
                hits[ent.id] = ent
    if len(hits) == 1:
        return next(iter(hits.values()))
    return None


def enrich_host_predicates(
    codemap: CodeMap,
    operator_root: str | Path = "",
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    if host_ir is None:
        return codemap
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    events = list(getattr(host_ir, "writes", None) or []) + list(
        getattr(host_ir, "local_writes", None) or []
    )
    for ev in events:
        rhs = str(getattr(ev, "rhs", "") or "")
        if "&&" not in rhs and "||" not in rhs:
            continue
        ev_file = str(getattr(ev, "file", "") or "")
        if ev_file and not host_ir_keeps_file(ev_file, arch):
            continue
        path = str(getattr(ev, "path", "") or "").replace("->", ".")
        target = _unique(codemap, path)
        if target is None:
            continue
        idents = []
        for ident in _IDENT_RE.findall(rhs):
            if ident in _SKIP or ident == path.rsplit(".", 1)[-1]:
                continue
            src = _unique(codemap, ident)
            if src is None or src.id == target.id:
                continue
            idents.append(src)
        if len(idents) < 2:
            continue
        for src in idents:
            codemap.link(
                RelationKind.DERIVES,
                src.id,
                target.id,
                attrs={
                    "provenance": "host_predicate_and",
                    "file": ev_file,
                    "line": int(getattr(ev, "line", 0) or 0),
                    "role": "conjunct",
                },
                status="confirmed",
            )
    return codemap
