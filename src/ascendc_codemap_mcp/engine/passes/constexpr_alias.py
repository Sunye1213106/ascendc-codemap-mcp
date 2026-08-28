# -*- coding: utf-8 -*-
"""Compile-time aliases: ``static constexpr T Name = (T)Other`` and object macros.

When a template argument or tiling dim is consumed under a different spelling,
the graph must keep identity. Occupancy ALIASES (shared integer sets) is a
different fact and is not reused here.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.source_layout import selected_host_files, selected_kernel_files

_ALIAS_RE = re.compile(
    r"\b(?:constexpr\s+static|static\s+constexpr|constexpr)\s+"
    r"(?:const\s+)?"
    r"(?:[\w:<>,\s*&]+?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*=\s*"
    r"(?:static_cast\s*<[^>]+>\s*\(|\(\s*[\w:]+(?:\s*<[^>]+>)?\s*\)\s*)?"
    r"(?P<src>[A-Za-z_]\w*)\s*\)?\s*;",
)
_DEFINE_ALIAS_RE = re.compile(
    r"^\s*#define\s+(?P<name>[A-Za-z_]\w*)\s+(?P<src>[A-Za-z_]\w*)\s*$",
    re.M,
)
_SOURCE_KINDS = (
    EntityKind.TEMPLATE_ARG,
    EntityKind.TILING_KEY,
    EntityKind.COMPILE_VAR,
    EntityKind.MACRO,
    EntityKind.FIELD,
    EntityKind.VARIABLE,
)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _resolve_src(codemap: CodeMap, name: str) -> Entity | None:
    hits: dict[str, Entity] = {}
    for kind in _SOURCE_KINDS:
        for ent in codemap.by_name(name, kind=kind):
            hits[ent.id] = ent
    if len(hits) == 1:
        return next(iter(hits.values()))
    preferred = [
        e
        for e in hits.values()
        if e.kind_name() in {EntityKind.TEMPLATE_ARG.value, EntityKind.TILING_KEY.value}
    ]
    if len(preferred) == 1:
        return preferred[0]
    return None


def enrich_constexpr_aliases(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    files = list(selected_kernel_files(root, architecture=arch)) + list(
        selected_host_files(root, architecture=arch)
    )
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        try:
            text = read_text(path)
        except OSError:
            continue
        file = _rel(root, path)
        for rx, provenance in (
            (_ALIAS_RE, "source_constexpr_alias"),
            (_DEFINE_ALIAS_RE, "source_macro_alias"),
        ):
            for match in rx.finditer(text):
                name = str(match.group("name") or "").strip()
                src_name = str(match.group("src") or "").strip()
                if not name or not src_name or name == src_name:
                    continue
                key = (file, name, src_name)
                if key in seen:
                    continue
                seen.add(key)
                line = text.count("\n", 0, match.start()) + 1
                dst = codemap.upsert(
                    EntityKind.COMPILE_VAR,
                    name,
                    eid=f"SRCCONST::{file}::{name}",
                    attrs={
                        "value_expr": src_name,
                        "alias_of": src_name,
                        "provenance": provenance,
                        "architecture": arch,
                    },
                    file=file,
                    line=line,
                    status="confirmed",
                )
                src = _resolve_src(codemap, src_name)
                if src is None:
                    src = codemap.upsert(
                        EntityKind.COMPILE_VAR,
                        src_name,
                        eid=f"SRCCONST::{file}::{src_name}",
                        attrs={"provenance": provenance, "architecture": arch},
                        file=file,
                        line=line,
                        status="extracted",
                    )
                extra = {"file": file, "line": line, "via": provenance}
                codemap.mint_candidate_relation(
                    RelationKind.MATERIALIZES_AS,
                    src.id,
                    dst.id,
                    provenance=provenance,
                    extra=extra,
                )
    return codemap
