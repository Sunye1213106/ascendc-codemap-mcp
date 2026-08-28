# -*- coding: utf-8 -*-
"""Workspace size accumulation and kernel workspace ABI.

Host ``workspaceSize += expr`` and ``set_*Offset`` / ``set_*Workspace*`` are
ALLOCATES / WRITES facts. Kernel parameters whose name contains ``workspace``
are kept (they are skipped by INPUT/OUTPUT ABI bind) and joined to
``SetGlobalBuffer`` sites.
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

_SIZE_ADD_RE = re.compile(
    r"\b(?P<lhs>[A-Za-z_]\w*[Ww]orkspace\w*[Ss]ize|[A-Za-z_]\w*[Ss]ize)\s*\+=\s*(?P<rhs>[^;]+);"
)
_SETTER_RE = re.compile(
    r"(?:\.|->)\s*set_(?P<field>[A-Za-z_]\w*(?:[Oo]ffset|[Ww]orkspace\w*))\s*\(\s*(?P<rhs>[^;)]+)\)"
)
_SET_GLOBAL_RE = re.compile(
    r"\b(?P<recv>[A-Za-z_]\w*)\s*(?:\.|->)\s*SetGlobalBuffer\s*\("
)
_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
_SKIP = frozenset(
    {
        "true",
        "false",
        "sizeof",
        "static_cast",
        "uint32_t",
        "uint64_t",
        "int32_t",
        "int64_t",
        "size_t",
        "this",
        "return",
        "if",
        "else",
        "nullptr",
    }
)
_FIELD_KINDS = (
    EntityKind.TILING_FIELD,
    EntityKind.FIELD,
    EntityKind.VARIABLE,
    EntityKind.TILING_KEY,
)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def is_workspace_name(name: str) -> bool:
    compact = str(name or "").replace("_", "").lower()
    return "workspace" in compact


def is_offset_name(name: str) -> bool:
    leaf = str(name or "").replace("::", ".").rsplit(".", 1)[-1]
    low = leaf.lower()
    return low.endswith("offset") or "workspace" in low


def _unique_field(codemap: CodeMap, name: str) -> Entity | None:
    leaf = str(name or "").replace("->", ".").rsplit(".", 1)[-1]
    if not leaf or leaf in _SKIP:
        return None
    hits: dict[str, Entity] = {}
    for kind in _FIELD_KINDS:
        for ent in codemap.by_name(leaf, kind=kind):
            hits[ent.id] = ent
    if len(hits) == 1:
        return next(iter(hits.values()))
    return None


def _workspace_node(codemap: CodeMap, *, arch: str, file: str, line: int) -> Entity:
    hits = [e for e in codemap.by_kind(EntityKind.VARIABLE) if is_workspace_name(e.name)]
    if len(hits) == 1:
        return hits[0]
    named = codemap.by_name("workspace", kind=EntityKind.VARIABLE)
    if named:
        return named[0]
    return codemap.upsert(
        EntityKind.VARIABLE,
        "workspace",
        eid="SRCWS::workspace",
        attrs={"layer": "abi", "provenance": "source_workspace_abi", "architecture": arch},
        file=file,
        line=line,
        status="confirmed",
    )


def _allocates(
    codemap: CodeMap,
    src: Entity,
    dst: Entity,
    *,
    file: str,
    line: int,
    role: str,
) -> None:
    if src.id == dst.id:
        return
    extra = {"file": file, "line": line, "role": role}
    codemap.mint_candidate_relation(
        RelationKind.ALLOCATES,
        src.id,
        dst.id,
        provenance="source_workspace_abi",
        extra=extra,
    )


def enrich_workspace_abi(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    ws: Entity | None = None

    for path in selected_host_files(root, architecture=arch):
        try:
            text = read_text(path)
        except OSError:
            continue
        file = _rel(root, path)
        for match in _SIZE_ADD_RE.finditer(text):
            lhs = str(match.group("lhs") or "")
            rhs = str(match.group("rhs") or "")
            if "workspace" not in lhs.lower() and "workspace" not in rhs.lower():
                continue
            line = text.count("\n", 0, match.start()) + 1
            if ws is None:
                ws = _workspace_node(codemap, arch=arch, file=file, line=line)
            lhs_ent = _unique_field(codemap, lhs) or ws
            for ident in _IDENT_RE.findall(rhs):
                if ident in _SKIP or ident == lhs:
                    continue
                src = _unique_field(codemap, ident)
                if src is None:
                    continue
                _allocates(codemap, src, lhs_ent, file=file, line=line, role="workspace_size")
            _allocates(codemap, lhs_ent, ws, file=file, line=line, role="workspace_accum")
        for match in _SETTER_RE.finditer(text):
            field_name = str(match.group("field") or "")
            if not is_offset_name(field_name):
                continue
            line = text.count("\n", 0, match.start()) + 1
            if ws is None:
                ws = _workspace_node(codemap, arch=arch, file=file, line=line)
            field = _unique_field(codemap, field_name)
            if field is None:
                field = codemap.upsert(
                    EntityKind.TILING_FIELD,
                    field_name,
                    attrs={"provenance": "source_workspace_abi", "architecture": arch},
                    file=file,
                    line=line,
                    status="extracted",
                )
            _allocates(codemap, ws, field, file=file, line=line, role="workspace_offset")
            rhs = str(match.group("rhs") or "")
            for ident in _IDENT_RE.findall(rhs):
                if ident in _SKIP or ident == field_name:
                    continue
                src = _unique_field(codemap, ident)
                if src is None:
                    continue
                _allocates(codemap, src, field, file=file, line=line, role="offset_expr")

    for path in selected_kernel_files(root, architecture=arch):
        try:
            text = read_text(path)
        except OSError:
            continue
        file = _rel(root, path)
        for match in _SET_GLOBAL_RE.finditer(text):
            recv = str(match.group("recv") or "")
            if not is_workspace_name(recv) and recv.lower() not in {"workspace", "usrworkspace"}:
                continue
            line = text.count("\n", 0, match.start()) + 1
            if ws is None:
                ws = _workspace_node(codemap, arch=arch, file=file, line=line)
            recv_ent = _unique_field(codemap, recv) or ws
            api = None
            ops = list(codemap.by_name("SetGlobalBuffer", kind=EntityKind.OPERATION))
            if not ops:
                ops = list(codemap.by_name("SetGlobalBuffer", kind=EntityKind.METHOD))
            if ops:
                api = ops[0]
            else:
                api = codemap.upsert(
                    EntityKind.OPERATION,
                    "SetGlobalBuffer",
                    attrs={"catalog": "ascendc", "provenance": "source_workspace_abi"},
                    status="extracted",
                )
            _allocates(codemap, recv_ent, api, file=file, line=line, role="kernel_workspace")
            codemap.mint_candidate_relation(
                RelationKind.FLOWS_TO,
                recv_ent.id,
                api.id,
                provenance="source_workspace_abi",
                extra={"file": file, "line": line, "kernel_param": recv},
            )
    return codemap
