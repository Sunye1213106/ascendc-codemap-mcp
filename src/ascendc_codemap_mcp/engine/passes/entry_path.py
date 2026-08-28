# -*- coding: utf-8 -*-
"""Entry-path facts from Host bailout / early-return premises.

A tiling/kernel entry is any function that WRITES a TILING_FIELD (or LAUNCHES
a kernel). Bailout PathConds on that function are the guard; writes before the
successful return are the then-body. Not a function-name heuristic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.source_layout import host_ir_keeps_file


def _pc_field(pc: Any, name: str, default: Any = "") -> Any:
    if isinstance(pc, dict):
        return pc.get(name, default)
    return getattr(pc, name, default)


def _is_bailout(pc: Any) -> bool:
    kind = str(_pc_field(pc, "kind", "") or "")
    if kind == "bailout":
        return True
    flag = getattr(pc, "is_bailout", None)
    if callable(flag):
        return bool(flag())
    return bool(flag)


def _writer_functions(codemap: CodeMap) -> dict[str, Entity]:
    out: dict[str, Entity] = {}
    tiling = {e.id for e in codemap.by_kind(EntityKind.TILING_FIELD)}
    kernels = {e.id for e in codemap.by_kind(EntityKind.KERNEL)}
    for rel in codemap.relations.values():
        kind = rel.kind_name()
        src = codemap.entities.get(rel.src)
        if src is None:
            continue
        if src.kind_name() not in {EntityKind.FUNCTION.value, EntityKind.METHOD.value}:
            continue
        if kind == RelationKind.WRITES.value and rel.dst in tiling:
            out[src.id] = src
        elif kind == RelationKind.LAUNCHES.value and rel.dst in kernels:
            out[src.id] = src
    return out


def _predicate(
    codemap: CodeMap,
    *,
    text: str,
    file: str,
    line: int,
    function: str,
    arch: str,
    role: str,
) -> Entity:
    eid = f"ENTRY::{arch}::{file}::{line}::{function}"
    return codemap.upsert(
        EntityKind.PREDICATE,
        (text or "entry_path")[:160],
        eid=eid,
        attrs={
            "predicate_role": "entry_path",
            "entry_role": role,
            "function": function,
            "architecture": arch,
            "provenance": "source_entry_path",
        },
        file=file,
        line=line,
        status="confirmed",
    )


def enrich_entry_paths(
    codemap: CodeMap,
    operator_root: str | Path = "",
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    writers = _writer_functions(codemap)
    writer_names = {e.name.split("::")[-1]: e for e in writers.values()}

    premises: list[Any] = []
    if host_ir is not None and hasattr(host_ir, "legality_premises"):
        try:
            premises = list(host_ir.legality_premises() or [])
        except Exception:  # noqa: BLE001
            premises = []
    for item in premises:
        if not isinstance(item, (tuple, list)) or len(item) < 4:
            continue
        text, fn_name, file, line = item[0], item[1], item[2], item[3]
        leaf = str(fn_name or "").split("::")[-1]
        if leaf not in writer_names and str(fn_name or "") not in writer_names:
            continue
        if file and not host_ir_keeps_file(str(file), arch):
            continue
        fn = writer_names.get(leaf) or writer_names.get(str(fn_name or ""))
        pred = _predicate(
            codemap,
            text=str(text or ""),
            file=str(file or ""),
            line=int(line or 0),
            function=str(fn_name or ""),
            arch=arch,
            role="bailout",
        )
        if fn is not None:
            codemap.link(
                RelationKind.CONTROLS,
                pred.id,
                fn.id,
                attrs={
                    "provenance": "source_entry_path",
                    "file": str(file or ""),
                    "line": int(line or 0),
                    "role": "entry_bailout",
                },
                status="confirmed",
            )

    if host_ir is None:
        return codemap
    for site in getattr(host_ir, "call_sites", None) or []:
        file = str(getattr(site, "file", "") or "")
        if file and not host_ir_keeps_file(file, arch):
            continue
        caller = str(getattr(site, "caller", "") or "")
        leaf = caller.split("::")[-1]
        fn = writer_names.get(leaf) or writer_names.get(caller)
        if fn is None:
            continue
        for pc in getattr(site, "path_conditions", None) or []:
            if not _is_bailout(pc):
                continue
            text = str(_pc_field(pc, "text", "") or "")
            if not text:
                continue
            pred = _predicate(
                codemap,
                text=text,
                file=str(_pc_field(pc, "file", "") or file),
                line=int(_pc_field(pc, "line", 0) or getattr(site, "line", 0) or 0),
                function=caller,
                arch=arch,
                role="bailout",
            )
            codemap.link(
                RelationKind.CONTROLS,
                pred.id,
                fn.id,
                attrs={
                    "provenance": "source_entry_path",
                    "role": "entry_bailout",
                    "file": pred.file,
                    "line": pred.line_start,
                },
                status="confirmed",
            )
    return codemap
