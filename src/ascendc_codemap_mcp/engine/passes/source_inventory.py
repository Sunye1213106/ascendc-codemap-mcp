# -*- coding: utf-8 -*-
"""Inventory current operator source files into the unified CodeMap.

A CodeMap should represent files even when a particular file contributes no
entity selected by a narrower semantic parser.  This prevents source coverage
from depending on incidental extraction hits and gives impact/navigation queries
stable FILE roots.  Only the requested architecture and shared operator entry
files that explicitly include/reference that architecture are admitted.
"""
from __future__ import annotations

from pathlib import Path

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.source_layout import selected_host_files, selected_kernel_files

_SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}


def inventory_source_files(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    archs = codemap.by_name(architecture, kind=EntityKind.ARCH)
    arch = archs[0] if archs else codemap.upsert(EntityKind.ARCH, architecture)

    files: dict[Path, str] = {}
    for role, directory in (
        ("api", root / "op_graph"),
    ):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES:
                files[path.resolve()] = role
    for path in selected_host_files(root, architecture):
        files[path.resolve()] = "host"
    for path in selected_kernel_files(root, architecture):
        files.setdefault(path.resolve(), "kernel")

    for path, role in sorted(files.items(), key=lambda item: item[0].as_posix()):
        rel = _rel(root, path)
        ent = codemap.upsert(
            EntityKind.FILE,
            rel,
            eid=f"FILE::{rel}",
            attrs={
                "role": role,
                "architecture": architecture if role != "api" else "shared",
                "provenance": "source_inventory",
            },
            file=rel,
            line=1,
            status="confirmed",
        )
        if role != "api":
            codemap.link(
                RelationKind.AVAILABLE_ON,
                ent.id,
                arch.id,
                attrs={"provenance": "source_inventory"},
                status="confirmed",
            )

    codemap.meta["source_inventory_file_count"] = len(files)
    codemap.meta["source_inventory_roles"] = {
        role: sum(1 for value in files.values() if value == role)
        for role in sorted(set(files.values()))
    }
    return codemap


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()
