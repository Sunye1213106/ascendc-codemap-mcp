# -*- coding: utf-8 -*-
"""Bind clang MACRO_INSTANTIATION sites to already-inventoried operator MACROs."""
from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap, _unique_named
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

_PARENT_KINDS: dict[str, tuple[EntityKind, ...]] = {
    "FUNCTION_DECL": (EntityKind.FUNCTION, EntityKind.METHOD, EntityKind.KERNEL),
    "FUNCTION_TEMPLATE": (EntityKind.FUNCTION, EntityKind.METHOD, EntityKind.KERNEL),
    "CXX_METHOD": (EntityKind.METHOD, EntityKind.FUNCTION),
    "CONSTRUCTOR": (EntityKind.METHOD, EntityKind.FUNCTION),
    "STRUCT_DECL": (EntityKind.TYPE,),
    "CLASS_DECL": (EntityKind.TYPE,),
    "CLASS_TEMPLATE": (EntityKind.TYPE,),
    "UNION_DECL": (EntityKind.TYPE,),
}


def bind_clang_macro_uses(codemap: CodeMap, kernel_ir: Any) -> None:
    """EXPANDS_TO from an existing MACRO to the clang parent FUNCTION/METHOD/TYPE.

    Does not mint MACRO entities. CANN-header instantiations stay off the graph
    unless the operator already inventoried that name.
    """
    if kernel_ir is None:
        return
    for use in getattr(kernel_ir, "macro_uses", None) or []:
        if isinstance(use, dict):
            name = str(use.get("name") or "")
            file = str(use.get("file") or "")
            line = int(use.get("line") or 0)
            parent_name = str(use.get("parent_name") or "")
            parent_kind = str(use.get("parent_kind") or "")
        else:
            name = str(getattr(use, "name", "") or "")
            file = str(getattr(use, "file", "") or "")
            line = int(getattr(use, "line", 0) or 0)
            parent_name = str(getattr(use, "parent_name", "") or "")
            parent_kind = str(getattr(use, "parent_kind", "") or "")
        if not name:
            continue
        macro = _macro_for_use(codemap, name, file)
        if macro is None:
            continue
        target = None
        if parent_name:
            kinds = _PARENT_KINDS.get(parent_kind) or (
                EntityKind.FUNCTION,
                EntityKind.METHOD,
                EntityKind.TYPE,
            )
            target = _unique_named(codemap, parent_name, kinds)
            if target is None:
                target = _named_in_file(codemap, parent_name, file, kinds)
        if target is None:
            target = _file_entity(codemap, file)
        if target is None:
            continue
        rel = codemap.link(
            RelationKind.EXPANDS_TO,
            macro.id,
            target.id,
            attrs={
                "provenance": "source_clang_macro_instantiation",
                "file": file,
                "line": line,
            },
            status="confirmed",
        )
        rel.attrs["provenance"] = "source_clang_macro_instantiation"


def run(
    codemap: CodeMap,
    operator_root: Any = None,
    *,
    architecture: str = "",
    kernel_ir: Any = None,
    host_ir: Any = None,
    **_kwargs: Any,
) -> CodeMap:
    del operator_root, architecture, host_ir
    bind_clang_macro_uses(codemap, kernel_ir)
    return codemap


def _norm_file(path: str) -> str:
    text = str(path or "").replace("\\", "/")
    if not text:
        return ""
    drive = ""
    rest = text
    if len(text) >= 2 and text[1] == ":":
        drive = text[:2]
        rest = text[2:]
        if rest.startswith("/"):
            rest = rest[1:]
            drive += "/"
    parts: list[str] = []
    for part in rest.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    body = "/".join(parts)
    if drive.endswith("/"):
        return drive + body
    if drive:
        return f"{drive}/{body}" if body else drive
    return body


def _same_source_file(left: str, right: str) -> bool:
    a = _norm_file(left)
    b = _norm_file(right)
    if not a or not b:
        return False
    if a == b:
        return True
    return a.endswith("/" + b) or b.endswith("/" + a)


def _macro_for_use(codemap: CodeMap, name: str, file: str) -> Entity | None:
    hits = list(codemap.by_name(name, kind=EntityKind.MACRO))
    if not hits:
        return None
    if file:
        same = [ent for ent in hits if _same_source_file(ent.file, file)]
        if len(same) == 1:
            return same[0]
        if len(same) > 1:
            return None
    if len(hits) == 1:
        return hits[0]
    return None


def _file_entity(codemap: CodeMap, file: str) -> Entity | None:
    hits = [
        ent
        for ent in codemap.by_kind(EntityKind.FILE)
        if _same_source_file(ent.file or ent.name, file)
    ]
    return hits[0] if len(hits) == 1 else None


def _named_in_file(
    codemap: CodeMap,
    name: str,
    file: str,
    kinds: tuple[EntityKind, ...],
) -> Entity | None:
    hits: list[Entity] = []
    for kind in kinds:
        for ent in codemap.by_name(name, kind=kind):
            if file and not _same_source_file(ent.file, file):
                continue
            hits.append(ent)
    if len(hits) == 1:
        return hits[0]
    return None
