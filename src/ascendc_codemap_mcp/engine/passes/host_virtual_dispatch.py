# -*- coding: utf-8 -*-
"""Record host virtual families that clang collapsed by short name.

``_remember_func`` keeps one FuncRecord per spelling. The empty
``virtual GetSparseUnpadBlockInfo() {}`` loses to the varlen override, and
``DoSparse:1110`` is stored as a direct CALLS to that override. This pass
re-attaches the missing base, the override, and the inheritance edge so the
next snapshot does not depend on query-time reconstruction.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.query.virtual_dispatch import (
    build_family,
    leaf_name,
)
from ascendc_codemap_mcp.engine.source_layout import host_ir_keeps_file

PROVENANCE = "source_host_virtual_dispatch"
_CALL_KINDS = {RelationKind.CALLS.value, RelationKind.CALLS_UNDER_GUARD.value}
_HOST_KINDS = {EntityKind.FUNCTION.value, EntityKind.METHOD.value}


def _short(name: str) -> str:
    return str(name or "").replace(".", "::").split("::")[-1].strip()


def _decl_window(root: Path, file: str, line: int) -> str:
    if not file or line <= 0:
        return ""
    try:
        from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

        text = read_text(root / file) if not Path(file).is_absolute() else read_text(file)
    except OSError:
        return ""
    rows = text.splitlines()
    start = max(0, int(line) - 1)
    return "\n".join(rows[start : start + 3])


def _type_owner_by_file(codemap: CodeMap) -> dict[str, str]:
    by_file: dict[str, list[str]] = defaultdict(list)
    for ent in codemap.entities.values():
        if ent.kind_name() != EntityKind.TYPE.value:
            continue
        file = str(ent.file or "").replace("\\", "/")
        name = _short(ent.name)
        if not file or not name:
            continue
        kind = str(ent.attrs.get("cpp_kind") or "").lower()
        if kind and kind not in {"class", "struct", ""}:
            continue
        by_file[file].append(name)
    return {file: names[0] for file, names in by_file.items() if len(names) == 1}


def _callable_members(codemap: CodeMap) -> dict[str, list[Entity]]:
    groups: dict[str, list[Entity]] = defaultdict(list)
    for ent in codemap.entities.values():
        if ent.kind_name() not in _HOST_KINDS:
            continue
        if str(ent.attrs.get("layer") or "host") not in {"", "host"}:
            continue
        leaf = leaf_name(ent.name)
        if not leaf or not ent.file:
            continue
        groups[leaf].append(ent)
    return groups


def _family_for(
    leaf: str,
    ents: list[Entity],
    *,
    root: Path,
    owners_by_file: dict[str, str],
) -> dict[str, Any] | None:
    members: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    for ent in ents:
        file = str(ent.file or "").replace("\\", "/")
        line = int(ent.line_start or 0)
        text = _decl_window(root, file, line)
        texts[f"{file}:{line}"] = text
        members.append(
            {
                "id": ent.id,
                "kind": ent.kind_name(),
                "name": ent.name,
                "file": file,
                "line": line,
                "line_start": line,
                "line_end": int(ent.line_end or 0),
                "attrs": dict(ent.attrs or {}),
                "text": text,
            }
        )
    return build_family(members, texts=texts, owners_by_file=owners_by_file)


def _stamp_entity(ent: Entity, row: dict[str, Any]) -> None:
    if row.get("virtual"):
        ent.attrs["is_virtual"] = True
    if row.get("override"):
        ent.attrs["is_override"] = True
    if row.get("empty"):
        ent.attrs["empty_body"] = True
    owner = str(row.get("owner") or "")
    if owner and not ent.attrs.get("owner"):
        ent.attrs["owner"] = owner
    ent.attrs.setdefault("virtual_role", "override" if row.get("override") or row.get("has_body") else "base")


def _mint_specializes(codemap: CodeMap, host_ir: Any, architecture: str) -> int:
    if host_ir is None:
        return 0
    types = {
        _short(e.name): e
        for e in codemap.entities.values()
        if e.kind_name() == EntityKind.TYPE.value and _short(e.name)
    }
    minted = 0
    for bd in getattr(host_ir, "base_decls", ()) or ():
        derived = _short(getattr(bd, "derived_name", "") or "")
        base = _short(getattr(bd, "base_name", "") or "")
        file = str(getattr(bd, "file", "") or "")
        if not derived or not base or derived == base:
            continue
        if file and not host_ir_keeps_file(file, architecture):
            continue
        src = types.get(derived)
        dst = types.get(base)
        if src is None or dst is None:
            continue
        rel = codemap.link(
            RelationKind.SPECIALIZES,
            src.id,
            dst.id,
            attrs={
                "provenance": PROVENANCE,
                "via": "inherits",
                "file": file,
                "line": int(getattr(bd, "line", 0) or 0),
            },
            status="confirmed",
        )
        if rel is not None:
            minted += 1
    return minted


def enrich_host_virtual_dispatch(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,  # noqa: ARG001 — pipeline needs_irs
    **_extra: Any,
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    owners_by_file = _type_owner_by_file(codemap)
    families: dict[str, dict[str, Any]] = {}
    for leaf, ents in _callable_members(codemap).items():
        if len(ents) < 2:
            continue
        family = _family_for(leaf, ents, root=root, owners_by_file=owners_by_file)
        if family is None:
            continue
        families[leaf] = family
        by_id = {e.id: e for e in ents}
        for row in list(family.get("base") or []) + list(family.get("overrides") or []):
            ent = by_id.get(str(row.get("id") or ""))
            if ent is not None:
                _stamp_entity(ent, row)
                ent.attrs["virtual_dispatch"] = {
                    "leaf": family["leaf"],
                    "base": family.get("base") or [],
                    "overrides": family.get("overrides") or [],
                }

    annotated = 0
    for rel in list(codemap.relations.values()):
        if rel.kind_name() not in _CALL_KINDS:
            continue
        dst = codemap.entities.get(rel.dst)
        if dst is None:
            continue
        family = families.get(leaf_name(dst.name))
        if family is None:
            continue
        rel.attrs["virtual"] = True
        rel.attrs["virtual_dispatch"] = {
            "leaf": family["leaf"],
            "base": family.get("base") or [],
            "overrides": family.get("overrides") or [],
        }
        annotated += 1

    specializes = _mint_specializes(codemap, host_ir, arch)
    codemap.meta["host_virtual_dispatch"] = {
        "families": len(families),
        "calls_annotated": annotated,
        "specializes": specializes,
    }
    return codemap
