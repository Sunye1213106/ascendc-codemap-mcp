# -*- coding: utf-8 -*-
"""Model CANN TilingKey → TilingData registrations from current source.

``REGISTER_TILING_FOR_TILINGKEY(expr, Struct)`` is an explicit source contract:
when the packed TilingKey matches ``expr`` the Kernel interprets ``tiling_data``
as ``Struct``.  This pass records the packed predicate, links source-declared
TilingKey fields covered by a constant bit mask, and links the predicate to the
registered TilingData type.  It never guesses a field when the expression
cannot be decoded.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.source_layout import KERNEL_ENTRY_NAME_RE, selected_kernel_files

_CPP_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_REGISTER_RE = re.compile(
    r"REGISTER_TILING_FOR_TILINGKEY\s*\(\s*\"(?P<expr>[^\"]+)\"\s*,\s*(?P<type>[A-Za-z_:][A-Za-z0-9_:]*)\s*\)",
    re.S,
)
_MASK_RE = re.compile(r"TILING_KEY_VAR\s*&\s*(0x[0-9A-Fa-f]+|\d+)")


def enrich_tiling_registrations(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    registrations = 0
    decoded_field_edges = 0

    for path in _candidate_files(root, architecture):
        text = read_text(path)
        for match in _REGISTER_RE.finditer(text):
            expr = match.group("expr").strip()
            type_name = match.group("type").split("::")[-1]
            line = text.count("\n", 0, match.start()) + 1
            target = _find_tiling_data(codemap, type_name)
            if target is None:
                target = codemap.upsert(
                    EntityKind.TILING_DATA,
                    type_name,
                    attrs={
                        "registered_only": True,
                        "architecture": architecture,
                        "provenance": "source_register_tiling_for_key",
                    },
                    file=_rel(root, path),
                    line=line,
                    status="confirmed",
                )

            predicate = codemap.upsert(
                EntityKind.PREDICATE,
                expr,
                eid=f"TILINGREG::{_rel(root, path)}::{line}::{type_name}",
                attrs={
                    "predicate_role": "packed_tiling_key_registration",
                    "packed_key_expression": expr,
                    "registered_tiling_data": type_name,
                    "architecture": architecture,
                    "provenance": "source_register_tiling_for_key",
                },
                file=_rel(root, path),
                line=line,
                status="confirmed",
            )
            codemap.link(
                RelationKind.SELECTS,
                predicate.id,
                target.id,
                attrs={
                    "provenance": "source_register_tiling_for_key",
                    "packed_key_expression": expr,
                    "file": _rel(root, path),
                    "line": line,
                },
                status="confirmed",
            )
            registrations += 1

            for kernel in _kernels_in_registration_tu(codemap, root, path, text):
                codemap.link(
                    RelationKind.FLOWS_TO,
                    target.id,
                    kernel.id,
                    attrs={
                        "provenance": "source_register_tiling_for_key",
                        "packed_key_expression": expr,
                        "file": _rel(root, path),
                        "line": line,
                    },
                    status="confirmed",
                )

            mask = _constant_mask(expr)
            if mask is None or mask == 0:
                continue
            for key in _keys_intersecting_mask(codemap, mask):
                codemap.link(
                    RelationKind.CONTROLS,
                    key.id,
                    predicate.id,
                    attrs={
                        "provenance": "source_tiling_key_mask",
                        "mask": hex(mask),
                        "packed_key_expression": expr,
                    },
                    status="confirmed",
                )
                decoded_field_edges += 1

    codemap.meta["source_tiling_registrations"] = registrations
    codemap.meta["source_tiling_registration_decoded_key_edges"] = decoded_field_edges
    return codemap


def _candidate_files(root: Path, architecture: str) -> list[Path]:
    return selected_kernel_files(root, architecture)


def _kernels_in_registration_tu(codemap: CodeMap, root: Path, path: Path, text: str):
    """Kernels in the same translation unit as a REGISTER_TILING_* site."""
    file_rel = _rel(root, path)
    entry_names = {
        m.group(1)
        for m in KERNEL_ENTRY_NAME_RE.finditer(text)
    }
    matched = []
    for kernel in codemap.by_kind(EntityKind.KERNEL):
        kfile = str(kernel.file or "").replace("\\", "/")
        if kernel.name in entry_names or kfile == file_rel:
            matched.append(kernel)
    if matched:
        return matched
    return list(codemap.by_kind(EntityKind.KERNEL))


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _find_tiling_data(codemap: CodeMap, name: str):
    hits = codemap.by_name(name, kind=EntityKind.TILING_DATA)
    return hits[0] if hits else None


def _constant_mask(expr: str) -> int | None:
    match = _MASK_RE.search(expr)
    if not match:
        return None
    try:
        return int(match.group(1), 0)
    except ValueError:
        return None


def _keys_intersecting_mask(codemap: CodeMap, mask: int):
    keys = sorted(
        (k for k in codemap.by_kind(EntityKind.TILING_KEY) if k.attrs.get("source_declared")),
        key=lambda k: int(k.attrs.get("decl_order") or 0),
    )
    offset = 0
    out = []
    for key in keys:
        width = int(key.attrs.get("bit_width") or 0)
        if width <= 0:
            # Unknown width makes all following offsets uncertain. Fail closed.
            break
        key.attrs.setdefault("bit_offset", offset)
        key.attrs.setdefault("bit_end", offset + width - 1)
        field_mask = ((1 << width) - 1) << offset
        if field_mask & mask:
            out.append(key)
        offset += width
    return out
