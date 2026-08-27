# -*- coding: utf-8 -*-
"""TilingDataBinding: per KernelEntry / CompileVariant, not THE operator type.

Identity comes from use-sites (GET_TILING_DATA*, registration macros,
reinterpret_cast of a tiling pointer). Names are candidate_score only.
No extra source walk: callers pass already-read text.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.evidence import SOURCE_DSL, TRUST_DERIVED, mint_payload
from ascendc_codemap_mcp.engine.ir.relation import Relation, RelationKind
from ascendc_codemap_mcp.engine.source_layout import GLOBAL_KERNEL_RE

_AICORE_FN_RE = re.compile(
    r"(?:inline\s+)?__aicore__\s+(?:inline\s+)?(?:void|[A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\("
)

KEY_CHAIN_PRODUCER = "producer"
KEY_CHAIN_PACKING = "packing"
KEY_CHAIN_TRANSPORT = "transport"
KEY_CHAIN_CONSUMER = "consumer"
KEY_CHAIN_DISPATCH = "dispatch"


def kernels_for_use_site(
    codemap: CodeMap, path: Path, text: str, root: Path
) -> list[Entity]:
    """Kernels that consume a tiling use-site in this file. Never all-kernels."""
    kernels = list(codemap.by_kind(EntityKind.KERNEL))
    if not kernels:
        return []
    entry_names = {m.group("name") for m in GLOBAL_KERNEL_RE.finditer(text or "")}
    file_rel = str(path).replace("\\", "/")
    try:
        file_rel = str(path.resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        file_rel = str(path).replace("\\", "/")
    same_file = [
        k
        for k in kernels
        if k.name in entry_names or str(k.file or "").replace("\\", "/") == file_rel
    ]
    if same_file:
        return same_file
    aicore_names = _AICORE_FN_RE.findall(text or "")
    if not aicore_names:
        return []
    scored: list[tuple[int, Entity]] = []
    for kernel in kernels:
        score = 0
        for fn in aicore_names:
            if fn.startswith(kernel.name) or kernel.name.startswith(fn):
                score = max(score, len(kernel.name))
        if score:
            scored.append((score, kernel))
    if not scored:
        return []
    best = max(score for score, _k in scored)
    return [k for score, k in scored if score == best]


def link_tiling_data_binding(
    codemap: CodeMap,
    type_ent: Entity,
    kernel_ent: Entity,
    *,
    provenance: str,
    file: str = "",
    extra: dict[str, Any] | None = None,
) -> Relation:
    """Mint one KernelEntry→type consumer binding. Does not cartesian-link."""
    payload = mint_payload(
        provenance=provenance,
        source=SOURCE_DSL,
        trust=TRUST_DERIVED,
        extra={
            "file": file,
            "binding_role": "consumer",
            "kernel_entry": kernel_ent.name,
            "compile_variant": str(codemap.meta.get("build_context_id") or ""),
            "tiling_data_type": type_ent.name,
            **dict(extra or {}),
        },
    )
    bound = list(type_ent.attrs.get("bound_kernel_entries") or [])
    if kernel_ent.name not in bound:
        bound.append(kernel_ent.name)
    type_ent.attrs["bound_kernel_entries"] = bound
    type_ent.attrs["selected_by_kernel_entry"] = True
    return codemap.link(
        RelationKind.FLOWS_TO,
        type_ent.id,
        kernel_ent.id,
        attrs=payload,
        status="confirmed",
    )


def summarize_tiling_data_bindings(codemap: CodeMap) -> list[dict[str, Any]]:
    """Linear scan of in-memory edges. No source I/O."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for rel in codemap.relations.values():
        if rel.kind_name() != RelationKind.FLOWS_TO.value:
            continue
        if str(rel.attrs.get("binding_role") or "") != "consumer":
            continue
        key = (str(rel.attrs.get("kernel_entry") or ""), str(rel.attrs.get("tiling_data_type") or ""))
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "kernel_entry": key[0],
                "type": key[1],
                "compile_variant": str(rel.attrs.get("compile_variant") or ""),
                "file": str(rel.attrs.get("file") or ""),
                "provenance": str(rel.attrs.get("provenance") or ""),
            }
        )
    rows.sort(key=lambda r: (r["kernel_entry"], r["type"], r["file"]))
    return rows
