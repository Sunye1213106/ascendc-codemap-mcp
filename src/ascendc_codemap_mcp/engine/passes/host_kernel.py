# -*- coding: utf-8 -*-
"""Host/Kernel finalisation pass.

This pass may add architecture availability facts, but it must never invent
TilingKey → Template/Kernel relations. Selection/launch/instantiation edges are
semantic facts and must come from the frontend or a deterministic binding pass
with evidence.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

# Relations that can carry a real semantic dependency from an API/host value to
# a kernel. Deliberately exclude AVAILABLE_ON / ACTIVE_UNDER and other metadata
# edges: those do not prove selection.
_SEMANTIC_FLOW = {
    RelationKind.DERIVES.value,
    RelationKind.FLOWS_TO.value,
    RelationKind.CONTROLS.value,
    RelationKind.BINDS.value,
    RelationKind.SELECTS.value,
    RelationKind.INSTANTIATES.value,
    RelationKind.LAUNCHES.value,
    RelationKind.READS.value,
    RelationKind.WRITES.value,
}


def evidence_backed_host_kernel_path_exists(codemap: CodeMap) -> bool:
    """Return True only when graph relations form an actual input→kernel path."""
    starts = codemap.by_kind(EntityKind.INPUT)
    kernels = {e.id for e in codemap.by_kind(EntityKind.KERNEL)}
    if not starts or not kernels:
        return False

    adj: dict[str, list[str]] = defaultdict(list)
    for rel in codemap.relations.values():
        if rel.kind_name() in _SEMANTIC_FLOW:
            adj[rel.src].append(rel.dst)

    q: deque[str] = deque(e.id for e in starts)
    seen = set(q)
    while q:
        cur = q.popleft()
        if cur in kernels:
            return True
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def run(codemap: CodeMap, *, context: dict[str, Any] | None = None) -> CodeMap:
    del context  # reserved
    kernels = codemap.by_kind(EntityKind.KERNEL)
    instances = codemap.by_kind(EntityKind.TEMPLATE_INSTANCE)
    archs = codemap.by_kind(EntityKind.ARCH)

    # Architecture membership is safe to materialise from the active build
    # variant. It does not imply semantic selection.
    for inst in instances:
        for arch in archs:
            codemap.link(
                RelationKind.AVAILABLE_ON,
                inst.id,
                arch.id,
                attrs={"provenance": "build_variant"},
                status="derived",
            )
    for kernel in kernels:
        for arch in archs:
            codemap.link(
                RelationKind.AVAILABLE_ON,
                kernel.id,
                arch.id,
                attrs={"provenance": "build_variant"},
                status="derived",
            )

    codemap.meta["host_kernel_bind_pass"] = "v2-sound"
    codemap.meta["has_evidence_backed_host_kernel_path"] = (
        evidence_backed_host_kernel_path_exists(codemap)
    )
    return codemap
