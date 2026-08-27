# -*- coding: utf-8 -*-
"""Semantic gap helpers — only locate-relevant residuals reach the agent."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

_LOCATE_REASONS = frozenset({"field_owner_unknown", "field_owner_ambiguous"})
_SETTLED_ROOTS = frozenset({"REACHED", "PROJECT", "BUILTIN"})
_HOST_LEAF_PROV = "source_host_unresolved_dependency"

# Catalog calls cannbot actually locates. Unproven ones are catalog_unproven,
# not a dump of every partial entity.
_KERNEL_API_NEEDLES = frozenset(
    {
        "EnQue",
        "DeQue",
        "InitBuffer",
        "DataCopy",
        "DataCopyPad",
        "DataCopyScatter",
        "Cast",
        "SetFlag",
        "WaitFlag",
        "CrossCoreSetFlag",
        "CrossCoreWaitFlag",
        "PipeBarrier",
        "DataSyncBarrier",
        "SyncAll",
        "WaitPreBlock",
        "NotifyNextBlock",
        "SetNextTaskStart",
        "WaitPreTaskEnd",
        "LocalMemBar",
        "AllocEventID",
        "ReleaseEventID",
        "Copy",
        "LoadData",
        "LoadAlign",
        "SetGlobalBuffer",
        "AllocTensor",
        "FreeTensor",
        "LocalMemBar",
        "PopStackBuffer",
        "InitShareBufStart",
        "InitShareBufEnd",
        "CreateMask",
        "StoreAlign",
        "StoreUnAlign",
        "UpdateMask",
        "DataCopyScatter",
    }
)


def classify_gap_entity(ent: Entity) -> str | None:
    """Bucket an entity for unresolved.yaml, or None if it is not a gap.

    host_runtime_leaf — Host def-use runtime leaves (not locate failure).
    locate_blocking — field owner missing, unknown status, audit-class holes.
    catalog_unproven — kernel API / storage wrapper still UNRESOLVED.
    """
    attrs = ent.attrs or {}
    eid = str(ent.id or "")
    prov = str(attrs.get("provenance") or "")
    if eid.startswith("HOSTUNRESOLVED::") or prov == _HOST_LEAF_PROV:
        return "host_runtime_leaf"
    reason = str(attrs.get("reason") or "")
    if reason in _LOCATE_REASONS:
        return "locate_blocking"
    rs = str(attrs.get("root_status") or "")
    if rs in _SETTLED_ROOTS:
        return None
    kind = ent.kind_name()
    callee = str(attrs.get("callee") or ent.name or "")
    if kind == EntityKind.OPERATION.value and rs == "UNRESOLVED":
        if callee in _KERNEL_API_NEEDLES:
            return "catalog_unproven"
        if str(ent.status).lower() in {"partial", "unresolved"}:
            return "catalog_unproven"
        return None
    if (
        kind == EntityKind.TYPE.value
        and rs == "UNRESOLVED"
        and str(attrs.get("role") or "") == "storage_wrapper_type"
    ):
        return "catalog_unproven"
    status = str(ent.status).lower()
    if status in {"unresolved", "not_extracted", "unknown"}:
        return "locate_blocking"
    return None


def _entity_gap_row(ent: Entity, bucket: str) -> dict[str, Any]:
    return {
        "code": "entity_status" if bucket != "host_runtime_leaf" else "host_runtime_leaf",
        "entity_id": ent.id,
        "name": ent.name,
        "status": ent.status,
        "reason": ent.attrs.get("reason"),
        "resolution_blocker": ent.attrs.get("resolution_blocker"),
        "bucket": bucket,
        "root_status": ent.attrs.get("root_status"),
    }


def summarize_gaps(gaps: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(g.get("bucket") or "") for g in gaps if isinstance(g, dict))
    locate = int(counts.get("locate_blocking") or 0) + int(counts.get("audit_blocking") or 0)
    return {
        "locate_blocking": locate,
        "host_runtime_leaf": int(counts.get("host_runtime_leaf") or 0),
        "catalog_unproven": int(counts.get("catalog_unproven") or 0),
        "audit_blocking": int(counts.get("audit_blocking") or 0),
        "total": len(gaps),
    }


def list_gaps(codemap: CodeMap, audit: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return locate-relevant gaps. Settled PROJECT/BUILTIN/REACHED are omitted.

    Audit blocking codes stay. Partial entities are classified; host runtime
    leaves are kept but bucketed so they do not look like locate failure.
    Pass ``audit`` when the caller already ran ``audit_codemap`` — same rows,
    no second full-graph walk.
    """
    if audit is None:
        from ascendc_codemap_mcp.engine.diagnostics.audit import audit_codemap

        audit = audit_codemap(codemap)
    gaps: list[dict[str, Any]] = []
    code_map = {
        "MISSING_KERNEL": "missing_kernel",
        "MISSING_TILING_KEY": "missing_tiling_key",
        "MISSING_TILING_DATA": "missing_tiling_data",
        "MISSING_INPUT": "missing_input",
        "MISSING_OUTPUT": "missing_output",
        "MISSING_EVIDENCE_BACKED_HOST_KERNEL_PATH": "missing_host_kernel_path",
        "MISSING_INPUT_TILINGKEY_KERNEL_PATH": "missing_input_tilingkey_kernel_path",
        "MISSING_TILINGDATA_KERNEL_PATH": "missing_tilingdata_kernel_path",
        "MISSING_INPUT_OUTPUT_PATH": "missing_input_output_path",
        "TILING_KEY_CARDINALITY_MISMATCH": "tiling_key_cardinality_mismatch",
        "SUSPICIOUS_CARTESIAN_KEY_KERNEL": "suspicious_cartesian_key_kernel",
    }
    for item in audit.get("blocking") or []:
        raw = str(item.get("code") or "")
        gaps.append(
            {
                "code": code_map.get(raw, raw.lower() or "audit_blocking"),
                "message": str(item.get("detail") or raw),
                "audit_code": raw,
                "bucket": "audit_blocking",
                **{k: v for k, v in item.items() if k not in {"code", "detail"}},
            }
        )

    for ent in codemap.entities.values():
        bucket = classify_gap_entity(ent)
        if not bucket:
            continue
        gaps.append(_entity_gap_row(ent, bucket))
    return gaps


def merge_resolutions(codemap: CodeMap, patches: list[dict[str, Any]]) -> CodeMap:
    """Apply agent gap patches as entities/relations (non-authoritative until commit)."""
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        kind = patch.get("entity_kind") or patch.get("kind")
        name = str(patch.get("name") or "")
        if kind and name:
            ent = codemap.upsert(str(kind), name, attrs=dict(patch.get("attrs") or {}))
            ent.status = str(patch.get("status") or "extracted")
        rel = patch.get("relation") or {}
        if rel.get("kind") and rel.get("src") and rel.get("dst"):
            codemap.link(
                str(rel["kind"]),
                str(rel["src"]),
                str(rel["dst"]),
                attrs=dict(rel.get("attrs") or {}),
            )
        if patch.get("derives_from") and patch.get("name"):
            src = codemap.upsert(EntityKind.INPUT, str(patch["derives_from"]))
            dst = codemap.upsert(EntityKind.TILING_KEY, name)
            codemap.link(
                RelationKind.DERIVES,
                src.id,
                dst.id,
                attrs={"provenance": "semantic_gap_patch"},
            )
    return codemap
