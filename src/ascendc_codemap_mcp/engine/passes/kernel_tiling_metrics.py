# -*- coding: utf-8 -*-
"""Compute strict source-backed Kernel/TilingData closure metrics.

The verified ``TILING_DATA -> KERNEL`` edge is a certificate edge: it exists
only when the source closure invariant below is true.  Therefore the generic UO
review cannot pass merely because one TilingData type was unpacked somewhere.
"""
from __future__ import annotations

from collections import defaultdict, deque

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

_BOUND_CALLS = {
    "source_kernel_call_bound_v2",
    "source_kernel_macro_call_bound_v2",
    "source_kernel_call_bound_v3",
    "source_kernel_call_dispatch_set_v3",
}
_CERT_EDGE_PROVENANCE = "source_tiling_registration_verified"


def finalize_kernel_tiling_metrics(codemap: CodeMap) -> CodeMap:
    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})
    starts = {e.id for e in codemap.by_kind(EntityKind.KERNEL) if e.attrs.get("source_signature")}
    adj: dict[str, set[str]] = defaultdict(set)
    for rel in codemap.relations.values():
        if rel.kind_name() == RelationKind.CALLS.value and str(rel.attrs.get("provenance") or "") in _BOUND_CALLS:
            adj[rel.src].add(rel.dst)
    reachable = set(starts)
    q = deque(starts)
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in reachable:
                reachable.add(nxt)
                q.append(nxt)

    incoming: dict[str, list] = defaultdict(list)
    for rel in codemap.relations.values():
        incoming[rel.dst].append(rel)

    consumed: set[str] = set()
    unresolved_reads = 0
    unresolved_calls = 0
    for rel in codemap.relations.values():
        provenance = str(rel.attrs.get("provenance") or "")
        if provenance == "source_tilingdata_read_verified" and rel.src in reachable:
            consumed.add(rel.dst)
        elif provenance == "source_tilingdata_read_unresolved_verified" and rel.src in reachable:
            unresolved_reads += 1
        elif provenance == "source_kernel_call_unresolved_v2" and rel.src in reachable:
            unresolved_calls += 1

    with_producer: list[str] = []
    without_producer: list[str] = []
    for field_id in sorted(consumed):
        field = codemap.entities.get(field_id)
        if field is None:
            continue
        has_writer = any(
            rel.kind_name() == RelationKind.WRITES.value
            and str(rel.attrs.get("provenance") or "") == "source_tilingdata_host_write_verified"
            for rel in incoming.get(field_id, ())
        )
        has_default = "default_initializer" in field.attrs
        qualified = str(field.attrs.get("qualified_name") or f"{field.attrs.get('owner')}::{field.name}")
        if has_writer or has_default:
            with_producer.append(qualified)
        else:
            without_producer.append(qualified)

    selected_types = set(closure.get("tiling_selected_type_closure") or ())
    selected_field_ids = {
        e.id for e in codemap.by_kind(EntityKind.TILING_FIELD)
        if str(e.attrs.get("owner") or "") in selected_types
    }
    consumed_selected = consumed & selected_field_ids if selected_field_ids else consumed
    declared_key_count = int(codemap.meta.get("source_declared_tiling_key_count") or 0)
    template_count = int(closure.get("kernel_template_args") or 0)
    template_complete = template_count > 0 and (not declared_key_count or template_count == declared_key_count)

    strict = bool(
        closure.get("architecture_pure")
        and int(closure.get("kernel_entries") or 0) > 0
        and template_complete
        and int(closure.get("kernel_abi_links") or 0) > 0
        and len(reachable) > len(starts)
        and unresolved_calls == 0
        and len(consumed) > 0
        and unresolved_reads == 0
        and not without_producer
        and int(closure.get("tiling_ambiguous_writer_sites") or 0) == 0
    )

    closure.update(
        {
            "kernel_reachable_scopes": len(reachable),
            "kernel_reachable_unresolved_internal_call_sites": unresolved_calls,
            "kernel_template_binding_complete": template_complete,
            "tiling_entry_reachable_fields": len(consumed),
            "tiling_entry_reachable_selected_fields": len(consumed_selected),
            "tiling_entry_reachable_unresolved_read_sites": unresolved_reads,
            "tiling_consumed_fields_with_producer": len(with_producer),
            "tiling_consumed_fields_without_producer": without_producer,
            "tiling_consumed_field_producer_coverage": f"{len(with_producer)}/{len(consumed)}",
            "strict_closure_ok": strict,
            "metrics_policy": "entry-reachable-consumed-fields/v2",
        }
    )

    cert_relations = [
        rid for rid, rel in codemap.relations.items()
        if str(rel.attrs.get("provenance") or "") == _CERT_EDGE_PROVENANCE
    ]
    # Registration-verified TILING_DATA→KERNEL is a source contract
    # (REGISTER_TILING_* in the kernel TU). Strict consumed-field metrics may
    # fail independently; they must not retract that contract. Only synthesize a
    # cert edge when the strict invariant holds and none exists yet.
    if strict and not cert_relations:
        kernels = codemap.by_kind(EntityKind.KERNEL)
        roots = set(closure.get("tiling_registered_root_types") or ())
        for td in codemap.by_kind(EntityKind.TILING_DATA):
            if td.name not in roots:
                continue
            for kernel in kernels:
                codemap.link(
                    RelationKind.FLOWS_TO,
                    td.id,
                    kernel.id,
                    attrs={"provenance": _CERT_EDGE_PROVENANCE, "strict_closure": True},
                    status="confirmed",
                )

    codemap.meta["kernel_tiling_closure"] = closure
    codemap.meta["has_strict_kernel_tiling_closure"] = strict
    return codemap
