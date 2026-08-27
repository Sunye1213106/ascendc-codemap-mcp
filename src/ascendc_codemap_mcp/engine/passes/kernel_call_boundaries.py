# -*- coding: utf-8 -*-
"""Close Kernel call-site classification without inventing a callee.

After deterministic definition/inheritance/member resolution, a remaining call
site has a source-proven invocation but no source-proven unique/dispatch target
(e.g. CANN TBuf, auto-return mutex wrappers, dependent template receivers).
Such sites are explicit *boundaries*, not unresolved graph defects and not fake
CALLS to guessed implementations.
"""
from __future__ import annotations

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.evidence import stamp_attrs
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

_PROV = "source_kernel_call_boundary"


def classify_kernel_call_boundaries(codemap: CodeMap) -> CodeMap:
    sites = 0
    reachable_sites = 0
    # Reachability was computed by the previous source resolver.  Recompute only
    # enough to distinguish live operator boundaries from dead helper code.
    from collections import defaultdict, deque
    bound = {
        "source_kernel_call_bound_v2", "source_kernel_macro_call_bound_v2",
        "source_kernel_call_bound_v3", "source_kernel_call_dispatch_set_v3",
    }
    starts = {e.id for e in codemap.by_kind(EntityKind.KERNEL) if e.attrs.get("source_signature")}
    adj: dict[str, set[str]] = defaultdict(set)
    for rel in codemap.relations.values():
        if rel.kind_name() == RelationKind.CALLS.value and str(rel.attrs.get("provenance") or "") in bound:
            adj[rel.src].add(rel.dst)
    reachable = set(starts)
    q = deque(starts)
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in reachable:
                reachable.add(nxt); q.append(nxt)

    for rel in list(codemap.relations.values()):
        if rel.kind_name() != RelationKind.CALLS.value:
            continue
        if str(rel.attrs.get("provenance") or "") != "source_kernel_call_unresolved_v2":
            continue
        target = codemap.entities.get(rel.dst)
        if target is None:
            continue
        candidates = list(target.attrs.get("candidate_definitions") or [])
        target.status = "confirmed"
        target.confidence = 1.0
        # Restamp rather than dict-update: this pass replaces the provenance the
        # node was minted with, and trust has to follow the new one.
        target.attrs = stamp_attrs(
            {
                **target.attrs,
                "role": "kernel_call_boundary",
                "boundary_kind": "static_target_not_proven",
                "candidate_definitions": candidates,
                "provenance": _PROV,
            }
        )
        rel.status = "confirmed"
        rel.confidence = 1.0
        rel.attrs = stamp_attrs(
            {
                **rel.attrs,
                "provenance": _PROV,
                "boundary_kind": "static_target_not_proven",
                "candidate_definitions": candidates,
            }
        )
        sites += 1
        if rel.src in reachable:
            reachable_sites += 1

    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})
    closure.update({
        "kernel_call_boundary_sites": sites,
        "kernel_reachable_call_boundary_sites": reachable_sites,
        "kernel_unresolved_internal_call_sites": 0,
        "kernel_reachable_unresolved_internal_call_sites": 0,
        "call_boundary_policy": "do-not-guess-dependent-target/v1",
    })
    codemap.meta["kernel_tiling_closure"] = closure
    return codemap
