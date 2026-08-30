# -*- coding: utf-8 -*-
"""Soundness/completeness audit for a committed ``.uo`` CodeMap.

A TilingKey is structurally closed when current source provides at least one
packing formula with a concrete producer/root chain. Leftover pass-through
``SetTilingKey`` sites without sources must not veto a real formula. Arbitrary
branch ``CONTROLS`` paths are excluded from Key rooting so a constant in an
unrelated condition cannot make a missing Host producer look complete.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.identity import (
    declaration_key,
    is_declaration_kind,
    is_forbidden_callable_name,
)
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.host_kernel import evidence_backed_host_kernel_path_exists

_FLOW_KINDS = {
    RelationKind.DERIVES.value,
    RelationKind.FLOWS_TO.value,
    RelationKind.CONTROLS.value,
    RelationKind.BINDS.value,
    RelationKind.SELECTS.value,
    RelationKind.INSTANTIATES.value,
    RelationKind.LAUNCHES.value,
    RelationKind.CALLS.value,
    RelationKind.READS.value,
    RelationKind.WRITES.value,
}
_ROOT_FLOW_KINDS = {RelationKind.DERIVES.value, RelationKind.FLOWS_TO.value}
_RUNTIME_KINDS = {EntityKind.VARIABLE.value, EntityKind.FIELD.value}
_LEAF_TYPE_ID_RE = re.compile(r"^SRCPOL::[^:]+::[A-Za-z_]\w*$")


def _integrity_false_confirmed(codemap: CodeMap) -> list[dict[str, Any]]:
    """Hard commit blockers: confirmed wrong facts. Coverage holes do not belong here."""
    out: list[dict[str, Any]] = []

    def _push(row: dict[str, Any]) -> bool:
        out.append(row)
        return len(out) >= 24

    for rel in codemap.relations.values():
        if str(rel.status or "").lower() != "confirmed":
            continue
        if rel.src not in codemap.entities or rel.dst not in codemap.entities:
            if _push(
                {
                    "code": "CONFIRMED_DANGLING_EDGE",
                    "detail": f"{rel.kind_name()} {rel.src!s} → {rel.dst!s}",
                    "relation_id": rel.id,
                }
            ):
                return out
            continue
        for end in (rel.src, rel.dst):
            ent = codemap.entities.get(end)
            if ent is None:
                continue
            state = str(ent.attrs.get("semantic_state") or "").lower()
            if state == "unresolved":
                if _push(
                    {
                        "code": "CONFIRMED_ON_UNRESOLVED_IDENTITY",
                        "detail": f"{rel.kind_name()} {rel.src!s} → {rel.dst!s} via {ent.name}",
                        "relation_id": rel.id,
                        "entity_id": ent.id,
                    }
                ):
                    return out
            if ent.kind_name() in {EntityKind.METHOD.value, EntityKind.FUNCTION.value}:
                if is_forbidden_callable_name(ent.name):
                    if _push(
                        {
                            "code": "KEYWORD_CALLABLE_NAME",
                            "detail": f"{ent.kind_name()} {ent.name!s}",
                            "entity_id": ent.id,
                        }
                    ):
                        return out
    for kind in (EntityKind.METHOD, EntityKind.FUNCTION):
        for ent in codemap.by_kind(kind):
            if not is_forbidden_callable_name(ent.name):
                continue
            if _push(
                {
                    "code": "KEYWORD_CALLABLE_NAME",
                    "detail": f"{ent.kind_name()} {ent.name!s}",
                    "entity_id": ent.id,
                }
            ):
                return out
    triples: dict[tuple[str, str, str], str] = {}
    for rel in codemap.relations.values():
        if str(rel.status or "").lower() != "confirmed":
            continue
        key = (rel.kind_name(), str(rel.src), str(rel.dst))
        prev = triples.get(key)
        if prev and prev != rel.id:
            if _push(
                {
                    "code": "DUPLICATE_CONFIRMED_TRIPLE",
                    "detail": f"{key[0]} {key[1]} → {key[2]}",
                    "relation_id": rel.id,
                }
            ):
                return out
        else:
            triples[key] = rel.id
    seen_decl: dict[tuple[str, str, str, str], str] = {}
    for ent in codemap.entities.values():
        if not is_declaration_kind(ent.kind_name()):
            continue
        if str(ent.status or "").lower() != "confirmed":
            continue
        if not str(ent.file or "").strip() or not _leaf_name(ent.name):
            continue
        key = declaration_key(ent)
        prev = seen_decl.get(key)
        if prev and prev != ent.id:
            if _push(
                {
                    "code": "DUPLICATE_DECLARATION",
                    "detail": f"{key[0]} {ent.name} @ {key[1]} owner={key[2]!s}",
                    "entity_id": ent.id,
                }
            ):
                return out
        else:
            seen_decl[key] = ent.id
    for ent in codemap.by_kind(EntityKind.TYPE):
        if str(ent.status or "").lower() != "confirmed":
            continue
        eid = str(ent.id or "")
        name = str(ent.name or "")
        if _LEAF_TYPE_ID_RE.match(eid) and "::" not in name:
            if _push(
                {
                    "code": "LEAF_ONLY_TYPE_IDENTITY",
                    "detail": f"{name} minted as {eid}",
                    "entity_id": eid,
                }
            ):
                return out
    return out


def _leaf_name(name: str) -> str:
    return str(name or "").replace(".", "::").split("::")[-1].strip()


def _flow_adjacency(codemap: CodeMap, kinds: set[str]) -> dict[str, list[str]]:
    """Forward edges of `kinds`. Build once per audit, not once per question.

    `_path_exists` rebuilt this over every relation on each of its four calls,
    which is where 116k of the stage's relation-kind reads came from. The graph
    does not change between the four questions.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    for rel in codemap.relations.values():
        if rel.kind_name() in kinds:
            adj[rel.src].append(rel.dst)
    return adj


def _path_exists(
    codemap: CodeMap,
    *,
    start_kind: EntityKind,
    end_kind: EntityKind,
    require_kind: EntityKind | None = None,
    adj: dict[str, list[str]] | None = None,
) -> bool:
    starts = codemap.by_kind(start_kind)
    ends = {e.id for e in codemap.by_kind(end_kind)}
    if not starts or not ends:
        return False
    if adj is None:
        adj = _flow_adjacency(codemap, _FLOW_KINDS)
    required = require_kind.value if require_kind is not None else ""
    q: deque[tuple[str, bool]] = deque((e.id, e.kind_name() == required if required else True) for e in starts)
    seen = set(q)
    while q:
        cur, has_required = q.popleft()
        if cur in ends and has_required:
            return True
        for nxt in adj.get(cur, ()):
            ent = codemap.entities.get(nxt)
            next_required = has_required or bool(ent and ent.kind_name() == required)
            state = (nxt, next_required)
            if state not in seen:
                seen.add(state)
                q.append(state)
    return False


def _trusted_compile_root(entity: Entity) -> bool:
    provenance = str(entity.attrs.get("provenance") or "")
    origin = str(entity.attrs.get("origin") or "")
    return bool(entity.attrs.get("compile_root") or provenance.startswith("source_") or provenance.startswith("source_host_") or origin == "constexpr_or_define")


_ALWAYS_ROOT = {EntityKind.INPUT.value, EntityKind.BUILD_VARIANT.value, EntityKind.ARCH.value}
_COMPILE_ROOT = {EntityKind.COMPILE_VAR.value, EntityKind.MACRO.value}


def _trusted_root(entity: Entity) -> bool:
    kind = entity.kind_name()
    if kind in _ALWAYS_ROOT:
        return True
    if kind in _COMPILE_ROOT:
        return _trusted_compile_root(entity)
    return False


def _source_rooted_entities(codemap: CodeMap) -> set[str]:
    roots = {ent.id for ent in codemap.entities.values() if _trusted_root(ent)}
    adj: dict[str, list[str]] = defaultdict(list)
    for rel in codemap.relations.values():
        if rel.kind_name() in _ROOT_FLOW_KINDS:
            adj[rel.src].append(rel.dst)
    seen = set(roots)
    queue = deque(roots)
    while queue:
        cur = queue.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _incoming(codemap: CodeMap) -> dict[str, list[Any]]:
    incoming: dict[str, list[Any]] = defaultdict(list)
    for rel in codemap.relations.values():
        incoming[rel.dst].append(rel)
    return incoming


_PACKING_NODE_PROVS = {
    "source_get_tpl_tiling_key",
    "source_packing_helper",
    "source_set_tiling_key",
    "source_get_tiling_key",
    "source_assign_tiling_key",
    "source_bitpack_dim",
}
_PACKING_SOURCE_PROVS = {
    "source_get_tpl_tiling_key_symbol",
    "source_get_tpl_tiling_key_literal",
    "source_set_tiling_key",
    "source_get_tiling_key",
    "source_assign_tiling_key",
}


def _packing_nodes(codemap: CodeMap, key: Entity, incoming: dict[str, list[Any]]) -> list[Entity]:
    out: list[Entity] = []
    for rel in incoming.get(key.id, ()):
        if rel.kind_name() != RelationKind.DERIVES.value:
            continue
        if str(rel.attrs.get("provenance") or "") not in _PACKING_NODE_PROVS:
            continue
        node = codemap.entities.get(rel.src)
        if node is not None:
            out.append(node)
    return out


def _packing_sources(codemap: CodeMap, node: Entity, incoming: dict[str, list[Any]]) -> list[Entity]:
    out: list[Entity] = []
    for rel in incoming.get(node.id, ()):
        if rel.kind_name() != RelationKind.DERIVES.value:
            continue
        provenance = str(rel.attrs.get("provenance") or "")
        if provenance not in _PACKING_SOURCE_PROVS:
            continue
        source = codemap.entities.get(rel.src)
        if source is not None:
            out.append(source)
    return out


_SOURCE_PRODUCER_PROVS = {
    "source_host_defuse",
    "source_host_tiling_input",
    "source_host_api_accessor",
}


def _source_has_producer(source: Entity, incoming: dict[str, list[Any]]) -> bool:
    if _trusted_root(source):
        return True
    if source.kind_name() not in _RUNTIME_KINDS:
        return False
    if any(
        rel.kind_name() == RelationKind.DERIVES.value
        and str(rel.attrs.get("provenance") or "") in _SOURCE_PRODUCER_PROVS
        for rel in incoming.get(source.id, ())
    ):
        return True
    if int(source.attrs.get("producer_site_count") or 0) <= 0:
        return False
    return any(
        rel.kind_name() == RelationKind.DERIVES.value
        and str(rel.attrs.get("provenance") or "") == "source_host_defuse"
        for rel in incoming.get(source.id, ())
    )


def _upstream_unresolved(codemap: CodeMap, start_id: str, incoming: dict[str, list[Any]]) -> list[str]:
    seen = {start_id}
    q = deque([start_id])
    unresolved: list[str] = []
    while q:
        cur = q.popleft()
        ent = codemap.entities.get(cur)
        if ent is not None and ent.attrs.get("dependency_unresolved"):
            unresolved.append(ent.name)
        for rel in incoming.get(cur, ()):
            if rel.kind_name() not in _ROOT_FLOW_KINDS:
                continue
            if rel.src not in seen:
                seen.add(rel.src)
                q.append(rel.src)
    return sorted(set(unresolved))


def _key_evidence(codemap: CodeMap, key: Entity, *, rooted: set[str], incoming: dict[str, list[Any]]) -> dict[str, Any]:
    packing = _packing_nodes(codemap, key, incoming)
    per_call: list[dict[str, Any]] = []
    any_producer = False
    any_rooted = False
    unresolved: set[str] = set()
    producer_sites: list[dict[str, Any]] = []
    for node in packing:
        sources = _packing_sources(codemap, node, incoming)
        source_rows = []
        has_producer = False
        has_root = node.id in rooted
        for source in sources:
            producer = _source_has_producer(source, incoming)
            has_producer = has_producer or producer
            for site in source.attrs.get("producer_sites") or []:
                if isinstance(site, dict) and site not in producer_sites:
                    producer_sites.append(site)
            source_rows.append({"id": source.id, "name": source.name, "kind": source.kind_name(), "producer": producer, "rooted": source.id in rooted})
        any_producer = any_producer or has_producer
        any_rooted = any_rooted or (has_producer and has_root)
        unresolved.update(_upstream_unresolved(codemap, node.id, incoming))
        per_call.append({"packing_node": node.id, "expression": node.attrs.get("expression") or node.name, "has_source_producer": has_producer, "has_trusted_root": has_root, "sources": source_rows})
    if any_producer and key.id in rooted:
        any_rooted = True
    return {
        "key": key.name,
        "packed": bool(packing),
        "producer": any_producer,
        "rooted": any_rooted,
        "dependency_complete": not unresolved,
        "unresolved_dependencies": sorted(unresolved),
        "producer_sites": producer_sites,
        "packing": per_call,
    }


def audit_codemap(codemap: CodeMap) -> dict[str, Any]:
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    inputs = codemap.by_kind(EntityKind.INPUT)
    tensor_inputs = [e for e in inputs if e.attrs.get("api_kind") == "tensor"]
    attributes = [e for e in inputs if e.attrs.get("api_kind") == "attribute"]
    outputs = codemap.by_kind(EntityKind.OUTPUT)
    hosts = codemap.by_kind(EntityKind.FUNCTION) + codemap.by_kind(EntityKind.FIELD) + codemap.by_kind(EntityKind.VARIABLE)
    keys = codemap.by_kind(EntityKind.TILING_KEY)
    declared_keys = sorted((e for e in keys if e.attrs.get("source_declared")), key=lambda e: int(e.attrs.get("decl_order") or 0))
    tiling_data = codemap.by_kind(EntityKind.TILING_DATA)
    tiling_fields = codemap.by_kind(EntityKind.TILING_FIELD)
    kernels = codemap.by_kind(EntityKind.KERNEL)
    instances = codemap.by_kind(EntityKind.TEMPLATE_INSTANCE)

    def block(code: str, detail: str, **extra: Any) -> None:
        blocking.append({"code": code, "detail": detail, **extra})

    def warn(code: str, detail: str, **extra: Any) -> None:
        warnings.append({"code": code, "detail": detail, **extra})

    if not hosts:
        block("MISSING_HOST", "no Host function/field/variable entities")
    if not inputs:
        block("MISSING_INPUT", "no API input/attribute entities")
    if not outputs:
        stats = codemap.meta.get("source_contract_stats") or {}
        proto_seen = int(stats.get("api_source_files") or 0) > 0
        proto_outputs = int(stats.get("api_outputs") or 0)
        if proto_seen and proto_outputs == 0:
            warn("SOURCE_HAS_NO_OUTPUT", "REG_OP / OpDef declares no OUTPUT; fusion send-style ops are honest")
        else:
            block("MISSING_OUTPUT", "no API output entities")
    if not keys:
        stats = codemap.meta.get("source_contract_stats") or {}
        declared_n = int(
            stats.get("source_declared_tiling_keys")
            or codemap.meta.get("source_declared_tiling_key_count")
            or 0
        )
        has_sites = bool(stats.get("source_has_tiling_key_sites"))
        if declared_n == 0 and not has_sites:
            warn("SOURCE_HAS_NO_TILING_KEY", "current source has no TPL / TILING_KEY_IS / packing helper; barrier-style ops are honest")
        else:
            block("MISSING_TILING_KEY", "no TilingKey entities")
    if not tiling_data or not tiling_fields:
        block("MISSING_TILING_DATA", "no structured TilingData class/field model")
    if not kernels:
        block("MISSING_KERNEL", "no Kernel entities")

    source_key_count = int(codemap.meta.get("source_declared_tiling_key_count") or 0)
    if source_key_count and len(declared_keys) != source_key_count:
        block(
            "TILING_KEY_CARDINALITY_MISMATCH",
            f"current source declares {source_key_count} TilingKeys but CodeMap contains {len(declared_keys)} source-declared keys",
            declared=codemap.meta.get("source_declared_tiling_keys") or [],
        )

    host_packing = codemap.meta.get("host_tiling_key_packing") or {}
    packing_calls = int(host_packing.get("calls") or 0)
    packing_bound = int(host_packing.get("fields_bound") or 0)
    packing_mismatches = list(host_packing.get("argument_count_mismatches") or [])
    packing_missing = [e.name for e in declared_keys if not e.attrs.get("host_packing_expressions")]
    packing_complete = packing_bound == len(declared_keys) and not packing_missing
    if packing_calls and not packing_complete:
        block("INCOMPLETE_HOST_TILINGKEY_PACKING", f"Host packing covers {packing_bound}/{len(declared_keys)} source-declared TilingKeys", missing=packing_missing, argument_count_mismatches=packing_mismatches)
    elif packing_mismatches and packing_complete:
        warn(
            "HOST_TILINGKEY_PACKING_EXTRA_ARITY",
            "Additional GET_TPL_TILING_KEY sites with a different arity are ignored once every declared key is bound",
            argument_count_mismatches=packing_mismatches,
        )

    rooted_entities = _source_rooted_entities(codemap)
    incoming = _incoming(codemap)
    evidence_rows = [_key_evidence(codemap, key, rooted=rooted_entities, incoming=incoming) for key in declared_keys]
    producer_keys = [row["key"] for row in evidence_rows if row["producer"]]
    rooted_keys = [row["key"] for row in evidence_rows if row["rooted"]]
    dependency_complete_keys = [row["key"] for row in evidence_rows if row["dependency_complete"]]
    producer_missing = [row["key"] for row in evidence_rows if not row["producer"]]
    unrooted_keys = [row["key"] for row in evidence_rows if not row["rooted"]]
    dependency_partial = [row for row in evidence_rows if not row["dependency_complete"]]

    if declared_keys and producer_missing:
        block("MISSING_HOST_TILINGKEY_PRODUCERS", f"{len(producer_missing)}/{len(declared_keys)} source-declared TilingKeys lack a current-source Host producer", missing=producer_missing)
    if declared_keys and unrooted_keys:
        block("UNROOTED_TILING_KEYS", f"{len(unrooted_keys)}/{len(declared_keys)} source-declared TilingKeys have no source-producer-backed API/compile root", unrooted=unrooted_keys)
    if dependency_partial:
        warn("PARTIAL_TILINGKEY_DEPENDENCY_SKELETON", f"{len(dependency_partial)}/{len(declared_keys)} TilingKeys retain unresolved runtime dependency leaves", examples=[{"key": row["key"], "unresolved": row["unresolved_dependencies"][:12]} for row in dependency_partial[:12]])

    strict_path = evidence_backed_host_kernel_path_exists(codemap)
    flow_adj = _flow_adjacency(codemap, _FLOW_KINDS)
    input_key_kernel = _path_exists(codemap, start_kind=EntityKind.INPUT, end_kind=EntityKind.KERNEL, require_kind=EntityKind.TILING_KEY, adj=flow_adj)
    tdata_kernel = _path_exists(codemap, start_kind=EntityKind.TILING_DATA, end_kind=EntityKind.KERNEL, adj=flow_adj)
    input_output = _path_exists(codemap, start_kind=EntityKind.INPUT, end_kind=EntityKind.OUTPUT, adj=flow_adj)
    if kernels and not strict_path:
        block("MISSING_EVIDENCE_BACKED_HOST_KERNEL_PATH", "no semantic INPUT→…→KERNEL path; node presence alone is insufficient")
    if inputs and keys and kernels and not input_key_kernel:
        key_to_kernel = _path_exists(
            codemap, start_kind=EntityKind.TILING_KEY, end_kind=EntityKind.KERNEL, adj=flow_adj
        )
        # Dtype / compile-rooted keys never flow from INPUT. KEY→KERNEL plus
        # INPUT→KERNEL is still a source-backed selection path.
        if not (strict_path and key_to_kernel):
            block("MISSING_INPUT_TILINGKEY_KERNEL_PATH", "no source-backed INPUT→…→TILING_KEY→…→KERNEL selection path")
    if tiling_data and kernels and not tdata_kernel:
        block("MISSING_TILINGDATA_KERNEL_PATH", "TilingData is present but no source-backed TILING_DATA→KERNEL consumption path exists")
    if inputs and outputs and kernels and not input_output:
        block("MISSING_INPUT_OUTPUT_PATH", "no source-backed INPUT→…→KERNEL→OUTPUT execution/data path exists")

    legacy_summary = codemap.summary()
    legacy_path = bool(legacy_summary.get("has_host_kernel_path"))
    if legacy_path and not strict_path:
        warn("SUMMARY_HOST_KERNEL_PATH_FALSE_POSITIVE", "CodeMap.summary() reports a Host→Kernel path without an evidence-backed semantic path")
    strict_summary = dict(legacy_summary)
    strict_summary["has_host_kernel_path"] = strict_path
    strict_summary["has_input_tilingkey_kernel_path"] = input_key_kernel
    strict_summary["has_tilingdata_kernel_path"] = tdata_kernel
    strict_summary["has_input_output_path"] = input_output
    strict_summary["tiling_key_declaration_coverage"] = f"{len(declared_keys)}/{source_key_count or len(declared_keys)}"
    strict_summary["tiling_key_host_packing_coverage"] = f"{packing_bound}/{len(declared_keys)}"
    strict_summary["tiling_key_host_producer_coverage"] = f"{len(producer_keys)}/{len(declared_keys)}"
    strict_summary["tiling_key_root_coverage"] = f"{len(rooted_keys)}/{len(declared_keys)}"
    strict_summary["tiling_key_dependency_coverage"] = f"{len(dependency_complete_keys)}/{len(declared_keys)}"

    select_pairs = {(rel.src, rel.dst) for rel in codemap.relations.values() if rel.kind_name() in {RelationKind.SELECTS.value, RelationKind.LAUNCHES.value}}
    tpl_dims = [
        key
        for key in keys
        if str(key.attrs.get("provenance") or "") == "source_tpl_args_decl"
    ]
    if len(keys) > 1 and len(kernels) > 1 and len(tpl_dims) != len(keys):
        universal = all((key.id, kernel.id) in select_pairs for key in keys for kernel in kernels)
        if universal:
            block("SUSPICIOUS_CARTESIAN_KEY_KERNEL", f"all {len(keys)} TilingKeys select/launch all {len(kernels)} Kernels")

    outgoing: dict[str, list[Any]] = defaultdict(list)
    graph_incoming: dict[str, list[Any]] = defaultdict(list)
    for rel in codemap.relations.values():
        outgoing[rel.src].append(rel)
        graph_incoming[rel.dst].append(rel)
    selection_kinds = {RelationKind.SELECTS.value, RelationKind.CONTROLS.value, RelationKind.BINDS.value, RelationKind.INSTANTIATES.value, RelationKind.LAUNCHES.value}
    unbound_keys = [e.name for e in keys if not any(r.kind_name() in selection_kinds for r in outgoing.get(e.id, ()))]
    if unbound_keys:
        warn("UNBOUND_TILING_KEYS", f"{len(unbound_keys)} TilingKeys have no outgoing selection/binding edge", examples=sorted(unbound_keys)[:20])
    kernel_incoming = {RelationKind.SELECTS.value, RelationKind.CONTROLS.value, RelationKind.INSTANTIATES.value, RelationKind.LAUNCHES.value, RelationKind.FLOWS_TO.value, RelationKind.CALLS.value}
    unbound_kernels = [e.name for e in kernels if not any(r.kind_name() in kernel_incoming for r in graph_incoming.get(e.id, ()))]
    if unbound_kernels:
        warn("UNBOUND_KERNELS", f"{len(unbound_kernels)} Kernels have no incoming semantic edge", examples=sorted(unbound_kernels)[:20])
    unbound_instances = [e.name for e in instances if not any(r.kind_name() == RelationKind.INSTANTIATES.value and codemap.entities.get(r.dst) and codemap.entities[r.dst].kind_name() == EntityKind.KERNEL.value for r in outgoing.get(e.id, ()))]
    if unbound_instances:
        warn("UNBOUND_TEMPLATE_INSTANCES", f"{len(unbound_instances)} template instances have no explicit Kernel target", examples=sorted(unbound_instances)[:20])

    unresolved_entities = [e.id for e in codemap.entities.values() if str(e.status).lower() in {"unresolved", "partial", "unknown"}]
    unresolved_relations = [r.id for r in codemap.relations.values() if str(r.status).lower() in {"unresolved", "partial", "unknown"}]
    if unresolved_entities or unresolved_relations:
        warn("UNRESOLVED_FACTS", f"entities={len(unresolved_entities)} relations={len(unresolved_relations)}")
    low_conf_entities = [e.id for e in codemap.entities.values() if float(e.confidence) < 0.8]
    low_conf_relations = [r.id for r in codemap.relations.values() if float(r.confidence) < 0.8]
    if low_conf_entities or low_conf_relations:
        warn("LOW_CONFIDENCE_FACTS", f"entities={len(low_conf_entities)} relations={len(low_conf_relations)}")

    integrity_blocking = _integrity_false_confirmed(codemap)

    # Meta-only Kernel root-trace quality (does not block verify).
    ke = None
    if isinstance(codemap.meta, dict):
        ke = codemap.meta.get("kernel_root_trace") or codemap.meta.get("kernel_execution")
    kernel_root_trace_quality: dict[str, Any] | None = None
    if isinstance(ke, dict) and not ke.get("skipped"):
        quality = dict(ke.get("quality") or {})
        quality.setdefault("ops", ke.get("operations"))
        quality.setdefault("buffers", ke.get("buffers"))
        quality.setdefault("reached_buffers", ke.get("reached_buffers"))
        quality.setdefault("reached_registers", ke.get("reached_registers"))
        quality.setdefault("reached_operations", ke.get("reached_operations"))
        quality.setdefault("gap_count", ke.get("gap_count"))
        kernel_root_trace_quality = quality

    return {
        "ok": not blocking,
        "op_name": codemap.op_name,
        "architecture": codemap.architecture,
        "summary": strict_summary,
        "legacy_summary_has_host_kernel_path": legacy_path,
        "evidence_backed_host_kernel_path": strict_path,
        "evidence_backed_input_tilingkey_kernel_path": input_key_kernel,
        "evidence_backed_tilingdata_kernel_path": tdata_kernel,
        "evidence_backed_input_output_path": input_output,
        "tiling_key_rooted": rooted_keys,
        "tiling_key_unrooted": unrooted_keys,
        "tiling_key_producer_missing": producer_missing,
        "tiling_key_evidence": evidence_rows,
        "kernel_root_trace_quality": kernel_root_trace_quality,
        "kernel_execution_quality": kernel_root_trace_quality,  # compat alias

        "counts": {
            "inputs": len(inputs), "tensor_inputs": len(tensor_inputs), "attributes": len(attributes), "outputs": len(outputs),
            "host_entities": len(hosts), "tiling_keys": len(keys), "source_declared_tiling_keys": source_key_count,
            "host_packing_bound_tiling_keys": packing_bound, "producer_tiling_keys": len(producer_keys), "rooted_tiling_keys": len(rooted_keys),
            "dependency_complete_tiling_keys": len(dependency_complete_keys), "unrooted_tiling_keys": len(unrooted_keys),
            "tiling_data": len(tiling_data), "tiling_fields": len(tiling_fields), "kernels": len(kernels), "template_instances": len(instances),
            "unbound_tiling_keys": len(unbound_keys), "unbound_kernels": len(unbound_kernels), "unbound_template_instances": len(unbound_instances),
            "unresolved_entities": len(unresolved_entities), "unresolved_relations": len(unresolved_relations),
        },
        "blocking": blocking,
        "integrity_blocking": integrity_blocking,
        "warnings": warnings,
    }


def audit_uo(path: str | Path) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.store.reader import read_codemap, read_meta
    product = Path(path).expanduser().resolve()
    cm = read_codemap(product)
    report = audit_codemap(cm)
    report["product"] = str(product)
    report["size_bytes"] = product.stat().st_size
    report["meta"] = read_meta(product)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a committed UO CodeMap binary")
    parser.add_argument("uo", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON (default is also JSON-safe text)")
    args = parser.parse_args(argv)
    try:
        report = audit_uo(args.uo)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
