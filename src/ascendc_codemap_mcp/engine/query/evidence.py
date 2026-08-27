# -*- coding: utf-8 -*-
"""Skill-useful CodeMap evidence projection for uo-query and CE."""

from __future__ import annotations

import re
from typing import Any

from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.semantics.ascendc_sync import (
    BARRIER_CALLEES,
    FLAG_SYNC_CALLEES,
    SYNC_MECHANISM,
    TPIPE_CALLEES,
    TQUE_CALLEES,
    canonical_sync_name,
    is_flag_sync,
    is_sync_root,
)

USEFUL_EDGE_KINDS: tuple[str, ...] = (
    "WRITES",
    "READS",
    "CALLS",
    "CONTROLS",
    "DERIVES",
    "SELECTS",
    "LAUNCHES",
    "SIGNALS",
    "AWAITS",
    "FLOWS_TO",
    "BINDS",
    "WRAPS",
    "DECLARES",
    "ROOTED_AT",
    "ALIASES",
    "GUARDED_BY",
    "PRECEDES",
    "ACTIVE_UNDER",
    "CONTAINS",
    "RETURNS",
)

_FIELD_EDGE_KINDS = frozenset({"WRITES", "READS", "DERIVES", "CONTROLS"})

_FLAG_SYNC_CALLEES = FLAG_SYNC_CALLEES
_TQUE_CALLEES = TQUE_CALLEES
_TPIPE_CALLEES = TPIPE_CALLEES
_BARRIER_CALLEES = BARRIER_CALLEES
_SYNC_CALLEES = _FLAG_SYNC_CALLEES | _TQUE_CALLEES | _TPIPE_CALLEES | _BARRIER_CALLEES | frozenset(SYNC_MECHANISM)
_PRECISION_CALLEES = frozenset({"Cast", "DataCopy", "DataCopyPad"})
_KERNEL_API_CALLEES = _SYNC_CALLEES | _PRECISION_CALLEES | frozenset(
    {
        "LoadData",
        "SetGlobalBuffer",
        "GetPhyAddr",
        "LoadAlign",
        "StoreAlign",
        "StoreUnAlign",
        "CreateMask",
        "UpdateMask",
        "InitOutput",
        "PopStackBuffer",
        "InitShareBufStart",
        "InitShareBufEnd",
    }
)

_KIND_FACTS: dict[str, tuple[str, ...]] = {
    EntityKind.TILING_KEY.value: (
        "bit_lo",
        "bit_hi",
        "bit_width",
        "domain",
        "value_domain",
        "packing_value_sites",
        "packing_expr",
    ),
    EntityKind.TILING_FIELD.value: (
        "owner",
        "ctype",
        "rhs",
        "host_writer_sites",
        "value_defining_sites",
        "producer_sites",
        "check_sites",
        "fused_outer_candidates",
        "local_aliases",
        "write_sites",
    ),
    EntityKind.FIELD.value: (
        "layer",
        "rhs",
        "guards",
        "check_sites",
        "owner",
        "cpp_type",
        "default_initializer",
        "definition_sites",
        "write_sites",
    ),
    EntityKind.TYPE.value: (
        "cpp_kind",
        "role",
        "alias_of",
        "root",
        "root_kind",
        "type_name",
        "owner",
        "catalog",
        "spelling",
        "wraps_storage",
        "wraps_lock",
        "wraps_flag",
        "conditional_flag",
    ),
    EntityKind.BUFFER.value: (
        "memory_space",
        "tposition",
        "wrapper",
        "scope",
        "type_name",
        "role",
        "allocated",
        "stack_pop",
        "wraps_lock",
        "wraps_storage",
        "conditional_flag",
        "mutex_policy",
    ),
    EntityKind.REGISTER.value: ("register_class", "memory_space", "scope", "type_name"),
    EntityKind.OPERATION.value: (
        "callee",
        "receiver",
        "function",
        "args",
        "argument",
        "template_args",
        "category",
        "mechanism",
        "flag_paired",
        "kernel_phase",
        "layer",
        "catalog",
    ),
    EntityKind.PIPE.value: (
        "identity",
        "scope",
        "type_name",
        "pipe_ordinal",
        "kernel_phase",
        "role",
        "catalog",
        "pointer",
        "kernel_file",
    ),
    EntityKind.EVENT.value: ("identity", "scope", "event_type", "mechanism", "cross_core"),
    EntityKind.QUEUE.value: ("identity", "scope", "type_name", "tposition", "memory_space"),
    EntityKind.BRANCH.value: ("predicate", "condition", "branch_kind", "layer", "function", "dimensions"),
    EntityKind.KERNEL.value: ("source_signature", "variants"),
    EntityKind.INPUT.value: (
        "dtype",
        "shape",
        "optional",
        "declaration",
        "check_sites",
        "api_kind",
        "api_index",
        "api_attr_index",
    ),
    EntityKind.OUTPUT.value: ("dtype", "shape", "declaration", "api_kind", "api_index"),
    EntityKind.FUNCTION.value: ("definition_sites", "write_sites"),
    EntityKind.METHOD.value: ("definition_sites", "write_sites"),
    EntityKind.VARIABLE.value: ("layer", "rhs", "write_sites"),
    EntityKind.MACRO.value: ("value", "value_expr", "definition", "layer"),
    EntityKind.COMPILE_VAR.value: ("value", "value_expr", "origin", "layer"),
    EntityKind.PREDICATE.value: (
        "predicate_role",
        "class",
        "priority",
        "arch_expr",
        "is_capable_file",
        "is_capable_line",
    ),
}

_DROP_ATTRS = frozenset({"type_text", "snippet"})


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Entity):
        return {
            "id": value.id,
            "kind": value.kind_name(),
            "name": value.name,
            "status": value.status,
            "file": value.file,
            "line_start": value.line_start,
            "line_end": value.line_end,
            "attrs": dict(value.attrs),
        }
    if not isinstance(value, dict):
        return {}
    attrs = value.get("attrs")
    if not isinstance(attrs, dict):
        data = value.get("data")
        attrs = data if isinstance(data, dict) else {
            k: v
            for k, v in value.items()
            if k
            not in {
                "id",
                "kind",
                "name",
                "status",
                "confidence",
                "file",
                "line_start",
                "line_end",
                "span_file",
                "evidence_tier",
                "located_via",
                "why",
                "facts",
                "distance",
            }
        }
    return {
        "id": value.get("id") or "",
        "kind": value.get("kind") or "",
        "name": value.get("name") or "",
        "status": value.get("status") or "",
        "file": value.get("file") or "",
        "line_start": value.get("line_start") or 0,
        "line_end": value.get("line_end") or 0,
        "attrs": dict(attrs or {}),
    }


_LONG_EXPR_KEYS = frozenset({"rhs", "expression", "guard", "predicate", "condition", "value_expr", "definition"})
_FULL_FACT_KEYS = frozenset(
    {
        "packing_value_sites",
        "host_writer_sites",
        "value_defining_sites",
        "producer_sites",
        "check_sites",
        "definition_sites",
        "fused_outer_candidates",
        "local_aliases",
        "write_sites",
    }
)
_ARG_CUT_RE = re.compile(r";|\bif\b")


def _sanitize_expr(text: str) -> str:
    cut = _ARG_CUT_RE.search(text)
    if cut:
        text = text[: cut.start()].rstrip()
    return text[:120]


def _short_args(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return value
    if isinstance(value, str):
        return value[:120] if len(value) > 120 else value
    if isinstance(value, list):
        return [_short_args(item, depth=depth + 1) for item in value[:8]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _LONG_EXPR_KEYS and isinstance(item, str):
                out[key] = item[:120]
            else:
                out[key] = _short_args(item, depth=depth + 1)
        return out
    return value


def _first_rhs(attrs: dict[str, Any]) -> str:
    for key in ("value_defining_sites", "host_writer_sites", "producer_sites"):
        for site in attrs.get(key) or []:
            if not isinstance(site, dict):
                continue
            for field in ("rhs", "expression"):
                text = str(site.get(field) or "").strip()
                if text:
                    return text[:120]
    return str(attrs.get("rhs") or attrs.get("default_initializer") or "").strip()[:120]


def project_entity(
    value: Any,
    *,
    why: str = "",
    distance: int | None = None,
    require_span_for_branch: bool = True,
) -> dict[str, Any] | None:
    """Project one CodeMap entity into a skill-useful evidence hit."""
    raw = _as_mapping(value)
    kind = str(raw.get("kind") or "")
    file = str(raw.get("file") or "")
    line_start = int(raw.get("line_start") or 0)
    line_end = int(raw.get("line_end") or line_start or 0)
    if kind == EntityKind.BRANCH.value and require_span_for_branch and (not file or line_start <= 0):
        return None
    attrs = dict(raw.get("attrs") or {})
    facts: dict[str, Any] = {}
    for key in _KIND_FACTS.get(kind, ()):
        if key not in attrs or attrs[key] in (None, "", [], {}):
            continue
        value = attrs[key]
        if key in _FULL_FACT_KEYS:
            facts[key] = value
        elif key in _LONG_EXPR_KEYS and isinstance(value, str):
            facts[key] = value
        else:
            facts[key] = _short_args(value)
    if kind == EntityKind.OPERATION.value and "args" in facts:
        args = facts["args"]
        if isinstance(args, str):
            facts["args"] = _sanitize_expr(args)
        elif isinstance(args, list):
            facts["args"] = [
                _sanitize_expr(item) if isinstance(item, str) else item for item in args[:8]
            ]
    if kind in {EntityKind.TILING_FIELD.value, EntityKind.FIELD.value} and not facts.get("rhs"):
        rhs = _first_rhs(attrs)
        if rhs:
            facts["rhs"] = rhs
    hit: dict[str, Any] = {
        "id": raw.get("id") or "",
        "kind": kind,
        "name": raw.get("name") or "",
        "status": raw.get("status") or "",
        "file": file,
        "line_start": line_start,
        "line_end": line_end,
        "why": why,
        "facts": facts,
    }
    if distance is not None:
        hit["distance"] = int(distance)
    return hit


def project_record(value: Any, *, why: str = "") -> dict[str, Any]:
    """Always return a dict (CE anchors must not disappear)."""
    hit = project_entity(value, why=why, require_span_for_branch=False)
    if hit is not None:
        return hit
    raw = _as_mapping(value)
    return {
        "id": raw.get("id") or "",
        "kind": raw.get("kind") or "",
        "name": raw.get("name") or "",
        "status": raw.get("status") or "",
        "file": raw.get("file") or "",
        "line_start": int(raw.get("line_start") or 0),
        "line_end": int(raw.get("line_end") or 0),
        "why": why or "seed",
        "facts": {},
    }


def project_relation(value: Any) -> dict[str, Any]:
    if hasattr(value, "kind_name"):
        return {
            "id": getattr(value, "id", ""),
            "kind": value.kind_name(),
            "src": getattr(value, "src", ""),
            "dst": getattr(value, "dst", ""),
            "status": getattr(value, "status", ""),
        }
    if isinstance(value, dict):
        return {
            "id": value.get("id") or "",
            "kind": value.get("kind") or "",
            "src": value.get("src") or "",
            "dst": value.get("dst") or "",
            "status": value.get("status") or "",
            "evidence_tier": value.get("evidence_tier") or "",
        }
    return {}


def skill_bucket(hit: dict[str, Any]) -> str:
    kind = str(hit.get("kind") or "")
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    name = canonical_sync_name(str(facts.get("callee") or hit.get("name") or ""))
    if kind in {EntityKind.TILING_KEY.value, EntityKind.TEMPLATE.value, EntityKind.PREDICATE.value}:
        return "dispatch"
    if kind in {EntityKind.TILING_FIELD.value, EntityKind.TILING_DATA.value}:
        return "layout"
    if kind in {EntityKind.BUFFER.value, EntityKind.REGISTER.value}:
        return "memory"
    if kind in {EntityKind.PIPE.value, EntityKind.EVENT.value, EntityKind.QUEUE.value}:
        return "sync"
    if kind == EntityKind.OPERATION.value:
        if name in _PRECISION_CALLEES:
            return "precision"
        if name in _SYNC_CALLEES or is_sync_root(name):
            return "sync"
        return "memory"
    if kind in {EntityKind.INPUT.value, EntityKind.OUTPUT.value}:
        return "contract"
    if kind in {EntityKind.BRANCH.value, EntityKind.KERNEL.value}:
        return "dispatch"
    return "other"


def bucket_hits(hits: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        "dispatch": [],
        "layout": [],
        "memory": [],
        "sync": [],
        "precision": [],
        "contract": [],
        "other": [],
    }
    for hit in hits:
        out.setdefault(skill_bucket(hit), []).append(hit)
    return {key: rows for key, rows in out.items() if rows}


def is_kernel_api_name(name: str) -> bool:
    short = canonical_sync_name(name)
    return short in _KERNEL_API_CALLEES or str(name or "").split("::")[-1] in _KERNEL_API_CALLEES


def is_tque_api_name(name: str) -> bool:
    return str(name or "").split("::")[-1] in _TQUE_CALLEES


def is_flag_sync_api_name(name: str) -> bool:
    return is_flag_sync(name)


def field_edge_kinds() -> frozenset[str]:
    return _FIELD_EDGE_KINDS


def drop_noise_attr(key: str) -> bool:
    return str(key) in _DROP_ATTRS


_SURFACE_KINDS = frozenset(
    {
        EntityKind.BUFFER.value,
        EntityKind.QUEUE.value,
        EntityKind.PIPE.value,
        EntityKind.EVENT.value,
        EntityKind.REGISTER.value,
    }
)


def surface_facts(kind: str, facts: dict[str, Any] | None) -> dict[str, Any]:
    """Identity attrs agents need on the name card, not only inside ``facts``."""
    if str(kind or "") not in _SURFACE_KINDS or not isinstance(facts, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _KIND_FACTS.get(str(kind), ()):
        if key in _FULL_FACT_KEYS:
            continue
        value = facts.get(key)
        if value in (None, "", [], {}):
            continue
        out[key] = value
    return out
