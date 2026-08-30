# -*- coding: utf-8 -*-
"""Symbol Bundle projector: attrs-first, edge fallback. Shared with evidence."""
from __future__ import annotations

import json
import re
from typing import Any

from ascendc_codemap_mcp.engine.ir.entity import EntityKind

KIND_FACTS: dict[str, tuple[str, ...]] = {
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
        "producer_sites",
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
    EntityKind.EVENT.value: (
        "identity",
        "scope",
        "event_type",
        "mechanism",
        "cross_core",
        "paired",
        "signal_count",
        "await_count",
    ),
    EntityKind.QUEUE.value: ("identity", "scope", "type_name", "tposition", "memory_space"),
    EntityKind.BRANCH.value: (
        "predicate",
        "condition",
        "branch_kind",
        "layer",
        "function",
        "dimensions",
        "operators",
        "literals",
        "references",
        "enum_values",
        "expr_ast",
    ),
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
        "entry_role",
        "class",
        "priority",
        "arch_expr",
        "is_capable_file",
        "is_capable_line",
        "operators",
        "literals",
        "references",
        "enum_values",
    ),
}

FULL_FACT_KEYS = frozenset(
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

_SITE_ATTR_KEYS = (
    "value_defining_sites",
    "host_writer_sites",
    "producer_sites",
    "packing_value_sites",
    "write_sites",
    "definition_sites",
)

_LOG_GUARD_RE = re.compile(r"CheckLogLevel|\bOP_LOG|Dlog", re.IGNORECASE)
_LOOP_GUARD_RE = re.compile(r"^\s*(for|while|do)\b", re.IGNORECASE)
_CONFIRMED = ("confirmed", "extracted", "verified")

_LAYOUT_LABEL = {
    "dqWorkSpaceOffset": "dq",
    "dkWorkSpaceOffset": "dk",
    "dvWorkSpaceOffset": "dv",
    "dropMaskGmOffset": "dropMask",
    "sfmgWorkSpaceOffset": "sfmg",
    "dsinkWorkSpaceOffset": "dsink",
    "deterWorkSpaceOffset": "deterWorkSpace",
    "deterGmOffset": "deterGm",
}


def _parse_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _leaf(name: str) -> str:
    return str(name or "").replace(".", "::").rsplit("::", 1)[-1]


def _norm_path(file: str) -> str:
    return str(file or "").replace("\\", "/")


_PARTITION_SITE_KEYS = (
    "value_defining_sites",
    "host_writer_sites",
    "producer_sites",
    "packing_value_sites",
)


def data_has_partition_sites(data: Any) -> bool:
    blob = data if isinstance(data, dict) else _parse_json(data)
    for key in _PARTITION_SITE_KEYS:
        if blob.get(key):
            return True
    return False


def data_has_site_attrs(data: Any) -> bool:
    blob = data if isinstance(data, dict) else _parse_json(data)
    for key in _SITE_ATTR_KEYS:
        if blob.get(key):
            return True
    return False


def first_rhs(attrs: dict[str, Any]) -> str:
    for key in ("value_defining_sites", "host_writer_sites", "producer_sites"):
        for site in attrs.get(key) or []:
            if not isinstance(site, dict):
                continue
            for field in ("rhs", "expression"):
                text = str(site.get(field) or "").strip()
                if text:
                    return text[:120]
    return str(attrs.get("rhs") or attrs.get("default_initializer") or "").strip()[:120]


def is_log_or_loop_guard(name: str, data: dict[str, Any] | None = None) -> bool:
    raw = str(name or "")
    if _LOG_GUARD_RE.search(raw):
        return True
    kind = str((data or {}).get("branch_kind") or "").lower()
    if kind in {"for", "while", "loop", "do"}:
        return True
    if _LOOP_GUARD_RE.search(raw):
        return True
    return False


def _site_when(site: dict[str, Any]) -> str:
    guards = site.get("guards") if isinstance(site.get("guards"), list) else []
    for item in guards:
        if isinstance(item, dict):
            cond = str(item.get("condition") or item.get("name") or "").strip()
        else:
            cond = str(item or "").strip()
        if cond and not is_log_or_loop_guard(cond):
            return cond
    return str(site.get("when") or "").strip()


def normalize_sites(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for site in raw or []:
        if not isinstance(site, dict):
            continue
        file = _norm_path(str(site.get("file") or ""))
        line = int(site.get("line") or 0)
        rhs = str(site.get("rhs") or site.get("expression") or "").strip()
        if line <= 0:
            continue
        key = (file, line, rhs)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "file": file,
                "line": line,
                "rhs": rhs,
                "function": str(site.get("function") or ""),
                "when": _site_when(site),
                "receiver": str(site.get("receiver") or ""),
                "expression": str(site.get("expression") or ""),
            }
        )
    return out


def _is_kernel_path(file: str) -> bool:
    blob = _norm_path(file)
    return "/op_kernel/" in f"/{blob}" or blob.startswith("op_kernel/")


def _is_host_path(file: str) -> bool:
    blob = _norm_path(file)
    return "/op_host/" in f"/{blob}" or blob.startswith("op_host/")


def fold_branches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse BRANCH rows by (name, function, layer)."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name or is_log_or_loop_guard(name, row.get("data") if isinstance(row.get("data"), dict) else None):
            continue
        fn = str(row.get("function") or "")
        layer = str(row.get("layer") or "")
        key = (name, fn, layer)
        if key not in groups:
            groups[key] = {
                "name": name,
                "function": fn,
                "layer": layer,
                "count": 0,
                "lines": [],
            }
            order.append(key)
        item = groups[key]
        item["count"] += 1
        line = int(row.get("line") or 0)
        if line and line not in item["lines"]:
            item["lines"].append(line)
    out = []
    for key in order:
        item = groups[key]
        item["lines"].sort()
        out.append(item)
    return out


def layout_label(name: str) -> str:
    leaf = _leaf(name)
    return _LAYOUT_LABEL.get(leaf, leaf)


def _load_entities(query: Any, ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    ph = ",".join("?" for _ in ids)
    with query._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, kind, name, file, line_start, data
            FROM entity WHERE id IN ({ph})
            """,
            ids,
        ).fetchall()
    by_id = {str(r[0]): r for r in rows}
    out: list[dict[str, Any]] = []
    for eid in ids:
        row = by_id.get(eid)
        if row is None:
            continue
        out.append(
            {
                "id": str(row[0]),
                "kind": str(row[1] or ""),
                "name": str(row[2] or ""),
                "file": _norm_path(str(row[3] or "")),
                "line": int(row[4] or 0),
                "data": _parse_json(row[5]),
            }
        )
    return out


def _collect_attr_sites(entities: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for ent in entities:
        for site in normalize_sites((ent.get("data") or {}).get(key)):
            mark = (site["file"], site["line"], site["rhs"])
            if mark in seen:
                continue
            seen.add(mark)
            sites.append(site)
    return sites


def _kernel_consumers(query: Any, field_ids: list[str]) -> list[dict[str, Any]]:
    if not field_ids:
        return []
    ph = ",".join("?" for _ in field_ids)
    with query._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT src.name, src.file, src.line_start, src.kind, r.data
            FROM relation r
            JOIN entity src ON src.id = r.src
            WHERE r.dst IN ({ph})
              AND r.kind IN ('READS', 'CALLS_UNDER_GUARD')
              AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
            """,
            field_ids,
        ).fetchall()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, file, line_start, kind, raw in rows:
        kind_u = str(kind or "").upper()
        if kind_u not in {EntityKind.METHOD.value, EntityKind.FUNCTION.value}:
            continue
        nm = str(name or "")
        leaf = _leaf(nm)
        if not leaf or leaf.lower() in {"min", "max"}:
            continue
        if _LOG_GUARD_RE.search(nm):
            continue
        data = _parse_json(raw)
        file_n = _norm_path(str(data.get("file") or file or ""))
        line = int(data.get("line") or 0)
        if line <= 0:
            for site in data.get("sites") or []:
                if isinstance(site, dict) and int(site.get("line") or 0) > 0:
                    line = int(site["line"])
                    file_n = _norm_path(str(site.get("file") or file_n))
                    break
        if line <= 0:
            line = int(line_start or 0)
            file_n = _norm_path(str(file or file_n))
        if not file_n or line <= 0:
            continue
        if not _is_kernel_path(file_n):
            continue
        if leaf in seen:
            continue
        seen.add(leaf)
        out.append({"name": leaf, "file": file_n, "line": line, "qualified": nm})
    return out


def _consumed_names(query: Any, field_ids: list[str]) -> list[str]:
    try:
        return list(query._consumed_by_names(field_ids) or [])
    except Exception:  # noqa: BLE001
        return []


def _workspace_layout(query: Any, seed_ids: list[str], seed_name: str) -> list[dict[str, Any]]:
    ids = list(seed_ids)
    leaf = _leaf(seed_name)
    with query._connect() as conn:
        if not ids and leaf:
            rows = conn.execute(
                """
                SELECT id FROM entity
                WHERE kind IN ('VARIABLE', 'FIELD', 'FUNCTION', 'METHOD')
                  AND (
                    name = ? COLLATE NOCASE
                    OR name LIKE '%::' || ? COLLATE NOCASE
                    OR name LIKE '%.' || ? COLLATE NOCASE
                  )
                LIMIT 24
                """,
                (leaf, leaf, leaf),
            ).fetchall()
            ids = [str(r[0]) for r in rows if r[0]]
        if not ids:
            return []
        ph = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""
            SELECT r.data, src.name, dst.name, dst.file, dst.line_start, src.file, src.line_start
            FROM relation r
            JOIN entity src ON src.id = r.src
            JOIN entity dst ON dst.id = r.dst
            WHERE r.kind = 'ALLOCATES'
              AND (r.src IN ({ph}) OR r.dst IN ({ph}))
              AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified', '')
            """,
            (*ids, *ids),
        ).fetchall()
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw, src_name, dst_name, dst_file, dst_line, src_file, src_line in rows:
        data = _parse_json(raw)
        role = str(data.get("role") or "")
        if role not in {"workspace_offset", "offset_expr", "workspace_accum", "workspace_size"}:
            continue
        if role not in {"workspace_offset", "offset_expr"}:
            continue
        label = layout_label(str(dst_name or ""))
        if label in {"workspaceSize", "workspace", "qSize"}:
            continue
        line = int(data.get("line") or 0)
        if line <= 0:
            sites = data.get("sites") if isinstance(data.get("sites"), list) else []
            for site in sites:
                if isinstance(site, dict) and int(site.get("line") or 0) > 0:
                    line = int(site["line"])
                    break
        if line <= 0:
            line = int(src_line or dst_line or 0)
        key = (label, line)
        if not label or line <= 0 or key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "label": label,
                "line": line,
                "name": _leaf(str(dst_name or "")),
                "file": _norm_path(str(dst_file or src_file or "")),
                "role": role,
            }
        )
    items.sort(key=lambda r: (int(r["line"]), str(r["label"])))
    # Prefer workspace_offset over offset_expr duplicates.
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in items:
        lab = item["label"]
        if lab not in best:
            best[lab] = item
            order.append(lab)
        elif item["role"] == "workspace_offset" and best[lab]["role"] != "workspace_offset":
            best[lab] = item
    return [best[k] for k in order]


def _resource_projection(query: Any, entity_id: str, kind: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not entity_id:
        return out
    with query._connect() as conn:
        rows = conn.execute(
            """
            SELECT r.kind, r.data, e.kind, e.name, e.file, e.line_start, e.data
            FROM relation r
            JOIN entity e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
            WHERE (r.src = ? OR r.dst = ?)
              AND r.kind IN ('BACKED_BY', 'SIGNALS', 'AWAITS', 'PRECEDES', 'WRAPS', 'ALLOCATES')
            """,
            (entity_id, entity_id, entity_id),
        ).fetchall()
    backing: list[dict[str, Any]] = []
    sync_edges: list[dict[str, Any]] = []
    order: list[dict[str, Any]] = []
    for rkind, rdata, ekind, name, file, line, edata in rows:
        rel = _parse_json(rdata)
        if str(rkind) == "BACKED_BY":
            backing.append(
                {
                    "name": str(name or ""),
                    "kind": str(ekind or ""),
                    "physical_space": str(rel.get("physical_space") or ""),
                    "via": str(rel.get("via") or ""),
                    "file": _norm_path(str(rel.get("file") or file or "")),
                    "line": int(rel.get("line") or line or 0),
                }
            )
        elif str(rkind) in {"SIGNALS", "AWAITS"}:
            sync_edges.append(
                {
                    "rel": str(rkind),
                    "name": str(name or ""),
                    "file": _norm_path(str(file or "")),
                    "line": int(line or 0),
                }
            )
        elif str(rkind) == "PRECEDES" and str(rel.get("via") or "") in {"pipe_destroy", ""}:
            order.append(
                {
                    "name": str(name or ""),
                    "via": str(rel.get("via") or ""),
                    "file": _norm_path(str(rel.get("file") or file or "")),
                    "line": int(rel.get("line") or line or 0),
                }
            )
    if backing:
        backing.sort(key=lambda r: (int(r.get("line") or 0), str(r.get("name") or "")))
        ping = [b for b in backing if "ping" in str(b.get("name") or "").lower()]
        out["backing"] = (ping or backing)[:1]
    if kind == EntityKind.EVENT.value or sync_edges:
        out["sync"] = {
            "paired": data.get("paired"),
            "signal_count": data.get("signal_count"),
            "await_count": data.get("await_count"),
            "mechanism": data.get("mechanism"),
            "edges": sync_edges,
        }
    if order:
        out["order"] = order
    return out


def _control_projection(query: Any, entity_id: str) -> dict[str, Any]:
    if not entity_id:
        return {}
    with query._connect() as conn:
        rows = conn.execute(
            """
            SELECT r.kind, e.kind, e.name, e.file, e.line_start, e.data
            FROM relation r
            JOIN entity e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
            WHERE (r.src = ? OR r.dst = ?)
              AND r.kind IN ('GUARDED_BY', 'CONTROLS')
              AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
            """,
            (entity_id, entity_id, entity_id),
        ).fetchall()
    guarded: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for rkind, ekind, name, file, line, raw in rows:
        data = _parse_json(raw)
        nm = str(name or "")
        if is_log_or_loop_guard(nm, data):
            continue
        row = {
            "name": nm,
            "kind": str(ekind or ""),
            "file": _norm_path(str(file or "")),
            "line": int(line or 0),
            "function": str(data.get("function") or ""),
            "layer": str(data.get("layer") or ""),
            "data": data,
        }
        if str(rkind) == "GUARDED_BY":
            guarded.append(row)
        else:
            controls.append(row)
    out: dict[str, Any] = {}
    folded = fold_branches(guarded)
    if folded:
        out["guarded_by"] = folded
    folded_c = fold_branches(controls) if any(r["kind"] == EntityKind.BRANCH.value for r in controls) else []
    ctrl_names: list[str] = []
    seen: set[str] = set()
    for row in controls:
        if row["kind"] == EntityKind.BRANCH.value:
            continue
        nm = _leaf(row["name"])
        if not nm or nm in seen:
            continue
        seen.add(nm)
        ctrl_names.append(nm)
    if folded_c:
        out["controls_folded"] = folded_c
    if ctrl_names:
        out["controls"] = ctrl_names
    return out


def build_symbol_bundle(query: Any, ident: str) -> dict[str, Any] | None:
    """Attrs-first Symbol Bundle for resolve(symbol)."""
    leaf = _leaf(ident)
    if not leaf:
        return None
    buckets = query._field_ids_named(ident)
    if not isinstance(buckets, dict):
        ids = [str(x) for x in (buckets or []) if x]
        buckets = {EntityKind.FIELD.value: ids}
    tiling_ids = list(buckets.get(EntityKind.TILING_FIELD.value) or [])
    field_ids = list(buckets.get(EntityKind.FIELD.value) or [])
    key_ids = list(buckets.get(EntityKind.TILING_KEY.value) or [])
    tiling = _load_entities(query, tiling_ids)
    fields = _load_entities(query, field_ids)
    keys = _load_entities(query, key_ids)

    host_defs = _collect_attr_sites(tiling, "value_defining_sites")
    transport = _collect_attr_sites(tiling, "host_writer_sites")
    assignments = _collect_attr_sites(fields, "producer_sites")
    if not assignments:
        assignments = _collect_attr_sites(fields, "write_sites")

    all_fieldish = tiling_ids + field_ids

    consumers = _kernel_consumers(query, tiling_ids or all_fieldish)
    consumed = _consumed_names(query, all_fieldish or tiling_ids or field_ids)
    layout_ids = all_fieldish
    workspace = _workspace_layout(query, layout_ids, ident)
    if not workspace and leaf.lower() in {"workspacesize", "workspace"}:
        workspace = _workspace_layout(query, [], ident)

    resource: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    primary_ids = tiling_ids or field_ids or key_ids
    if primary_ids:
        primary = (_load_entities(query, primary_ids[:1]) or [{}])[0]
        kind = str(primary.get("kind") or "")
        if kind in {
            EntityKind.BUFFER.value,
            EntityKind.QUEUE.value,
            EntityKind.PIPE.value,
            EntityKind.EVENT.value,
            EntityKind.REGISTER.value,
        }:
            resource = _resource_projection(query, str(primary.get("id") or ""), kind, primary.get("data") or {})
        if kind in {EntityKind.FIELD.value, EntityKind.TILING_FIELD.value, EntityKind.BRANCH.value}:
            controls = _control_projection(query, str(primary.get("id") or ""))
        if kind == EntityKind.VARIABLE.value:
            workspace = workspace or _workspace_layout(query, [str(primary.get("id") or "")], ident)

    if not any(
        (
            host_defs,
            transport,
            consumers,
            assignments,
            workspace,
            resource,
            controls,
            consumed,
        )
    ):
        return None
    return {
        "host_value_definitions": host_defs,
        "transport": transport,
        "kernel_consumers": consumers,
        "assignments": assignments,
        "consumed_by": consumed,
        "workspace_layout": workspace,
        "resource": resource,
        "controls": controls,
        "tiling_keys": [e.get("name") for e in keys if e.get("name")],
    }


def attach_entity_projections(query: Any, entity: dict[str, Any]) -> dict[str, Any]:
    """Per-card resource / control facets for BUFFER/EVENT/FIELD/… seeds."""
    eid = str(entity.get("id") or entity.get("_entity_id") or "")
    kind = str(entity.get("kind") or "")
    data = entity.get("data") if isinstance(entity.get("data"), dict) else {}
    if not data and eid:
        loaded = _load_entities(query, [eid])
        if loaded:
            data = loaded[0].get("data") or {}
            kind = kind or str(loaded[0].get("kind") or "")
    out: dict[str, Any] = {}
    if kind in {
        EntityKind.BUFFER.value,
        EntityKind.QUEUE.value,
        EntityKind.PIPE.value,
        EntityKind.EVENT.value,
        EntityKind.REGISTER.value,
        EntityKind.VARIABLE.value,
    }:
        res = _resource_projection(query, eid, kind, data)
        if kind in {EntityKind.VARIABLE.value, EntityKind.FUNCTION.value, EntityKind.METHOD.value}:
            layout = _workspace_layout(query, [eid] if eid else [], str(entity.get("name") or ""))
            if layout:
                res["workspace_layout"] = layout
        if res:
            out["resource"] = res
    if kind in {EntityKind.FIELD.value, EntityKind.TILING_FIELD.value, EntityKind.BRANCH.value}:
        ctrl = _control_projection(query, eid)
        if ctrl:
            out["controls_proj"] = ctrl
    return out
