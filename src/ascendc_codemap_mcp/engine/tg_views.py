# -*- coding: utf-8 -*-
"""Project TG-facing view blobs from a CodeMap.

These projections are disposable views embedded in the single ``.uo`` product.
TG must not require a parallel YAML export tree to consume them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind


def graph_fingerprint(codemap: CodeMap) -> str:
    ent_kinds = sorted(Counter(e.kind_name() for e in codemap.entities.values()).items())
    rel_kinds = sorted(Counter(r.kind_name() for r in codemap.relations.values()).items())
    key_names = sorted(
        e.name for e in codemap.by_kind(EntityKind.TILING_KEY) if e.attrs.get("source_declared")
    )
    payload = {
        "op": codemap.op_name,
        "arch": codemap.architecture,
        "entities": ent_kinds,
        "relations": rel_kinds,
        "tiling_keys": key_names,
        "legal_key_count": int(codemap.meta.get("legal_key_count") or 0),
        "args_sel_group_count": int(codemap.meta.get("args_sel_group_count") or 0),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def project_operator_graph(codemap: CodeMap, *, fingerprint: str = "") -> dict[str, Any]:
    fp = fingerprint or graph_fingerprint(codemap)
    return {
        "schema": "uo-operator-graph/v1",
        "fingerprint": fp,
        "op_name": codemap.op_name,
        "architecture": codemap.architecture,
        "node_count": len(codemap.entities),
        "edge_count": len(codemap.relations),
        "entities_by_kind": dict(Counter(e.kind_name() for e in codemap.entities.values())),
        "relations_by_kind": dict(Counter(r.kind_name() for r in codemap.relations.values())),
        "closure": {
            "legal_key_count": int(codemap.meta.get("legal_key_count") or 0),
            "args_sel_group_count": int(codemap.meta.get("args_sel_group_count") or 0),
            "tiling_key_host_packing": codemap.meta.get("host_tiling_key_packing") or {},
            "has_strict_kernel_tiling_closure": bool(codemap.meta.get("has_strict_kernel_tiling_closure")),
        },
        "nodes": [],
        "edges": [],
    }


def project_tg_host_view(codemap: CodeMap, *, fingerprint: str = "") -> dict[str, Any]:
    fp = fingerprint or graph_fingerprint(codemap)
    keys = sorted(codemap.by_kind(EntityKind.TILING_KEY), key=lambda e: int(e.attrs.get("decl_order") or 0))
    fields: list[dict[str, Any]] = []
    predicates: list[dict[str, Any]] = []
    declared_keys: dict[str, Any] = {}
    incoming: dict[str, list[Any]] = defaultdict(list)
    for rel in codemap.relations.values():
        incoming[rel.dst].append(rel)

    for key in keys:
        packing = list(key.attrs.get("host_packing_expressions") or [])
        declared_keys[key.name] = {
            "decl_order": key.attrs.get("decl_order"),
            "bit_offset": key.attrs.get("bit_offset"),
            "bit_width": key.attrs.get("bit_width") or key.attrs.get("bw"),
            "allowed_values": list(key.attrs.get("allowed_values") or key.attrs.get("value_domain") or []),
            "packing": packing,
        }
        host_syms = _host_symbols_for_key(codemap, key, incoming)
        for sym in host_syms:
            field_preds = _predicates_for_entity(codemap, sym, incoming)
            predicates.extend(field_preds)
            fields.append({
                "name": sym.name,
                "kind": "key_dim_host",
                "tiling_key": key.name,
                "exactness": sym.attrs.get("exactness") or "",
                "writers": list(sym.attrs.get("producer_sites") or []),
                "reads": _read_roots(codemap, sym, incoming),
                "packing": packing,
                "rooted": bool(sym.attrs.get("rooted_by_current_source")),
                "entity_id": sym.id,
            })
        if not host_syms and packing:
            fields.append({
                "name": key.name,
                "kind": "key_dim",
                "tiling_key": key.name,
                "exactness": "",
                "writers": [],
                "reads": [],
                "packing": packing,
                "rooted": False,
                "entity_id": key.id,
            })

    return {
        "schema": "tg-host-view/v1",
        "compat_schema": "codemap/v2",
        "source": {"graph_fingerprint": fp, "generated_by": "uo_init.tg_views.project_tg_host_view", "authority": "uo_codemap", "role": "tg_host_projection"},
        "fields": fields,
        "predicates": predicates,
        "declared_keys": declared_keys,
        "platform_gates": [],
    }


_COMPARE_RE = re.compile(
    r"(?P<field>[A-Za-z_]\w*)\s*(?P<op>==|!=|<=|>=|<|>)\s*(?P<value>-?\d+|true|false)",
    re.IGNORECASE,
)
_RISK_TOKENS = ("overflow", "align", "alignment", "tail", "zero", "minimum", "maximum", "min", "max")


def _infer_stage(attrs: dict[str, Any], condition: str, key_dims: list[str], td_fields: list[str]) -> str:
    explicit = str(attrs.get("stage") or attrs.get("evaluation_stage") or "").strip().lower()
    if explicit in {"constexpr", "runtime", "compile_time", "host"}:
        return "constexpr" if explicit in {"constexpr", "compile_time"} else "runtime"
    # Constexpr-like when only tiling-key dims appear; runtime when TilingData is involved.
    if td_fields and not key_dims:
        return "runtime"
    if td_fields and key_dims:
        return "runtime"
    if "constexpr" in condition or condition.startswith("std::is_same") or "sizeof" in condition:
        return "constexpr"
    if key_dims and not td_fields:
        return "constexpr"
    if td_fields:
        return "runtime"
    return explicit or "runtime"


def _value_classes_from_expressions(field: str, expressions: list[str]) -> list[dict[str, Any]]:
    """Extract semantic boundary predicates; never enumerate full domains."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for expr in expressions:
        for match in _COMPARE_RE.finditer(str(expr or "")):
            if match.group("field") != field:
                continue
            op = match.group("op")
            raw = match.group("value")
            value: Any
            if raw.lower() in {"true", "false"}:
                value = raw.lower() == "true"
            else:
                try:
                    value = int(raw)
                except ValueError:
                    value = raw
            pred = f"{field} {op} {value}"
            if pred in seen:
                continue
            seen.add(pred)
            out.append({"field": field, "op": op, "value": value, "predicate": pred})
            # Companion polarity for equality / inequality and threshold cuts.
            if op == "==":
                companion = f"{field} != {value}"
                if companion not in seen:
                    seen.add(companion)
                    out.append({"field": field, "op": "!=", "value": value, "predicate": companion})
            elif op == "!=":
                companion = f"{field} == {value}"
                if companion not in seen:
                    seen.add(companion)
                    out.append({"field": field, "op": "==", "value": value, "predicate": companion})
            elif op == ">=":
                companion = f"{field} < {value}"
                if companion not in seen:
                    seen.add(companion)
                    out.append({"field": field, "op": "<", "value": value, "predicate": companion})
            elif op == "<":
                companion = f"{field} >= {value}"
                if companion not in seen:
                    seen.add(companion)
                    out.append({"field": field, "op": ">=", "value": value, "predicate": companion})
            elif op == ">":
                companion = f"{field} <= {value}"
                if companion not in seen:
                    seen.add(companion)
                    out.append({"field": field, "op": "<=", "value": value, "predicate": companion})
            elif op == "<=":
                companion = f"{field} > {value}"
                if companion not in seen:
                    seen.add(companion)
                    out.append({"field": field, "op": ">", "value": value, "predicate": companion})
    return out


def _classify_tilingdata_field(
    *,
    name: str,
    attrs: dict[str, Any],
    readers: list[dict[str, Any]],
    writers: list[Any],
    value_classes: list[dict[str, Any]],
) -> str:
    explicit = str(attrs.get("coverage_class") or attrs.get("field_class") or "").strip().lower()
    if explicit in {"control", "boundary", "derived", "payload"}:
        return explicit
    if attrs.get("derived_from") or attrs.get("derived") or str(attrs.get("exactness") or "") == "derived":
        return "derived"
    reader_text = " ".join(str(r.get("expression") or "") for r in readers).lower()
    name_l = name.lower()
    if value_classes or any(tok in reader_text for tok in ("if", "?", "for", "while", "switch")):
        if any(tok in name_l or tok in reader_text for tok in ("tail", "inner", "outer", "split", "block", "align")):
            return "boundary"
        return "control"
    if any(tok in name_l or tok in reader_text for tok in _RISK_TOKENS):
        return "payload"
    if writers and not readers:
        return "payload"
    if readers and not value_classes:
        return "payload"
    return "payload"


def project_kernel_view(codemap: CodeMap, *, fingerprint: str = "") -> dict[str, Any]:
    """Project kernel branch evidence required by TG certification (schema v2)."""
    fp = fingerprint or graph_fingerprint(codemap)
    key_names = [e.name for e in codemap.by_kind(EntityKind.TILING_KEY)]
    td_names = [e.name for e in codemap.by_kind(EntityKind.TILING_FIELD)]
    rows: list[dict[str, Any]] = []
    for ent in codemap.by_kind(EntityKind.BRANCH):
        attrs = ent.attrs or {}
        condition = str(attrs.get("condition") or attrs.get("predicate") or attrs.get("expression") or ent.name or "")
        dims = list(attrs.get("dimensions") or attrs.get("tiling_key_dims") or [])
        if not dims and condition:
            dims = [
                name
                for name in key_names
                if name and name in condition and re.search(rf"\b{re.escape(name)}\b", condition)
            ]
        td_fields = list(attrs.get("tilingdata_fields") or attrs.get("tiling_data_fields") or [])
        if not td_fields and condition:
            td_fields = [
                name
                for name in td_names
                if name and name in condition and re.search(rf"\b{re.escape(name)}\b", condition)
            ]
        stage = _infer_stage(attrs, condition, dims, td_fields)
        key_specialization = {
            "tiling_key_dims": dims,
            "fixes_branch_when_constant": bool(dims) and stage == "constexpr",
        }
        rows.append({
            "id": ent.id,
            "name": ent.name,
            "stage": stage,
            "condition": condition,
            "dimensions": dims,
            "tilingdata_fields": td_fields,
            "key_specialization": key_specialization,
            "finite_predicate": attrs.get("finite_predicate"),
            "file": ent.file,
            "line": ent.line_start,
            "function": attrs.get("function") or attrs.get("owner") or "",
            "status": getattr(ent, "status", "extracted"),
        })
    return {
        "schema": "uo-kernel-view/v2",
        "compat_schema": "uo-kernel-view/v1",
        "source": {"graph_fingerprint": fp, "authority": "uo_codemap", "generated_by": "uo_init.tg_views.project_kernel_view"},
        "branches": rows,
    }


def project_tilingdata_view(codemap: CodeMap, *, fingerprint: str = "") -> dict[str, Any]:
    """Project TilingData ABI fields with A/B/C/D classification (schema v2)."""
    fp = fingerprint or graph_fingerprint(codemap)
    fields = list(codemap.by_kind(EntityKind.TILING_FIELD))
    readers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rel in codemap.relations.values():
        if rel.kind_name() != RelationKind.READS.value:
            continue
        dst = codemap.entities.get(rel.dst)
        src = codemap.entities.get(rel.src)
        if dst is None or dst.kind_name() != EntityKind.TILING_FIELD.value:
            continue
        readers[dst.id].append({
            "entity_id": rel.src,
            "function": src.name if src is not None else str(rel.attrs.get("function") or ""),
            "file": rel.attrs.get("file") or (src.file if src is not None else ""),
            "line": rel.attrs.get("line") or (src.line_start if src is not None else None),
            "expression": rel.attrs.get("expression") or "",
            "architecture": rel.attrs.get("architecture") or codemap.architecture,
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_writer: list[str] = []
    no_reader: list[str] = []
    class_counts: Counter[str] = Counter()
    for ent in fields:
        attrs = ent.attrs or {}
        owner = str(attrs.get("owner") or attrs.get("struct") or "TilingData")
        writers = list(attrs.get("host_writer_sites") or attrs.get("producer_sites") or [])
        rds = readers.get(ent.id, [])
        expressions = [str(r.get("expression") or "") for r in rds if r.get("expression")]
        for w in writers:
            if isinstance(w, dict) and w.get("expression"):
                expressions.append(str(w.get("expression")))
        value_classes = _value_classes_from_expressions(ent.name, expressions)
        field_class = _classify_tilingdata_field(
            name=ent.name, attrs=attrs, readers=rds, writers=writers, value_classes=value_classes
        )
        class_counts[field_class] += 1
        risk_markers = [tok for tok in _RISK_TOKENS if tok in ent.name.lower() or any(tok in str(e).lower() for e in expressions)]
        writer_formula = ""
        for w in writers:
            if isinstance(w, dict) and (w.get("expression") or w.get("formula")):
                writer_formula = str(w.get("expression") or w.get("formula"))
                break
        value_defining = list(attrs.get("value_defining_sites") or [])
        row = {
            "id": ent.id,
            "name": ent.name,
            "type": attrs.get("data_type") or attrs.get("type") or "",
            "ordinal": attrs.get("ordinal"),
            "array_extent": attrs.get("array_extent"),
            "writers": writers,
            "value_defining_sites": value_defining,
            "readers": rds,
            "file": ent.file,
            "line": ent.line_start,
            "field_class": field_class,
            "value_classes": value_classes,
            "writer_formula": writer_formula,
            "risk_markers": risk_markers,
            "coverage_priority": field_class in {"control", "boundary"}
            or (field_class == "payload" and bool(risk_markers)),
        }
        grouped[owner].append(row)
        if not writers:
            no_writer.append(ent.name)
        if not rds:
            no_reader.append(ent.name)

    structs = [
        {"name": name, "fields": sorted(rows, key=lambda x: (x.get("ordinal") is None, x.get("ordinal") or 0, x["name"]))}
        for name, rows in sorted(grouped.items())
    ]
    return {
        "schema": "uo-tilingdata-view/v2",
        "compat_schema": "uo-tilingdata-view/v1",
        "source": {"graph_fingerprint": fp, "authority": "uo_codemap", "generated_by": "uo_init.tg_views.project_tilingdata_view"},
        "structs": structs,
        "defects": {"no_writer": sorted(set(no_writer)), "no_reader": sorted(set(no_reader))},
        "class_counts": dict(class_counts),
        "classification": {
            "control": "participates in kernel if/loop/dispatch",
            "boundary": "affects loop count / tail / block split / offset",
            "derived": "fully determined by other values; never alone creates a case",
            "payload": "address/length/compute params; obligations only with risk markers",
        },
    }


def finalize_tg_views(codemap: CodeMap, *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge TPL blobs with all TG projections and stamp one graph identity.

    Callers that persist via ``write_codemap`` should finalize **after** any
    canonical mutations (e.g. dropping unproven edges); ``write_codemap``
    re-finalizes itself so commit order stays: semantic → finalize → digest →
    stamp provenance → validate → atomic write.
    """
    from ascendc_codemap_mcp.engine.projection_provenance import (
        canonical_graph_digest,
        stamp_all_views,
    )

    views = dict(existing or {})
    fp = graph_fingerprint(codemap)
    codemap.meta["graph_fingerprint"] = fp
    codemap.meta["canonical_graph_digest"] = canonical_graph_digest(codemap)
    if not codemap.meta.get("canonical_revision"):
        codemap.meta["canonical_revision"] = str(codemap.meta["canonical_graph_digest"])[:16]
    if isinstance(views.get("tiling/exhaustive_key_space.yaml"), dict):
        views["tiling/exhaustive_key_space.yaml"]["fingerprint"] = fp
    views["ir/operator_graph.yaml"] = project_operator_graph(codemap, fingerprint=fp)
    views["ir/tg_host_view.yaml"] = project_tg_host_view(codemap, fingerprint=fp)
    views["views/kernel.yaml"] = project_kernel_view(codemap, fingerprint=fp)
    views["views/tilingdata.yaml"] = project_tilingdata_view(codemap, fingerprint=fp)
    return stamp_all_views(views, codemap)


#: Hoisted out of `_host_symbols_for_key`: rebuilding it per candidate cost
#: 768k enum property reads on one operator, the largest single source of them.
_DERIVES = RelationKind.DERIVES.value


def _host_symbols_for_key(codemap: CodeMap, key: Any, incoming: dict[str, list[Any]]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    packing = [str(x) for x in (key.attrs.get("host_packing_expressions") or [])]
    tokens = {tok for expr in packing for tok in _extract_symbols(expr)}
    derive_from_key = {
        rel.src for rel in incoming.get(key.id, []) if rel.kind_name() == _DERIVES
    }
    mid_src = {rel.src for mid in derive_from_key for rel in incoming.get(mid, [])}
    # By kind rather than over every entity: this runs once per tiling key, and
    # the scan asked 67k entities what kind they were to keep two kinds.
    for ent in (*codemap.by_kind(EntityKind.FIELD), *codemap.by_kind(EntityKind.VARIABLE)):
        if not (ent.attrs.get("host_key_argument") or ent.attrs.get("producer_sites") or ent.name in tokens or any(ent.name.endswith(t) or t.endswith(ent.name) for t in tokens)):
            continue
        related = bool(ent.attrs.get("host_key_argument") and (str(ent.attrs.get("tiling_key") or "") == key.name or key.name in str(ent.attrs.get("host_key_dims") or "")))
        if ent.id in derive_from_key:
            related = True
        if not related and ent.id in mid_src:
            related = True
        if related or (ent.attrs.get("host_key_argument") and tokens and any(t in ent.name or ent.name in t for t in tokens)):
            if ent.id not in seen:
                seen.add(ent.id)
                out.append(ent)
    return out


def _predicates_for_entity(codemap: CodeMap, ent: Any, incoming: dict[str, list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in incoming.get(ent.id, []):
        if rel.kind_name() != RelationKind.DERIVES.value:
            continue
        src = codemap.entities.get(rel.src)
        if src is None or src.kind_name() != EntityKind.PREDICATE.value:
            continue
        out.append({"id": src.id, "file": rel.attrs.get("file") or src.file, "line": rel.attrs.get("line") or src.line_start, "function": rel.attrs.get("function") or src.attrs.get("function"), "condition": src.name, "fields": [ent.name], "lhs": src.attrs.get("lhs"), "guards": list(src.attrs.get("guards") or [])})
    return out


def _read_roots(codemap: CodeMap, ent: Any, incoming: dict[str, list[Any]]) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for rel in incoming.get(ent.id, []):
        if rel.kind_name() not in {RelationKind.DERIVES.value, RelationKind.FLOWS_TO.value}:
            continue
        src = codemap.entities.get(rel.src)
        if src is not None and src.kind_name() in {EntityKind.INPUT.value, EntityKind.COMPILE_VAR.value, EntityKind.MACRO.value}:
            roots.append({"root": src.name, "kind": src.kind_name(), "entity_id": src.id, "provenance": rel.attrs.get("provenance") or ""})
    return roots


def _extract_symbols(expr: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b", expr):
        out.append(m.group(1))
    for m in re.finditer(r"\b([A-Za-z_][\w]*)\b", expr):
        tok = m.group(1)
        if tok not in {"static_cast", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int"}:
            out.append(tok)
    return out
