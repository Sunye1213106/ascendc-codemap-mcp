# -*- coding: utf-8 -*-
"""Cannbot locate-surface quality for a CodeMap — not unresolved-count."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ascendc_codemap_mcp.engine.diagnostics.source_api import (
    GRAPH_API_NEEDLES,
    count_graph_kernel_api,
    precision_gaps,
    source_api_from_codemap,
)
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.evidence import summarize_graph
from ascendc_codemap_mcp.engine.resolve.semantic_gap import list_gaps, summarize_gaps

_KERNEL_API_NEEDLES = GRAPH_API_NEEDLES


def _has_span(entity: Entity) -> bool:
    return bool(str(entity.file or "").strip()) and int(entity.line_start or 0) > 0


def _sites_with_span(attrs: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        for site in attrs.get(key) or []:
            if not isinstance(site, dict):
                continue
            if str(site.get("file") or "").strip() and int(
                site.get("line") or site.get("line_start") or 0
            ) > 0:
                return True
    return False


def source_schema_tiling_keys(keys: list[Any]) -> list[Any]:
    """Packing / locate coverage is over the source-contract schema.

    ``TILING_KEY_IS`` catalogs and sibling-arch TPL dims stay on the graph as
    selection facts. They are not extra packing dimensions.
    """
    declared = [
        e
        for e in keys
        if bool((getattr(e, "attrs", None) or {}).get("source_declared"))
    ]
    return declared or list(keys)


def _surface(ok: bool, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"ok": bool(ok)}
    row.update(extra)
    return row


def codemap_quality(
    codemap: CodeMap,
    *,
    integrity_ok: bool = True,
    audit: dict[str, Any] | None = None,
    gaps: list[dict[str, Any]] | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Scorecard aligned with cannbot locate / field / buffer / kernel_api.

    integrity_ok is graph legality (verify pass/fail). grade is whether the
    product can replace grep for structural questions.
    """
    if audit is None:
        from ascendc_codemap_mcp.engine.diagnostics.audit import audit_codemap

        audit = audit_codemap(codemap)
    if gaps is None:
        gaps = list_gaps(codemap)
    buckets = summarize_gaps(gaps)
    summary = dict((audit or {}).get("summary") or {})

    by_kind: dict[str, list[Entity]] = {}
    for ent in codemap.entities.values():
        by_kind.setdefault(ent.kind_name(), []).append(ent)

    keys = by_kind.get(EntityKind.TILING_KEY.value) or []
    schema_keys = source_schema_tiling_keys(keys)
    fields = by_kind.get(EntityKind.TILING_FIELD.value) or []
    kernels = by_kind.get(EntityKind.KERNEL.value) or []
    inputs = by_kind.get(EntityKind.INPUT.value) or []
    buffers = (by_kind.get(EntityKind.BUFFER.value) or []) + (
        by_kind.get(EntityKind.QUEUE.value) or []
    )
    ops = by_kind.get(EntityKind.OPERATION.value) or []
    host_checks = [
        e
        for e in (by_kind.get(EntityKind.BRANCH.value) or [])
        if str((e.attrs or {}).get("branch_kind") or "") == "host_check"
    ]

    key_span_n = sum(1 for e in keys if _has_span(e))
    schema_span_n = sum(1 for e in schema_keys if _has_span(e))
    kernel_span_n = sum(1 for e in kernels if _has_span(e))
    input_span_n = sum(1 for e in inputs if _has_span(e))
    pack_n = sum(
        1
        for e in schema_keys
        if _sites_with_span(e.attrs or {}, "packing_value_sites", "producer_sites")
        or e.attrs.get("host_packing_expressions")
    )
    owner_n = sum(1 for e in fields if str((e.attrs or {}).get("owner") or "").strip())
    writer_n = sum(
        1
        for e in fields
        if _sites_with_span(
            e.attrs or {},
            "host_writer_sites",
            "value_defining_sites",
            "check_sites",
        )
        or str((e.attrs or {}).get("rhs") or "").strip()
    )
    field_owner_unknown = sum(
        1
        for e in fields
        if str((e.attrs or {}).get("reason") or "")
        in {"field_owner_unknown", "field_owner_ambiguous"}
    )
    host_check_span_n = sum(1 for e in host_checks if _has_span(e))
    placed_buf = 0
    for e in buffers:
        attrs = e.attrs or {}
        space = str(attrs.get("memory_space") or "")
        tpos = str(attrs.get("tposition") or "")
        if tpos or (space and space != "UNKNOWN"):
            placed_buf += 1

    dtype_n = 0
    for e in inputs:
        attrs = e.attrs or {}
        facts = attrs.get("facts") if isinstance(attrs.get("facts"), dict) else {}
        if attrs.get("dtype") or facts.get("dtype"):
            dtype_n += 1

    kernel_api = count_graph_kernel_api(ops)
    api_ok = True
    any_api = False
    for needle in _KERNEL_API_NEEDLES:
        row = kernel_api.get(needle) or {}
        n = int(row.get("n") or 0)
        spanned = int(row.get("with_span") or 0)
        reached = int(row.get("reached") or 0)
        if n:
            any_api = True
            if spanned < n and reached < n:
                api_ok = False
    source_counts: dict[str, int] | None = None
    try:
        source_counts = source_api_from_codemap(
            codemap, source_root, str(codemap.architecture or "")
        )
    except Exception:  # noqa: BLE001
        source_counts = None
    src_gaps = precision_gaps(source_counts, kernel_api) if source_counts is not None else []
    if src_gaps:
        api_ok = False
    kt_meta = {}
    if isinstance(codemap.meta, dict):
        kt_meta = dict(codemap.meta.get("kernel_root_trace") or {})
    if kt_meta.get("gated_fill_complete") is False:
        api_ok = False
    if not any_api:
        any_source = bool(source_counts) and any(
            int(source_counts.get(name) or 0) > 0 for name in _KERNEL_API_NEEDLES
        )
        # Empty graph is honest only when the same include closure has no needles.
        # Without a source root the emptiness cannot be proven.
        api_ok = bool(source_counts is not None and not any_source)

    outputs = by_kind.get(EntityKind.OUTPUT.value) or []
    # Fusion send-style proto may declare no .OUTPUT. Verify already treats that
    # as SOURCE_HAS_NO_OUTPUT (warning, not block). Requiring a path to a node
    # that does not exist would keep grade at usable after an honest verify pass.
    io_ok = (not outputs) or bool(summary.get("has_input_output_path"))
    paths_ok = bool(
        summary.get("has_host_kernel_path")
        and summary.get("has_tilingdata_kernel_path")
        and io_ok
    )
    if not kernels:
        paths_ok = False

    tiling_key_ok = (not schema_keys) or (pack_n == len(schema_keys) and schema_span_n >= 1)
    field_ok = field_owner_unknown == 0 and (
        not fields or ((owner_n / len(fields) >= 0.9) and writer_n >= 1)
    )
    host_check_ok = (not host_checks) or host_check_span_n == len(host_checks)
    buffer_ok = (not buffers) or (placed_buf / len(buffers) >= 0.5)
    dtype_ok = (not inputs) or dtype_n >= 1
    symbol_ok = kernel_span_n >= 1 and input_span_n >= 1 and ((not keys) or key_span_n >= 1)

    surfaces = {
        "symbol_span": _surface(
            symbol_ok,
            kernel=f"{kernel_span_n}/{len(kernels)}",
            input=f"{input_span_n}/{len(inputs)}",
            tiling_key=f"{key_span_n}/{len(keys)}",
        ),
        "tiling_key": _surface(
            tiling_key_ok,
            packing=f"{pack_n}/{len(schema_keys)}",
            coverage=summary.get("tiling_key_host_packing_coverage"),
        ),
        "field_rw": _surface(
            field_ok,
            owner=f"{owner_n}/{len(fields)}",
            writer=f"{writer_n}/{len(fields)}",
            field_owner_unknown=field_owner_unknown,
        ),
        "host_check": _surface(
            host_check_ok, span=f"{host_check_span_n}/{len(host_checks)}"
        ),
        "buffer": _surface(
            buffer_ok,
            placed=f"{placed_buf}/{len(buffers)}",
        ),
        "kernel_api": _surface(api_ok, apis=kernel_api, source_gaps=src_gaps),
        "dtype": _surface(dtype_ok, count=f"{dtype_n}/{len(inputs)}"),
        "paths": _surface(
            paths_ok,
            host_kernel=bool(summary.get("has_host_kernel_path")),
            tilingdata_kernel=bool(summary.get("has_tilingdata_kernel_path")),
            input_output=bool(summary.get("has_input_output_path")),
            output_n=len(outputs),
        ),
    }
    surfaces_ok = all(bool(v.get("ok")) for v in surfaces.values())
    locate_blocking = int(buckets.get("locate_blocking") or 0)

    not_ready_reasons: list[str] = []
    if not integrity_ok:
        not_ready_reasons.append("integrity_fail")
    if kernel_span_n < 1:
        not_ready_reasons.append("no_kernel_span")
    if schema_keys and pack_n == 0:
        not_ready_reasons.append("no_tiling_key_packing_site")
    if field_owner_unknown > 0:
        not_ready_reasons.append("field_owner_unknown")
    if not api_ok:
        not_ready_reasons.append("kernel_api_gap")
    if kt_meta.get("gated_fill_complete") is False:
        not_ready_reasons.append("gated_fill_truncated")
    if kernels and not bool(summary.get("has_host_kernel_path")):
        not_ready_reasons.append("missing_host_kernel_path")
    if (by_kind.get(EntityKind.TILING_DATA.value)) and not bool(
        summary.get("has_tilingdata_kernel_path")
    ):
        not_ready_reasons.append("missing_tilingdata_kernel_path")

    if not_ready_reasons:
        grade = "not_ready"
    elif integrity_ok and surfaces_ok and locate_blocking == 0:
        grade = "ready"
    else:
        grade = "usable"

    blockers = [
        g
        for g in gaps
        if isinstance(g, dict) and str(g.get("bucket") or "") in {"locate_blocking", "audit_blocking"}
    ][:20]

    ke = {}
    if isinstance(codemap.meta, dict):
        ke = dict(codemap.meta.get("kernel_root_trace") or {})
    kq = dict(ke.get("quality") or {})

    locate_ready = grade == "ready"
    source_lookup = (
        "minimal windows at recorded file:line"
        if grade in {"ready", "usable"}
        else "CodeMap is not a substitute for source; locate surface is incomplete"
    )
    cm_summary = codemap.summary()
    trust_summary = {}
    if isinstance(codemap.meta, dict) and isinstance(codemap.meta.get("trust_summary"), dict):
        trust_summary = dict(codemap.meta.get("trust_summary") or {})
    if not trust_summary:
        trust_summary = summarize_graph(codemap.entities.values(), codemap.relations.values())
    roles_meta = dict((codemap.meta or {}).get("symbol_roles") or {})
    role_candidates = int(roles_meta.get("candidate_symbol_count") or 0)
    role_resolved = int(roles_meta.get("resolved_symbol_count") or 0)
    role_rate = (role_resolved / role_candidates) if role_candidates else None
    return {
        "schema": "uo-product-quality/v1",
        "grade": grade,
        "locate_ready": locate_ready,
        "graph": {
            "entity_count": int(cm_summary.get("entity_count") or 0),
            "relation_count": int(cm_summary.get("relation_count") or 0),
            "entities_by_kind": dict(cm_summary.get("entities_by_kind") or {}),
            "relations_by_kind": dict(cm_summary.get("relations_by_kind") or {}),
        },
        "trust": trust_summary,
        "integrity": "pass" if integrity_ok else "fail",
        "surfaces": surfaces,
        "unresolved": {
            "locate_blocking": locate_blocking,
            "host_runtime_leaf": int(buckets.get("host_runtime_leaf") or 0),
            "catalog_unproven": int(buckets.get("catalog_unproven") or 0),
            "total": int(buckets.get("total") or 0),
        },
        "source_lookup": source_lookup,
        "blockers": blockers,
        "not_ready_reasons": not_ready_reasons,
        "aux": {
            "tiling_key_dependency_coverage": summary.get("tiling_key_dependency_coverage"),
            "unpaired_flag_sync": kq.get("unpaired_flag_sync") or ke.get("unpaired_flag_sync"),
            "reached_operations": kq.get("reached_operations") or ke.get("reached_operations"),
            "reached_buffers": kq.get("reached_buffers") or ke.get("reached_buffers"),
            "reached_registers": kq.get("reached_registers") or ke.get("reached_registers"),
            "tiling_data_binding_count": len(list((codemap.meta or {}).get("tiling_data_bindings") or [])),
            "role_resolution_rate": role_rate,
        },
    }


def ready_status_fields(quality: dict[str, Any] | None) -> dict[str, Any]:
    """Compact grade / graph counts for status-only and host_step.done."""
    q = quality if isinstance(quality, dict) else {}
    graph = q.get("graph") if isinstance(q.get("graph"), dict) else {}
    unresolved = q.get("unresolved") if isinstance(q.get("unresolved"), dict) else {}
    out: dict[str, Any] = {}
    if q.get("grade") is not None:
        out["grade"] = q.get("grade")
    if q.get("integrity") is not None:
        out["integrity"] = q.get("integrity")
    if "locate_ready" in q:
        out["locate_ready"] = q.get("locate_ready")
    if unresolved.get("locate_blocking") is not None:
        out["locate_blocking"] = unresolved.get("locate_blocking")
    if unresolved.get("total") is not None:
        out["unresolved_total"] = unresolved.get("total")
    if graph.get("entity_count") is not None:
        out["entity_count"] = graph.get("entity_count")
    if graph.get("relation_count") is not None:
        out["relation_count"] = graph.get("relation_count")
    return out


def load_product_ready_status(product: Path | None) -> dict[str, Any]:
    """Read verify's quality.yaml next to the .uo; never open the binary."""
    if product is None:
        return {}
    p = Path(product)
    candidates: list[Path] = []
    if p.is_dir():
        candidates.append(p / "checks" / "quality.yaml")
        candidates.append(p / "quality.yaml")
    else:
        candidates.append(p.parent / "checks" / "quality.yaml")
    for quality_path in candidates:
        if not quality_path.is_file():
            continue
        try:
            doc = yaml.safe_load(quality_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(doc, dict):
            return ready_status_fields(doc)
    return {}
