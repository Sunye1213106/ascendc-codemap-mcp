# -*- coding: utf-8 -*-
"""Operator-agnostic cannbot locate-surface checks for a CodeMap product.

FAG-specific numbers (Key ≥ 19, locate s1Inner) stay in tools that call this
as a control sample. This module only encodes clauses that hold for any op.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.diagnostics.source_api import (
    count_graph_kernel_api,
    precision_gaps,
    source_api_from_codemap,
)
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.source_layout import is_foreign_arch_entry_tu, is_other_arch_path

_TPL_HINT_RE = re.compile(r"\b(?:ASCENDC_TPL_|GET_TPL_TILING_KEY|TILING_KEY_IS)\b")


def _has_span(entity: Any) -> bool:
    return bool(str(getattr(entity, "file", "") or "").strip()) and int(
        getattr(entity, "line_start", 0) or 0
    ) > 0


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


def _source_declares_tiling_key(source_root: Path, architecture: str) -> bool:
    roots = [source_root / "op_host", source_root / "op_kernel"]
    for base in roots:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}:
                continue
            if is_other_arch_path(path, architecture):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _TPL_HINT_RE.search(text):
                return True
    return False


def check_cannbot_product(
    codemap: CodeMap,
    *,
    source_root: str | Path,
    architecture: str = "",
) -> dict[str, Any]:
    """Job-facts that any operator must satisfy to stand in for grep."""
    root = Path(source_root)
    arch = str(architecture or codemap.architecture or "")
    keys = list(codemap.by_kind(EntityKind.TILING_KEY))
    fields = list(codemap.by_kind(EntityKind.TILING_FIELD))
    kernels = list(codemap.by_kind(EntityKind.KERNEL))
    inputs = list(codemap.by_kind(EntityKind.INPUT))
    ops = list(codemap.by_kind(EntityKind.OPERATION))
    bufs = list(codemap.by_kind(EntityKind.BUFFER)) + list(codemap.by_kind(EntityKind.QUEUE))
    branches = list(codemap.by_kind(EntityKind.BRANCH))
    other_n = sum(1 for e in codemap.entities.values() if e.kind_name() == EntityKind.OTHER.value)

    pack_n = sum(
        1
        for e in keys
        if _sites_with_span(e.attrs or {}, "packing_value_sites", "producer_sites")
        or (e.attrs or {}).get("host_packing_expressions")
    )
    source_has_key_schema = _source_declares_tiling_key(root, arch)
    if keys:
        packing_ok = pack_n == len(keys) and any(_has_span(e) for e in keys)
        packing_warning = None
    elif source_has_key_schema:
        packing_ok = False
        packing_warning = "source_declares_key_schema_but_graph_has_zero_keys"
    else:
        packing_ok = True
        packing_warning = "no_tiling_key_schema"

    locate_ok = True
    locate_probes: list[dict[str, Any]] = []
    for kind, ents in (
        ("TILING_FIELD", fields),
        ("TILING_KEY", keys),
        ("KERNEL", kernels),
        ("INPUT", inputs),
    ):
        if not ents:
            continue
        sample = next((e for e in ents if _has_span(e)), None)
        ok = sample is not None
        locate_probes.append(
            {
                "kind": kind,
                "name": (sample or ents[0]).name,
                "ok": ok,
                "sample": (
                    f"{sample.file}:{sample.line_start}" if sample is not None else None
                ),
            }
        )
        if not ok:
            locate_ok = False

    source_api = source_api_from_codemap(codemap, root, arch) or {}
    graph_api = count_graph_kernel_api(ops)
    needle_gaps = precision_gaps(source_api, graph_api)
    needles_ok = not needle_gaps

    src_sync = int(source_api.get("SetFlag") or 0) + int(source_api.get("WaitFlag") or 0)
    graph_sync = int((graph_api.get("SetFlag") or {}).get("n") or 0) + int(
        (graph_api.get("WaitFlag") or {}).get("n") or 0
    )
    kernel_api_sync_ok = src_sync <= 0 or graph_sync > 0

    placed = 0
    for e in bufs:
        attrs = e.attrs or {}
        tpos = str(attrs.get("tposition") or "")
        space = str(attrs.get("memory_space") or "")
        if tpos or (space and space != "UNKNOWN"):
            placed += 1
    buffer_ok = (not bufs) or (placed / len(bufs) >= 0.5)

    dtype_n = 0
    for e in inputs:
        attrs = e.attrs or {}
        facts = attrs.get("facts") if isinstance(attrs.get("facts"), dict) else {}
        if attrs.get("dtype") or facts.get("dtype"):
            dtype_n += 1
    dtype_ok = (not inputs) or dtype_n >= 1

    dummy_kernels = [
        e
        for e in kernels
        if not e.file and not e.attrs.get("source_signature") and not e.attrs.get("variants")
    ]
    foreign = [
        str(e.file)
        for e in codemap.entities.values()
        if e.file and is_foreign_arch_entry_tu(Path(str(e.file)), arch)
    ]

    expected = {
        "key_packing_full": packing_ok,
        "locate_existing_kinds": locate_ok,
        "needles_graph_ge_source": needles_ok,
        "kernel_api_sync": kernel_api_sync_ok,
        "buffer_tposition": buffer_ok,
        "input_dtype_declared": dtype_ok,
        "no_dummy_kernel": not dummy_kernels,
        "no_foreign_arch": not foreign,
        "other_count_zero": other_n == 0,
    }
    return {
        "ok": all(expected.values()),
        "expected": expected,
        "failures": [k for k, v in expected.items() if not v],
        "warning": packing_warning,
        "counts": {
            "tiling_keys": len(keys),
            "tiling_fields": len(fields),
            "kernels": len(kernels),
            "inputs_with_dtype": dtype_n,
            "inputs": len(inputs),
            "buffers": len(bufs),
            "placed_buffers": placed,
            "host_checks": sum(
                1
                for e in branches
                if str((e.attrs or {}).get("branch_kind") or "") == "host_check"
            ),
            "other": other_n,
            "dummy_kernels": len(dummy_kernels),
            "foreign_arch": len(foreign),
        },
        "source_api": source_api,
        "kernel_api": graph_api,
        "precision_gaps": needle_gaps,
        "locate_probes": locate_probes,
        "foreign_arch_sample": foreign[:10],
        "dummy_kernels": [e.id for e in dummy_kernels[:10]],
    }
