# -*- coding: utf-8 -*-
"""UO CodeMap compiler entry — assemble semantic passes and commit one ``.uo``."""

from __future__ import annotations

from ascendc_codemap_mcp.engine.paths import require_architecture
import pickle
import time
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.frontend.build_variant import build_variant_from_context
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.frontier_resolution import resolve_class_frontiers
from ascendc_codemap_mcp.engine.passes.host_defuse import trace_host_key_roots
from ascendc_codemap_mcp.engine.passes.host_defuse_validate import validate_host_defuse
from ascendc_codemap_mcp.engine.passes.host_tiling_key import bind_host_tiling_key_expressions
from ascendc_codemap_mcp.engine.passes.kernel_call_boundaries import classify_kernel_call_boundaries
from ascendc_codemap_mcp.engine.passes.kernel_call_read_refine import refine_kernel_calls_and_tiling_reads
from ascendc_codemap_mcp.engine.passes.clang_macro_uses import run as bind_clang_macro_uses_pass
from ascendc_codemap_mcp.engine.passes.kernel_call_resolution import resolve_kernel_call_frontiers
from ascendc_codemap_mcp.engine.passes.kernel_identity import preserve_verified_kernel_identity
from ascendc_codemap_mcp.engine.passes.kernel_root_trace import finalize_kernel_root_trace
from ascendc_codemap_mcp.engine.passes.kernel_tiling_closure import finalize_kernel_tiling_closure
from ascendc_codemap_mcp.engine.passes.kernel_tiling_metrics import finalize_kernel_tiling_metrics
from ascendc_codemap_mcp.engine.passes.kernel_tiling_truth import finalize_kernel_tiling_truth
from ascendc_codemap_mcp.engine.passes.manager import run_analyze_passes
from ascendc_codemap_mcp.engine.passes.source_contract import enrich_codemap_from_operator_source
from ascendc_codemap_mcp.engine.passes.source_inventory import inventory_source_files
from ascendc_codemap_mcp.engine.passes.source_resolution import resolve_source_gaps
from ascendc_codemap_mcp.engine.passes.symbol_roles import project_symbol_roles
from ascendc_codemap_mcp.engine.passes.tiling_field_complete import complete_tiling_fields
from ascendc_codemap_mcp.engine.passes.tiling_host_writes import enrich_tiling_host_writes
from ascendc_codemap_mcp.engine.passes.value_defining_sites import enrich_value_defining_sites
from ascendc_codemap_mcp.engine.passes.host_checks import enrich_host_checks
from ascendc_codemap_mcp.engine.passes.tiling_context_apis import enrich_tiling_context_apis
from ascendc_codemap_mcp.engine.passes.guarded_calls import enrich_guarded_calls
from ascendc_codemap_mcp.engine.passes.constexpr_alias import enrich_constexpr_aliases
from ascendc_codemap_mcp.engine.passes.host_predicates import enrich_host_predicates
from ascendc_codemap_mcp.engine.passes.compile_policy import enrich_compile_policy
from ascendc_codemap_mcp.engine.passes.workspace_abi import enrich_workspace_abi
from ascendc_codemap_mcp.engine.passes.consumer_role import enrich_consumer_roles
from ascendc_codemap_mcp.engine.passes.entry_path import enrich_entry_paths
from ascendc_codemap_mcp.engine.passes.contracts import enrich_contracts
from ascendc_codemap_mcp.engine.passes.tiling_kernel_reads import rebuild_verified_tiling_reads
from ascendc_codemap_mcp.engine.passes.tiling_registration import enrich_tiling_registrations
from ascendc_codemap_mcp.engine.passes.tiling_template_registry import enrich_tiling_template_registry
from ascendc_codemap_mcp.engine.resolve.semantic_gap import list_gaps
from ascendc_codemap_mcp.engine.store.writer import uo_product_path, write_codemap
from ascendc_codemap_mcp.engine.timing import log as _tlog, timing_enabled

# Same-process reuse between analyze (commit=False) and commit. Avoids paying
# the full source-enrichment stack twice in one uo-init run.
_COMPILE_MEM: dict[str, dict[str, Any]] = {}
ANALYZE_CACHE_VERSION = 2


def _cache_key(op_root: Path, op_name: str, architecture: str) -> str:
    return f"{op_root.resolve()}|{op_name}|{architecture}"


def _cache_path(op_root: Path, architecture: str) -> Path:
    return (
        Path(op_root).expanduser().resolve()
        / ".ascendc-codemap" / architecture
        / "ir"
        / "_codemap_compile_cache.pkl"
    )


def store_compile_cache(
    op_root: Path,
    op_name: str,
    architecture: str,
    result: dict[str, Any],
    *,
    extract_fingerprint: str = "",
) -> None:
    """Keep analyze's compile result for a later commit in this (or next) process."""
    key = _cache_key(op_root, op_name, architecture)
    payload = {
        "op_name": op_name,
        "architecture": architecture,
        "codemap": result.get("codemap"),
        "views": result.get("_merged_views") or {},
        "summary": result.get("summary") or {},
        "gaps": result.get("gaps") or [],
        "audit": result.get("audit"),
        "tg_views": result.get("tg_views") or {},
        "extract_fingerprint": extract_fingerprint
        or str(result.get("extract_fingerprint") or ""),
        "analyze_cache_version": ANALYZE_CACHE_VERSION,
    }
    _COMPILE_MEM[key] = payload
    try:
        path = _cache_path(op_root, architecture)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:  # noqa: BLE001
        pass


def load_compile_cache(
    op_root: Path,
    op_name: str,
    architecture: str,
) -> dict[str, Any] | None:
    key = _cache_key(op_root, op_name, architecture)
    hit = _COMPILE_MEM.get(key)
    if hit is not None and hit.get("codemap") is not None:
        return hit
    try:
        path = _cache_path(op_root, architecture)
        if not path.is_file():
            return None
        data = pickle.loads(path.read_bytes())
        if not isinstance(data, dict) or data.get("codemap") is None:
            return None
        if data.get("op_name") != op_name or data.get("architecture") != architecture:
            return None
        _COMPILE_MEM[key] = data
        return data
    except Exception:  # noqa: BLE001
        return None


def load_fresh_compile_cache(
    op_root: Path,
    op_name: str,
    architecture: str,
    *,
    extract_fingerprint: str,
) -> dict[str, Any] | None:
    """Reuse analyze's compile payload only when extract identity still matches."""
    if not extract_fingerprint:
        return None
    cached = load_compile_cache(op_root, op_name, architecture)
    if cached is None or cached.get("codemap") is None:
        return None
    if int(cached.get("analyze_cache_version") or 0) != ANALYZE_CACHE_VERSION:
        return None
    if str(cached.get("extract_fingerprint") or "") != str(extract_fingerprint):
        return None
    return cached


def drop_compile_mem(op_root: Path | None = None, architecture: str | None = None) -> None:
    """Drop in-process compile payloads. Disk pickle stays for crash/restart."""
    if op_root is None:
        _COMPILE_MEM.clear()
        return
    prefix = f"{Path(op_root).expanduser().resolve()}|"
    arch = str(architecture or "")
    for key in list(_COMPILE_MEM):
        if not key.startswith(prefix):
            continue
        if arch and not key.endswith(f"|{arch}"):
            continue
        _COMPILE_MEM.pop(key, None)


def clear_compile_cache(op_root: Path | None = None, architecture: str | None = None) -> None:
    if op_root is None:
        _COMPILE_MEM.clear()
        return
    arch = require_architecture(architecture)
    key_prefix = f"{Path(op_root).expanduser().resolve()}|"
    for k in list(_COMPILE_MEM):
        if k.startswith(key_prefix):
            _COMPILE_MEM.pop(k, None)
    try:
        path = _cache_path(Path(op_root), arch)
        if path.is_file():
            path.unlink()
    except Exception:  # noqa: BLE001
        pass


def _span(name: str, t0: float) -> None:
    dt = time.perf_counter() - t0
    if timing_enabled():
        _tlog(f"{dt:7.3f}s  compile.{name}")
    try:
        from ascendc_codemap_mcp.engine.perf import record_pass

        record_pass(name, dt)
    except Exception:  # noqa: BLE001
        pass


def compile_codemap(
    *,
    op_name: str,
    architecture: str = "",
    op_root: str | Path | None = None,
    host_ir: Any = None,
    kernel_ir: Any = None,
    tiling_ir: Any = None,
    kb: Any = None,
    key_fields: list[dict[str, Any]] | None = None,
    declared: dict[str, Any] | None = None,
    inputs: list[str] | None = None,
    build_context: Any = None,
    template_bindings: list[dict[str, Any]] | None = None,
    views: dict[str, Any] | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Compile deterministic facts + current source into the unified CodeMap.

    Compiler-derived Host/Kernel IR remains authoritative where available. When
    an operator source root exists, deterministic source passes additionally
    inventory the selected architecture, recover API/Key/TilingData contracts,
    bind Host packed-key arguments, trace and lexically revalidate their def-use
    roots, complete scalar and array TilingData ABI fields, and finally rebuild
    an architecture-pure Kernel call/read/write closure from qualified current-
    source symbols before the strict completeness audit runs. Dependent/external
    calls that cannot be uniquely bound remain explicit call boundaries rather
    than guessed edges.
    """
    t_all = time.perf_counter()
    arch = require_architecture(architecture)
    variant = build_variant_from_context(architecture=arch, build_context=build_context, name=arch)
    variant_doc = variant.to_dict()
    from ascendc_codemap_mcp.engine.ir.evidence import build_context_id

    cm = CodeMap(op_name=op_name, architecture=arch)
    cm.meta["build_context_id"] = build_context_id(variant_doc)
    cm.meta["build_context"] = {
        "context_id": cm.meta["build_context_id"],
        "arch": arch,
        "dtype_variant": str(variant_doc.get("dtype_variant") or ""),
    }
    bv = cm.upsert(EntityKind.BUILD_VARIANT, variant.name, attrs=variant_doc)
    arch_e = cm.upsert(EntityKind.ARCH, arch)
    cm.link(RelationKind.ACTIVE_UNDER, arch_e.id, bv.id, attrs={"provenance": "build_variant"}, status="confirmed")

    context: dict[str, Any] = {
        "host_ir": host_ir,
        "kernel_ir": kernel_ir,
        "tiling_ir": tiling_ir,
        "key_fields": key_fields or [],
        "declared": declared or {},
        "inputs": inputs or [],
        "build_variant": variant.to_dict(),
        "template_bindings": template_bindings or [],
        "op_name": op_name,
        "op_root": str(op_root or ""),
        "clang_ctx": build_context,
    }
    if kb is not None:
        CodeMap.from_kb(kb, codemap=cm)
    t0 = time.perf_counter()
    cm = run_analyze_passes(cm, context=context)
    _span("analyze_passes", t0)

    source_root = Path(op_root).expanduser().resolve() if op_root is not None else None
    if source_root is not None and _looks_like_operator_source(source_root):
        from ascendc_codemap_mcp.engine.passes.source_text_cache import clear as clear_source_text
        from ascendc_codemap_mcp.engine.perf import reset_file_reads
        from ascendc_codemap_mcp.engine.source_index import reset_index_cache

        clear_source_text()
        reset_index_cache()
        reset_file_reads()
        for name, fn, kwargs in (
            ("inventory", inventory_source_files, {}),
            ("source_contract", enrich_codemap_from_operator_source, {"needs_irs": True}),
            ("symbol_roles", project_symbol_roles, {"needs_irs": True}),
            ("tiling_template_registry", enrich_tiling_template_registry, {}),
            ("tiling_fields", complete_tiling_fields, {"needs_irs": True}),
            ("host_tiling_key", bind_host_tiling_key_expressions, {}),
            ("host_defuse", trace_host_key_roots, {"needs_irs": True}),
            ("host_defuse_validate", validate_host_defuse, {}),
            ("tiling_registration", enrich_tiling_registrations, {}),
            ("source_gaps", resolve_source_gaps, {}),
            ("class_frontiers", resolve_class_frontiers, {}),
            ("kernel_tiling_closure", finalize_kernel_tiling_closure, {"needs_irs": True, "rebuild_bodies": False}),
            ("kernel_identity", preserve_verified_kernel_identity, {"skip_arch": True}),
            ("kernel_call_refine", refine_kernel_calls_and_tiling_reads, {}),
            ("clang_macro_uses", bind_clang_macro_uses_pass, {"needs_irs": True}),
            ("kernel_call_frontiers", resolve_kernel_call_frontiers, {}),
            ("kernel_call_boundaries", classify_kernel_call_boundaries, {"skip_arch": True}),
            ("tiling_reads", rebuild_verified_tiling_reads, {}),
            ("tiling_host_writes", enrich_tiling_host_writes, {"needs_irs": True}),
            ("value_defining_sites", enrich_value_defining_sites, {}),
            ("host_checks", enrich_host_checks, {}),
            ("kernel_tiling_truth", finalize_kernel_tiling_truth, {"skip_arch": True}),
            ("kernel_tiling_metrics", finalize_kernel_tiling_metrics, {"skip_arch": True}),
            # Kernel Root Trace (UO canonical): wrappers / calls → AscendC root.
            ("kernel_root_trace", finalize_kernel_root_trace, {}),
            ("guarded_calls", enrich_guarded_calls, {"needs_irs": True}),
            ("constexpr_alias", enrich_constexpr_aliases, {}),
            ("host_predicates", enrich_host_predicates, {"needs_host_ir": True}),
            ("compile_policy", enrich_compile_policy, {}),
            ("workspace_abi", enrich_workspace_abi, {}),
            ("consumer_roles", enrich_consumer_roles, {}),
            ("entry_paths", enrich_entry_paths, {"needs_host_ir": True}),
            ("contracts", enrich_contracts, {}),
            # After root-trace purge: host TilingContext APIs are not kernel ops.
            ("tiling_context_apis", enrich_tiling_context_apis, {"needs_host_ir": True}),
        ):
            t0 = time.perf_counter()
            extra = {
                key: value
                for key, value in kwargs.items()
                if key not in {"skip_arch", "needs_host_ir", "needs_irs"}
            }
            if kwargs.get("skip_arch"):
                fn(cm, **extra)  # type: ignore[misc]
            elif kwargs.get("needs_host_ir"):
                fn(cm, source_root, architecture=arch, host_ir=host_ir, **extra)  # type: ignore[misc]
            elif kwargs.get("needs_irs"):
                fn(cm, source_root, architecture=arch, host_ir=host_ir, kernel_ir=kernel_ir, **extra)  # type: ignore[misc]
            else:
                fn(cm, source_root, architecture=arch, **extra)  # type: ignore[misc]
            _span(name, t0)
        clear_source_text()
        reset_index_cache()
        cm.meta["production_source_enrichment"] = True
    else:
        cm.meta["production_source_enrichment"] = False

    from ascendc_codemap_mcp.engine.diagnostics.audit import audit_codemap
    from ascendc_codemap_mcp.engine.passes import tpl_schema as tpl_schema_pass
    from ascendc_codemap_mcp.engine.tg_views import finalize_tg_views

    # Refresh TPL/D after source inventory. An early analyze pass may have
    # parsed a DECL-only header and stamped views that commit cannot rebuild;
    # the include walk here also sees ARGS_SEL siblings.
    t0 = time.perf_counter()
    if source_root is not None:
        context["op_root"] = str(source_root)
        context["architecture"] = arch
    cm = tpl_schema_pass.run(cm, context=context)
    _span("tpl_schema", t0)
    t0 = time.perf_counter()
    merged_views = dict(views or {})
    merged_views.update(context.get("tg_views") or {})
    merged_views = finalize_tg_views(cm, existing=merged_views)
    _span("finalize_views", t0)

    from ascendc_codemap_mcp.engine.tg_projection import require_commit_views

    if commit and source_root is not None:
        missing = require_commit_views(merged_views)
        if missing:
            return {
                "ok": False,
                "error": "TG_VIEW_INCOMPLETE",
                "missing": missing,
                "summary": {},
                "audit": {},
                "gaps": list_gaps(cm),
                "codemap": cm,
                "_merged_views": merged_views,
                "tg_views": {"view_names": sorted(merged_views), "missing": missing},
            }

    t0 = time.perf_counter()
    audit = audit_codemap(cm)
    _span("audit", t0)
    result: dict[str, Any] = {
        "ok": True,
        "summary": dict(audit["summary"]),
        "audit": audit,
        "gaps": list_gaps(cm, audit=audit),
        "codemap": cm,
        "_merged_views": merged_views,
        "tg_views": {
            "legal_key_count": int(cm.meta.get("legal_key_count") or 0),
            "view_names": sorted(merged_views),
        },
    }
    if commit and source_root is not None:
        t0 = time.perf_counter()
        path = uo_product_path(source_root, op_name, arch)
        from ascendc_codemap_mcp.engine.store.writer import detect_source_revision

        revision = detect_source_revision(source_root)
        written = write_codemap(
            cm,
            path,
            views=merged_views,
            summary=dict(audit["summary"]),
            meta={"source_revision": revision} if revision else None,
        )
        result["uo"] = written
        result["path"] = written.get("path")
        _span("write_uo", t0)
    _span("total", t_all)
    return result


def _looks_like_operator_source(root: Path) -> bool:
    return root.is_dir() and any((root / name).is_dir() for name in ("op_graph", "op_host", "op_kernel"))
