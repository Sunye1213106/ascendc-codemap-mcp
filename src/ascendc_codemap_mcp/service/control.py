# -*- coding: utf-8 -*-
"""Control plane: discover, status, doctor, index, update."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import PRODUCT_DIR_NAME, SERVER_VERSION
from ascendc_codemap_mcp.service.envelope import (
    STATE_BUILDING,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_IDLE,
    STATE_NEEDS_CONFIRMATION,
    envelope,
    fail,
)
from ascendc_codemap_mcp.service.identity import (
    CodemapRef,
    bind,
    is_ref,
    list_products,
    public_handle,
    ref_from_product,
    resolve,
)
from ascendc_codemap_mcp.service import runtime


def _meta(product: Path | None) -> dict[str, Any]:
    if product is None or not Path(product).is_file():
        return {}
    from ascendc_codemap_mcp.engine.store.reader import read_meta

    try:
        return dict(read_meta(product))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": str(exc)[:200]}


def _freshness_for(ref: CodemapRef, meta: dict[str, Any]) -> dict[str, Any]:
    from ascendc_codemap_mcp.service.freshness import compute

    return compute(
        ref.project,
        meta=meta,
        building=runtime.is_building(ref.id),
        blocked=runtime.is_blocked(ref.id),
    )


def discover(
    *,
    project: str = "",
    architecture: str = "",
) -> dict[str, Any]:
    from ascendc_codemap_mcp.service.identity import env_architecture, env_project

    proj = str(project or "").strip() or env_project()
    arch = str(architecture or "").strip() or env_architecture()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_ref(ref: CodemapRef) -> None:
        runtime.registry.put(ref)
        meta = _meta(ref.product)
        info = _freshness_for(ref, meta) if ref.product else {
            "freshness": "unknown",
            "source_revision": "",
            "indexed_revision": "",
            "dirty": False,
            "changed_files": 0,
            "semantic_completeness": None,
        }
        handle = public_handle(ref, meta=meta, freshness_info=info)
        handle["indexed"] = bool(ref.product and Path(ref.product).is_file())
        key = handle.get("path") or handle["id"]
        if key in seen:
            return
        seen.add(str(key))
        rows.append(handle)

    if proj:
        root = Path(proj).expanduser().resolve()
        if not root.is_dir():
            return fail(
                f"operator directory not found: {root}",
                error_code="OPERATOR_DIR_NOT_FOUND",
            )
        products = list_products(root)
        if arch:
            products = [p for p in products if p.name.endswith(f".{arch}.uo")]
        for product in products:
            ref = ref_from_product(product, project=root)
            if ref is not None:
                add_ref(ref)
        if not products and arch:
            add_ref(bind(project=root, architecture=arch, registry=runtime.registry))
    else:
        for ref in runtime.registry.all():
            if arch and ref.architecture != arch:
                continue
            add_ref(ref)

    hint = ""
    if not rows:
        hint = (
            "No CodeMap registered. Pass project= to scan an operator directory, "
            "then codemap_index with architecture (e.g. arch35)."
        )
    return envelope(
        ok=True,
        state=STATE_IDLE,
        data={"codemaps": rows, "count": len(rows), "hint": hint},
        extra={
            "engine": "codemap_discover",
            "version": SERVER_VERSION,
            "codemaps": rows,
            "count": len(rows),
            "hint": hint,
            "project": proj,
            "architecture": arch,
        },
    )


def status(
    *,
    project: str = "",
    architecture: str = "",
    codemap_id: str = "",
) -> dict[str, Any]:
    ref = resolve(
        codemap_id=codemap_id,
        project=project,
        architecture=architecture,
        registry=runtime.registry,
        require_indexed=False,
    )
    if not is_ref(ref):
        return ref  # type: ignore[return-value]
    product = ref.product
    indexed = bool(product is not None and Path(product).is_file())
    meta = _meta(product if indexed else None)
    info = _freshness_for(ref, meta) if indexed else {
        "freshness": "unknown",
        "source_revision": "",
        "indexed_revision": "",
        "dirty": False,
        "changed_files": 0,
        "semantic_completeness": None,
    }
    handle = public_handle(ref, meta=meta, freshness_info=info)
    mtime = 0
    if indexed and product is not None:
        try:
            mtime = int(product.stat().st_mtime)
        except OSError:
            mtime = 0
    extra = {
        "engine": "codemap_status",
        "indexed": indexed,
        "mtime": mtime,
        "architecture": ref.architecture,
        "op_name": ref.op_name,
        "project": str(ref.project),
        "path": str(product) if product else "",
        "semantic_completeness": handle.get("semantic_completeness"),
        "entity_count": meta.get("entity_count"),
        "relation_count": meta.get("relation_count"),
        "freshness": info.get("freshness"),
        "source_revision": info.get("source_revision"),
        "indexed_revision": info.get("indexed_revision"),
        "dirty": info.get("dirty"),
        "changed_files": info.get("changed_files"),
        "snapshot_id": handle.get("snapshot_id"),
        "expected": (
            f"{PRODUCT_DIR_NAME}/{ref.architecture}/<op>.{ref.architecture}.uo"
            if not indexed
            else ""
        ),
    }
    state = STATE_BUILDING if runtime.is_building(ref.id) else STATE_IDLE
    if runtime.is_blocked(ref.id):
        state = STATE_NEEDS_CONFIRMATION
    return envelope(
        ok=True,
        state=state,
        updated=False,
        codemap=handle,
        extra=extra,
    )


def doctor(
    *,
    project: str = "",
    architecture: str = "",
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.paths import explain, require_cann_ready

    issues: list[str] = []
    root = Path(project).expanduser() if str(project or "").strip() else None
    if root is not None and not root.is_dir():
        issues.append(f"operator directory not found: {root}")
    arch = str(architecture or "").strip()
    if not arch:
        issues.append("architecture is required (e.g. arch35)")
    try:
        import clang.cindex  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        issues.append(f"libclang import failed: {exc}")
    cann, cann_issues = require_cann_ready()
    issues.extend(cann_issues)
    stats = runtime.cache_stats()
    return {
        "ok": not issues,
        "engine": "codemap_doctor",
        "version": SERVER_VERSION,
        "project": str(root) if root else "",
        "architecture": arch,
        "cann_root": str(cann) if cann else None,
        "issues": issues,
        "explain": explain(),
        "product_dir": (
            str(root / PRODUCT_DIR_NAME / arch) if root is not None and arch else ""
        ),
        "runtime": stats,
    }


def _index_unlocked(
    root: Path,
    arch: str,
    *,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.codemap_engines import (
        analyze,
        commit,
        extract,
        prepare,
    )
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product

    ctx: dict[str, Any] = {"architecture": arch, "arch_dir": arch}
    t0 = time.perf_counter()
    steps: list[dict[str, Any]] = []
    pipeline = (
        ("prepare", prepare),
        ("extract", extract),
        ("analyze", analyze),
        ("commit", commit),
    )
    total = len(pipeline)
    for index, (name, fn) in enumerate(pipeline, start=1):
        if should_stop is not None and should_stop():
            return {
                "ok": False,
                "cancelled": True,
                "failed_step": name,
                "error": "cancelled",
                "error_code": "CANCELLED",
                "steps": steps,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
        if on_progress is not None:
            on_progress(index - 1, total, name)
        out = fn(root, ctx)
        if isinstance(out, dict):
            for key in ("op_name", "run_id", "arch_dir", "architecture"):
                if out.get(key) and not ctx.get(key):
                    ctx[key] = out[key]
            if out.get("arch_dir"):
                ctx["architecture"] = out.get("architecture") or out["arch_dir"]
        ok = bool(isinstance(out, dict) and out.get("ok"))
        steps.append(
            {
                "step": name,
                "ok": ok,
                "error": (out or {}).get("error") if isinstance(out, dict) else "no_result",
            }
        )
        if not ok:
            return {
                "ok": False,
                "failed_step": name,
                "error": (out or {}).get("error")
                or (out or {}).get("message_zh")
                or f"{name} failed",
                "steps": steps,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
    if on_progress is not None:
        on_progress(total, total, "done")
    product = find_uo_product(root, architecture=arch)
    summary: dict[str, Any] = {}
    if product is not None:
        try:
            summary = _meta(product)
        except Exception:  # noqa: BLE001
            summary = {}
    return {
        "ok": True,
        "path": str(product) if product else "",
        "summary": {
            k: summary.get(k)
            for k in (
                "op_name",
                "architecture",
                "entity_count",
                "relation_count",
                "semantic_completeness",
            )
            if k in summary
        },
        "gaps_count": int(summary.get("gaps_count") or 0) if summary else 0,
        "steps": steps,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "product": product,
        "meta": summary,
    }


def index_operator(
    *,
    project: str,
    architecture: str,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    if not str(project or "").strip():
        return fail("project is required", error_code="PROJECT_REQUIRED")
    arch = str(architecture or "").strip()
    if not arch:
        return fail(
            "ARCHITECTURE_MISSING_IN_RUN_STATE: architecture is required",
            error_code="ARCHITECTURE_MISSING_IN_RUN_STATE",
        )
    root = Path(project).expanduser().resolve()
    if not root.is_dir():
        return fail(
            f"operator directory not found: {root}",
            error_code="OPERATOR_DIR_NOT_FOUND",
        )
    ref = bind(project=root, architecture=arch, registry=runtime.registry)
    if runtime.is_building(ref.id):
        return envelope(
            ok=True,
            state=STATE_BUILDING,
            updated=False,
            codemap=public_handle(ref),
            extra={"engine": "codemap_index", "error_code": "BUILDING"},
        )
    with runtime.locks.write(ref.id):
        runtime.mark_building(ref.id)
        if ref.product is not None:
            runtime.cache.drop(ref.product)
        try:
            result = _index_unlocked(
                root,
                arch,
                should_stop=should_stop,
                on_progress=on_progress,
            )
        finally:
            runtime.clear_building(ref.id)
        if not result.get("ok"):
            cancelled = bool(result.get("cancelled"))
            return envelope(
                ok=False,
                state=STATE_IDLE if cancelled else STATE_FAILED,
                updated=False,
                error=str(result.get("error") or "index failed"),
                error_code="CANCELLED" if cancelled else None,
                extra={
                    "engine": "index_operator",
                    "failed_step": result.get("failed_step"),
                    "steps": result.get("steps") or [],
                    "elapsed_s": result.get("elapsed_s"),
                },
            )
        product = result.get("product")
        meta = result.get("meta") or _meta(product)
        fresh = bind(project=root, architecture=arch, registry=runtime.registry)
        runtime.clear_blocked(fresh.id)
        info = _freshness_for(fresh, meta)
        handle = public_handle(fresh, meta=meta, freshness_info=info)
        return envelope(
            ok=True,
            state=STATE_COMPLETED,
            updated=True,
            codemap=handle,
            extra={
                "engine": "index_operator",
                "path": result.get("path") or "",
                "summary": result.get("summary") or {},
                "gaps_count": result.get("gaps_count") or 0,
                "steps": result.get("steps") or [],
                "elapsed_s": result.get("elapsed_s"),
            },
        )


def update_operator(
    *,
    project: str = "",
    architecture: str = "",
    confirm_scope: bool = False,
    codemap_id: str = "",
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.update import update_operator as apply_update

    ref = resolve(
        codemap_id=codemap_id,
        project=project,
        architecture=architecture,
        registry=runtime.registry,
        require_indexed=True,
    )
    if not is_ref(ref):
        err = ref  # type: ignore[assignment]
        if isinstance(err, dict):
            err["engine"] = "update_operator"
        return err  # type: ignore[return-value]
    if runtime.is_building(ref.id):
        return envelope(
            ok=True,
            state=STATE_BUILDING,
            updated=False,
            codemap=public_handle(ref),
            extra={"engine": "update_operator", "error_code": "BUILDING"},
        )
    t0 = time.perf_counter()
    with runtime.locks.write(ref.id):
        runtime.mark_building(ref.id)
        if should_stop is not None and should_stop():
            runtime.clear_building(ref.id)
            return envelope(
                ok=False,
                state=STATE_IDLE,
                updated=False,
                error="cancelled",
                error_code="CANCELLED",
                extra={
                    "engine": "update_operator",
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                },
            )
        if ref.product is not None:
            runtime.cache.drop(ref.product)
        if on_progress is not None:
            on_progress(0, 1, "update")
        try:
            result = apply_update(
                ref.project,
                ref.op_name,
                architecture=ref.architecture,
                confirm_scope=bool(confirm_scope),
                reuse_artifacts=True,
            )
        except Exception as exc:  # noqa: BLE001
            runtime.clear_building(ref.id)
            return envelope(
                ok=False,
                state=STATE_FAILED,
                updated=False,
                error=str(exc)[:800],
                extra={
                    "engine": "update_operator",
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                },
            )
        if on_progress is not None:
            on_progress(1, 1, "done")
        runtime.clear_building(ref.id)
        status_name = str((result or {}).get("status") or "")
        plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
        change_set = (
            result.get("change_set") if isinstance(result.get("change_set"), dict) else {}
        )
        if status_name == "blocked":
            runtime.mark_blocked(ref.id)
            meta = _meta(ref.product)
            info = _freshness_for(ref, meta)
            handle = public_handle(ref, meta=meta, freshness_info=info)
            return envelope(
                ok=True,
                state=STATE_NEEDS_CONFIRMATION,
                updated=False,
                error_code="SCOPE_CONFIRMATION_REQUIRED",
                codemap=handle,
                extra={
                    "engine": "update_operator",
                    "status": status_name,
                    "run_id": result.get("run_id"),
                    "mode": plan.get("mode"),
                    "affected_layers": plan.get("affected_layers") or [],
                    "actions": plan.get("actions") or [],
                    "in_scope_count": int(change_set.get("scoped_change_count") or 0),
                    "needs_scope_review": True,
                    "path": str(ref.product or ""),
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                    "error": None,
                },
            )
        if status_name == "fail":
            runtime.clear_blocked(ref.id)
            return envelope(
                ok=False,
                state=STATE_FAILED,
                updated=False,
                error=str((result.get("receipt") or {}).get("message") or "update failed"),
                extra={
                    "engine": "update_operator",
                    "status": status_name,
                    "run_id": result.get("run_id"),
                    "path": str(ref.product or ""),
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                },
            )
        runtime.clear_blocked(ref.id)
        if ref.product is not None:
            runtime.cache.drop(ref.product)
        fresh = bind(
            project=ref.project,
            architecture=ref.architecture,
            registry=runtime.registry,
        )
        meta = _meta(fresh.product)
        info = _freshness_for(fresh, meta)
        handle = public_handle(fresh, meta=meta, freshness_info=info)
        updated = status_name == "pass" and str(plan.get("mode") or "") != "noop"
        return envelope(
            ok=True,
            state=STATE_COMPLETED,
            updated=updated,
            codemap=handle,
            extra={
                "engine": "update_operator",
                "status": status_name,
                "run_id": result.get("run_id"),
                "mode": plan.get("mode"),
                "affected_layers": plan.get("affected_layers") or [],
                "actions": plan.get("actions") or [],
                "in_scope_count": int(change_set.get("scoped_change_count") or 0),
                "needs_scope_review": bool(plan.get("needs_scope_review")),
                "path": str(fresh.product or ""),
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "error": None,
            },
        )

