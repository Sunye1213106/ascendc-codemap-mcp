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
    building = runtime.is_building(ref.id)
    if building:
        meta: dict[str, Any] = {}
        info = {
            "freshness": "building",
            "source_revision": "",
            "indexed_revision": "",
            "dirty": False,
            "changed_files": 0,
            "semantic_completeness": None,
        }
    else:
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


def _env_next_steps(
    *,
    clang_exe: str | None,
    libclang_ok: bool,
    cann: Path | None,
) -> list[str]:
    """Actionable bootstrap for an agent when doctor is not green."""
    import os

    from ascendc_codemap_mcp.constants import (
        CANN_DOWNLOAD_CENTER,
        CANN_TOOLKIT_RUN_NAME,
    )
    from ascendc_codemap_mcp.engine.paths import repo_root

    dest = repo_root() / "_cann" / "pkg"
    steps: list[str] = []
    if not libclang_ok:
        steps.append("pip install 'libclang>=18.1.1'")
    if not clang_exe:
        if os.name == "nt":
            steps.append("winget install --id LLVM.LLVM --version 18.1.8 -e")
            steps.append(
                r"If clang is not on PATH, set CLANG_EXE=C:\Program Files\LLVM\bin\clang.exe"
            )
        else:
            steps.append("sudo apt-get update && sudo apt-get install -y clang")
        steps.append("clang --version")
    if cann is None:
        steps.append(
            f"Search the machine for {CANN_TOOLKIT_RUN_NAME} "
            "(Downloads, ~/Downloads) before asking the user to download again."
        )
        steps.append(
            "Download CANN Toolkit .run (Huawei/Ascend login required; "
            f"unsigned wget usually fails): {CANN_DOWNLOAD_CENTER}"
        )
        steps.append(
            "Need the Toolkit package, not kernels/nnal. On Windows still download "
            "the linux-x86_64 .run — cann-extract unpacks it and does not execute "
            "the installer."
        )
        steps.append(
            "python -m ascendc_codemap_mcp cann-extract "
            f"<path-to/{CANN_TOOLKIT_RUN_NAME}> --dest {dest}"
        )
        steps.append(
            f"python -m ascendc_codemap_mcp cann-extract --fixup --dest {dest}"
        )
        steps.append(
            f"Optional: set ASCENDC_CODEMAP_CANN_ROOT={dest} "
            "(User-level env; a session-only export disappears)."
        )
    return steps


def doctor(
    *,
    project: str = "",
    architecture: str = "",
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.clang_cmd import find_clang
    from ascendc_codemap_mcp.engine.paths import explain, require_cann_ready

    issues: list[str] = []
    root = Path(project).expanduser() if str(project or "").strip() else None
    if root is not None and not root.is_dir():
        issues.append(f"operator directory not found: {root}")
    arch = str(architecture or "").strip()
    if not arch:
        issues.append("architecture is required (e.g. arch35)")
    libclang_ok = False
    try:
        import clang.cindex  # noqa: F401

        libclang_ok = True
    except Exception as exc:  # noqa: BLE001
        issues.append(f"libclang import failed: {exc}")
    clang_exe = find_clang()
    if not clang_exe:
        issues.append(
            "clang executable not found (pip libclang is not enough; "
            "TPL preprocess needs clang -E). Install LLVM 18 and/or set CLANG_EXE."
        )
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
        "clang_exe": clang_exe,
        "libclang_ok": libclang_ok,
        "issues": issues,
        "next_steps": _env_next_steps(
            clang_exe=clang_exe,
            libclang_ok=libclang_ok,
            cann=cann,
        ),
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
        "coverage": index_coverage(summary),
        "steps": steps,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "product": product,
        "meta": summary,
    }


def index_coverage(meta: dict[str, Any] | None) -> dict[str, Any]:
    """What this build measured, and what it did not.

    The single ``gaps_count`` this replaces read 0 on a product whose meta
    table held no such key, next to a null completeness, while the same table
    recorded seven unresolved kernel sync pairs and a host/kernel path with no
    evidence behind it. One number cannot carry that, and defaulting it to zero
    reports the absence of a measurement as a clean bill of health.
    """
    meta = dict(meta or {})

    def _int(key: str) -> int | None:
        raw = meta.get(key)
        if raw in (None, ""):
            return None
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return None

    measured = str(meta.get("analyze_measured") or "") == "1"
    gaps = _int("analyze_gap_count")
    blocking = _int("analyze_locate_blocking")
    out: dict[str, Any] = {
        "semantic_completeness": str(meta.get("semantic_completeness") or "") or "not_measured",
        "analyze_gaps": gaps if measured and gaps is not None else "not_measured",
        "locate_blocking": blocking if measured and blocking is not None else "not_measured",
    }
    closure = str(meta.get("cm_has_strict_kernel_tiling_closure") or "")
    if closure:
        out["strict_kernel_tiling_closure"] = closure == "true"
    host_path = str(meta.get("cm_has_evidence_backed_host_kernel_path") or "")
    if host_path:
        out["evidence_backed_host_kernel_path"] = host_path == "true"
    keys = _int("cm_legal_key_count")
    if keys is not None:
        out["compiled_legal_keys"] = keys
    return out


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
        ref = bind(project=root, architecture=arch, registry=runtime.registry)
        if ref.product is not None and Path(ref.product).is_file():
            meta = _meta(ref.product)
            from ascendc_codemap_mcp.service.freshness import compute as compute_freshness

            info = compute_freshness(
                ref.project,
                meta=meta,
                building=False,
                blocked=runtime.is_blocked(ref.id),
            )
            handle = public_handle(ref, meta=meta, freshness_info=info)
            return envelope(
                ok=True,
                state=STATE_COMPLETED,
                updated=False,
                codemap=handle,
                extra={
                    "engine": "codemap_index",
                    "mode": "noop",
                    "error_code": "ALREADY_INDEXED",
                    "hint": "CodeMap already exists; use codemap_update",
                    "path": str(ref.product),
                },
            )
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
                "coverage": result.get("coverage") or {},
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
        ref = bind(
            project=ref.project,
            architecture=ref.architecture,
            registry=runtime.registry,
        )
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
        meta_now = _meta(ref.product)
        from ascendc_codemap_mcp.service.freshness import compute as compute_freshness

        fresh_now = compute_freshness(
            ref.project,
            meta=meta_now,
            building=False,
            blocked=runtime.is_blocked(ref.id),
        )
        if str(fresh_now.get("freshness") or "") == "fresh":
            runtime.clear_building(ref.id)
            handle = public_handle(ref, meta=meta_now, freshness_info=fresh_now)
            return envelope(
                ok=True,
                state=STATE_COMPLETED,
                updated=False,
                codemap=handle,
                extra={
                    "engine": "update_operator",
                    "status": "pass",
                    "mode": "noop",
                    "path": str(ref.product or ""),
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                    "error": None,
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
                should_stop=should_stop,
                on_progress=on_progress,
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
        if status_name == "cancelled":
            return envelope(
                ok=False,
                state=STATE_IDLE,
                updated=False,
                error="cancelled",
                error_code="CANCELLED",
                extra={
                    "engine": "update_operator",
                    "failed_step": result.get("failed_step"),
                    "run_id": result.get("run_id"),
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                },
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

