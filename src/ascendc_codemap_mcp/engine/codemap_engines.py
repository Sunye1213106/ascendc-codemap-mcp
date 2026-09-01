# -*- coding: utf-8 -*-
"""Five public uo-init Actions for the source-backed CodeMap compiler.

UO extracts facts an Agent can query.  It does not solve the operator's full
19-dimensional TilingKey function.  In particular the public analyze path does
not run ``derive_key_fields`` or a global host-reachability SAT pass. Test construction
and local lemma reasoning belong to TG.

Canonical ``.uo`` is compiler truth + deterministic derivation only.  Semantic
residuals stay in ``unresolved.yaml``; LLM must not patch them into the product.
Optional investigation lives under ``/uo-investigate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ascendc_codemap_mcp.engine.paths import require_architecture
from ascendc_codemap_mcp.engine import pilot_engines as pe


def _chain(
    project_root: Path,
    payload: dict[str, Any] | None,
    steps: list[tuple[str, Callable[..., dict[str, Any]]]],
    *,
    engine: str,
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.progress import emit

    ctx = dict(payload or {})
    results: list[dict[str, Any]] = []
    total = len(steps)
    for idx, (name, fn) in enumerate(steps, start=1):
        emit(f"{engine} ({idx}/{total}) {name} …")
        import time

        t0 = time.perf_counter()
        out = fn(project_root, ctx)
        dt = time.perf_counter() - t0
        ok = bool(out.get("ok", False))
        mark = "ok" if ok else "FAIL"
        emit(f"{engine} ({idx}/{total}) {name} {mark} ({dt:.1f}s)")
        results.append({"step": name, **{k: out.get(k) for k in ("ok", "error", "engine")}})
        if not ok:
            return {
                "ok": False,
                "engine": engine,
                "failed_step": name,
                "error": out.get("error") or out.get("message_zh") or f"{name} failed",
                "reason_code": out.get("reason_code") or out.get("error") or "",
                "message_zh": out.get("message_zh") or "",
                "steps": results,
                "detail": out,
            }
        for key in ("op_name", "run_id"):
            if out.get(key) and not ctx.get(key):
                ctx[key] = out[key]
        if out.get("arch_dir"):
            ctx["arch_dir"] = out["arch_dir"]
            ctx["architecture"] = out.get("architecture") or out["arch_dir"]
        elif out.get("architecture") and not ctx.get("architecture"):
            ctx["architecture"] = out["architecture"]
    return {"ok": True, "engine": engine, "steps": results}


def prepare(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve Source Scope from operator+arch; machine-validate; seed BuildVariant.

    User chooses the analysis target. Clang decides the authoritative source
    closure. There is no human file-list confirmation and no decision=yes bypass.
    """
    from ascendc_codemap_mcp.engine.perf import stage

    with stage("prepare"):
        ctx = dict(payload or {})
        out = _chain(
            project_root,
            ctx,
            [
                ("prepare_layout", pe.prepare_layout),
                ("scope_scan", pe.scope_scan),
                ("scope_validate", pe.scope_validate),
            ],
            engine="prepare",
        )
        if out.get("ok"):
            try:
                from ascendc_codemap_mcp.engine.op_spec import discover
                from ascendc_codemap_mcp.engine.platform_ini import kernel_macros_for_arch

                import yaml

                root = Path(project_root).expanduser().resolve()
                spec = discover(root, arch_dir=ctx.get("arch_dir") or ctx.get("architecture"))
                arch = require_architecture(spec.arch_dir)
                uo = pe._uo_root(root, arch=arch)
                probe = uo / "cache" / "cann_9201_overlay" / "probe.yaml"
                cann_9201 = {}
                if probe.is_file():
                    loaded = yaml.safe_load(probe.read_text(encoding="utf-8")) or {}
                    if isinstance(loaded, dict):
                        cann_9201 = loaded
                payload = {
                    "schema": "build-variant/v1",
                    "architecture": arch,
                    "name": arch,
                    "source": "uo_init.codemap_engines.prepare",
                    "kernel_macros": kernel_macros_for_arch(arch),
                }
                if cann_9201:
                    payload["cann_9201"] = cann_9201
                pe._dump(uo / "ir" / "build_variant.yaml", payload)
            except Exception as exc:  # noqa: BLE001
                out["build_variant_warning"] = str(exc)[:200]
    return out


def extract(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run deterministic Clang/frontend extraction."""
    from ascendc_codemap_mcp.engine.perf import stage

    with stage("extract"):
        return _chain(
            project_root,
            payload,
            [
                ("extract_host", pe.extract_host),
            ],
            engine="extract",
        )


def _compiler_inputs(
    project_root: Path, ctx: dict[str, Any]
) -> tuple[str, str, Any, Any, dict[str, Any], Path]:
    """Resolve only structural inputs for CodeMap compilation.

    ``tiling/key_space.yaml`` is a deterministic declaration/schema artefact and
    is allowed.  ``host_derivation.yaml`` and per-key value expressions are
    intentionally not loaded here.
    """
    from ascendc_codemap_mcp.engine.op_spec import discover

    root = project_root.expanduser().resolve()
    ctx = pe._ctx(ctx)
    spec = discover(root, arch_dir=pe._payload_arch(ctx))
    op_name = str(ctx.get("op_name") or spec.op_name)
    arch = require_architecture(pe._payload_arch(ctx) or spec.arch_dir)
    uo = pe._uo_root(root, arch=arch)

    host_ir = None
    kernel_ir = None
    try:
        bundle = pe._ensure_bundle(root, ctx)
        host_ir = bundle.get("host_ir")
        kernel_ir = bundle.get("kernel_ir")
    except Exception:
        # Current-source enrichment in compile_codemap remains authoritative;
        # missing compiler IR becomes an explicit structural gap, not a reason
        # to fall back to removed symbolic host-reachability logic.
        pass

    declared = pe._load(uo / "tiling" / "key_space.yaml") or pe._load(
        uo / "ir" / "tiling_key_bindings.yaml"
    ) or {}
    if not isinstance(declared, dict):
        declared = {}
    return op_name, arch, host_ir, kernel_ir, declared, uo


def analyze(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a structural CodeMap dry-run and emit only extraction gaps.

    This stage answers: did UO recover API, Host provenance, TilingKey packing,
    TilingData transport, template/kernel structure and evidence-backed paths?
    It explicitly does *not* answer whether every declared packed key is
    reachable or derive a closed-form formula for every key dimension.

    Residuals are recorded in ``ir/unresolved.yaml`` and retained — they are not
    LLM-resolved into canonical ``.uo``.
    """
    from ascendc_codemap_mcp.engine.build import compile_codemap, load_fresh_compile_cache, store_compile_cache
    from ascendc_codemap_mcp.engine.perf import stage
    from ascendc_codemap_mcp.engine.progress import step

    ctx = pe._ctx(payload)
    root = Path(project_root).expanduser().resolve()
    try:
        with stage("analyze"):
            with step("analyze.resolve_inputs"):
                op_name, arch, host_ir, kernel_ir, declared, uo = _compiler_inputs(root, ctx)
            extract_fp = pe._current_extract_fingerprint(root, ctx)
            cached = load_fresh_compile_cache(
                root, op_name, arch, extract_fingerprint=extract_fp
            )
            reused = cached is not None
            if reused:
                result = cached
            else:
                with step("analyze.compile_codemap"):
                    result = compile_codemap(
                        op_name=op_name,
                        architecture=arch,
                        op_root=root,
                        host_ir=host_ir,
                        kernel_ir=kernel_ir,
                        declared=declared,
                        key_fields=[],
                        commit=False,
                    )
                with step("analyze.store_cache"):
                    store_compile_cache(
                        root, op_name, arch, result, extract_fingerprint=extract_fp
                    )
    except Exception as exc:  # noqa: BLE001
        try:
            from ascendc_codemap_mcp.engine.runtime import end_session

            end_session(op_root=root, architecture=str(ctx.get("architecture") or ctx.get("arch_dir") or ""))
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "engine": "analyze", "error": str(exc)[:400]}

    gaps = [g for g in (result.get("gaps") or []) if isinstance(g, dict)]
    from ascendc_codemap_mcp.engine.resolve.semantic_gap import summarize_gaps

    buckets = summarize_gaps(gaps)
    locate_blocking = int(buckets.get("locate_blocking") or 0)
    unresolved = {
        "schema": "codemap-structural-gaps/v1",
        "status": "unresolved" if gaps else "closed",
        "blocker_count": len(gaps),
        "locate_blocking": locate_blocking,
        "host_runtime_leaf": int(buckets.get("host_runtime_leaf") or 0),
        "catalog_unproven": int(buckets.get("catalog_unproven") or 0),
        "derivation_blocker_count": 0,
        "blockers": gaps,
        "scope": "structural_source_extraction",
        "policy": "retain_unresolved_no_llm_patch",
        "non_goals": [
            "global_tilingkey_value_derivation",
            "global_host_reachability_sat",
            "container_cardinality_proofs",
            "read_coverage_implication_proofs",
            "llm_semantic_gap_patching",
        ],
    }
    receipt = {
        "ok": True,
        "engine": "analyze",
        "schema": "uo-codemap-analyze/v1",
        "op_name": op_name,
        "architecture": arch,
        "summary": dict(result.get("summary") or {}),
        "gap_count": len(gaps),
        "locate_blocking": locate_blocking,
        "analysis_policy": "structure_and_provenance_only",
        "deep_key_derivation": False,
        "global_sat": False,
        "compile_cached": True,
        "reused_compile_cache": reused,
        "semantic_completeness": "complete" if locate_blocking == 0 else "partial",
    }
    pe._dump(uo / "ir" / "unresolved.yaml", unresolved)
    pe._dump(uo / "ir" / "codemap_analyze_receipt.yaml", receipt)
    return receipt


def resolve(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Removed from ``/uo-init``. Use ``/uo-investigate`` for residuals."""
    return pe.resolve_gaps(project_root, payload)


def commit(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compile current structural facts/source into the single ``.uo`` product.

    Open semantic residuals are allowed: commit writes a valid but possibly
    incomplete CodeMap (``semantic_completeness=partial``). Hard extraction
    failures still fail this stage.
    """
    from ascendc_codemap_mcp.engine.perf import stage
    from ascendc_codemap_mcp.engine.progress import step

    ctx = dict(payload or {})
    root = Path(project_root).expanduser().resolve()
    with stage("commit"), step("commit.write_uo_product"):
        product = _commit_uo_product(root, ctx)
    if product.get("ok") and product.get("path"):
        _stamp_completeness(root, ctx, str(product.get("path")))
    if not product.get("ok"):
        try:
            from ascendc_codemap_mcp.engine.runtime import end_session

            end_session(
                op_root=root,
                architecture=str(ctx.get("architecture") or ctx.get("arch_dir") or ""),
                drop_compile_mem=False,
            )
        except Exception:  # noqa: BLE001
            pass
    return {
        "ok": bool(product.get("ok")),
        "engine": "commit",
        "uo_product": product,
        "path": product.get("path"),
        "summary": product.get("summary"),
        "gaps": product.get("gaps"),
        "reused_analyze": bool(product.get("reused_analyze")),
        **({"error": product.get("error") or "uo_commit_failed"} if not product.get("ok") else {}),
    }


def _stamp_completeness(root: Path, ctx: dict[str, Any], product: str) -> None:
    """Carry the analyze verdict into the product's ``meta`` table.

    It was only ever written to ``ir/codemap_analyze_receipt.yaml``, so nothing
    downstream could read it: ``semantic_completeness`` came back null and
    ``gaps_count`` came back 0 because both keys were absent, not because the
    graph was whole. An index that reports zero gaps it never counted is worse
    than one that reports nothing.
    """
    import yaml

    try:
        from ascendc_codemap_mcp.engine.store.writer import upsert_meta

        arch = require_architecture(pe._payload_arch(pe._ctx(dict(ctx))))
        receipt_path = pe._uo_root(root, arch=arch) / "ir" / "codemap_analyze_receipt.yaml"
        if not receipt_path.is_file():
            return
        receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8")) or {}
        if not isinstance(receipt, dict):
            return
        items = {
            "semantic_completeness": str(receipt.get("semantic_completeness") or "unknown"),
            "analyze_gap_count": int(receipt.get("gap_count") or 0),
            "analyze_locate_blocking": int(receipt.get("locate_blocking") or 0),
            "analyze_measured": "1",
        }
        upsert_meta(product, items)
    except Exception:  # noqa: BLE001
        return


def verify(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate graph legality and emit a cannbot locate quality receipt.

    Integrity (``checks/integrity.yaml``) is schema/invariant/dangling-edge
    legality and remains the workflow gate. Open unresolved blockers do not
    fail it. Quality (``checks/quality.yaml``) grades whether the CodeMap can
    replace grep for structural locate questions (ready / usable / not_ready).
    """
    import time

    from ascendc_codemap_mcp.engine.perf import record_stage
    from ascendc_codemap_mcp.engine.progress import step

    # verify emits performance.yaml itself, so its own duration is stamped just
    # before the dump rather than by the stage() context manager.
    verify_t0 = time.perf_counter()
    ctx = dict(payload or {})
    root = Path(project_root).expanduser().resolve()
    try:
        from ascendc_codemap_mcp.engine.diagnostics.audit import audit_uo
        from ascendc_codemap_mcp.engine.diagnostics.quality import codemap_quality
        from ascendc_codemap_mcp.engine.resolve.semantic_gap import list_gaps
        from ascendc_codemap_mcp.engine.store.reader import find_uo_product, read_codemap

        with step("verify.find_uo_product"):
            arch = require_architecture(ctx.get("arch_dir") or ctx.get("architecture"))
            op_name = str(ctx.get("op_name") or "")
            product = find_uo_product(root, op_name=op_name, architecture=arch)
        if product is None or product.suffix != ".uo":
            return {
                "ok": False,
                "engine": "verify",
                "error": "missing_uo_product",
                "message": "commit must write .ascendc-codemap/<arch>/<op>.<arch>.uo",
            }
        with step("verify.audit_uo"):
            report = audit_uo(product)
        ok = bool(report.get("ok"))
        with step("verify.write_integrity_receipt"):
            uo = pe._uo_root(root, arch=arch)
            integrity = {
                "version": 1,
                "schema": "uo-product-integrity/v1",
                "status": "pass" if ok else "fail",
                "ok": ok,
                "uo_product": str(product),
                "architecture": arch,
                "op_name": op_name or product.stem.split(".")[0],
                "audit_ok": ok,
                "source": "uo-init/verify",
            }
            pe._dump(uo / "checks" / "integrity.yaml", integrity)
        with step("verify.write_quality_receipt"):
            cm = read_codemap(product)
            quality = codemap_quality(
                cm,
                integrity_ok=ok,
                audit=report,
                gaps=list_gaps(cm),
                source_root=root,
            )
            quality["uo_product"] = str(product)
            quality["architecture"] = arch
            quality["op_name"] = op_name or product.stem.split(".")[0]
            pe._dump(uo / "checks" / "quality.yaml", quality)
        with step("verify.write_performance_receipt"):
            perf_path = None
            try:
                from ascendc_codemap_mcp.engine.perf import dump_yaml

                record_stage("verify", time.perf_counter() - verify_t0)
                perf_path = uo / "checks" / "performance.yaml"
                dump_yaml(
                    perf_path,
                    extra={
                        "operator": op_name or product.stem.split(".")[0],
                        "architecture": arch,
                    },
                )
            except Exception:  # noqa: BLE001
                perf_path = None
        try:
            from ascendc_codemap_mcp.engine.runtime import end_session

            end_session(op_root=root, architecture=arch)
        except Exception:  # noqa: BLE001
            pass
        try:
            from ascendc_codemap_mcp.engine.diagnostics.quality import ready_status_fields

            status = ready_status_fields(quality)
        except Exception:  # noqa: BLE001
            status = {}
        return {
            "ok": ok,
            "engine": "verify",
            "path": str(product),
            "audit": report,
            "verdict": "pass" if ok else "fail",
            **status,
            "integrity": str(uo / "checks" / "integrity.yaml"),
            "quality": str(uo / "checks" / "quality.yaml"),
            "performance": str(perf_path) if perf_path else None,
            "quality_grade": quality.get("grade"),
            "locate_ready": quality.get("locate_ready"),
        }
    except Exception as exc:  # noqa: BLE001
        try:
            from ascendc_codemap_mcp.engine.runtime import end_session

            end_session(op_root=root, architecture=str(ctx.get("architecture") or ctx.get("arch_dir") or ""))
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "engine": "verify", "error": str(exc)[:400], "verdict": "fail"}


def review(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Backward-compatible alias for :func:`verify`."""
    out = verify(project_root, payload)
    if isinstance(out, dict) and out.get("engine") == "verify":
        out = dict(out)
        out["engine"] = "review"
        out["alias_of"] = "verify"
    return out


def _commit_uo_product(project_root: Path, ctx: dict[str, Any]) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.build import compile_codemap, load_compile_cache
    from ascendc_codemap_mcp.engine.store.writer import uo_product_path, write_codemap
    from ascendc_codemap_mcp.engine.tg_projection import require_commit_views

    root = project_root.expanduser().resolve()
    try:
        op_name, arch, host_ir, kernel_ir, declared, _uo = _compiler_inputs(root, ctx)
        cached = load_compile_cache(root, op_name, arch)
        if cached is not None and cached.get("codemap") is not None:
            views = cached.get("views") or {}
            missing = require_commit_views(views)
            if missing:
                return {
                    "ok": False,
                    "error": "TG_VIEW_INCOMPLETE",
                    "missing": missing,
                    "reused_analyze": True,
                }
            path = uo_product_path(root, op_name, arch)
            from ascendc_codemap_mcp.engine.store.writer import detect_source_revision

            revision = detect_source_revision(root)
            written = write_codemap(
                cached["codemap"],
                path,
                views=views,
                summary=cached.get("summary"),
                meta={"source_revision": revision} if revision else None,
            )
            from ascendc_codemap_mcp.engine.build import drop_compile_mem

            drop_compile_mem(root, architecture=arch)
            return {
                "ok": bool(written.get("ok")),
                "path": written.get("path"),
                "summary": cached.get("summary"),
                "audit": cached.get("audit"),
                "gaps": cached.get("gaps"),
                "uo": written,
                "reused_analyze": True,
                "analysis_policy": "structure_and_provenance_only",
            }
        result = compile_codemap(
            op_name=op_name,
            architecture=arch,
            op_root=root,
            host_ir=host_ir,
            kernel_ir=kernel_ir,
            declared=declared,
            key_fields=[],
            commit=True,
        )
        from ascendc_codemap_mcp.engine.build import drop_compile_mem

        drop_compile_mem(root, architecture=arch)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:400]}

    return {
        "ok": bool(result.get("ok")),
        "path": result.get("path"),
        "summary": result.get("summary"),
        "audit": result.get("audit"),
        "gaps": result.get("gaps"),
        "uo": result.get("uo"),
        "error": result.get("error"),
        "missing": result.get("missing"),
        "reused_analyze": False,
        "analysis_policy": "structure_and_provenance_only",
    }
