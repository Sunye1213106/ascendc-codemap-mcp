# -*- coding: utf-8 -*-
"""Pilot Action engines for the clang-based uo-init workflow.

Each entrypoint has signature ``fn(project_root, payload) -> dict`` with an
``ok`` field.  Engines write under ``.ascendc-codemap/<arch>/`` only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ascendc_codemap_mcp.engine import paths
from ascendc_codemap_mcp.engine.source_layout import is_product_architecture


def _payload_arch(ctx: dict[str, Any] | None) -> str | None:
    ctx = ctx or {}
    for key in ("arch_dir", "architecture", "arch"):
        val = str(ctx.get(key) or "").strip()
        if val:
            return val
    return None


def _uo_root(project_root: Path, *, arch: str | None = None) -> Path:
    """Arch-scoped working tree ``.ascendc-codemap/<arch>/``."""
    from ascendc_codemap_mcp.engine.paths import product_dir
    from ascendc_codemap_mcp.engine.source_layout import is_product_architecture

    root = Path(project_root).expanduser().resolve()
    name = (arch or "").strip() or None
    if name:
        return product_dir(root, name)
    import os

    env_arch = (
        os.environ.get("ASCENDC_CODEMAP_ARCH")
        or os.environ.get("UO_ARCH")
        or os.environ.get("ASCENDC_ARCH")
        or ""
    ).strip()
    if env_arch:
        return product_dir(root, env_arch)
    product_root = root / ".ascendc-codemap"
    if product_root.is_dir():
        arch_dirs = sorted(
            p
            for p in product_root.iterdir()
            if p.is_dir() and is_product_architecture(p.name)
        )
        with_product = [p for p in arch_dirs if any(p.glob("*.uo"))]
        chosen = (
            with_product[0]
            if len(with_product) == 1
            else (arch_dirs[0] if len(arch_dirs) == 1 else None)
        )
        if chosen is not None:
            return chosen
    raise ValueError("ARCHITECTURE_MISSING_IN_RUN_STATE: architecture required")


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from yaml import CSafeDumper as _Dumper
    except ImportError:
        from yaml import SafeDumper as _Dumper
    path.write_text(
        yaml.dump(
            payload,
            Dumper=_Dumper,
            allow_unicode=True,
            sort_keys=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )


def _quote_unquoted_snippets(text: str) -> str:
    """LLM proposals often leave ``snippet: !foo && bar`` unquoted; quote them."""
    import re

    def _fix(m: re.Match[str]) -> str:
        indent, val = m.group(1), m.group(2)
        if not val or val[:1] in "\"'|[{":
            return m.group(0)
        if not any(ch in val for ch in "!&*:{}[],"):
            return m.group(0)
        esc = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'{indent}snippet: "{esc}"'

    return re.sub(r"^([ \t]*)snippet: (.+)$", _fix, text, flags=re.M)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        data = yaml.safe_load(_quote_unquoted_snippets(text)) or {}
    return data if isinstance(data, dict) else {}


def _load_yaml_scalar(path: Path, key: str) -> str:
    """Read one top-level YAML scalar without parsing a large graph file."""
    if not path.is_file():
        return ""
    prefix = f"{key}:"
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text.startswith(prefix):
                    continue
                value = text[len(prefix):].strip()
                if not value:
                    return ""
                if " #" in value:
                    value = value.split(" #", 1)[0].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                return value
    except OSError:
        return ""
    return ""


def _ctx(payload: dict[str, Any] | None) -> dict[str, Any]:
    ctx = dict(payload or {})
    arch = _payload_arch(ctx)
    if arch:
        ctx.setdefault("arch_dir", arch)
        ctx.setdefault("architecture", arch)
    return ctx


def _flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def _cann_root(ctx: dict[str, Any]) -> str:
    found = paths.cann_root(ctx.get("cann_root"))
    if found is None:
        # Returning a path that does not exist gives clang a clearer failure
        # than returning None does three frames further down.
        raise FileNotFoundError(f"CANN packages not found.\n{paths.explain()}")
    return str(found)


def _cann_env_block(engine: str, ctx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Fail closed only when cann_root is missing or not a CANN tree.

    Official toolkit/install packages are complete. Do not block prepare on a
    hardcoded file inventory; clang + include_heal align -I to the real tree.
    """
    root, issues = paths.require_cann_ready((ctx or {}).get("cann_root"))
    if not issues:
        return None
    detail = "; ".join(issues[:8])
    return {
        "ok": False,
        "engine": engine,
        "error": "CANN_ENV_NOT_READY",
        "cann_root": str(root) if root else None,
        "issues": issues,
        "message_zh": (
            "UO 解析前未找到 CANN 根目录。"
            f"{detail}。"
            "把 toolkit 解到当前仓库 _cann/pkg（自动发现），"
            "或设置 ASCENDC_CODEMAP_CANN_ROOT / ASCEND_CANN_PACKAGE_PATH 指向解包后的 cann-* 根，"
            "或官方安装的 ASCEND_HOME_PATH。"
            "官方 CANN 包不缺头文件；配好 cann_root 后 prepare 不再按单个相对路径失败。"
            "doctor / check_cann.py / prepare 共用 require_cann_ready；"
            "可先执行: python scripts/dev/check_cann.py"
        ),
    }


def _ops_root(ctx: dict[str, Any], project_root: Path) -> str | None:
    raw = ctx.get("ops_root")
    if raw:
        return str(raw)
    # Typical layout: …/ops-transformer/attention/<op>. Confirm by shape rather
    # than by existence, or an operator two levels below anything at all would
    # silently hand clang an include root with no headers in it.
    parent = project_root.parent.parent
    if (parent / "common" / "include").is_dir():
        return str(parent)
    found = paths.ops_root()
    return str(found) if found is not None else None


def _run_dir(uo: Path, ctx: dict[str, Any]) -> Path:
    run_id = str(ctx.get("run_id") or "default").strip() or "default"
    d = uo / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_layout(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Discover operator layout and seed the prepare working tree.

    Resets ``.ascendc-codemap/<arch>/`` to runs/summary (+ cache),
    writes manifest / operator / layout_receipt. Does **not** seed legacy layered
    KB stubs (``flow/``, ``tiling/``, ``kernel/``, ``data_model``, ``pipeline``, …)
    — canonical product is the single ``.uo`` CodeMap written at commit.
    """
    from ascendc_codemap_mcp.engine.op_spec import discover

    ctx = _ctx(payload)
    root = Path(project_root).expanduser().resolve()
    run_id = str(ctx.get("run_id") or "").strip()
    if not run_id:
        from datetime import datetime, timezone

        run_id = datetime.now(timezone.utc).strftime("idx-%Y%m%dT%H%M%SZ")
        ctx["run_id"] = run_id
    blocked = _cann_env_block("prepare_layout", ctx)
    if blocked is not None:
        return blocked
    try:
        spec = discover(root, arch_dir=ctx.get("arch_dir"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "engine": "prepare_layout", "error": str(exc)[:400]}

    product_name = (
        f"{spec.op_name}.{spec.arch_dir}.uo" if spec.arch_dir else f"{spec.op_name}.uo"
    )
    uo = _uo_root(root, arch=spec.arch_dir)
    scrub = _reset_uo_skeleton(uo, run_id=run_id, keep_other_runs=bool(ctx.get("keep_other_runs")))
    try:
        from ascendc_codemap_mcp.engine.store.writer import detect_source_revision

        source_revision = detect_source_revision(root) or "unknown"
    except Exception:  # noqa: BLE001
        source_revision = "unknown"

    manifest = {
        "version": 1,
        "status": "prepared",
        "authority": "uo",
        "product": product_name,
        "derived_index": product_name,
        "op_name": spec.op_name,
        "architecture": spec.arch_dir,
        "schema": "uo-codemap/v1",
        "run_id": run_id,
        "source": {
            "kind": "uo_init.pilot_engines.prepare_layout",
            "revision": source_revision,
            "root": str(root),
        },
        "source_revision": source_revision,
        "workflow": "uo-init",
        "contract": "clang-codemap",
    }
    operator = {
        "version": 1,
        "status": "prepared",
        "op_name": spec.op_name,
        "op_snake": spec.op_snake,
        "architecture": spec.arch_dir,
        "op_spec": spec.to_dict(),
        "ambiguities": list(spec.ambiguities),
    }
    _dump(uo / "manifest.yaml", manifest)
    _dump(uo / "operator.yaml", operator)
    scope = uo / "runs" / run_id / "scope"
    _dump(
        scope / "layout_receipt.yaml",
        {
            "ok": True,
            "op_name": spec.op_name,
            "run_id": run_id,
            "schema": "uo-codemap/v1",
            "scrubbed": scrub,
        },
    )
    return {
        "ok": True,
        "engine": "prepare_layout",
        "op_name": spec.op_name,
        "run_id": run_id,
        "manifest": (uo / "manifest.yaml").as_posix(),
        "ambiguous": bool(spec.ambiguities),
        "scrubbed_paths": scrub.get("removed") or [],
        "seeded_not_extracted": list(scrub.get("seeded_not_extracted") or []),
        "layout_reset": bool(scrub.get("removed")),
        "arch_dir": spec.arch_dir,
        "architecture": spec.arch_dir,
    }


# Prepare-only working dirs under <arch>/uo/.
# Canonical product is `.ascendc-codemap/<arch>/<op>.<arch>.uo` (commit).
# Empty tiling/kernel folders were leftovers from layered-KB receipts that
# extract no longer writes; views live inside the ``.uo`` product.
_UO_SEED_DIRS = (
    "summary",  # scope mirrors for gates / update fallback
    "runs",
)

_DISALLOWED_TOP_DIRS = (
    "analysis",
    "diff",
    "docs_cache",
    "test",
    "generated",
    "ledger",
    # Legacy layered-KB product tree — replaced by single .uo CodeMap.
    "flow",
    "tiling",
    "kernel",
)

# Created by extract/analyze/export; prepare must not leave them around.
_DEFER_UNTIL_EXPORT = (
    "ir",
    "checks",
    "cross_layer",
    "indexes",
    "review",
)

# Leftover files from the old prepare stub / layered-KB path.
_LEGACY_STUB_FILES = (
    "tiling/data_model.yaml",
    "tiling/key_space.yaml",
    "tiling/key_derivations.yaml",
    "kernel/pipeline.yaml",
    "kernel/resources.yaml",
    "flow/golden_model.yaml",
    "flow/numerical_model.yaml",
    "ir/_host_bundle_meta.yaml",
    "ir/full_init_timing_report.json",
)


def _reset_uo_skeleton(uo: Path, *, run_id: str, keep_other_runs: bool = False) -> dict[str, Any]:
    """Reset uo/ to the prepare-allowed skeleton (no layered-KB stubs)."""
    import shutil

    removed: list[str] = []
    uo.mkdir(parents=True, exist_ok=True)

    for name in _DISALLOWED_TOP_DIRS:
        path = uo / name
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(name)

    # Product dirs created by extract/export — remove at prepare so the tree
    # only contains what this Action is allowed to seed.
    for name in _DEFER_UNTIL_EXPORT:
        path = uo / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(name)

    for name in _UO_SEED_DIRS:
        path = uo / name
        if name == "runs":
            path.mkdir(parents=True, exist_ok=True)
            if not keep_other_runs:
                for child in list(path.iterdir()):
                    if child.name == run_id:
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                    removed.append(f"runs/{child.name}")
            (path / run_id / "scope").mkdir(parents=True, exist_ok=True)
            continue
        path.mkdir(parents=True, exist_ok=True)

    # Keep durable libclang WalkResult cache across prepare — deleting it forces
    # a full host||kernel rewalk (FAG cold ~5min). Product YAML/IR still reset.
    keep_top = set(_UO_SEED_DIRS) | {
        "manifest.yaml",
        "operator.yaml",
        "quality.yaml",
        "cache",
    }
    for child in list(uo.iterdir()):
        # Durable CodeMap stays queryable until commit replaces it. Cross-workflow
        # start ("不删正式产物") must not make `acp uo-query` see "found none".
        if child.is_file() and child.suffix == ".uo":
            continue
        if child.name in keep_top:
            if child.name == "quality.yaml" and child.is_file():
                child.unlink()
                removed.append("quality.yaml")
            continue
        if child.name in _DISALLOWED_TOP_DIRS or child.name in _DEFER_UNTIL_EXPORT:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(child.name)

    for rel in _LEGACY_STUB_FILES:
        path = uo / rel
        if path.is_file():
            path.unlink()
            removed.append(rel)

    return {"removed": sorted(set(removed)), "seeded_not_extracted": []}


def scope_scan(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve Source Scope: layout bootstrap → Clang authoritative closure → probe."""
    import os

    from ascendc_codemap_mcp.engine import scope_scan as sscan
    from ascendc_codemap_mcp.engine.build_context import BuildContext
    from ascendc_codemap_mcp.engine.clang_walk import probe_diagnostics
    from ascendc_codemap_mcp.engine.op_spec import discover

    ctx = _ctx(payload)
    root = Path(project_root).expanduser().resolve()
    uo = _uo_root(root, arch=_payload_arch(ctx))
    run = _run_dir(uo, ctx)
    blocked = _cann_env_block("scope_scan", ctx)
    if blocked is not None:
        return blocked
    try:
        spec = discover(root, arch_dir=ctx.get("arch_dir"))
        cann = _cann_root(ctx)
        bctx = BuildContext.load(
            cann_root=cann,
            ops_root=_ops_root(ctx, root),
            op_dir=str(spec.op_dir),
            arch_dir=spec.arch_dir,
            apply_saved_extras=False,
        )
        from ascendc_codemap_mcp.engine.kernel_tiling_view import install_kernel_tiling_view
        from ascendc_codemap_mcp.engine.include_heal import HealReport, apply_saved_extras, clear_saved_extras, save_extras

        run_id = str(ctx.get("run_id") or "").strip() or None
        # Script extras are per-prepare; heal_promote -I must survive the rerun.
        clear_saved_extras(spec.op_dir, spec.arch_dir, run_id=run_id)
        apply_saved_extras(bctx)
        install_kernel_tiling_view(spec, bctx)
        save_extras(bctx, HealReport(enabled=True), run_id=run_id)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "engine": "scope_scan", "error": str(exc)[:400]}

    # All selected Host TUs + current-arch kernel entry — never a fixed [:N] cut.
    # Architecture only selects which entry TUs Clang parses; the file list is
    # whatever those TUs actually include.
    hosts = [p for p in spec.host_targets if p.exists()]
    kernel = spec.kernel_entry if spec.kernel_entry and spec.kernel_entry.exists() else None
    if kernel is not None:
        from ascendc_codemap_mcp.engine.source_layout import (
            architecture_in_scope,
            architectures_match,
            path_owned_architecture,
        )

        owns_path = path_owned_architecture(kernel)
        owns_inc = sscan.entry_architecture(kernel)
        arch = (spec.arch_dir or "").strip()
        drop = False
        if owns_path:
            drop = bool(arch and not architectures_match(owns_path, arch))
        elif owns_inc and arch:
            if not architecture_in_scope(owns_inc, arch):
                drop = True
            elif spec.scope is not None and not architectures_match(owns_inc, arch):
                spec.scope.notes.append(
                    f"kernel_entry_kept_last_tu: {kernel.name} includes {owns_inc}"
                )
        if drop:
            if spec.scope is not None:
                spec.scope.notes.append(
                    f"kernel_entry_other_arch: {kernel.name} builds "
                    f"{owns_path or owns_inc}; not used as clang entry"
                )
            kernel = None
            spec.kernel_entry = None

    clang_status = "incomplete"
    clang_tus_expected = 0
    clang_tus_parsed = 0
    clang_errors: list[str] = []
    enrich_probes: list[dict[str, Any]] = []
    include_heal_report: dict[str, Any] = {}
    scope = spec.scope
    if scope is not None:
        try:
            from ascendc_codemap_mcp.engine.include_heal import enrich_scope_with_heal

            base_scope = scope
            run_id = str(ctx.get("run_id") or "").strip() or None

            def _enrich():
                return sscan.enrich_with_clang(
                    base_scope,
                    host_args=bctx.host_args(),
                    kernel_args=bctx.kernel_args(
                        dtype_variant="DT_FLOAT16", source_path=kernel
                    ),
                    host_tus=hosts,
                    kernel_tu=kernel,
                    walk_ctx=bctx,
                )

            enrichment, heal_rep = enrich_scope_with_heal(
                ctx=bctx,
                host_tus=hosts,
                kernel_tu=kernel,
                enrich_fn=_enrich,
                run_id=run_id,
            )
            include_heal_report = heal_rep.to_dict()
            scope = enrichment.scope
            spec.scope = scope
            clang_status = enrichment.status
            clang_tus_expected = enrichment.tus_expected
            clang_tus_parsed = enrichment.tus_parsed
            clang_errors = list(enrichment.errors)
            enrich_probes = list(enrichment.probes or [])
        except Exception as exc:  # noqa: BLE001
            clang_status = "incomplete"
            clang_errors = [f"clang_enrichment_failed:{str(exc)[:200]}"]
            if scope is not None:
                scope.notes.append(clang_errors[0])

    probes: list[dict[str, Any]] = []
    host_errors = 0
    kernel_errors = 0
    # Test/dev escape hatch only — never decision=yes / force_confirm on product path.
    allow_unverified = str(
        os.environ.get("UO_TEST_ALLOW_UNVERIFIED_SCOPE") or ""
    ).strip().lower() in {"1", "true", "yes"}

    def _probe_score(res: dict[str, Any]) -> tuple[int, int, list[str]]:
        """Relevant probe errors: operator-source + any fatal (missing includes)."""
        if "probe_relevant_errors" in res:
            relevant = int(res.get("probe_relevant_errors") or 0)
        else:
            # Back-compat for older cached probe payloads.
            op_errs = int(res.get("operator_error_count") or res.get("error_count") or 0)
            fatals = int(res.get("fatal_count") or 0)
            relevant = op_errs + (0 if fatals == 0 else fatals)
        fatals = int(res.get("fatal_count") or 0)
        samples = [str(s) for s in (res.get("samples") or [])[:5]]
        return relevant, fatals, samples

    def _probe_entry(path: Path, side: str, res: dict[str, Any], kind: str) -> tuple[dict[str, Any], int]:
        errs, fatals, samples = _probe_score(res)
        entry: dict[str, Any] = {
            "file": path.as_posix(),
            "errors": errs,
            "fatal": fatals,
            "raw_error_count": int(res.get("error_count") or 0),
            "operator_error_count": int(res.get("operator_error_count") or 0),
            "side": side,
            "probe": kind,
        }
        if samples:
            entry["samples"] = samples
        if "benign_external_decl_only" in res:
            entry["benign_external_decl_only"] = bool(res.get("benign_external_decl_only"))
        return entry, errs

    from ascendc_codemap_mcp.engine.progress import emit as _progress

    reused_includes = False
    expected = [(p, "host") for p in hosts]
    if kernel is not None:
        expected.append((kernel, "kernel"))
    by_file = {
        str(row.get("file") or "").replace("\\", "/").lower(): row
        for row in enrich_probes
        if row.get("file")
    }
    if expected and all(
        p.as_posix().replace("\\", "/").lower() in by_file for p, _side in expected
    ):
        _progress("prepare probe reused from clang include parse")
        reused_includes = True
        for path, side in expected:
            res = by_file[path.as_posix().replace("\\", "/").lower()]
            entry, errs = _probe_entry(path, side, res, "clang_includes")
            probes.append(entry)
            if side == "host":
                host_errors += max(errs, 0)
            else:
                kernel_errors = errs
        # Full-body include parse of kernel TUs reports NPU asm / vector types
        # that declarations-only probe (SKIP_FUNCTION_BODIES) correctly ignores.
        # Fall back only for dirty TUs so the gate stays equivalent.
        dirty = [
            i
            for i, row in enumerate(probes)
            if int(row.get("errors") or 0) > 0 or row.get("error")
        ]
        for i in dirty:
            path = Path(str(probes[i].get("file") or ""))
            side = str(probes[i].get("side") or "host")
            _progress(f"prepare probe fallback declarations_only {side} {path.name}")
            try:
                if side == "kernel":
                    res = probe_diagnostics(
                        str(path), bctx, side="kernel", dtype_variant="DT_FLOAT16"
                    )
                else:
                    res = probe_diagnostics(str(path), bctx, side="host")
                entry, _errs = _probe_entry(path, side, res, "declarations_only")
                probes[i] = entry
            except Exception as exc:  # noqa: BLE001
                probes[i] = {
                    "file": path.as_posix(),
                    "error": str(exc)[:200],
                    "side": side,
                }
        from ascendc_codemap_mcp.engine.include_heal import (
            HealReport as _HealReport,
            heal_missing_includes,
            missing_includes_from_probes,
            save_extras as _save_heal_extras,
        )

        extra_missing = missing_includes_from_probes(probes, [])
        if extra_missing:
            added = heal_missing_includes(
                bctx, extra_missing, round_no=99, source="fallback_probe"
            )
            if added:
                _progress(
                    "prepare include-heal fallback "
                    + ", ".join(f"{h.include} -> {h.include_dir}" for h in added[:4])
                )
                _save_heal_extras(bctx, _HealReport(enabled=True))
                for i, row in enumerate(probes):
                    if str(row.get("side") or "") != "kernel":
                        continue
                    if int(row.get("errors") or 0) <= 0 and not row.get("error"):
                        continue
                    path = Path(str(row.get("file") or ""))
                    if not path.is_file():
                        continue
                    try:
                        res = probe_diagnostics(
                            str(path), bctx, side="kernel", dtype_variant="DT_FLOAT16"
                        )
                        probes[i], _ = _probe_entry(path, "kernel", res, "declarations_only")
                    except Exception as exc:  # noqa: BLE001
                        probes[i] = {
                            "file": path.as_posix(),
                            "error": str(exc)[:200],
                            "side": "kernel",
                        }
        host_errors = 0
        kernel_errors = 0
        for row in probes:
            side = str(row.get("side") or "host")
            if row.get("error"):
                if side == "kernel":
                    kernel_errors = -1
                else:
                    host_errors += 1
                continue
            errs = int(row.get("errors") or 0)
            if side == "kernel":
                kernel_errors = errs
            else:
                host_errors += max(errs, 0)

    if not reused_includes:
        probe_total = len(expected)
        probe_i = 0
        for path, side in expected:
            probe_i += 1
            _progress(f"prepare probe {side} ({probe_i}/{probe_total}) {path.name}")
            try:
                if side == "kernel":
                    res = probe_diagnostics(
                        str(path), bctx, side="kernel", dtype_variant="DT_FLOAT16"
                    )
                else:
                    res = probe_diagnostics(str(path), bctx, side="host")
                entry, errs = _probe_entry(path, side, res, "declarations_only")
                probes.append(entry)
                if side == "host":
                    host_errors += max(errs, 0)
                else:
                    kernel_errors = errs
            except Exception as exc:  # noqa: BLE001
                probes.append(
                    {"file": path.as_posix(), "error": str(exc)[:200], "side": side}
                )
                if side == "host":
                    host_errors += 1
                else:
                    kernel_errors = -1

    probe_clean = host_errors == 0 and kernel_errors == 0
    if not probe_clean and host_errors == 0 and clang_status == "complete":
        from ascendc_codemap_mcp.engine.diag_scope import is_benign_kernel_probe_residual

        if is_benign_kernel_probe_residual(probes):
            # Kernel declarations-only residuals (unknown TCubeTiling /
            # SoftMaxTiling after a complete Clang include closure) must not
            # hard-block prepare. Arbitrary kernel errors and probe exceptions
            # stay fail-closed.
            probe_clean = True
            probes.append(
                {
                    "probe": "kernel_residuals_after_complete_clang_scope",
                    "kernel_errors": kernel_errors,
                }
            )
    if allow_unverified and not probe_clean:
        probes.append({"probe": "unverified_override", "reason": "UO_TEST_ALLOW_UNVERIFIED_SCOPE"})
        probe_clean = True
    if allow_unverified and clang_status != "complete":
        clang_status = "complete"
        clang_errors = []
        probes.append(
            {"clang_scope": "unverified_override", "reason": "UO_TEST_ALLOW_UNVERIFIED_SCOPE"}
        )

    scope_dict = scope.to_dict() if scope is not None else {}
    confirmed = list(scope_dict.get("confirmed_source_files") or [])
    candidate = {
        "version": 3,
        "status": "extracted",
        "op_name": spec.op_name,
        "arch_dir": spec.arch_dir,
        "available_archs": list(spec.available_archs),
        "host_targets": [p.as_posix() for p in hosts],
        "kernel_entry": kernel.as_posix() if kernel else "",
        "tiling_key_header": (
            spec.tiling_key_header.as_posix() if spec.tiling_key_header else ""
        ),
        "ambiguities": list(spec.ambiguities),
        "probes": probes,
        "host_probe_errors": host_errors,
        "kernel_probe_errors": kernel_errors,
        "probe_clean": probe_clean,
        "probe_skipped": False,
        "probe_reused_includes": reused_includes,
        "clang_scope_status": clang_status,
        "clang_scope_tus_expected": clang_tus_expected,
        "clang_scope_tus_parsed": clang_tus_parsed,
        "clang_scope_errors": clang_errors,
        "confirmed_source_files": confirmed,
        "scope_files": len(scope.files) if scope is not None else 0,
        "scope_shared": (
            sum(1 for f in scope.files if f.shared) if scope is not None else 0
        ),
        "scope_notes": list(scope.notes) if scope is not None else [],
        "arch_user_specified": bool(str(ctx.get("arch_dir") or "").strip()),
        "include_heal": include_heal_report,
        "build_context_extras": {
            "host": list(getattr(bctx, "extra_host_includes", None) or []),
            "kernel": list(getattr(bctx, "extra_kernel_includes", None) or []),
        },
    }
    out = run / "scope" / "candidates.yaml"
    _dump(out, candidate)
    _dump(uo / "summary" / "scope_candidates.yaml", candidate)
    if scope_dict:
        _dump(run / "scope" / "scope_set.yaml", scope_dict)
        _dump(uo / "summary" / "scope_set.yaml", scope_dict)
        # Run-level lease scope: same Clang set (posix, op-relative).
        try:
            from ascendc_codemap_mcp.engine.paths import product_dir as _product_dir

            run_id = str(ctx.get("run_id") or "").strip()
            if run_id:
                roots = sorted(
                    {
                        p.split("/", 1)[0]
                        for p in confirmed
                        if "/" in p and not p.startswith(".")
                    }
                )
                src_scope = {
                    "version": 1,
                    "run_id": run_id,
                    "allowed_source_roots": roots or ["op_host", "op_kernel"],
                    "allowed_source_files": confirmed,
                    "clang_scope_status": clang_status,
                }
                _dump(
                    _product_dir(root, spec.arch_dir) / "runs" / run_id / "source_scope.yaml",
                    src_scope,
                )
        except Exception:  # noqa: BLE001
            pass
    return {
        "ok": True,
        "engine": "scope_scan",
        "probe_clean": candidate["probe_clean"],
        "clang_scope_status": clang_status,
        "ambiguous": bool(spec.ambiguities),
        "candidates": out.as_posix(),
        "host_probe_errors": host_errors,
        "kernel_probe_errors": kernel_errors,
        "scope_files": candidate["scope_files"],
        "scope_shared": candidate["scope_shared"],
        "confirmed_source_files": len(confirmed),
        "include_heal": include_heal_report,
        "arch_dir": spec.arch_dir,
        "architecture": spec.arch_dir,
    }


# Soft discover notes that never justify a human file-list review once the user
# already fixed operator + architecture. Hard blockers still fail prepare.
_SOFT_AMBIGUITY_PREFIXES = (
    "host_targets_from_glob:",
    "op_name_from_filename:",
    "op_name_from_directory:",
    "display_name_hint:",
    "multiple_opdef:",
    "multiple_kernel_entry:",
    "multiple_tiling_key_header:",
    # Many families pack keys in host GetTilingKey() without the TPL DSL header.
    # extract already skips fold when the header is missing.
    "tiling_key_header_not_found:",
    "kernel_entry_kept_last_tu:",
    "host_targets_from_sibling_kernel_include:",
    "unified_implementation:",
)


def _hard_scope_blockers(
    ambiguities: list[str],
    *,
    arch_user_specified: bool,
    probe_clean: bool,
    clang_scope_status: str,
    hosts: list[Any],
    kernel_entry: str,
) -> list[str]:
    """Failures that stop prepare; never become a 'confirm these files?' prompt."""
    blockers: list[str] = []
    for item in ambiguities:
        text = str(item)
        if text.startswith(_SOFT_AMBIGUITY_PREFIXES):
            continue
        if text.startswith("multiple_arch_dirs:"):
            if arch_user_specified:
                continue
            blockers.append(text)
            continue
        if text.startswith(
            (
                "arch_not_present:",
                "opdef_not_found:",
                "host_targets_not_found:",
                "kernel_entry_not_found:",
                "override_missing_",
            )
        ):
            blockers.append(text)
            continue
        # Unknown ambiguity: soft when probe is clean, else hard.
        if not probe_clean:
            blockers.append(text)
    if not hosts:
        blockers.append("host_targets_empty")
    if not str(kernel_entry or "").strip():
        blockers.append("kernel_entry_empty")
    if not probe_clean:
        blockers.append("clang_probe_unclean")
    if clang_scope_status != "complete":
        blockers.append("SCOPE_CLANG_CLOSURE_INCOMPLETE")
    return blockers


def scope_validate(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Machine gate: write scope receipt or fail as blocker.

    Requires Clang authoritative include closure (``clang_scope_status=complete``)
    and a clean probe. Never asks a human to confirm a file list, and never
    accepts ``decision=yes`` / ``force_confirm`` as a compiler bypass.
    """
    ctx = _ctx(payload)
    root = Path(project_root).expanduser().resolve()
    uo = _uo_root(root, arch=_payload_arch(ctx))
    run = _run_dir(uo, ctx)
    scope = run / "scope"
    cand = _load(scope / "candidates.yaml") or _load(uo / "summary" / "scope_candidates.yaml") or {}
    _legacy_scope_receipt = "scope_" + "confirmed.yaml"
    validated_path = scope / "scope_validated.yaml"
    summary_validated = uo / "summary" / "scope_validated.yaml"
    legacy_run = scope / _legacy_scope_receipt
    legacy_summary = uo / "summary" / _legacy_scope_receipt
    if not validated_path.is_file() and legacy_run.is_file():
        return {
            "ok": False,
            "engine": "scope_validate",
            "reason_code": "STALE_RUN_LAYOUT",
            "error": "STALE_RUN_LAYOUT",
            "message": f"legacy {_legacy_scope_receipt} present; re-run uo-init",
        }
    if (
        not validated_path.is_file()
        and not summary_validated.is_file()
        and legacy_summary.is_file()
    ):
        return {
            "ok": False,
            "engine": "scope_validate",
            "reason_code": "STALE_RUN_LAYOUT",
            "error": "STALE_RUN_LAYOUT",
            "message": f"legacy {_legacy_scope_receipt} present; re-run uo-init",
        }
    prior = _load(validated_path) or _load(summary_validated)
    if isinstance(prior, dict) and str(prior.get("status") or "") == "confirmed":
        prior_action = str(prior.get("action_id") or "").strip()
        prior_run = str(prior.get("run_id") or "").strip()
        ctx_run = str(ctx.get("run_id") or "").strip()
        source = str(prior.get("source") or "").strip().lower()
        auto = prior.get("auto")
        machine_ok = source == "machine" or auto is True or str(auto).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # Reuse only canonical machine stamps for this run. Legacy prepare stamps
        # and mismatched run ids must be rewritten (ses_00bb sticky false-green).
        reusable = (
            prior_action == "scope_validated"
            and machine_ok
            and (not ctx_run or prior_run == ctx_run)
        )
        if reusable:
            return {
                "ok": True,
                "engine": "scope_validate",
                "auto": bool(prior.get("auto", True)),
                "already_validated": True,
                "receipt": prior,
            }
    probe_clean = bool(cand.get("probe_clean", False))
    clang_scope_status = str(cand.get("clang_scope_status") or "incomplete")
    arch_user_specified = bool(
        cand.get("arch_user_specified")
        or str(ctx.get("arch_dir") or ctx.get("architecture") or "").strip()
    )
    ambiguities = [str(x) for x in (cand.get("ambiguities") or [])]
    hosts = list(cand.get("host_targets") or [])
    kernel_entry = str(cand.get("kernel_entry") or "")
    blockers = _hard_scope_blockers(
        ambiguities,
        arch_user_specified=arch_user_specified,
        probe_clean=probe_clean,
        clang_scope_status=clang_scope_status,
        hosts=hosts,
        kernel_entry=kernel_entry,
    )
    if blockers:
        err = (
            "SCOPE_CLANG_CLOSURE_INCOMPLETE"
            if "SCOPE_CLANG_CLOSURE_INCOMPLETE" in blockers
            else "SCOPE_VALIDATE_BLOCKED"
        )
        probe_samples: list[str] = []
        dirty_probes: list[dict[str, Any]] = []
        for item in cand.get("probes") or []:
            if not isinstance(item, dict):
                continue
            errs = int(item.get("errors") or 0)
            if errs <= 0 and not item.get("error"):
                continue
            dirty = {
                "file": item.get("file"),
                "side": item.get("side"),
                "errors": errs,
                "fatal": item.get("fatal"),
                "samples": list(item.get("samples") or [])[:5],
            }
            if item.get("error"):
                dirty["error"] = item.get("error")
            dirty_probes.append(dirty)
            for sample in dirty["samples"]:
                if sample and sample not in probe_samples:
                    probe_samples.append(str(sample))
        detail_zh = "范围校验失败（Clang Source Scope 不完整或探针失败），已记为 blocker；不要求人工确认文件清单"
        if probe_samples:
            detail_zh = (
                f"{detail_zh}；探针样例: " + "; ".join(probe_samples[:3])
            )
        missing = [s for s in probe_samples if "file not found" in s.lower()]
        if missing:
            unresolved = [
                str(x.get("include") or x)
                for x in ((cand.get("include_heal") or {}).get("unresolved") or [])
            ]
            healed_n = len((cand.get("include_heal") or {}).get("healed") or [])
            if unresolved:
                detail_zh = (
                    f"{detail_zh}。include-heal 在当前 cann_root 下仍找不到: "
                    + ", ".join(unresolved[:4])
                    + "。进入 heal：propose_include_heal 对照真实 CANN/ops 树写 staging，"
                    "heal_promote 校验后写入 extras；不要手改 extras 或 spec/build_context.yaml"
                )
            elif healed_n:
                detail_zh = (
                    f"{detail_zh}。include-heal 已补 {healed_n} 条 -I 仍有缺头文件；"
                    "看 candidates.yaml 的 include_heal，不要用 UO_TEST_ALLOW_UNVERIFIED_SCOPE"
                )
            else:
                detail_zh = (
                    f"{detail_zh}。缺头文件由 prepare include-heal 自动补 -I；"
                    "若仍失败，到解包树确认头文件是否存在，不要当算子噪声，"
                    "也不要用 UO_TEST_ALLOW_UNVERIFIED_SCOPE 走产品路径"
                )
        heal_unresolved = [
            str(x.get("include") or x)
            for x in ((cand.get("include_heal") or {}).get("unresolved") or [])
        ]
        if heal_unresolved:
            err = "INCLUDE_HEAL_UNRESOLVED"
        return {
            "ok": False,
            "engine": "scope_validate",
            "blocker": True,
            "need_human": False,
            "blockers": blockers,
            "ambiguous": bool(ambiguities),
            "probe_clean": probe_clean,
            "clang_scope_status": clang_scope_status,
            "probe_samples": probe_samples,
            "dirty_probes": dirty_probes,
            "unresolved": heal_unresolved,
            "reason_code": err,
            "message_zh": detail_zh,
            "error": err,
        }
    scope_set = (
        _load(scope / "scope_set.yaml")
        or _load(uo / "summary" / "scope_set.yaml")
        or {}
    )
    confirmed_files = list(
        cand.get("confirmed_source_files")
        or scope_set.get("confirmed_source_files")
        or []
    )
    receipt = {
        "version": 3,
        "status": "confirmed",
        "validated": True,
        "source": "machine",
        "clang_scope_status": clang_scope_status,
        "clang_scope_tus_expected": cand.get("clang_scope_tus_expected"),
        "clang_scope_tus_parsed": cand.get("clang_scope_tus_parsed"),
        # Run-scoped identity is required by the Pilot output contract.
        "run_id": str(ctx.get("run_id") or ""),
        "workflow_id": str(ctx.get("workflow_id") or "uo-init"),
        # Gate identity for scope_receipt — always machine clang validation.
        # Do NOT stamp the parent Action id (`prepare`); that caused
        # SCOPE_RECEIPT_ACTION_MISMATCH after auto drain (ses_00bf).
        "action_id": "scope_validated",
        "op_name": cand.get("op_name") or root.name,
        "arch_dir": cand.get("arch_dir") or "",
        "host_targets": hosts,
        "kernel_entry": kernel_entry,
        "auto": True,
        "probe_clean": probe_clean,
        "scope_files": cand.get("scope_files"),
        "scope_shared": cand.get("scope_shared"),
        "confirmed_source_files": confirmed_files,
        "frozen_scope": {"confirmed_source_files": confirmed_files},
        "soft_ambiguities": ambiguities,
    }
    _dump(scope / "scope_validated.yaml", receipt)
    _dump(scope / "receipt.yaml", {"ok": True, "gate": "scope_receipt", **receipt})
    _dump(uo / "summary" / "scope_validated.yaml", receipt)
    return {"ok": True, "engine": "scope_validate", "auto": True, "receipt": receipt}


def _bundle_cache(uo: Path) -> Path:
    """Extract receipt is the only host-bundle sidecar (no duplicate meta file)."""
    return uo / "ir" / "host_extract_receipt.yaml"


def _dump_ir_pickle(path: Path, obj: Any) -> None:
    """Stream pickle to disk. ``dumps`` + ``write_bytes`` doubles FAG HostIR in RAM."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _load_ir_pickle(path: Path) -> Any:
    import pickle

    with path.open("rb") as fh:
        return pickle.load(fh)


def _persist_bundle_ir(uo: Path, bundle: dict[str, Any]) -> dict[str, str]:
    from ascendc_codemap_mcp.engine.timing import log as _tlog

    written: dict[str, str] = {}
    ir = uo / "ir"
    hir = bundle.get("host_ir")
    kir = bundle.get("kernel_ir")
    if hir is not None:
        try:
            _dump_ir_pickle(ir / "host_ir.pkl", hir)
            written["host_ir"] = "pkl"
        except Exception as exc:  # noqa: BLE001
            _tlog(f"persist.host_ir_pkl_failed  {type(exc).__name__}: {exc}")
    if kir is not None:
        try:
            _dump_ir_pickle(ir / "kernel_ir.pkl", kir)
            written["kernel_ir"] = "pkl"
        except Exception as exc:  # noqa: BLE001
            _tlog(f"persist.kernel_ir_pkl_failed  {type(exc).__name__}: {exc}")
    return written


def _restore_extracted_bundle(
    project_root: Path, uo: Path, ctx: dict[str, Any]
) -> dict[str, Any] | None:
    """Reload extract's HostIR/KernelIR. Missing host pickle means no reuse."""
    from ascendc_codemap_mcp.engine.kernel_ir import kernel_ir_from_dict
    from ascendc_codemap_mcp.engine.op_spec import discover
    from ascendc_codemap_mcp.engine.timing import log as _tlog

    host_pkl = uo / "ir" / "host_ir.pkl"
    if not host_pkl.is_file():
        return None
    try:
        hir = _load_ir_pickle(host_pkl)
    except Exception as exc:  # noqa: BLE001
        _tlog(f"restore.host_ir_pkl_failed  {type(exc).__name__}: {exc}")
        return None
    kir = None
    kir_pkl = uo / "ir" / "kernel_ir.pkl"
    if kir_pkl.is_file():
        try:
            kir = _load_ir_pickle(kir_pkl)
        except Exception as exc:  # noqa: BLE001
            _tlog(f"restore.kernel_ir_pkl_failed  {type(exc).__name__}: {exc}")
            kir = None
    if kir is None:
        persisted = _load(uo / "ir" / "kernel_ir.yaml")
        if isinstance(persisted, dict) and persisted.get("branches"):
            kir = kernel_ir_from_dict(persisted)
    arch = _payload_arch(ctx)
    bundle = {
        "spec": discover(project_root, arch_dir=arch),
        "host_ir": hir,
        "kernel_ir": kir,
        "restored_from": "host_ir.pkl",
    }
    _tlog("restore.extracted_bundle  host_ir.pkl")
    return bundle


def extract_host(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build host IR ∥ uninstantiated kernel IR (one product path)."""
    from ascendc_codemap_mcp.engine.extract_bundle import extract_host_bundle
    from ascendc_codemap_mcp.engine.extract_cache import (
        compute_extract_fingerprint,
        skip_reextract_for_unchanged_tus,
        store_extract_fingerprint,
    )
    from ascendc_codemap_mcp.engine.init_profile import default_kernel_max_variants, default_with_kernel
    from ascendc_codemap_mcp.engine.progress import emit as _progress
    from ascendc_codemap_mcp.engine.timing import log as _tlog

    _progress("extract_host: building host/kernel IR bundle …")

    ctx = _ctx(payload)
    root = Path(project_root).expanduser().resolve()
    arch = _payload_arch(ctx)
    uo = _uo_root(root, arch=arch)
    with_kernel = default_with_kernel(ctx)
    kernel_max_variants = default_kernel_max_variants(ctx)
    skip_plan = skip_reextract_for_unchanged_tus(root, uo_root=uo, arch=arch)
    restored = None
    if skip_plan.get("skip_reextract"):
        restored = _restore_extracted_bundle(root, uo, ctx)
        if restored is None:
            _tlog("extract_host.skip_reextract_miss  fingerprint matched but host_ir.pkl missing")
    if restored is not None:
        bundle = restored
        _tlog("extract_host.reused_pickle  skip_reextract=true")
    else:
        try:
            bundle = extract_host_bundle(
                op_dir=root,
                cann_root=_cann_root(ctx),
                ops_root=_ops_root(ctx, root),
                arch_dir=arch,
                with_kernel=with_kernel,
                kernel_max_variants=kernel_max_variants,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc()
            try:
                from ascendc_codemap_mcp.engine.runtime import end_session

                end_session(op_root=root, architecture=arch)
            except Exception:  # noqa: BLE001
                pass
            return {
                "ok": False,
                "engine": "extract_host",
                "error": str(exc)[:400],
                "traceback": tb[-1200:],
            }

    spec = bundle.get("spec")
    arch = str(getattr(spec, "arch_dir", "") or arch or "")
    uo = _uo_root(root, arch=arch)
    _put_bundle(root, ctx, bundle, spec=spec)
    persisted = _persist_bundle_ir(uo, bundle)

    fp_meta = compute_extract_fingerprint(root, uo_root=uo, arch=arch)
    store_extract_fingerprint(uo, fp_meta)
    kir = bundle.get("kernel_ir")
    kernel_branches = len(getattr(kir, "branches", []) or [])
    if kir is not None and hasattr(kir, "to_persist_dict"):
        _dump(uo / "ir" / "kernel_ir.yaml", kir.to_persist_dict())
    meta = {
        "version": 1,
        "status": "extracted",
        "op_name": getattr(spec, "op_name", ""),
        "architecture": arch,
        "with_kernel": bool(with_kernel),
        "kernel_max_variants": int(kernel_max_variants or 0),
        "kernel_branches": kernel_branches,
        "extract_fingerprint": fp_meta.get("extract_fingerprint"),
        "sources_unchanged_at_start": bool(skip_plan.get("skip_reextract")),
        "restored_from": bundle.get("restored_from") or "",
        "persisted_ir": persisted,
    }
    _dump(
        uo / "ir" / "host_extract_receipt.yaml",
        {"ok": True, "engine": "extract_host", **meta},
    )
    _put_bundle(root, ctx, bundle, spec=spec, extract_fingerprint=str(meta.get("extract_fingerprint") or ""))
    return {
        "ok": True,
        "engine": "extract_host",
        "kernel_branches": meta["kernel_branches"],
        "extract_fingerprint": meta["extract_fingerprint"],
        "sources_unchanged_at_start": meta["sources_unchanged_at_start"],
        "restored_from": meta["restored_from"],
    }


_STORE: dict[str, Any] = {}


def _bundle_key(
    project_root: Path,
    ctx: dict[str, Any],
    *,
    spec: Any = None,
    extract_fingerprint: str = "",
) -> tuple[str, str, str]:
    from ascendc_codemap_mcp.engine.runtime import bundle_identity

    op_name = str(getattr(spec, "op_name", "") or ctx.get("op_name") or "")
    arch = str(getattr(spec, "arch_dir", "") or _payload_arch(ctx) or "")
    del extract_fingerprint
    return bundle_identity(
        project_root,
        ctx,
        op_name=op_name,
        architecture=arch,
    )


def _current_extract_fingerprint(
    project_root: Path,
    ctx: dict[str, Any],
    *,
    spec: Any = None,
) -> str:
    ctx = ctx or {}
    fp = str(ctx.get("extract_fingerprint") or "")
    if fp:
        return fp
    try:
        from ascendc_codemap_mcp.engine.extract_cache import compute_extract_fingerprint, load_extract_fingerprint

        arch = str(getattr(spec, "arch_dir", "") or _payload_arch(ctx) or "")
        uo = _uo_root(Path(project_root), arch=arch)
        try:
            meta = compute_extract_fingerprint(Path(project_root), uo_root=uo, arch=arch)
            got = str(meta.get("extract_fingerprint") or "")
            if got:
                return got
        except Exception:  # noqa: BLE001
            pass
        stored = load_extract_fingerprint(uo)
        return str(stored.get("extract_fingerprint") or "")
    except Exception:  # noqa: BLE001
        return ""


def _put_bundle(
    project_root: Path,
    ctx: dict[str, Any],
    bundle: dict[str, Any],
    *,
    spec: Any = None,
    extract_fingerprint: str = "",
) -> None:
    _STORE["bundle"] = bundle
    _STORE["bundle_key"] = _bundle_key(project_root, ctx, spec=spec)
    _STORE["bundle_fp"] = str(extract_fingerprint or "")


def _ensure_bundle(project_root: Path, ctx: dict[str, Any]) -> dict[str, Any]:
    ctx = _ctx(ctx)
    want = _bundle_key(project_root, ctx)
    cached = _STORE.get("bundle")
    stored_fp = str(_STORE.get("bundle_fp") or "")
    current_fp = _current_extract_fingerprint(project_root, ctx)
    if cached is not None and _STORE.get("bundle_key") == want:
        if not stored_fp or not current_fp or stored_fp == current_fp:
            return cached
    from ascendc_codemap_mcp.engine.extract_bundle import extract_host_bundle
    from ascendc_codemap_mcp.engine.kernel_ir import kernel_ir_from_dict
    from ascendc_codemap_mcp.engine.init_profile import default_kernel_max_variants, default_with_kernel
    from ascendc_codemap_mcp.engine.timing import log as _tlog

    root = Path(project_root).expanduser().resolve()
    arch = _payload_arch(ctx)
    uo = _uo_root(root, arch=arch)
    restored = _restore_extracted_bundle(root, uo, ctx)
    if restored is not None:
        _put_bundle(root, ctx, restored, spec=restored.get("spec"))
        return restored

    cached_meta = _load(_bundle_cache(uo))
    persisted_kir = _load(uo / "ir" / "kernel_ir.yaml")
    has_persist = isinstance(persisted_kir, dict) and bool(persisted_kir.get("branches"))
    _tlog(
        "ensure_bundle.reextract  "
        f"reason={'no_host_ir_pkl' if not (uo / 'ir' / 'host_ir.pkl').is_file() else 'restore_failed'}"
    )

    with_kernel = False if (cached_meta and has_persist) else default_with_kernel(ctx)
    kernel_max_variants = default_kernel_max_variants(ctx)
    if "with_kernel" in ctx:
        with_kernel = bool(ctx.get("with_kernel"))
    if "kernel_max_variants" in ctx:
        try:
            kernel_max_variants = int(ctx.get("kernel_max_variants"))
        except (TypeError, ValueError):
            pass
    bundle = extract_host_bundle(
        op_dir=root,
        cann_root=_cann_root(ctx),
        ops_root=_ops_root(ctx, root),
        arch_dir=arch,
        with_kernel=with_kernel,
        kernel_max_variants=kernel_max_variants,
    )
    if not getattr(bundle.get("kernel_ir"), "branches", None) and has_persist:
        restored = kernel_ir_from_dict(persisted_kir)
        if restored is not None:
            bundle["kernel_ir"] = restored
    elif bundle.get("kernel_ir") is not None and hasattr(
        bundle["kernel_ir"], "to_persist_dict"
    ):
        _dump(uo / "ir" / "kernel_ir.yaml", bundle["kernel_ir"].to_persist_dict())
    _persist_bundle_ir(uo, bundle)
    _put_bundle(root, ctx, bundle, spec=bundle.get("spec"))
    return bundle


def resolve_gaps(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Removed from ``/uo-init``. Residual analysis is ``/uo-investigate``."""
    del project_root, payload
    return {
        "ok": False,
        "error": "RESOLVE_GAPS_REMOVED",
        "engine": "resolve_gaps",
        "message_zh": "uo-init 不再做 LLM 缺口补齐；未闭合项请用 /uo-investigate。",
    }


def export_tg_host_view(
    project_root: Path, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Project live HostIR into tg_host_view.yaml stamped with the graph fingerprint.

    Prefer the committed ``.uo`` product; working-tree ``ir/operator_graph.yaml``
    is accepted during extract before commit. Does not require sqlite.
    """
    from ascendc_codemap_mcp.engine.host_codemap import (
        TG_HOST_VIEW_YAML,
        export_tg_host_view as _export_view,
        rebuild_codemap_index,
    )

    ctx = _ctx(payload)
    root = Path(project_root).expanduser().resolve()
    uo = _uo_root(root, arch=_payload_arch(ctx))
    graph_path = uo / "ir" / "operator_graph.yaml"
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product

    product = find_uo_product(
        root, architecture=str(_payload_arch(ctx) or "")
    )
    if product is None and not graph_path.is_file():
        return {
            "ok": False,
            "engine": "export_tg_host_view",
            "error": "missing .uo product; run /uo-init commit first",
        }
    try:
        fingerprint = ""
        if graph_path.is_file():
            fingerprint = _load_yaml_scalar(graph_path, "fingerprint")
        if not fingerprint:
            from ascendc_codemap_mcp.engine.store.reader import find_uo_product, load_production_view, read_meta

            product = find_uo_product(
                root, architecture=str(_payload_arch(ctx) or "")
            )
            if product is not None:
                graph = load_production_view(product, "ir/operator_graph.yaml") or {}
                fingerprint = str(
                    graph.get("fingerprint")
                    or read_meta(product).get("graph_fingerprint")
                    or read_meta(product).get("cm_graph_fingerprint")
                    or ""
                )
        manifest = _load(uo / "manifest.yaml")
        if not isinstance(manifest, dict):
            manifest = {}
        manifest_hash = str(manifest.get("content_hash") or manifest.get("hash") or "")
        manifest_source = manifest.get("source")
        if not isinstance(manifest_source, dict):
            manifest_source = {}
        source_revision = str(
            manifest.get("source_revision")
            or manifest_source.get("revision")
            or ""
        )
        existing_view = _load(uo / TG_HOST_VIEW_YAML)
        if isinstance(existing_view, dict):
            source = existing_view.get("source")
            if not isinstance(source, dict):
                source = {}
            view_fp = str(source.get("graph_fingerprint") or "")
            view_manifest_hash = str(source.get("manifest_hash") or "")
            view_source_revision = str(source.get("source_revision") or "")
            same_manifest = (
                not manifest_hash
                or not view_manifest_hash
                or view_manifest_hash == manifest_hash
            )
            same_revision = (
                not source_revision
                or not view_source_revision
                or view_source_revision == source_revision
            )
            if (
                fingerprint
                and view_fp == fingerprint
                and same_manifest
                and same_revision
                and (existing_view.get("fields") or existing_view.get("predicates"))
            ):
                summary = rebuild_codemap_index(uo)
                receipt = {
                    "ok": bool(summary.get("ok", True)),
                    "engine": "export_tg_host_view",
                    "cached": True,
                    "graph_fingerprint": fingerprint,
                    "schema": existing_view.get("schema"),
                    "yaml": str(uo / TG_HOST_VIEW_YAML),
                    "alias_yaml": "",
                    "fields": len(existing_view.get("fields") or []),
                    "writers": sum(
                        len(f.get("writers") or [])
                        for f in existing_view.get("fields") or []
                        if isinstance(f, dict)
                    ),
                    "predicates": len(existing_view.get("predicates") or []),
                    **summary,
                }
                _dump(uo / "checks" / "tg_host_view_receipt.yaml", receipt)
                return receipt

        local_ctx = dict(ctx)
        local_ctx.setdefault("with_kernel", False)
        bundle = _ensure_bundle(root, local_ctx)
        host_ir = bundle.get("host_ir")
        if host_ir is None:
            return {
                "ok": False,
                "engine": "export_tg_host_view",
                "error": "bundle has no host_ir; re-run extract_host",
            }
        derive_fields: list[dict[str, Any]] | None = None
        kd = _load(uo / "tiling" / "key_derivations.yaml")
        if isinstance(kd, dict):
            derive_fields = list(kd.get("fields") or []) or None

        declared: dict[str, Any] | None = None
        try:
            from testcase_agent.closure import workspace as WS

            sch = WS.schema()
            declared = {
                "count": len(WS.declared()),
                "dims": [
                    {
                        "name": d.name,
                        "bw": getattr(d, "bw", 0),
                        "domain": list(getattr(d, "value_domain", []) or []),
                    }
                    for d in sch.dims
                ],
            }
        except Exception:
            declared = None

        result = _export_view(
            host_ir,
            uo,
            derive_fields=derive_fields,
            declared=declared,
            graph_fingerprint=fingerprint,
            source_revision=source_revision,
            manifest_hash=manifest_hash,
        )
        receipt = {
            "ok": bool(result.get("ok")),
            "engine": "export_tg_host_view",
            "graph_fingerprint": fingerprint,
            **{k: v for k, v in result.items() if k != "ok"},
        }
        _dump(uo / "checks" / "tg_host_view_receipt.yaml", receipt)
        return receipt
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "engine": "export_tg_host_view", "error": str(exc)[:400]}


def export_adapter_pack(
    project_root: Path, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Export TG adapter YAML from host_derivation into ``tg/adapter/``."""
    from ascendc_codemap_mcp.engine.adapter_pack import export_adapter_pack as _export

    ctx = _ctx(payload)
    root = Path(project_root).expanduser().resolve()
    arch = str(ctx.get("architecture") or ctx.get("arch") or "").strip() or None
    write_package = _flag(ctx.get("write_package"), default=False)
    sampling_grid = ctx.get("sampling_grid")
    if sampling_grid is not None and not isinstance(sampling_grid, dict):
        return {
            "ok": False,
            "engine": "export_adapter_pack",
            "error": "sampling_grid must be a mapping",
        }
    return _export(
        root,
        arch=arch,
        write_package=write_package,
        sampling_grid=sampling_grid,
    )


def export_integrity(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    del payload
    from ascendc_codemap_mcp.engine.host_codemap import (
        TG_HOST_VIEW_YAML,
        CODEMAP_YAML,
        load_tg_host_view,
        migrate_load_host_view_from_yaml,
    )
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product, load_production_view, read_meta

    uo = _uo_root(project_root)
    graph = uo / "ir" / "operator_graph.yaml"
    quality = uo / "checks" / "quality.yaml"
    quality_legacy = uo / "quality.yaml"
    unresolved = uo / "ir" / "unresolved.yaml"
    hashes = uo / "checks" / "artifact_hashes.yaml"
    view_path = uo / TG_HOST_VIEW_YAML
    alias_path = uo / CODEMAP_YAML
    errors: list[str] = []

    product = find_uo_product(Path(project_root).expanduser().resolve())
    if product is None:
        product = find_uo_product(uo)
    uo_ready = product is not None and product.suffix == ".uo" and product.is_file()
    if not uo_ready and not graph.is_file():
        errors.append("missing .uo product (and no working-tree ir/operator_graph.yaml)")

    q = _load(quality) or _load(quality_legacy)
    if not q and uo_ready:
        blob = load_production_view(product, "checks/quality.yaml") or load_production_view(
            product, "quality.yaml"
        )
        q = blob if isinstance(blob, dict) else {}
    if not q:
        errors.append("missing checks/quality.yaml")

    if not hashes.is_file() and uo_ready:
        blob = load_production_view(product, "checks/artifact_hashes.yaml")
        if blob is None:
            errors.append("missing checks/artifact_hashes.yaml")
    elif not hashes.is_file() and not uo_ready:
        errors.append("missing checks/artifact_hashes.yaml")

    graph_fp = ""
    if graph.is_file():
        graph_fp = _load_yaml_scalar(graph, "fingerprint")
    if not graph_fp and uo_ready:
        payload_graph = load_production_view(product, "ir/operator_graph.yaml") or {}
        meta = read_meta(product)
        graph_fp = str(
            payload_graph.get("fingerprint")
            or meta.get("graph_fingerprint")
            or meta.get("cm_graph_fingerprint")
            or ""
        )

    view = load_tg_host_view(uo)
    if not view:
        view = migrate_load_host_view_from_yaml(uo)
    if not view and uo_ready:
        blob = load_production_view(product, "ir/tg_host_view.yaml") or load_production_view(
            product, "tg_host_view"
        )
        view = blob if isinstance(blob, dict) else {}
    if not view_path.is_file() and not alias_path.is_file() and not view:
        errors.append(f"missing {TG_HOST_VIEW_YAML} (run export_tg_host_view)")
    elif view:
        view_source = view.get("source") if isinstance(view, dict) else {}
        if not isinstance(view_source, dict):
            view_source = {}
        view_fp = str(view_source.get("graph_fingerprint") or "")
        if not view_fp:
            errors.append("tg_host_view missing source.graph_fingerprint")
        elif graph_fp and view_fp != graph_fp:
            errors.append(
                f"tg_host_view fingerprint drift: view={view_fp!r} graph={graph_fp!r}"
            )

    ur = _load(unresolved)
    if not ur and uo_ready:
        blob = load_production_view(product, "ir/unresolved.yaml")
        ur = blob if isinstance(blob, dict) else {}
    blocker_count = int(ur.get("blocker_count") or len(ur.get("blockers") or []))
    doc = {
        "version": 1,
        "status": "pass" if not errors else "fail",
        "ok": not errors,
        "blocker_count": blocker_count,
        "source_closure": q.get("source_closure") if isinstance(q, dict) else None,
        "graph_fingerprint": graph_fp,
        "errors": errors,
    }
    _dump(uo / "checks" / "integrity.yaml", doc)
    return {"ok": not errors, "engine": "export_integrity", **doc}


def _codemap_engine(name: str):
    """Lazy import to avoid circular import with codemap_engines."""
    from ascendc_codemap_mcp.engine import codemap_engines as ce

    return getattr(ce, name)


# Stable names for ENGINE_REGISTRY adapters.
# Public CodeMap surface: prepare / extract / analyze / commit / verify,
# plus failure-only heal_promote (LLM writes staging; this engine promotes extras).
# Fine-grained names remain for internal chaining. LLM gap resolve is gone;
# residuals go to /uo-investigate.


def heal_promote(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate staged include dirs and append heal_promote extras."""
    import yaml

    from ascendc_codemap_mcp.engine.build_context import BuildContext
    from ascendc_codemap_mcp.engine.include_heal import promote_include_dirs
    from ascendc_codemap_mcp.engine.op_spec import discover

    ctx = _ctx(payload)
    root = Path(project_root).expanduser().resolve()
    blocked = _cann_env_block("heal_promote", ctx)
    if blocked is not None:
        return blocked
    try:
        spec = discover(root, arch_dir=ctx.get("arch_dir"))
        bctx = BuildContext.load(
            cann_root=_cann_root(ctx),
            ops_root=_ops_root(ctx, root),
            op_dir=str(spec.op_dir),
            arch_dir=spec.arch_dir,
            apply_saved_extras=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "engine": "heal_promote", "error": str(exc)[:400]}
    run_id = str(ctx.get("run_id") or "").strip()
    if not run_id:
        return {
            "ok": False,
            "engine": "heal_promote",
            "error": "INCLUDE_HEAL_STAGING_MISSING",
            "reason_code": "INCLUDE_HEAL_STAGING_MISSING",
            "message_zh": "缺少 run_id，无法读取 propose_include_heal/staging.yaml",
        }
    try:
        from ascendc_codemap_mcp.engine.paths import product_dir as _product_dir

        staging_path = (
            _product_dir(root, spec.arch_dir)
            / "runs"
            / run_id
            / "actions"
            / "propose_include_heal"
            / "staging.yaml"
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "engine": "heal_promote",
            "error": str(exc)[:400],
            "reason_code": "INCLUDE_HEAL_STAGING_MISSING",
        }
    if not staging_path.is_file():
        return {
            "ok": False,
            "engine": "heal_promote",
            "error": "INCLUDE_HEAL_STAGING_MISSING",
            "reason_code": "INCLUDE_HEAL_STAGING_MISSING",
            "staging_path": staging_path.as_posix(),
            "message_zh": "缺少 propose_include_heal/staging.yaml；LLM 只写 staging，不得直接改 extras",
        }
    try:
        staging = yaml.safe_load(staging_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {
            "ok": False,
            "engine": "heal_promote",
            "error": "INCLUDE_HEAL_STAGING_INVALID",
            "reason_code": "INCLUDE_HEAL_STAGING_INVALID",
            "message_zh": f"staging.yaml 无法解析: {exc}",
        }
    if not isinstance(staging, dict):
        return {
            "ok": False,
            "engine": "heal_promote",
            "error": "INCLUDE_HEAL_STAGING_INVALID",
            "reason_code": "INCLUDE_HEAL_STAGING_INVALID",
            "message_zh": "staging.yaml 必须是 mapping（host/kernel 列表）",
        }
    out = promote_include_dirs(bctx, staging, run_id=run_id)
    out.setdefault("engine", "heal_promote")
    out["staging_path"] = staging_path.as_posix()
    return out


ENGINES: dict[str, Any] = {
    "prepare": lambda project_root, payload=None: _codemap_engine("prepare")(
        project_root, payload
    ),
    "extract": lambda project_root, payload=None: _codemap_engine("extract")(
        project_root, payload
    ),
    "analyze": lambda project_root, payload=None: _codemap_engine("analyze")(
        project_root, payload
    ),
    "commit": lambda project_root, payload=None: _codemap_engine("commit")(
        project_root, payload
    ),
    "verify": lambda project_root, payload=None: _codemap_engine("verify")(
        project_root, payload
    ),
    "heal_promote": heal_promote,
    # Compatibility aliases (not in default /uo-init pipeline).
    "review": lambda project_root, payload=None: _codemap_engine("review")(
        project_root, payload
    ),
    # Internal / merge helpers (also used by composites).
    "prepare_layout": prepare_layout,
    "scope_scan": scope_scan,
    "scope_validate": scope_validate,
    "extract_host": extract_host,
    "compile": lambda project_root, payload=None: _codemap_engine("analyze")(
        project_root, payload
    ),
    "export_tg_host_view": export_tg_host_view,
    "export_adapter_pack": export_adapter_pack,
}

