# -*- coding: utf-8 -*-
"""CodeMap doctor / index / status / query tools."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import PRODUCT_DIR_NAME, SERVER_VERSION

_QUERY_CACHE: dict[str, Any] = {}


def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


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
    }


def index_operator(
    *,
    project: str,
    architecture: str,
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.codemap_engines import (
        analyze,
        commit,
        extract,
        prepare,
    )
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product

    root = Path(project).expanduser().resolve()
    arch = str(architecture or "").strip()
    if not str(project or "").strip():
        return {"ok": False, "error": "project is required"}
    if not arch:
        return {
            "ok": False,
            "error": "ARCHITECTURE_MISSING_IN_RUN_STATE: architecture is required",
        }
    if not root.is_dir():
        return {"ok": False, "error": f"operator directory not found: {root}"}
    ctx: dict[str, Any] = {"architecture": arch, "arch_dir": arch}
    t0 = time.perf_counter()
    steps: list[dict[str, Any]] = []
    for name, fn in (
        ("prepare", prepare),
        ("extract", extract),
        ("analyze", analyze),
        ("commit", commit),
    ):
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
                "engine": "index_operator",
                "failed_step": name,
                "error": (out or {}).get("error")
                or (out or {}).get("message_zh")
                or f"{name} failed",
                "steps": steps,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
    product = find_uo_product(root, architecture=arch)
    summary: dict[str, Any] = {}
    if product is not None:
        from ascendc_codemap_mcp.engine.store.reader import read_meta

        try:
            summary = read_meta(product)
        except Exception:  # noqa: BLE001
            summary = {}
    return {
        "ok": True,
        "engine": "index_operator",
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
    }


def _op_name_for(root: Path, architecture: str, product: Path) -> str:
    from ascendc_codemap_mcp.engine.store.reader import read_meta
    from ascendc_codemap_mcp.engine.yaml_io import read_yaml

    try:
        name = str(read_meta(product).get("op_name") or "").strip()
    except Exception:  # noqa: BLE001
        name = ""
    if name:
        return name
    yaml_name = str(
        (read_yaml(product.parent / "operator.yaml") or {}).get("op_name") or ""
    ).strip()
    if yaml_name:
        return yaml_name
    del architecture
    return root.name


def update_operator(
    *,
    project: str,
    architecture: str,
    confirm_scope: bool = False,
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product
    from ascendc_codemap_mcp.engine.update import update_operator as apply_update

    root = Path(project).expanduser().resolve()
    arch = str(architecture or "").strip()
    if not str(project or "").strip():
        return {"ok": False, "error": "project is required"}
    if not arch:
        return {
            "ok": False,
            "error": "ARCHITECTURE_MISSING_IN_RUN_STATE: architecture is required",
        }
    if not root.is_dir():
        return {"ok": False, "error": f"operator directory not found: {root}"}
    product = find_uo_product(root, architecture=arch)
    if product is None or product.suffix != ".uo":
        return {
            "ok": False,
            "engine": "update_operator",
            "error": (
                f"no .uo product under {root}; expected "
                f"{PRODUCT_DIR_NAME}/{arch}/<op>.{arch}.uo. "
                "Run index_operator first."
            ),
        }
    op_name = _op_name_for(root, arch, product)
    t0 = time.perf_counter()
    try:
        result = apply_update(
            root,
            op_name,
            architecture=arch,
            confirm_scope=bool(confirm_scope),
            reuse_artifacts=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "engine": "update_operator",
            "error": str(exc)[:800],
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    status = str((result or {}).get("status") or "")
    plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
    change_set = (
        result.get("change_set") if isinstance(result.get("change_set"), dict) else {}
    )
    _QUERY_CACHE.pop(str(product), None)
    return _jsonable(
        {
            "ok": status in {"pass", "blocked"},
            "engine": "update_operator",
            "status": status,
            "run_id": result.get("run_id"),
            "mode": plan.get("mode"),
            "affected_layers": plan.get("affected_layers") or [],
            "actions": plan.get("actions") or [],
            "in_scope_count": int(change_set.get("scoped_change_count") or 0),
            "needs_scope_review": bool(plan.get("needs_scope_review")),
            "path": str(product),
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "error": None
            if status != "fail"
            else str((result.get("receipt") or {}).get("message") or "update failed"),
        }
    )


def status(
    *,
    project: str,
    architecture: str,
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product, read_meta

    root = Path(project).expanduser().resolve() if str(project or "").strip() else None
    arch = str(architecture or "").strip()
    if root is None:
        return {"ok": False, "indexed": False, "error": "project is required"}
    if not arch:
        return {
            "ok": False,
            "indexed": False,
            "error": "ARCHITECTURE_MISSING_IN_RUN_STATE: architecture is required",
        }
    product = find_uo_product(root, architecture=arch)
    if product is None or not product.is_file():
        return {
            "ok": True,
            "indexed": False,
            "engine": "codemap_status",
            "project": str(root),
            "architecture": arch,
            "expected": f"{PRODUCT_DIR_NAME}/{arch}/<op>.{arch}.uo",
        }
    meta: dict[str, Any] = {}
    try:
        meta = read_meta(product)
    except Exception as exc:  # noqa: BLE001
        meta = {"read_error": str(exc)[:200]}
    try:
        mtime = int(product.stat().st_mtime)
    except OSError:
        mtime = 0
    return {
        "ok": True,
        "indexed": True,
        "engine": "codemap_status",
        "path": str(product),
        "mtime": mtime,
        "architecture": str(meta.get("architecture") or arch),
        "op_name": meta.get("op_name"),
        "semantic_completeness": meta.get("semantic_completeness"),
        "entity_count": meta.get("entity_count"),
        "relation_count": meta.get("relation_count"),
    }


def query_codemap(
    *,
    project: str = "",
    architecture: str = "",
    pattern: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product

    project = str(project or os.environ.get("ASCENDC_CODEMAP_PROJECT") or "").strip()
    architecture = str(
        architecture or os.environ.get("ASCENDC_CODEMAP_ARCHITECTURE") or ""
    ).strip()
    if not project:
        raise ValueError("project is required")
    if not architecture:
        raise ValueError(
            "ARCHITECTURE_MISSING_IN_RUN_STATE: architecture is required"
        )
    root = Path(project).expanduser()
    product = find_uo_product(root, architecture=architecture)
    if product is None or product.suffix != ".uo":
        raise FileNotFoundError(
            f"no .uo product under {root}; expected "
            f"{PRODUCT_DIR_NAME}/{architecture}/<op>.{architecture}.uo. "
            "Run index_operator first."
        )
    try:
        mtime = int(product.stat().st_mtime_ns)
    except OSError:
        mtime = 0
    cached = _QUERY_CACHE.get(str(product))
    q = None
    if isinstance(cached, tuple) and len(cached) == 2 and cached[1] == mtime:
        q = cached[0]
    if q is None:
        q = UoSqlQuery(product)
        _QUERY_CACHE[str(product)] = (q, mtime)
    payload = q.agent_query(
        pattern=str(pattern or ""),
        file=str(file or ""),
        line=int(line or 0),
        line_end=int(line_end or 0),
    )
    payload["engine"] = "query_codemap"
    return _jsonable(payload)
