# -*- coding: utf-8 -*-
"""Read TG view blobs from an existing ``.uo``. TG never writes CodeMap products."""

from __future__ import annotations

from ascendc_codemap_mcp.engine.paths import require_architecture
import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.passes.tpl_schema import run as run_tpl_schema
from ascendc_codemap_mcp.engine.store.reader import find_uo_product, load_view_blob, read_codemap, read_meta
from ascendc_codemap_mcp.engine.store.writer import write_codemap
from ascendc_codemap_mcp.engine.tg_views import finalize_tg_views

REQUIRED_TG_VIEWS = (
    "tiling/exhaustive_key_space.yaml",
    "tiling/legal_key_index.jsonl",
    "ir/tg_host_view.yaml",
    "ir/operator_graph.yaml",
    "views/kernel.yaml",
    "views/tilingdata.yaml",
)


#: The subset a commit can always satisfy. Both are projected from CodeMap
#: entities alone, so a product that omits them omitted them by accident.
#: The TilingKey domain views are excluded on purpose: they need a discoverable
#: TPL header, and a tree without one is a legitimate partial build rather than
#: a broken product.
REQUIRED_COMMIT_VIEWS = (
    "views/kernel.yaml",
    "views/tilingdata.yaml",
)

_RERUN_UO_INIT = "rerun /uo-init; TG must not write .uo"


def require_tg_views(views: dict[str, Any] | None) -> list[str]:
    """Names in ``REQUIRED_TG_VIEWS`` that are absent from ``views``."""
    docs = views or {}
    return [name for name in REQUIRED_TG_VIEWS if name not in docs or docs.get(name) is None]


def require_commit_views(views: dict[str, Any] | None) -> list[str]:
    """Kernel / TilingData views a commit must carry.

    TG reads branch and field coverage domains from these two; a product without
    them looks to the solver like an operator with no branches and no fields,
    which reads as "nothing to cover" instead of "the product is incomplete".
    """
    docs = views or {}
    return [name for name in REQUIRED_COMMIT_VIEWS if name not in docs or docs.get(name) is None]


def load_tg_view(uo_path: str | Path, name: str) -> dict[str, Any] | list[Any] | None:
    """Load one view with fail-closed provenance. Never returns a stale blob."""
    from ascendc_codemap_mcp.engine.store.reader import load_view_blob_checked

    checked = load_view_blob_checked(uo_path, name)
    if not checked.get("ok"):
        return None
    view = checked.get("view")
    return view if view is not None else None


def list_view_names(uo_path: str | Path) -> list[str]:
    conn = sqlite3.connect(str(uo_path))
    try:
        return [str(r[0]) for r in conn.execute("SELECT name FROM view_blob ORDER BY name")]
    finally:
        conn.close()


def _existing_views(product: Path) -> dict[str, Any]:
    """Read every embedded view before ``write_codemap`` atomically replaces DB.

    Backfill (UO-side) preserves on-disk blobs, including ones TG would reject
    as stale; TG read paths use ``load_tg_view`` / ``load_view_blob_checked``.
    """
    out: dict[str, Any] = {}
    for name in list_view_names(product):
        if name == "summary":
            continue
        value = load_view_blob(product, name)
        if value is not None:
            out[name] = value
    return out


def legal_key_rows(uo_path: str | Path) -> list[dict[str, Any]]:
    blob = load_tg_view(uo_path, "tiling/legal_key_index.jsonl")
    if isinstance(blob, dict):
        rows = blob.get("rows")
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    return []


def legal_key_count(uo_path: str | Path) -> int:
    space = load_tg_view(uo_path, "tiling/exhaustive_key_space.yaml")
    if isinstance(space, dict):
        n = int(space.get("legal_key_count") or 0)
        if n > 0:
            return n
    return len(legal_key_rows(uo_path))


def _view_legal_key_count(views: dict[str, Any]) -> int:
    space = views.get("tiling/exhaustive_key_space.yaml")
    if isinstance(space, dict):
        n = int(space.get("legal_key_count") or 0)
        if n > 0:
            return n
    rows = views.get("tiling/legal_key_index.jsonl")
    if isinstance(rows, dict):
        rows = rows.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def backfill_from_source(
    project_root: str | Path,
    *,
    op_name: str = "",
    architecture: str = "",
    tiling_key_header: str | Path | None = None,
    uo_path: str | Path | None = None,
) -> dict[str, Any]:
    """Upgrade a CodeMap product in place while preserving existing view blobs.

    UO-side only. TG/CE must not call this; they read via ``ensure_tg_views``.
    """
    root = Path(project_root).expanduser().resolve()
    product = Path(uo_path).expanduser().resolve() if uo_path else find_uo_product(root, op_name=op_name, architecture=architecture)
    if product is None or not product.is_file() or product.suffix != ".uo":
        return {"ok": False, "error": "missing .uo CodeMap product"}

    meta = read_meta(product)
    op = op_name or str(meta.get("op_name") or "")
    arch = require_architecture(architecture or meta.get("architecture"))
    cm = read_codemap(product)
    cm.op_name = cm.op_name or op
    cm.architecture = cm.architecture or arch

    existing = _existing_views(product)
    ctx: dict[str, Any] = {
        "op_root": str(root),
        "architecture": arch,
        "op_name": op,
        "tg_views": dict(existing),
    }
    if tiling_key_header:
        ctx["tiling_key_header"] = str(Path(tiling_key_header).expanduser().resolve())

    before_d = _view_legal_key_count(existing)
    tpl_rerun = bool(tiling_key_header) or before_d <= 0
    if tpl_rerun:
        cm = run_tpl_schema(cm, context=ctx)

    views = finalize_tg_views(cm, existing=dict(ctx.get("tg_views") or existing))
    d_count = _view_legal_key_count(views)
    if d_count <= 0:
        return {
            "ok": False,
            "error": "TPL ARGS_SEL expansion produced empty D (header missing?)",
            "path": str(product),
            "header": ctx.get("tiling_key_header") or (cm.meta.get("tpl_schema") or {}).get("header"),
        }
    missing = require_tg_views(views)
    if missing:
        return {"ok": False, "error": "TG_VIEW_INCOMPLETE", "missing": missing, "path": str(product)}

    written = write_codemap(cm, product, views=views)
    return {
        "ok": True,
        "path": str(product),
        "sha256": hashlib.sha256(product.read_bytes()).hexdigest(),
        "legal_key_count": d_count,
        "args_sel_group_count": int(cm.meta.get("args_sel_group_count") or 0),
        "views": sorted(views),
        "graph_fingerprint": str(cm.meta.get("graph_fingerprint") or ""),
        "tpl_rerun": tpl_rerun,
        "preserved_view_count": len(existing),
        "uo": written,
    }


def ensure_tg_views(
    project_root: str | Path,
    *,
    op_name: str = "",
    architecture: str = "",
    tiling_key_header: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only TG readiness. Missing views fail closed; never ``write_codemap``.

    ``tiling_key_header`` is accepted for call-site compatibility and ignored:
    TG cannot backfill a CodeMap product.
    """
    del tiling_key_header
    root = Path(project_root).expanduser().resolve()
    arch = str(architecture or "").strip()
    if not arch:
        return {"ok": False, "error": "ARCHITECTURE_MISSING_IN_RUN_STATE", "backfilled": False}
    product = find_uo_product(root, op_name=op_name, architecture=arch)
    if product is None:
        return {
            "ok": False,
            "error": f"missing .uo CodeMap product; {_RERUN_UO_INIT}",
            "backfilled": False,
        }
    docs: dict[str, Any] = {}
    missing: list[str] = []
    for name in REQUIRED_TG_VIEWS:
        view = load_tg_view(product, name)
        if view is None:
            missing.append(name)
        else:
            docs[name] = view
    count = legal_key_count(product)
    if count > 0 and not missing:
        graph = docs["ir/operator_graph.yaml"] if isinstance(docs.get("ir/operator_graph.yaml"), dict) else {}
        return {
            "ok": True,
            "path": str(product),
            "legal_key_count": count,
            "backfilled": False,
            "graph_fingerprint": str(graph.get("fingerprint") or ""),
            "views": list(REQUIRED_TG_VIEWS),
        }
    reason = []
    if missing:
        reason.append(f"missing views {missing}")
    if count <= 0:
        reason.append("legal_key_count is 0")
    return {
        "ok": False,
        "error": f"TG views not ready ({'; '.join(reason) or 'incomplete'}); {_RERUN_UO_INIT}",
        "missing": missing,
        "path": str(product),
        "legal_key_count": count,
        "backfilled": False,
    }
