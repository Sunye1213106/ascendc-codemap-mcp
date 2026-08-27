# -*- coding: utf-8 -*-
"""Export uo/diff product from change_set + update_plan over the new KB graph."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.update.artifacts import resolve_uo_root
from ascendc_codemap_mcp.engine.yaml_io import read_yaml, write_yaml


def export_diff_product(
    repo_root: Path,
    op_name: str,
    *,
    change_set: dict[str, Any] | None = None,
    update_plan: dict[str, Any] | None = None,
    status: str | None = None,
    write: bool = True,
    architecture: str = "",
) -> dict[str, Any]:
    del op_name
    repo_root = Path(repo_root).expanduser().resolve()
    uo_root = resolve_uo_root(repo_root, architecture=architecture)
    change_set = change_set or read_yaml(uo_root / "diff" / "change_set.yaml")
    update_plan = update_plan or read_yaml(uo_root / "summary" / "update_plan.yaml")
    if not change_set:
        raise FileNotFoundError("diff/change_set.yaml missing")
    if not update_plan:
        raise FileNotFoundError("summary/update_plan.yaml missing")

    graph = read_yaml(uo_root / "ir" / "operator_graph.yaml")
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]

    changed_paths = {
        str(item.get("path") or "").replace("\\", "/")
        for item in (change_set.get("files") or [])
        if isinstance(item, dict) and item.get("in_scope")
    }

    matched_ids: list[str] = []
    for node in nodes:
        fpath = str(node.get("file_path") or node.get("file") or "").replace("\\", "/")
        if fpath and any(fpath.endswith(p) or p in fpath for p in changed_paths):
            nid = str(node.get("id") or "")
            if nid:
                matched_ids.append(nid)

    st = status or ("blocked" if update_plan.get("mode") == "blocked_scope" else "ready")
    impact = {
        "version": 1,
        "status": st,
        "affected_layers": list(update_plan.get("affected_layers") or []),
        "matched_node_ids": sorted(dict.fromkeys(matched_ids)),
        "scoped_changed_files": list(update_plan.get("scoped_changed_files") or []),
        "engine": "uo_init.update",
    }
    unresolved = {
        "version": 1,
        "items": [],
        "needs_scope_review": bool(update_plan.get("needs_scope_review")),
    }
    index = {
        "version": 1,
        "status": st,
        "change_set": "diff/change_set.yaml",
        "update_plan": "summary/update_plan.yaml",
        "impact": "diff/impact.yaml",
        "unresolved": "diff/unresolved.yaml",
        "engine": "uo_init.update",
    }
    product = {"index": index, "impact": impact, "unresolved": unresolved, "status": st}
    if write:
        diff = uo_root / "diff"
        diff.mkdir(parents=True, exist_ok=True)
        write_yaml(diff / "index.yaml", index)
        write_yaml(diff / "impact.yaml", impact)
        write_yaml(diff / "unresolved.yaml", unresolved)
        # keep change_set in place
        if not (diff / "change_set.yaml").is_file():
            write_yaml(diff / "change_set.yaml", change_set)
    return product
