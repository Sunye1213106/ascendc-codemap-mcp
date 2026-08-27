# -*- coding: utf-8 -*-
"""Plan which new-engine rebuild phases an update should run."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.update.artifacts import (
    compute_plan_fingerprint,
    current_scope_identity,
    resolve_uo_root,
)
from ascendc_codemap_mcp.engine.yaml_io import read_yaml, write_yaml

# Layers map onto uo_init.pilot_engines actions. Persistence is compile/commit
# of the arch-scoped ``.uo`` product — never sqlite export_kb / build_index.
ALL_LAYERS = ("host", "kernel", "compile", "commit")


def plan_kb_update(
    repo_root: Path,
    op_name: str,
    *,
    change_set: dict[str, Any] | None = None,
    write: bool = True,
    architecture: str = "",
) -> dict[str, Any]:
    del op_name
    repo_root = Path(repo_root).expanduser().resolve()
    uo_root = resolve_uo_root(repo_root, architecture=architecture)
    change_set = change_set or read_yaml(uo_root / "diff" / "change_set.yaml")
    if not change_set:
        raise FileNotFoundError("diff/change_set.yaml missing; run detect_kb_changes first")

    layers: set[str] = set()
    reasons: list[str] = []
    needs_scope = bool(change_set.get("needs_scope_review"))
    scoped_files = [
        f for f in (change_set.get("files") or []) if isinstance(f, dict) and f.get("in_scope")
    ]

    for item in scoped_files:
        role = str(item.get("role") or "other").lower()
        path = str(item.get("path") or "").replace("\\", "/").lower()
        mapped = _layers_for_role(role, path)
        if mapped:
            layers.update(mapped)
            reasons.append(f"{item.get('path')}: role={role} -> {sorted(mapped)}")

    if layers & {"host", "kernel"}:
        layers.add("compile")
        layers.add("commit")

    common_or_header = any(
        str(f.get("role") or "").lower() in {"common", "headers"}
        or "/common/" in str(f.get("path") or "").replace("\\", "/").lower()
        for f in scoped_files
    )
    if common_or_header:
        layers.update(ALL_LAYERS)
        reasons.append("common/headers change -> full uo_init rebuild")

    mode = "selective"
    if needs_scope and not scoped_files:
        mode = "blocked_scope"
    elif len(layers) >= len(ALL_LAYERS) - 1:
        mode = "full_extract"
        layers.update(ALL_LAYERS)
    if not layers and not needs_scope:
        mode = "noop"

    actions = _actions_for_layers(layers)
    scope_id = current_scope_identity(uo_root)
    scope_fingerprint = str(
        change_set.get("scope_fingerprint") or scope_id.get("scope_fingerprint") or ""
    )
    change_set_fingerprint = str(
        change_set.get("change_set_fingerprint") or change_set.get("fingerprint") or ""
    )
    plan_fingerprint = compute_plan_fingerprint(
        head_revision=str(change_set.get("head_revision") or ""),
        base_revision=str(change_set.get("base_revision") or ""),
        scope_fingerprint=scope_fingerprint,
        change_set_fingerprint=change_set_fingerprint,
        mode=mode,
        affected_layers=sorted(layers),
    )
    plan = {
        "version": 1,
        "op_name": change_set.get("op_name"),
        "base_revision": change_set.get("base_revision"),
        "head_revision": change_set.get("head_revision"),
        "scope_fingerprint": scope_fingerprint,
        "change_set_fingerprint": change_set_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "mode": mode,
        "affected_layers": sorted(layers),
        "actions": actions,
        "needs_scope_review": needs_scope,
        "needs_llm_resolve": False,
        "scoped_changed_files": [str(f.get("path")) for f in scoped_files],
        "reasons": reasons,
        "engine": "uo_init.update",
    }
    if write:
        out_dir = uo_root / "summary"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(out_dir / "update_plan.yaml", plan)
    return plan


def _layers_for_role(role: str, path: str) -> set[str]:
    if role in {"tilingkey"} or "template_tiling_key" in path:
        return {"host", "compile", "commit"}
    if role in {"kernel"}:
        return {"kernel", "compile", "commit"}
    if role in {"host", "tiling"}:
        return {"host", "compile", "commit"}
    if role in {"golden"}:
        return {"compile", "commit"}
    if role in {"api", "input_output", "proto"}:
        return {"host", "compile", "commit"}
    if role in {"common", "headers"}:
        return set(ALL_LAYERS)
    if role in {"other", ""}:
        return {"host", "kernel", "compile", "commit"}
    return set()


def _actions_for_layers(layers: set[str]) -> list[str]:
    ordered: list[str] = []
    if "host" in layers or "kernel" in layers:
        ordered.append("extract_host")
    if "compile" in layers or "commit" in layers or layers:
        ordered.extend(["compile", "commit"])
    # de-dupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for a in ordered:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out
