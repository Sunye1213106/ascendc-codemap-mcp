# -*- coding: utf-8 -*-
"""Detect in-scope source changes against the KB manifest revision."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.update.artifacts import (
    OPERATOR_PATH_MARKERS,
    SOURCE_SUFFIXES,
    compute_change_set_fingerprint,
    current_scope_identity,
    git_head,
    infer_role,
    inspect_git_changes,
    is_kb_artifact_path,
    load_scope_index,
    resolve_uo_root,
    revision_sha,
)
from ascendc_codemap_mcp.engine.yaml_io import read_yaml, write_yaml


def detect_kb_changes(
    repo_root: Path,
    op_name: str,
    *,
    base: str | None = None,
    head: str | None = None,
    write: bool = True,
    architecture: str = "",
) -> dict[str, Any]:
    del op_name  # op identity comes from project layout / manifest
    repo_root = Path(repo_root).expanduser().resolve()
    uo_root = resolve_uo_root(repo_root, architecture=architecture)
    if not (uo_root / "manifest.yaml").exists():
        raise FileNotFoundError(f"UO working tree missing at {uo_root}; run /uo-init first")

    manifest = read_yaml(uo_root / "manifest.yaml") or {}
    base_revision = _resolve_base_revision(
        manifest,
        uo_root=uo_root,
        repo_root=repo_root,
        explicit=base,
    )
    head_sha = revision_sha(head) or git_head(repo_root) or base_revision or "unknown"
    if not base_revision:
        base_revision = head_sha if head_sha != "unknown" else "unknown"

    skip_plan: dict[str, Any] | None = None
    try:
        from ascendc_codemap_mcp.engine.extract_cache import skip_reextract_for_unchanged_tus

        skip_plan = skip_reextract_for_unchanged_tus(
            repo_root, uo_root=uo_root, arch=architecture or None
        )
    except Exception:  # noqa: BLE001
        skip_plan = None

    inspected = inspect_git_changes(repo_root, base=base_revision, head=head_sha)
    name_status = list(inspected.get("rows") or [])
    warnings: list[str] = []
    detection = "git"
    if not inspected.get("git_ok"):
        detection = "content_fingerprint"
        warnings.append(
            "git unavailable or not a repository; falling back to confirmed-source content fingerprint"
        )
        name_status = _content_fallback_rows(repo_root, uo_root, skip_plan=skip_plan)

    scope_index = load_scope_index(uo_root)
    files: list[dict[str, Any]] = []
    needs_scope_review = False
    for status, path in name_status:
        norm = path.replace("\\", "/")
        if is_kb_artifact_path(norm):
            continue
        role = scope_index.get(norm, "")
        in_scope = norm in scope_index
        suspicious = (not in_scope) and _looks_like_operator_source(norm)
        if suspicious:
            needs_scope_review = True
        files.append(
            {
                "path": norm,
                "status": status,
                "in_scope": in_scope,
                "role": role or infer_role(norm),
                "suspicious_out_of_scope": suspicious,
            }
        )

    if skip_plan is not None:
        from ascendc_codemap_mcp.engine.extract_cache import align_scoped_changes

        files = align_scoped_changes(files, skip_plan)

    worktree_dirty = bool(inspected.get("worktree_dirty"))
    worktree_fingerprint = str(inspected.get("worktree_fingerprint") or "")
    if detection == "content_fingerprint":
        worktree_dirty = any(item.get("in_scope") for item in files)
        worktree_fingerprint = compute_change_set_fingerprint(
            head_revision=head_sha,
            base_revision=base_revision,
            scope_fingerprint="content",
            changed_files=files,
        )
    if worktree_dirty and head_sha != "unknown":
        head_revision = f"{head_sha}+dirty:{worktree_fingerprint[:12]}"
    else:
        head_revision = head_sha

    scope_id = current_scope_identity(uo_root)
    scope_fingerprint = str(scope_id.get("scope_fingerprint") or "")
    change_set_fingerprint = compute_change_set_fingerprint(
        head_revision=head_revision,
        base_revision=base_revision,
        scope_fingerprint=scope_fingerprint,
        changed_files=files,
    )
    payload = {
        "version": 1,
        "op_name": str(manifest.get("op_name") or repo_root.name),
        "base_revision": base_revision,
        "head_revision": head_revision,
        "head_sha": head_sha,
        "worktree_dirty": worktree_dirty,
        "worktree_fingerprint": worktree_fingerprint,
        "detection": detection,
        "git_ok": bool(inspected.get("git_ok")),
        "warnings": warnings,
        "scope_revision": scope_id.get("scope_revision"),
        "scope_fingerprint": scope_fingerprint,
        "confirmed_sources_hash": scope_id.get("confirmed_sources_hash"),
        "change_set_fingerprint": change_set_fingerprint,
        "fingerprint": change_set_fingerprint,
        "needs_scope_review": needs_scope_review,
        "scoped_change_count": sum(1 for item in files if item["in_scope"]),
        "files": files,
        "engine": "uo_init.update",
    }
    if write:
        out = uo_root / "diff" / "change_set.yaml"
        write_yaml(out, payload)
    return payload


def _resolve_base_revision(
    manifest: dict[str, Any],
    *,
    uo_root: Path,
    repo_root: Path,
    explicit: str | None,
) -> str:
    if explicit:
        sha = revision_sha(explicit)
        if sha and sha != "unknown":
            return sha
    source = manifest.get("source")
    if isinstance(source, dict):
        sha = revision_sha(source.get("revision") or "")
        if sha and sha != "unknown":
            return sha
    top = revision_sha(manifest.get("source_revision") or "")
    if top and top != "unknown":
        return top
    product = _product_source_revision(uo_root, repo_root, manifest)
    if product:
        return product
    return git_head(repo_root) or ""


def _product_source_revision(uo_root: Path, repo_root: Path, manifest: dict[str, Any]) -> str:
    try:
        from ascendc_codemap_mcp.engine.store.reader import find_uo_product, read_meta
    except Exception:  # noqa: BLE001
        return ""
    product = None
    try:
        product = find_uo_product(
            repo_root,
            op_name=str(manifest.get("op_name") or ""),
            architecture=str(manifest.get("architecture") or ""),
        )
    except Exception:  # noqa: BLE001
        product = None
    if product is None:
        hits = sorted(path for path in uo_root.glob("*.uo") if path.is_file())
        product = hits[0] if len(hits) == 1 else None
    if product is None:
        return ""
    try:
        meta = read_meta(product)
    except Exception:  # noqa: BLE001
        return ""
    sha = revision_sha(meta.get("source_revision") or "")
    return sha if sha and sha != "unknown" else ""


def _content_fallback_rows(
    repo_root: Path,
    uo_root: Path,
    *,
    skip_plan: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """When git cannot list diffs, treat confirmed-source content drift as edits."""
    plan = skip_plan
    if plan is None:
        try:
            from ascendc_codemap_mcp.engine.extract_cache import skip_reextract_for_unchanged_tus

            plan = skip_reextract_for_unchanged_tus(repo_root, uo_root=uo_root)
        except Exception:  # noqa: BLE001
            plan = None
    if plan is not None:
        return [("M", path) for path in (plan.get("changed_or_cold") or [])]
    scope_index = load_scope_index(uo_root)
    if not scope_index:
        return []
    return [("M", path) for path in sorted(scope_index)]


def _looks_like_operator_source(path: str) -> bool:
    lower = path.lower()
    if Path(path).suffix.lower() not in SOURCE_SUFFIXES:
        return False
    return any(marker in lower for marker in OPERATOR_PATH_MARKERS)
