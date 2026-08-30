# -*- coding: utf-8 -*-
"""Freshness: is this CodeMap the same revision as the sources the agent sees?"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.store.schema import SCHEMA_COMPAT
from ascendc_codemap_mcp.engine.update.artifacts import (
    git_operator_scope,
    is_kb_artifact_path,
    revision_sha,
    run_git,
)
from ascendc_codemap_mcp.service.envelope import (
    FRESHNESS_BLOCKED,
    FRESHNESS_BUILDING,
    FRESHNESS_DIRTY,
    FRESHNESS_FRESH,
    FRESHNESS_INCOMPATIBLE,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
)

# Query/status hit this on every MCP call. Full inspect_git_changes hashes dirty
# files and spawns several git processes — too heavy for a 123MB FAG tree.
_PROBE_TTL_S = 2.0
_probe_lock_cache: dict[str, tuple[float, tuple[int, int], dict[str, Any], Path]] = {}
_scope_cache: dict[str, tuple[Path, list[str], str]] = {}


def reset_probe_cache() -> None:
    _probe_lock_cache.clear()
    _scope_cache.clear()


def _cached_scope(op: Path) -> tuple[Path, list[str], str]:
    key = str(op)
    hit = _scope_cache.get(key)
    if hit is not None:
        return hit
    scope = git_operator_scope(op)
    _scope_cache[key] = scope
    return scope


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_mtime_token(git_cwd: Path) -> tuple[int, int]:
    git_dir = git_cwd / ".git"
    head = git_dir / "HEAD"
    index = git_dir / "index"
    def _ns(path: Path) -> int:
        try:
            return int(path.stat().st_mtime_ns)
        except OSError:
            return 0
    return _ns(head), _ns(index)


def _parse_porcelain_v2(stdout: str, prefix: str) -> tuple[str, int]:
    head = ""
    changed = 0
    for raw in (stdout or "").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("# branch.oid "):
            oid = line.split(" ", 2)[-1].strip()
            if oid and oid != "(initial)":
                head = oid
            continue
        path = ""
        if line.startswith("? "):
            path = line[2:].strip()
        elif line.startswith(("1 ", "u ")):
            parts = line.split(" ", 8)
            path = parts[-1].strip() if parts else ""
        elif line.startswith("2 "):
            parts = line.split("\t")
            path = parts[-1].strip() if parts else ""
        if not path:
            continue
        rel = path.replace("\\", "/")
        if prefix and rel.startswith(prefix):
            rel = rel[len(prefix) :]
        if is_kb_artifact_path(rel):
            continue
        changed += 1
    return head, changed


def _probe_operator_git_uncached(project: Path) -> dict[str, Any]:
    """One `git status` for HEAD + dirty count. No worktree file hashing."""
    op = Path(project).expanduser().resolve()
    git_cwd, pathspec, prefix = _cached_scope(op)
    args = ["status", "--porcelain=v2", "--branch", "--untracked-files=normal", *pathspec]
    proc = run_git(git_cwd, args)
    if proc is None:
        return {"git_ok": False, "head": "", "dirty": False, "changed_files": 0}
    err = f"{proc.stderr or ''} {proc.stdout or ''}".lower()
    if proc.returncode not in (0, 1) or "unknown option" in err or "unknown switch" in err:
        from ascendc_codemap_mcp.engine.update.artifacts import git_head

        head = git_head(git_cwd)
        proc_v1 = run_git(
            git_cwd,
            ["status", "--porcelain", "--untracked-files=normal", *pathspec],
        )
        if proc_v1 is None or proc_v1.returncode not in (0, 1):
            return {"git_ok": False, "head": head, "dirty": False, "changed_files": 0}
        changed = 0
        for line in (proc_v1.stdout or "").splitlines():
            path = line[3:].strip().replace("\\", "/") if len(line) > 3 else ""
            if prefix and path.startswith(prefix):
                path = path[len(prefix) :]
            if path and not is_kb_artifact_path(path):
                changed += 1
        return {"git_ok": bool(head), "head": head, "dirty": changed > 0, "changed_files": changed}
    head, changed = _parse_porcelain_v2(proc.stdout or "", prefix)
    if not head:
        from ascendc_codemap_mcp.engine.update.artifacts import git_head

        head = git_head(git_cwd)
    return {
        "git_ok": True,
        "head": head,
        "dirty": changed > 0,
        "changed_files": changed,
    }


def probe_operator_git(project: Path) -> dict[str, Any]:
    op = Path(project).expanduser().resolve()
    key = str(op)
    now = time.monotonic()
    hit = _probe_lock_cache.get(key)
    if hit is not None:
        ts, prev_token, payload, git_cwd = hit
        token = _git_mtime_token(git_cwd)
        if prev_token == token and (now - ts) < _PROBE_TTL_S:
            return dict(payload)
    git_cwd, _, _ = _cached_scope(op)
    token = _git_mtime_token(git_cwd)
    payload = _probe_operator_git_uncached(op)
    _probe_lock_cache[key] = (now, token, dict(payload), git_cwd)
    return dict(payload)


def compute(
    project: Path,
    *,
    meta: dict[str, Any] | None = None,
    building: bool = False,
    blocked: bool = False,
) -> dict[str, Any]:
    meta = dict(meta or {})
    indexed_revision = revision_sha(meta.get("source_revision") or "")
    schema = str(meta.get("schema") or "")
    completeness = _as_float(meta.get("semantic_completeness"))

    if building:
        freshness = FRESHNESS_BUILDING
        source_revision = ""
        dirty = False
        changed_files = 0
    elif schema and schema not in SCHEMA_COMPAT:
        freshness = FRESHNESS_INCOMPATIBLE
        source_revision = ""
        dirty = False
        changed_files = 0
    elif blocked:
        freshness = FRESHNESS_BLOCKED
        source_revision = ""
        dirty = False
        changed_files = 0
    else:
        probed = probe_operator_git(project)
        git_ok = bool(probed.get("git_ok"))
        source_revision = str(probed.get("head") or "")
        dirty = bool(probed.get("dirty"))
        changed_files = int(probed.get("changed_files") or 0)
        if not git_ok or not indexed_revision or not source_revision:
            freshness = FRESHNESS_UNKNOWN
        elif indexed_revision != revision_sha(source_revision):
            freshness = FRESHNESS_STALE
        elif dirty:
            freshness = FRESHNESS_DIRTY
        else:
            freshness = FRESHNESS_FRESH

    return {
        "freshness": freshness,
        "source_revision": source_revision if not building else (indexed_revision or ""),
        "indexed_revision": indexed_revision,
        "dirty": bool(freshness == FRESHNESS_DIRTY),
        "changed_files": int(changed_files),
        "schema": schema,
        "semantic_completeness": completeness,
    }
