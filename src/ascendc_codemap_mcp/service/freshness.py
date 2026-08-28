# -*- coding: utf-8 -*-
"""Freshness: is this CodeMap the same revision as the sources the agent sees?"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.store.schema import SCHEMA_COMPAT
from ascendc_codemap_mcp.engine.update.artifacts import inspect_git_changes, revision_sha
from ascendc_codemap_mcp.service.envelope import (
    FRESHNESS_BLOCKED,
    FRESHNESS_BUILDING,
    FRESHNESS_DIRTY,
    FRESHNESS_FRESH,
    FRESHNESS_INCOMPATIBLE,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        from ascendc_codemap_mcp.engine.store.writer import detect_source_revision

        source_revision = detect_source_revision(project) or ""
        inspected: dict[str, Any] = {}
        try:
            inspected = inspect_git_changes(
                project,
                base=indexed_revision,
                head=source_revision,
            )
        except Exception:  # noqa: BLE001
            inspected = {"git_ok": False}
        git_ok = bool(inspected.get("git_ok"))
        dirty = bool(inspected.get("worktree_dirty"))
        rows = list(inspected.get("rows") or [])
        changed_files = len(rows)
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
