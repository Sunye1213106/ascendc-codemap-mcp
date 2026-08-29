# -*- coding: utf-8 -*-
"""Unified read/control envelope for the CodeMap service."""
from __future__ import annotations

import json
from typing import Any

FRESHNESS_FRESH = "fresh"
FRESHNESS_DIRTY = "dirty"
FRESHNESS_STALE = "stale"
FRESHNESS_BUILDING = "building"
FRESHNESS_BLOCKED = "blocked"
FRESHNESS_INCOMPATIBLE = "incompatible"
FRESHNESS_UNKNOWN = "unknown"

STATE_COMPLETED = "completed"
STATE_NEEDS_CONFIRMATION = "needs_confirmation"
STATE_FAILED = "failed"
STATE_BUILDING = "building"
STATE_IDLE = "idle"

VERDICT_ANSWERED = "ANSWERED"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_UNKNOWN = "UNKNOWN"


def jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def envelope(
    *,
    ok: bool,
    codemap: dict[str, Any] | None = None,
    verdict: str | None = None,
    layer: str | None = None,
    data: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None,
    next_cursor: str | None = None,
    state: str | None = None,
    updated: bool | None = None,
    error: str | None = None,
    error_code: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": bool(ok)}
    if state is not None:
        out["state"] = state
    if updated is not None:
        out["updated"] = bool(updated)
    if error_code:
        out["error_code"] = error_code
    if error:
        out["error"] = error
    if codemap is not None:
        out["codemap"] = codemap
    if verdict is not None:
        out["verdict"] = verdict
    if layer is not None:
        out["layer"] = layer
    if data is not None:
        out["data"] = data
    if evidence is not None:
        out["evidence"] = evidence
    if coverage is not None:
        out["coverage"] = coverage
    if next_cursor:
        out["next_cursor"] = next_cursor
    if extra:
        for key, value in extra.items():
            if key not in out:
                out[key] = value
    return jsonable(out)


def fail(
    error: str,
    *,
    error_code: str = "",
    ok: bool = False,
    state: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return envelope(
        ok=ok,
        error=error,
        error_code=error_code or None,
        state=state,
        **kwargs,
    )
