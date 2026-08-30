# -*- coding: utf-8 -*-
"""Collect post-evidence query metrics against a FAG snapshot.

Emits JSON with queries_per_question / native_escape / repeat_resolve /
invalid_query / server_ms. Does not call codemap_evidence.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query

FAG = Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad")

# Single-function vs cross Host/Kernel review questions from the plan.
QUESTIONS: list[dict[str, object]] = [
    {
        "id": "Q_bn2",
        "kind": "cross",
        "steps": [
            {"operation": "search", "name": "isBn2MultiBlk"},
            {"operation": "resolve", "symbol": "isBn2MultiBlk"},
        ],
    },
    {
        "id": "Q_deter",
        "kind": "cross",
        "steps": [
            {"operation": "resolve", "symbol": "deterMaxRound"},
        ],
    },
    {
        "id": "Q_select",
        "kind": "single_fn",
        "steps": [
            {"operation": "resolve", "symbol": "SelectDeterBandSchedule"},
        ],
    },
    {
        "id": "Q_sync",
        "kind": "single_fn",
        "steps": [
            {"operation": "resolve", "symbol": "SyncALLCores"},
        ],
    },
    {
        "id": "Q_vec",
        "kind": "single_fn",
        "steps": [
            {"operation": "resolve", "symbol": "vecInPing"},
        ],
    },
    {
        "id": "Q_ws",
        "kind": "cross",
        "steps": [
            {"operation": "resolve", "symbol": "workspaceSize"},
        ],
    },
]


def _run_step(step: dict[str, object]) -> dict[str, object]:
    payload = query(project=str(FAG), architecture="arch35", **step)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    text = str(data.get("text") or "")
    return {
        "ok": bool(payload.get("ok")),
        "error_code": str(payload.get("error_code") or ""),
        "server_ms": float(payload.get("server_ms") or 0.0),
        "render_ms": float(payload.get("render_ms") or 0.0),
        "response_chars": int(payload.get("response_chars") or 0),
        "text_len": len(text),
        "operation": str(step.get("operation") or ""),
        "symbol": str(step.get("symbol") or ""),
        "name": str(step.get("name") or ""),
    }


def collect() -> dict[str, object]:
    status(project=str(FAG), architecture="arch35")
    per_q: list[dict[str, object]] = []
    resolve_keys: list[str] = []
    invalid = 0
    server_ms: list[float] = []
    for spec in QUESTIONS:
        rows = []
        for step in spec["steps"]:  # type: ignore[union-attr]
            row = _run_step(step)  # type: ignore[arg-type]
            rows.append(row)
            server_ms.append(float(row["server_ms"]))
            if row["error_code"] == "INVALID_QUERY" or not row["ok"]:
                invalid += 1
            if str(row["operation"]) == "resolve" and row.get("symbol"):
                resolve_keys.append(str(row["symbol"]))
        per_q.append(
            {
                "id": spec["id"],
                "kind": spec["kind"],
                "queries": len(rows),
                "steps": rows,
            }
        )
    repeats = Counter(resolve_keys)
    repeat_resolve = sum(1 for n in repeats.values() if n > 1)
    return {
        "queries_per_question": {str(r["id"]): int(r["queries"]) for r in per_q},
        "native_escape": None,
        "native_escape_note": (
            "scripted queries cannot observe grep/read; measure native_escape "
            "from an exported session with benchmarks/session_metrics.py"
        ),
        "repeat_resolve": repeat_resolve,
        "invalid_query": invalid,
        "server_ms": {
            "samples": server_ms,
            "max": max(server_ms) if server_ms else 0.0,
            "mean": (sum(server_ms) / len(server_ms)) if server_ms else 0.0,
        },
        "questions": per_q,
    }


def main() -> int:
    blob = collect()
    print(json.dumps(blob, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
