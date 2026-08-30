# -*- coding: utf-8 -*-
"""Measure real session metrics from an exported agent transcript.

native_escape is the count of non-CodeMap tool calls (grep/read/bash/…).
This is the number collect_query_metrics.py used to hard-code as 0.

Usage:
  python benchmarks/session_metrics.py path/to/session-ses_facb.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOOL_RE = re.compile(r"^\*\*Tool:\s*(.+?)\*\*\s*$")
USER_RE = re.compile(r"^## User\s*$")
ASSIST_RE = re.compile(r"^## Assistant .*?(\d+(?:\.\d+)?)s\)\s*$")
MCP_NEEDLE = "codemap"


def _json_block(lines: list[str], start: int) -> tuple[str, int]:
    k = start
    while k < len(lines) and not lines[k].startswith("```"):
        k += 1
    k += 1
    buf: list[str] = []
    while k < len(lines) and not lines[k].startswith("```"):
        buf.append(lines[k])
        k += 1
    return "\n".join(buf), k


def parse_session(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    events: list[dict[str, Any]] = []
    q = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        if USER_RE.match(ln):
            q += 1
            events.append({"t": "user", "q": q, "line": i + 1})
            i += 1
            continue
        m = ASSIST_RE.match(ln)
        if m:
            events.append({"t": "turn", "q": q, "line": i + 1, "sec": float(m.group(1))})
            i += 1
            continue
        m = TOOL_RE.match(ln)
        if m:
            name = m.group(1).strip()
            j = i + 1
            while j < min(i + 8, len(lines)) and "**Input:**" not in lines[j]:
                j += 1
            payload = ""
            k = j
            if j < len(lines) and "**Input:**" in lines[j]:
                payload, k = _json_block(lines, j + 1)
            out = ""
            k2 = k
            while k2 < len(lines) and "**Output:**" not in lines[k2]:
                if lines[k2].startswith("## "):
                    break
                k2 += 1
            if k2 < len(lines) and "**Output:**" in lines[k2]:
                out, _ = _json_block(lines, k2 + 1)
            events.append(
                {
                    "t": "tool",
                    "q": q,
                    "line": i + 1,
                    "name": name,
                    "input": payload,
                    "out": out,
                }
            )
        i += 1

    tools = [e for e in events if e["t"] == "tool"]
    by_name = Counter(str(t["name"]) for t in tools)
    per_q_mcp: dict[int, int] = defaultdict(int)
    per_q_native: dict[int, int] = defaultdict(int)
    per_q_invalid: dict[int, int] = defaultdict(int)
    per_q_empty: dict[int, int] = defaultdict(int)
    resolve_targets: dict[int, Counter[str]] = defaultdict(Counter)
    op_counter: Counter[str] = Counter()

    for t in tools:
        qi = int(t["q"])
        out = str(t["out"] or "")
        if MCP_NEEDLE in str(t["name"]).lower():
            per_q_mcp[qi] += 1
            try:
                data = json.loads(t["input"]) if str(t["input"]).strip() else {}
            except json.JSONDecodeError:
                data = {}
            op = str(data.get("operation") or str(t["name"]).rsplit("_", 1)[-1])
            op_counter[op] += 1
            if "INVALID_QUERY" in out:
                per_q_invalid[qi] += 1
            if re.search(r"\b0 matches\b|count\": 0|returned 0", out):
                per_q_empty[qi] += 1
            if op == "resolve":
                key = json.dumps(
                    {
                        k: data.get(k)
                        for k in ("symbol", "name", "pattern", "file", "line")
                        if data.get(k)
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                resolve_targets[qi][key] += 1
        else:
            per_q_native[qi] += 1

    questions = []
    tot_repeat = 0
    for qi in sorted(set(list(per_q_mcp) + list(per_q_native))):
        rep = sum(c - 1 for c in resolve_targets[qi].values() if c > 1)
        tot_repeat += rep
        questions.append(
            {
                "id": qi,
                "mcp": per_q_mcp[qi],
                "native": per_q_native[qi],
                "invalid": per_q_invalid[qi],
                "empty": per_q_empty[qi],
                "repeat_resolve": rep,
            }
        )
    native_escape = int(sum(per_q_native.values()))
    return {
        "file": path.name,
        "user_turns": q,
        "assistant_turns": sum(1 for e in events if e["t"] == "turn"),
        "assistant_seconds": sum(float(e.get("sec") or 0) for e in events if e["t"] == "turn"),
        "tool_calls": len(tools),
        "tool_counts": dict(by_name),
        "operations": dict(op_counter),
        "queries_per_question": {str(r["id"]): r["mcp"] + r["native"] for r in questions},
        "native_escape": native_escape,
        "native_escape_per_question": {str(r["id"]): r["native"] for r in questions},
        "invalid_query": int(sum(per_q_invalid.values())),
        "empty_search": int(sum(per_q_empty.values())),
        "repeat_resolve": tot_repeat,
        "questions": questions,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python benchmarks/session_metrics.py SESSION.md", file=sys.stderr)
        return 2
    blob = parse_session(Path(args[0]))
    print(json.dumps(blob, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
