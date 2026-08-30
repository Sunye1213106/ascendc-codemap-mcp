# -*- coding: utf-8 -*-
"""Agent-facing DTO: name/kind/file/line/summary/source/facets/counts.

Internal ids, USR, attrs, and pass names stay off this schema. Query renders
from these cards; ``_INTERNAL_ID_RE`` is a CI assertion, not the product path.
"""
from __future__ import annotations

import re
from typing import Any

AGENT_FIELDS = ("name", "kind", "file", "line", "summary", "source", "facets", "counts")

_UNIT_START_RE = re.compile(
    r"\b(?:using|typedef|if|static\s+constexpr|constexpr|consteval)\b"
)
_INTERNAL_ID_RE = re.compile(
    r"(?:SRCPOL(?:COND)?|SRCMACRO|SRCFRONTIER|E_TYPE_|REL::|entity_id=)[^\s]*"
)


def to_agent_card(
    hit: dict[str, Any] | None,
    *,
    summary: str = "",
    source: str = "",
    facets: dict[str, Any] | None = None,
    counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = hit if isinstance(hit, dict) else {}
    card: dict[str, Any] = {
        "name": str(row.get("name") or ""),
        "kind": str(row.get("kind") or ""),
        "file": str(row.get("file") or ""),
        "line": int(row.get("line") or row.get("line_start") or 0),
    }
    if summary:
        card["summary"] = summary
    src = source or str(row.get("snippet") or "")
    if src:
        card["source"] = src
    if facets:
        card["facets"] = facets
    if counts:
        card["counts"] = counts
    return card


def clip_logical_unit(
    rows: list[tuple[int, str]],
    center: int,
    *,
    max_lines: int = 48,
) -> list[tuple[int, str]]:
    """Whole ``using`` / constexpr / assignment / if-condition, not a 3-line slice."""
    if not rows:
        return []
    indexed = {int(ln): txt for ln, txt in rows}
    numbers = sorted(indexed)
    if not numbers:
        return []
    if center not in indexed:
        nearest = min(numbers, key=lambda n: abs(n - center))
        center = nearest
    start_i = numbers.index(center)
    lo = start_i
    while lo > 0:
        prev = indexed[numbers[lo - 1]].rstrip()
        cur = indexed[numbers[lo]].lstrip()
        if prev.endswith(("\\", ",", "<", "(", "{")) or cur.startswith(("=", "<", "{")):
            lo -= 1
            continue
        if _UNIT_START_RE.search(indexed[numbers[lo - 1]]) and ";" not in indexed[numbers[lo - 1]]:
            lo -= 1
            continue
        break
    depth_paren = 0
    depth_brace = 0
    depth_angle = 0
    hi = start_i
    while hi < len(numbers):
        text = indexed[numbers[hi]]
        for ch in text:
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)
            elif ch == "<":
                depth_angle += 1
            elif ch == ">":
                depth_angle = max(0, depth_angle - 1)
        closed = ";" in text or (depth_brace == 0 and "}" in text)
        if closed and depth_paren == 0 and depth_brace == 0 and depth_angle == 0:
            break
        if hi - lo + 1 >= max_lines:
            break
        hi += 1
        if hi >= len(numbers):
            hi = len(numbers) - 1
            break
    chosen = numbers[lo : hi + 1]
    return [(n, indexed[n]) for n in chosen]


def assert_no_internal_ids(text: str) -> None:
    if _INTERNAL_ID_RE.search(str(text or "")):
        raise AssertionError("agent card leaked an internal identity")
