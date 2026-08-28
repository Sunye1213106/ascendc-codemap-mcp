# -*- coding: utf-8 -*-
"""Phenomenon candidate retrieval: FTS + identifier / camelCase tokens.

Ranking only. A hit here is never COMPLETE. Lineage still comes from the graph.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.query.hints import identifier_tokens

_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def tokenize_phenomenon(text: str) -> list[str]:
    """Split NL / camelCase into graph-name needles. Order preserved, unique."""
    raw = str(text or "").strip()
    out: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        tok = str(token or "").strip()
        if len(tok) < 2:
            return
        key = tok.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(tok)

    for ident in identifier_tokens(raw):
        add(ident)
        if ident.isupper() or ident.islower():
            continue
        for part in _CAMEL_RE.findall(ident):
            if len(part) >= 3 and not part.isdigit():
                add(part)
    return out


def rank_name_hits(
    needles: Iterable[str],
    hits: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Prefer exact identifier matches, then prefix, then kind weight."""
    want = [str(n) for n in needles if str(n)]
    want_l = [n.lower() for n in want]

    def score(hit: dict[str, Any]) -> tuple:
        name = str(hit.get("name") or "")
        leaf = name.replace("::", ".").rsplit(".", 1)[-1]
        low = leaf.lower()
        exact = 0 if low in want_l else 1
        prefix = 0 if any(low.startswith(n) or n in low for n in want_l) else 1
        kind_rank = {
            "TILING_FIELD": 0,
            "TILING_KEY": 1,
            "TEMPLATE_ARG": 2,
            "PREDICATE": 3,
            "BRANCH": 4,
            "COMPILE_VAR": 5,
            "FUNCTION": 6,
        }.get(str(hit.get("kind") or ""), 9)
        return (exact, prefix, kind_rank, low)

    ranked = sorted(hits, key=score)
    uniq: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in ranked:
        eid = str(hit.get("id") or "")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        uniq.append(hit)
        if len(uniq) >= limit:
            break
    return uniq
