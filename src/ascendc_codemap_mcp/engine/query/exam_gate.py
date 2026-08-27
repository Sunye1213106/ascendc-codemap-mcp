# -*- coding: utf-8 -*-
"""Score frozen uo-query exam recipes: gold hits vs noise vs latency."""
from __future__ import annotations

import json
import re
from typing import Any

_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{1,})\b")
_SKIP_NAMES = frozenset(
    {
        "ok", "shape", "count", "file", "line", "kind", "name", "id", "hint",
        "next", "cards", "snippet", "coverage", "truncated", "pattern",
        "true", "false", "none", "null", "int", "void", "bool", "auto",
    }
)
LATENCY_BAND = 0.10


def payload_blob(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str).lower()


def basenames(text: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"([A-Za-z0-9_+\-]+\.(?:cpp|h|hpp|cc|cu))", text, re.I):
        found.add(match.group(1).lower())
    return found


def names_from_payload(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: Any) -> None:
        name = str(raw or "").strip()
        leaf = name.split("::")[-1].split(".")[-1]
        if not leaf or leaf.lower() in _SKIP_NAMES or not _IDENT_RE.fullmatch(leaf):
            return
        key = leaf.lower()
        if key not in seen:
            seen.add(key)
            out.append(leaf)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("name", "canonical", "pipe", "symbol"):
                if node.get(key):
                    _add(node.get(key))
            for row in node.get("next") or []:
                _add(row)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return out


def score_question(spec: dict[str, Any], payloads: list[dict[str, Any]], ms: float) -> dict[str, Any]:
    blob = "\n".join(payload_blob(p) for p in payloads)
    must_names = [str(x) for x in (spec.get("must_names") or [])]
    must_files = [str(x) for x in (spec.get("must_files") or [])]
    must_needles = [str(x) for x in (spec.get("must_needles") or [])]
    allow = {str(x).lower() for x in (spec.get("related_allow") or [])}
    allow.update(n.lower() for n in must_names)

    name_hits = [n for n in must_names if n.lower() in blob]
    files_present = basenames(blob)

    def _file_hit(must: str) -> bool:
        needle = must.lower()
        return needle in files_present or any(needle in p for p in files_present)

    file_hits = [f for f in must_files if _file_hit(f)]
    needle_hits = [n for n in must_needles if n.lower() in blob]

    gold_hits = len(name_hits) + len(file_hits) + len(needle_hits)
    gold_total = len(must_names) + len(must_files) + len(must_needles)

    extracted: list[str] = []
    for payload in payloads:
        extracted.extend(names_from_payload(payload))
    noise_names = [n for n in extracted if n.lower() not in allow]
    extra_files = sorted(
        f
        for f in files_present
        if f not in allow
        and not any(must.lower() in f or f in must.lower() for must in must_files)
    )

    first_kind_miss = 0
    expect = spec.get("expect_first_kinds") if isinstance(spec.get("expect_first_kinds"), dict) else {}
    for payload, query in zip(payloads, spec.get("queries") or []):
        pattern = str((query or {}).get("pattern") or "")
        wanted = expect.get(pattern)
        if not wanted:
            continue
        cards = payload.get("cards") or payload.get("seeds") or payload.get("hits") or []
        first = cards[0] if cards and isinstance(cards[0], dict) else {}
        kind = str(first.get("kind") or "")
        if kind and kind not in wanted:
            first_kind_miss += 1

    tokens = max(1, len(blob.encode("utf-8")) // 4)
    noise = len(noise_names) + len(extra_files) + first_kind_miss
    return {
        "id": spec.get("id"),
        "gold_hits": gold_hits,
        "gold_total": gold_total,
        "name_hits": name_hits,
        "file_hits": file_hits,
        "needle_hits": needle_hits,
        "noise": noise,
        "noise_names": noise_names[:24],
        "extra_files": extra_files[:12],
        "first_kind_miss": first_kind_miss,
        "tokens": tokens,
        "ms": round(ms, 1),
        "ok": bool(payloads),
    }


def compare(gold: dict[str, Any], current: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    by_gold = {str(r.get("id")): r for r in (gold.get("questions") or [])}
    by_cur = {str(r.get("id")): r for r in (current.get("questions") or [])}
    for qid in sorted(set(by_gold) | set(by_cur)):
        if qid not in by_gold:
            diffs.append(f"{qid}: ADDED")
            continue
        if qid not in by_cur:
            diffs.append(f"{qid}: MISSING")
            continue
        g, c = by_gold[qid], by_cur[qid]
        if int(c.get("gold_hits") or 0) < int(g.get("gold_hits") or 0):
            diffs.append(
                f"{qid}: gold_hits {g.get('gold_hits')} -> {c.get('gold_hits')} (useful info dropped)"
            )
        if int(c.get("noise") or 0) > int(g.get("noise") or 0):
            extra = [n for n in (c.get("noise_names") or []) if n not in (g.get("noise_names") or [])]
            diffs.append(
                f"{qid}: noise {g.get('noise')} -> {c.get('noise')} extra={extra[:8]}"
            )
        g_ms = float(g.get("ms") or 0)
        c_ms = float(c.get("ms") or 0)
        if g_ms > 0 and c_ms > g_ms * (1.0 + LATENCY_BAND) and (c_ms - g_ms) > 8:
            diffs.append(f"{qid}: slower {g_ms}ms -> {c_ms}ms")
    return diffs
