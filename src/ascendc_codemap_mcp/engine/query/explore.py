# -*- coding: utf-8 -*-
"""Unified explore card: seed resolve + contract + impact + completeness.

Seed resolution is syntactic (ident / Dim= / file:line / else phenomenon).
There is no NL intent classifier. Completeness is independent of how the seed
was found.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.query.closure import semantic_impact_closure_sql
from ascendc_codemap_mcp.engine.query.completeness import (
    AMBIGUOUS,
    COMPLETE,
    INCOMPLETE,
    UNKNOWN,
    fence_contract,
    pick_transport,
)
from ascendc_codemap_mcp.engine.query.hints import looks_like_nl_or_multi_token
from ascendc_codemap_mcp.engine.query.phenomenon import rank_name_hits, tokenize_phenomenon

_PROOF_LINES = 8
_PRODUCER_KINDS = {
    RelationKind.WRITES.value,
    RelationKind.DERIVES.value,
    RelationKind.ALLOCATES.value,
}
_CONSUMER_KINDS = {
    RelationKind.READS.value,
    RelationKind.CALLS_UNDER_GUARD.value,
    RelationKind.BINDS.value,
    RelationKind.MATERIALIZES_AS.value,
    RelationKind.SELECTS.value,
}


def _clip_source(op_root: Path | None, file: str, line: int, *, max_lines: int = _PROOF_LINES) -> str:
    if op_root is None or not file or int(line or 0) <= 0:
        return ""
    path = Path(file)
    if not path.is_file():
        path = Path(op_root) / str(file).replace("\\", "/")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    rows = text.splitlines()
    idx = int(line) - 1
    if idx < 0 or idx >= len(rows):
        return ""
    start = max(0, idx - 2)
    end = min(len(rows), start + max_lines)
    if end - start < max_lines:
        start = max(0, end - max_lines)
    out = []
    for i, raw in enumerate(rows[start:end], start + 1):
        out.append(f"{i}:{raw}")
    return "\n".join(out)


def _loc_row(ent: dict[str, Any], *, role: str, snippet: str = "") -> dict[str, Any]:
    line = int(ent.get("line_start") or ent.get("line") or 0)
    row = {
        "id": ent.get("id") or "",
        "name": ent.get("name") or "",
        "kind": ent.get("kind") or "",
        "file": ent.get("file") or "",
        "line": line,
        "role": role,
        "consumer_role": ent.get("consumer_role") or "",
    }
    if snippet:
        row["snippet"] = snippet
    return row


def build_contract_card(
    query: Any,
    seed: dict[str, Any],
    *,
    extra_seeds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One seed → producers / transport / consumers / sinks / entry / fence."""
    sid = str(seed.get("id") or "")
    kind = str(seed.get("kind") or "")
    op_root = getattr(query, "_op_root", None)
    arch = str(getattr(query, "_architecture", "") or "")
    producers: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    binds: list[dict[str, Any]] = []
    kernel_repr: list[dict[str, Any]] = []
    entry: list[dict[str, Any]] = []
    has_branch = False
    import json

    with query._connect() as conn:
        rel_rows = conn.execute(
            """
            SELECT kind, src, dst, data FROM relation
            WHERE src = ? OR dst = ?
            LIMIT 400
            """,
            (sid, sid),
        ).fetchall()
        for rel in rel_rows:
            rkind = str(rel["kind"] or "")
            src_id = str(rel["src"] or "")
            dst_id = str(rel["dst"] or "")
            try:
                rdata = json.loads(rel["data"] or "{}")
            except json.JSONDecodeError:
                rdata = {}
            if not isinstance(rdata, dict):
                rdata = {}
            other_id = dst_id if src_id == sid else src_id
            row = query._entity_row(conn, other_id)
            if row is None:
                continue
            try:
                edata = json.loads(row["data"] or "{}")
            except json.JSONDecodeError:
                edata = {}
            if not isinstance(edata, dict):
                edata = {}
            hit = {
                "id": str(row["id"]),
                "name": str(row["name"] or ""),
                "kind": str(row["kind"] or ""),
                "file": str(rdata.get("file") or row["file"] or ""),
                "line_start": int(rdata.get("line") or row["line_start"] or 0),
                "consumer_role": rdata.get("consumer_role") or "",
            }
            if str(row["kind"] or "") == EntityKind.BRANCH.value:
                has_branch = True
            if rkind in _PRODUCER_KINDS and dst_id == sid:
                snippet = _clip_source(op_root, hit["file"], hit["line_start"])
                producers.append(_loc_row(hit, role="producer", snippet=snippet))
            if rkind in _CONSUMER_KINDS and (dst_id == sid or src_id == sid):
                snippet = _clip_source(op_root, hit["file"], hit["line_start"])
                consumers.append(_loc_row(hit, role="consumer", snippet=snippet))
            if rkind == RelationKind.BINDS.value:
                binds.append(_loc_row(hit, role="bind"))
            if rkind == RelationKind.MATERIALIZES_AS.value:
                kernel_repr.append(
                    _loc_row(
                        hit,
                        role="kernel_repr",
                        snippet=_clip_source(op_root, hit["file"], hit["line_start"]),
                    )
                )
            if (
                rkind == RelationKind.CONTROLS.value
                and str(row["kind"] or "") == EntityKind.PREDICATE.value
                and str(edata.get("predicate_role") or "") == "entry_path"
            ):
                entry.append(
                    _loc_row(
                        hit,
                        role="entry",
                        snippet=_clip_source(op_root, hit["file"], hit["line_start"]),
                    )
                )
        closure = semantic_impact_closure_sql(
            conn, [sid], entity_row=query._entity_row
        )
    sinks = []
    for row in closure.get("sinks") or []:
        snippet = _clip_source(op_root, str(row.get("file") or ""), int(row.get("line") or 0))
        sinks.append(
            {
                **row,
                "snippet": snippet,
            }
        )
    transport = pick_transport(seed_kind=kind, has_branch_reader=has_branch, has_bind=bool(binds))
    seed_row = _loc_row(seed, role="seed", snippet=_clip_source(op_root, str(seed.get("file") or ""), int(seed.get("line_start") or seed.get("line") or 0)))
    fence = fence_contract(
        seeds=[seed_row] + list(extra_seeds or []),
        producers=producers,
        consumers=consumers,
        sinks=sinks,
        transport=transport,
        binds=binds,
        kernel_repr=kernel_repr or binds,
        architecture=arch,
        seed_arch=str(seed.get("architecture") or ""),
    )
    proofs = []
    for row in [seed_row, *producers[:4], *consumers[:4], *sinks[:4], *entry[:2]]:
        if row.get("snippet") and row.get("file"):
            proofs.append(
                {
                    "name": row.get("name"),
                    "file": row.get("file"),
                    "line": row.get("line"),
                    "snippet": row.get("snippet"),
                    "role": row.get("role") or row.get("via") or "",
                }
            )
    return {
        "seed": seed_row,
        "transport": transport,
        "producers": producers[:12],
        "consumers": consumers[:12],
        "impact_sinks": sinks[:12],
        "binds": binds[:8],
        "kernel_repr": kernel_repr[:8],
        "entry": entry[:8],
        "completeness": fence["completeness"],
        "unresolved_reason": fence.get("unresolved_reason") or "",
        "unresolved_reasons": fence.get("unresolved_reasons") or [],
        "checks": fence.get("checks") or {},
        "windows": fence.get("windows") or [],
        "proof": proofs[:12],
        "impact_nodes": len(closure.get("nodes") or []),
    }


def attach_explore_fields(query: Any, payload: dict[str, Any], *, pattern: str) -> dict[str, Any]:
    """Enrich a name/cover/around payload with a contract card when there is one seed."""
    cards = list(payload.get("cards") or [])
    if not cards and payload.get("seeds"):
        cards = list(payload.get("seeds") or [])
    if not cards:
        payload.setdefault("completeness", UNKNOWN)
        return payload
    primary = cards[0]
    if not isinstance(primary, dict) or not primary.get("id"):
        payload.setdefault("completeness", UNKNOWN)
        return payload
    extras = [c for c in cards[1:3] if isinstance(c, dict) and c.get("id")]
    try:
        extra_for_fence = extras if str(payload.get("match") or "") == "phenomenon" and extras else None
        card = build_contract_card(query, primary, extra_seeds=extra_for_fence)
    except Exception:  # noqa: BLE001
        payload.setdefault("completeness", INCOMPLETE)
        payload.setdefault("unresolved_reason", "CONTRACT_BUILD_FAILED")
        return payload
    payload["contract"] = card
    payload["impact_sinks"] = card.get("impact_sinks") or []
    payload["entry"] = card.get("entry") or []
    payload["proof"] = card.get("proof") or []
    if len(cards) > 1 and str(payload.get("shape") or "") == "name":
        payload["completeness"] = AMBIGUOUS
        payload["unresolved_reason"] = "MULTIPLE_SEEDS"
    else:
        payload["completeness"] = card.get("completeness") or INCOMPLETE
        payload["unresolved_reason"] = card.get("unresolved_reason") or ""
    payload["unresolved_reasons"] = card.get("unresolved_reasons") or []
    payload["checks"] = card.get("checks") or {}
    if payload.get("completeness") == COMPLETE:
        payload["ok"] = True
    payload.setdefault("explore_pattern", pattern)
    return payload


def phenomenon_payload(query: Any, text: str, *, limit: int = 8) -> dict[str, Any]:
    needles = tokenize_phenomenon(text)
    hits: list[dict[str, Any]] = []
    for needle in needles[:8]:
        found = list(query._exact_name_hits(needle, limit=max(int(limit) * 4, 16)) or [])
        if not found:
            found = list(query._prefix_name_hits(needle, limit=max(int(limit), 8)) or [])
        hits.extend(found)
    ranked = rank_name_hits(needles, hits, limit=3)
    payload: dict[str, Any] = {
        "ok": bool(ranked),
        "shape": "explore",
        "pattern": text,
        "match": "phenomenon",
        "needles": needles[:12],
        "cards": ranked,
        "count": len(ranked),
        "completeness": UNKNOWN if not ranked else AMBIGUOUS if len(ranked) > 1 else INCOMPLETE,
    }
    if not ranked:
        payload["unresolved_reason"] = "NO_SEED"
        payload["hint"] = (
            "No graph identifier matched this phenomenon. "
            "Needles tried: " + ", ".join(needles[:6])
        )
        return payload
    payload = attach_explore_fields(query, payload, pattern=text)
    # Retrieval must not claim COMPLETE even if one candidate's fence passes.
    if payload.get("completeness") == COMPLETE:
        payload["completeness"] = INCOMPLETE
        payload["unresolved_reason"] = payload.get("unresolved_reason") or "PHENOMENON_CANDIDATE"
        reasons = list(payload.get("unresolved_reasons") or [])
        if "PHENOMENON_CANDIDATE" not in reasons:
            reasons.append("PHENOMENON_CANDIDATE")
        payload["unresolved_reasons"] = reasons
    return payload


def should_use_phenomenon(pattern: str) -> bool:
    return looks_like_nl_or_multi_token(pattern)
