# -*- coding: utf-8 -*-
"""Bounded directed slices over an in-memory or committed CodeMap."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.evidence import TRUST_ADVISORY, is_authoritative_trust
from ascendc_codemap_mcp.engine.store.reader import read_codemap


def _codemap(value: Any) -> CodeMap:
    if isinstance(value, CodeMap):
        return value
    nested = getattr(value, "codemap", None)
    if isinstance(nested, CodeMap):
        return nested
    path = Path(value).expanduser().resolve()
    if path.is_file() and path.suffix == ".uo":
        return read_codemap(path)
    from ascendc_codemap_mcp.engine.query.engine import open_codemap_query

    return open_codemap_query(path).codemap


def _tier(status: str, provenance: str, trust: str = "") -> str:
    """Return a conservative A/B/C evidence hint, never a proof upgrade.

    ``trust`` is authoritative; ``status=confirmed`` does not promote advisory facts.
    """
    if str(trust or "") == TRUST_ADVISORY:
        return "C"
    if is_authoritative_trust(trust):
        return "A" if str(trust) == "authoritative" else "B"
    state = str(status or "").lower()
    prov = str(provenance or "").lower()
    if state in {"unresolved", "partial", "unknown", "external"}:
        return "C"
    if any(mark in prov for mark in ("llm", "heuristic", "inferred", "lexical")):
        return "C"
    if any(mark in prov for mark in ("clang", "compiler", "source_", "source-", "ast")):
        return "B"
    return "B"


def _row(value: Any) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.query.evidence import project_entity, project_relation

    if hasattr(value, "src") and hasattr(value, "dst"):
        hit = project_relation(value)
    else:
        hit = project_entity(value, require_span_for_branch=False) or value.to_dict()
    hit["evidence_tier"] = _tier(
        str(getattr(value, "status", "") or ""),
        str(getattr(value, "attrs", {}).get("provenance") or ""),
        str(getattr(value, "attrs", {}).get("trust") or ""),
    )
    return hit


def _slice(
    codemap_or_product: Any,
    seed_ids: Iterable[str],
    *,
    edge_kinds: Iterable[str] | None,
    depth: int,
    budget: int,
    direction: str,
    include_advisory: bool = False,
) -> dict[str, Any]:
    codemap = _codemap(codemap_or_product)
    max_depth = max(0, int(depth))
    cap = max(1, int(budget))
    wanted = {str(kind.value if hasattr(kind, "value") else kind).upper() for kind in (edge_kinds or ())}
    if not wanted:
        from ascendc_codemap_mcp.engine.query.evidence import USEFUL_EDGE_KINDS

        wanted = set(USEFUL_EDGE_KINDS)

    outgoing: dict[str, list[Any]] = defaultdict(list)
    for rel in codemap.relations.values():
        if wanted and rel.kind_name().upper() not in wanted:
            continue
        if not include_advisory and str(rel.attrs.get("trust") or "") == TRUST_ADVISORY:
            continue
        key = rel.src if direction == "forward" else rel.dst
        outgoing[key].append(rel)
    for rows in outgoing.values():
        rows.sort(key=lambda rel: (rel.kind_name(), rel.src, rel.dst, rel.id))

    seeds = [str(seed) for seed in seed_ids if str(seed) in codemap.entities]
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    for seed in seeds:
        if seed not in seen and len(seen) < cap:
            seen.add(seed)
            queue.append((seed, 0))

    included_edges: dict[str, Any] = {}
    truncated = len(set(seeds)) > cap
    while queue:
        current, distance = queue.popleft()
        if distance >= max_depth:
            continue
        for rel in outgoing.get(current, ()):
            other = rel.dst if direction == "forward" else rel.src
            if other not in codemap.entities:
                continue
            if other not in seen:
                if len(seen) >= cap:
                    truncated = True
                    continue
                seen.add(other)
                queue.append((other, distance + 1))
            included_edges[rel.id] = rel

    nodes = [_row(codemap.entities[eid]) for eid in sorted(seen)]
    edges = [_row(included_edges[rid]) for rid in sorted(included_edges)]
    hints = Counter(str(row["evidence_tier"]) for row in nodes + edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "evidence_tier_hints": dict(sorted(hints.items())),
        "truncated": truncated,
    }


def slice_forward(
    codemap_or_product: Any,
    seed_ids: Iterable[str],
    *,
    edge_kinds: Iterable[str] | None = None,
    depth: int = 3,
    budget: int = 500,
    include_advisory: bool = False,
) -> dict[str, Any]:
    """Return a bounded outgoing subgraph rooted at ``seed_ids``.

    Advisory edges are excluded by default so lexical candidates cannot
    expand the semantic closure.
    """
    return _slice(
        codemap_or_product,
        seed_ids,
        edge_kinds=edge_kinds,
        depth=depth,
        budget=budget,
        direction="forward",
        include_advisory=include_advisory,
    )


def slice_backward(
    codemap_or_product: Any,
    seed_ids: Iterable[str],
    *,
    edge_kinds: Iterable[str] | None = None,
    depth: int = 3,
    budget: int = 500,
    include_advisory: bool = False,
) -> dict[str, Any]:
    """Return a bounded incoming subgraph rooted at ``seed_ids``."""
    return _slice(
        codemap_or_product,
        seed_ids,
        edge_kinds=edge_kinds,
        depth=depth,
        budget=budget,
        direction="backward",
        include_advisory=include_advisory,
    )
