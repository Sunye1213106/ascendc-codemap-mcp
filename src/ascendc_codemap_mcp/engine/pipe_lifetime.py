# -*- coding: utf-8 -*-
"""TPipe launch order from Destroy → next construct in one continued body.

Physical line order of ``TPipe`` decls is not execution order when later
phases live in a ``#define`` body above the constructing function.  Destroy
and the next ``TPipe`` name in the same backslash-continued range *are*
adjacent in that body; pipes with no such edge keep declaration order.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def continued_line_ranges(text: str) -> list[tuple[int, int]]:
    """Inclusive 1-based physical line spans; ``\\`` continuations stay one span."""
    lines = str(text or "").splitlines()
    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        start = i + 1
        while i + 1 < n and lines[i].rstrip().endswith("\\"):
            i += 1
        ranges.append((start, i + 1))
        i += 1
    return ranges


def lifetime_edges(
    constructs: Sequence[tuple[str, int]],
    destroys: Sequence[tuple[int, str]],
    ranges: Sequence[tuple[int, int]],
) -> list[tuple[str, str]]:
    """``Destroy(A)`` then next construct ``B`` inside the same continued span."""
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for lo, hi in ranges:
        if hi < lo:
            continue
        events: list[tuple[int, int, str]] = []
        for name, line in constructs:
            loc = int(line or 0)
            ident = str(name or "").strip()
            if ident and lo <= loc <= hi:
                events.append((loc, 1, ident))
        for line, recv in destroys:
            loc = int(line or 0)
            ident = str(recv or "").strip()
            if ident and lo <= loc <= hi:
                events.append((loc, 0, ident))
        events.sort()
        pending = ""
        for _loc, kind, ident in events:
            if kind == 0:
                pending = ident
                continue
            if pending and ident and ident != pending:
                pair = (pending, ident)
                if pair not in seen:
                    seen.add(pair)
                    edges.append(pair)
                pending = ""
    return edges


def topo_pipe_ordinals(
    names: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    line_of: Mapping[str, int],
) -> dict[str, int]:
    """Ordinal 1..n. Sources first (then successors); disconnected keep line order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        ident = str(name or "").strip()
        if not ident or ident in seen:
            continue
        seen.add(ident)
        ordered.append(ident)
    if not ordered:
        return {}
    incoming = {name: 0 for name in ordered}
    succ: dict[str, list[str]] = {name: [] for name in ordered}
    for src, dst in edges:
        if src not in incoming or dst not in incoming or src == dst:
            continue
        if dst in succ[src]:
            continue
        succ[src].append(dst)
        incoming[dst] += 1

    def _key(name: str) -> tuple[int, str]:
        return (int(line_of.get(name) or 0), name)

    ready = sorted((name for name in ordered if incoming[name] == 0), key=_key)
    rank: list[str] = []
    while ready:
        cur = ready.pop(0)
        rank.append(cur)
        nxt_ready: list[str] = []
        for nxt in succ[cur]:
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                nxt_ready.append(nxt)
        if nxt_ready:
            ready.extend(nxt_ready)
            ready.sort(key=_key)
    for name in ordered:
        if name not in rank:
            rank.append(name)
    return {name: idx for idx, name in enumerate(rank, start=1)}


def order_pipe_names(
    names: Sequence[str],
    constructs: Sequence[tuple[str, int]],
    destroys: Sequence[tuple[int, str]],
    ranges: Sequence[tuple[int, int]],
    *,
    line_of: Mapping[str, int],
) -> list[str]:
    edges = lifetime_edges(constructs, destroys, ranges)
    if not edges:
        return sorted(
            dict.fromkeys(str(n or "").strip() for n in names if str(n or "").strip()),
            key=lambda name: (int(line_of.get(name) or 0), name),
        )
    ordinals = topo_pipe_ordinals(names, edges, line_of=line_of)
    return sorted(
        ordinals,
        key=lambda name: (int(ordinals.get(name) or 0), int(line_of.get(name) or 0), name),
    )


def receiver_leaf(text: str) -> str:
    raw = str(text or "").strip().replace("->", ".")
    if not raw:
        return ""
    leaf = raw.rsplit(".", 1)[-1].strip()
    return leaf if leaf.isidentifier() else ""
