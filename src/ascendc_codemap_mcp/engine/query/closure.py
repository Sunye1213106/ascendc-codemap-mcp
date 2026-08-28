# -*- coding: utf-8 -*-
"""Deterministic semantic impact BFS over the contract-edge whitelist."""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity
from ascendc_codemap_mcp.engine.query.evidence import CLOSURE_EDGE_KINDS

_SINK_KINDS = frozenset(
    {
        "OPERATION",
        "BUFFER",
        "QUEUE",
        "PIPE",
        "KERNEL",
        "EVENT",
    }
)
_SINK_EDGE = frozenset({"CALLS_UNDER_GUARD", "ALLOCATES", "ROOTED_AT", "LAUNCHES"})
_MAX_HOPS = 8
_MAX_NODES = 80


def _kind_name(ent: Entity | None) -> str:
    if ent is None:
        return ""
    return ent.kind_name()


def semantic_impact_closure_mem(
    codemap: CodeMap,
    seed_ids: Iterable[str],
    *,
    max_hops: int = _MAX_HOPS,
) -> dict[str, Any]:
    """In-memory closure used at index tests and compile. Bidirectional whitelist."""
    wanted = set(CLOSURE_EDGE_KINDS)
    seeds = [str(s) for s in seed_ids if str(s)]
    seen: dict[str, int] = {sid: 0 for sid in seeds}
    queue: deque[tuple[str, int]] = deque((sid, 0) for sid in seeds)
    sinks: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    while queue and len(seen) < _MAX_NODES:
        cur, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for rel in codemap.relations.values():
            if rel.kind_name() not in wanted:
                continue
            nxt = ""
            if rel.src == cur:
                nxt = rel.dst
            elif rel.dst == cur:
                nxt = rel.src
            if not nxt or nxt in seen:
                if nxt == cur:
                    continue
                if nxt and nxt in seen and rel.kind_name() in _SINK_EDGE:
                    dst = codemap.entities.get(rel.dst if rel.src == cur else rel.src)
                    if dst is not None and _kind_name(dst) in _SINK_KINDS:
                        sinks.append(
                            {
                                "id": dst.id,
                                "name": dst.name,
                                "kind": dst.kind_name(),
                                "file": dst.file,
                                "line": dst.line_start,
                                "via": rel.kind_name(),
                                "consumer_role": rel.attrs.get("consumer_role") or "",
                            }
                        )
                continue
            seen[nxt] = dist + 1
            queue.append((nxt, dist + 1))
            other = codemap.entities.get(nxt)
            edges.append(
                {
                    "kind": rel.kind_name(),
                    "src": rel.src,
                    "dst": rel.dst,
                    "consumer_role": rel.attrs.get("consumer_role") or "",
                    "file": rel.attrs.get("file") or "",
                    "line": rel.attrs.get("line") or 0,
                }
            )
            if other is not None and (
                _kind_name(other) in _SINK_KINDS or rel.kind_name() in {"CALLS_UNDER_GUARD", "ALLOCATES"}
            ):
                sinks.append(
                    {
                        "id": other.id,
                        "name": other.name,
                        "kind": other.kind_name(),
                        "file": other.file,
                        "line": other.line_start,
                        "via": rel.kind_name(),
                        "consumer_role": rel.attrs.get("consumer_role") or "",
                    }
                )
    uniq: dict[str, dict[str, Any]] = {}
    for row in sinks:
        uniq.setdefault(str(row.get("id") or ""), row)
    return {
        "seeds": seeds,
        "nodes": sorted(seen, key=lambda i: (seen[i], i)),
        "distance": seen,
        "sinks": list(uniq.values())[:40],
        "edges": edges[:80],
    }


def semantic_impact_closure_sql(
    conn: Any,
    seed_ids: Iterable[str],
    *,
    max_hops: int = _MAX_HOPS,
    entity_row=None,
) -> dict[str, Any]:
    """SQLite walk. ``entity_row(conn, id)`` optional for sink names."""
    wanted = tuple(sorted(CLOSURE_EDGE_KINDS))
    placeholders = ",".join("?" for _ in wanted)
    seeds = [str(s) for s in seed_ids if str(s)]
    seen: dict[str, int] = {sid: 0 for sid in seeds}
    queue: deque[tuple[str, int]] = deque((sid, 0) for sid in seeds)
    sinks: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    sql = f"""
        SELECT id, kind, src, dst, data FROM relation
        WHERE (src = ? OR dst = ?) AND kind IN ({placeholders})
    """
    while queue and len(seen) < _MAX_NODES:
        cur, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for row in conn.execute(sql, (cur, cur, *wanted)):
            kind = str(row["kind"] or "")
            src = str(row["src"] or "")
            dst = str(row["dst"] or "")
            nxt = dst if src == cur else src
            if not nxt:
                continue
            data = row["data"]
            role = ""
            file = ""
            line = 0
            if isinstance(data, str) and data:
                import json

                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    role = str(parsed.get("consumer_role") or "")
                    file = str(parsed.get("file") or "")
                    line = int(parsed.get("line") or 0)
            if nxt not in seen:
                seen[nxt] = dist + 1
                queue.append((nxt, dist + 1))
                edges.append(
                    {
                        "kind": kind,
                        "src": src,
                        "dst": dst,
                        "consumer_role": role,
                        "file": file,
                        "line": line,
                    }
                )
            is_sink_edge = kind in {"CALLS_UNDER_GUARD", "ALLOCATES"}
            if is_sink_edge or nxt not in seeds:
                info = {"id": nxt, "name": "", "kind": "", "file": file, "line": line, "via": kind, "consumer_role": role}
                if callable(entity_row):
                    rec = entity_row(conn, nxt)
                    if rec is not None:
                        info["name"] = str(rec["name"] if "name" in rec.keys() else rec.get("name") or "")
                        info["kind"] = str(rec["kind"] if "kind" in rec.keys() else rec.get("kind") or "")
                        if not info["file"]:
                            info["file"] = str(rec["file"] if "file" in rec.keys() else rec.get("file") or "")
                        if not info["line"]:
                            info["line"] = int(
                                rec["line_start"] if "line_start" in rec.keys() else rec.get("line_start") or 0
                            )
                if info["kind"] in _SINK_KINDS or is_sink_edge:
                    sinks.append(info)
    uniq: dict[str, dict[str, Any]] = {}
    for row in sinks:
        uniq.setdefault(str(row.get("id") or ""), row)
    return {
        "seeds": seeds,
        "nodes": sorted(seen, key=lambda i: (seen[i], i)),
        "distance": seen,
        "sinks": list(uniq.values())[:40],
        "edges": edges[:80],
    }
