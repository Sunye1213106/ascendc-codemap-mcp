# -*- coding: utf-8 -*-
"""Stable evidence handles. ``file:line`` is a locator; ``span:{entity}`` is identity."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.service.identity import snapshot_id as make_snapshot_id


def _stat_fingerprint(root: Path, file: str) -> str:
    rel = str(file or "").replace("\\", "/").lstrip("./")
    if not rel:
        return ""
    path = root / rel
    if not path.is_file():
        return ""
    try:
        st = path.stat()
    except OSError:
        return ""
    blob = f"{rel}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def mint(
    *,
    entity_id: str = "",
    file: str = "",
    line: int = 0,
    op_root: Path | None = None,
    snapshot: str = "",
) -> dict[str, Any] | None:
    eid = str(entity_id or "").strip()
    path = str(file or "").replace("\\", "/").strip()
    line_n = int(line or 0)
    if not eid and (not path or line_n <= 0):
        return None
    ev_id = f"span:{eid}" if eid else (
        "ev:" + hashlib.sha256(f"{path}|{line_n}".encode("utf-8")).hexdigest()[:16]
    )
    fingerprint = _stat_fingerprint(op_root, path) if op_root is not None else ""
    return {
        "id": ev_id,
        "entity_id": eid,
        "file": path,
        "line": line_n,
        "source_stat_fingerprint": fingerprint,
        "snapshot_id": snapshot,
    }


def collect(payload: dict[str, Any], *, op_root: Path, snapshot: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: Any) -> None:
        if not isinstance(row, dict):
            return
        item = mint(
            entity_id=str(row.get("id") or row.get("entity_id") or row.get("_entity_id") or ""),
            file=str(row.get("file") or ""),
            line=int(row.get("line") or row.get("line_start") or 0),
            op_root=op_root,
            snapshot=snapshot,
        )
        if item is None or item["id"] in seen:
            return
        seen.add(item["id"])
        found.append(item)

    for key in ("cards", "hits", "seeds", "sel_sites", "phases", "neighbors"):
        rows = payload.get(key)
        if isinstance(rows, list):
            for row in rows:
                add(row)
    for key in ("enclosing", "entry", "definition", "match"):
        add(payload.get(key))
    host = payload.get("host") if isinstance(payload.get("host"), dict) else {}
    kernel = payload.get("kernel") if isinstance(payload.get("kernel"), dict) else {}
    for row in list(host.get("writers") or [])[:8]:
        add(row)
    for row in list(kernel.get("readers") or [])[:8]:
        add(row)
    defn = payload.get("definition")
    if isinstance(defn, dict):
        add(defn)
    elif isinstance(defn, list):
        for row in defn[:4]:
            add(row)
    return found[:16]


def lookup_span(query: Any, evidence_id: str) -> dict[str, Any] | None:
    eid = str(evidence_id or "").strip()
    if not eid:
        return None
    span_id = eid
    entity_id = ""
    if eid.startswith("span:"):
        entity_id = eid[len("span:") :]
    elif eid.startswith("ev:"):
        span_id = ""
    else:
        entity_id = eid
        span_id = f"span:{eid}"
    try:
        with query._connect() as conn:  # noqa: SLF001 — same SQLite the query engine uses
            row = None
            if span_id:
                row = conn.execute(
                    "SELECT id, entity_id, file, line_start, line_end FROM source_span WHERE id = ?",
                    (span_id,),
                ).fetchone()
            if row is None and entity_id:
                row = conn.execute(
                    "SELECT id, entity_id, file, line_start, line_end FROM source_span "
                    "WHERE entity_id = ? LIMIT 1",
                    (entity_id,),
                ).fetchone()
            if row is None and entity_id:
                ent = conn.execute(
                    "SELECT id, file, line_start, line_end FROM entity WHERE id = ?",
                    (entity_id,),
                ).fetchone()
                if ent is not None:
                    return {
                        "id": f"span:{ent[0]}",
                        "entity_id": str(ent[0]),
                        "file": str(ent[1] or ""),
                        "line": int(ent[2] or 0),
                    }
            if row is None:
                return None
            return {
                "id": str(row[0]),
                "entity_id": str(row[1] or ""),
                "file": str(row[2] or ""),
                "line": int(row[3] or 0),
            }
    except Exception:  # noqa: BLE001
        return None


def attach_snapshot(items: list[dict[str, Any]], product: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    sid = make_snapshot_id(product, meta)
    for item in items:
        item.setdefault("snapshot_id", sid)
    return items
