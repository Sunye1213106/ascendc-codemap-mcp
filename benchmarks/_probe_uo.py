# -*- coding: utf-8 -*-
"""One-shot UO probe for the 20-question MCP surface review."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

UO = Path(
    r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"
    r"\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
)
OUT = Path(r"d:\TEST\ascendc-codemap-mcp\benchmarks\_uo_probe.json")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(UO))
    conn.row_factory = sqlite3.Row
    return conn


def tables(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")]


def fts_hits(conn: sqlite3.Connection, query: str, limit: int = 8) -> list[dict]:
    sql = """
    SELECT sl.file, sl.line, sl.text
    FROM source_fts
    JOIN source_line sl ON source_fts.rowid = sl.rowid
    WHERE source_fts MATCH ?
    LIMIT ?
    """
    try:
        rows = conn.execute(sql, (query, limit)).fetchall()
        return [{"file": r["file"], "line": r["line"], "text": (r["text"] or "")[:160]} for r in rows]
    except sqlite3.Error as exc:
        return [{"error": str(exc)}]


def name_hits(conn: sqlite3.Connection, like: str, limit: int = 8) -> list[dict]:
    sql = """
    SELECT kind, name, file, line_start AS line
    FROM entity
    WHERE name LIKE ? COLLATE NOCASE
    ORDER BY length(name), kind
    LIMIT ?
    """
    rows = conn.execute(sql, (like, limit)).fetchall()
    return [{"kind": r["kind"], "name": r["name"], "file": r["file"], "line": r["line"]} for r in rows]


def exact_name(conn: sqlite3.Connection, name: str, limit: int = 12) -> list[dict]:
    sql = """
    SELECT kind, name, file, line_start AS line
    FROM entity
    WHERE name = ? OR name LIKE ?
    ORDER BY kind
    LIMIT ?
    """
    rows = conn.execute(sql, (name, f"%{name}", limit)).fetchall()
    return [{"kind": r["kind"], "name": r["name"], "file": r["file"], "line": r["line"]} for r in rows]


def writers_readers(conn: sqlite3.Connection, leaf: str, limit: int = 10) -> dict:
    sql = """
    SELECT r.kind AS rel, e.kind AS ekind, e.name, e.file, e.line_start AS line
    FROM relation r
    JOIN entity e ON e.id = r.src
    JOIN entity t ON t.id = r.dst
    WHERE t.name LIKE ? AND r.kind IN ('WRITES','READS','DERIVES','BINDS','SELECTS')
    LIMIT ?
    """
    rows = conn.execute(sql, (f"%{leaf}%", limit)).fetchall()
    return [
        {
            "rel": r["rel"],
            "kind": r["ekind"],
            "name": r["name"],
            "file": r["file"],
            "line": r["line"],
        }
        for r in rows
    ]


def dim_counts(conn: sqlite3.Connection, dim: str) -> list[dict]:
    sql = """
    SELECT value, COUNT(*) AS n
    FROM legal_key_dim
    WHERE dim = ?
    GROUP BY value
    ORDER BY n DESC
    """
    try:
        rows = conn.execute(sql, (dim,)).fetchall()
        return [{"value": r["value"], "n": r["n"]} for r in rows]
    except sqlite3.Error as exc:
        return [{"error": str(exc)}]


def combo(conn: sqlite3.Connection, filters: dict[str, str]) -> int:
    # filters: dim -> value
    sql = """
    SELECT COUNT(DISTINCT k.key_id)
    FROM legal_key k
    """
    joins = []
    args: list[str] = []
    for i, (dim, value) in enumerate(filters.items()):
        alias = f"d{i}"
        sql += f" JOIN legal_key_dim {alias} ON {alias}.key_id = k.key_id AND {alias}.dim = ? AND {alias}.value = ?"
        args.extend([dim, value])
    try:
        return int(conn.execute(sql, args).fetchone()[0])
    except sqlite3.Error:
        # try name/value columns
        return -1


def legal_schema(conn: sqlite3.Connection) -> dict:
    out = {}
    for table in ("legal_key", "legal_key_dim", "source_fts", "source_line", "entity", "relation"):
        try:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
            out[table] = cols
        except sqlite3.Error as exc:
            out[table] = str(exc)
    return out


def dims(conn: sqlite3.Connection) -> list[str]:
    try:
        return [r[0] for r in conn.execute("SELECT DISTINCT dim FROM legal_key_dim ORDER BY 1")]
    except sqlite3.Error:
        return []


def main() -> None:
    conn = connect()
    report: dict = {
        "tables": tables(conn),
        "counts": {},
        "schema": legal_schema(conn),
        "dims": dims(conn),
    }
    for t in [
        "entity",
        "relation",
        "source_span",
        "source_line",
        "source_fts",
        "legal_key",
        "legal_key_dim",
    ]:
        try:
            report["counts"][t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.Error as exc:
            report["counts"][t] = str(exc)

    # FTS match syntax probe
    for q in ["IS_SMALL_D_PRELOAD", "empty tensor", "IsEmptyTensor", "keepProb", "dropMask"]:
        report.setdefault("fts_probe", {})[q] = fts_hits(conn, q, 3)

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tables": report["tables"], "counts": report["counts"], "dims": report["dims"], "schema": report["schema"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
