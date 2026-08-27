# -*- coding: utf-8 -*-
"""Query-acceleration side tables for a committed ``.uo``.

The agent-facing name lookup used to be ``e.name = ? COLLATE NOCASE OR
e.name LIKE '%::' || ?``.  A leading wildcard plus ``OR`` makes SQLite drop
every index on ``entity``, so each hop scanned all rows.  These tables turn
that into an indexed equality probe.

Everything here is derived from ``entity`` / ``source_span`` and can be
rebuilt at any time; ``.uo`` files without it still answer correctly through
the legacy path.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.paths import CANN_MARKER, resolve_operator_file

ACCEL_VERSION = "uo-accel/v2"
_SEL_MACRO_RE = re.compile(r"ASCENDC_TPL_ARGS_SEL\s*\(")

#: What an in-place `upgrade()` may leave behind. This is the complement of
#: `view_projection.NOT_SHIPPED`, not a shorter list: dropping anything else
#: would delete a blob no projector can rebuild.
KEEP_QUERY_BLOBS = (
    "tiling/exhaustive_key_space.yaml",
    "tiling/template_blocks.yaml",
    "tiling/tpl_schema.yaml",
    "tiling/legal_key_index.jsonl",
    "ir/operator_graph.yaml",
    "summary",
)

TEMPLATE_BLOCK_SQL = """
CREATE TABLE IF NOT EXISTS template_block(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  sel_group_index INTEGER,
  file TEXT,
  line_start INTEGER,
  line_end INTEGER,
  product_count INTEGER,
  data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS template_block_dim(
  block_id INTEGER NOT NULL,
  dim TEXT NOT NULL,
  value TEXT NOT NULL,
  is_fixed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tbd_dim_value ON template_block_dim(dim, value, block_id);
CREATE INDEX IF NOT EXISTS idx_tbd_block ON template_block_dim(block_id);
CREATE INDEX IF NOT EXISTS idx_tb_name ON template_block(name);
"""

ACCEL_SQL = """
CREATE TABLE IF NOT EXISTS entity_name_leaf(
  leaf TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  is_ascendc INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_name_leaf ON entity_name_leaf(leaf, is_ascendc);
"""

# `source_span` only holds entity definition snippets, so scanning it can only
# recall names the graph already knows. `source_line` holds every line of the
# operator tree, which makes the fallback a real grep and lets a card report an
# exact total instead of a sample.
#
# `id INTEGER PRIMARY KEY` is load-bearing, not decoration: `source_fts` indexes
# this table by rowid via `content=`, and VACUUM is free to renumber the rowids
# of a table that has no explicit integer key. Without it the index would
# silently point at the wrong lines after the vacuum in `upgrade()`.
SOURCE_LINE_SQL = """
CREATE TABLE IF NOT EXISTS source_line(
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  line INTEGER NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_line_path ON source_line(path, line);
"""

SOURCE_SUFFIXES = (".h", ".hpp", ".cpp", ".cc", ".c", ".cuh")


_IDENT_LEAF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_EXPR_NAME_RE = re.compile(r"[()=!<>&|+\-*/%]")


def _leaf_variants(name: str) -> list[str]:
    """Lowercased full name plus its ``::`` / ``.`` qualified leaf."""
    low = str(name or "").strip().lower()
    if not low:
        return []
    out = [low]
    for sep in ("::", "."):
        if sep in low:
            leaf = low.rsplit(sep, 1)[-1].strip()
            if leaf and leaf not in out:
                out.append(leaf)
    return out


def _indexable_leaves(kind: str, name: str) -> list[str]:
    """Name-lookup leaves. BRANCH expressions are not identifiers.

    A host check named ``!(dim0 == fBaseParams.b)`` used to occupy the leaf
    index as a whole string, crowding out real symbols. Keep a leading
    identifier (``OP_CHECK_IF(...)``) so a name query still lands; drop the
    rest.
    """
    text = str(name or "").strip()
    if not text:
        return []
    if str(kind or "") == "BRANCH" and _EXPR_NAME_RE.search(text):
        match = _IDENT_LEAF_RE.match(text)
        return [match.group(0).lower()] if match else []
    return _leaf_variants(text)


def has_accel(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity_name_leaf' LIMIT 1"
    ).fetchone()
    return row is not None


def build_name_leaf(conn: sqlite3.Connection) -> int:
    """(Re)build the leaf-name inverted index. Returns row count."""
    conn.executescript(ACCEL_SQL)
    conn.execute("DELETE FROM entity_name_leaf")
    rows: list[tuple[str, str, int]] = []
    for eid, kind, name, data in conn.execute(
        "SELECT id, kind, name, IFNULL(data,'') FROM entity "
        "WHERE name IS NOT NULL AND name <> ''"
    ):
        is_ascendc = 1 if '"catalog": "ascendc"' in data or '"catalog":"ascendc"' in data else 0
        for leaf in _indexable_leaves(str(kind or ""), str(name or "")):
            rows.append((leaf, eid, is_ascendc))
    conn.executemany(
        "INSERT INTO entity_name_leaf(leaf, entity_id, is_ascendc) VALUES (?,?,?)", rows
    )
    return len(rows)


def has_source_line(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_line' LIMIT 1"
    ).fetchone()
    return row is not None


def build_source_line(
    conn: sqlite3.Connection, op_root: Path, *, architecture: str = ""
) -> tuple[int, int]:
    """Index every source line the graph can cite. Returns (files, lines).

    A `.uo` is built per architecture, so lines under a foreign `archNN/` are
    skipped: they cannot be a valid citation for this product, and they roughly
    halve the index.

    Shared code beside the operator (`../common/...`) is indexed too, but only
    the files the graph actually references. Walking the sibling tree would pull
    in every other operator in the domain; leaving it out entirely meant every
    entity in shared code was uncitable, which reads as "no such code" rather
    than "not indexed".
    """
    # Dropped rather than emptied so an older product picks up the current
    # schema: an in-place upgrade of a table built before `id` existed would
    # otherwise keep the old shape and break the FTS `content_rowid`.
    conn.execute("DROP TABLE IF EXISTS source_fts")
    conn.execute("DROP TABLE IF EXISTS source_line")
    conn.executescript(SOURCE_LINE_SQL)
    root = Path(op_root)
    arch = str(architecture or "").strip().lower()
    files = 0
    rows: list[tuple[str, int, str]] = []

    def skip(rel_parts: tuple[str, ...]) -> bool:
        # The operator tree itself often lives under `.ascendc-pr`, so only
        # generated artifacts *below* the root are excluded.
        if any(part.startswith(".ascendc-") for part in rel_parts):
            return True
        return bool(arch) and any(
            part.lower().startswith("arch") and part.lower() != arch
            for part in rel_parts
        )

    def index(path: Path, rel: str) -> bool:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        for no, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                rows.append((rel, no, line))
        return True

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if skip(rel.parts):
            continue
        if index(path, rel.as_posix()):
            files += 1

    for rel in _referenced_outside_paths(conn):
        if skip(tuple(rel.split("/"))):
            continue
        resolved = resolve_operator_file(root, rel)
        if resolved is None or resolved.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        # Indexed under the spelling the graph cites, so the recall join is an
        # equality rather than another round of path guessing.
        if index(resolved, rel):
            files += 1

    conn.executemany(
        "INSERT INTO source_line(path, line, text) VALUES (?,?,?)", rows
    )
    return files, len(rows)


def _referenced_outside_paths(conn: sqlite3.Connection) -> list[str]:
    """Distinct locations the graph cites from outside the operator directory.

    Reads the graph rather than the filesystem so the files that get indexed are
    exactly the ones something points at: the sibling tree holds every other
    operator in the domain, and the toolkit holds thousands of headers.
    """
    seen: dict[str, None] = {}
    for table in ("entity", "source_span"):
        for (value,) in conn.execute(
            f"SELECT DISTINCT file FROM {table} "
            "WHERE IFNULL(file,'') LIKE '../%' OR IFNULL(file,'') LIKE ?",
            (CANN_MARKER + "%",),
        ):
            seen.setdefault(str(value), None)
    return list(seen)


def clamp_spans_to_file_length(conn: sqlite3.Connection) -> dict[str, int]:
    """Cut entity spans back to the end of the file they claim.

    Name-keyed entities merge across sites, and a bad merge can leave
    ``line_end`` past end-of-file. Recall maps a text hit to an entity by span
    containment, so an over-long span silently captures every hit in the file.
    Requires ``source_line``; returns counts so the defect stays visible
    instead of being quietly absorbed.
    """
    if not has_source_line(conn):
        return {"clamped": 0, "checked": 0, "unmatched": 0}

    extent: dict[str, int] = {
        str(path): int(mx or 0)
        for path, mx in conn.execute(
            "SELECT path, MAX(line) FROM source_line GROUP BY path"
        )
    }
    # Index by basename so an entity's absolute or partially relative path can
    # still be matched to the indexed operator-relative path.
    by_base: dict[str, list[tuple[str, int]]] = {}
    for path, mx in extent.items():
        by_base.setdefault(path.rsplit("/", 1)[-1], []).append((path, mx))

    stats = {"clamped": 0, "checked": 0, "unmatched": 0}
    updates: list[tuple[int, str]] = []
    for eid, file, line_start, line_end in conn.execute(
        "SELECT id, file, IFNULL(line_start,0), IFNULL(line_end,0) FROM entity "
        "WHERE IFNULL(file,'') <> '' AND IFNULL(line_end,0) > 0"
    ):
        key = str(file or "").replace("\\", "/")
        candidates = by_base.get(key.rsplit("/", 1)[-1]) or []
        limit = 0
        for path, mx in candidates:
            if key.endswith(path) or path.endswith(key):
                limit = mx
                break
        if not limit:
            stats["unmatched"] += 1
            continue
        stats["checked"] += 1
        if int(line_end) > limit:
            updates.append((max(limit, int(line_start)), eid))
    if updates:
        conn.executemany("UPDATE entity SET line_end = ? WHERE id = ?", updates)
        stats["clamped"] = len(updates)
    return stats


def has_template_block(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='template_block' LIMIT 1"
    ).fetchone()
    return row is not None


def _load_template_block_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    row = conn.execute(
        "SELECT data FROM view_blob WHERE name = ?",
        ("tiling/template_blocks.yaml",),
    ).fetchone()
    if row is None:
        return []
    try:
        blob = json.loads(row[0] or "{}")
    except json.JSONDecodeError:
        return []
    if not isinstance(blob, dict):
        return []
    for key in ("groups", "blocks", "rows", "template_blocks"):
        rows = blob.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


def sel_lines_from_header(path: Path) -> list[int]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [text.count("\n", 0, m.start()) + 1 for m in _SEL_MACRO_RE.finditer(text)]


def _resolve_sel_header(conn: sqlite3.Connection, op_root: Path | None) -> Path | None:
    row = conn.execute(
        "SELECT file FROM entity WHERE kind = 'TEMPLATE' AND name LIKE 'ARGS_SEL%' "
        "AND IFNULL(file,'') <> '' LIMIT 1"
    ).fetchone()
    if row is None or not op_root:
        return None
    rel = str(row[0] or "").replace("\\", "/")
    resolved = resolve_operator_file(op_root, rel)
    if resolved is not None:
        return resolved
    # Basename search across the tree is a guess, and `patch_sel_lines` stamps
    # whatever it returns onto every TEMPLATE row, so it only runs when the
    # stored path names a file inside this operator.
    name = Path(rel).name
    if name and not rel.startswith("../") and not rel.startswith(CANN_MARKER):
        hits = sorted(op_root.rglob(name))
        if len(hits) == 1:
            return hits[0]
    return None


def patch_sel_lines(conn: sqlite3.Connection, op_root: Path | None) -> int:
    """Write ARGS_SEL file:line onto TEMPLATE entities. Returns rows updated."""
    header = _resolve_sel_header(conn, op_root)
    if header is None:
        return 0
    lines = sel_lines_from_header(header)
    if not lines:
        return 0
    rel = header.as_posix()
    if op_root:
        try:
            rel = header.relative_to(op_root).as_posix()
        except ValueError:
            pass
    updated = 0
    for index, start in enumerate(lines):
        end = (lines[index + 1] - 1) if index + 1 < len(lines) else start + 24
        name = f"ARGS_SEL_{index}"
        cur = conn.execute(
            "UPDATE entity SET file = ?, line_start = ?, line_end = ? "
            "WHERE kind = 'TEMPLATE' AND name = ?",
            (rel, int(start), int(end), name),
        )
        updated += int(cur.rowcount or 0)
    return updated


def build_template_blocks(conn: sqlite3.Connection) -> int:
    """Materialize SEL blocks as relational rows. Returns block count."""
    conn.executescript(TEMPLATE_BLOCK_SQL)
    conn.execute("DELETE FROM template_block_dim")
    conn.execute("DELETE FROM template_block")
    rows = _load_template_block_rows(conn)
    dim_rows: list[tuple[Any, ...]] = []
    for index, row in enumerate(rows):
        name = str(row.get("name") or f"ARGS_SEL_{index}")
        loc = conn.execute(
            "SELECT file, line_start, line_end FROM entity "
            "WHERE kind = 'TEMPLATE' AND name = ? LIMIT 1",
            (name,),
        ).fetchone()
        file = str((loc[0] if loc else "") or "")
        line_start = int((loc[1] if loc else 0) or 0)
        line_end = int((loc[2] if loc else 0) or 0)
        conn.execute(
            "INSERT INTO template_block("
            "id, name, sel_group_index, file, line_start, line_end, product_count, data"
            ") VALUES (?,?,?,?,?,?,?,?)",
            (
                index,
                name,
                row.get("sel_group_index"),
                file,
                line_start,
                line_end,
                int(row.get("product_count") or 0),
                json.dumps(row, ensure_ascii=False, default=str),
            ),
        )
        fixed = row.get("fixed_fields") if isinstance(row.get("fixed_fields"), dict) else {}
        domains = row.get("field_domains") if isinstance(row.get("field_domains"), dict) else {}
        for dim, value in fixed.items():
            dim_rows.append((index, str(dim), str(value), 1))
        for dim, domain in domains.items():
            values = domain if isinstance(domain, (list, tuple, set)) else [domain]
            for value in values:
                dim_rows.append((index, str(dim), str(value), 0))
    if dim_rows:
        conn.executemany(
            "INSERT INTO template_block_dim(block_id, dim, value, is_fixed) VALUES (?,?,?,?)",
            dim_rows,
        )
    return len(rows)


def build_source_fts(conn: sqlite3.Connection) -> bool:
    """Trigram FTS over `source_line`. Returns False if FTS5 is unavailable.

    Recall answers `pattern` misses by scanning source text for a substring,
    which `LIKE '%x%'` cannot index -- 10ms of full scan per needle, and the
    slowest name cards spent most of their time there. A trigram tokenizer
    indexes substrings, so it answers the same question in well under a
    millisecond.

    `content='source_line'` matters as much as the tokenizer. A standalone FTS5
    table keeps its own copy of the indexed text, which on this product was a
    4.9MB duplicate of bytes `source_line` already held; pointing the index at
    that table instead costs a rowid join and takes the whole feature from
    +13.8MB to +5.8MB. Measured on FAG/arch35: recall went 151ms -> 9ms across
    14 needles with byte-identical row sets, including the `kernel_deter` style
    of identifier that a word tokenizer would have split and lost.
    """
    if not has_source_line(conn):
        return False
    try:
        conn.execute("DROP TABLE IF EXISTS source_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE source_fts USING fts5("
            "text, content='source_line', content_rowid='id', tokenize='trigram')"
        )
        conn.execute("INSERT INTO source_fts(rowid, text) SELECT id, text FROM source_line")
    except sqlite3.OperationalError:
        return False
    return True


def has_source_fts(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_fts' LIMIT 1"
    ).fetchone()
    return row is not None


def prune_view_blobs(conn: sqlite3.Connection, keep: Iterable[str]) -> int:
    """Drop view blobs the agent query path never reads. Returns bytes freed."""
    keep_set = {str(k) for k in keep}
    freed = 0
    victims: list[str] = []
    for name, size in conn.execute(
        "SELECT name, LENGTH(IFNULL(data,'')) FROM view_blob"
    ):
        if name not in keep_set:
            victims.append(name)
            freed += int(size or 0)
    for name in victims:
        conn.execute("DELETE FROM view_blob WHERE name = ?", (name,))
    return freed


def upgrade(
    path: str | Path,
    *,
    op_root: str | Path | None = None,
    architecture: str = "",
    prune: Iterable[str] | None = KEEP_QUERY_BLOBS,
    vacuum: bool = True,
) -> dict[str, Any]:
    """Add acceleration tables to an existing ``.uo`` in place."""
    db = Path(path).expanduser().resolve()
    before = db.stat().st_size
    conn = sqlite3.connect(str(db))
    root = Path(op_root).expanduser().resolve() if op_root is not None else None
    stats: dict[str, Any] = {"product": str(db), "size_before": before}
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        stats["name_leaf_rows"] = build_name_leaf(conn)
        stats["sel_lines_patched"] = patch_sel_lines(conn, root)
        stats["template_blocks"] = build_template_blocks(conn)
        if root is not None:
            files, lines = build_source_line(
                conn, root, architecture=architecture
            )
            stats["source_files"] = files
            stats["source_lines"] = lines
            stats["source_fts"] = build_source_fts(conn)
        if prune is not None:
            stats["view_blob_bytes_freed"] = prune_view_blobs(conn, prune)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
            ("accel_version", ACCEL_VERSION),
        )
        conn.commit()
        if vacuum:
            conn.execute("VACUUM")
            conn.commit()
    finally:
        conn.close()
    stats["size_after"] = db.stat().st_size
    stats["saved_mb"] = round((before - stats["size_after"]) / 1048576, 2)
    return stats
