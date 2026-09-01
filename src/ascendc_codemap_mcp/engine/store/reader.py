# -*- coding: utf-8 -*-
"""Read CodeMap / views from a ``.uo`` SQLite product."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

_CONN_LOCK = threading.Lock()
# (resolved path, thread ident) → connection. MCP SDK runs sync tools on a
# worker pool; sqlite3 connections are not shareable across threads and must
# not be used concurrently even with check_same_thread=False.
_CONN: dict[tuple[str, int], sqlite3.Connection] = {}
_IDLE_TIMERS: dict[str, threading.Timer] = {}
_IDLE_GEN: dict[str, int] = {}


def _idle_sec() -> float:
    # Agent think-time between tool calls is tens of seconds. Closing the
    # handle after 2s made every OpenCode round trip a cold SQLite open of a
    # 100MB+ product, which is the difference between 40ms and multiple seconds.
    raw = str(os.environ.get("ASCENDC_CODEMAP_SQLITE_IDLE_SEC", "120") or "120").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 120.0


@lru_cache(maxsize=256)
def _resolved_product_key(text: str) -> str:
    return str(Path(text).expanduser().resolve())


def _product_key(path: str | Path) -> str:
    # One resolve() is a filesystem round trip, and a single query asks for the
    # connection well over a hundred times.
    return _resolved_product_key(str(path))


def _configure_readonly(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Read-only SQLite. Daemon holds one connection, so mmap is safe on POSIX.

    Windows: mmap keeps a mapping that can survive ``close()`` and still block
    DeleteFile / rename of the ``.uo``. Keep mmap off there.
    """
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = 1")
        conn.execute("PRAGMA cache_size = -32000")
        if sys.platform == "win32":
            conn.execute("PRAGMA mmap_size = 0")
        else:
            conn.execute("PRAGMA mmap_size = 67108864")
        conn.execute("PRAGMA temp_store = MEMORY")
    except sqlite3.Error:
        pass
    return conn


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    db = Path(path).expanduser().resolve()
    if not db.is_file():
        raise FileNotFoundError(f"missing .uo product: {db}")
    # Queries stay on the creating thread via (path, thread_ident).
    # check_same_thread=False only so a writer can close idle handles after
    # SnapshotLocks has drained readers.
    conn = sqlite3.connect(
        f"file:{db.as_posix()}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    return _configure_readonly(conn)


def _pool_max() -> int:
    raw = str(os.environ.get("ASCENDC_CODEMAP_SQLITE_POOL", "8") or "8").strip()
    try:
        return max(1, min(32, int(raw)))
    except ValueError:
        return 8


# Connections leased for the duration of one query, reusable across threads.
# Thread-local slots remain as a fallback for callers that never go through
# QueryCache.open (tests, dump helpers).
_TLS = threading.local()
_POOL: dict[str, list[sqlite3.Connection]] = {}
_POOL_LIVE: dict[str, int] = {}
_POOL_COND = threading.Condition(_CONN_LOCK)


def lease_query_connection(path: str | Path) -> sqlite3.Connection:
    """Pin one pooled connection to this thread for the current query.

    Nested leases of the same product reuse the same connection, which is how
    ``_connect`` is used inside a query. A new thread checks out an idle handle
    instead of opening another 100MB mapping.
    """
    key = _product_key(path)
    depth = int(getattr(_TLS, "depth", 0) or 0)
    if depth > 0 and getattr(_TLS, "key", None) == key and getattr(_TLS, "conn", None) is not None:
        _TLS.depth = depth + 1
        return _TLS.conn
    conn = _pool_checkout(key)
    _TLS.conn = conn
    _TLS.key = key
    _TLS.depth = 1
    return conn


def release_query_connection(path: str | Path) -> None:
    depth = int(getattr(_TLS, "depth", 0) or 0)
    if depth > 1:
        _TLS.depth = depth - 1
        return
    conn = getattr(_TLS, "conn", None)
    key = getattr(_TLS, "key", None)
    _TLS.depth = 0
    _TLS.conn = None
    _TLS.key = None
    if conn is not None and key is not None:
        _pool_checkin(key, conn)


def _pool_checkout(key: str) -> sqlite3.Connection:
    create = False
    with _POOL_COND:
        while True:
            idle = _POOL.setdefault(key, [])
            if idle:
                return idle.pop()
            live = int(_POOL_LIVE.get(key, 0) or 0)
            if live < _pool_max():
                _POOL_LIVE[key] = live + 1
                create = True
                break
            _POOL_COND.wait(timeout=1.0)
    if create:
        return _connect_readonly(key)
    # Waited and still nothing: open a short-lived extra rather than hang.
    return _connect_readonly(key)


def _pool_checkin(key: str, conn: sqlite3.Connection) -> None:
    with _POOL_COND:
        idle = _POOL.setdefault(key, [])
        if len(idle) < _pool_max():
            idle.append(conn)
            _POOL_COND.notify()
            return
    _close_conn(conn)


def shared_uo(path: str | Path) -> sqlite3.Connection:
    """Read-only connection for one ``.uo`` path.

    Prefer the connection leased for this query so concurrent MCP workers share
    a small pool instead of one mapping per thread. Fall back to a thread-local
    handle for callers that never leased.
    """
    key = _product_key(path)
    _cancel_idle(key)
    leased = getattr(_TLS, "conn", None)
    if leased is not None and getattr(_TLS, "key", None) == key:
        return leased
    tid = threading.get_ident()
    slot = (key, tid)
    with _CONN_LOCK:
        hit = _CONN.get(slot)
        if hit is not None:
            return hit
    conn = _connect_readonly(key)
    with _CONN_LOCK:
        again = _CONN.get(slot)
        if again is not None:
            _close_conn(conn)
            return again
        _CONN[slot] = conn
        return conn


def _close_conn(conn: sqlite3.Connection) -> None:
    try:
        from ascendc_codemap_mcp.engine.query.sql import drop_source_line_cache

        drop_source_line_cache(conn)
    except Exception:  # noqa: BLE001
        pass
    try:
        conn.execute("PRAGMA mmap_size = 0")
    except sqlite3.Error:
        pass
    try:
        conn.close()
    except sqlite3.ProgrammingError as exc:
        if "closed" not in str(exc).lower():
            raise
    except sqlite3.Error:
        pass


def _cancel_idle(key: str) -> None:
    with _CONN_LOCK:
        timer = _IDLE_TIMERS.pop(key, None)
        _IDLE_GEN[key] = _IDLE_GEN.get(key, 0) + 1
    if timer is not None:
        timer.cancel()


def mark_uo_in_use(path: str | Path) -> None:
    """A query is starting; do not idle-close this product."""
    _cancel_idle(_product_key(path))


def mark_uo_idle(path: str | Path) -> None:
    """No in-flight query; close pooled handles after a short idle."""
    key = _product_key(path)
    delay = _idle_sec()
    if delay <= 0:
        close_uo_connections(key)
        return
    _cancel_idle(key)
    with _CONN_LOCK:
        gen = _IDLE_GEN.get(key, 0)

    def fire() -> None:
        with _CONN_LOCK:
            if _IDLE_GEN.get(key) != gen:
                return
            _IDLE_TIMERS.pop(key, None)
        close_uo_connections(key)

    timer = threading.Timer(delay, fire)
    timer.daemon = True
    with _CONN_LOCK:
        _IDLE_TIMERS[key] = timer
    timer.start()


def close_uo_connections(path: str | Path | None = None) -> None:
    """Close pooled read connections so Windows can replace the ``.uo``.

    Safe to call from the writer thread: connections are opened with
    ``check_same_thread=False`` solely so this close can run after readers
    have drained. SQL still runs only on the creating thread.
    """
    pooled: list[sqlite3.Connection] = []
    with _CONN_LOCK:
        if path is None:
            items = list(_CONN.items())
            _CONN.clear()
            timers = list(_IDLE_TIMERS.values())
            _IDLE_TIMERS.clear()
            _IDLE_GEN.clear()
            for idle in _POOL.values():
                pooled.extend(idle)
            _POOL.clear()
            _POOL_LIVE.clear()
            _META_CACHE.clear()
            _POOL_COND.notify_all()
        else:
            key = _product_key(path)
            items = [(slot, conn) for slot, conn in list(_CONN.items()) if slot[0] == key]
            for slot, _ in items:
                _CONN.pop(slot, None)
            timers = []
            idle = _IDLE_TIMERS.pop(key, None)
            if idle is not None:
                timers.append(idle)
            _IDLE_GEN[key] = _IDLE_GEN.get(key, 0) + 1
            pooled.extend(_POOL.pop(key, []))
            _POOL_LIVE.pop(key, None)
            _META_CACHE.pop(key, None)
            _POOL_COND.notify_all()
    for timer in timers:
        timer.cancel()
    errors: list[BaseException] = []
    for _, conn in items:
        try:
            _close_conn(conn)
        except sqlite3.ProgrammingError as exc:
            errors.append(exc)
    for conn in pooled:
        try:
            _close_conn(conn)
        except sqlite3.ProgrammingError as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def open_handle_count(path: str | Path | None = None) -> int:
    with _CONN_LOCK:
        if path is None:
            pooled = sum(len(v) for v in _POOL.values())
            return len(_CONN) + pooled
        key = _product_key(path)
        local = sum(1 for slot in _CONN if slot[0] == key)
        return local + len(_POOL.get(key) or [])

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.evidence import (
    SOURCE_UNSPECIFIED,
    STATE_RESOLVED,
    TRUST_LEGACY_UNKNOWN,
    grow_evidence_attrs,
)
from ascendc_codemap_mcp.engine.ir.relation import Relation, RelationKind


def _legacy_trust_attrs(
    attrs: dict[str, Any], *, legacy: bool, build_context_id: str = ""
) -> dict[str, Any]:
    """Rebuild the evidence fields the writer left out.

    v1 products: missing trust is unknown, not lexical -- there is no
    provenance to derive it from. v2 stores only what its provenance does not
    already say, so the absent fields are re-derived here and every consumer
    still sees a complete record.
    """
    if not legacy:
        return grow_evidence_attrs(attrs, build_context_id=build_context_id)
    if str(attrs.get("trust") or "") in {TRUST_LEGACY_UNKNOWN, "authoritative", "derived", "advisory"}:
        return attrs
    attrs["trust"] = TRUST_LEGACY_UNKNOWN
    attrs.setdefault("evidence_source", SOURCE_UNSPECIFIED)
    attrs.setdefault("semantic_state", STATE_RESOLVED)
    return attrs


def open_uo(path: str | Path) -> sqlite3.Connection:
    """Exclusive short-lived connection. Caller must ``close()``."""
    return _connect_readonly(path)


_META_CACHE: dict[str, tuple[int, dict[str, str]]] = {}


def read_meta(path: str | Path) -> dict[str, str]:
    key = _product_key(path)
    try:
        mtime = int(Path(key).stat().st_mtime_ns)
    except OSError:
        mtime = 0
    hit = _META_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return dict(hit[1])
    conn = shared_uo(path) if getattr(_TLS, "conn", None) is not None else open_uo(path)
    close_after = conn is not getattr(_TLS, "conn", None)
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
        meta = {str(r["key"]): str(r["value"]) for r in rows}
    finally:
        if close_after:
            _close_conn(conn)
    _META_CACHE[key] = (mtime, meta)
    return dict(meta)


def read_codemap(
    path: str | Path,
    *,
    entity_kinds: Iterable[str] | None = None,
    relation_kinds: Iterable[str] | None = None,
) -> CodeMap:
    """Materialize the graph, or the slice of it named by the kind filters.

    A projection needs a handful of kinds, not all 58k rows: loading BRANCH
    alone is ~40x cheaper than the whole graph. When a filter is given the
    canonical counts come from `meta` instead of the loaded rows, so a slice
    still stamps the identity of the graph it was cut from rather than
    advertising itself as a smaller graph.
    """
    conn = open_uo(path)
    try:
        meta = {str(r["key"]): str(r["value"]) for r in conn.execute("SELECT key, value FROM meta")}
        cm = CodeMap(
            op_name=meta.get("op_name") or "",
            architecture=meta.get("architecture") or "",
        )
        cm.meta = {k[3:]: _maybe_json(v) for k, v in meta.items() if k.startswith("cm_")}
        partial = entity_kinds is not None or relation_kinds is not None
        if partial:
            for key in ("entity_count", "relation_count"):
                if meta.get(key):
                    cm.meta[key] = int(meta[key])
        from ascendc_codemap_mcp.engine.store.schema import SCHEMA_COMPAT

        schema = str(meta.get("schema") or "")
        legacy = schema == "codemap-uo/v1" or schema not in SCHEMA_COMPAT
        if legacy:
            cm.meta["trust_model"] = "legacy_unknown"
        # Rows carry this only when they disagree with the product; putting it
        # back here is what keeps it out of 58k copies on disk.
        context_id = str(meta.get("cm_build_context_id") or "")
        ent_sql = (
            "SELECT id, kind, name, status, confidence, file, line_start, line_end, data"
            " FROM entity"
        )
        ent_args: tuple[str, ...] = ()
        if entity_kinds is not None:
            wanted = tuple(sorted({str(k) for k in entity_kinds}))
            if not wanted:
                ent_sql += " WHERE 0"
            else:
                ent_sql += f" WHERE kind IN ({','.join('?' * len(wanted))})"
                ent_args = wanted
        for row in conn.execute(ent_sql, ent_args):
            data = json.loads(row["data"] or "{}")
            attrs = {
                k: v
                for k, v in data.items()
                if k
                not in {
                    "id",
                    "kind",
                    "name",
                    "status",
                    "confidence",
                    "file",
                    "line_start",
                    "line_end",
                }
            }
            kind_name = str(row["kind"])
            try:
                kind: EntityKind | str = EntityKind(kind_name)
            except ValueError:
                kind = kind_name
            cm.add_entity(
                Entity(
                    id=str(row["id"]),
                    kind=kind,
                    name=str(row["name"] or ""),
                    attrs=_legacy_trust_attrs(attrs, legacy=legacy, build_context_id=context_id),
                    file=str(row["file"] or ""),
                    line_start=int(row["line_start"] or 0),
                    line_end=int(row["line_end"] or 0),
                    status=str(row["status"] or "extracted"),
                    confidence=float(row["confidence"] or 1.0),
                ),
                stamp=not legacy,
            )
        rel_sql = "SELECT id, kind, src, dst, status, confidence, data FROM relation"
        rel_args: tuple[str, ...] = ()
        if relation_kinds is not None:
            wanted_rel = tuple(sorted({str(k) for k in relation_kinds}))
            if not wanted_rel:
                rel_sql += " WHERE 0"
            else:
                rel_sql += f" WHERE kind IN ({','.join('?' * len(wanted_rel))})"
                rel_args = wanted_rel
        for row in conn.execute(rel_sql, rel_args):
            data = json.loads(row["data"] or "{}")
            attrs = {
                k: v
                for k, v in data.items()
                if k not in {"id", "kind", "src", "dst", "status", "confidence"}
            }
            kind_name = str(row["kind"])
            try:
                rkind: RelationKind | str = RelationKind(kind_name)
            except ValueError:
                rkind = kind_name
            cm.relations[str(row["id"])] = Relation(
                id=str(row["id"]),
                kind=rkind,
                src=str(row["src"]),
                dst=str(row["dst"]),
                attrs=_legacy_trust_attrs(attrs, legacy=legacy, build_context_id=context_id),
                status=str(row["status"] or "extracted"),
                confidence=float(row["confidence"] or 1.0),
            )
        return cm
    finally:
        conn.close()


def load_view_blob(
    path: str | Path,
    name: str,
    *,
    expand_legal_keys: bool = True,
) -> Any | None:
    conn = open_uo(path)
    try:
        row = conn.execute(
            "SELECT data FROM view_blob WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        blob = json.loads(row["data"])
    finally:
        _close_conn(conn)
    if (
        expand_legal_keys
        and name == "tiling/legal_key_index.jsonl"
        and isinstance(blob, dict)
    ):
        from ascendc_codemap_mcp.engine.query.legal_key_cache import expand_legal_key_rows

        blob = dict(blob)
        blob["rows"] = expand_legal_key_rows(blob)
    return blob


def load_production_view(path: str | Path, name: str) -> Any | None:
    """Load a view for production callers. Stale blobs are never returned."""
    checked = load_view_blob_checked(path, name)
    if checked.get("ok"):
        return checked.get("view")
    return None


def _architecture_from_uo_name(path: Path) -> str:
    from ascendc_codemap_mcp.engine.source_layout import is_product_architecture

    name = path.name
    if not name.endswith(".uo"):
        return ""
    stem = name[: -len(".uo")]
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and is_product_architecture(parts[1]):
        return parts[1]
    return ""


def load_view_blob_checked(
    path: str | Path,
    name: str,
    *,
    codemap: CodeMap | None = None,
    fallback_canonical: bool = True,
    expand_legal_keys: bool = True,
) -> dict[str, Any]:
    """Load a projection with fail-closed provenance validation.

    A stale/legacy blob is never returned as the usable ``view``.  Known
    projections are rebuilt from the canonical CodeMap engine-side; unknown
    projections return ``view=None`` so callers cannot accidentally consume an
    unverifiable shortcut.  ``stale_blob`` is retained only for diagnostics.
    """
    from ascendc_codemap_mcp.engine.projection_provenance import (
        LEGACY_VIEW_UNVERIFIED,
        VIEW_STALE,
        validate_view_against_codemap,
    )
    from ascendc_codemap_mcp.engine.tg_views import (
        finalize_tg_views,
        project_kernel_view,
        project_operator_graph,
        project_tilingdata_view,
        project_tg_host_view,
    )

    blob = load_view_blob(path, name, expand_legal_keys=expand_legal_keys)
    if blob is None:
        # A projection is derived data, so an absent blob is a cache miss rather
        # than missing information: rebuild it from the graph the same way a
        # stale one is rebuilt below. Products stop shipping these documents,
        # and no caller has to learn a second way to ask.
        from ascendc_codemap_mcp.engine.store.view_projection import is_projectable, project_view

        if fallback_canonical and codemap is None and is_projectable(name):
            projected = project_view(path, name)
            if projected is not None:
                return {
                    "ok": True,
                    "reason_code": "",
                    "name": name,
                    "view": projected,
                    "fallback": "projected",
                }
        return {"ok": False, "reason_code": "VIEW_MISSING", "name": name, "view": None}
    stored_digest = str(_maybe_json(read_meta(path).get("cm_canonical_graph_digest") or "") or "")
    prov = blob.get("provenance") if isinstance(blob, dict) else None
    actual_digest = str((prov or {}).get("canonical_graph_digest") or "") if isinstance(prov, dict) else ""
    if stored_digest and actual_digest and stored_digest == actual_digest:
        return {"ok": True, "reason_code": "", "name": name, "view": blob}
    if codemap is None and not fallback_canonical:
        return {
            "ok": False,
            "reason_code": VIEW_STALE if actual_digest else LEGACY_VIEW_UNVERIFIED,
            "name": name,
            "view": None,
            "stale_blob": blob,
            "check": {
                "ok": False,
                "reason_code": "DIGEST_META_MISMATCH",
                "expected": {"canonical_graph_digest": stored_digest},
                "actual": {"canonical_graph_digest": actual_digest},
            },
        }
    cm = codemap if codemap is not None else read_codemap(path)
    check = validate_view_against_codemap(blob, cm)
    if check.get("ok"):
        return {"ok": True, "reason_code": "", "name": name, "view": blob}
    result: dict[str, Any] = {
        "ok": False,
        "reason_code": check.get("reason_code") or VIEW_STALE,
        "name": name,
        "view": None,
        "stale_blob": blob,
        "check": check,
    }
    if not fallback_canonical:
        return result
    rebuilt: Any = None
    if name == "ir/operator_graph.yaml":
        rebuilt = project_operator_graph(cm)
    elif name == "ir/tg_host_view.yaml":
        rebuilt = project_tg_host_view(cm)
    elif name == "views/kernel.yaml":
        rebuilt = project_kernel_view(cm)
    elif name == "views/tilingdata.yaml":
        rebuilt = project_tilingdata_view(cm)
    elif name == "summary":
        rebuilt = {
            "entity_count": len(cm.entities),
            "relation_count": len(cm.relations),
            "graph_fingerprint": cm.meta.get("graph_fingerprint"),
        }
    if rebuilt is not None:
        from ascendc_codemap_mcp.engine.projection_provenance import stamp_provenance

        # Ensure fingerprint meta exists for stamp.
        if not cm.meta.get("graph_fingerprint"):
            finalize_tg_views(cm, existing={})
        result["ok"] = True
        result["fallback"] = "canonical"
        result["view"] = stamp_provenance(rebuilt, cm)
    return result


def list_views(path: str | Path) -> list[str]:
    conn = open_uo(path)
    try:
        return [str(r["name"]) for r in conn.execute("SELECT name FROM view_blob ORDER BY name")]
    finally:
        _close_conn(conn)


def find_uo_product(
    op_root: str | Path,
    *,
    op_name: str = "",
    architecture: str = "",
) -> Path | None:
    """Locate the CodeMap product ``.ascendc-codemap/<arch>/<op>.<arch>.uo``.

    Production authority is arch-scoped ``*.<arch>.uo`` only. Top-level
    ``.ascendc-codemap/uo/*.uo`` and ``indexes/kb_graph.sqlite`` are not products.

    Without ``architecture``, only a unique arch among candidates is accepted.
    Multiple arches never return ``candidates[0]`` (arch22 would sort first).
    """
    from ascendc_codemap_mcp.engine.store.writer import uo_product_dir, uo_product_path

    root = Path(op_root).expanduser().resolve()
    if root.is_file() and root.suffix == ".uo":
        arch = (architecture or "").strip()
        if arch and not root.name.endswith(f".{arch}.uo"):
            return None
        if op_name and not root.name.startswith(f"{op_name}."):
            return None
        return root

    search_dirs: list[Path] = []
    arch = (architecture or "").strip()

    def _add_dir(path: Path) -> None:
        if path not in search_dirs:
            search_dirs.append(path)

    from ascendc_codemap_mcp.engine.source_layout import is_product_architecture

    if root.is_dir():
        if is_product_architecture(root.name):
            _add_dir(root)

    if op_name and arch:
        try:
            p = uo_product_path(root, op_name, arch)
            if p.is_file():
                return p
        except Exception:
            pass

    if arch:
        try:
            _add_dir(uo_product_dir(root, architecture=arch))
        except Exception:
            _add_dir(root / ".ascendc-codemap" / arch)
        _add_dir(root / ".ascendc-codemap" / arch)

    pilot = root / ".ascendc-codemap"
    if not pilot.is_dir() and is_product_architecture(root.name) and root.parent.name == ".ascendc-codemap":
        pilot = root.parent
        root = pilot.parent
    if not pilot.is_dir() and root.name == ".ascendc-codemap":
        pilot = root

    if pilot.is_dir():
        for child in sorted(pilot.iterdir()):
            if child.is_dir() and is_product_architecture(child.name):
                _add_dir(child)
        # Intentionally skip legacy top-level ``.ascendc-codemap/uo/``.

    candidates: list[Path] = []
    seen: set[Path] = set()
    for product_dir in search_dirs:
        if not product_dir.is_dir():
            continue
        for p in sorted(product_dir.glob("*.uo")):
            if p.is_file() and p not in seen:
                seen.add(p)
                candidates.append(p)

    if arch:
        narrowed = [c for c in candidates if c.name.endswith(f".{arch}.uo")]
        if narrowed:
            if op_name:
                for c in narrowed:
                    if c.name.startswith(f"{op_name}."):
                        return c
            if len(narrowed) == 1:
                return narrowed[0]
            # Same arch can leave both snake_case and CamelCase products after
            # discover() spelling changes. The newest commit is the authority.
            return max(narrowed, key=lambda p: p.stat().st_mtime)
        return None
    by_arch: dict[str, list[Path]] = {}
    for c in candidates:
        a = _architecture_from_uo_name(c)
        if not a:
            continue
        by_arch.setdefault(a, []).append(c)
    if len(by_arch) != 1:
        return None
    arch_candidates = next(iter(by_arch.values()))
    if op_name:
        for c in arch_candidates:
            if c.name.startswith(f"{op_name}."):
                return c
        return None
    if len(arch_candidates) == 1:
        return arch_candidates[0]
    return None


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text
