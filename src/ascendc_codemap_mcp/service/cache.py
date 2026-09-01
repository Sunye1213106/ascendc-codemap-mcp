# -*- coding: utf-8 -*-
"""Bounded LRU of open ``UoSqlQuery`` handles for a long-lived MCP process."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MAX_OPEN_CODEMAPS = int(os.environ.get("ASCENDC_CODEMAP_MAX_OPEN", "4") or 4)
HARD_MAX_OPEN = max(MAX_OPEN_CODEMAPS, int(os.environ.get("ASCENDC_CODEMAP_HARD_MAX_OPEN", "8") or 8))


class _Entry:
    __slots__ = ("query", "mtime_ns", "inuse")

    def __init__(self, query: Any, mtime_ns: int) -> None:
        self.query = query
        self.mtime_ns = mtime_ns
        self.inuse = 0


class QueryCache:
    """Service-level owner of query facades.

    SQLite connections live in ``engine.store.reader`` (thread-local per path).
    ``drop`` / ``close_all`` always close those handles so a later writer can
    replace the ``.uo`` and the next query cannot see snapshot N.
    """

    def __init__(self, max_open: int = MAX_OPEN_CODEMAPS) -> None:
        self.max_open = max(1, int(max_open))
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def stats(self) -> dict[str, int]:
        from ascendc_codemap_mcp.engine.store.reader import open_handle_count

        with self._lock:
            inuse = sum(1 for e in self._entries.values() if e.inuse)
            return {
                "cache_size": len(self._entries),
                "open_sqlite_handles": open_handle_count(),
                "inuse": inuse,
                "max_open_codemaps": self.max_open,
            }

    def drop(self, product: str | Path | None) -> None:
        if product is None:
            return
        key = str(Path(product).expanduser().resolve())
        with self._lock:
            entry = self._entries.pop(key, None)
        if entry is not None:
            self._close(entry)
        self._close_sqlite(key)

    def close_all(self) -> None:
        with self._lock:
            items = list(self._entries.items())
            self._entries.clear()
        for _, entry in items:
            self._close(entry)
        self._close_sqlite(None)

    def _close_sqlite(self, product: str | Path | None) -> None:
        from ascendc_codemap_mcp.engine.store.reader import close_uo_connections

        close_uo_connections(product)

    def _close(self, entry: _Entry) -> None:
        try:
            entry.query.close()
        except Exception:  # noqa: BLE001
            pass

    def _mtime(self, product: Path) -> int:
        try:
            return int(product.stat().st_mtime_ns)
        except OSError:
            return 0

    def _evict_unlocked(self) -> None:
        if len(self._entries) < self.max_open:
            return
        for key, entry in list(self._entries.items()):
            if entry.inuse:
                continue
            self._entries.pop(key, None)
            self._close(entry)
            if len(self._entries) < self.max_open:
                return
        # Never close an in-use facade. Grow past HARD_MAX until queries finish.
        for key, entry in list(self._entries.items()):
            if len(self._entries) < HARD_MAX_OPEN:
                return
            if entry.inuse:
                continue
            self._entries.pop(key, None)
            self._close(entry)

    def acquire(self, product: str | Path) -> Any:
        from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery
        from ascendc_codemap_mcp.engine.store.reader import lease_query_connection, mark_uo_in_use

        path = Path(product).expanduser().resolve()
        key = str(path)
        mark_uo_in_use(path)
        mtime = self._mtime(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.mtime_ns != mtime and entry.inuse == 0:
                self._entries.pop(key, None)
                self._close(entry)
                entry = None
            if entry is None:
                self._evict_unlocked()
                entry = _Entry(UoSqlQuery(path), mtime)
                self._entries[key] = entry
            else:
                self._entries.move_to_end(key)
            entry.inuse += 1
            query = entry.query
        try:
            lease_query_connection(path)
        except Exception:
            self.release(path)
            raise
        return query

    def release(self, product: str | Path) -> None:
        key = str(Path(product).expanduser().resolve())
        idle = False
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                from ascendc_codemap_mcp.engine.store.reader import release_query_connection

                release_query_connection(key)
                return
            if entry.inuse > 0:
                entry.inuse -= 1
            idle = entry.inuse == 0
        from ascendc_codemap_mcp.engine.store.reader import release_query_connection

        release_query_connection(key)
        if idle:
            from ascendc_codemap_mcp.engine.store.reader import mark_uo_idle

            mark_uo_idle(key)

    @contextmanager
    def open(self, product: str | Path) -> Iterator[Any]:
        query = self.acquire(product)
        try:
            yield query
        finally:
            self.release(product)
