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
    def __init__(self, max_open: int = MAX_OPEN_CODEMAPS) -> None:
        self.max_open = max(1, int(max_open))
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def stats(self) -> dict[str, int]:
        with self._lock:
            inuse = sum(1 for e in self._entries.values() if e.inuse)
            return {
                "cache_size": len(self._entries),
                "open_sqlite_handles": len(self._entries),
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

    def close_all(self) -> None:
        with self._lock:
            items = list(self._entries.items())
            self._entries.clear()
        for _, entry in items:
            self._close(entry)

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
        while len(self._entries) >= HARD_MAX_OPEN:
            key, entry = next(iter(self._entries.items()))
            self._entries.pop(key, None)
            self._close(entry)

    def acquire(self, product: str | Path) -> Any:
        from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery

        path = Path(product).expanduser().resolve()
        key = str(path)
        mtime = self._mtime(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.mtime_ns != mtime:
                self._entries.pop(key, None)
                if entry.inuse <= 0:
                    self._close(entry)
                entry = None
            if entry is None:
                self._evict_unlocked()
                query = UoSqlQuery(path)
                entry = _Entry(query, mtime)
                self._entries[key] = entry
            else:
                self._entries.move_to_end(key)
            entry.inuse += 1
            return entry.query

    def release(self, product: str | Path) -> None:
        key = str(Path(product).expanduser().resolve())
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.inuse > 0:
                entry.inuse -= 1

    @contextmanager
    def open(self, product: str | Path) -> Iterator[Any]:
        query = self.acquire(product)
        try:
            yield query
        finally:
            self.release(product)
