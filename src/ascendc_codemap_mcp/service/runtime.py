# -*- coding: utf-8 -*-
"""Process-wide CodeMap service runtime (registry, cache, locks, flags)."""
from __future__ import annotations

import threading

from ascendc_codemap_mcp.service.cache import QueryCache
from ascendc_codemap_mcp.service.identity import Registry
from ascendc_codemap_mcp.service.locks import SnapshotLocks

registry = Registry()
cache = QueryCache()
locks = SnapshotLocks()

_FLAG_LOCK = threading.Lock()
_BUILDING: set[str] = set()
_BLOCKED: set[str] = set()
_CANCEL: dict[str, threading.Event] = {}


def mark_building(codemap_id: str) -> None:
    with _FLAG_LOCK:
        _BUILDING.add(codemap_id)


def clear_building(codemap_id: str) -> None:
    with _FLAG_LOCK:
        _BUILDING.discard(codemap_id)


def is_building(codemap_id: str) -> bool:
    if locks.is_writing(codemap_id):
        return True
    with _FLAG_LOCK:
        return codemap_id in _BUILDING


def mark_blocked(codemap_id: str) -> None:
    with _FLAG_LOCK:
        _BLOCKED.add(codemap_id)


def clear_blocked(codemap_id: str) -> None:
    with _FLAG_LOCK:
        _BLOCKED.discard(codemap_id)


def is_blocked(codemap_id: str) -> bool:
    with _FLAG_LOCK:
        return codemap_id in _BLOCKED


def cancel_event(codemap_id: str) -> threading.Event:
    with _FLAG_LOCK:
        ev = _CANCEL.get(codemap_id)
        if ev is None:
            ev = threading.Event()
            _CANCEL[codemap_id] = ev
        return ev


def request_cancel(codemap_id: str) -> None:
    cancel_event(codemap_id).set()


def clear_cancel(codemap_id: str) -> None:
    with _FLAG_LOCK:
        ev = _CANCEL.pop(codemap_id, None)
    if ev is not None:
        ev.set()


def cache_stats() -> dict[str, int]:
    return cache.stats()


def shutdown() -> None:
    cache.close_all()
    with _FLAG_LOCK:
        _BUILDING.clear()
        _BLOCKED.clear()
        events = list(_CANCEL.values())
        _CANCEL.clear()
    for ev in events:
        ev.set()
