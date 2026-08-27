# -*- coding: utf-8 -*-
"""Per-file CPU maps. Merge results on the calling thread (CodeMap is not shared)."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _requested_workers() -> int | None:
    """``UO_MAP_WORKERS`` override, or None to size the pool from the machine.

    ``1`` runs every item inline. The knob exists because the benefit here is
    not obvious and should not be assumed: Python's ``re`` holds the GIL, so a
    pool of eight threads scanning with regexes is one thread doing the work
    while seven pay for scheduling and lock handoffs. File reads *do* release
    it, and these callbacks mix reading with scanning in different proportions,
    so which way a given call site comes out is an empirical question. Being
    able to run the same build both ways is how it gets answered.
    """
    raw = str(os.environ.get("UO_MAP_WORKERS") or "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _default_map_backend() -> str:
    """Linux/WSL uses process pools to escape the GIL; Windows spawn is too costly."""
    if sys.platform.startswith("linux"):
        return "process"
    return "thread"


def _map_backend() -> str:
    """``thread`` or ``process``.

    Process pools escape the GIL for pure-Python regex scans, but Windows
    spawn costs ~0.5s per worker and the callback must pickle. Nested
    closures (most current call sites) fall back to threads automatically.
    Default: process on Linux/WSL, thread on Windows. Override with
    ``UO_MAP_BACKEND``.
    """
    raw = str(os.environ.get("UO_MAP_BACKEND") or "").strip().lower()
    if raw in {"process", "proc", "spawn"}:
        return "process"
    if raw in {"thread", "threads"}:
        return "thread"
    return _default_map_backend()


def map_files(
    items: Sequence[T] | Iterable[T], fn: Callable[[T], R], *, workers: int | None = None
) -> list[R]:
    """Apply ``fn`` per item. One item stays in-process; many files use a pool."""
    rows = list(items)
    if len(rows) <= 1:
        return [fn(row) for row in rows]
    override = _requested_workers()
    if override is not None:
        n = min(override, len(rows))
    elif workers is not None:
        n = workers
    else:
        n = min(len(rows), os.cpu_count() or 4, 8)
    if n <= 1:
        return [fn(row) for row in rows]
    n = max(2, n)
    if _map_backend() == "process":
        try:
            with ProcessPoolExecutor(max_workers=n) as pool:
                return list(pool.map(fn, rows))
        except Exception:
            # Nested closures, pickle failures, and Windows spawn errors must
            # not drop the work; the thread pool still returns the same list.
            pass
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(fn, rows))
