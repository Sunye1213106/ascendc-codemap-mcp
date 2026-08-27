# -*- coding: utf-8 -*-
"""Always-on stderr progress for long UO deterministic steps.

Distinct from ``uo_init.timing`` (which can be disabled via ``UO_TIMING=0``):
these lines exist so Host/users know the process is alive.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator


def emit(msg: str) -> None:
    sys.stderr.write(f"[ascendc-codemap] {msg}\n")
    sys.stderr.flush()


@contextmanager
def step(name: str, *, total: int | None = None, index: int | None = None) -> Iterator[None]:
    if total is not None and index is not None:
        head = f"({index}/{total}) {name}"
    else:
        head = name
    emit(f"{head} …")
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        emit(f"{head} FAIL ({time.perf_counter() - t0:.1f}s): {exc}"[:200])
        raise
    else:
        emit(f"{head} ok ({time.perf_counter() - t0:.1f}s)")
