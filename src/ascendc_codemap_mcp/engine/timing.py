# -*- coding: utf-8 -*-
"""Lightweight phase timing for uo-init extract.

Enable with ``UO_TIMING=1`` (default on). Any single phase over
``UO_PHASE_BUDGET_S`` (default 180) is flagged as ``SLOW`` — that is a
failed algorithm for that step, not something to wait out.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator


def timing_enabled() -> bool:
    raw = os.environ.get("UO_TIMING", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def phase_budget_s() -> float:
    try:
        return float(os.environ.get("UO_PHASE_BUDGET_S", "180"))
    except ValueError:
        return 180.0


def log(msg: str) -> None:
    if not timing_enabled():
        return
    sys.stderr.write(f"[uo-timing] {msg}\n")
    sys.stderr.flush()


@contextmanager
def span(name: str, **extra: Any) -> Iterator[dict[str, Any]]:
    """Time a named phase; always records, prints when timing is on."""
    row: dict[str, Any] = {"phase": name, **extra}
    t0 = time.perf_counter()
    try:
        yield row
    finally:
        dt = time.perf_counter() - t0
        row["seconds"] = round(dt, 3)
        budget = phase_budget_s()
        flag = " SLOW" if dt > budget else ""
        bits = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        log(f"{dt:7.3f}s{flag}  {name}" + (f"  {bits}" if bits else ""))


class PhaseTimer:
    """Accumulate phase rows for a receipt / return value."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, **extra: Any) -> Iterator[dict[str, Any]]:
        with span(name, **extra) as row:
            yield row
        self.rows.append(dict(row))

    def total(self) -> float:
        return round(time.perf_counter() - self.t0, 3)

    def summary(self) -> dict[str, Any]:
        budget = phase_budget_s()
        slow = [r for r in self.rows if float(r.get("seconds") or 0) > budget]
        return {
            "total_seconds": self.total(),
            "phase_budget_s": budget,
            "slow_phases": [r["phase"] for r in slow],
            "phases": list(self.rows),
        }
