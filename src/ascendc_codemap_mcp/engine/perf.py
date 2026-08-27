# -*- coding: utf-8 -*-
"""Process-local performance counters and a single TimeBudget.

One collector per uo-init run. Stages, analyze passes, source reads, pickle
loads and clang parses all land here so ``performance.yaml`` can explain wall
time instead of only five coarse phases.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_LOCK = threading.Lock()

_PASSES: dict[str, dict[str, Any]] = {}
_STAGES: dict[str, float] = {}
_COUNTERS: dict[str, Any] = {
    "read_text_count": 0,
    "read_text_bytes": 0,
    "read_text_cache_hits": 0,
    "pickle_load": 0,
    "pickle_deserialize": 0,
    "clang_tu_parse": 0,
    "regex_scan": 0,
    "ast_cursors": 0,
    "walk_bundle_loads": 0,
    "walk_bundle_hits": 0,
}
_FILE_READS: dict[str, int] = {}
_META: dict[str, Any] = {}
_T0 = time.perf_counter()


class TimeBudget:
    """Optional deadline for one pass. ``seconds <= 0`` means no deadline."""

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        if self.seconds <= 0:
            self.deadline = float("inf")
        else:
            self.deadline = time.perf_counter() + self.seconds

    def remaining(self) -> float:
        if self.deadline == float("inf"):
            return float("inf")
        return max(0.0, self.deadline - time.perf_counter())

    def expired(self) -> bool:
        return self.deadline != float("inf") and self.remaining() <= 0.0


def kernel_root_trace_budget_s(profile: str | None = None) -> float:
    """Seconds before kernel_root_trace may stop filling. ``0`` = no cutoff.

    Graph fill must not be truncated to hit a wall-clock target. Set
    ``UO_KERNEL_ROOT_TRACE_BUDGET_S`` only when an explicit cap is required.
    """
    del profile  # profile no longer picks a default cutoff
    override = str(os.environ.get("UO_KERNEL_ROOT_TRACE_BUDGET_S") or "").strip()
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    return 0.0


def reset() -> None:
    global _T0
    with _LOCK:
        _PASSES.clear()
        _STAGES.clear()
        _FILE_READS.clear()
        _META.clear()
        for key in list(_COUNTERS):
            _COUNTERS[key] = 0
        _T0 = time.perf_counter()


def set_meta(**kwargs: Any) -> None:
    with _LOCK:
        _META.update(kwargs)


def record_stage(name: str, seconds: float) -> None:
    with _LOCK:
        _STAGES[name] = round(float(seconds), 3)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time one uo-init stage, including the path that raises.

    A stage that fails still costs wall time, so the receipt records it either
    way; otherwise a crash in extract would silently reappear as unattributed
    time.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record_stage(name, time.perf_counter() - t0)


def elapsed_since_start() -> float:
    """Wall seconds since this process first imported the collector."""
    return time.perf_counter() - _T0


def record_pass(name: str, seconds: float | None = None, **extra: Any) -> None:
    with _LOCK:
        row = dict(_PASSES.get(name) or {})
        if seconds is not None:
            row["wall_s"] = round(float(seconds), 3)
        row.update({k: v for k, v in extra.items() if v is not None})
        _PASSES[name] = row


def reset_file_reads() -> None:
    """Isolate analyze-phase disk reads from prepare/extract."""
    with _LOCK:
        _FILE_READS.clear()
        _COUNTERS["read_text_count"] = 0
        _COUNTERS["read_text_bytes"] = 0
        _COUNTERS["read_text_cache_hits"] = 0
        _COUNTERS["regex_scan"] = 0


def bump(key: str, n: int = 1) -> None:
    with _LOCK:
        _COUNTERS[key] = int(_COUNTERS.get(key) or 0) + int(n)


def record_read(path: str, nbytes: int, *, cache_hit: bool) -> None:
    key = str(path or "").replace("\\", "/")
    with _LOCK:
        if cache_hit:
            _COUNTERS["read_text_cache_hits"] = int(_COUNTERS["read_text_cache_hits"]) + 1
        else:
            _COUNTERS["read_text_count"] = int(_COUNTERS["read_text_count"]) + 1
            _COUNTERS["read_text_bytes"] = int(_COUNTERS["read_text_bytes"]) + max(0, int(nbytes))
            _FILE_READS[key] = int(_FILE_READS.get(key) or 0) + 1


def snapshot() -> dict[str, Any]:
    with _LOCK:
        files = {
            path: {"read_count": n}
            for path, n in sorted(_FILE_READS.items(), key=lambda kv: (-kv[1], kv[0]))
            if n > 0
        }
        hot = {p: row for p, row in files.items() if int(row["read_count"]) > 3}
        stages = dict(_STAGES)
        wall = round(time.perf_counter() - _T0, 3)
        total = round(sum(float(v) for v in stages.values()), 3)
        if not total:
            total = wall
        bytes_n = int(_COUNTERS.get("read_text_bytes") or 0)
        return {
            "schema": "uo-performance/v1",
            "total_s": total,
            "wall_s": wall,
            "unattributed_s": round(max(0.0, wall - total), 3),
            "stages": stages,
            "passes": dict(_PASSES),
            "counters": {
                **dict(_COUNTERS),
                "read_text_mb": round(bytes_n / (1024 * 1024), 3),
            },
            "files": files,
            "hot_files": hot,
            "meta": dict(_META),
        }


def dump_yaml(path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    import yaml

    payload = snapshot()
    if extra:
        payload.update(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out
