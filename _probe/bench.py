# -*- coding: utf-8 -*-
"""Measure query latency: single-shot and concurrent."""
from __future__ import annotations

import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROJECT = r"D:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"

from ascendc_codemap_mcp.service import query as query_mod  # noqa: E402
from ascendc_codemap_mcp.service.control import status  # noqa: E402

CASES = [
    ("search HIFLOAT8", dict(operation="search", pattern="DT_HIFLOAT8")),
    ("trace isBn2MultiBlk", dict(operation="trace", symbol="isBn2MultiBlk")),
    ("trace DoSparse", dict(operation="trace", symbol="DoSparse")),
    ("trace DoOpTiling→DoSparse", dict(operation="trace", symbol="DoOpTiling", to_symbol="DoSparse")),
    ("trace dim=IsRope=1", dict(operation="trace", dim="IsRope", value="1")),
    (
        "source DoSparse:1099",
        dict(
            operation="source",
            file="op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp",
            line=1099,
        ),
    ),
]


def _one(**kw) -> tuple[float, dict]:
    t0 = time.perf_counter()
    payload = query_mod.query(project=PROJECT, architecture="arch35", **kw)
    ms = (time.perf_counter() - t0) * 1000.0
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    timing = extra or payload
    return ms, {
        "ok": payload.get("ok"),
        "error": payload.get("error_code"),
        "server_ms": timing.get("server_ms"),
        "render_ms": timing.get("render_ms"),
        "chars": timing.get("response_chars"),
    }


def _stats(xs: list[float]) -> str:
    xs = sorted(xs)
    p50 = statistics.median(xs)
    p95 = xs[max(0, int(round(0.95 * (len(xs) - 1))))]
    return f"n={len(xs)} min={min(xs):.1f} p50={p50:.1f} p95={p95:.1f} max={max(xs):.1f} mean={statistics.mean(xs):.1f}"


def main() -> None:
    st = status(project=PROJECT, architecture="arch35")
    print("status", st.get("ok"), (st.get("codemap") or {}).get("path") or st.get("path"))
    print("--- warmup ---")
    for name, kw in CASES:
        ms, meta = _one(**kw)
        print(f"  {name:28s} {ms:7.1f} ms  ok={meta['ok']} err={meta['error']} server={meta['server_ms']} render={meta['render_ms']} chars={meta['chars']}")

    print("\n--- single, 8 repeats after warmup ---")
    for name, kw in CASES:
        xs = []
        for _ in range(8):
            ms, _ = _one(**kw)
            xs.append(ms)
        print(f"  {name:28s} {_stats(xs)}")

    mix = [kw for _, kw in CASES]
    print("\n--- concurrent mix ---")
    for workers in (1, 4, 8, 16):
        jobs = mix * max(1, 16 // len(mix))
        # First pass fills the connection pool; second is the steady state.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda kw: _one(**kw), jobs))
        t0 = time.perf_counter()
        xs: list[float] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, **kw) for kw in jobs]
            for fut in as_completed(futs):
                ms, _ = fut.result()
                xs.append(ms)
        wall = (time.perf_counter() - t0) * 1000.0
        print(f"  workers={workers:2d} wall={wall:7.1f} ms  per-call {_stats(xs)}")


if __name__ == "__main__":
    main()
