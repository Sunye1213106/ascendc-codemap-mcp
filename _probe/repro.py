# -*- coding: utf-8 -*-
"""Reproduce the ses_fa7c defects against the local .uo, and score them.

Run: python _probe/repro.py
Scores four invariants the OpenCode session violated:
  I1  primary entity of a symbol query == the requested symbol
  I2  a card that prints a body never claims "no definition either"
  I3  a known caller edge (DoOpTiling -> DoSparse @819) is never rendered absent
  I4  compiled legal keys for IsRope=1 are reachable and exclude FP32 / BN2S2
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROJECT = r"D:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"
ARCH = "arch35"

from ascendc_codemap_mcp.service import query as query_mod  # noqa: E402


def run(**kw):
    kw.setdefault("project", PROJECT)
    kw.setdefault("architecture", ARCH)
    payload = query_mod.query(**kw)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    return payload, str(data.get("text") or "")


def first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return "<empty>"


# The session's 40 resolve calls, all of this shape. A representative sample.
SPAN_CASES = [
    ("SetSplitAxis", "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp", 1657, 1719),
    ("DoSparse", "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp", 1077, 1149),
    ("DoOpTiling", "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp", 810, 880),
    ("DoPreTiling", "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp", 1, 0),
    ("GetDTemplateType", "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp", 1, 0),
]

BAD_PHRASE = "no definition either"
NOT_CALLED = "no resolved caller"


def main() -> int:
    print("=" * 78)
    print("I1/I2  symbol + file + line + line_end  (the session's only resolve shape)")
    print("=" * 78)
    i1_bad = i2_bad = 0
    for sym, file, line, line_end in SPAN_CASES:
        if line_end <= line:
            continue
        _, text = run(operation="resolve", symbol=sym, file=file, line=line, line_end=line_end)
        head = first_line(text)
        title_ok = sym.lower() in head.lower()
        body = "Source" in text
        claims_no_def = BAD_PHRASE in text
        i1_bad += 0 if title_ok else 1
        i2_bad += 1 if (body and claims_no_def) else 0
        print(f"  ask={sym:22} head={head[:56]:56} title_ok={title_ok} body={body} no_def={claims_no_def}")
    print(f"  -> I1 title mismatches: {i1_bad}/{len([c for c in SPAN_CASES if c[3] > c[2]])}")
    print(f"  -> I2 body+no_def violations: {i2_bad}")

    print()
    print("=" * 78)
    print("I3  DoSparse caller (graph truth: DoOpTiling @819)")
    print("=" * 78)
    for label, kw in (
        ("span   ", dict(operation="resolve", symbol="DoSparse",
                         file="op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp",
                         line=1077, line_end=1149)),
        ("symbol ", dict(operation="resolve", symbol="DoSparse")),
    ):
        _, text = run(**kw)
        has_caller = "DoOpTiling" in text and "819" in text
        print(f"  {label} says-no-caller={NOT_CALLED in text}  claims-no-def={BAD_PHRASE in text}  "
              f"shows-DoOpTiling@819={has_caller}  chars={len(text)}")

    print()
    print("=" * 78)
    print("I4  compiled legal keys, IsRope=1  (truth: 224, FP16/BF16 only, no BN2S2)")
    print("=" * 78)
    for label, kw in (
        ("dim+value ", dict(operation="resolve", dim="IsRope", value="1")),
        ("symbol    ", dict(operation="resolve", symbol="IsRope")),
    ):
        try:
            _, text = run(**kw)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label} RAISED {type(exc).__name__}: {exc}")
            continue
        print(f"  {label} chars={len(text)} has224={'224' in text} "
              f"hasDTemplate192={'192' in text} head={first_line(text)[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
