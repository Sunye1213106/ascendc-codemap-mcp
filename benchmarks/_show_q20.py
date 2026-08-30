# -*- coding: utf-8 -*-
"""Print compact evidence from Q1-Q20 run JSON."""
from __future__ import annotations

import json
from pathlib import Path


def show_fts(fts_list, max_win=1):
    for item in fts_list or []:
        print(f"  FTS {item.get('q')!r} total={item.get('total')} err={item.get('error')}")
        for h in (item.get("hits") or [])[:3]:
            print(f"    hit {h['file']}:{h['line']}: {h['text'][:110]}")
        for w in (item.get("windows") or [])[:max_win]:
            print(f"    WINDOW {w['at']}")
            for line in w.get("window") or []:
                print("     ", line[:140])


def show_q(qid, q):
    print("\n" + "=" * 72)
    print(qid, "START", q.get("start"))
    show_fts(q.get("fts"), 1)
    if q.get("kernel_gate"):
        print("  kernel_gate:")
        show_fts([q["kernel_gate"]], 1)
    if q.get("host_can_emit"):
        print("  host_can_emit:")
        show_fts([q["host_can_emit"]], 1)
    if q.get("entry"):
        print("  entry:")
        show_fts([q["entry"]], 1)
    for k in ("names", "legal", "legal_S1", "legal_fp32_d", "legal_drop", "legal_tnd", "legal_456", "matrix", "sync_files", "dims", "sel_groups", "legal_n", "template_block_dtype"):
        if k in q and q[k]:
            val = q[k]
            if isinstance(val, list) and len(val) > 12:
                print(f"  {k}: {val[:12]} ...")
            else:
                print(f"  {k}: {val}")
    if q.get("edges"):
        print("  edges:")
        for e in q["edges"][:6]:
            print("   ", e)
    for ek in ("edges_coreNum", "edges_aicNum", "edges_IsDrop", "edges_dsink", "edges_isSeq"):
        if q.get(ek):
            print(f"  {ek}:")
            for e in q[ek][:6]:
                print("   ", e)
    if q.get("apt_empty_lines"):
        print("  apt:")
        for line in q["apt_empty_lines"][:12]:
            print("   ", line[:140])
    if q.get("tiling_cpp_1_180"):
        # only print lines mentioning empty
        print("  tiling.cpp empty-related:")
        for line in q["tiling_cpp_1_180"]:
            low = line.lower()
            if any(x in low for x in ("empty", "numel", "dqnum", "return", "tilingkey", "op_tiling")):
                print("   ", line[:140])


def main():
    a = json.loads(Path(r"d:\TEST\ascendc-codemap-mcp\benchmarks\_run_q20.json").read_text(encoding="utf-8"))
    b_path = Path(r"d:\TEST\ascendc-codemap-mcp\benchmarks\_run_q20_b.json")
    b = json.loads(b_path.read_text(encoding="utf-8")) if b_path.exists() else {}
    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        show_q(qid, a[qid])
    for qid in [f"Q{i}" for i in range(6, 21)]:
        if qid in b:
            show_q(qid, b[qid])


if __name__ == "__main__":
    main()
