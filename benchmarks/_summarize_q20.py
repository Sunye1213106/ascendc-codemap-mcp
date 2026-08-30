# -*- coding: utf-8 -*-
import json
from pathlib import Path

p = Path(r"d:\TEST\ascendc-codemap-mcp\benchmarks\_uo_q20.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("legal_status", data["legal_status"])
for qid, q in data["questions"].items():
    print("\n====", qid, "====")
    print("tokens:", q.get("agent_tokens"))
    if "fts" in q:
        for f in q["fts"]:
            hits = f.get("hits") or []
            top = ""
            if hits:
                h = hits[0]
                top = f"{h['file'].split('/')[-1]}:{h['line']}"
            print(f"  FTS {f.get('q')!r} total={f.get('total')} err={f.get('error')} top={top}")
    if "names" in q:
        for k, rows in q["names"].items():
            kinds = [f"{r['kind']}:{r['name']}" for r in (rows or [])[:3]]
            print(f"  NAME {k}: n={len(rows or [])} {kinds}")
    if "edges" in q and q["edges"]:
        print("  EDGES", q["edges"][:4])
    if "edges_coreNum" in q:
        print("  coreNum edges", q["edges_coreNum"][:5])
    if "legal" in q:
        print("  LEGAL", q["legal"])
    if "legal_S1" in q:
        print("  S1", q["legal_S1"])
    if "legal_deter" in q:
        print("  DeterType", q["legal_deter"])
    if "tnd_deter_matrix" in q:
        print("  TND matrix", q["tnd_deter_matrix"])
    if "sel_groups" in q:
        print("  sel_groups", q["sel_groups"], "legal_n", q.get("legal_n"), "dims", q.get("dims"))
    if "sync_ops" in q:
        print("  sync", q["sync_ops"][:6])
