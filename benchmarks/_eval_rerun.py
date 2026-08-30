# -*- coding: utf-8 -*-
"""Replay the FAG/SFAG CANN-scenario queries used in the pre-fix measurement."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from ascendc_codemap_mcp.service.control import discover
from ascendc_codemap_mcp.service.query import query

FAG_P = r"D:\TEST\ops-transformer\attention\flash_attention_score_grad"
SFAG_P = r"D:\TEST\ops-transformer\attention\sparse_flash_attention_grad"
FAG = "p:462155::FlashAttentionScoreGrad@arch35"
SFAG = "p:3efbc1::SparseFlashAttentionGrad@arch35"
OUT = Path(__file__).resolve().parent / "_eval" / "rerun.json"


def dump_size(obj) -> tuple[int, int, int, int]:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    text = ""
    data = obj.get("data") if isinstance(obj, dict) else None
    if isinstance(data, dict):
        text = str(data.get("text") or "")
    return len(s), s.count(": null"), s.count("snippet"), len(text.encode("utf-8"))


def source_line_stats(uo: Path) -> dict:
    conn = sqlite3.connect(f"file:{uo}?mode=ro", uri=True)
    paths = [r[0] for r in conn.execute("SELECT DISTINCT path FROM source_line")]
    conn.close()
    return {
        "files": len(paths),
        "common": sum(1 for p in paths if p.startswith("../common/")),
        "fag_leak": any("flash_attention_score_grad" in p for p in paths),
        "has_pse_arch": any("pse_arch35" in p for p in paths),
    }


def summarize(name: str, payload: dict, t_ms: float) -> dict:
    json_n, nuls, snips, text_n = dump_size(payload)
    data = payload.get("data") or {}
    cards = data.get("cards") if isinstance(data.get("cards"), list) else []
    text = str(data.get("text") or "")
    kinds = [c.get("kind") for c in cards if isinstance(c, dict)]
    names = [c.get("name") for c in cards if isinstance(c, dict)]
    readers = []
    writers = []
    if cards and isinstance(cards[0], dict):
        readers = cards[0].get("readers") or []
        writers = cards[0].get("writers") or []
    leak = "flash_attention_score_grad" in text and "sparse" in name.lower()
    return {
        "case": name,
        "ok": payload.get("ok"),
        "verdict": payload.get("verdict"),
        "layer": payload.get("layer"),
        "completeness": data.get("completeness"),
        "unresolved": data.get("unresolved_reason") or payload.get("error_code") or data.get("empty_reason"),
        "json_bytes": json_n,
        "text_bytes": text_n,
        "nulls": nuls,
        "snippet_mentions": snips,
        "ms": round(t_ms, 1),
        "card_kinds": kinds,
        "card_names": names,
        "n_cards": len(cards),
        "n_readers": len(readers) if isinstance(readers, list) else 0,
        "n_writers": len(writers) if isinstance(writers, list) else 0,
        "has_pseValue": "pseValue" in text,
        "has_IS_PSE": "IS_PSE" in text,
        "has_QUEUE": "QUEUE" in kinds or any("Que" in str(n) for n in names),
        "sfag_leaks_fag": leak or ("flash_attention_score_grad/" in json.dumps(payload, default=str) and "SFAG" in name),
        "dim_coverage": data.get("dim_coverage"),
        "legal_key_count": data.get("legal_key_count"),
        "top_keys": list(payload.keys()),
        "text_head": text[:400],
    }


def run_case(name: str, fn) -> dict:
    t0 = time.perf_counter()
    payload = fn()
    return summarize(name, payload, (time.perf_counter() - t0) * 1000)


def main() -> None:
    discover(project=FAG_P, architecture="arch35")
    discover(project=SFAG_P, architecture="arch35")
    cases = [
        ("FAG index", lambda: query(codemap_id=FAG, limit=8)),
        ("FAG IsPse", lambda: query(codemap_id=FAG, symbol="IsPse")),
        ("FAG Dim=IsPse", lambda: query(codemap_id=FAG, dim="IsPse")),
        ("FAG workspace", lambda: query(codemap_id=FAG, symbol="workspace")),
        ("FAG SetFlag", lambda: query(codemap_id=FAG, symbol="SetFlag")),
        ("FAG TQue", lambda: query(codemap_id=FAG, symbol="TQue")),
        ("FAG TBuf", lambda: query(codemap_id=FAG, symbol="TBuf")),
        ("FAG IsPse by alias only", lambda: query(codemap_id="FlashAttentionScoreGrad@arch35", symbol="IsPse")),
        ("SFAG IsPse", lambda: query(codemap_id=SFAG, symbol="IsPse")),
        ("SFAG IsRope", lambda: query(codemap_id=SFAG, symbol="IsRope")),
        ("SFAG Deterministic", lambda: query(codemap_id=SFAG, symbol="Deterministic")),
        ("SFAG workspace", lambda: query(codemap_id=SFAG, symbol="workspace")),
    ]
    rows = [run_case(name, fn) for name, fn in cases]
    fag_uo = Path(FAG_P) / ".ascendc-codemap" / "arch35" / "FlashAttentionScoreGrad.arch35.uo"
    sfag_uo = Path(SFAG_P) / ".ascendc-codemap" / "arch35" / "SparseFlashAttentionGrad.arch35.uo"
    report = {
        "source_line": {
            "FAG": source_line_stats(fag_uo),
            "SFAG": source_line_stats(sfag_uo),
        },
        "cases": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'case':28} {'verdict':10} {'comp':12} {'jsonKB':7} {'textKB':7} {'null':5} {'cards':5} note")
    for r in rows:
        note = r.get("unresolved") or ""
        if r["has_pseValue"]:
            note = (note + " pseValue").strip()
        if r["has_IS_PSE"]:
            note = (note + " IS_PSE").strip()
        if r["has_QUEUE"]:
            note = (note + " QUEUE").strip()
        if r["sfag_leaks_fag"]:
            note = (note + " LEAK").strip()
        print(
            f"{r['case']:28} {str(r['verdict'] or ''):10} {str(r['completeness'] or ''):12} "
            f"{r['json_bytes']/1024:7.1f} {r['text_bytes']/1024:7.1f} {r['nulls']:5} {r['n_cards']:5} {note[:40]}"
        )
    print("source_line", json.dumps(report["source_line"], ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
