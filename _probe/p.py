# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ascendc_codemap_mcp.service.query import query  # noqa: E402

CID = "p:462155::FlashAttentionScoreGrad@arch35"


def run(label: str, *, show: int = 0, **kw: object) -> None:
    out = query(codemap_id=CID, **kw)  # type: ignore[arg-type]
    d = out.get("data") if isinstance(out.get("data"), dict) else {}
    text = str(d.get("text") or "")
    print(
        json.dumps(
            {
                "label": label,
                "verdict": out.get("verdict"),
                "completeness": d.get("completeness"),
                "count": d.get("count"),
                "total": d.get("total"),
                "truncated": d.get("truncated"),
                "chars": len(text),
                "hint": str(d.get("hint") or "")[:200],
                "err": out.get("error_code"),
                "dym": [
                    " ".join(f"{k}={v}" for k, v in c.items())
                    for c in (out.get("did_you_mean") or [])
                ],
            },
            ensure_ascii=False,
        )
    )
    if show:
        print("\n".join(text.splitlines()[:show]))
    print("-" * 70)


run("A resolve name=*Buffer* (wrong op, glob value)", operation="resolve", name="*Buffer*")
run("B find name=BufferNum (miss)", operation="find", name="BufferNum", show=12)
run("C find name=*buf* (broad discovery)", operation="find", name="*buf*", show=40)
run("D find name=*BuffSelector*", operation="find", name="*BuffSelector*", show=12)
run("E resolve SetSplitAxis (long fn tail)", operation="resolve", symbol="SetSplitAxis", show=200)
