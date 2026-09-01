# -*- coding: utf-8 -*-
"""Print one rendered card. Usage: python _probe/show.py <op> k=v k=v ..."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROJECT = r"D:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"

from ascendc_codemap_mcp.service import query as query_mod  # noqa: E402

op = sys.argv[1]
kw: dict[str, object] = {"project": PROJECT, "architecture": "arch35", "operation": op}
for arg in sys.argv[2:]:
    k, _, v = arg.partition("=")
    kw[k] = int(v) if v.isdigit() else v

payload = query_mod.query(**kw)
data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
text = str(data.get("text") or "")
print(f"--- ok={payload.get('ok')} error={payload.get('error_code')} chars={len(text)} ---")
print(text or payload.get("error") or "<no text>")
