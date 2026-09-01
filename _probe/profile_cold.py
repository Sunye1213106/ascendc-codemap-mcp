# -*- coding: utf-8 -*-
from __future__ import annotations

import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROJECT = r"D:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"

from ascendc_codemap_mcp.service import query as query_mod  # noqa: E402
from ascendc_codemap_mcp.service.control import status  # noqa: E402

status(project=PROJECT, architecture="arch35")
pr = cProfile.Profile()
pr.enable()
query_mod.query(project=PROJECT, architecture="arch35", operation="trace", symbol="DoSparse")
pr.disable()
buf = StringIO()
pstats.Stats(pr, stream=buf).sort_stats("cumtime").print_stats(35)
print(buf.getvalue())
