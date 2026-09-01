# -*- coding: utf-8 -*-
"""cProfile one warmed query."""
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

KW = dict(
    project=PROJECT,
    architecture="arch35",
    operation=sys.argv[1] if len(sys.argv) > 1 else "source",
)
if KW["operation"] == "source":
    KW.update(
        file="op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp",
        line=1099,
    )
elif KW["operation"] == "trace":
    KW["symbol"] = sys.argv[2] if len(sys.argv) > 2 else "isBn2MultiBlk"
elif KW["operation"] == "legal":
    KW["operation"] = "trace"
    KW.update(dim="IsRope", value="1")

# warmup
query_mod.query(**KW)
query_mod.query(**KW)

pr = cProfile.Profile()
pr.enable()
query_mod.query(**KW)
pr.disable()
buf = StringIO()
ps = pstats.Stats(pr, stream=buf).sort_stats("cumtime")
ps.print_stats(40)
print(buf.getvalue())
