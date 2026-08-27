# -*- coding: utf-8 -*-
"""Deterministic CodeMap passes.

Pass order (canonical, see manager.ANALYZE_PASSES):
  Reachability → CoreCodeMap → Macro → TplSchema → Dataflow → Kernel → HostKernelBind

Current-source enrichment (source_contract, tiling_*, kernel_root_trace) runs
in compile_codemap after this structural list.
"""

from ascendc_codemap_mcp.engine.passes.manager import PassManager, run_analyze_passes

__all__ = ["PassManager", "run_analyze_passes"]
