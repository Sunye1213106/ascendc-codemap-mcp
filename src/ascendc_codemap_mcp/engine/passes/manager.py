# -*- coding: utf-8 -*-
"""PassManager — run deterministic structural CodeMap analyze passes in order."""

from __future__ import annotations

from typing import Any, Callable

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.passes import (
    dataflow,
    host_kernel,
    kernel,
    macro,
    reachability,
    symbol,
    tpl_schema,
)

PassFn = Callable[..., CodeMap]

# Canonical structural analyze order (after BuildVariant + Clang frontend).
# Host API→packed-key roots are recovered later from *current source* by
# host_tiling_key + host_defuse. The retired ``input_root`` pass consumed
# derive_key_fields/host_derivation and therefore has no place in the new UO
# product pipeline.
ANALYZE_PASSES: list[tuple[str, PassFn]] = [
    ("reachability", reachability.run),
    ("core_codemap", symbol.run),
    ("macro", macro.run),
    ("tpl_schema", tpl_schema.run),
    ("dataflow", dataflow.run),
    ("kernel", kernel.run),
    ("host_kernel_bind", host_kernel.run),
]


class PassManager:
    def __init__(self, passes: list[tuple[str, PassFn]] | None = None) -> None:
        self.passes = list(passes or ANALYZE_PASSES)

    def run(self, codemap: CodeMap, *, context: dict[str, Any] | None = None) -> CodeMap:
        ctx = dict(context or {})
        ran: list[str] = []
        import time

        from ascendc_codemap_mcp.engine.perf import record_pass

        for name, fn in self.passes:
            t0 = time.perf_counter()
            codemap = fn(codemap, context=ctx)
            record_pass(name, time.perf_counter() - t0)
            ran.append(name)
        codemap.meta["passes_run"] = ran
        return codemap


def run_analyze_passes(
    codemap: CodeMap,
    *,
    context: dict[str, Any] | None = None,
) -> CodeMap:
    return PassManager().run(codemap, context=context)
