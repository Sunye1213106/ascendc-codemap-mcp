# -*- coding: utf-8 -*-
"""KernelPass — kernel IR → KERNEL / BRANCH / AVAILABLE_ON edges."""

from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap


def run(codemap: CodeMap, *, context: dict[str, Any] | None = None) -> CodeMap:
    ctx = context or {}
    kernel_ir = ctx.get("kernel_ir")
    if kernel_ir is not None:
        mint = getattr(kernel_ir, "mint_ids", None)
        if callable(mint):
            mint(str(ctx.get("op_root") or ""))
        CodeMap.from_kernel_ir(
            kernel_ir,
            op_name=codemap.op_name or str(ctx.get("op_name") or ""),
            architecture=codemap.architecture,
            codemap=codemap,
            op_root=str(ctx.get("op_root") or ""),
        )
    codemap.meta["kernel_pass"] = "v1"
    return codemap
