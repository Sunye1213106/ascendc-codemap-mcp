# -*- coding: utf-8 -*-
"""BuildVariant only. Compiler facts come from host_ir / kernel_ir → clang_walk."""

from ascendc_codemap_mcp.engine.frontend.build_variant import BuildVariant, build_variant_from_context

__all__ = [
    "BuildVariant",
    "build_variant_from_context",
]
