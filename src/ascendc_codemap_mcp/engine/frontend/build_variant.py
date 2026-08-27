# -*- coding: utf-8 -*-
"""BuildVariantPass — architecture / macros / includes as a first-class unit."""

from __future__ import annotations
from ascendc_codemap_mcp.engine.paths import require_architecture

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BuildVariant:
    name: str
    architecture: str = ""
    soc: str = ""
    compiler_target: str = ""
    host_defines: list[str] = field(default_factory=list)
    kernel_defines: list[str] = field(default_factory=list)
    include_paths: list[str] = field(default_factory=list)
    dtype_variant: str = ""
    compile_flags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "soc": self.soc,
            "compiler_target": self.compiler_target,
            "host_defines": list(self.host_defines),
            "kernel_defines": list(self.kernel_defines),
            "include_paths": list(self.include_paths),
            "dtype_variant": self.dtype_variant,
            "compile_flags": list(self.compile_flags),
            **dict(self.extra),
        }


def build_variant_from_context(
    *,
    architecture: str = "",
    build_context: Any = None,
    dtype_variant: str = "",
    name: str = "",
) -> BuildVariant:
    """Lift ``BuildContext`` / YAML into a CodeMap BUILD_VARIANT."""
    arch = require_architecture(architecture)
    host_defs: list[str] = []
    kernel_defs: list[str] = []
    includes: list[str] = []
    flags: list[str] = []
    soc = ""
    target = ""

    ctx = build_context
    if ctx is not None:
        # Prefer BuildContext accessor methods (defines are dicts of name→value).
        if hasattr(ctx, "host_defines") and callable(ctx.host_defines):
            try:
                hd = ctx.host_defines() or {}
                host_defs = [f"{k}={v}" if v not in (None, "") else str(k) for k, v in dict(hd).items()]
            except Exception:
                host_defs = []
        if hasattr(ctx, "kernel_defines") and callable(ctx.kernel_defines):
            try:
                kd = ctx.kernel_defines() or {}
                kernel_defs = [f"{k}={v}" if v not in (None, "") else str(k) for k, v in dict(kd).items()]
            except Exception:
                kernel_defs = []
        if hasattr(ctx, "host_includes") and callable(ctx.host_includes):
            try:
                includes.extend(str(x) for x in (ctx.host_includes() or []))
            except Exception:
                pass
        if hasattr(ctx, "kernel_includes") and callable(ctx.kernel_includes):
            try:
                includes.extend(str(x) for x in (ctx.kernel_includes() or []))
            except Exception:
                pass
        if hasattr(ctx, "base_flags") and callable(getattr(ctx, "base_flags", None)):
            try:
                flags.extend(str(x) for x in (ctx.base_flags() or []))
            except Exception:
                pass
        elif hasattr(ctx, "raw") and isinstance(ctx.raw, dict):
            flags.extend(str(x) for x in (ctx.raw.get("base_flags") or []))

        host = getattr(ctx, "host", None) or {}
        kernel = getattr(ctx, "kernel", None) or {}
        if not host_defs and isinstance(host, dict):
            defs = host.get("defines") or {}
            if isinstance(defs, dict):
                host_defs = [f"{k}={v}" if v not in (None, "") else str(k) for k, v in defs.items()]
            else:
                host_defs = [str(x) for x in defs]
            includes.extend(str(x) for x in (host.get("include_paths") or host.get("includes") or []))
            flags.extend(str(x) for x in (host.get("flags") or []))
        if not kernel_defs and isinstance(kernel, dict):
            defs = kernel.get("defines") or {}
            if isinstance(defs, dict):
                kernel_defs = [f"{k}={v}" if v not in (None, "") else str(k) for k, v in defs.items()]
            else:
                kernel_defs = [str(x) for x in defs]
            includes.extend(str(x) for x in (kernel.get("include_paths") or kernel.get("includes") or []))
        soc = str(getattr(ctx, "soc", "") or "")
        target = str(getattr(ctx, "compiler_target", "") or getattr(ctx, "target", "") or "")
        if not target and hasattr(ctx, "raw") and isinstance(ctx.raw, dict):
            target = str(ctx.raw.get("target") or "")

    return BuildVariant(
        name=name or arch,
        architecture=arch,
        soc=soc,
        compiler_target=target,
        host_defines=host_defs,
        kernel_defines=kernel_defs,
        include_paths=includes,
        dtype_variant=dtype_variant,
        compile_flags=flags,
    )
