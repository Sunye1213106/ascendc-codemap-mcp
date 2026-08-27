# -*- coding: utf-8 -*-
"""MacroPass — promote macros / defines to first-class CodeMap entities."""

from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

#: These macros are the ``-D`` flags of the build variant, read off the
#: BuildContext. Nothing is parsed or guessed, so the facts are derived.
_PROV = "build_variant"


def run(codemap: CodeMap, *, context: dict[str, Any] | None = None) -> CodeMap:
    ctx = context or {}
    build_variant = ctx.get("build_variant") or {}
    defines: list[str] = []
    for key in ("host_defines", "kernel_defines", "defines"):
        raw = build_variant.get(key) or ctx.get(key) or []
        if isinstance(raw, dict):
            defines.extend(f"{k}={v}" for k, v in raw.items())
        else:
            defines.extend(str(x) for x in raw)

    bv_name = str(
        build_variant.get("name")
        or build_variant.get("architecture")
        or codemap.architecture
        or "default"
    )
    bv = codemap.upsert(
        EntityKind.BUILD_VARIANT,
        bv_name,
        attrs={
            "architecture": codemap.architecture,
            **dict(build_variant),
            "provenance": _PROV,
        },
    )

    for item in defines:
        text = str(item).lstrip("-D")
        if not text:
            continue
        if "=" in text:
            name, value = text.split("=", 1)
        else:
            name, value = text, "1"
        macro = codemap.upsert(
            EntityKind.MACRO,
            name.strip(),
            attrs={
                "value": value.strip(),
                "definition": text,
                "layer": "compile",
                "provenance": _PROV,
            },
        )
        codemap.link(RelationKind.ACTIVE_UNDER, macro.id, bv.id, attrs={"provenance": _PROV})
        cvar = codemap.upsert(
            EntityKind.COMPILE_VAR,
            name.strip(),
            attrs={"value": value.strip(), "layer": "compile", "provenance": _PROV},
        )
        codemap.link(RelationKind.EXPANDS_TO, macro.id, cvar.id, attrs={"provenance": _PROV})
        codemap.link(RelationKind.BINDS, cvar.id, macro.id, attrs={"provenance": _PROV})

    codemap.meta["macro_pass"] = "v1"
    return codemap
