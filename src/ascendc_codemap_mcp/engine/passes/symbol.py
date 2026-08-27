# -*- coding: utf-8 -*-
"""CoreCodeMapPass — symbol / call / read / write / control facts → CodeMap."""

from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind


def run(codemap: CodeMap, *, context: dict[str, Any] | None = None) -> CodeMap:
    ctx = context or {}
    host_ir = ctx.get("host_ir")
    if host_ir is not None:
        CodeMap.from_host_ir(
            host_ir,
            op_name=codemap.op_name,
            architecture=codemap.architecture,
            codemap=codemap,
        )
    # Surface INPUT entities from context.
    for name in ctx.get("inputs") or ():
        codemap.upsert(
            EntityKind.INPUT,
            str(name),
            attrs={"layer": "api", "provenance": "source_op_def"},
        )
    for name in ctx.get("outputs") or ():
        codemap.upsert(
            EntityKind.OUTPUT,
            str(name),
            attrs={"layer": "api", "provenance": "source_op_def"},
        )
    # Platform / arch node.
    if codemap.architecture:
        arch = codemap.upsert(
            EntityKind.ARCH,
            codemap.architecture,
            attrs={"provenance": "source_arch_file"},
        )
        for fn in codemap.by_kind(EntityKind.FUNCTION):
            if fn.attrs.get("reachable"):
                codemap.link(
                    RelationKind.ACTIVE_UNDER,
                    fn.id,
                    arch.id,
                    attrs={"provenance": "source_arch_file"},
                )
    codemap.meta["core_codemap_pass"] = "v1"
    return codemap
