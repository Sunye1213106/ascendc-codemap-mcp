# -*- coding: utf-8 -*-
"""DataflowPass — def-use / lifecycle edges on CodeMap."""

from __future__ import annotations

from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.source_layout import host_ir_keeps_file

#: Writes and reads are read straight off the clang walk summaries.
_CLANG = "clang_walk"
#: FLOWS_TO is not in the summaries. It joins a write and a read on an equal
#: path spelling, which is a deterministic step over two clang facts rather
#: than a clang fact -- so it is derived, not authoritative.
_JOIN = "source_host_dataflow_join"


def run(codemap: CodeMap, *, context: dict[str, Any] | None = None) -> CodeMap:
    ctx = context or {}
    host_ir = ctx.get("host_ir")
    if host_ir is None:
        codemap.meta["dataflow_pass"] = "v1-skip"
        return codemap

    # Field write → FLOWS_TO consumers that read the same path tail.
    writes_by_path: dict[str, list[str]] = {}
    arch = str(codemap.architecture or ctx.get("architecture") or "")
    for ev in getattr(host_ir, "writes", None) or []:
        path = str(getattr(ev, "path", "") or "")
        fn = str(getattr(ev, "function", "") or "")
        ev_file = str(getattr(ev, "file", "") or "")
        if not path:
            continue
        if ev_file and not host_ir_keeps_file(ev_file, arch):
            continue
        field = codemap.upsert(
            EntityKind.FIELD, path, attrs={"layer": "host", "provenance": _CLANG}
        )
        if fn:
            writer = codemap.upsert(
                EntityKind.FUNCTION, fn, attrs={"layer": "host", "provenance": _CLANG}
            )
            codemap.link(
                RelationKind.WRITES, writer.id, field.id, attrs={"provenance": _CLANG}
            )
            writes_by_path.setdefault(path, []).append(writer.id)

    for name, summary in (getattr(host_ir, "summaries", None) or {}).items():
        reads = list(getattr(summary, "reads", None) or [])
        if not reads:
            continue
        summary_file = str(getattr(summary, "file", "") or "")
        if not summary_file or not host_ir_keeps_file(summary_file, arch):
            continue
        reader = codemap.upsert(
            EntityKind.FUNCTION, str(name), attrs={"layer": "host", "provenance": _CLANG}
        )
        for r in getattr(summary, "reads", None) or []:
            path = str(r)
            var = codemap.upsert(
                EntityKind.VARIABLE, path, attrs={"layer": "host", "provenance": _CLANG}
            )
            codemap.link(
                RelationKind.READS, reader.id, var.id, attrs={"provenance": _CLANG}
            )
            for writer_id in writes_by_path.get(path, ()):
                codemap.link(
                    RelationKind.FLOWS_TO,
                    writer_id,
                    reader.id,
                    attrs={"provenance": _JOIN},
                )
                field = codemap.by_name(path, kind=EntityKind.FIELD)
                if field:
                    codemap.link(
                        RelationKind.FLOWS_TO,
                        field[0].id,
                        var.id,
                        attrs={"provenance": _JOIN},
                    )

    codemap.meta["dataflow_pass"] = "v1"
    return codemap
