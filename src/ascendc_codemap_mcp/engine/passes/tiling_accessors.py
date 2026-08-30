# -*- coding: utf-8 -*-
"""Link fileless set_/get_ callee stubs to the TILING_FIELD they name.

HostIR records `tilingData.set_deterMaxRound(...)` as a FUNCTION callee with no
declaration site. The real ABI location lives on the TILING_FIELD. This pass
adds a REFERENCES edge and `accessor_of` so query can jump to the field without
forging a fake source span on the stub.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

_ACCESSOR_RE = re.compile(r"^(?:set_|get_)([A-Za-z_]\w*)$")
_PROVENANCE = "source_tiling_accessor_stub"


def link_tiling_accessors(
    codemap: CodeMap,
    operator_root: str | Path = "",
    *,
    architecture: str = "",
) -> CodeMap:
    del operator_root, architecture
    located: dict[str, list] = defaultdict(list)
    for field in codemap.by_kind(EntityKind.TILING_FIELD):
        if str(field.file or "").strip() and int(field.line_start or 0) > 0:
            located[field.name].append(field)

    linked = 0
    skipped_ambiguous = 0
    for kind in (EntityKind.FUNCTION, EntityKind.METHOD):
        for stub in codemap.by_kind(kind):
            if str(stub.file or "").strip() and int(stub.line_start or 0) > 0:
                continue
            leaf = str(stub.name or "").split("::")[-1]
            match = _ACCESSOR_RE.match(leaf)
            if match is None:
                continue
            field_name = match.group(1)
            hits = located.get(field_name) or []
            if len(hits) != 1:
                if hits:
                    skipped_ambiguous += 1
                continue
            field = hits[0]
            stub.attrs["accessor_of"] = field_name
            stub.attrs.setdefault("provenance", _PROVENANCE)
            rel = codemap.link(
                RelationKind.REFERENCES,
                stub.id,
                field.id,
                attrs={
                    "provenance": _PROVENANCE,
                    "accessor_of": field_name,
                },
                status="confirmed",
            )
            rel.attrs["provenance"] = _PROVENANCE
            linked += 1

    stats = dict(codemap.meta.get("tiling_accessor_links") or {})
    stats.update({"linked": linked, "skipped_ambiguous": skipped_ambiguous})
    codemap.meta["tiling_accessor_links"] = stats
    return codemap
