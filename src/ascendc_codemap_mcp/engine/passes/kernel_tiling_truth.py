# -*- coding: utf-8 -*-
"""Final truth normalization for selected TilingData source facts.

Two source distinctions matter after the individual extraction passes:

* a setter with no current-architecture TilingData receiver evidence is outside
  this TilingData domain (for example a shared Host file's legacy arch22 type),
  not an ambiguous arch35 write; and
* conditional members may mention several possible TilingData classes in their
  C++ type, so selected-type closure must retain every source-mentioned current
  TilingData type rather than only a simplified outer type.
"""
from __future__ import annotations

import re
from collections import deque

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

_WORD_RE = re.compile(r"\b[A-Za-z_]\w*\b")


def finalize_kernel_tiling_truth(codemap: CodeMap) -> CodeMap:
    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})

    # Remove writer diagnostics that never entered the current TilingData type
    # domain. They are usually shared-source legacy setters. A real ambiguity
    # must have at least one current-domain candidate owner/field.
    outside_domain: list[str] = []
    true_ambiguous: list[str] = []
    for ent in list(codemap.entities.values()):
        if str(ent.attrs.get("provenance") or "") != "source_tilingdata_host_write_unresolved":
            continue
        candidates = [x for x in (ent.attrs.get("candidate_fields") or []) if x]
        if candidates:
            true_ambiguous.append(ent.id)
        else:
            outside_domain.append(ent.id)
    for eid in outside_domain:
        # These nodes have no semantic field edge by construction. Fail closed
        # if a future change adds one rather than silently removing the target.
        if codemap.has_incident(eid):
            true_ambiguous.append(eid)
            continue
        codemap.entities.pop(eid, None)

    types = {e.name: e for e in codemap.by_kind(EntityKind.TILING_DATA)}
    type_names = set(types)
    selected = set(closure.get("tiling_registered_root_types") or ())
    queue = deque(selected)
    while queue:
        owner = queue.popleft()
        owner_ent = types.get(owner)
        if owner_ent is None:
            continue
        for rel, field in codemap.neighbors(
            owner_ent.id, kind=RelationKind.DECLARES, direction="out"
        ):
            if field is None or field.kind_name() != EntityKind.TILING_FIELD.value:
                continue
            referenced = set(_WORD_RE.findall(str(field.attrs.get("cpp_type") or ""))) & type_names
            for nested in referenced:
                if nested not in selected:
                    selected.add(nested)
                    queue.append(nested)

    # Mark every source-selected nested type explicitly; only root registrations
    # retain the direct TILING_DATA->KERNEL relation created by the registration
    # pass. Nested types are selection closure facts, not independent entry ABI.
    for name, ent in types.items():
        ent.attrs["selected_type_closure"] = name in selected

    closure.update(
        {
            "tiling_selected_type_closure": sorted(selected),
            "tiling_selected_type_count": len(selected),
            "tiling_outside_domain_writer_sites": len(outside_domain),
            "tiling_ambiguous_writer_sites": len(set(true_ambiguous)),
            "truth_policy": "current-domain-writer+conditional-type-closure/v1",
        }
    )
    codemap.meta["kernel_tiling_closure"] = closure
    return codemap
