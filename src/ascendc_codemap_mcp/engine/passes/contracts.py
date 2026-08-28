# -*- coding: utf-8 -*-
"""Index-time CONTRACT nodes: one per configuration seed + transport.

Identity is op + architecture + seed id + transport. Query-time closure and
completeness fill the card; this pass only mints the stable node and CONTAINS
the seed so explore does not invent contracts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

_SEED_KINDS = (
    EntityKind.TILING_FIELD,
    EntityKind.TILING_KEY,
    EntityKind.TEMPLATE_ARG,
    EntityKind.COMPILE_VAR,
)


def _transport_for(ent: Entity, codemap: CodeMap) -> str:
    kind = ent.kind_name()
    if kind == EntityKind.TILING_FIELD.value:
        return "tiling_data"
    if kind in {EntityKind.TILING_KEY.value, EntityKind.TEMPLATE_ARG.value}:
        return "dispatch"
    if kind == EntityKind.COMPILE_VAR.value:
        return "dispatch"
    for rel in codemap.relations.values():
        if rel.dst != ent.id:
            continue
        if rel.kind_name() == RelationKind.READS.value:
            src = codemap.entities.get(rel.src)
            if src is not None and src.kind_name() == EntityKind.BRANCH.value:
                return "control"
        if rel.attrs.get("role") == "conjunct":
            return "control"
    return "unknown"


def enrich_contracts(
    codemap: CodeMap,
    operator_root: str | Path = "",
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    op = str(codemap.op_name or "")
    minted = 0
    for kind in _SEED_KINDS:
        for ent in codemap.by_kind(kind):
            transport = _transport_for(ent, codemap)
            eid = f"CONTRACT::{op}::{arch}::{ent.id}::{transport}"
            contract = codemap.upsert(
                EntityKind.CONTRACT,
                ent.name,
                eid=eid,
                attrs={
                    "transport": transport,
                    "seed_id": ent.id,
                    "seed_kind": ent.kind_name(),
                    "architecture": arch,
                    "op_name": op,
                    "provenance": "source_contract_synth",
                },
                file=str(ent.file or ""),
                line=int(ent.line_start or 0),
                status="confirmed",
            )
            codemap.link(
                RelationKind.CONTAINS,
                contract.id,
                ent.id,
                attrs={"provenance": "source_contract_synth", "role": "seed"},
                status="confirmed",
            )
            minted += 1
    # Control seeds: flags that BRANCH reads.
    seen = {e.id for e in codemap.by_kind(EntityKind.CONTRACT)}
    for rel in list(codemap.relations.values()):
        if rel.kind_name() != RelationKind.READS.value:
            continue
        src = codemap.entities.get(rel.src)
        dst = codemap.entities.get(rel.dst)
        if src is None or dst is None:
            continue
        if src.kind_name() != EntityKind.BRANCH.value:
            continue
        if dst.kind_name() not in {
            EntityKind.VARIABLE.value,
            EntityKind.FIELD.value,
            EntityKind.TILING_FIELD.value,
            EntityKind.TILING_KEY.value,
            EntityKind.INPUT.value,
        }:
            continue
        eid = f"CONTRACT::{op}::{arch}::{dst.id}::control"
        if eid in seen:
            continue
        seen.add(eid)
        contract = codemap.upsert(
            EntityKind.CONTRACT,
            dst.name,
            eid=eid,
            attrs={
                "transport": "control",
                "seed_id": dst.id,
                "seed_kind": dst.kind_name(),
                "architecture": arch,
                "op_name": op,
                "provenance": "source_contract_synth",
            },
            file=str(dst.file or ""),
            line=int(dst.line_start or 0),
            status="confirmed",
        )
        codemap.link(
            RelationKind.CONTAINS,
            contract.id,
            dst.id,
            attrs={"provenance": "source_contract_synth", "role": "seed"},
            status="confirmed",
        )
        minted += 1
    codemap.meta["contract_count"] = minted
    return codemap
