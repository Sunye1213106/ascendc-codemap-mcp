# -*- coding: utf-8 -*-
"""Rebuild TPL materialized views from canonical CodeMap entities.

``TplSchemaPass`` persists every declared dimension as a TILING_KEY entity and
every ARGS_SEL group as a TEMPLATE entity. Those canonical entities contain
enough information to reconstruct the four TPL query views without trusting a
previously materialized blob or reparsing source.
"""

from __future__ import annotations

import itertools
from typing import Any

from ascendc_codemap_mcp.engine.ids import named_id
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind

TPL_VIEW_NAMES = (
    "tiling/tpl_schema.yaml",
    "tiling/template_blocks.yaml",
    "tiling/exhaustive_key_space.yaml",
    "tiling/legal_key_index.jsonl",
)


def _canonical_dims(codemap: CodeMap) -> tuple[list[dict[str, Any]], str]:
    keys = [
        e
        for e in codemap.by_kind(EntityKind.TILING_KEY)
        if bool(e.attrs.get("source_declared"))
        and str(e.attrs.get("provenance") or "") == "source_tpl_args_decl"
    ]
    keys.sort(key=lambda e: (int(e.attrs.get("decl_order") or 0), e.name, e.id))
    dims: list[dict[str, Any]] = []
    header = ""
    for ent in keys:
        attrs = ent.attrs or {}
        domain = [str(v) for v in (attrs.get("value_domain") or attrs.get("allowed_values") or [])]
        bw = int(attrs.get("bit_width") or attrs.get("bw") or 0)
        if not domain or bw <= 0:
            return [], ""
        bit_lo = int(
            attrs.get("bit_lo")
            if attrs.get("bit_lo") is not None
            else attrs.get("bit_offset") or 0
        )
        bit_hi = int(
            attrs.get("bit_hi")
            if attrs.get("bit_hi") is not None
            else bit_lo + bw - 1
        )
        dims.append(
            {
                "name": ent.name,
                "kind": str(attrs.get("decl_kind") or attrs.get("kind_tpl") or "UINT"),
                "bw": bw,
                "bit_lo": bit_lo,
                "bit_hi": bit_hi,
                "value_domain": domain,
            }
        )
        header = header or str(ent.file or "")
    return dims, header


def _canonical_templates(codemap: CodeMap) -> list[Entity]:
    rows = [
        e
        for e in codemap.by_kind(EntityKind.TEMPLATE)
        if str(e.attrs.get("tpl_role") or "") == "args_sel_group"
    ]
    rows.sort(
        key=lambda e: (
            int(e.attrs.get("sel_group_index") or 0),
            e.name,
            e.id,
        )
    )
    return rows


def _encode_dim(dim: dict[str, Any], raw: str) -> int:
    kind = str(dim.get("kind") or "UINT")
    domain = [str(v) for v in (dim.get("value_domain") or [])]
    if kind == "UINT":
        try:
            return domain.index(str(raw))
        except ValueError as exc:
            raise ValueError(f"{dim.get('name')}={raw!r} not in {domain}") from exc
    if kind == "BOOL":
        return 1 if raw in {"1", "true", "True"} else 0
    return int(raw)


def _encode_key(dims: list[dict[str, Any]], instance: dict[str, str]) -> int:
    key = 0
    shift = 0
    for dim in dims:
        domain = list(dim.get("value_domain") or [])
        raw = str(instance.get(str(dim["name"]), domain[0] if domain else "0"))
        encoded = _encode_dim(dim, raw)
        bw = int(dim.get("bw") or 0)
        key |= (encoded & ((1 << bw) - 1)) << shift
        shift += bw
    if shift > 64:
        raise ValueError(f"tiling key bits {shift} exceed 64")
    return key


def _template_block(template: Entity, dims: list[dict[str, Any]]) -> dict[str, Any] | None:
    attrs = template.attrs or {}
    fixed = attrs.get("fixed_fields") or {}
    domains = attrs.get("field_domains") or {}
    if not isinstance(fixed, dict) or not isinstance(domains, dict):
        return None
    dim_names = {str(d["name"]) for d in dims}
    fixed_out = {str(k): str(v) for k, v in fixed.items() if str(k) in dim_names}
    domains_out: dict[str, list[str]] = {}
    for key, raw in domains.items():
        name = str(key)
        if name not in dim_names:
            continue
        values = [str(v) for v in raw] if isinstance(raw, list) else [str(raw)]
        if values:
            domains_out[name] = values
    if not fixed_out and not domains_out:
        return None
    product = 1
    for values in domains_out.values():
        product *= max(1, len(values))
    index = int(attrs.get("sel_group_index") or 0)
    return {
        "id": named_id("TemplateBinding", f"sel{index}"),
        "name": template.name or f"ARGS_SEL_{index}",
        "fixed_fields": fixed_out,
        "field_domains": domains_out,
        "product_count": int(attrs.get("product_count") or product),
        "sel_group_index": index,
    }


def project_tpl_views_from_codemap(codemap: CodeMap) -> dict[str, Any]:
    """Return all four TPL views, or ``{}`` when canonical TPL facts are partial."""
    dims, header_ref = _canonical_dims(codemap)
    templates = _canonical_templates(codemap)
    if not dims or not templates:
        return {}

    blocks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    fallback = {str(d["name"]): str((d.get("value_domain") or ["0"])[0]) for d in dims}

    for template in templates:
        block = _template_block(template, dims)
        if block is None:
            continue
        blocks.append(block)
        fixed = dict(block["fixed_fields"])
        domains = dict(block["field_domains"])
        axes: list[tuple[str, list[str]]] = []
        sels: list[dict[str, Any]] = []
        for dim in dims:
            name = str(dim["name"])
            if name in fixed:
                values = [str(fixed[name])]
            elif name in domains:
                values = [str(v) for v in domains[name]]
            else:
                values = [fallback[name]]
            axes.append((name, values))
            if name in fixed or name in domains:
                sels.append(
                    {
                        "name": name,
                        "kind": str(dim.get("kind") or "UINT"),
                        "vals": values,
                    }
                )
        selections.append(
            {
                "sel_group_index": int(block["sel_group_index"]),
                "sels": sels,
            }
        )
        names = [name for name, _ in axes]
        for combo in itertools.product(*[values for _, values in axes]):
            full = dict(zip(names, combo))
            try:
                key = _encode_key(dims, full)
            except (ValueError, KeyError):
                continue
            rows.append(
                {
                    "index": len(rows),
                    "tiling_key": key,
                    "tiling_key_hex": f"0x{key:016x}",
                    "dims": full,
                    "sel_group_id": str(block["id"]),
                    "status": "template_admissible",
                }
            )

    meta = codemap.meta.get("tpl_schema") or {}
    op_tag = str(meta.get("op_tag") or codemap.op_name or "") if isinstance(meta, dict) else codemap.op_name
    return {
        "tiling/tpl_schema.yaml": {
            "schema": "uo-tpl-schema/v1",
            "op_tag": op_tag,
            "header": header_ref,
            "dims": dims,
            "selections": selections,
        },
        "tiling/template_blocks.yaml": {
            "schema": "uo-template-blocks/v1",
            "blocks": blocks,
            "count": len(blocks),
        },
        "tiling/exhaustive_key_space.yaml": {
            "schema": "uo-exhaustive-key-space/v1",
            "legal_key_count": len(rows),
            "legal_key_index": "tiling/legal_key_index.jsonl",
            "template_blocks": blocks,
            "header": header_ref,
            "status": "template_admissible",
        },
        "tiling/legal_key_index.jsonl": {
            "schema": "uo-legal-key-index/v1",
            "count": len(rows),
            "rows": rows,
        },
    }
