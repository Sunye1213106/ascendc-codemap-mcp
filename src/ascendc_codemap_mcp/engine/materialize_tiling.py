# -*- coding: utf-8 -*-
"""TPL template blocks and legal-key rows (no KnowledgeBase landing)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ids import named_id, slug
from ascendc_codemap_mcp.engine.kb_model import CONTROLLABLE_ROOTS, STATUS_EXTRACTED, STATUS_PARTIAL
from ascendc_codemap_mcp.engine.tpl_bind import BindingResult
from ascendc_codemap_mcp.engine.tpl_dsl import TplSchema, expand_legal_instances, parse_file
from ascendc_codemap_mcp.engine.variable_model import VariableModel

KEY_REACHABLE = "reachable"
KEY_UNREACHABLE = "unreachable"
KEY_UNKNOWN = "unknown"
KEY_UNDERIVABLE = "underivable"
LAYER_TEMPLATE = "template"
LAYER_NOT_COMPUTED = "not_computed"

REASON_OK = "OK"
REASON_BIND_INCOMPLETE = "BIND_INCOMPLETE"
REASON_NOT_INPUT_DERIVABLE = "NOT_INPUT_DERIVABLE"
REASON_PREDICATE_UNRESOLVED = "PREDICATE_UNRESOLVED"
REASON_REALIZATION_MISSING = "REALIZATION_MISSING"
REASON_DOMAIN_OPEN = "DOMAIN_OPEN"
REASON_HOST_ENCODE_CONFLICT = "HOST_ENCODE_CONFLICT"
REASON_HOST_UNREACHABLE = "HOST_UNREACHABLE"
REASON_HOST_UNKNOWN = "HOST_UNKNOWN"

REASON_CODES = frozenset(
    {
        REASON_OK,
        REASON_BIND_INCOMPLETE,
        REASON_NOT_INPUT_DERIVABLE,
        REASON_PREDICATE_UNRESOLVED,
        REASON_REALIZATION_MISSING,
        REASON_DOMAIN_OPEN,
        REASON_HOST_ENCODE_CONFLICT,
        REASON_HOST_UNREACHABLE,
        REASON_HOST_UNKNOWN,
    }
)


@dataclass
class TemplateBlock:
    id: str
    name: str
    fixed_fields: dict[str, str]
    field_domains: dict[str, list[str]]
    product_count: int
    sel_group_index: int
    line_start: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "fixed_fields": dict(self.fixed_fields),
            "field_domains": {k: list(v) for k, v in self.field_domains.items()},
            "product_count": self.product_count,
            "sel_group_index": self.sel_group_index,
            "line_start": int(self.line_start or 0),
        }


@dataclass
class LegalKeyRow:
    index: int
    tiling_key: int
    tiling_key_hex: str
    dims: dict[str, str]
    sel_group_id: str
    reason_code: str = REASON_OK
    #: Default is the weakest status, not the strongest. A row that nobody
    #: classified must not read as "a host run produces this".
    status: str = KEY_UNKNOWN
    detail: str = ""
    blocker_ids: list[str] = field(default_factory=list)
    #: Which check produced the status.
    layer: str = LAYER_TEMPLATE

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "tiling_key": self.tiling_key,
            "tiling_key_hex": self.tiling_key_hex,
            "dims": dict(self.dims),
            "sel_group_id": self.sel_group_id,
            "reason_code": self.reason_code,
            "status": self.status,
            "detail": self.detail,
            "blocker_ids": list(self.blocker_ids),
            "layer": self.layer,
        }
        return out


def _sel_domain(sel: dict[str, Any]) -> list[str]:
    from ascendc_codemap_mcp.engine.tpl_dsl import canonicalize_sel_vals

    vals = list(sel.get("vals") or [])
    if vals and ("UI_LIST" in str(vals[0]) or "UI_RANGE" in str(vals[0])):
        vals = vals[1:]
    return canonicalize_sel_vals(str(sel.get("kind") or ""), [str(v) for v in vals])


def build_template_blocks(schema: TplSchema) -> list[TemplateBlock]:
    blocks: list[TemplateBlock] = []
    for gi, group in enumerate(schema.selections):
        fixed: dict[str, str] = {}
        domains: dict[str, list[str]] = {}
        product = 1
        for sel in group:
            name = str(sel["name"])
            domain = _sel_domain(sel)
            if not domain:
                continue
            if len(domain) == 1:
                fixed[name] = domain[0]
            else:
                domains[name] = domain
                product *= len(domain)
        if not fixed and not domains:
            continue
        if product < 1:
            product = 1
        bid = named_id("TemplateBinding", f"sel{gi}")
        line_start = 0
        for sel in group:
            line_start = int(sel.get("line") or 0)
            if line_start:
                break
        blocks.append(
            TemplateBlock(
                id=bid,
                name=f"ARGS_SEL_{gi}",
                fixed_fields=fixed,
                field_domains=domains,
                product_count=product,
                sel_group_index=gi,
                line_start=line_start,
            )
        )
    return blocks


def expand_legal_with_groups(schema: TplSchema) -> list[tuple[int, dict[str, str]]]:
    """Return (sel_group_index, dims) for every legal instance."""
    import itertools

    out: list[tuple[int, dict[str, str]]] = []
    for gi, group in enumerate(schema.selections):
        axes: list[tuple[str, list[str]]] = []
        for sel in group:
            axes.append((str(sel["name"]), _sel_domain(sel)))
        if not axes:
            continue
        names = [a[0] for a in axes]
        for combo in itertools.product(*[a[1] for a in axes]):
            out.append((gi, dict(zip(names, combo))))
    return out


def _bind_complete(binding: BindingResult | None, schema: TplSchema) -> tuple[bool, str]:
    if binding is None:
        return False, "tpl_bind missing"
    if not binding.bindings:
        return False, "tpl_bind empty"
    bound = {b.decl.name for b in binding.bindings}
    missing = [d.name for d in schema.dims if d.name not in bound]
    if missing:
        return False, "unbound dims: " + ",".join(missing[:8])
    return True, ""


def _legal_key_status(
    dims: dict[str, str],
    schema: TplSchema,
    *,
    bind_ok: bool,
    bind_detail: str,
) -> tuple[str, str, str, str]:
    """Return (status, reason_code, detail, layer) for a legal key row."""

    if not bind_ok:
        return KEY_UNDERIVABLE, REASON_NOT_INPUT_DERIVABLE, bind_detail, LAYER_TEMPLATE
    for dim in schema.dims:
        val = dims.get(dim.name)
        if val is None:
            return KEY_UNDERIVABLE, REASON_NOT_INPUT_DERIVABLE, f"missing {dim.name}", LAYER_TEMPLATE
        if str(val) not in [str(x) for x in dim.value_domain]:
            return (
                KEY_UNDERIVABLE,
                REASON_NOT_INPUT_DERIVABLE,
                f"{dim.name}={val} not in domain",
                LAYER_TEMPLATE,
            )
    return (
        KEY_UNKNOWN,
        REASON_HOST_UNKNOWN,
        "host reachability is not computed by UO; TG closes it with replay or reviewed evidence",
        LAYER_NOT_COMPUTED,
    )


def build_legal_key_rows(
    schema: TplSchema,
    *,
    binding: BindingResult | None = None,
    blocker_ids: Iterable[str] = (),
) -> list[LegalKeyRow]:
    blockers = list(blocker_ids)
    bind_ok, bind_detail = _bind_complete(binding, schema)
    rows: list[LegalKeyRow] = []
    for idx, (gi, dims) in enumerate(expand_legal_with_groups(schema)):
        full = {d.name: str(dims.get(d.name, d.value_domain[0])) for d in schema.dims}
        key = schema.encode_tiling_key(full)
        status, reason, detail, layer = _legal_key_status(
            full,
            schema=schema,
            bind_ok=bind_ok,
            bind_detail=bind_detail,
        )
        rows.append(
            LegalKeyRow(
                index=idx,
                tiling_key=key,
                tiling_key_hex=f"0x{key:016x}",
                dims=full,
                sel_group_id=named_id("TemplateBinding", f"sel{gi}"),
                reason_code=reason,
                status=status,
                detail=detail,
                blocker_ids=blockers if reason == REASON_PREDICATE_UNRESOLVED else [],
                layer=layer,
            )
        )
    return rows

def write_legal_key_index(uo_root: Path, legal_keys: list[dict[str, Any]]) -> Path:
    path = uo_root / "tiling" / "legal_key_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in legal_keys:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def write_key_index(uo_root: Path, fields: list[dict[str, Any]]) -> Path:
    """Lightweight closure-facing index (~8% of derive payload).

    Carries def_sites / status / exactness / value_leaves / input_roots.
    """
    import yaml

    root = Path(uo_root)
    rows = []
    for f in fields or []:
        if not isinstance(f, dict):
            continue
        rows.append({
            "name": f.get("name"),
            "index": f.get("index"),
            "status": f.get("status"),
            "exactness": f.get("exactness"),
            "value_leaves": list(f.get("value_leaves") or []),
            "input_roots": list(f.get("input_roots") or []),
            "def_sites": list(f.get("def_sites") or []),
            "free_vars": list(f.get("free_vars") or []),
        })
    path = root / "tiling" / "key_index.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {"schema": "uo-key-index/v1", "fields": rows},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


