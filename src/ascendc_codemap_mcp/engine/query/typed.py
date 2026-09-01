# -*- coding: utf-8 -*-
"""Typed structured query: operation enum + closed filter set.

The agent plans; this module validates and runs a deterministic graph lookup.
Illegal filters become INVALID_QUERY. There is no NL / phenomenon fallback.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.consumer_role import CONSUMER_ROLES
from ascendc_codemap_mcp.engine.query.completeness import COMPLETE, UNKNOWN
from ascendc_codemap_mcp.engine.query.contract import (
    PUBLIC_OPERATIONS,
    RELATION_FAMILY_NAMES,
    SEARCH_FILTERS,
    SEARCH_PATTERN_ALIASES,
    expand_relation,
)
from ascendc_codemap_mcp.engine.query.predicate_ast import OPERATORS as AST_OPERATORS

OPERATIONS = (
    "resolve",
    "contract",
    "impact",
    "entry",
    "find",
    "search",
    "trace",
    "source",
)
PROJECTIONS = ("summary", "source", "locations")
LAYERS = ("host", "kernel", "tiling", "template", "arch")
ENTRY_ROLES = ("bailout", "guard_clause", "then_body")

FIND_KINDS = (
    EntityKind.BRANCH.value,
    EntityKind.OPERATION.value,
    EntityKind.PREDICATE.value,
    EntityKind.TILING_FIELD.value,
    EntityKind.TILING_KEY.value,
    EntityKind.FUNCTION.value,
    EntityKind.METHOD.value,
    EntityKind.KERNEL.value,
    EntityKind.MACRO.value,
    EntityKind.COMPILE_VAR.value,
    EntityKind.BUFFER.value,
    EntityKind.QUEUE.value,
    EntityKind.PIPE.value,
    EntityKind.EVENT.value,
    EntityKind.REGISTER.value,
    EntityKind.TYPE.value,
)

FILTER_KEYS = (
    "symbol",
    "name",
    "pattern",
    "entity_id",
    "file",
    "line",
    "line_end",
    "kind",
    "layer",
    "callee",
    "referenced_symbol",
    "referenced_value",
    "literal",
    "operator",
    "dim",
    "value",
    "relation",
    "consumer_role",
    "from_symbol",
    "to_symbol",
    "entry_role",
    "function",
)

_AST_FILTERS = frozenset({"literal", "operator", "referenced_value"})

_RESOLVE_FILTERS = frozenset(
    {"symbol", "entity_id", "file", "line", "line_end", "kind", "dim", "value"}
)
_CONTRACT_FILTERS = _RESOLVE_FILTERS
_IMPACT_FILTERS = frozenset({"symbol", "entity_id", "file", "line", "kind"})
_ENTRY_FILTERS = frozenset({"layer", "entry_role", "function", "referenced_symbol"})
# trace is symbol space. It never takes file/line: a caller who has both was
# answered about the line and told the symbol's name, which is the defect this
# split exists to remove.
_TRACE_FILTERS = frozenset(
    {"symbol", "entity_id", "from_symbol", "to_symbol", "relation", "kind", "dim", "value"}
)
#: What the MCP tool actually exposes. `entity_id` / `from_symbol` still work
#: on the engine, but listing them on INVALID_QUERY advertised filters the
#: caller cannot send.
_TRACE_ADVERTISED_FILTERS = frozenset(
    {"symbol", "to_symbol", "relation", "kind", "dim", "value"}
)
_DIM_CATALOG = frozenset({"*", "all", "dims"})
# source is line space. It never takes a symbol, so nothing can silently
# outrank the range the caller asked to read.
_SOURCE_FILTERS = frozenset({"file", "line", "line_end", "kind"})
_SEARCH_FILTERS = frozenset(SEARCH_FILTERS)
# `pattern` is regex over indexed source lines. `name` is a silent alias
# (SEARCH_PATTERN_ALIASES) and is normalized before the illegal-filter check.
_FIND_COMMON = frozenset({"kind", "layer", "function", "name"})
_FIND_BY_KIND: dict[str, frozenset[str]] = {
    EntityKind.BRANCH.value: _FIND_COMMON
    | {"referenced_symbol", "referenced_value", "literal", "operator"},
    EntityKind.OPERATION.value: _FIND_COMMON | {"callee"},
    EntityKind.PREDICATE.value: _FIND_COMMON
    | {"entry_role", "referenced_symbol", "referenced_value", "literal", "operator"},
    EntityKind.TILING_FIELD.value: _FIND_COMMON | {"dim", "consumer_role"},
    EntityKind.TILING_KEY.value: _FIND_COMMON | {"dim", "value"},
    EntityKind.FUNCTION.value: _FIND_COMMON | {"callee"},
    EntityKind.METHOD.value: _FIND_COMMON | {"callee"},
    EntityKind.KERNEL.value: _FIND_COMMON,
    EntityKind.MACRO.value: _FIND_COMMON | {"callee"},
    EntityKind.COMPILE_VAR.value: _FIND_COMMON | {"dim"},
    EntityKind.BUFFER.value: _FIND_COMMON,
    EntityKind.QUEUE.value: _FIND_COMMON,
    EntityKind.PIPE.value: _FIND_COMMON,
    EntityKind.EVENT.value: _FIND_COMMON,
    EntityKind.REGISTER.value: _FIND_COMMON,
    EntityKind.TYPE.value: _FIND_COMMON,
}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUAL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+$")
_WS_RE = re.compile(r"\s")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Identifier filters that can be rebound onto each other. `dim` is a declared
# tiling-key name, not a generic ident slot — rebinding symbol/name onto dim=
# produces a ready-made call the agent will fire and miss.
_IDENT_VALUED = (
    "symbol",
    "callee",
    "referenced_symbol",
    "from_symbol",
    "to_symbol",
    "function",
    "name",
)
_MAX_SUGGESTIONS = 3
_GRAPH_ID_RE = re.compile(r"^(?P<prefix>[A-Z][A-Z0-9]*)_[A-Za-z0-9_]+$")

_OP_FILTERS = {
    "resolve": _RESOLVE_FILTERS,
    "contract": _CONTRACT_FILTERS,
    "impact": _IMPACT_FILTERS,
    "entry": _ENTRY_FILTERS,
    "trace": _TRACE_FILTERS,
    "source": _SOURCE_FILTERS,
    "search": _SEARCH_FILTERS,
}

#: Operations whose seed is a name rather than a location.
_SYMBOL_OPS = frozenset({"resolve", "contract", "impact", "trace"})


@dataclass
class QueryPlan:
    operation: str
    projection: str = "summary"
    symbol: str = ""
    name: str = ""
    pattern: str = ""
    entity_id: str = ""
    file: str = ""
    line: int = 0
    line_end: int = 0
    kind: str = ""
    layer: str = ""
    callee: str = ""
    referenced_symbol: str = ""
    referenced_value: str = ""
    literal: str = ""
    operator: str = ""
    dim: str = ""
    value: str = ""
    relation: str = ""
    #: Families named by ``relation=``, empty when the caller did not narrow.
    #: Empty means walk every family as a menu, not one mixed shortest path.
    relation_families: frozenset[str] = frozenset()
    consumer_role: str = ""
    from_symbol: str = ""
    to_symbol: str = ""
    entry_role: str = ""
    function: str = ""
    limit: int = 8
    offset: int = 0
    filled: dict[str, str] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)

    def unique_seed(self) -> bool:
        return self.operation in {"resolve", "contract", "impact"}


class InvalidQuery(ValueError):
    error_code = "INVALID_QUERY"

    def __init__(
        self,
        message: str,
        *,
        legal_filters: list[str] | None = None,
        parsed_tokens: list[str] | None = None,
        operation: str = "",
        did_you_mean: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.legal_filters = list(legal_filters or [])
        self.parsed_tokens = list(parsed_tokens or [])
        self.operation = operation
        self.did_you_mean = list(did_you_mean or [])


def _call_repr(call: dict[str, Any]) -> str:
    op = str(call.get("operation") or "")
    rest = " ".join(
        f"{k}={v}" for k, v in call.items() if k != "operation" and v not in (None, "")
    )
    return f"{op} {rest}".strip()


def suggest_calls(
    operation: str,
    *,
    illegal: list[str],
    filled: dict[str, str],
    allowed: set[str],
) -> list[dict[str, Any]]:
    """Re-bind rejected identifier values onto the legal filters of this operation.

    Purely mechanical: no ranking, no source lookup. An empty list means the
    caller gave nothing that could be moved. Suggestions stay on the public
    operations ``search`` and ``resolve``.
    """
    kept = {k: v for k, v in filled.items() if k in allowed}
    targets = [f for f in _IDENT_VALUED if f in allowed and f not in kept]
    out: list[dict[str, Any]] = []
    name_pat = str(filled.get("pattern") or filled.get("name") or "")
    if (
        operation in _SYMBOL_OPS
        and name_pat
        and ("name" in illegal or "pattern" in illegal)
    ):
        out.append({"operation": "search", "pattern": name_pat})
        if "*" not in name_pat and "?" not in name_pat and _looks_like_ident(name_pat):
            out.append({"operation": "trace", "symbol": name_pat})
    for key in illegal:
        value = str(filled.get(key) or "").strip()
        if not value:
            continue
        if not _looks_like_ident(value) and not _looks_like_name_pattern(value):
            continue
        if not _looks_like_ident(value):
            out.append({"operation": "search", "pattern": value})
            if len(out) >= _MAX_SUGGESTIONS:
                return out
            continue
        if operation not in PUBLIC_OPERATIONS:
            kind = str(kept.get("kind") or "")
            search_call = {"operation": "search", "pattern": value}
            if kind:
                search_call["kind"] = kind
            if search_call not in out:
                out.append(search_call)
            trace_call = {"operation": "trace", "symbol": value}
            if trace_call not in out:
                out.append(trace_call)
            if len(out) >= _MAX_SUGGESTIONS:
                return out
            continue
        for target in targets:
            call = {"operation": operation, **kept, target: value}
            if call not in out:
                out.append(call)
            if len(out) >= _MAX_SUGGESTIONS:
                return out
    return out


def _norm_kind(kind: str) -> str:
    return str(kind or "").strip().upper()


def _filled_filters(**kwargs: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in FILTER_KEYS:
        val = kwargs.get(key)
        if val is None:
            continue
        if isinstance(val, int):
            if val == 0:
                continue
            out[key] = str(val)
            continue
        text = str(val).strip()
        if text:
            out[key] = text
    return out


def legal_filters_for(operation: str, kind: str = "") -> list[str]:
    op = str(operation or "resolve").strip().lower()
    if op == "find":
        k = _norm_kind(kind)
        allowed = _FIND_BY_KIND.get(k, _FIND_COMMON | {"kind"})
        return sorted(allowed)
    if op == "search":
        return sorted(_SEARCH_FILTERS)
    return sorted(_OP_FILTERS.get(op, _RESOLVE_FILTERS))


def parsed_tokens(*parts: Any) -> list[str]:
    seen: list[str] = []
    for part in parts:
        for tok in _TOKEN_RE.findall(str(part or "")):
            if tok not in seen:
                seen.append(tok)
    return seen[:12]


def _looks_like_ident(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if _IDENT_RE.fullmatch(s) or _QUAL_RE.fullmatch(s):
        return True
    if "." in s and all(_IDENT_RE.fullmatch(p) for p in s.split(".") if p):
        return True
    return False


def _looks_like_name_pattern(text: str) -> bool:
    """Ident, or a glob of ident characters (`*Buf*`, `Foo?ar`)."""
    s = str(text or "").strip()
    if not s:
        return False
    if _looks_like_ident(s):
        return True
    core = s.replace("*", "").replace("?", "").replace("::", "").replace(".", "")
    return bool(core) and all(c.isalnum() or c == "_" for c in core)


def _is_dim_catalog(dim: str) -> bool:
    return str(dim or "").strip().lower() in _DIM_CATALOG


def _looks_like_graph_id(text: str) -> bool:
    """True for a CodeMap entity id, not a C++ ident the caller stuffed in entity_id=."""
    s = str(text or "").strip()
    if not s:
        return False
    if s.startswith("span:") or "::" in s:
        return True
    match = _GRAPH_ID_RE.fullmatch(s)
    if not match:
        return False
    from ascendc_codemap_mcp.engine.ids import PREFIX_KIND

    return match.group("prefix") in PREFIX_KIND


def _has_concrete_seed(operation: str, filled: dict[str, str]) -> bool:
    if operation == "source":
        return bool(filled.get("file") and filled.get("line"))
    if operation == "trace":
        return bool(
            filled.get("symbol") or filled.get("from_symbol") or filled.get("entity_id")
            or filled.get("dim")
        )
    if operation in {"resolve", "contract", "impact"}:
        return bool(
            filled.get("symbol")
            or filled.get("entity_id")
            or filled.get("file")
            or filled.get("dim")
        )
    if operation == "find":
        return bool(
            filled.get("name")
            or filled.get("callee")
            or filled.get("dim")
            or filled.get("function")
            or filled.get("literal")
            or filled.get("referenced_symbol")
            or filled.get("referenced_value")
            or filled.get("operator")
        )
    if operation == "search":
        return bool(filled.get("pattern") or filled.get("name"))
    if operation == "entry":
        return True
    return False


def validate_plan(**kwargs: Any) -> QueryPlan:
    operation = str(kwargs.get("operation") or "resolve").strip().lower() or "resolve"
    projection = str(kwargs.get("projection") or "summary").strip().lower() or "summary"
    if operation not in OPERATIONS:
        raise InvalidQuery(
            f"unknown operation: {operation}",
            legal_filters=list(PUBLIC_OPERATIONS),
            parsed_tokens=parsed_tokens(operation),
            operation=operation,
        )
    if projection not in PROJECTIONS:
        raise InvalidQuery(
            f"unknown projection: {projection}",
            legal_filters=list(PROJECTIONS),
            parsed_tokens=[projection],
            operation=operation,
        )
    filled = _filled_filters(**kwargs)
    kind = _norm_kind(filled.get("kind") or kwargs.get("kind") or "")
    dropped: list[str] = []
    # trace(from_symbol=) was the two-endpoint spelling before trace also served
    # a single symbol. Both name the same slot.
    if operation == "trace" and filled.get("from_symbol") and not filled.get("symbol"):
        filled["symbol"] = filled.pop("from_symbol")
    # entity_id= that is just a C++ ident is a name, not a graph row.
    eid = str(filled.get("entity_id") or "")
    if eid and not _looks_like_graph_id(eid):
        if not filled.get("symbol"):
            filled["symbol"] = eid
        filled.pop("entity_id", None)
        dropped.append("entity_id")
    # search: pattern is canonical; name is a silent alias.
    if operation == "search":
        for alias_key in SEARCH_PATTERN_ALIASES:
            alias = str(filled.get(alias_key) or "").strip()
            if alias and not filled.get("pattern"):
                filled["pattern"] = alias
                filled.pop(alias_key, None)
            elif alias_key in filled:
                filled.pop(alias_key, None)
                dropped.append(alias_key)
    # trace(name=) / resolve(name=) is symbol= when name is an identifier.
    if operation in _SYMBOL_OPS and filled.get("name") and not filled.get("symbol"):
        raw = str(filled.get("name") or "")
        if _looks_like_ident(raw):
            filled["symbol"] = raw
            filled.pop("name", None)
            dropped.append("name")
    # Already has a seed: extra name= is not a second question.
    if operation in _SYMBOL_OPS and filled.get("symbol") and filled.get("name"):
        filled.pop("name", None)
        dropped.append("name")
    if operation in _SYMBOL_OPS and filled.get("pattern"):
        if not filled.get("symbol") and _looks_like_ident(str(filled.get("pattern") or "")):
            filled["symbol"] = filled["pattern"]
        filled.pop("pattern", None)
        dropped.append("pattern")
    tokens = parsed_tokens(*filled.values())
    name_pattern = str(filled.get("pattern") or filled.get("name") or "")
    if operation == "search" and not name_pattern:
        raise InvalidQuery(
            "search requires pattern (regex over source lines)",
            legal_filters=sorted(_SEARCH_FILTERS),
            parsed_tokens=tokens,
            operation=operation,
        )
    if operation == "find" and not kind and not name_pattern:
        raise InvalidQuery(
            "find requires kind, or name to discover idents",
            legal_filters=list(FIND_KINDS),
            parsed_tokens=tokens,
            operation=operation,
            did_you_mean=suggest_calls(
                operation,
                illegal=[k for k in filled if k not in _FIND_COMMON],
                filled=filled,
                allowed=set(_FIND_COMMON),
            ),
        )
    allowed = set(legal_filters_for(operation, kind))
    illegal = sorted(k for k in filled if k not in allowed)
    if illegal:
        remaining = {k: v for k, v in filled.items() if k in allowed}
        ident_dropped = [
            k
            for k in illegal
            if _looks_like_ident(str(filled.get(k) or ""))
            or _looks_like_name_pattern(str(filled.get(k) or ""))
        ]
        remaining_ident = bool(
            remaining.get("symbol")
            or remaining.get("entity_id")
            or remaining.get("dim")
            or remaining.get("pattern")
            or remaining.get("name")
        )
        if operation in _SYMBOL_OPS and ident_dropped and not remaining_ident:
            ident_val = str(filled.get(ident_dropped[0]) or "")
            if _looks_like_ident(ident_val):
                suggestions = [
                    {"operation": "trace", "symbol": ident_val},
                    {"operation": "search", "pattern": ident_val},
                ]
            else:
                suggestions = [{"operation": "search", "pattern": ident_val}]
            raise InvalidQuery(
                f"unsupported filter: {ident_dropped[0]}",
                legal_filters=sorted(allowed),
                parsed_tokens=tokens,
                operation=operation,
                did_you_mean=suggestions,
            )
        if _has_concrete_seed(operation, remaining):
            dropped.extend(illegal)
            filled = remaining
            name_pattern = str(filled.get("pattern") or filled.get("name") or "")
            tokens = parsed_tokens(*filled.values())
        else:
            suggestions = suggest_calls(
                operation, illegal=illegal, filled=filled, allowed=allowed
            )
            raise InvalidQuery(
                f"unsupported filter: {illegal[0]}",
                legal_filters=sorted(allowed),
                parsed_tokens=tokens,
                operation=operation,
                did_you_mean=suggestions,
            )

    layer = str(filled.get("layer") or "")
    if layer and layer not in LAYERS:
        raise InvalidQuery(
            f"unsupported filter: layer={layer}",
            legal_filters=list(LAYERS),
            parsed_tokens=tokens,
            operation=operation,
        )
    role = str(filled.get("consumer_role") or "")
    if role and role not in CONSUMER_ROLES:
        raise InvalidQuery(
            f"unsupported filter: consumer_role={role}",
            legal_filters=sorted(CONSUMER_ROLES),
            parsed_tokens=tokens,
            operation=operation,
        )
    entry_role = str(filled.get("entry_role") or "")
    if entry_role and entry_role not in ENTRY_ROLES:
        raise InvalidQuery(
            f"unsupported filter: entry_role={entry_role}",
            legal_filters=list(ENTRY_ROLES),
            parsed_tokens=tokens,
            operation=operation,
        )
    operator = str(filled.get("operator") or "").upper()
    if operator and operator not in AST_OPERATORS:
        raise InvalidQuery(
            f"unsupported filter: operator={operator}",
            legal_filters=list(AST_OPERATORS),
            parsed_tokens=tokens,
            operation=operation,
        )
    relation = str(filled.get("relation") or "").strip()
    relation_kinds, relation_families = expand_relation(relation)
    if relation:
        legal_rel = {k.value for k in RelationKind}
        unknown = sorted(relation_kinds - legal_rel)
        if unknown:
            raise InvalidQuery(
                f"unsupported filter: relation={','.join(unknown)}",
                legal_filters=[*RELATION_FAMILY_NAMES, *sorted(legal_rel)],
                parsed_tokens=tokens,
                operation=operation,
            )

    symbol = str(filled.get("symbol") or "")
    if operation in _SYMBOL_OPS and symbol and not _looks_like_ident(symbol):
        raise InvalidQuery(
            "unsupported filter: symbol (identifier required; natural-language text is not a query)",
            legal_filters=sorted(allowed),
            parsed_tokens=tokens or parsed_tokens(symbol),
            operation=operation,
            did_you_mean=[{"operation": "search", "pattern": symbol}],
        )
    if operation == "find" and kind and kind not in _FIND_BY_KIND:
        raise InvalidQuery(
            f"unsupported filter: kind={kind}",
            legal_filters=list(FIND_KINDS),
            parsed_tokens=tokens,
            operation=operation,
        )
    if operation == "trace" and not _has_concrete_seed("trace", filled):
        # to_symbol alone names a destination with no journey. Route it to the
        # call that answers it rather than only listing the missing filter.
        endpoint = str(filled.get("to_symbol") or "")
        alternatives: list[dict[str, Any]] = []
        if endpoint and _looks_like_ident(endpoint):
            alternatives = [{"operation": "trace", "symbol": endpoint}]
        raise InvalidQuery(
            "trace requires symbol (add to_symbol for a path, or dim for the compiled key space; dim=* lists every dim)",
            legal_filters=sorted(_TRACE_ADVERTISED_FILTERS),
            parsed_tokens=tokens,
            operation=operation,
            did_you_mean=alternatives or [{"operation": "trace", "dim": "*"}],
        )
    if operation == "source" and not _has_concrete_seed("source", filled):
        missing = "line" if filled.get("file") else "file"
        raise InvalidQuery(
            f"source requires file and line ({missing} is missing); "
            "search first if you do not have a location yet",
            legal_filters=sorted(_SOURCE_FILTERS),
            parsed_tokens=tokens,
            operation=operation,
            did_you_mean=[{"operation": "search", "pattern": str(filled.get("file") or "")}]
            if filled.get("file")
            else [],
        )

    return QueryPlan(
        operation=operation,
        projection=projection,
        symbol=symbol,
        name=name_pattern,
        pattern=str(filled.get("pattern") or "") if operation == "search" else "",
        entity_id=str(filled.get("entity_id") or ""),
        file=str(filled.get("file") or ""),
        # From `filled`, not `kwargs`: a location dropped for being illegal on
        # this operation left `file` cleared but `line` set, so the plan carried
        # a coordinate it had already reported as ignored.
        line=int(filled.get("line") or 0),
        line_end=int(filled.get("line_end") or 0),
        kind=kind,
        layer=layer,
        callee=str(filled.get("callee") or ""),
        referenced_symbol=str(filled.get("referenced_symbol") or ""),
        referenced_value=str(filled.get("referenced_value") or ""),
        literal=str(filled.get("literal") or ""),
        operator=operator,
        dim=str(filled.get("dim") or ""),
        value=str(filled.get("value") or ""),
        relation=",".join(sorted(relation_kinds)),
        relation_families=relation_families,
        consumer_role=role,
        from_symbol=str(filled.get("from_symbol") or (symbol if operation == "trace" else "")),
        to_symbol=str(filled.get("to_symbol") or ""),
        entry_role=entry_role,
        function=str(filled.get("function") or ""),
        limit=max(
            1,
            int(
                kwargs.get("limit")
                or (20 if operation == "search" else 8)
            ),
        ),
        offset=max(0, int(kwargs.get("offset") or 0)),
        filled=filled,
        dropped=dropped,
    )


def invalid_payload(exc: InvalidQuery) -> dict[str, Any]:
    hint = (
        f"INVALID_QUERY: {exc}. "
        f"legal filters: {', '.join(exc.legal_filters) or '(none)'}. "
        f"parsed tokens: {', '.join(exc.parsed_tokens) or '(none)'}"
    )
    if exc.did_you_mean:
        hint += ". did you mean: " + " | ".join(
            _call_repr(call) for call in exc.did_you_mean
        )
    return {
        "ok": False,
        "shape": "invalid",
        "completeness": UNKNOWN,
        "unresolved_reason": "INVALID_QUERY",
        "error": str(exc),
        "error_code": "INVALID_QUERY",
        "legal_filters": exc.legal_filters,
        "parsed_tokens": exc.parsed_tokens,
        "did_you_mean": exc.did_you_mean,
        "operation": exc.operation,
        "cards": [],
        "count": 0,
        "hint": hint,
    }


def snapshot_has_ast(query: Any) -> bool:
    try:
        with query._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM entity
                WHERE kind IN ('BRANCH', 'PREDICATE')
                  AND json_extract(data, '$.operators') IS NOT NULL
                LIMIT 1
                """
            ).fetchone()
        return row is not None
    except Exception:  # noqa: BLE001
        return False


def execute(query: Any, plan: QueryPlan) -> dict[str, Any]:
    if plan.operation == "search":
        payload = query.query_search(
            plan, limit=plan.limit, offset=int(getattr(plan, "offset", 0) or 0)
        )
        return _attach(query, payload, plan, unique_seed=False)
    if plan.operation == "find":
        if _AST_FILTERS & set(plan.filled) and not snapshot_has_ast(query):
            raise InvalidQuery(
                "unsupported filter: literal/operator/referenced_value "
                "(needs predicate AST; this snapshot predates it)",
                legal_filters=legal_filters_for("find", plan.kind),
                parsed_tokens=parsed_tokens(*plan.filled.values()),
                operation="find",
            )
        payload = query.query_find(plan, limit=plan.limit)
        return _attach(query, payload, plan, unique_seed=False)
    if plan.operation == "entry":
        payload = query.query_entry(plan, limit=plan.limit)
        return _attach(query, payload, plan, unique_seed=False)
    if plan.operation == "source":
        payload = _source_seed(query, plan)
        return _attach(query, payload, plan, unique_seed=False)
    if plan.operation == "trace":
        if plan.to_symbol:
            return query.query_trace(plan, limit=plan.limit)
        payload = _symbol_seed(query, plan)
        return _attach(query, payload, plan, unique_seed=True)
    if plan.operation == "impact":
        payload = _resolve_seed(query, plan)
        payload = _attach(query, payload, plan, unique_seed=True)
        return payload
    if plan.operation == "contract":
        payload = _resolve_seed(query, plan)
        return _attach(query, payload, plan, unique_seed=True)
    payload = _resolve_seed(query, plan)
    site = _seeds_from_location(plan)
    return _attach(query, payload, plan, unique_seed=not site)


def _seeds_from_location(plan: QueryPlan) -> bool:
    """Whether this plan is answered from a line rather than from a name."""
    if plan.operation == "source":
        return True
    if plan.symbol or plan.entity_id or plan.dim:
        return False
    return bool(plan.file and int(plan.line or 0) > 0)


def _resolve_seed(query: Any, plan: QueryPlan) -> dict[str, Any]:
    """Compat ``operation=resolve``: a name outranks a location.

    The reverse order shipped, and a caller who passed both got a card about
    whatever entity happened to start on that line -- a predicate, a log
    statement -- while still being told it was the answer for the symbol they
    named. Callers who want the lines now say ``operation=source``.
    """
    if _seeds_from_location(plan):
        return _source_seed(query, plan)
    return _symbol_seed(query, plan)


def _source_seed(query: Any, plan: QueryPlan) -> dict[str, Any]:
    site = getattr(query, "query_site_unit", None)
    # Identity is the function that owns the window: enclosing ``line`` when
    # that function still covers most of [line, line_end], otherwise the
    # function that covers the bulk (a search range that starts in the
    # previous function's tail). `line_end` still crops the snippet.
    if callable(site):
        return site(
            plan.file,
            plan.line,
            highlight=str(plan.symbol or ""),
            limit=plan.limit,
            line_end=int(plan.line_end or 0),
        )
    return query.query_around(
        plan.file, plan.line, line_end=int(plan.line_end or plan.line), limit=plan.limit
    )


def _prefer_file(payload: dict[str, Any], file: str) -> dict[str, Any]:
    """Keep the cards that live in ``file``, when the caller named one.

    file/line no longer routes the query; it still disambiguates one name that
    several translation units declare.
    """
    needle = str(file or "").strip().replace("\\", "/").lower()
    if not needle:
        return payload
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    kept = [
        c
        for c in cards
        if needle in str(c.get("file") or "").replace("\\", "/").lower()
        or str(c.get("file") or "").replace("\\", "/").lower().endswith(needle)
    ]
    if kept and len(kept) < len(cards):
        payload["cards"] = kept
        payload["count"] = len(kept)
    return payload


def _symbol_seed(query: Any, plan: QueryPlan) -> dict[str, Any]:
    symbol = plan.symbol
    if plan.entity_id:
        if _looks_like_graph_id(plan.entity_id):
            payload = query.query_name_card(plan.entity_id, limit=plan.limit)
            cards = [
                c
                for c in (payload.get("cards") or [])
                if isinstance(c, dict) and str(c.get("id") or "") == plan.entity_id
            ]
            if cards:
                payload["cards"] = cards
                payload["count"] = len(cards)
            return payload
        symbol = symbol or plan.entity_id
    if plan.dim:
        if _is_dim_catalog(plan.dim) and not plan.value:
            return query.query_index(limit=plan.limit)
        pattern = f"{plan.dim}={plan.value}" if plan.value else f"Dim={plan.dim}"
        return query.query_cover(pattern, limit=plan.limit)
    if not symbol:
        return query.query_index(limit=plan.limit)
    payload = query.query_name_card(symbol, limit=plan.limit)
    wanted = str(plan.kind or "").strip().upper()
    if wanted:
        cards = [
            c
            for c in (payload.get("cards") or [])
            if isinstance(c, dict) and str(c.get("kind") or "").upper() == wanted
        ]
        if cards:
            payload["cards"] = cards
            payload["count"] = len(cards)
    return _prefer_file(payload, plan.file)


def _attach(query: Any, payload: dict[str, Any], plan: QueryPlan, *, unique_seed: bool) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.query.explore import attach_explore_fields
    from ascendc_codemap_mcp.engine.query.sql import _fit_payload

    pattern = (
        plan.symbol
        or plan.callee
        or plan.pattern
        or plan.name
        or plan.dim
        or (f"{plan.file}:{plan.line}" if plan.file else "")
    )
    if plan.relation_families:
        # Set before rendering: the renderer narrows the card it is about to
        # emit, and states which families it left out.
        payload["relation_families"] = sorted(plan.relation_families)
    payload = attach_explore_fields(
        query,
        payload,
        pattern=pattern,
        unique_seed=unique_seed,
        projection=plan.projection,
        operation=plan.operation,
        seed_kind=plan.kind,
        file_filter="" if _seeds_from_location(plan) else plan.file,
    )
    if plan.dropped:
        note = "ignored filters: " + ", ".join(plan.dropped)
        existing = str(payload.get("hint") or "")
        payload["hint"] = f"{existing} {note}".strip() if existing else note
        text = str(payload.get("text") or "")
        if text and note not in text:
            payload["text"] = text.rstrip() + "\n" + note + "\n"
    return _fit_payload(payload)


def plan_fingerprint(plan: QueryPlan) -> str:
    blob = json.dumps(
        {
            "op": plan.operation,
            "proj": plan.projection,
            **plan.filled,
            "l": plan.line,
            "le": plan.line_end,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    import hashlib

    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
