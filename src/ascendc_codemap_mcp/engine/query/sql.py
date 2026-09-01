# -*- coding: utf-8 -*-
"""Indexed SQLite query over a committed ``.uo`` product.

Agent-facing ``acp uo-query`` must never hydrate the full CodeMap.  All
navigation uses ``entity`` / ``relation`` / ``source_span`` indexes.  Dump /
audit helpers that truly need the in-memory graph stay on
:class:`uo_init.query.engine.CodeMapQuery` and are lazy here.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
import threading
from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ascendc_codemap_mcp.engine.common_paths import strip_dot_slash
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.evidence import TRUST_ADVISORY
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.pipe_lifetime import (
    continued_line_ranges,
    order_pipe_names,
    receiver_leaf,
)
from ascendc_codemap_mcp.engine.query.evidence import (
    USEFUL_EDGE_KINDS,
    bucket_hits,
    catalog_kind_alias,
    field_edge_kinds,
    project_entity,
    project_relation,
    surface_facts,
)
from ascendc_codemap_mcp.engine.query.hints import (
    attach_query_hints,
    identifier_tokens,
    search_needles,
)
from ascendc_codemap_mcp.engine.query.legal_key_cache import _pattern_filters, normalize_cover_pattern
from ascendc_codemap_mcp.engine.source_locator import locations_from_attr_sites

SNIPPET_LINES = 40
SNIPPET_BEFORE = 3
MACRO_CONT_MAX = 80
STATEMENT_BEFORE = 8
STATEMENT_AFTER = 8
SITE_UNIT_HARD = 240
NO_ENCLOSE_RADIUS = 120
_STMT_EXPAND_KINDS = {
    EntityKind.COMPILE_VAR.value,
    EntityKind.MACRO.value,
    EntityKind.TILING_KEY.value,
    EntityKind.TYPE.value,
}
PRIMARY_CANDIDATES = 3
MAX_PAYLOAD_CHARS = 24_000
MAX_REL_HOPS = 4
MIN_LIST_KEEP = 5
_PROTECTED_PAYLOAD_KEYS = frozenset(
    {
        "coverage",
        "files",
        "dim_coverage",
        "nearby",
        "matching_block_count",
        "legal_key_count",
        "phases",
        "first_query",
        "answer_contract",
        "filters",
        "occupancy_axis",
        "fixed_coverage",
        "other_kernels",
        "cards",
        "next",
        "omitted",
        "dim_names",
        "tiling_data_names",
        "shape",
        "total_matched",
        "host",
        "kernel",
        "flow",
        "definition",
        "related",
        "impact",
        "enclosing",
        "seeds",
        "match",
        "match_note",
        "text_hits",
        "text_hits_total",
        "text_hits_complete",
        "compiled_support",
        "exhaustive",
        "returned",
        "used_at",
        "completeness",
        "impact_sinks",
        "entry",
        "checks",
        "unresolved_reason",
        "unresolved_reasons",
        "windows",
        "text",
        "state_changes",
        "unit_start",
        "unit_end",
        "function_start",
        "function_end",
        "resolve_mode",
        "assignments",
        "host_kernel",
        "calls",
        "bundle",
        "units",
        "leftover",
        "template_lines",
        "source_units",
        "field_bundles",
        "recovered_token",
        "original_pattern",
        "highlight",
        "trimmed_lists",
        "other_definitions",
        "matched_entities",
    }
)
PACKING_RHS_TRIM = 400
_GET_OFFSET_RE = re.compile(r"\bGet\w*Offset\b")
_DATA_MOVE_RE = re.compile(r"\bDataCopy(?:Pad)?\b|\bLoadData\b")
_TRIVIAL_RHS_RE = re.compile(r"^(?:true|false|0|1|nullptr)?$", re.IGNORECASE)
_DECISION_RHS_RE = re.compile(r"&&|\|\||==|!=|>=|<=|>(?!=)|<(?!=)")
_NEXT_EDGE_PRIORITY = (
    "WRITES",
    "READS",
    "CALLS",
    "BINDS",
    "DERIVES",
    "RETURNS",
    "GUARDED_BY",
    "SELECTS",
    "CONTROLS",
)
_NEXT_EDGE_SKIP = frozenset(
    {
        "FLOWS_TO",
        "PRECEDES",
        "DECLARES",
        "ACTIVE_UNDER",
        "CONTAINS",
        "LAUNCHES",
    }
)
_FOCUS_SKIP_IDENTS = frozenset(
    {
        "true",
        "false",
        "bool",
        "auto",
        "const",
        "static_cast",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int",
        "int32_t",
        "int64_t",
        "void",
        "static",
        "nullptr",
        "this",
        "if",
        "else",
        "return",
        "ge",
    }
)
NEIGHBOR_REL_KINDS = (
    "SELECTS",
    "CONTROLS",
    "CALLS",
    "SIGNALS",
    "AWAITS",
    "DECLARES",
    "WRAPS",
    "BINDS",
    "READS",
    "WRITES",
)
CARD_EDGE_KINDS = NEIGHBOR_REL_KINDS + (
    "ROOTED_AT",
    "RETURNS",
    "GUARDED_BY",
    "DERIVES",
    "PRECEDES",
    "ACTIVE_UNDER",
    "ALIASES",
    "CONTAINS",
    "FLOWS_TO",
    "LAUNCHES",
    "REFERENCES",
    "BACKED_BY",
    "INSTANCE_OF",
    "MATERIALIZES_AS",
)
# Neighbor bodies on the card (codegraph trail / CBM 1-hop). Other CARD_EDGE_KINDS
# stay as count-only so completeness is visible without dumping the hop list.
CARD_NEIGHBOR_RELS = (
    "WRITES",
    "READS",
    "CALLS",
    "DERIVES",
    "RETURNS",
    "ROOTED_AT",
    "BINDS",
    "ALIASES",
)
EDGES_PER_KIND = 8
NEXT_HOP_LIMIT = 6
DERIVES_PER_TILING_KEY = 3
CARD_SNIPPET_MAX_LINES = 8
AROUND_SEED_LIMIT = 16
AROUND_NEIGHBORS_PER_KIND = 4
#: trace(symbol, to_symbol) walk bounds. Depth keeps a hit short enough to read;
#: nodes and fan-out keep a miss from scanning the whole graph.
_TRACE_NODE_BUDGET = 6000
_TRACE_MAX_DEPTH = 12
_TRACE_FANOUT = 200
_TRACE_FAMILY_NODE_BUDGET = 80
_TRACE_FAMILY_MAX_DEPTH = 5
_TRACE_FAMILY_FANOUT = 12
_TRACE_COMPILE_NODE_BUDGET = 40
_TRACE_COMPILE_MAX_DEPTH = 3
_DECLARES_KIND_SQL = (
    "CASE e.kind WHEN 'METHOD' THEN 0 WHEN 'FUNCTION' THEN 1 "
    "WHEN 'FIELD' THEN 2 WHEN 'BUFFER' THEN 3 ELSE 4 END"
)
TEMPLATE_BLOCK_EXEMPLARS = 1
MAX_NAME_CARDS = 4
MAX_DEFINITION_SITES = 12
MAX_WRITE_TIMELINE = 8
MAX_SAME_VALUE = 8
_VIEW_CACHE_MAX = 1
_TEMPLATE_BLOCKS_LOCK = threading.Lock()
_TEMPLATE_BLOCKS_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SQL_IN_CHUNK = 400


def _chunks(items: Iterable[Any], size: int = _SQL_IN_CHUNK) -> Iterator[list[Any]]:
    seq = [item for item in items if item not in (None, "")]
    for i in range(0, len(seq), max(1, size)):
        yield seq[i : i + size]
_IDENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AROUND_KIND_PRIORITY = {
    EntityKind.TILING_FIELD.value: 0,
    EntityKind.TILING_DATA.value: 1,
    EntityKind.TILING_KEY.value: 2,
    EntityKind.FIELD.value: 3,
    EntityKind.PIPE.value: 4,
    EntityKind.KERNEL.value: 5,
}
_EXACT_KINDS = {EntityKind.TYPE.value}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HW_PIPE_RE = re.compile(r"^PIPE_[A-Z][A-Z0-9]*$")
_OCCUPANCY_IDENT_RE = re.compile(
    r"(?:^|[_.])(?:coreNum|aicNum|aivNum|blockDim|blockOuter|fusedOuter|"
    r"usedCoreNum|cubeCoreNum|vecCoreNum|coreSplit|splitCount|"
    r"tschBlockDim|blockIdx)(?:$|[_.])",
    re.IGNORECASE,
)
_WRITER_SKIP_KINDS = {EntityKind.INPUT.value, EntityKind.OUTPUT.value}
_KERNEL_READER_KINDS = {
    RelationKind.READS.value,
    RelationKind.BINDS.value,
    RelationKind.SELECTS.value,
    RelationKind.CALLS_UNDER_GUARD.value,
    RelationKind.MATERIALIZES_AS.value,
}
_LEGACY_KIND_MAP: dict[str, set[str]] = {
    "Variable": {"VARIABLE", "COMPILE_VAR", "MACRO"},
    "Input": {"INPUT"},
    "OptionalInput": {"INPUT"},
    "Output": {"OUTPUT"},
    "TilingDataField": {"TILING_FIELD"},
    "TilingKeyDim": {"TILING_KEY"},
    "HostBranch": {"BRANCH"},
    "KernelBranch": {"BRANCH"},
    "Predicate": {"PREDICATE"},
    "TemplateBinding": {"TEMPLATE", "TEMPLATE_ARG", "TEMPLATE_INSTANCE"},
}


def _kind_names(kinds: Iterable[str]) -> set[str]:
    out: set[str] = set()
    valid = {k.value for k in EntityKind}
    for raw in kinds:
        text = str(raw or "").strip()
        if not text:
            continue
        if text in _LEGACY_KIND_MAP:
            out.update(_LEGACY_KIND_MAP[text])
        upper = text.upper()
        if upper in valid:
            out.add(upper)
    return out


def _parse_data(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_advisory_data(raw: Any) -> bool:
    return str(_parse_data(raw).get("trust") or "") == TRUST_ADVISORY


def _skip_composition_neighbor(
    entity_kind: str, rel_kind: str, other_kind: str
) -> bool:
    """Hide inverted TYPE composition that floods MutexBuffer-style cards."""
    rel = str(rel_kind or "")
    other = str(other_kind or "").upper()
    kind_u = str(entity_kind or "").upper()
    if rel == "WRAPS" and other == EntityKind.BUFFER.value:
        return kind_u in {"", EntityKind.TYPE.value}
    if rel == "CONTAINS" and other == EntityKind.TYPE.value:
        return True
    return False


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def _strip_dot_slash(text: str) -> str:
    """Drop leading ``./`` segments. Never touches ``../``."""
    return strip_dot_slash(text)


def _rel_key(file: str) -> str:
    """Collapse ``.`` / ``..`` so source_line.path and entity.file can match."""
    text = str(file or "").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    parts: list[str] = []
    for part in text.split("/"):
        if part in {"", "."}:
            continue
        if part == ".." and parts and parts[-1] != "..":
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _norm_file(file: str) -> str:
    """Display/lookup spelling of a stored location.

    A path the writer already canonicalized is passed through: cards are what an
    agent copies back into ``--file``, so shortening ``../common/op_kernel/x.h``
    to ``op_kernel/x.h`` would name a file that is not there -- and would
    collide with the operator's own ``op_kernel/x.h``. Only a legacy product's
    absolute paths still get cut down.
    """
    text = str(file or "").replace("\\", "/")
    if not text:
        return ""
    if not _is_machine_path(text):
        return _strip_dot_slash(text)
    for marker in ("/op_kernel/", "/op_host/", "/include/"):
        idx = text.lower().find(marker)
        if idx >= 0:
            return text[idx + 1 :]
    return Path(text).name


def _is_machine_path(text: str) -> bool:
    """Rooted at a drive or at ``/``: names a location on the build machine.

    The ``<cann>/`` marker is not one of these -- it is already portable, and
    cutting it down would strip the only thing that says which tree it is in.
    """
    return text.startswith("/") or (len(text) > 1 and text[1] == ":")


def _architecture_from_name(path: Path) -> str:
    from ascendc_codemap_mcp.engine.source_layout import is_product_architecture

    name = path.name
    if not name.endswith(".uo"):
        return ""
    stem = name[: -len(".uo")]
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and is_product_architecture(parts[1]):
        return parts[1]
    return ""


def _op_root_from_product(product: Path) -> Path | None:
    parts = product.resolve().parts
    try:
        idx = parts.index(".ascendc-codemap")
    except ValueError:
        return None
    if idx <= 0:
        return None
    return Path(*parts[:idx])


def _is_truncated_branch_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if "..." in raw:
        return True
    # ``ptr->field`` and ``a > b`` are not truncated templates.
    stripped = raw.replace("->", " ")
    if "<" in stripped and stripped.count("<") != stripped.count(">"):
        return True
    return len(raw) > 80 and "<" in stripped


def _keep_branch(kind: str, name: str, data: dict[str, Any]) -> bool:
    if kind != EntityKind.BRANCH.value:
        return True
    cond = str(data.get("condition") or name or "")
    return not _is_truncated_branch_text(cond)


def _last_ident(name: str) -> str:
    text = str(name or "").replace(".", "::")
    return text.split("::")[-1].strip()


def _dim_aliases(ident: str) -> list[str]:
    """Host spellings that join a persisted legal_key dim (hasRope ↔ IsRope)."""
    leaf = _last_ident(ident)
    if not leaf:
        return []
    out = [leaf]
    low = leaf.lower()
    if low.startswith("has") and len(leaf) > 3:
        rest = leaf[3:]
        cap = rest[:1].upper() + rest[1:]
        out.extend([f"Is{cap}", f"is{rest}"])
    elif low.startswith("is") and len(leaf) > 2:
        rest = leaf[2:]
        cap = rest[:1].upper() + rest[1:]
        out.extend([f"has{cap}", f"Has{cap}"])
    if leaf.endswith("Num"):
        out.append(leaf[:-3])
    else:
        out.append(leaf + "Num")
    seen: set[str] = set()
    uniq: list[str] = []
    for name in out:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(name)
    return uniq


_DEF_CARD_KINDS = {
    EntityKind.FUNCTION.value,
    EntityKind.METHOD.value,
    EntityKind.KERNEL.value,
}
_ENCLOSE_KINDS = _DEF_CARD_KINDS
#: Kinds whose card is the read. Wider than `_DEF_CARD_KINDS`, which is the set
#: that can *enclose* a line and must stay free of types.
_UNIT_SNIPPET_KINDS = _DEF_CARD_KINDS | {EntityKind.TYPE.value}
_ASCENDC_CATALOG_SQL = "IFNULL(json_extract(e.data, '$.catalog'), '') != 'ascendc'"


def _is_recorded_definition(hit: dict[str, Any]) -> bool:
    kind = str(hit.get("kind") or "").upper()
    start = int(hit.get("line_start") or hit.get("line") or 0)
    end = int(hit.get("line_end") or 0)
    return kind in _DEF_CARD_KINDS and end > start


def _card_snippet_for_hit(hit: dict[str, Any]) -> tuple[str, bool]:
    text = str(hit.get("snippet") or "")
    kind = str(hit.get("kind") or "").upper()
    if (
        _is_recorded_definition(hit)
        or kind == EntityKind.MACRO.value
        or kind in _STMT_EXPAND_KINDS
    ):
        return text, bool(hit.get("truncated"))
    return _clip_definition_snippet(text)


def _same_path(left: str, right: str) -> bool:
    a = _norm_file(left)
    b = _norm_file(right)
    if not a or not b:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a) or a.endswith(b) or b.endswith(a)


def _occupancy_rank(hit: dict[str, Any]) -> int:
    """Prefer located, written identities over empty VARIABLE / bare BRANCH."""
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    kind = str(hit.get("kind") or "").upper()
    catalog = str(facts.get("catalog") or "")
    if catalog == "ge.graphStatus":
        return 0
    if catalog == "ascendc":
        return 9
    file = str(hit.get("file") or "").strip()
    writes = (
        facts.get("write_sites")
        or facts.get("host_writer_sites")
        or facts.get("value_defining_sites")
        or facts.get("packing_value_sites")
    )
    has_writes = isinstance(writes, list) and bool(writes)
    if file and (
        has_writes
        or kind
        in {
            EntityKind.FIELD.value,
            EntityKind.TILING_FIELD.value,
            EntityKind.TILING_KEY.value,
        }
    ):
        return 0
    if file:
        return 1
    if kind == EntityKind.VARIABLE.value:
        return 3
    if kind == EntityKind.BRANCH.value:
        return 4
    return 2


def _leaf_name_where(needle: str) -> tuple[str, list[str]]:
    full = str(needle or "").strip()
    ident = _last_ident(full.replace(".", "::"))
    if not ident:
        ident = full
    clause = f"""
      {_ASCENDC_CATALOG_SQL}
      AND (
        e.name = ? COLLATE NOCASE
        OR e.name = ? COLLATE NOCASE
        OR e.name LIKE '%.' || ? COLLATE NOCASE
        OR e.name LIKE '%::' || ? COLLATE NOCASE
      )
    """
    return clause, [full, ident, ident, ident]


def _leaf_name_where_indexed(needle: str) -> tuple[str, list[str]]:
    """Same predicate as `_leaf_name_where` over the leaf-name inverted index.

    The legacy clause combines `OR` with a leading-wildcard `LIKE`, which makes
    SQLite abandon every index on `entity` and scan the table on every hop.
    """
    full = str(needle or "").strip().lower()
    ident = _last_ident(full.replace(".", "::")) or full
    leaves = [full] if ident == full else [full, ident]
    placeholders = ",".join("?" for _ in leaves)
    clause = (
        "e.id IN (SELECT entity_id FROM entity_name_leaf "
        f"WHERE leaf IN ({placeholders}) AND is_ascendc = 0)"
    )
    return clause, leaves


def _encloses_line(hit: dict[str, Any], line: int) -> bool:
    kind = str(hit.get("kind") or "").upper()
    if kind not in _ENCLOSE_KINDS:
        return False
    start = int(hit.get("line_start") or 0)
    end = int(hit.get("line_end") or 0)
    if start <= 0 or end <= start:
        return False
    loc = int(line or 0)
    return start <= loc <= end


_SITE_UNIT_HARD = 240
_NO_ENCLOSE_RADIUS = 120

#: Kinds that can hold a call site. A candidate list is a guess by definition,
#: so the guesses at least have to be things capable of calling; a queue or a
#: field named as a possible caller is not a weak answer but a wrong one.
_CALLER_KINDS = frozenset(
    {
        EntityKind.FUNCTION.value,
        EntityKind.METHOD.value,
        EntityKind.KERNEL.value,
        EntityKind.MACRO.value,
        EntityKind.TEMPLATE.value,
        EntityKind.TEMPLATE_INSTANCE.value,
    }
)


def _is_noise_name_sql(name: str) -> bool:
    from ascendc_codemap_mcp.engine.query.explore import _is_noise_name

    return _is_noise_name(name)


# A member chain, optionally called once. `->` and `::` are spelled as units on
# purpose: a character class holding `-` and `>` separately also admits `a > b`
# and `a - b`, and dropping the parens off `!(a > b)` yields `!a > b`, which is
# the opposite claim.
_ATOMIC_GUARD_RE = re.compile(
    r"^[A-Za-z_]\w*(?:(?:->|::|\.)[A-Za-z_]\w*)*(?:\s*\([^()]*\))?$"
)


def _matching_paren(text: str, open_at: int) -> int:
    """Index of the ``)`` closing the ``(`` at `open_at`, or -1."""
    depth = 0
    for index in range(open_at, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _is_atomic_guard(text: str) -> bool:
    """A term that binds tighter than ``!``, so dropping its parens is safe."""
    return bool(_ATOMIC_GUARD_RE.match(text.strip()))


def _simplify_negation(text: str) -> str:
    """Drop parentheses that ``!`` does not need. Never applies De Morgan.

    ``!(!x)`` collapses only when the inner ``!`` covers the whole term;
    ``!(!a && b)`` is left alone because rewriting it would change its meaning.
    """
    current = str(text or "").strip()
    for _ in range(4):
        if not current.startswith("!(") or _matching_paren(current, 1) != len(current) - 1:
            return current
        inner = current[2:-1].strip()
        if inner.startswith("!"):
            term = inner[1:].strip()
            if _is_atomic_guard(term):
                current = term
                continue
            if term.startswith("(") and _matching_paren(term, 0) == len(term) - 1:
                current = term
                continue
            return current
        if _is_atomic_guard(inner):
            return "!" + inner
        return current
    return current


_LOOP_HEADER_RE = re.compile(r"^(?:for|while|do|switch|cxx_for_range)\s*\(")


def _needs_guard_parens(text: str) -> bool:
    """True when joining this guard with && would change how it reads."""
    depth = 0
    for index, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "|" and depth == 0 and text[index + 1 : index + 2] == "|":
            return True
    return False


def _guards_from_spans(spans: list[tuple[int, int, str]], loc: int) -> list[str]:
    """Innermost enclosing guards at ``loc``, skipping a branch written on it."""
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for start, end, name in spans:
        if start > loc:
            break
        if loc > end or name in seen or start == end == loc:
            continue
        seen.add(name)
        found.append((end - start, name))
    found.sort(key=lambda item: item[0])
    return [name for _width, name in found[:3]]


def _join_guards(guards: Iterable[str]) -> str:
    """Conjoin enclosing guards without silently re-associating them.

    Guards nest, so the conjunction is the truth; but a guard that is itself a
    disjunction binds looser than ``&&`` and would be misread once joined.

    Loop and switch headers are dropped: ``for (i < n)`` says the site runs on
    some iteration, not that ``i < n`` holds where it sits, so conjoining it
    with real conditions would state something the code does not promise.
    """
    parts: list[str] = []
    for raw in guards:
        text = _simplify_negation(str(raw or "").strip())
        if not text or _LOOP_HEADER_RE.match(text):
            continue
        parts.append(f"({text})" if _needs_guard_parens(text) else text)
    return " && ".join(parts)


def _file_same(left: str, right: str) -> bool:
    a = _norm_file(str(left or "")).replace("\\", "/")
    b = _norm_file(str(right or "")).replace("\\", "/")
    if not a or not b:
        return False
    if a == b:
        return True
    if _rel_key(a) == _rel_key(b):
        return True
    return a.endswith("/" + b) or b.endswith("/" + a) or a.rsplit("/", 1)[-1] == b.rsplit("/", 1)[-1]


def _smallest_compound_span(
    rows: list[tuple[int, str]], center: int, *, cap: int
) -> tuple[int, int] | None:
    """Innermost brace block that contains `center`, capped at `cap` lines."""
    if not rows:
        return None
    indexed = {int(ln): str(txt or "") for ln, txt in rows}
    numbers = sorted(indexed)
    if not numbers:
        return None
    if center not in indexed:
        center = min(numbers, key=lambda n: abs(n - center))
    depth = 0
    start = numbers[0]
    for n in reversed([x for x in numbers if x <= center]):
        for ch in reversed(indexed[n]):
            if ch == "}":
                depth += 1
            elif ch == "{":
                if depth == 0:
                    start = n
                    depth = -1
                    break
                depth -= 1
        if depth < 0:
            break
    depth = 0
    end = numbers[-1]
    started = False
    for n in [x for x in numbers if x >= start]:
        for ch in indexed[n]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth = max(0, depth - 1)
                if started and depth == 0:
                    end = n
                    break
        if started and depth == 0 and end >= start:
            break
    if end < start:
        return None
    if end - start + 1 > cap:
        lo = max(start, center - cap // 2)
        hi = min(end, lo + cap - 1)
        return lo, hi
    return start, end


def _edge_evidence_rank(
    rel_kind: str,
    rel_data: Any,
    other_kind: str,
    other_file: str,
    seed_file: str,
) -> int:
    data = _parse_data(rel_data)
    rel_file = str(data.get("file") or other_file or "")
    same = _same_path(rel_file, seed_file) or _same_path(str(other_file or ""), seed_file)
    formula = bool(
        str(data.get("rhs") or data.get("expression") or data.get("formula") or "").strip()
    )
    predicate = bool(
        data.get("predicate")
        or data.get("guards")
        or rel_kind in {"GUARDED_BY", "CONTROLS"}
    )
    is_input = str(other_kind or "").upper() == EntityKind.INPUT.value
    if same and formula:
        return 0
    if same and predicate:
        return 1
    if (not same) and (formula or rel_kind == "BINDS"):
        return 2
    if is_input and not formula:
        return 3
    return 4


def _prefix_continues_ident(name: str, needle: str) -> bool:
    """True when ``name`` keeps the same ident token after ``needle`` (FooBar vs Foo_)."""
    if not needle or not name.startswith(needle) or len(name) <= len(needle):
        return False
    return name[len(needle)].isalnum()


def _name_rank(
    name: str,
    eid: str,
    needle: str,
    *,
    exact_kind: bool,
    kind: str = "",
) -> int | None:
    if not needle:
        return 3
    low_name = str(name or "").lower()
    ident = _last_ident(low_name)
    id_ident = _last_ident(str(eid or "").replace("/", "::"))
    needle_ident = _last_ident(needle.replace(".", "::"))
    member = low_name.startswith(needle + "::") or (
        bool(needle_ident) and low_name.startswith(needle_ident + "::")
    )
    exact = (
        low_name == needle
        or ident == needle
        or id_ident == needle
        or (bool(needle_ident) and ident == needle_ident)
        or (bool(needle_ident) and id_ident == needle_ident)
    )
    if member:
        return 0
    if exact:
        return 1 if str(kind or "").upper() == EntityKind.METHOD.value else 0
    if exact_kind:
        return None
    if low_name.startswith(needle) or ident.startswith(needle):
        cont = _prefix_continues_ident(low_name, needle) if low_name.startswith(needle) else False
        if not cont and ident.startswith(needle):
            cont = _prefix_continues_ident(ident, needle)
        return 3 if cont else 2
    if needle in low_name:
        return 4
    return None


def _prefer_src_id(eid: str) -> int:
    text = str(eid or "")
    if text.startswith(("SRCTYPE::", "SRCFIELD::", "SRCKDEF")):
        return 0
    if text.startswith("TYPE_"):
        return 2
    return 1


def _definition_rank(kind: str, name: str, eid: str, facts: dict[str, Any] | None) -> tuple[int, int]:
    src = _prefer_src_id(eid)
    kind_u = str(kind or "").upper()
    facts = facts if isinstance(facts, dict) else {}
    prov = str(facts.get("provenance") or "")
    role = str(facts.get("role") or "")
    if kind_u == EntityKind.METHOD.value:
        if facts.get("source_definition") or "source_kernel_definition" in prov:
            use = 0
        elif role == "kernel_call_boundary" or "call_boundary" in prov:
            use = 4
        elif "::" in str(name or ""):
            use = 1
        else:
            use = 2
    elif kind_u == EntityKind.TYPE.value:
        cpp = str(facts.get("cpp_kind") or "").lower()
        role = str(facts.get("role") or "")
        use = 0 if cpp in {"class", "struct"} or role == "storage_wrapper_type" else 1
    else:
        use = 0
    return (src, use)


def _arch_file_rank(file: str, architecture: str) -> int:
    """Prefer operator op_host/op_kernel (arch-scoped first) over ../common."""
    text = str(file or "").replace("\\", "/").lower()
    arch = str(architecture or "").strip().lower()
    if not text:
        return 4
    blob = f"/{text.strip('/')}/"
    if "/common/" in blob:
        return 3
    if arch and f"/{arch}/" in blob:
        return 0
    if "/op_host/" in blob or blob.startswith("/op_host/"):
        return 1
    if "/op_kernel/" in blob or blob.startswith("/op_kernel/"):
        return 1
    return 2


def _is_occupancy_ident(name: str) -> bool:
    leaf = _last_ident(str(name or ""))
    if not leaf:
        return False
    return _OCCUPANCY_IDENT_RE.search(f"_{leaf}_") is not None


def _is_launch_pipe_hit(hit: dict[str, Any]) -> bool:
    """TPipe instances with a source file — not HardEvent hardware pipe enums."""
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    if str(facts.get("catalog") or "") == "ascendc":
        return False
    if facts.get("pointer"):
        return False
    role = str(facts.get("role") or "")
    if role in {"src_pipe", "dst_pipe"}:
        return False
    name = str(hit.get("name") or "")
    if _HW_PIPE_RE.match(name):
        return False
    file = str(hit.get("file") or "").strip()
    return bool(file)


def _launch_group_key(hit: dict[str, Any], architecture: str) -> tuple[Any, ...]:
    file = str(hit.get("file") or "").replace("\\", "/")
    fname = file.rsplit("/", 1)[-1].lower()
    return (
        _arch_file_rank(file, architecture),
        0 if "entry" in fname else 1,
        1 if fname.endswith("_apt.cpp") else 0,
        file.lower(),
    )


def _select_launch_phases(
    hits: list[dict[str, Any]], *, architecture: str, limit: int
) -> tuple[list[dict[str, Any]], list[str]]:
    launchable = [hit for hit in hits if _is_launch_pipe_hit(hit)]
    if not launchable:
        return [], []
    launchable.sort(key=lambda hit: _launch_group_key(hit, architecture))
    winner_file = str(launchable[0].get("file") or "").replace("\\", "/").lower()
    selected = [
        hit
        for hit in launchable
        if str(hit.get("file") or "").replace("\\", "/").lower() == winner_file
    ]
    other_files: list[str] = []
    seen: set[str] = set()
    for hit in launchable:
        file = str(hit.get("file") or "").replace("\\", "/")
        if file.lower() == winner_file or not file or file in seen:
            continue
        seen.add(file)
        other_files.append(file)
    selected.sort(
        key=lambda hit: (
            int((hit.get("facts") or {}).get("pipe_ordinal") or 0)
            if isinstance(hit.get("facts"), dict)
            else 0,
            int(hit.get("line_start") or 0),
            str(hit.get("name") or ""),
        )
    )
    return selected[: max(0, int(limit))], other_files


def _is_compile_unit_placeholder(name: str) -> bool:
    return "source_scope" in str(name or "").lower()


def _site_loc(row: dict[str, Any]) -> tuple[str, int]:
    file = _norm_file(str(row.get("file") or ""))
    line = int(row.get("line") or row.get("line_start") or 0)
    return file, line


def _is_use_site_hit(kind: str, *bags: dict[str, Any]) -> bool:
    """Whether a hit is a place the name is *used* rather than declared.

    An API the tree only calls — ``CrossCoreSetFlag`` and its neighbours — has
    no declaration in the snapshot, so every entity carrying the name was
    minted at a call. Counting those as definition sites told a reader there
    were twelve implementations to compare and sent them to twelve callers.
    The same holds for a type named inside an alias: ``using T =
    std::conditional_t<…, A, B>`` mentions ``A``, it does not declare it.
    """
    if kind == EntityKind.OPERATION.value:
        return True
    for bag in bags:
        if not isinstance(bag, dict):
            continue
        if (
            bag.get("call_target")
            or bag.get("internal_unresolved")
            or bag.get("reference_only")
        ):
            return True
        if str(bag.get("cpp_kind") or "") in {"call", "call_expr"}:
            return True
    return False


def _definition_sites_from_hits(
    hits: Sequence[dict[str, Any]], *, needle: str, limit: int = MAX_DEFINITION_SITES
) -> tuple[list[dict[str, Any]], bool]:
    ident = _last_ident(str(needle or "").replace(".", "::")).lower()
    sites: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for hit in hits:
        name = str(hit.get("name") or "")
        if ident and _last_ident(name.replace(".", "::")).lower() != ident:
            if name.lower() != str(needle or "").lower():
                continue
        file = str(hit.get("file") or "").replace("\\", "/")
        line = int(hit.get("line_start") or hit.get("line") or 0)
        kind = str(hit.get("kind") or "")
        if not file or line <= 0:
            continue
        key = (file, line, kind)
        if key in seen:
            continue
        seen.add(key)
        facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
        attrs = hit.get("attrs") if isinstance(hit.get("attrs"), dict) else {}
        data = hit.get("data") if isinstance(hit.get("data"), dict) else {}
        cpp_kind = str(
            facts.get("cpp_kind") or attrs.get("cpp_kind") or data.get("cpp_kind") or ""
        )
        if _is_use_site_hit(kind, facts, attrs, data):
            continue
        sites.append(
            {
                "file": file,
                "line": line,
                "line_end": int(hit.get("line_end") or line),
                "kind": kind,
                "name": name,
                "snippet": str(hit.get("snippet") or ""),
                "cpp_kind": cpp_kind,
            }
        )
    complete = len(sites) <= int(limit)
    return sites[: max(0, int(limit))], complete


_STATEMENT_READER_KINDS = {
    EntityKind.BRANCH.value,
    EntityKind.PREDICATE.value,
}


def _prefer_statement_readers(
    readers: list[dict[str, Any]],
    definition_sites: Sequence[dict[str, Any]],
    field_ident: str,
) -> list[dict[str, Any]]:
    ident = str(field_ident or "").lower()
    if not ident:
        return readers
    preferred: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for site in definition_sites:
        kind = str(site.get("kind") or "").upper()
        if kind not in _STATEMENT_READER_KINDS:
            continue
        if _last_ident(str(site.get("name") or "")).lower() != ident:
            continue
        file = str(site.get("file") or "")
        line = int(site.get("line") or 0)
        if not file or line <= 0:
            continue
        key = (file, line)
        if key in seen:
            continue
        seen.add(key)
        preferred.append(
            {
                "name": site.get("name"),
                "kind": kind,
                "file": file,
                "line": line,
            }
        )
    if not preferred:
        return readers
    skip_kinds = {
        EntityKind.METHOD.value,
        EntityKind.FUNCTION.value,
        EntityKind.KERNEL.value,
    }
    out = list(preferred)
    for row in readers:
        kind = str(row.get("kind") or "").upper()
        if kind in skip_kinds:
            continue
        key = (str(row.get("file") or ""), int(row.get("line") or 0))
        if key in seen or key[1] <= 0:
            continue
        seen.add(key)
        out.append(row)
    return out[:12]


def _timeline_site(row: dict[str, Any], *, role: str) -> dict[str, Any] | None:
    file, line = _site_loc(row)
    if not file or line <= 0:
        return None
    item = {
        "file": file,
        "line": line,
        "rhs": str(row.get("rhs") or (row.get("facts") or {}).get("rhs") or ""),
        "role": role,
    }
    fn = str(row.get("function") or row.get("name") or "")
    if fn:
        item["function"] = fn
    kind = str(row.get("kind") or "")
    if kind:
        item["kind"] = kind
    return item


def _order_pipes_by_lifetime(
    selected: list[dict[str, Any]],
    *,
    destroys: Sequence[tuple[int, str]],
    source_text: str,
) -> list[dict[str, Any]]:
    if not selected:
        return selected
    line_of = {
        str(hit.get("name") or ""): int(hit.get("line_start") or 0) for hit in selected
    }
    names = [str(hit.get("name") or "") for hit in selected]
    constructs = [(name, line_of[name]) for name in names if name]
    ranges = continued_line_ranges(source_text) if source_text else []
    ordered = order_pipe_names(
        names, constructs, destroys, ranges, line_of=line_of
    )
    if not ordered:
        return selected
    by_name = {str(hit.get("name") or ""): hit for hit in selected}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, name in enumerate(ordered, start=1):
        hit = by_name.get(name)
        if hit is None or name in seen:
            continue
        seen.add(name)
        item = dict(hit)
        facts = dict(item.get("facts") or {}) if isinstance(item.get("facts"), dict) else {}
        facts["pipe_ordinal"] = idx
        item["facts"] = facts
        out.append(item)
    for hit in selected:
        name = str(hit.get("name") or "")
        if name in seen:
            continue
        out.append(hit)
    return out


def _collapse_locate_hits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One location per (kind, name, file); extra lines stay on definition_sites."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        file = str(row.get("file") or "").replace("\\", "/")
        name = str(row.get("name") or "")
        kind = str(row.get("kind") or "")
        key = (kind, name, file)
        line = int(row.get("line_start") or row.get("line") or 0)
        if key not in grouped:
            host = dict(row)
            facts = dict(host.get("facts") or {}) if isinstance(host.get("facts"), dict) else {}
            sites = list(facts.get("definition_sites") or [])
            facts["definition_sites"] = sites
            host["facts"] = facts
            grouped[key] = host
            order.append(key)
        host = grouped[key]
        facts = host["facts"]
        sites = facts.setdefault("definition_sites", [])
        if line > 0 and file:
            if not any(
                int(site.get("line") or site.get("line_start") or 0) == line
                and str(site.get("file") or "").replace("\\", "/") == file
                for site in sites
                if isinstance(site, dict)
            ):
                sites.append({"file": file, "line": line, "line_start": line, "name": name})
        host_line = int(host.get("line_start") or 0)
        if line > 0 and (host_line <= 0 or line < host_line):
            host["line_start"] = line
            if row.get("snippet"):
                host["snippet"] = row["snippet"]
    return [grouped[key] for key in order]


def _dim_coverage_restricted(
    blocks: list[dict[str, Any]], filters: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """Union coverage, but pin filtered dims to the queried values."""
    pinned = {
        str(k): str(v)
        for k, v in dict(filters or {}).items()
        if str(k).strip() and str(v).strip()
    }
    coverage: dict[str, set[str]] = {}
    for row in blocks:
        fixed = row.get("fixed_fields") or {}
        domains = row.get("field_domains") or {}
        if isinstance(fixed, dict):
            for name, value in fixed.items():
                key = str(name)
                if key in pinned:
                    coverage.setdefault(key, set()).add(pinned[key])
                else:
                    coverage.setdefault(key, set()).add(str(value))
        if isinstance(domains, dict):
            for name, domain in domains.items():
                key = str(name)
                bucket = coverage.setdefault(key, set())
                if key in pinned:
                    bucket.add(pinned[key])
                    continue
                if isinstance(domain, (list, tuple, set)):
                    bucket.update(str(v) for v in domain)
                elif domain is not None:
                    bucket.add(str(domain))
    return {name: sorted(vals) for name, vals in coverage.items()}


def _fixed_coverage(blocks: list[dict[str, Any]]) -> dict[str, list[str]]:
    coverage: dict[str, set[str]] = {}
    for row in blocks:
        fixed = row.get("fixed_fields") or {}
        if not isinstance(fixed, dict):
            continue
        for name, value in fixed.items():
            coverage.setdefault(str(name), set()).add(str(value))
    return {name: sorted(vals) for name, vals in coverage.items()}


def _alias_ident_names(facts: dict[str, Any] | None) -> list[str]:
    names: list[str] = []
    blob = facts if isinstance(facts, dict) else {}
    for key in ("local_aliases", "fused_outer_candidates"):
        raw = blob.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            ident = str(item.get("name") or "").strip()
            if ident:
                names.append(ident)
    return names


def _alias_hit_rank(hit: dict[str, Any], needle: str) -> tuple[Any, ...]:
    """Prefer the tiling field with the strongest occupancy alias to this local."""
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    needle_l = str(needle or "").lower()
    count = 0
    hops = 99
    occupancy = 1
    for item in list(facts.get("local_aliases") or []) + list(
        facts.get("fused_outer_candidates") or []
    ):
        if not isinstance(item, dict):
            continue
        if str(item.get("name") or "").lower() != needle_l:
            continue
        count += 1
        hops = min(hops, int(item.get("hops") or 99))
        rhs = str(item.get("rhs") or "").lower()
        if any(tok in rhs for tok in ("aicnum", "corenum", "aivnum")):
            occupancy = 0
    kind = 0 if str(hit.get("kind") or "").upper() == EntityKind.TILING_FIELD.value else 1
    return (-count, occupancy, hops, kind, str(hit.get("name") or ""))


def _kind_priority(hit: dict[str, Any], needle: str) -> int:
    """Prefer tiling/host/method identities over VF ops, getters, and TYPE."""
    kind = str(hit.get("kind") or "").upper()
    ident = _last_ident(str(hit.get("name") or "")).lower()
    file = str(hit.get("file") or "").replace("\\", "/").lower()
    last_needle = _last_ident(str(needle or "").lower().replace(".", "::"))
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    if kind == EntityKind.TYPE.value and str(facts.get("catalog") or "") == "ge.graphStatus":
        return 0
    table = {
        EntityKind.TILING_KEY.value: 0,
        EntityKind.TILING_FIELD.value: 0,
        EntityKind.TILING_DATA.value: 0,
        EntityKind.INPUT.value: 0,
        EntityKind.OUTPUT.value: 0,
        EntityKind.MACRO.value: 0,
        EntityKind.KERNEL.value: 0,
        EntityKind.VARIABLE.value: 1,
        EntityKind.METHOD.value: 1,
        EntityKind.FIELD.value: 1,
        EntityKind.FUNCTION.value: 2,
        EntityKind.BUFFER.value: 1,
        EntityKind.REGISTER.value: 1,
        EntityKind.QUEUE.value: 1,
        EntityKind.PIPE.value: 1,
        EntityKind.EVENT.value: 1,
        EntityKind.COMPILE_VAR.value: 2,
        EntityKind.BRANCH.value: 3,
        EntityKind.PREDICATE.value: 3,
        EntityKind.OPERATION.value: 4,
        EntityKind.TYPE.value: 5,
    }
    score = table.get(kind, 3)
    if ident.startswith("get_") and ident != last_needle:
        score += 4
    if "/vector_api/" in file:
        score += 3
    if kind == EntityKind.TYPE.value and last_needle == "process":
        score += 2
    return score


def _units_cover_all(extra: Any, total: int) -> bool:
    """Whether the unit grouping already accounts for every match."""
    if not isinstance(extra, dict) or total <= 0:
        return False
    units = extra.get("units")
    if not isinstance(units, list) or not units:
        return False
    seen = sum(
        len(u.get("hits") or []) for u in units if isinstance(u, dict)
    ) + len(extra.get("leftover") or [])
    return seen >= total


def _constructor_rank(name: str, needle: str) -> int:
    """Demote `X::X` when the caller asked for a bare `X`.

    Candidates are ordered by kind, and `METHOD` sorts before `TYPE`, so a class
    loses to its own constructor for a name that named the class. Only that pair
    is reordered; any other same-named member keeps whatever rank it earned.
    """
    want = str(needle or "").strip().replace(".", "::")
    if not want or "::" in want:
        return 0
    return 1 if str(name or "").replace(".", "::") == f"{want}::{want}" else 0


def _agent_sort_key(
    hit: dict[str, Any], needle: str, *, architecture: str = ""
) -> tuple[Any, ...]:
    kind = str(hit.get("kind") or "")
    name = str(hit.get("name") or "")
    eid = str(hit.get("id") or "")
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    match = _name_rank(name, eid, needle.lower().strip(), exact_kind=False, kind=kind)
    if match is None:
        match = 9
    src, use = _definition_rank(kind, name, eid, facts)
    return (
        # An explicit `Owner::` outranks every other signal. The caller already
        # told us which of the same-named members they meant.
        int(hit.get("owner_rank") or 0),
        # Without this, kind order alone hands `Buffer` to `Buffer::Buffer`.
        _constructor_rank(name, needle),
        match,
        _occupancy_rank(hit),
        _kind_priority(hit, needle),
        _arch_file_rank(str(hit.get("file") or ""), architecture),
        _entry_rank(hit),
        src,
        use,
        int(hit.get("line_start") or 0),
        eid,
    )


def _drop_redundant_type_hashes(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    src_names = {
        str(hit.get("name") or "").lower()
        for hit in hits
        if str(hit.get("id") or "").startswith("SRCTYPE::")
    }
    if not src_names:
        return hits
    return [
        hit
        for hit in hits
        if not (
            str(hit.get("id") or "").startswith("TYPE_")
            and str(hit.get("name") or "").lower() in src_names
        )
    ]


def _round_robin_by_file(hits: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Keep rank order, but do not let one file fill the page."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for hit in hits:
        file = str(hit.get("file") or "").replace("\\", "/") or "(unknown)"
        if file not in buckets:
            buckets[file] = []
            order.append(file)
        buckets[file].append(hit)
    out: list[dict[str, Any]] = []
    depth = 0
    cap = max(1, int(limit))
    while len(out) < cap:
        progressed = False
        for file in order:
            bucket = buckets[file]
            if depth < len(bucket):
                out.append(bucket[depth])
                progressed = True
                if len(out) >= cap:
                    break
        if not progressed:
            break
        depth += 1
    return out


def _diversify_by_file(
    hits: list[dict[str, Any]], *, limit: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts: dict[str, int] = {}
    exemplars: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        file = str(hit.get("file") or "").replace("\\", "/") or "(unknown)"
        counts[file] = counts.get(file, 0) + 1
        if file in seen:
            continue
        seen.add(file)
        if len(exemplars) < max(0, int(limit)):
            exemplars.append(hit)
    return exemplars, counts


def _entry_rank(hit: dict[str, Any]) -> int:
    kind = str(hit.get("kind") or "").upper()
    name = str(hit.get("name") or "").lower()
    file = str(hit.get("file") or "").replace("\\", "/").lower()
    ident = _last_ident(name)
    if kind == EntityKind.KERNEL.value:
        return 0
    fname = file.rsplit("/", 1)[-1]
    if "entry" in fname:
        return 1
    if ident.startswith("invoke_"):
        return 1
    if file.endswith("_apt.cpp"):
        return 3
    if "processvec" in ident or ident in {"process", "processvec1", "processvec2"}:
        return 4
    return 2


def _hit_function(hit: dict[str, Any]) -> str:
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    return str(facts.get("function") or "").strip() or "(unknown)"


def _diversify_by_function(
    hits: list[dict[str, Any]], *, limit: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One exemplar per file first, then remaining unique functions.

    Ranking already put the richest bodies first; without a file pass, three
    functions in one translation unit crowd out the kernel entry ``if``.
    """
    counts: dict[str, int] = {}
    for hit in hits:
        fn = _hit_function(hit)
        counts[fn] = counts.get(fn, 0) + 1
    cap = max(0, int(limit))
    exemplars: list[dict[str, Any]] = []
    seen_fn: set[str] = set()
    seen_file: set[str] = set()
    for hit in hits:
        file = str(hit.get("file") or "").replace("\\", "/") or "(unknown)"
        if file in seen_file:
            continue
        seen_file.add(file)
        seen_fn.add(_hit_function(hit))
        exemplars.append(hit)
        if len(exemplars) >= cap:
            break
    for hit in hits:
        if len(exemplars) >= cap:
            break
        fn = _hit_function(hit)
        if fn in seen_fn:
            continue
        seen_fn.add(fn)
        exemplars.append(hit)
    return exemplars, counts


def _trivial_rhs(rhs: str) -> bool:
    return bool(_TRIVIAL_RHS_RE.match(str(rhs or "").strip().rstrip(";")))


def _hit_expr(hit: dict[str, Any]) -> str:
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    for key in ("rhs", "expression", "packing_expr"):
        text = str(facts.get(key) or "").strip()
        if text:
            return text
    return str(hit.get("name") or "").strip()


def _packing_site_sort_key(site: Any) -> tuple[Any, ...]:
    if not isinstance(site, dict):
        return (9, 9, 0, 9, 0)
    rhs = str(site.get("rhs") or "")
    fn = str(site.get("function") or "").strip()
    return (
        1 if _trivial_rhs(rhs) else 0,
        0 if fn else 1,
        -len(rhs),
        0 if site.get("guards") else 1,
        int(site.get("line") or 0),
    )


def _field_value_rank(hit: dict[str, Any]) -> tuple[Any, ...]:
    rhs = _hit_expr(hit)
    return (
        1 if _trivial_rhs(rhs) else 0,
        -len(rhs),
        -int(hit.get("line_start") or 0),
    )


def _write_site_sort_key(hit: dict[str, Any]) -> tuple[Any, ...]:
    rhs = _hit_expr(hit)
    return (
        1 if _trivial_rhs(rhs) else 0,
        -len(rhs),
        -int(hit.get("line_start") or 0),
        str(hit.get("id") or ""),
    )


def _branch_sort_key(hit: dict[str, Any]) -> tuple[Any, ...]:
    snippet = str(hit.get("snippet") or "")
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    fn = str(facts.get("function") or "")
    line = int(hit.get("line_start") or 0)
    constexpr_n = snippet.count("if constexpr")
    fn_offset = 1 if fn.endswith("Offset") else 0
    body_assigns = 0
    body_lines = 0
    data_move = 0
    offset_calls = 0
    started = False
    hit_indent: int | None = None
    for raw in snippet.splitlines():
        no = _snippet_line_no(raw)
        text = raw.split(":", 1)[1] if no is not None and ":" in raw else raw
        if no is not None and line > 0 and no < line:
            continue
        stripped = text.strip()
        indent = len(text) - len(text.lstrip(" \t"))
        if not started:
            if no is not None and line > 0 and no != line:
                continue
            started = True
            hit_indent = indent
            if _DATA_MOVE_RE.search(text):
                data_move = 1
            continue
        if stripped.startswith("}") and hit_indent is not None and indent <= hit_indent:
            break
        if not stripped or stripped.startswith("//"):
            continue
        body_lines += 1
        if _DATA_MOVE_RE.search(text):
            data_move = 1
        offset_calls += len(_GET_OFFSET_RE.findall(text))
        if "=" in stripped and not stripped.lstrip().startswith(("if", "for", "while")):
            body_assigns += 1
    return (
        fn_offset,
        0 if data_move else 1,
        0 if constexpr_n >= 2 else 1,
        -min(body_assigns, 12),
        offset_calls,
        -body_lines,
        line,
        str(hit.get("id") or ""),
    )


def _candidate_limit(limit: int) -> int:
    return max(0, min(int(limit), PRIMARY_CANDIDATES))


def _snippet_line_no(raw: str) -> int | None:
    prefix = str(raw or "").split(":", 1)[0].strip()
    if prefix.isdigit():
        return int(prefix)
    return None


def _snippet_covers_line(snippet: str, line: int) -> bool:
    want = int(line or 0)
    if want <= 0:
        return bool(str(snippet or "").strip())
    return any(_snippet_line_no(row) == want for row in str(snippet or "").splitlines())


def _clip_snippet_around_line(snippet: str, line: int, *, max_lines: int) -> str:
    rows = str(snippet or "").splitlines()
    if len(rows) <= max_lines:
        return str(snippet or "")
    idx = 0
    want = int(line or 0)
    if want > 0:
        for i, raw in enumerate(rows):
            if _snippet_line_no(raw) == want:
                idx = i
                break
    before = min(SNIPPET_BEFORE, idx)
    start = max(0, idx - before)
    end = min(len(rows), start + max_lines)
    if end - start < max_lines:
        start = max(0, end - max_lines)
    return "\n".join(rows[start:end])


def _cap_snippet(text: str, line_start: int) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    lines = raw.splitlines() or [raw]
    lines = lines[:SNIPPET_LINES]
    start = int(line_start or 0)
    if start <= 0 or (lines and lines[0][:1].isdigit() and (":" in lines[0][:8] or "|" in lines[0][:8])):
        return "\n".join(lines)
    return "\n".join(f"{start + offset}:{line}" for offset, line in enumerate(lines))


#: `entity` rows overlapping a line range in one file. Two things kept the
#: planner off `idx_entity_file_line` here: `file = ? OR file LIKE '%' || ?` in
#: one statement, where the leading wildcard rules the index out for the whole
#: disjunction, and `IFNULL(file, '') = ?`, which is a function of the column
#: rather than the column. Splitting the probes and comparing `file` directly
#: turns a 20k-row scan into a seek. The needle is non-empty, so dropping IFNULL
#: changes no result: NULL never equals it either way. The suffix probe stays for
#: a caller that pastes a path spelled from somewhere else.
_SEED_BY_FILE_SQL = """
    SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
           IFNULL(s.snippet, '') AS snippet
    FROM entity e
    LEFT JOIN source_span s ON s.entity_id = e.id
    WHERE {predicate}
      AND IFNULL(e.line_end, e.line_start) >= ?
      AND IFNULL(e.line_start, 0) <= ?
      AND IFNULL(e.line_start, 0) > 0
    {order}
    LIMIT ?
"""


def _alternate_file_spelling(needle: str) -> str:
    """A second spelling to try when the first found nothing, or ``''``.

    Callers paste paths from logs and editors, so a needle may carry a prefix the
    product never stored. Cutting back to the operator subdirectory, or failing
    that to the basename, is what turns such a paste into a lookup.
    """
    rel = needle
    for marker in ("op_host/", "op_kernel/", "common/", "tests/"):
        at = needle.find(marker)
        if at >= 0:
            rel = needle[at:]
            break
    alt = rel if rel != needle else needle.rsplit("/", 1)[-1]
    return alt if alt and alt != needle else ""


def _seed_rows_for_file(
    conn: sqlite3.Connection,
    needle: str,
    start: int,
    end: int,
    limit: int,
    *,
    order: str = "",
) -> list[Any]:
    """Entities overlapping `start..end` in `needle`, cheapest probe first."""
    if not needle:
        return []
    exact = conn.execute(
        _SEED_BY_FILE_SQL.format(predicate="e.file = ?", order=order),
        (needle, start, end, limit),
    ).fetchall()
    if exact:
        return exact
    return conn.execute(
        _SEED_BY_FILE_SQL.format(
            predicate="IFNULL(e.file, '') LIKE '%' || ?", order=order
        ),
        ("/" + needle.lstrip("/"), start, end, limit),
    ).fetchall()


# `source_line` is indexed by (path, line), but an agent spells a file the way
# a card cited it. Matching that spelling inside the WHERE clause
# (`REPLACE(path, ...) LIKE '%' || ?`) makes SQLite drop `idx_source_line_path`
# and scan every indexed line. One resolve issues well over a hundred of these.
# A committed `.uo` is immutable, so the spelling is resolved to the stored path
# once per snapshot and the file body is held whole.
_SRC_STATE_LOCK = threading.Lock()
_SRC_STATE: "OrderedDict[int, tuple[sqlite3.Connection, dict[str, Any]]]" = OrderedDict()
_SRC_STATE_MAX = 8
_SRC_FILE_MAX = 24


def reset_source_line_cache() -> None:
    with _SRC_STATE_LOCK:
        _SRC_STATE.clear()


def drop_source_line_cache(conn: sqlite3.Connection) -> None:
    """Forget one connection's snapshot state.

    The cache is keyed by ``id(conn)`` and pins the object so the id cannot be
    reused. That also keeps a closed handle from being collected, which on
    Windows blocks deleting the ``.uo``; the reader drops the entry as part of
    closing.
    """
    with _SRC_STATE_LOCK:
        _SRC_STATE.pop(id(conn), None)


def _src_state(conn: sqlite3.Connection) -> dict[str, Any]:
    key = id(conn)
    with _SRC_STATE_LOCK:
        hit = _SRC_STATE.get(key)
        if hit is not None and hit[0] is conn:
            _SRC_STATE.move_to_end(key)
            return hit[1]
        state: dict[str, Any] = {
            "has_source_line": None,
            "paths": None,
            "resolved": {},
            "files": OrderedDict(),
        }
        _SRC_STATE[key] = (conn, state)
        _SRC_STATE.move_to_end(key)
        while len(_SRC_STATE) > _SRC_STATE_MAX:
            _SRC_STATE.popitem(last=False)
        return state


def _has_source_line(conn: sqlite3.Connection) -> bool:
    state = _src_state(conn)
    if state["has_source_line"] is None:
        from ascendc_codemap_mcp.engine.store.accel import has_source_line

        state["has_source_line"] = bool(has_source_line(conn))
    return bool(state["has_source_line"])


def _source_paths(conn: sqlite3.Connection) -> list[str]:
    state = _src_state(conn)
    if state["paths"] is None:
        try:
            rows = conn.execute("SELECT DISTINCT path FROM source_line").fetchall()
        except sqlite3.OperationalError:
            rows = []
        state["paths"] = sorted({str(r[0] or "") for r in rows if r and r[0]})
    return list(state["paths"])


def _snapshot_path_for(conn: sqlite3.Connection, file: str) -> str:
    """Stored ``source_line.path`` for the spelling an agent passed."""
    needle = str(file or "").replace("\\", "/")
    if not needle:
        return ""
    state = _src_state(conn)
    cache: dict[str, str] = state["resolved"]
    hit = cache.get(needle)
    if hit is not None:
        return hit
    leaf = needle.rsplit("/", 1)[-1]
    rel = _rel_key(needle)
    exact: list[str] = []
    suffix: list[str] = []
    by_leaf: list[str] = []
    for path in _source_paths(conn):
        norm = path.replace("\\", "/")
        if norm == needle or _rel_key(norm) == rel:
            exact.append(path)
        elif norm.endswith("/" + needle) or needle.endswith("/" + norm):
            suffix.append(path)
        elif leaf and (norm == leaf or norm.endswith("/" + leaf)):
            by_leaf.append(path)
    chosen = (exact or suffix or by_leaf or [""])[0]
    cache[needle] = chosen
    return chosen


_CLASS_DECL_RE = re.compile(r"^\s*(?:class|struct)\s+\w")
#: A member is at most this far below its class header. Bounds the upward walk
#: so a file with no enclosing template does not scan to line 1.
_TEMPLATE_SCAN_LIMIT = 600


def _enclosing_template_header(text: dict[int, str], line: int) -> str:
    """The ``template <...>`` header of the class enclosing *line*, if any.

    Walks up to the nearest ``class`` / ``struct`` header, then takes the
    preceding non-blank line when it opens a template. Multi-line headers are
    joined until the angle brackets balance.
    """
    start = 0
    for probe in range(int(line) - 1, max(0, int(line) - _TEMPLATE_SCAN_LIMIT), -1):
        if _CLASS_DECL_RE.match(text.get(probe, "")):
            start = probe
            break
    if not start:
        return ""
    above = 0
    for probe in range(start - 1, max(0, start - 12), -1):
        src = text.get(probe, "").strip()
        if not src or src.startswith(("//", "/*", "*")):
            continue
        above = probe
        break
    if not above or not text.get(above, "").strip().startswith("template"):
        return ""
    parts: list[str] = []
    for probe in range(above, start):
        src = text.get(probe, "")
        if not src:
            continue
        parts.append(src.strip())
        joined = " ".join(parts)
        if joined.count("<") and joined.count("<") == joined.count(">"):
            return joined
    return " ".join(parts)


def _snapshot_file_lines(conn: sqlite3.Connection, path: str) -> list[tuple[int, str]]:
    if not path:
        return []
    state = _src_state(conn)
    files: "OrderedDict[str, list[tuple[int, str]]]" = state["files"]
    hit = files.get(path)
    if hit is not None:
        files.move_to_end(path)
        return hit
    try:
        rows = conn.execute(
            "SELECT line, text FROM source_line WHERE path = ? ORDER BY line",
            (path,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    body = [(int(r[0] or 0), str(r[1] or "")) for r in rows]
    files[path] = body
    files.move_to_end(path)
    while len(files) > _SRC_FILE_MAX:
        files.popitem(last=False)
    return body


def _source_line_window(
    conn: sqlite3.Connection,
    file: str,
    line: int,
    *,
    before: int = STATEMENT_BEFORE,
    after: int = STATEMENT_AFTER,
) -> str:
    """~16-line window from indexed `source_line`. Empty if the table is absent."""
    if not file or int(line or 0) <= 0:
        return ""
    centre = int(line)
    rows = _source_line_rows(conn, file, max(1, centre - int(before)), centre + int(after))
    if not rows:
        return ""
    return "\n".join(f"{ln}:{txt}" for ln, txt in rows)


def _source_line_rows(
    conn: sqlite3.Connection, file: str, start: int, end: int
) -> list[tuple[int, str]]:
    if not file or int(start or 0) <= 0:
        return []
    if not _has_source_line(conn):
        return []
    path = _snapshot_path_for(conn, file)
    if not path:
        return []
    body = _snapshot_file_lines(conn, path)
    if not body:
        return []
    lo = max(1, int(start))
    hi = max(lo, int(end))
    left = bisect_left(body, lo, key=lambda row: row[0])
    right = bisect_right(body, hi, key=lambda row: row[0])
    return body[left:right]


def _snapshot_line_range(
    conn: sqlite3.Connection, file: str
) -> tuple[int, int] | None:
    """First and last line the snapshot holds for a file, if it holds any."""
    if not file or not _has_source_line(conn):
        return None
    path = _snapshot_path_for(conn, file)
    if not path:
        return None
    body = _snapshot_file_lines(conn, path)
    if not body:
        return None
    return int(body[0][0]), int(body[-1][0])


def _squash_path(text: str) -> str:
    """Letters and digits only, so `PostRegbase` can meet `post_regbase`."""
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _glob_nearest(conn: sqlite3.Connection, glob: str) -> list[str] | None:
    """Paths a `file=` glob was reaching for, when it selects no file at all.

    A glob that names no file and a glob that names files holding no match are
    different failures, and only the second says anything about the pattern.
    Returning None for the second lets a caller tell them apart instead of
    reporting both as a plain zero.
    """
    from ascendc_codemap_mcp.engine.query.rg import path_matches

    pat = str(glob or "").strip()
    if not pat:
        return None
    paths = _source_paths(conn)
    if any(path_matches(p, pat) for p in paths):
        return None
    # Globs get built out of CamelCase class names while files are spelled in
    # snake_case, so the comparison drops separators and case on both sides.
    literals = sorted(
        (t for t in re.split(r"[*?/\[\]]+", pat) if len(_squash_path(t)) >= 4),
        key=len,
        reverse=True,
    )
    for token in literals:
        key = _squash_path(token)
        hits = [p for p in paths if key in _squash_path(p)]
        if hits:
            return sorted(hits)[:6]
    return []


def _restore_blank_lines(
    rows: list[tuple[int, str]], stop: int
) -> list[tuple[int, str]]:
    """Put back the blank lines the index does not store.

    `source_line` holds no row for an empty line, so a body read back from it
    arrives with holes: the numbers jump, and a coverage note comparing rows to
    span length reports a complete body as truncated. Readers then spend a call
    fetching lines that were never missing.
    """
    if not rows:
        return rows
    out: list[tuple[int, str]] = []
    expect = rows[0][0]
    for number, text in rows:
        while expect < number:
            out.append((expect, ""))
            expect += 1
        out.append((number, text))
        expect = number + 1
    while expect <= int(stop or 0):
        out.append((expect, ""))
        expect += 1
    return out


_TEMPLATE_HEAD_RE = re.compile(r"^\s*template\s*<")
_TEMPLATE_HEAD_KINDS = frozenset(
    {
        EntityKind.TYPE.value,
        EntityKind.METHOD.value,
        EntityKind.FUNCTION.value,
        EntityKind.KERNEL.value,
    }
)


def _template_head_start(
    conn: sqlite3.Connection, file: str, start: int, *, look_back: int = 6
) -> int:
    """First line of the ``template <…>`` header above a declaration.

    A class body is not readable without it. Every mention of ``bufferType``
    or ``syncMode`` inside refers to a parameter declared on the line above the
    snippet, together with the default that decides the behaviour — and a
    reader who cannot see it spends two searches recovering what one line of
    context would have carried. Returns ``start`` when there is no header.
    """
    if int(start or 0) <= 1:
        return int(start or 1)
    rows = _source_line_rows(conn, file, max(1, start - look_back), start - 1)
    for index, (line_no, text) in enumerate(rows):
        if not _TEMPLATE_HEAD_RE.match(str(text or "")):
            continue
        # A header can wrap over several lines, so it is only this declaration's
        # once its angle brackets close by the line above.
        joined = " ".join(str(t or "") for _n, t in rows[index:])
        if joined.count("<") <= joined.count(">"):
            return int(line_no)
    return int(start)


def _snapshot_window(
    conn: sqlite3.Connection,
    file: str,
    line: int,
    *,
    kind: str = "",
    line_end: int = 0,
) -> tuple[str, bool, list[dict[str, Any]]]:
    """Snippet from indexed source_line only. Never reads the working tree."""
    empty: list[dict[str, Any]] = []
    if not file or int(line or 0) <= 0:
        return "", False, empty
    centre = int(line)
    kind_u = str(kind or "").upper()
    span_end = int(line_end or 0)
    if kind_u in _STMT_EXPAND_KINDS:
        window = _source_line_window(conn, file, line)
        return (window, False, empty) if window else ("", False, empty)
    start = centre if kind_u in {
        EntityKind.METHOD.value,
        EntityKind.FUNCTION.value,
        EntityKind.KERNEL.value,
    } else max(1, centre - SNIPPET_BEFORE)
    if kind_u in _TEMPLATE_HEAD_KINDS:
        start = min(start, _template_head_start(conn, file, start))
    if span_end > centre:
        end = min(span_end, start + SNIPPET_LINES - 1)
        truncated = span_end > end
    else:
        end = start + SNIPPET_LINES - 1
        truncated = False
    rows = _source_line_rows(conn, file, start, end)
    if not rows:
        return "", False, empty
    return "\n".join(f"{ln}:{txt}" for ln, txt in rows), truncated, empty


def _source_file_text(conn: sqlite3.Connection, file: str) -> str:
    rows = _source_line_rows(conn, file, 1, 100000)
    if not rows:
        return ""
    return "\n".join(txt for _ln, txt in rows)


def _snapshot_statement(conn: sqlite3.Connection, file: str, line: int) -> str:
    rows = _source_line_rows(conn, file, int(line or 0), int(line or 0) + 23)
    buf: list[str] = []
    for _ln, txt in rows:
        buf.append(str(txt).rstrip())
        if ";" in txt:
            break
    return " ".join(buf)


def _snapshot_snippet(
    conn: sqlite3.Connection, file: str, line: int, *, kind: str = "", line_end: int = 0
) -> str:
    text, _truncated, _omitted = _snapshot_window(
        conn, file, line, kind=kind, line_end=line_end
    )
    return text


def _is_trivial_decl(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind") or "")
    rhs = str(row.get("rhs") or "").strip()
    return kind == "declaration" and bool(_TRIVIAL_RHS_RE.match(rhs))


def _rhs_looks_decision(rhs: str) -> bool:
    return bool(_DECISION_RHS_RE.search(str(rhs or "")))


def _focus_value_write(extras: dict[str, Any] | None) -> dict[str, Any] | None:
    rows = [
        row
        for row in (extras or {}).get("value_writes") or []
        if isinstance(row, dict)
    ]
    decision = [
        row
        for row in rows
        if not _is_trivial_decl(row) and _rhs_looks_decision(str(row.get("rhs") or ""))
    ]
    if not decision:
        return None

    def _rank(row: dict[str, Any]) -> tuple[int, int]:
        file = str(row.get("file") or "").replace("\\", "/").lower()
        cpp = 0 if file.endswith(".cpp") else 1
        return (cpp, int(row.get("line") or 0))

    return sorted(decision, key=_rank)[0]


def _focus_idents(rhs: str, needle: str) -> list[str]:
    needle_l = _last_ident(needle).lower()
    preferred: list[str] = []
    rest: list[str] = []
    for tok in _TOKEN_RE.findall(str(rhs or "")):
        low = tok.lower()
        if low in _FOCUS_SKIP_IDENTS or low == needle_l:
            continue
        if tok.isupper() or (tok[:1].isdigit()):
            continue
        if tok.startswith(("fBase", "tndBase", "context")):
            continue
        mixed = any(ch.isupper() for ch in tok[1:]) and any(ch.islower() for ch in tok)
        tagged = tok.endswith(("Cond", "Limit", "Support")) or any(
            part in tok for part in ("Cond", "Limit", "Support")
        )
        if tagged:
            if tok not in preferred:
                preferred.append(tok)
        elif mixed and tok not in rest and tok not in preferred:
            rest.append(tok)
        if len(preferred) >= 3:
            break
    return (preferred + rest)[:3]


def _overlay_value_write(
    grouped: dict[str, Any],
    focus: dict[str, Any],
    needle: str,
) -> dict[str, Any]:
    grouped = dict(grouped)
    writes = dict(grouped.get("WRITES") or {"count": 0, "neighbors": []})
    rhs = str(focus.get("rhs") or "")
    idents = _focus_idents(rhs, needle)
    fn = str(focus.get("function") or "").strip()
    if fn:
        name, kind = fn, EntityKind.FUNCTION.value
    elif idents:
        name, kind = idents[0], EntityKind.VARIABLE.value
    else:
        name, kind = str(needle or ""), EntityKind.VARIABLE.value
    focus_row = {
        "name": name,
        "kind": kind,
        "file": str(focus.get("file") or ""),
        "line": int(focus.get("line") or 0),
    }
    loc = (focus_row["file"].replace("\\", "/"), focus_row["line"])
    rest: list[dict[str, Any]] = []
    for row in list(writes.get("neighbors") or []):
        other = (str(row.get("file") or "").replace("\\", "/"), int(row.get("line") or 0))
        if other == loc:
            continue
        rest.append(row)
    writes["neighbors"] = [focus_row] + rest[: max(0, EDGES_PER_KIND - 1)]
    writes["count"] = max(int(writes.get("count") or 0), len(writes["neighbors"]))
    grouped["WRITES"] = writes
    return grouped


def _queryable_hop(name: str) -> bool:
    """True when ``name`` can be fed back as a uo-query identifier."""
    text = str(name or "").strip()
    if not text or text.startswith("ARGS_SEL"):
        return False
    if any(ch in text for ch in (" ", "(", ")", "!", "<", ">", "=", ",", "[", "]")):
        return False
    parts = text.replace("::", ".").split(".")
    return bool(parts) and all(_IDENT_NAME_RE.fullmatch(part) for part in parts if part)


def _append_next_name(
    names: list[str],
    seen: set[str],
    self_names: set[str],
    candidate: str,
    *,
    limit: int = NEXT_HOP_LIMIT,
) -> bool:
    other = str(candidate or "").strip()
    if (
        not other
        or not _queryable_hop(other)
        or other.lower() in self_names
        or other.lower() in seen
    ):
        return False
    seen.add(other.lower())
    names.append(other)
    return len(names) >= limit


def _extend_next_from_edges(
    names: list[str],
    seen: set[str],
    self_names: set[str],
    grouped: dict[str, Any],
    *,
    limit: int = NEXT_HOP_LIMIT,
) -> None:
    queues: list[list[Any]] = []

    def _queue(group: Any) -> None:
        if not isinstance(group, dict):
            return
        rows = [row for row in list(group.get("neighbors") or []) if isinstance(row, dict)]
        if rows:
            queues.append(rows)

    for rel in _NEXT_EDGE_PRIORITY:
        _queue(grouped.get(rel))
    for rel, group in grouped.items():
        if rel in _NEXT_EDGE_PRIORITY or rel in _NEXT_EDGE_SKIP:
            continue
        _queue(group)
    if not queues:
        return
    index = 0
    while len(names) < limit and any(queues):
        bucket = queues[index % len(queues)]
        index += 1
        if not bucket:
            continue
        row = bucket.pop(0)
        if not str(row.get("file") or "").strip():
            continue
        if str(row.get("kind") or "").upper() == EntityKind.TYPE.value:
            continue
        if _append_next_name(
            names, seen, self_names, str(row.get("name") or ""), limit=limit
        ):
            return


def _extend_next_from_value_expr(
    names: list[str],
    seen: set[str],
    self_names: set[str],
    expr: str,
    *,
    limit: int = NEXT_HOP_LIMIT,
) -> None:
    for tok in identifier_tokens(expr):
        if tok.lower() in _FOCUS_SKIP_IDENTS:
            continue
        if _append_next_name(names, seen, self_names, tok, limit=limit):
            return


def _rhs_looks_truncated(rhs: str) -> bool:
    text = str(rhs or "")
    if len(text) >= PACKING_RHS_TRIM:
        return True
    stripped = text.rstrip()
    return stripped.endswith(("&&", "||", "(", ",", "+"))


def _template_block_rows(blob: Any) -> list[dict[str, Any]]:
    if not isinstance(blob, dict):
        return []
    for key in ("groups", "blocks", "rows", "template_blocks"):
        rows = blob.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


def _load_template_blocks_cached(product: Path) -> dict[str, Any]:
    key = str(product)
    try:
        mtime_ns = product.stat().st_mtime_ns
    except OSError:
        return {"ok": False, "reason_code": "UO_PRODUCT_MISSING", "rows": [], "mtime_ns": -1}
    with _TEMPLATE_BLOCKS_LOCK:
        hit = _TEMPLATE_BLOCKS_CACHE.get(key)
        if hit and hit.get("mtime_ns") == mtime_ns:
            _TEMPLATE_BLOCKS_CACHE.move_to_end(key)
            return hit
    from ascendc_codemap_mcp.engine.store.reader import load_view_blob_checked

    checked = load_view_blob_checked(
        product,
        "tiling/template_blocks.yaml",
        fallback_canonical=False,
    )
    entry = {
        "ok": bool(checked.get("ok")),
        "reason_code": str(checked.get("reason_code") or ""),
        "mtime_ns": mtime_ns,
        "rows": _template_block_rows(checked.get("view")) if checked.get("ok") else [],
    }
    with _TEMPLATE_BLOCKS_LOCK:
        _TEMPLATE_BLOCKS_CACHE[key] = entry
        _TEMPLATE_BLOCKS_CACHE.move_to_end(key)
        while len(_TEMPLATE_BLOCKS_CACHE) > _VIEW_CACHE_MAX:
            _TEMPLATE_BLOCKS_CACHE.popitem(last=False)
    return entry


def _value_matches_domain(value: str, domain: Any) -> bool:
    from ascendc_codemap_mcp.engine.tpl_dsl import bool_value_aliases

    want = {str(value)}
    want.update(bool_value_aliases(value))
    want_l = {v.lower() for v in want}
    values: list[str]
    if isinstance(domain, (list, tuple, set)):
        values = [str(v) for v in domain]
    else:
        values = [str(domain)]
    return any(item in want or item.lower() in want_l for item in values)


def _template_block_matches(row: dict[str, Any], filters: dict[str, str]) -> bool:
    fixed = row.get("fixed_fields") or {}
    domains = row.get("field_domains") or {}
    if not isinstance(fixed, dict):
        fixed = {}
    if not isinstance(domains, dict):
        domains = {}
    for name, value in filters.items():
        if name in fixed:
            if not _value_matches_domain(str(value), [fixed[name]]):
                return False
            continue
        if name in domains:
            if not _value_matches_domain(str(value), domains[name]):
                return False
            continue
        return False
    return True


def _fts_match_query(needle: str) -> str:
    """Quote a substring so FTS5 trigram treats it as a phrase."""
    text = str(needle or "").replace('"', " ").strip()
    return f'"{text}"' if text else ""


def _fts_and_query(tokens: Sequence[str]) -> str:
    """AND of quoted phrases. Does not expand abbreviations."""
    parts = [_fts_match_query(tok) for tok in tokens if str(tok or "").strip()]
    return " AND ".join(part for part in parts if part)


def _search_phrase_tokens(phrase: str) -> list[str]:
    """Split a missed phrase into AND tokens. No abbrev expansion."""
    text = str(phrase or "").strip()
    seen: list[str] = []
    for tok in text.split():
        core = tok.replace("*", "").replace("?", "").replace("%", "").strip()
        if len(core) >= 3 and core not in seen:
            seen.append(core)
    for tok in _ident_tokens(text):
        if len(tok) >= 3 and tok not in seen:
            seen.append(tok)
    return seen


def _compact_template_block(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id") or "",
        "name": row.get("name") or "",
        "sel_group_index": row.get("sel_group_index"),
        "fixed_fields": row.get("fixed_fields") or {},
        "field_domains": row.get("field_domains") or {},
        "product_count": row.get("product_count"),
    }


def _collect_block_dim_values(row: dict[str, Any], dim_name: str) -> list[str]:
    values: list[str] = []
    fixed = row.get("fixed_fields") or {}
    domains = row.get("field_domains") or {}
    if isinstance(fixed, dict) and dim_name in fixed:
        values.append(str(fixed[dim_name]))
    domain = domains.get(dim_name) if isinstance(domains, dict) else None
    if isinstance(domain, (list, tuple, set)):
        values.extend(str(v) for v in domain)
    elif domain is not None:
        values.append(str(domain))
    return values


def _dim_coverage(blocks: list[dict[str, Any]]) -> dict[str, list[str]]:
    coverage: dict[str, set[str]] = {}
    for row in blocks:
        fixed = row.get("fixed_fields") or {}
        domains = row.get("field_domains") or {}
        if isinstance(fixed, dict):
            for name, value in fixed.items():
                coverage.setdefault(str(name), set()).add(str(value))
        if isinstance(domains, dict):
            for name, domain in domains.items():
                bucket = coverage.setdefault(str(name), set())
                if isinstance(domain, (list, tuple, set)):
                    bucket.update(str(v) for v in domain)
                elif domain is not None:
                    bucket.add(str(domain))
    return {name: sorted(vals) for name, vals in coverage.items()}


def _template_nearby(
    all_blocks: list[dict[str, Any]], filters: dict[str, str]
) -> list[dict[str, Any]]:
    nearby: list[dict[str, Any]] = []
    for dropped in filters:
        remaining = {k: v for k, v in filters.items() if k != dropped}
        matched = [
            row
            for row in all_blocks
            if not remaining or _template_block_matches(row, remaining)
        ]
        values: set[str] = set()
        for row in matched:
            values.update(_collect_block_dim_values(row, dropped))
        nearby.append(
            {
                "dropped": dropped,
                "remaining_filters": remaining,
                "matching_block_count": len(matched),
                "values": sorted(values),
            }
        )
    return nearby


def _file_index_entry(hit: dict[str, Any]) -> dict[str, Any]:
    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
    return {
        "id": hit.get("id"),
        "name": hit.get("name"),
        "line": hit.get("line_start"),
        "function": facts.get("function") or "",
    }


def _group_by_file(hits: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        file = str(hit.get("file") or "")
        if not file:
            continue
        grouped.setdefault(file, []).append(_file_index_entry(hit))
    return grouped


def _clip_hit_snippets(rows: list[Any], *, max_lines: int) -> None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        snip = row.get("snippet")
        if not isinstance(snip, str) or snip.count("\n") + 1 <= max_lines:
            continue
        row["snippet"] = _clip_snippet_around_line(
            snip, int(row.get("line_start") or 0), max_lines=max_lines
        )


def _clip_definition_snippet(text: str, *, max_lines: int = CARD_SNIPPET_MAX_LINES) -> tuple[str, bool]:
    lines = str(text or "").splitlines()
    if len(lines) <= max_lines:
        return str(text or ""), False
    return "\n".join(lines[:max_lines]), True


def _compact_kind_card(hit: dict[str, Any]) -> dict[str, Any]:
    """Search-row pointer: kind + location, no second copy of extras/snippet."""
    return {
        "kind": str(hit.get("kind") or ""),
        "name": str(hit.get("name") or ""),
        "id": str(hit.get("id") or ""),
        "file": str(hit.get("file") or ""),
        "line": int(hit.get("line_start") or hit.get("line") or 0),
    }


def _matched_entity_rows(hits: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    """The entities a name matched, so a count of them can be checked.

    The coverage line quotes how many things the name reached. Without the
    list, a reader cannot tell three definitions from one definition and two
    look-alikes, and both readings change what they are allowed to conclude.
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        name = str(hit.get("name") or "")
        kind = str(hit.get("kind") or "")
        file = _norm_file(str(hit.get("file") or ""))
        line = int(hit.get("line_start") or hit.get("line") or 0)
        key = (name, kind, file, line)
        if not name or key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "kind": kind, "file": file, "line": line})
        if len(out) >= limit:
            break
    return out


def _name_card_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "completeness",
        "answerable",
        "definition_sites_count",
        "definition_sites_complete",
        "matched_entities_count",
    )
    return {key: coverage[key] for key in keep if key in coverage}


def _compact_around_hit(hit: dict[str, Any], *, snippet: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": hit.get("name"),
        "kind": hit.get("kind"),
        "file": _norm_file(str(hit.get("file") or "")),
        "line": int(hit.get("line_start") or hit.get("line") or 0),
        "id": hit.get("id"),
    }
    if snippet:
        clipped, truncated = _card_snippet_for_hit(hit)
        if clipped:
            row["snippet"] = clipped
        if truncated:
            row["truncated"] = True
    return {key: value for key, value in row.items() if value not in (None, "")}


def _clip_snippets(
    payload: dict[str, Any], *, max_lines: int, keep_primary: bool = False
) -> None:
    for key in (
        "rows",
        "branches",
        "calls",
        "buffers",
        "hits",
        "locations",
        "keys",
        "templates",
        "candidates",
        "writers",
        "readers",
        "fields",
        "phases",
        "cards",
    ):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        # The body of the thing that was asked for is the answer, not context;
        # trimming it to three lines while every neighbour list survives spends
        # the budget backwards.
        if keep_primary and key == "cards" and rows:
            _clip_hit_snippets(rows[1:], max_lines=max_lines)
        else:
            _clip_hit_snippets(rows, max_lines=max_lines)
    field = payload.get("field")
    if isinstance(field, dict):
        _clip_hit_snippets([field], max_lines=max_lines)
    contract = payload.get("contract")
    if isinstance(contract, dict):
        for key in ("producers", "consumers", "impact_sinks", "binds", "kernel_repr", "entry"):
            rows = contract.get(key)
            if isinstance(rows, list):
                _clip_hit_snippets(rows, max_lines=max_lines)


def _clip_relationships(payload: dict[str, Any], *, max_rels: int = MAX_REL_HOPS) -> None:
    for key in (
        "rows",
        "branches",
        "calls",
        "buffers",
        "hits",
        "locations",
        "keys",
        "templates",
        "fields",
    ):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            rels = row.get("relationships")
            if isinstance(rels, list) and len(rels) > max_rels:
                row["relationships"] = rels[:max_rels]


def _payload_size(payload: dict[str, Any]) -> int:
    """Budget the facts, not the rendering of them.

    `text` is a markdown mirror of the same cards, regenerated after fitting, so
    counting it charges every fact twice and the second charge is paid by the
    definition body -- the one part of the card that cannot be recovered from
    anywhere else.
    """
    if "text" in payload:
        payload = {k: v for k, v in payload.items() if k != "text"}
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))


def _page_by_exactness(
    hits: list[dict[str, Any]], needle: str, *, limit: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep exact ident / ``::`` members on the first page; substring later."""
    exact: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    n = str(needle or "").lower().strip()
    exact_ids: set[str] = set()
    for hit in hits:
        eid = str(hit.get("id") or hit.get("entity_id") or "")
        rank = _name_rank(
            str(hit.get("name") or ""),
            eid,
            n,
            exact_kind=False,
            kind=str(hit.get("kind") or ""),
        )
        if rank in (0, 1):
            exact.append(hit)
            if eid:
                exact_ids.add(eid)
        else:
            rest.append(hit)
    if exact_ids:
        kept_rest: list[dict[str, Any]] = []
        for hit in rest:
            if str(hit.get("id") or hit.get("entity_id") or "") in exact_ids:
                exact.append(hit)
            else:
                kept_rest.append(hit)
        rest = kept_rest
    cap = max(0, int(limit))
    if exact:
        page = exact[:cap]
        return page, {
            "total": len(exact),
            "clipped": len(exact) > cap,
            "substring_only": False,
            "all_matched": len(hits),
        }
    page = rest[:cap]
    return page, {
        "total": len(rest),
        "clipped": len(rest) > cap,
        "substring_only": True,
        "all_matched": len(hits),
    }


def _hits_coverage(
    hits: list[dict[str, Any]],
    *,
    total: int | None = None,
    dim_coverage: dict[str, Any] | None = None,
    clipped: bool = False,
    needle: str = "",
    substring_only: bool = False,
) -> dict[str, Any]:
    sibling_files: list[str] = []
    seen: set[str] = set()
    mutex: list[str] = []
    phases: list[str] = []
    def_count = 0
    fused_count = 0
    for hit in hits:
        file = str(hit.get("file") or "").replace("\\", "/")
        if file and file not in seen:
            seen.add(file)
            sibling_files.append(file)
        facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
        policy = str(facts.get("mutex_policy") or "").strip()
        if facts.get("wraps_lock") and "lock" not in mutex:
            mutex.append("lock")
        if facts.get("allocated") and "allocated" not in mutex:
            mutex.append("allocated")
        if policy and policy not in mutex:
            mutex.append(policy)
        phase = str(facts.get("pipe_ordinal") or facts.get("kernel_phase") or "").strip()
        if phase and phase not in phases:
            phases.append(phase)
        if _is_use_site_hit(
            str(hit.get("kind") or ""),
            facts,
            hit.get("attrs") if isinstance(hit.get("attrs"), dict) else {},
            hit.get("data") if isinstance(hit.get("data"), dict) else {},
        ):
            continue
        sites = facts.get("definition_sites")
        if isinstance(sites, list):
            # Two entity kinds can share one declaration: `ping_` is both the
            # FIELD and the BUFFER minted from it, on the same line. Counting
            # rows reported that as two sites, so the list rendered under the
            # count was always shorter than the count claimed.
            distinct = {
                (str(s.get("file") or ""), int(s.get("line") or 0))
                for s in sites
                if isinstance(s, dict) and s.get("file")
            }
            def_count = max(def_count, len(distinct) or len(sites))
        elif isinstance(sites, int):
            def_count = max(def_count, int(sites))
        fused = facts.get("fused_outer_candidates")
        if isinstance(fused, list):
            fused_count = max(fused_count, len(fused))
    total_matched = int(total if total is not None else len(hits))
    exact_unique = False
    if needle and len(hits) == 1 and total_matched == 1 and not substring_only:
        rank = _name_rank(
            str(hits[0].get("name") or ""),
            str(hits[0].get("id") or ""),
            str(needle).lower().strip(),
            exact_kind=False,
            kind=str(hits[0].get("kind") or ""),
        )
        exact_unique = rank in (0, 1)
    if dim_coverage:
        completeness = "coverage_checked"
        answerable = True
    elif clipped:
        completeness = "first_hit" if len(hits) <= 1 else "page_clipped"
        answerable = False
    elif substring_only:
        completeness = "first_hit"
        answerable = False
    elif def_count > 1 or len(sibling_files) > 1 or total_matched > 1:
        completeness = "siblings_checked"
        answerable = True
    elif exact_unique:
        completeness = "first_hit"
        answerable = True
    else:
        completeness = "first_hit"
        answerable = False
    return {
        "sibling_files": sibling_files,
        # Two different quantities used to share this key. Falling back to the
        # match total reported `SetCrossCore` as "2 definition sites" when the
        # second match was `SetCrossCoreID`, and `IsEmptyOutput` as three when
        # two of them were a BRANCH and a call. Count definitions only.
        "definition_sites_count": def_count,
        "matched_entities_count": total_matched,
        "total_matched": total_matched,
        "fused_outer_candidates_count": fused_count,
        "mutex_policies": mutex,
        "kernel_phases": phases,
        "completeness": completeness,
        "answerable": answerable,
        **({"dim_coverage": dim_coverage} if dim_coverage else {}),
    }


def _downgrade_coverage_after_clip(payload: dict[str, Any]) -> None:
    cov = payload.get("coverage")
    if not isinstance(cov, dict):
        return
    total = cov.get("total_matched")
    shown = 0
    for key in ("locations", "rows", "calls", "buffers", "branches", "hits", "phases"):
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            shown = max(shown, len(rows))
    if not isinstance(total, int) or shown >= total:
        return
    # Universe coverage (dim_coverage / coverage_checked) is not a clipped
    # first page. Do not downgrade it to first_hit when template_blocks shrink.
    if cov.get("dim_coverage") or cov.get("completeness") == "coverage_checked":
        return
    cov = dict(cov)
    if shown <= 1:
        cov["completeness"] = "first_hit"
        cov["answerable"] = False
    else:
        cov["completeness"] = "siblings_checked"
        cov["answerable"] = True
    payload["coverage"] = cov


#: An include guard is named after its own header. The `#ifndef` / `#define`
#: pair gets indexed (as MACRO or as the BRANCH it opens) but carries no
#: semantics, so it must never outrank a real seed for a fuzzy needle.
_INCLUDE_GUARD_RE = re.compile(r"^_*[A-Z][A-Z0-9_]*_(?:H|HH|HPP|HXX|INC)_*$")


def is_include_guard(kind: str, name: str, data: Any = None) -> bool:
    ident = str(name or "").rsplit("::", 1)[-1]
    if not _INCLUDE_GUARD_RE.fullmatch(ident):
        return False
    if isinstance(data, dict):
        body = str(data.get("value") or data.get("body") or "").strip()
        if body:
            return False
    return True


_CAMEL_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_DISCOVERY_KIND_RANK = {
    EntityKind.TILING_KEY.value: 0,
    EntityKind.TILING_FIELD.value: 0,
    EntityKind.TYPE.value: 1,
    EntityKind.FUNCTION.value: 2,
    EntityKind.METHOD.value: 2,
    EntityKind.KERNEL.value: 2,
    EntityKind.BUFFER.value: 2,
    EntityKind.MACRO.value: 3,
    EntityKind.VARIABLE.value: 4,
    EntityKind.FIELD.value: 4,
    EntityKind.OPERATION.value: 5,
    EntityKind.BRANCH.value: 6,
    EntityKind.PREDICATE.value: 6,
    EntityKind.COMPILE_VAR.value: 7,
    EntityKind.CONTRACT.value: 7,
}


def _needle_core(pattern: str) -> str:
    return str(pattern or "").replace("*", "").replace("?", "").replace("%", "").strip().lower()


def _ident_tokens(name: str) -> list[str]:
    leaf = _last_ident(name)
    out: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", leaf):
        if not chunk:
            continue
        out.extend(t.lower() for t in (_CAMEL_TOKEN_RE.findall(chunk) or [chunk]) if t)
    return out


# Ident recovery is lexical, not embedding: the name table is small and the
# caller needs a real ident, not a nearest neighbor. Split camel / snake /
# digits, expand a few C++ abbreviations, then LIKE.
_RECOVERY_STOP = frozenset(
    {
        "get",
        "set",
        "the",
        "has",
        "is",
        "for",
        "and",
        "not",
        "num",
        "len",
        "max",
        "min",
        "all",
        "val",
        "tmp",
        "new",
        "old",
    }
)
_ABBREV_TO_FULL = {
    "buf": "buffer",
    "buff": "buffer",
    "ptr": "pointer",
    "cnt": "count",
    "idx": "index",
    "cfg": "config",
    "ctx": "context",
    "msg": "message",
    "sel": "selector",
}
_FULL_TO_ABBREV = {
    full: short
    for short, full in _ABBREV_TO_FULL.items()
    if len(short) >= 4
}


def _recovery_tokens(pattern: str) -> list[str]:
    """Tokens to retry after a name-discovery miss. Longest first, no dups."""
    core = str(pattern or "").replace("*", "").replace("?", "").replace("%", "").strip()
    if not core:
        return []
    core_l = core.lower()
    seen: set[str] = set()
    scored: list[tuple[int, str]] = []
    for token in _ident_tokens(core):
        if token.isdigit() or token in _RECOVERY_STOP:
            continue
        full = _ABBREV_TO_FULL.get(token, token)
        for cand in (full, token):
            if len(cand) < 4 or cand == core_l or cand in seen:
                continue
            seen.add(cand)
            scored.append((len(cand), cand))
        short = _FULL_TO_ABBREV.get(full, "")
        if short and short != core_l and short not in seen:
            seen.add(short)
            scored.append((len(short), short))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [token for _, token in scored[:6]]


_SALIENCE_KIND = {
    EntityKind.TILING_KEY.value: 8,
    EntityKind.TILING_FIELD.value: 8,
    EntityKind.CONTRACT.value: 6,
    EntityKind.PREDICATE.value: 6,
    EntityKind.TYPE.value: 3,
    EntityKind.BUFFER.value: 3,
    EntityKind.KERNEL.value: 3,
    EntityKind.FUNCTION.value: 2,
    EntityKind.METHOD.value: 1,
}
_KIND_MERGE_PREF = [
    EntityKind.TILING_KEY.value,
    EntityKind.CONTRACT.value,
    EntityKind.TILING_FIELD.value,
    EntityKind.TYPE.value,
    EntityKind.KERNEL.value,
    EntityKind.FUNCTION.value,
    EntityKind.BUFFER.value,
    EntityKind.METHOD.value,
    EntityKind.VARIABLE.value,
]
_MATCH_LABEL = {
    100: "exact",
    90: "canonical",
    75: "token",
    60: "glob",
    50: "substring",
    30: "abbrev",
}


def _is_glob_pattern(pattern: str) -> bool:
    return "*" in str(pattern or "") or "?" in str(pattern or "")


def _match_quality(pattern: str, name: str) -> int:
    """Deterministic match tier. A glob's stripped core is not an exact leaf."""
    needle = _needle_core(pattern)
    if not needle:
        return 50
    leaf = _last_ident(name).lower()
    tokens = _ident_tokens(name)
    glob = _is_glob_pattern(pattern)
    if glob:
        return 60 if needle in leaf or needle in name.lower() else 50
    if leaf == needle:
        return 100
    if needle in tokens:
        return 75
    if needle in leaf or needle in name.lower():
        return 50
    return 30


def _locality_score(file: str, architecture: str = "") -> int:
    text = str(file or "").replace("\\", "/").lower()
    if not text.strip():
        return -20
    blob = f"/{text.strip('/')}/"
    if "/common/" in blob:
        return -10
    if "/op_host/" in blob or "/op_kernel/" in blob:
        return 20
    if architecture and f"/{architecture.strip().lower()}/" in blob:
        return 10
    return 0


def _find_score(
    hit: dict[str, Any],
    pattern: str,
    *,
    architecture: str = "",
    total: int = 1,
    freq: dict[str, int] | None = None,
) -> float:
    name = str(hit.get("name") or "")
    kind = str(hit.get("kind") or "")
    file = str(hit.get("file") or "")
    leaf = _last_ident(name).lower()
    quality = _match_quality(pattern, name)
    loc = _locality_score(file, architecture)
    sal = _SALIENCE_KIND.get(kind, 0)
    freq = freq or {}
    df = int(freq.get(leaf) or 1)
    idf = math.log((max(int(total), 1) + 1) / max(df, 1))
    generic = -15 if df > max(3, int(total) * 0.02) and len(leaf) <= 8 else 0
    located = 4 if file.strip() and int(hit.get("line_start") or hit.get("line") or 0) > 0 else 0
    return quality + loc + sal + idf * 4 + generic + located


def _name_discovery_key(
    hit: dict[str, Any],
    pattern: str,
    *,
    architecture: str = "",
    total: int = 1,
    freq: dict[str, int] | None = None,
) -> tuple[Any, ...]:
    """Higher-salience, operator-local, rare names first. Glob is not exact leaf."""
    return (
        -_find_score(
            hit, pattern, architecture=architecture, total=total, freq=freq
        ),
        _last_ident(str(hit.get("name") or "")).lower(),
        str(hit.get("kind") or ""),
    )


def _collapse_by_name(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per identity: same leaf at the same span, kinds merged."""
    filtered: list[dict[str, Any]] = []
    for hit in hits:
        name = str(hit.get("name") or "")
        kind = str(hit.get("kind") or "")
        if not name:
            continue
        if str(kind) == EntityKind.FILE.value:
            continue
        if any(ch in name for ch in " =<>!&|"):
            continue
        if is_include_guard(kind, name, hit.get("data")):
            continue
        filtered.append(hit)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, int]] = []
    for hit in filtered:
        name = str(hit.get("name") or "")
        leaf = _last_ident(name).lower()
        file = str(hit.get("file") or "").replace("\\", "/")
        line = int(hit.get("line_start") or hit.get("line") or 0)
        key = (leaf, file, line) if file and line > 0 else (leaf, "", 0)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(hit)
    located_leaves = {key[0] for key in groups if key[1]}
    out: list[dict[str, Any]] = []
    for key in order:
        leaf, file, _line = key
        if not file and leaf in located_leaves:
            continue
        members = groups[key]
        members.sort(
            key=lambda hit: (
                _KIND_MERGE_PREF.index(str(hit.get("kind") or ""))
                if str(hit.get("kind") or "") in _KIND_MERGE_PREF
                else 50,
                str(hit.get("kind") or ""),
            )
        )
        primary = dict(members[0])
        kinds: list[str] = []
        for member in members:
            kind = str(member.get("kind") or "")
            if kind and kind not in kinds:
                kinds.append(kind)
        if kinds:
            primary["kinds"] = kinds
        out.append(primary)
    compile_vars = {
        str(hit.get("name") or "").lower()
        for hit in out
        if str(hit.get("kind") or "") == EntityKind.COMPILE_VAR.value
    }
    if compile_vars:
        out = [
            hit
            for hit in out
            if not (
                str(hit.get("kind") or "") == EntityKind.CONTRACT.value
                and str(hit.get("name") or "").lower() in compile_vars
            )
        ]
    return out


def _name_pattern_to_like(pattern: str) -> str:
    """Glob (* ?) when the caller wrote one, otherwise substring.

    `_` stays a LIKE single-char wildcard, which still matches the literal `_`
    that idents are full of, so patterns need no escaping.
    """
    text = str(pattern or "").strip()
    if not text:
        return "%"
    if "*" in text or "?" in text:
        return text.replace("*", "%").replace("?", "_")
    return f"%{text}%"


_NEIGHBOUR_LIST_KEYS = (
    "used_at",
    "references",
    "neighbors",
    "readers",
    "writers",
    "locations",
    "hits",
    # `calls` is a list of places to look next, same as the rest of these. It
    # was not shed with them, so on a big function it outweighed the body and
    # the body was cut instead.
    "calls",
)

# Cross-references on the primary card: who calls this, what it uses. Context
# for the body, and recoverable by resolving any of the names they cite, which
# the body is not.
_CARD_CONTEXT_KEYS = ("facets", "extras")


#: Entities that are declared inside a class or struct, so a bare name can name
#: several of them. A TYPE or a TILING_KEY is named once and owns itself.
_SCOPED_MEMBER_KINDS = frozenset(
    {"FIELD", "BUFFER", "QUEUE", "REGISTER", "EVENT", "PIPE", "METHOD"}
)


def _inexact_name_note(needle: str, ranked: list[dict[str, Any]]) -> str:
    """Say when nothing carries the asked-for name.

    `MutexBuffersPolicy` names no entity, and the substring fallback answered
    with `MutexBuffersPolicySingleBuffer::buffer_` under the asked-for heading.
    A near miss presented as a hit is worse than a miss.
    """
    want = str(needle or "").strip().replace(".", "::")
    if not want or not ranked:
        return ""
    leaf = _last_ident(want).lower()
    for hit in ranked:
        name = str(hit.get("name") or "").replace(".", "::")
        if name.lower() == want.lower() or _last_ident(name).lower() == leaf:
            return ""
    best = str(ranked[0].get("name") or "")
    if not best:
        return ""
    return (
        f"Nothing is named `{needle}`. The closest name in the graph is "
        f"`{best}`, shown below; search pattern={needle} for the text instead."
    )


def _shrink_primary_context(payload: dict[str, Any], *, max_chars: int) -> None:
    """Drop cross-reference blocks off the primary card before its body."""
    cards = payload.get("cards")
    if not isinstance(cards, list) or not cards or not isinstance(cards[0], dict):
        return
    for key in _CARD_CONTEXT_KEYS:
        block = cards[0].get(key)
        if not isinstance(block, dict) or not block:
            continue
        for name in sorted(
            block,
            key=lambda k: len(json.dumps(block[k], ensure_ascii=False, default=str)),
            reverse=True,
        ):
            block.pop(name, None)
            payload["context_trimmed"] = True
            _note_trim(payload, key)
            if _payload_size(payload) <= max_chars:
                return


def _shrink_neighbour_lists(payload: dict[str, Any], *, max_chars: int) -> None:
    """Trim reference-style lists before the body they point away from."""
    for key in _NEIGHBOUR_LIST_KEYS:
        block = payload.get(key)
        # `calls` arrives as {calls, called_by, possible_callers} and is protected
        # so the key survives; the guard also spared its contents, so on a large
        # function it held three quarters of the budget and the body paid. This
        # shortens the lists inside and leaves the key in place.
        if isinstance(block, dict):
            if _shrink_nested_lists(payload, block, max_chars=max_chars):
                return
            continue
        if key in _PROTECTED_PAYLOAD_KEYS:
            continue
        if not isinstance(block, list) or len(block) <= MIN_LIST_KEEP:
            continue
        payload[key] = block[:MIN_LIST_KEEP]
        _note_trim(payload, key)
        _downgrade_coverage_after_clip(payload)
        if _payload_size(payload) <= max_chars:
            return


#: Order in which a grouped neighbour block gives up its lists. The callee list
#: goes first because the body above it names the same calls; who calls this is
#: not written anywhere in the card.
_NESTED_SHED_ORDER = ("calls", "possible_callers", "called_by")


def _shrink_nested_lists(
    payload: dict[str, Any], block: dict[str, Any], *, max_chars: int
) -> bool:
    """Trim the lists inside a grouped neighbour block, cheapest loss first."""
    known = [k for k in _NESTED_SHED_ORDER if isinstance(block.get(k), list)]
    rest = sorted(
        (k for k, v in block.items() if isinstance(v, list) and k not in known),
        key=lambda k: len(json.dumps(block[k], ensure_ascii=False, default=str)),
        reverse=True,
    )
    for name in [*known, *rest]:
        rows = block[name]
        if len(rows) <= MIN_LIST_KEEP:
            continue
        block[name] = rows[:MIN_LIST_KEEP]
        _note_trim(payload, name)
        _downgrade_coverage_after_clip(payload)
        if _payload_size(payload) <= max_chars:
            return True
    return False


#: Reader-facing names for the lists a card can shed. "some lists were trimmed"
#: costs as much as saying nothing: one agent re-derived a caller list that had
#: come back complete, because the note did not say the callee list was the one.
_TRIM_LABELS = {
    "calls": "callee list",
    "called_by": "caller list",
    "possible_callers": "possible-caller list",
    "references": "reference list",
    "used_at": "use list",
    "readers": "reader list",
    "writers": "writer list",
    "neighbors": "neighbour list",
    "facets": "cross-reference blocks",
    "extras": "extra context",
}


def _note_trim(payload: dict[str, Any], key: str) -> None:
    label = _TRIM_LABELS.get(str(key), str(key))
    names = payload.setdefault("trimmed_lists", [])
    if isinstance(names, list) and label not in names:
        names.append(label)


def _fit_payload(payload: dict[str, Any], *, max_chars: int = MAX_PAYLOAD_CHARS) -> dict[str, Any]:
    if _payload_size(payload) <= max_chars:
        return payload
    out = dict(payload)
    out["truncated"] = True
    _clip_relationships(out)
    if _payload_size(out) <= max_chars:
        return out
    if "edges" not in _PROTECTED_PAYLOAD_KEYS:
        out.pop("edges", None)
    if _payload_size(out) <= max_chars:
        return out
    for key in ("readers", "writers", "neighbors"):
        rows = out.get(key)
        if not isinstance(rows, list) or len(rows) <= PRIMARY_CANDIDATES:
            continue
        out[key] = rows[:PRIMARY_CANDIDATES]
        if _payload_size(out) <= max_chars:
            _downgrade_coverage_after_clip(out)
            return out
    # Neighbour lists are context; the definition body is the answer. Shedding
    # context first is what keeps a large function readable instead of trading
    # its body for a full reference list.
    for max_lines in (12, 6, 3):
        _clip_snippets(out, max_lines=max_lines, keep_primary=True)
        if _payload_size(out) <= max_chars:
            return out
    _shrink_neighbour_lists(out, max_chars=max_chars)
    if _payload_size(out) <= max_chars:
        return out
    _shrink_primary_context(out, max_chars=max_chars)
    if _payload_size(out) <= max_chars:
        return out
    # Starting the ladder at 120 threw away half a body that fit. Step down from
    # the hard cap so the first rung that fits is the largest one that does.
    for max_lines in (240, 180, 120, 60, 24):
        _clip_snippets(out, max_lines=max_lines)
        if _payload_size(out) <= max_chars:
            return out
    for key in (
        "rows",
        "branches",
        "calls",
        "buffers",
        "hits",
        "locations",
        "keys",
        "templates",
        "macros_compile_vars",
        "template_blocks",
        "gaps",
        "tiling_data",
        "fields",
        "neighbors",
        "writers",
        "readers",
    ):
        if key in _PROTECTED_PAYLOAD_KEYS:
            continue
        rows = out.get(key)
        if not isinstance(rows, list) or len(rows) <= MIN_LIST_KEEP:
            continue
        while len(rows) > MIN_LIST_KEEP:
            rows.pop()
            probe = dict(out)
            probe[key] = rows
            if _payload_size(probe) <= max_chars:
                out[key] = rows
                _note_trim(out, key)
                _downgrade_coverage_after_clip(out)
                return out
    _downgrade_coverage_after_clip(out)
    return out


class UoSqlQuery:
    """Read-only query facade over ``*.uo`` SQLite indexes."""

    backend = "codemap"

    def __init__(self, product: str | Path):
        self.product = Path(product).expanduser().resolve()
        if not self.product.is_file() or self.product.suffix != ".uo":
            raise FileNotFoundError(self.product)
        self.database = self.product
        self._architecture = _architecture_from_name(self.product)
        self._op_root = _op_root_from_product(self.product)
        self._engine = None
        self._accel: bool | None = None
        self._launch_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._edges_cache: dict[str, list[dict[str, Any]]] | None = None
        self._named_fields_cache: dict[str, list[dict[str, Any]]] | None = None
        self._idf_cache: tuple[int, dict[str, int]] | None = None
        self._compiled_support_cache: dict[str, dict[str, Any]] = {}
        self._state_index_cache: dict[str, list[dict[str, Any]]] | None = None
        self._state_files_cache: dict[str, list[str]] = {}
        self._guard_rel_cache: dict[str, str] | None = None
        self._field_ids_cache: dict[str, dict[str, list[str]]] = {}
        self._entity_row_cache: dict[str, sqlite3.Row | None] = {}
        self._clip_source_cache: dict[tuple[str, int, int, int], str] = {}
        self._field_impact_cache: dict[str, dict[str, Any]] = {}
        self._call_site_count_cache: dict[str, int] = {}
        self._branch_index_cache: dict[tuple[str, int, int], list[tuple[int, int, str]]] = {}
        self._calls_for_cache: dict[str, dict[str, Any] | None] = {}
        self._branch_guard_cache: dict[tuple[str, int], list[str]] = {}
        self._type_span_cache: dict[str, list[tuple[int, int, str]]] = {}
        self._def_span_cache: dict[str, list[tuple[int, int]]] = {}
        self._enclosing_cache: dict[tuple[str, int], dict[str, Any] | None] = {}
        self._grouped_edges_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._operation_index_cache: dict[str, list[tuple[int, str, str]]] | None = None
        self._event_site_index_cache: (
            dict[str, list[tuple[int, str, str, str]]] | None
        ) = None
        self._index_files_cache: dict[tuple[int, str], list[str]] = {}
        self._alloc_index_cache: dict[str, list[dict[str, Any]]] | None = None
        self._reach_cache: dict[frozenset[str], dict[str, list[dict[str, Any]]]] = {}
        self._reach_lock = threading.Lock()
        self._virtual_family_cache: dict[str, dict[str, Any] | None] = {}
        self._legal_cross_cache: dict[tuple[str, str], dict[str, dict[str, int]]] = {}

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        from ascendc_codemap_mcp.engine.store.reader import shared_uo

        # Thread-local connection owned by reader.close_uo_connections / QueryCache.drop.
        yield shared_uo(self.product)

    def _ident_frequencies(self) -> tuple[int, dict[str, int]]:
        if self._idf_cache is not None:
            return self._idf_cache
        freq: dict[str, int] = {}
        total = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM entity WHERE IFNULL(name, '') != ''"
            ).fetchall()
        for (name,) in rows:
            leaf = _last_ident(str(name or "")).lower()
            if not leaf:
                continue
            freq[leaf] = freq.get(leaf, 0) + 1
            total += 1
        self._idf_cache = (max(total, 1), freq)
        return self._idf_cache

    def _accel_ready(self, conn: sqlite3.Connection) -> bool:
        """Whether this product carries the leaf-name inverted index."""
        if self._accel is None:
            from ascendc_codemap_mcp.engine.store.accel import has_accel

            self._accel = has_accel(conn)
        return bool(self._accel)

    def close(self) -> None:
        from ascendc_codemap_mcp.engine.query.legal_key_cache import clear_legal_key_cache
        from ascendc_codemap_mcp.engine.store.reader import close_uo_connections

        close_uo_connections(self.product)
        clear_legal_key_cache(self.product)
        reset_source_line_cache()
        key = str(self.product)
        with _TEMPLATE_BLOCKS_LOCK:
            _TEMPLATE_BLOCKS_CACHE.pop(key, None)
        self._engine = None
        self._accel = None
        self._launch_cache = {}
        self._idf_cache = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def architecture(self) -> str:
        if self._architecture:
            return self._architecture
        from ascendc_codemap_mcp.engine.store.reader import read_meta

        self._architecture = str(read_meta(self.product).get("architecture") or "")
        return self._architecture

    def _foreign_file(self, file: str) -> bool:
        text = str(file or "").replace("\\", "/")
        arch = self.architecture
        if not text or not arch:
            return False
        from ascendc_codemap_mcp.engine.source_layout import is_other_arch_path

        return is_other_arch_path(Path(text), arch)

    def _hit(
        self,
        row: sqlite3.Row,
        *,
        why: str = "",
        distance: int | None = None,
        require_span_for_branch: bool = False,
        with_snippet: bool = True,
        with_rels: bool = False,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any] | None:
        data = _parse_data(_row_get(row, "data", "{}"))
        kind = str(_row_get(row, "kind") or "")
        name = str(_row_get(row, "name") or "")
        if not _keep_branch(kind, name, data):
            return None
        file = _norm_file(str(_row_get(row, "file") or ""))
        if self._foreign_file(file):
            return None
        mapping = {
            "id": _row_get(row, "id") or "",
            "kind": kind,
            "name": name,
            "status": _row_get(row, "status") or "",
            "file": file,
            "line_start": int(_row_get(row, "line_start") or 0),
            "line_end": int(_row_get(row, "line_end") or 0),
            "attrs": data,
        }
        hit = project_entity(
            mapping,
            why=why,
            distance=distance,
            require_span_for_branch=require_span_for_branch,
        )
        if hit is None:
            return None
        if with_snippet:
            orig = str(_row_get(row, "file") or "")
            snippet = str(_row_get(row, "snippet") or "")
            line = int(hit.get("line_start") or 0)
            thin = (not snippet.strip()) or snippet.count("\n") < 2
            numbered = bool(snippet) and snippet[:1].isdigit() and (
                ":" in snippet[:8] or "|" in snippet[:8]
            )
            if line > 0 and (thin or not _snippet_covers_line(snippet, line)):
                line_end = int(hit.get("line_end") or 0)
                window, truncated, omitted = ("", False, [])
                if conn is not None:
                    window, truncated, omitted = _snapshot_window(
                        conn, orig or file, line, kind=kind, line_end=line_end
                    )
                if window:
                    snippet = window
                    numbered = True
                    if truncated:
                        hit["truncated"] = True
                    if omitted:
                        hit["omitted"] = omitted
            if str(kind or "").upper() == EntityKind.MACRO.value and line > 0 and conn is not None:
                rows = _source_line_rows(conn, orig or file, line, line + MACRO_CONT_MAX)
                end = line
                for ln, txt in rows:
                    end = ln
                    if not str(txt).rstrip().endswith("\\"):
                        break
                hit["line_end"] = max(int(hit.get("line_end") or 0), end)
            if str(kind or "").upper() in _STMT_EXPAND_KINDS and line > 0:
                window = ""
                if conn is not None:
                    window = _source_line_window(conn, orig or file, line)
                if window and (
                    str(snippet or "").count("\n") < 2 or len(window) > len(snippet)
                ):
                    snippet = window
                    numbered = True
            hit["snippet"] = snippet if numbered else _cap_snippet(snippet, line)
            if int(hit.get("line_end") or 0) > 0 and not _snippet_covers_line(
                str(hit.get("snippet") or ""), int(hit.get("line_end") or 0)
            ):
                hit["truncated"] = True
        if with_rels and conn is not None:
            rels = self._relationships(
                conn, str(hit.get("id") or ""), entity_kind=kind
            )
            if rels:
                hit["relationships"] = rels
        return hit

    def _relationships(
        self,
        conn: sqlite3.Connection,
        entity_id: str,
        *,
        limit: int = MAX_REL_HOPS,
        entity_kind: str = "",
    ) -> list[dict[str, Any]]:
        if not entity_id:
            return []
        placeholders = ",".join("?" for _ in NEIGHBOR_REL_KINDS)
        fetch = max(int(limit) * 4, 16)
        rows = conn.execute(
            f"""
            SELECT r.kind AS rel_kind, r.src AS src, r.dst AS dst, r.data AS rel_data,
                   e.id AS other_id, e.kind AS other_kind, e.name AS other_name,
                   e.file AS other_file, e.line_start AS other_line
            FROM relation r
            JOIN entity e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
            WHERE (r.src = ? OR r.dst = ?) AND r.kind IN ({placeholders})
            ORDER BY r.kind, e.kind, e.name
            LIMIT ?
            """,
            (entity_id, entity_id, entity_id, *NEIGHBOR_REL_KINDS, fetch),
        ).fetchall()
        skip_tpl = str(entity_kind or "").upper() == EntityKind.TILING_KEY.value
        out: list[dict[str, Any]] = []
        for row in rows:
            if _is_advisory_data(row["rel_data"]):
                continue
            other_kind = str(row["other_kind"] or "")
            other_name = str(row["other_name"] or "")
            rel_kind = str(row["rel_kind"] or "")
            if skip_tpl and rel_kind == "BINDS" and (
                other_kind == EntityKind.TEMPLATE.value or other_name.startswith("ARGS_SEL")
            ):
                continue
            if _skip_composition_neighbor(entity_kind, rel_kind, other_kind):
                continue
            out.append(
                {
                    "kind": rel_kind,
                    "src": str(row["src"] or ""),
                    "dst": str(row["dst"] or ""),
                    "other_id": str(row["other_id"] or ""),
                    "other_kind": other_kind,
                    "other_name": other_name,
                    "file": _norm_file(str(row["other_file"] or "")),
                    "line_start": int(row["other_line"] or 0),
                }
            )
            if len(out) >= int(limit):
                break
        return out

    def _select_entities(
        self,
        conn: sqlite3.Connection,
        *,
        kinds: Iterable[str] = (),
        extra_where: str = "",
        params: Iterable[Any] = (),
        limit: int = 50,
        order: str = "e.kind, e.name, e.id",
    ) -> list[sqlite3.Row]:
        allowed = [k for k in kinds if k]
        where: list[str] = []
        sql_params: list[Any] = []
        if allowed:
            placeholders = ",".join("?" for _ in allowed)
            where.append(f"e.kind IN ({placeholders})")
            sql_params.extend(allowed)
        if extra_where:
            where.append(f"({extra_where})")
            sql_params.extend(list(params))
        clause = " AND ".join(where) if where else "1=1"
        sql_params.append(max(0, int(limit)))
        return conn.execute(
            f"""
            SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                   IFNULL(s.snippet, '') AS snippet
            FROM entity e
            LEFT JOIN source_span s ON s.entity_id = e.id
            WHERE {clause}
            ORDER BY {order}
            LIMIT ?
            """,
            tuple(sql_params),
        ).fetchall()

    def _count_entities(
        self,
        conn: sqlite3.Connection,
        *,
        kinds: Iterable[str] = (),
        extra_where: str = "",
        params: Iterable[Any] = (),
    ) -> int:
        allowed = [k for k in kinds if k]
        where: list[str] = []
        sql_params: list[Any] = []
        if allowed:
            placeholders = ",".join("?" for _ in allowed)
            where.append(f"e.kind IN ({placeholders})")
            sql_params.extend(allowed)
        if extra_where:
            where.append(f"({extra_where})")
            sql_params.extend(list(params))
        clause = " AND ".join(where) if where else "1=1"
        row = conn.execute(
            f"SELECT COUNT(*) FROM entity e WHERE {clause}",
            tuple(sql_params),
        ).fetchone()
        return int(row[0] or 0) if row else 0

    def _hits_from_rows(
        self,
        conn: sqlite3.Connection,
        rows: Iterable[sqlite3.Row],
        *,
        why: str = "",
        with_snippet: bool = True,
        with_rels: bool = False,
        require_span_for_branch: bool = False,
    ) -> list[dict[str, Any]]:
        preferred: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        order: list[tuple[str, str, str, int]] = []
        for row in rows:
            hit = self._hit(
                row,
                why=why,
                require_span_for_branch=require_span_for_branch,
                with_snippet=with_snippet,
                with_rels=with_rels,
                conn=conn,
            )
            if hit is None:
                continue
            file = _norm_file(str(hit.get("file") or ""))
            line = int(hit.get("line_start") or 0)
            key = (
                str(hit.get("kind") or ""),
                str(hit.get("name") or "").lower(),
                file,
                line,
            )
            existing = preferred.get(key)
            if existing is None:
                preferred[key] = hit
                order.append(key)
                continue
            if _prefer_src_id(str(hit.get("id") or "")) < _prefer_src_id(str(existing.get("id") or "")):
                preferred[key] = hit
        return _drop_redundant_type_hashes([preferred[key] for key in order])

    def _entity_row(self, conn: sqlite3.Connection, name_or_id: str) -> sqlite3.Row | None:
        key = str(name_or_id or "")
        if not key:
            return None
        # Graph walks revisit the same node from several edges; a committed row
        # cannot change under us.
        if key in self._entity_row_cache:
            return self._entity_row_cache[key]
        row = self._entity_row_uncached(conn, key)
        if len(self._entity_row_cache) < 20000:
            self._entity_row_cache[key] = row
        return row

    def _entity_row_uncached(
        self, conn: sqlite3.Connection, key: str
    ) -> sqlite3.Row | None:
        row = conn.execute(
            """
            SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                   IFNULL(s.snippet, '') AS snippet
            FROM entity e
            LEFT JOIN source_span s ON s.entity_id = e.id
            WHERE e.id = ?
            LIMIT 1
            """,
            (key,),
        ).fetchone()
        if row is not None:
            return row
        return conn.execute(
            """
            SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                   IFNULL(s.snippet, '') AS snippet
            FROM entity e
            LEFT JOIN source_span s ON s.entity_id = e.id
            WHERE e.name = ?
            ORDER BY e.kind, e.id
            LIMIT 1
            """,
            (key,),
        ).fetchone()

    def search(
        self, pattern: str, *, kinds: Iterable[str] = (), limit: int = 50
    ) -> list[dict[str, Any]]:
        needles = search_needles(pattern)
        if len(needles) <= 1:
            return self._search_one(
                needles[0] if needles else str(pattern or ""),
                kinds=kinds,
                limit=limit,
            )
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for needle in needles:
            for hit in self._search_one(needle, kinds=kinds, limit=limit):
                eid = str(hit.get("id") or "")
                key = eid or f"{hit.get('file')}:{hit.get('line_start')}:{hit.get('name')}"
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
                if len(merged) >= max(0, int(limit)):
                    return merged
        return merged[: max(0, int(limit))]

    def _search_one(
        self, pattern: str, *, kinds: Iterable[str] = (), limit: int = 50, ranked_only: bool = False
    ) -> list[dict[str, Any]]:
        needle = str(pattern or "").lower().strip()
        allowed = _kind_names(kinds)
        fetch = max(int(limit) * 8, 32)
        with self._connect() as conn:
            kind_filter = ""
            kind_params: list[Any] = []
            if allowed:
                placeholders = ",".join("?" for _ in allowed)
                kind_filter = f" AND e.kind IN ({placeholders})"
                kind_params.extend(sorted(allowed))
            if needle:
                prefix = f"{needle}%"
                extra = """
                    e.name COLLATE NOCASE = ?
                    OR e.name COLLATE NOCASE LIKE ?
                    OR e.name COLLATE NOCASE LIKE '%::' || ?
                    OR e.name COLLATE NOCASE LIKE '%.' || ?
                    OR lower(IFNULL(e.id, '')) = ?
                    OR lower(IFNULL(e.id, '')) LIKE '%::' || ?
                    OR lower(IFNULL(e.id, '')) LIKE '%.' || ?
                    OR (
                      e.kind NOT IN ('TYPE')
                      AND e.name COLLATE NOCASE LIKE ?
                    )
                """
                sql_params: list[Any] = [
                    needle,
                    prefix,
                    needle,
                    needle,
                    needle,
                    needle,
                    needle,
                    f"%{needle}%",
                    *kind_params,
                ]
                order_sql = """
                ORDER BY CASE
                  WHEN e.name COLLATE NOCASE = ? THEN 0
                  WHEN e.name COLLATE NOCASE LIKE ? THEN 1
                  WHEN e.name COLLATE NOCASE LIKE '%::' || ? THEN 1
                  WHEN e.name COLLATE NOCASE LIKE ? THEN 2
                  ELSE 3
                END, e.id
                """
                sql_params.extend([needle, f"{needle}::%", needle, prefix])
            else:
                extra = "1=1"
                sql_params = [*kind_params]
                order_sql = "ORDER BY e.id"
            sql_params.append(fetch)
            rows = conn.execute(
                f"""
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE ({extra}) {kind_filter}
                {order_sql}
                LIMIT ?
                """,
                tuple(sql_params),
            ).fetchall()
            matched: list[sqlite3.Row] = []
            for row in rows:
                kind = str(row["kind"] or "")
                rank = _name_rank(
                    str(row["name"] or ""),
                    str(row["id"] or ""),
                    needle,
                    exact_kind=kind in _EXACT_KINDS,
                    kind=kind,
                )
                if rank is None:
                    continue
                matched.append(row)
            hits = self._hits_from_rows(
                conn,
                matched,
                why="search",
                with_snippet=True,
                with_rels=True,
            )
            hits.sort(
                key=lambda hit: _agent_sort_key(
                    hit, needle, architecture=self._architecture
                )
            )
            if ranked_only:
                return hits
            if allowed == {EntityKind.BRANCH.value} or (
                hits and all(str(hit.get("kind") or "") == EntityKind.BRANCH.value for hit in hits)
            ):
                hits, _ = _diversify_by_function(hits, limit=int(limit))
                return hits[: max(0, int(limit))]
            if hits and all(
                str(hit.get("kind") or "") in {EntityKind.FUNCTION.value, EntityKind.METHOD.value}
                for hit in hits
            ):
                hits, _ = _diversify_by_file(hits, limit=int(limit))
                return hits[: max(0, int(limit))]
            page, _meta = _page_by_exactness(hits, needle, limit=int(limit))
            return page

    def aggregate_search(
        self, pattern: str, *, kinds: Iterable[str] = (), limit: int = 8
    ) -> dict[str, Any]:
        needle = str(pattern or "").strip()
        ranked = self._search_one(needle, kinds=kinds, limit=int(limit), ranked_only=True)
        page, meta = _page_by_exactness(ranked, needle, limit=int(limit))
        fetch_cap = max(int(limit) * 8, 32)
        clipped = bool(meta["clipped"] or int(meta.get("all_matched") or 0) >= fetch_cap)
        coverage = _hits_coverage(
            page,
            total=int(meta["total"]),
            clipped=clipped,
            needle=needle,
            substring_only=bool(meta["substring_only"]),
        )
        payload = {
            "ok": True,
            "mode": "search",
            "pattern": needle,
            "kinds": [k for k in kinds if k],
            "count": len(page),
            "coverage": coverage,
            "rows": page,
            "files": _group_by_file(page),
        }
        attach_query_hints(payload, needle, count=len(page), kinds=kinds, mode="search")
        return _fit_payload(payload)

    def neighbors(
        self, entity_id: str, *, depth: int = 1, limit: int = 100
    ) -> list[dict[str, Any]]:
        max_depth = max(1, min(int(depth), 4))
        with self._connect() as conn:
            start = self._entity_row(conn, entity_id)
            if start is None:
                return []
            start_id = str(start["id"])
            seen = {start_id}
            queue: deque[tuple[str, int]] = deque([(start_id, 0)])
            ordered: list[tuple[str, int]] = [(start_id, 0)]
            while queue and len(ordered) < int(limit):
                cur, dist = queue.popleft()
                if dist >= max_depth:
                    continue
                for row in conn.execute(
                    """
                    SELECT CASE WHEN src = ? THEN dst ELSE src END AS other, data
                    FROM relation
                    WHERE src = ? OR dst = ?
                    """,
                    (cur, cur, cur),
                ):
                    if _is_advisory_data(row["data"]):
                        continue
                    other = str(row["other"] or "")
                    if not other or other in seen:
                        continue
                    seen.add(other)
                    queue.append((other, dist + 1))
                    ordered.append((other, dist + 1))
                    if len(ordered) >= int(limit):
                        break
            out: list[dict[str, Any]] = []
            for eid, dist in ordered[: int(limit)]:
                row = self._entity_row(conn, eid)
                if row is None:
                    continue
                hit = self._hit(row, distance=dist, with_snippet=False, with_rels=False)
                if hit is not None:
                    out.append(hit)
        return out

    def edges_of_many(
        self,
        ids: Iterable[str],
        *,
        kind: str = "",
        limit: int = 100,
    ) -> dict[str, list[dict[str, Any]]]:
        unique = []
        seen: set[str] = set()
        for raw in ids:
            eid = str(raw or "")
            if eid and eid not in seen:
                seen.add(eid)
                unique.append(eid)
        out: dict[str, list[dict[str, Any]]] = {eid: [] for eid in unique}
        if not unique:
            return out
        wanted = str(kind or "").upper()
        cap = max(1, int(limit or 100))
        with self._connect() as conn:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                sql = f"""
                    SELECT id, kind, src, dst, status FROM relation
                    WHERE (src IN ({placeholders}) OR dst IN ({placeholders}))
                """
                params: list[Any] = [*chunk, *chunk]
                if wanted:
                    sql += " AND kind = ?"
                    params.append(wanted)
                sql += " ORDER BY kind, src, dst"
                for row in conn.execute(sql, params):
                    rec = {
                        "id": str(row["id"] or ""),
                        "kind": str(row["kind"] or ""),
                        "src": str(row["src"] or ""),
                        "dst": str(row["dst"] or ""),
                        "status": str(row["status"] or ""),
                    }
                    for key in (rec["src"], rec["dst"]):
                        bucket = out.get(key)
                        if bucket is None or len(bucket) >= cap:
                            continue
                        bucket.append(rec)
        return out

    def edges_of(
        self, entity_id: str, *, kind: str = "", limit: int = 100
    ) -> list[dict[str, Any]]:
        eid = str(entity_id or "")
        cached = self._edges_cache
        if cached is not None and eid in cached and not kind:
            return list(cached.get(eid) or [])[: max(1, int(limit or 100))]
        with self._connect() as conn:
            start = self._entity_row(conn, entity_id)
            if start is None:
                return []
            eid = str(start["id"])
            wanted = str(kind or "").upper()
            if wanted:
                rows = conn.execute(
                    """
                    SELECT id, kind, src, dst, status FROM relation
                    WHERE (src = ? OR dst = ?) AND kind = ?
                    ORDER BY kind, src, dst LIMIT ?
                    """,
                    (eid, eid, wanted, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, kind, src, dst, status FROM relation
                    WHERE src = ? OR dst = ?
                    ORDER BY kind, src, dst LIMIT ?
                    """,
                    (eid, eid, int(limit)),
                ).fetchall()
        return [
            {
                "id": str(row["id"] or ""),
                "kind": str(row["kind"] or ""),
                "src": str(row["src"] or ""),
                "dst": str(row["dst"] or ""),
                "status": str(row["status"] or ""),
            }
            for row in rows
        ]

    def constraints_for(self, entity_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            start = self._entity_row(conn, entity_id)
            if start is None:
                return []
            eid = str(start["id"])
            ids = {eid}
            for row in conn.execute(
                """
                SELECT CASE WHEN src = ? THEN dst ELSE src END AS other, kind
                FROM relation
                WHERE src = ? OR dst = ?
                """,
                (eid, eid, eid),
            ):
                if str(row["kind"] or "") in {
                    RelationKind.GUARDED_BY.value,
                    RelationKind.CONTROLS.value,
                    RelationKind.DERIVES.value,
                    RelationKind.BINDS.value,
                }:
                    ids.add(str(row["other"] or ""))
            hits: list[dict[str, Any]] = []
            for other in ids:
                row = self._entity_row(conn, other)
                if row is None or str(row["kind"] or "") != EntityKind.PREDICATE.value:
                    continue
                hit = self._hit(row, with_snippet=False)
                if hit is not None:
                    hits.append(hit)
        return hits

    def _reachable_kinds(
        self, start_id: str, kinds: set[str], *, depth: int = 6, limit: int = 80
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            start = self._entity_row(conn, start_id)
            if start is None:
                return []
            sid = str(start["id"])
            seen = {sid}
            queue = deque([(sid, 0)])
            out: list[dict[str, Any]] = []
            while queue and len(out) < int(limit):
                cur, dist = queue.popleft()
                if dist >= int(depth):
                    continue
                for row in conn.execute(
                    """
                    SELECT CASE WHEN src = ? THEN dst ELSE src END AS other, data
                    FROM relation WHERE src = ? OR dst = ?
                    """,
                    (cur, cur, cur),
                ):
                    if _is_advisory_data(row["data"]):
                        continue
                    other = str(row["other"] or "")
                    if not other or other in seen:
                        continue
                    seen.add(other)
                    queue.append((other, dist + 1))
                    ent = self._entity_row(conn, other)
                    if ent is None:
                        continue
                    if str(ent["kind"] or "") in kinds:
                        hit = self._hit(ent, with_snippet=True, with_rels=False)
                        if hit is not None:
                            out.append(hit)
        return out[: int(limit)]

    def branches_for_key(self, key_id: str) -> list[dict[str, Any]]:
        return self._reachable_kinds(key_id, {EntityKind.BRANCH.value}, depth=4)

    def templates_for_key(self, key_id: str) -> list[dict[str, Any]]:
        return self._reachable_kinds(
            key_id,
            {
                EntityKind.TEMPLATE.value,
                EntityKind.TEMPLATE_ARG.value,
                EntityKind.TEMPLATE_INSTANCE.value,
            },
            depth=4,
        )

    def affected_shapes(self, entity_id: str) -> list[dict[str, Any]]:
        return self._reachable_kinds(
            entity_id,
            {EntityKind.INPUT.value, EntityKind.FIELD.value, EntityKind.TILING_FIELD.value},
            depth=4,
        )

    def controllability_of(self, branch_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            start = self._entity_row(conn, branch_id)
            if start is None:
                return []
            eid = str(start["id"])
            hits: list[dict[str, Any]] = []
            for row in conn.execute(
                """
                SELECT r.kind AS rel_kind,
                       e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM relation r
                JOIN entity e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE r.src = ? OR r.dst = ?
                """,
                (eid, eid, eid),
            ):
                rel = str(row["rel_kind"] or "")
                kind = str(row["kind"] or "")
                if rel not in {
                    RelationKind.CONTROLS.value,
                    RelationKind.GUARDED_BY.value,
                    RelationKind.DERIVES.value,
                } and kind not in {
                    EntityKind.PREDICATE.value,
                    EntityKind.TILING_KEY.value,
                    EntityKind.INPUT.value,
                }:
                    continue
                hit = self._hit(row, with_snippet=False)
                if hit is not None:
                    hits.append(hit)
        return hits

    def entities_in_files(self, files: Iterable[str]) -> list[dict[str, Any]]:
        normalized = sorted({_strip_dot_slash(str(p).replace("\\", "/")) for p in files if str(p).strip()})
        if not normalized:
            return []
        hits: list[dict[str, Any]] = []
        matched: set[str] = set()
        with self._connect() as conn:
            for chunk in _chunks(normalized):
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                           IFNULL(s.snippet, '') AS snippet
                    FROM entity e
                    LEFT JOIN source_span s ON s.entity_id = e.id
                    WHERE e.file IN ({placeholders})
                    ORDER BY e.kind, e.id
                    LIMIT ?
                    """,
                    (*chunk, min(200 * len(chunk), 4000)),
                ).fetchall()
                for row in rows:
                    matched.add(_strip_dot_slash(str(row["file"] or "").replace("\\", "/")))
                hits.extend(self._hits_from_rows(conn, rows, with_snippet=False))
            missing = [
                path
                for path in normalized
                if path not in matched
                and not any(got.endswith("/" + path) or got.endswith(path) for got in matched)
            ]
            for path in missing:
                suffix = "/" + path if not path.startswith("/") else path
                rows = conn.execute(
                    """
                    SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                           IFNULL(s.snippet, '') AS snippet
                    FROM entity e
                    LEFT JOIN source_span s ON s.entity_id = e.id
                    WHERE e.file = ?
                       OR e.file LIKE '%' || ?
                    ORDER BY e.kind, e.id
                    LIMIT 200
                    """,
                    (path, suffix),
                ).fetchall()
                hits.extend(self._hits_from_rows(conn, rows, with_snippet=False))
        hits.sort(key=lambda r: (str(r.get("kind")), str(r.get("id"))))
        return hits

    def impact_of(self, file: str, line_range: tuple[int, int]) -> dict[str, Any]:
        start, end = sorted((int(line_range[0]), int(line_range[1])))
        needle = _strip_dot_slash(str(file or "").replace("\\", "/"))
        useful = set(USEFUL_EDGE_KINDS)
        with self._connect() as conn:
            seed_rows = _seed_rows_for_file(conn, needle, start, end, 80)
            if not seed_rows:
                alt = _alternate_file_spelling(needle)
                if alt:
                    seed_rows = _seed_rows_for_file(conn, alt, start, end, 80)
            seeds = [row for row in seed_rows if self._file_matches(str(row["file"] or ""), needle)]
            seen: dict[str, int] = {str(row["id"]): 0 for row in seeds}
            queue: deque[tuple[str, int]] = deque((str(row["id"]), 0) for row in seeds)
            placeholders = ",".join("?" for _ in useful)
            useful_sorted = sorted(useful)
            while queue:
                cur, dist = queue.popleft()
                if dist >= 2:
                    continue
                for row in conn.execute(
                    f"""
                    SELECT dst, data FROM relation
                    WHERE src = ? AND kind IN ({placeholders})
                    """,
                    (cur, *useful_sorted),
                ):
                    if _is_advisory_data(row["data"]):
                        continue
                    other = str(row["dst"] or "")
                    if not other or other in seen:
                        continue
                    seen[other] = dist + 1
                    queue.append((other, dist + 1))
            hits: list[dict[str, Any]] = []
            for eid, dist in seen.items():
                row = self._entity_row(conn, eid)
                if row is None:
                    continue
                hit = self._hit(
                    row,
                    distance=dist,
                    why="seed" if dist == 0 else "slice_neighbor",
                    with_snippet=dist == 0,
                    with_rels=False,
                )
                if hit is not None:
                    hits.append(hit)
        hits.sort(key=lambda r: (int(r.get("distance") or 0), str(r.get("kind")), str(r.get("id"))))
        return {
            "ok": True,
            "seeds": [row for row in hits if int(row.get("distance") or 0) == 0],
            "hits": hits,
            "buckets": bucket_hits(hits),
            "count": len(hits),
        }

    @staticmethod
    def _file_matches(current: str, needle: str) -> bool:
        cur = _strip_dot_slash(str(current or "").replace("\\", "/"))
        want = _strip_dot_slash(str(needle or "").replace("\\", "/"))
        if not cur or not want:
            return False
        return cur == want or cur.endswith("/" + want) or want.endswith("/" + cur)

    def tiling_field(self, name_or_id: str) -> list[dict[str, Any]]:
        return self._named_fields(name_or_id, kinds=(EntityKind.TILING_FIELD.value,))

    def _named_fields(self, name_or_id: str, *, kinds: Iterable[str]) -> list[dict[str, Any]]:
        key = str(name_or_id or "").strip().lower()
        if not key:
            return []
        cached = self._named_fields_cache
        if cached is not None and key in cached:
            return list(cached.get(key) or [])
        kind_list = [k for k in kinds if k]
        placeholders = ",".join("?" for _ in kind_list)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE e.kind IN ({placeholders})
                  AND (
                    e.name COLLATE NOCASE = ?
                    OR e.name COLLATE NOCASE LIKE '%::' || ?
                    OR e.name COLLATE NOCASE LIKE '%.' || ?
                    OR lower(e.id) = ?
                    OR lower(e.id) LIKE '%::' || ?
                  )
                ORDER BY CASE e.kind
                    WHEN 'TILING_FIELD' THEN 0
                    WHEN 'FIELD' THEN 1
                    ELSE 2
                END, e.id
                LIMIT 40
                """,
                (*kind_list, key, key, key, key, key),
            ).fetchall()
            hits = self._hits_from_rows(conn, rows, why="field", with_snippet=True, with_rels=True)
            hits.sort(
                key=lambda hit: (
                    0
                    if str(hit.get("name") or "").lower() == key
                    or _last_ident(str(hit.get("name") or "")).lower() == key
                    else 1,
                    {
                        EntityKind.TILING_FIELD.value: 0,
                        EntityKind.FIELD.value: 1,
                    }.get(str(hit.get("kind") or ""), 2),
                    *_field_value_rank(hit),
                )
            )
            return hits

    def _fields_by_local_alias(
        self, ident: str, *, kinds: Iterable[str]
    ) -> list[dict[str, Any]]:
        """Resolve a host local name via facts already on tiling fields.

        Extra query round is preferred over a hardcoded alias table.
        """
        needle = str(ident or "").strip().lower()
        if not needle:
            return []
        kind_list = [k for k in kinds if k]
        placeholders = ",".join("?" for _ in kind_list)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE e.kind IN ({placeholders})
                  AND (
                    e.data LIKE '%local_aliases%'
                    OR e.data LIKE '%fused_outer_candidates%'
                  )
                LIMIT 80
                """,
                tuple(kind_list),
            ).fetchall()
            hits = self._hits_from_rows(conn, rows, why="field_alias", with_snippet=True, with_rels=True)
        matched: list[dict[str, Any]] = []
        for hit in hits:
            facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
            names = {n.lower() for n in _alias_ident_names(facts)}
            if needle in names:
                matched.append(hit)
        matched.sort(key=lambda hit: _alias_hit_rank(hit, needle))
        return matched

    def _named_fields_many(
        self, names: Iterable[str], *, kinds: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        keys: list[str] = []
        seen: set[str] = set()
        for raw in names:
            key = str(raw or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
        out: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
        if not keys:
            return out
        kind_list = [k for k in kinds if k]
        kind_ph = ",".join("?" for _ in kind_list)
        with self._connect() as conn:
            for chunk in _chunks(keys):
                name_ph = ",".join("?" for _ in chunk)
                likes = []
                like_params: list[str] = []
                for key in chunk:
                    likes.append("e.name COLLATE NOCASE LIKE '%::' || ?")
                    likes.append("e.name COLLATE NOCASE LIKE '%.' || ?")
                    like_params.extend([key, key])
                rows = conn.execute(
                    f"""
                    SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                           IFNULL(s.snippet, '') AS snippet
                    FROM entity e
                    LEFT JOIN source_span s ON s.entity_id = e.id
                    WHERE e.kind IN ({kind_ph})
                      AND (
                        lower(e.name) IN ({name_ph})
                        OR lower(e.id) IN ({name_ph})
                        OR {' OR '.join(likes)}
                      )
                    LIMIT 80
                    """,
                    (*kind_list, *chunk, *chunk, *like_params),
                ).fetchall()
                hits = self._hits_from_rows(conn, rows, why="field", with_snippet=True, with_rels=True)
                for hit in hits:
                    name = str(hit.get("name") or "").lower()
                    leaf = _last_ident(name).lower()
                    eid = str(hit.get("id") or "").lower()
                    for key in chunk:
                        if key in {name, leaf, eid} or name.endswith("::" + key) or name.endswith("." + key):
                            out[key].append(hit)
        return out

    def _prefetch_field_graph(self, names: Iterable[str]) -> None:
        unique = [str(n).strip() for n in names if str(n).strip()]
        if not unique:
            return
        field_kinds = (
            EntityKind.TILING_FIELD.value,
            EntityKind.FIELD.value,
            EntityKind.TILING_KEY.value,
        )
        by_key = self._named_fields_many(unique, kinds=field_kinds)
        self._named_fields_cache = by_key
        fids: list[str] = []
        for hits in by_key.values():
            for hit in hits[:1]:
                fid = str(hit.get("id") or "")
                if fid:
                    fids.append(fid)
        self._edges_cache = self.edges_of_many(fids, limit=300)

    def field_impact_many(self, names: Iterable[str]) -> dict[str, dict[str, Any]]:
        unique: list[str] = []
        seen: set[str] = set()
        for raw in names:
            name = str(raw or "").strip().strip('"').strip("'")
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                unique.append(name)
        if not unique:
            return {}
        self._prefetch_field_graph(unique)
        try:
            return {name: self.field_impact(name) for name in unique}
        finally:
            self._named_fields_cache = None
            self._edges_cache = None

    def field_impact(self, name_or_id: str) -> dict[str, Any]:
        # A field card asks for this from several facets, and a symbol with both
        # a FIELD and a TILING_KEY identity asks once per identity.
        key = str(name_or_id or "")
        cached = self._field_impact_cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
        payload = self._field_impact_uncached(key)
        if len(self._field_impact_cache) < 512:
            self._field_impact_cache[key] = copy.deepcopy(payload)
        return payload

    def _field_impact_uncached(self, name_or_id: str) -> dict[str, Any]:
        raw = str(name_or_id or "").strip().strip('"').strip("'")
        field_kinds = (
            EntityKind.TILING_FIELD.value,
            EntityKind.FIELD.value,
            EntityKind.TILING_KEY.value,
        )
        fields = self._named_fields(raw, kinds=field_kinds)
        alias_from = ""
        if not fields:
            fields = self._fields_by_local_alias(raw, kinds=field_kinds)
            if fields:
                alias_from = raw
        if not fields:
            payload = {
                "ok": False,
                "error": "tiling_field_not_found",
                "query": name_or_id,
            }
            attach_query_hints(payload, raw, count=0, mode="field")
            return payload
        primary = fields[0]
        fid = str(primary["id"])
        allowed = field_edge_kinds()
        edges = [
            project_relation(rel)
            for rel in self.edges_of(fid, limit=300)
            if str(rel.get("kind") or "") in allowed
        ]
        readers: list[dict[str, Any]] = []
        writers: list[dict[str, Any]] = []
        seen_readers: set[str] = set()
        with self._connect() as conn:
            for rel in edges:
                src_id = str(rel.get("src") or "")
                dst_id = str(rel.get("dst") or "")
                rkind = str(rel.get("kind") or "")
                if rkind in {RelationKind.WRITES.value, RelationKind.DERIVES.value} and dst_id == fid:
                    row = self._entity_row(conn, src_id)
                    hit = self._hit(row, why="host_writer", with_snippet=False) if row else None
                    if hit and str(hit.get("kind") or "") not in _WRITER_SKIP_KINDS:
                        writers.append(hit)
                if rkind in _KERNEL_READER_KINDS:
                    other = src_id if dst_id == fid else dst_id if src_id == fid else ""
                    if not other or other == fid:
                        continue
                    row = self._entity_row(conn, other)
                    hit = self._hit(row, why="kernel_reader", with_snippet=False) if row else None
                    hid = str((hit or {}).get("id") or "")
                    if hit and hid and hid not in seen_readers:
                        if str(hit.get("kind") or "") not in _WRITER_SKIP_KINDS:
                            seen_readers.add(hid)
                            readers.append(hit)
        for hit in writers[:12]:
            with self._connect() as conn:
                stmt = _snapshot_statement(
                    conn, str(hit.get("file") or ""), int(hit.get("line_start") or 0)
                )
            if not stmt:
                continue
            facts = dict(hit.get("facts") or {}) if isinstance(hit.get("facts"), dict) else {}
            if len(stmt) > len(str(facts.get("rhs") or "")):
                facts["rhs"] = stmt
                hit["facts"] = facts
        writers.sort(key=_write_site_sort_key)
        cap = _candidate_limit(PRIMARY_CANDIDATES)
        for hit in writers[:cap]:
            if str(hit.get("snippet") or "").strip():
                continue
            window = ""
            with self._connect() as conn:
                window = _snapshot_snippet(
                    conn,
                    str(hit.get("file") or ""),
                    int(hit.get("line_start") or 0),
                )
            if window:
                hit["snippet"] = window
        primary = dict(primary)
        facts = dict(primary.get("facts") or {}) if isinstance(primary.get("facts"), dict) else {}
        if writers:
            best = writers[0]
            best_facts = best.get("facts") if isinstance(best.get("facts"), dict) else {}
            best_rhs = str(best_facts.get("rhs") or "")
            if _trivial_rhs(str(facts.get("rhs") or "")) and not _trivial_rhs(best_rhs):
                facts["rhs"] = best_rhs
            facts["primary_write"] = {
                "id": best.get("id"),
                "name": best.get("name"),
                "file": best.get("file"),
                "line": best.get("line_start"),
                "rhs": best_rhs,
            }
            keep_snip = _snippet_covers_line(
                str(primary.get("snippet") or ""), int(primary.get("line_start") or 0)
            )
            if not keep_snip and best.get("snippet"):
                primary["snippet"] = best["snippet"]
            primary["facts"] = facts
        fused = list(facts.get("fused_outer_candidates") or [])
        if fused:
            facts["fused_outer_candidates"] = fused
            primary["facts"] = facts
        candidates = writers[:cap] or fields[:cap]
        if fused:
            patched: list[dict[str, Any]] = []
            for hit in candidates:
                item = dict(hit)
                hf = dict(item.get("facts") or {}) if isinstance(item.get("facts"), dict) else {}
                hf.setdefault("fused_outer_candidates", fused)
                item["facts"] = hf
                patched.append(item)
            candidates = patched
        occupancy = ""
        queried = alias_from or raw
        occupancy_names = [queried, str(primary.get("name") or "")] + _alias_ident_names(facts)
        if any(_is_occupancy_ident(name) for name in occupancy_names):
            occupancy = f"{queried} vs aicNum"
            facts["occupancy_axis"] = occupancy
            primary["facts"] = facts
        coverage = _hits_coverage(candidates + fields, total=len(fields) or len(candidates))
        coverage["fused_outer_candidates_count"] = max(
            int(coverage.get("fused_outer_candidates_count") or 0), len(fused)
        )
        if occupancy:
            coverage["occupancy_axis"] = occupancy
        if len(candidates) > 1 or fused:
            coverage["completeness"] = "siblings_checked"
            coverage["answerable"] = True
        payload = {
            "ok": True,
            "field": primary,
            "fields_matched": len(fields),
            "candidates": candidates,
            "writers": writers[:12],
            "readers": readers[:12],
            "edges": edges[:8],
            "coverage": coverage,
            "occupancy_axis": occupancy or None,
        }
        if alias_from:
            payload["alias_from"] = alias_from
            payload["canonical"] = str(primary.get("name") or "")
        return _fit_payload(payload)

    def constant(self, name: str) -> list[dict[str, Any]]:
        needle = str(name or "").lower()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE e.kind IN ('COMPILE_VAR', 'MACRO')
                  AND e.name COLLATE NOCASE LIKE ?
                LIMIT 20
                """,
                (f"%{needle}%",),
            ).fetchall()
            return self._hits_from_rows(conn, rows, with_snippet=True)

    def locate(
        self, query: str, *, kinds: Iterable[str] | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = self.search(query, kinds=kinds or (), limit=limit)
        return self._locations_with_sites(rows, limit=limit)

    def locate_dim(self, name: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE e.kind = 'TILING_KEY' AND IFNULL(e.name, '') = ?
                LIMIT ?
                """,
                (str(name or ""), int(limit)),
            ).fetchall()
            hits = self._hits_from_rows(conn, rows, why="locate_dim", with_snippet=True)
        return self._locations_with_sites(hits, limit=limit)

    def locate_branch(self, branch_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            row = self._entity_row(conn, branch_id)
            if row is None or str(row["kind"] or "") != EntityKind.BRANCH.value:
                return []
            hit = self._hit(row, with_snippet=True)
        return ([self._location(hit)] if hit else [])[:limit]

    def locate_field(self, name: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._locations_with_sites(self.tiling_field(name), limit=limit)

    def _locations_with_sites(
        self, rows: list[dict[str, Any]], *, limit: int
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()
        for row in rows:
            loc = self._location(row)
            file = str(loc.get("file") or "")
            line = int(loc.get("line_start") or 0)
            key = (str(loc.get("id") or ""), file, line)
            if file and line > 0 and key not in seen:
                seen.add(key)
                out.append(loc)
            attrs: dict[str, Any] = {}
            if isinstance(row.get("facts"), dict):
                attrs.update(row["facts"])
            if isinstance(row.get("data"), dict):
                attrs.update(row["data"])
            for extra in locations_from_attr_sites(
                str(row.get("id") or ""), str(row.get("kind") or ""), attrs or row
            ):
                extra_file = str(extra.file or "").replace("\\", "/")
                extra_line = int(extra.line_start or 0)
                extra_key = (extra.entity_id, extra_file, extra_line)
                if extra_key in seen:
                    continue
                seen.add(extra_key)
                if extra_file and extra_file == file.replace("\\", "/"):
                    host = out[-1] if out else loc
                    hf = dict(host.get("facts") or {}) if isinstance(host.get("facts"), dict) else {}
                    sites = list(hf.get("definition_sites") or [])
                    if extra_line > 0 and not any(
                        int(site.get("line") or site.get("line_start") or 0) == extra_line
                        for site in sites
                        if isinstance(site, dict)
                    ):
                        sites.append(
                            {
                                "file": extra_file,
                                "line": extra_line,
                                "line_start": extra_line,
                                "name": row.get("name"),
                            }
                        )
                        hf["definition_sites"] = sites
                        host["facts"] = hf
                    continue
                payload = extra.to_dict()
                payload.setdefault("id", extra.entity_id)
                payload.setdefault("name", row.get("name"))
                payload.setdefault("kind", extra.kind)
                if not payload.get("snippet"):
                    with self._connect() as conn:
                        payload["snippet"] = _snapshot_snippet(
                            conn, extra.file, extra.line_start
                        )
                payload["snippet"] = _cap_snippet(
                    str(payload.get("snippet") or ""), int(payload.get("line_start") or 0)
                )
                out.append(payload)
                if len(out) >= max(int(limit) * 4, 24):
                    break
        return _collapse_locate_hits(out)[: int(limit)]

    @staticmethod
    def _location(row: dict[str, Any]) -> dict[str, Any]:
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        return {
            "id": row.get("id"),
            "kind": row.get("kind"),
            "name": row.get("name"),
            "file": row.get("file"),
            "line_start": row.get("line_start"),
            "line_end": row.get("line_end"),
            "snippet": row.get("snippet") or facts.get("snippet") or "",
            "facts": facts,
            "relationships": row.get("relationships") or [],
        }

    def operator_api(self) -> dict[str, Any]:
        with self._connect() as conn:
            inputs = self._hits_from_rows(
                conn,
                self._select_entities(conn, kinds=("INPUT",), limit=200),
                with_snippet=False,
            )
            outputs = self._hits_from_rows(
                conn,
                self._select_entities(conn, kinds=("OUTPUT",), limit=200),
                with_snippet=False,
            )

        def _api_index(hit: dict[str, Any], key: str) -> int:
            facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
            return int(facts.get(key) or 0)

        tensor = [h for h in inputs if (h.get("facts") or {}).get("api_kind") == "tensor"]
        attrs = [h for h in inputs if (h.get("facts") or {}).get("api_kind") == "attribute"]
        tensor.sort(key=lambda h: _api_index(h, "api_index"))
        attrs.sort(key=lambda h: _api_index(h, "api_attr_index"))
        outputs.sort(key=lambda h: _api_index(h, "api_index"))
        return {"tensor_inputs": tensor, "attributes": attrs, "outputs": outputs}

    def input_roots(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return self._hits_from_rows(
                conn, self._select_entities(conn, kinds=("INPUT",), limit=400), with_snippet=False
            )

    def output_roots(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return self._hits_from_rows(
                conn, self._select_entities(conn, kinds=("OUTPUT",), limit=400), with_snippet=False
            )

    def _repair_packing_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        facts = hit.get("facts")
        if not isinstance(facts, dict):
            return hit
        sites = facts.get("packing_value_sites")
        if not isinstance(sites, list):
            return hit
        repaired: list[Any] = []
        for site in sites:
            if not isinstance(site, dict):
                repaired.append(site)
                continue
            item = dict(site)
            rhs = str(item.get("rhs") or "")
            stmt = ""
            with self._connect() as conn:
                stmt = _snapshot_statement(
                    conn, str(item.get("file") or ""), int(item.get("line") or 0)
                )
            if stmt and (
                len(stmt) > len(rhs)
                or _rhs_looks_truncated(rhs)
                or (_trivial_rhs(rhs) and not _trivial_rhs(stmt))
            ):
                item["rhs"] = stmt
            repaired.append(item)
        repaired.sort(key=_packing_site_sort_key)
        facts = dict(facts)
        facts["packing_value_sites"] = repaired
        hit = dict(hit)
        hit["facts"] = facts
        best = next((site for site in repaired if isinstance(site, dict)), None)
        if best is not None:
            window = ""
            with self._connect() as conn:
                window = _snapshot_snippet(
                    conn, str(best.get("file") or ""), int(best.get("line") or 0)
                )
            if window:
                hit["snippet"] = window
        return hit

    def tiling_keys(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet,
                       CAST(IFNULL(json_extract(e.data, '$.decl_order'), 0) AS INTEGER) AS decl_order
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE e.kind = 'TILING_KEY'
                ORDER BY decl_order, e.name
                """
            ).fetchall()
            hits = self._hits_from_rows(conn, rows, with_snippet=True, with_rels=True)
        return [self._repair_packing_hit(hit) for hit in hits]

    def tiling_data(self, name: str = "") -> list[dict[str, Any]]:
        needle = str(name or "").strip()
        with self._connect() as conn:
            if needle:
                rows = conn.execute(
                    """
                    SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                           IFNULL(s.snippet, '') AS snippet
                    FROM entity e
                    LEFT JOIN source_span s ON s.entity_id = e.id
                    WHERE (
                        e.kind = 'TILING_DATA'
                        OR (
                            e.kind = 'TYPE'
                            AND (
                                json_extract(e.data, '$.role') = 'type_alias'
                                OR IFNULL(json_extract(e.data, '$.alias_of'), '') != ''
                            )
                        )
                    )
                      AND (e.name = ? OR e.id = ? OR e.name LIKE ?)
                    LIMIT 40
                    """,
                    (needle, needle, f"%{needle}%"),
                ).fetchall()
            else:
                rows = self._select_entities(conn, kinds=("TILING_DATA",), limit=80)
            return self._hits_from_rows(conn, rows, with_snippet=True)

    def tiling_fields(self, owner: str = "") -> list[dict[str, Any]]:
        with self._connect() as conn:
            if owner:
                rows = conn.execute(
                    """
                    SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                           IFNULL(s.snippet, '') AS snippet
                    FROM entity e
                    LEFT JOIN source_span s ON s.entity_id = e.id
                    WHERE e.kind = 'TILING_FIELD'
                      AND json_extract(e.data, '$.owner') = ?
                    LIMIT 200
                    """,
                    (owner,),
                ).fetchall()
            else:
                rows = self._select_entities(conn, kinds=("TILING_FIELD",), limit=200)
            return self._hits_from_rows(conn, rows, with_snippet=False)

    def tiling_registrations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.id AS rel_id, r.kind AS rel_kind, r.src, r.dst, r.status AS rel_status,
                       s.id AS sid, s.kind AS skind, s.name AS sname, s.status AS sstatus,
                       s.file AS sfile, s.line_start AS sline, s.line_end AS slend,
                       d.id AS did, d.kind AS dkind, d.name AS dname, d.status AS dstatus,
                       d.file AS dfile, d.line_start AS dline, d.line_end AS dlend
                FROM relation r
                JOIN entity s ON s.id = r.src
                JOIN entity d ON d.id = r.dst
                WHERE r.kind = 'SELECTS'
                  AND s.kind = 'PREDICATE'
                  AND d.kind = 'TILING_DATA'
                  AND (
                    json_extract(s.data, '$.predicate_role') = 'packed_tiling_key_registration'
                    OR s.data LIKE '%packed_tiling_key_registration%'
                  )
                LIMIT 80
                """
            ).fetchall()
        return [
            {
                "predicate": {
                    "id": row["sid"],
                    "kind": row["skind"],
                    "name": row["sname"],
                    "status": row["sstatus"],
                    "file": row["sfile"],
                    "line_start": row["sline"],
                    "line_end": row["slend"],
                },
                "tiling_data": {
                    "id": row["did"],
                    "kind": row["dkind"],
                    "name": row["dname"],
                    "status": row["dstatus"],
                    "file": row["dfile"],
                    "line_start": row["dline"],
                    "line_end": row["dlend"],
                },
                "relation": {
                    "id": row["rel_id"],
                    "kind": row["rel_kind"],
                    "src": row["src"],
                    "dst": row["dst"],
                    "status": row["rel_status"],
                },
            }
            for row in rows
        ]

    def unresolved(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE lower(e.status) IN ('unresolved', 'partial', 'unknown')
                LIMIT 400
                """
            ).fetchall()
            return self._hits_from_rows(conn, rows, with_snippet=False)

    def _lazy_engine(self):
        if self._engine is None:
            from ascendc_codemap_mcp.engine.query.engine import CodeMapQuery
            from ascendc_codemap_mcp.engine.store.reader import read_codemap

            self._engine = CodeMapQuery(read_codemap(self.product), path=str(self.product))
        return self._engine

    def audit(self) -> dict[str, Any]:
        """Offline / expensive: hydrates the full CodeMap for path-existence audit."""
        return self._lazy_engine().audit()

    def summary(self) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.store.reader import load_view_blob, read_meta

        blob = load_view_blob(self.product, "summary", expand_legal_keys=False)
        if isinstance(blob, dict) and blob:
            return dict(blob)
        meta = read_meta(self.product)
        with self._connect() as conn:
            by_kind = {
                str(k): int(n)
                for k, n in conn.execute("SELECT kind, COUNT(*) FROM entity GROUP BY kind")
            }
            by_rel = {
                str(k): int(n)
                for k, n in conn.execute("SELECT kind, COUNT(*) FROM relation GROUP BY kind")
            }
            entity_count = int(conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0])
            relation_count = int(conn.execute("SELECT COUNT(*) FROM relation").fetchone()[0])
        return {
            "op_name": meta.get("op_name") or "",
            "architecture": meta.get("architecture") or "",
            "entity_count": entity_count,
            "relation_count": relation_count,
            "entities_by_kind": by_kind,
            "relations_by_kind": by_rel,
        }

    def _slice(
        self,
        seed_ids: Iterable[str],
        *,
        edge_kinds: Iterable[str] | None,
        depth: int,
        budget: int,
        direction: str,
        include_advisory: bool = False,
    ) -> dict[str, Any]:
        wanted = {
            str(kind.value if hasattr(kind, "value") else kind).upper()
            for kind in (edge_kinds or ())
        }
        if not wanted:
            wanted = set(USEFUL_EDGE_KINDS)
        max_depth = max(0, int(depth))
        cap = max(1, int(budget))
        placeholders = ",".join("?" for _ in wanted)
        col_from, col_to = ("src", "dst") if direction == "forward" else ("dst", "src")
        with self._connect() as conn:
            present: list[str] = []
            for seed in seed_ids:
                row = self._entity_row(conn, str(seed))
                if row is not None:
                    present.append(str(row["id"]))
            seen: set[str] = set()
            queue: deque[tuple[str, int]] = deque()
            for seed in present:
                if seed not in seen and len(seen) < cap:
                    seen.add(seed)
                    queue.append((seed, 0))
            included: list[sqlite3.Row] = []
            truncated = len(set(present)) > cap
            wanted_sorted = sorted(wanted)
            while queue:
                current, distance = queue.popleft()
                if distance >= max_depth:
                    continue
                for rel in conn.execute(
                    f"""
                    SELECT id, kind, src, dst, status, data
                    FROM relation
                    WHERE {col_from} = ? AND kind IN ({placeholders})
                    ORDER BY kind, src, dst, id
                    """,
                    (current, *wanted_sorted),
                ):
                    if not include_advisory and _is_advisory_data(rel["data"]):
                        continue
                    other = str(rel[col_to] or "")
                    if not other:
                        continue
                    if other not in seen:
                        if len(seen) >= cap:
                            truncated = True
                            continue
                        if self._entity_row(conn, other) is None:
                            continue
                        seen.add(other)
                        queue.append((other, distance + 1))
                    included.append(rel)
            nodes: list[dict[str, Any]] = []
            for eid in sorted(seen):
                row = self._entity_row(conn, eid)
                if row is None:
                    continue
                hit = self._hit(row, with_snippet=False)
                if hit is not None:
                    hit["evidence_tier"] = "B"
                    nodes.append(hit)
            edges = [
                {
                    "id": str(rel["id"] or ""),
                    "kind": str(rel["kind"] or ""),
                    "src": str(rel["src"] or ""),
                    "dst": str(rel["dst"] or ""),
                    "status": str(rel["status"] or ""),
                    "evidence_tier": "B",
                }
                for rel in included
            ]
        return {
            "nodes": nodes,
            "edges": edges,
            "evidence_tier_hints": {"B": len(nodes) + len(edges)},
            "truncated": truncated,
        }

    def slice_forward(
        self,
        seed_ids: Iterable[str],
        *,
        edge_kinds: Iterable[str] | None = None,
        depth: int = 3,
        budget: int = 500,
        include_advisory: bool = False,
    ) -> dict[str, Any]:
        return self._slice(
            seed_ids,
            edge_kinds=edge_kinds,
            depth=depth,
            budget=budget,
            direction="forward",
            include_advisory=include_advisory,
        )

    def slice_backward(
        self,
        seed_ids: Iterable[str],
        *,
        edge_kinds: Iterable[str] | None = None,
        depth: int = 3,
        budget: int = 500,
        include_advisory: bool = False,
    ) -> dict[str, Any]:
        return self._slice(
            seed_ids,
            edge_kinds=edge_kinds,
            depth=depth,
            budget=budget,
            direction="backward",
            include_advisory=include_advisory,
        )

    def find_path(self, start: str, end: str | None = None, *, end_kind: str = "") -> list[dict[str, Any]]:
        with self._connect() as conn:
            starts = conn.execute(
                """
                SELECT id, kind, name FROM entity
                WHERE name = ? OR id = ?
                ORDER BY CASE kind
                  WHEN 'INPUT' THEN 0 WHEN 'OUTPUT' THEN 1 WHEN 'TILING_KEY' THEN 2
                  ELSE 9 END, id
                LIMIT 8
                """,
                (start, start),
            ).fetchall()
            if not starts:
                return []
            end_id = None
            end_kinds: set[str] = set()
            if end_kind:
                end_kinds.add(str(end_kind).upper())
            if end:
                ends = conn.execute(
                    "SELECT id, kind FROM entity WHERE name = ? OR id = ? LIMIT 4",
                    (end, end),
                ).fetchall()
                if ends:
                    end_id = str(ends[0]["id"])
                elif str(end).upper() in {k.value for k in EntityKind}:
                    end_kinds.add(str(end).upper())
            if not end_id and not end_kinds:
                end_kinds.add("KERNEL")
            for src in starts:
                path = self._bfs_path(conn, str(src["id"]), end_id=end_id, end_kinds=end_kinds)
                if path:
                    hits: list[dict[str, Any]] = []
                    for eid in path:
                        row = self._entity_row(conn, eid)
                        hit = self._hit(row, with_snippet=False) if row else None
                        if hit:
                            hits.append(hit)
                    return hits
        return []

    def _bfs_path(
        self,
        conn: sqlite3.Connection,
        start_id: str,
        *,
        end_id: str | None,
        end_kinds: set[str],
        max_depth: int = 16,
    ) -> list[str]:
        prev: dict[str, str | None] = {start_id: None}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        found: str | None = None
        while queue:
            cur, dist = queue.popleft()
            row = self._entity_row(conn, cur)
            kind = str(row["kind"] or "") if row is not None else ""
            if end_id and cur == end_id:
                found = cur
                break
            if end_kinds and kind in end_kinds and cur != start_id:
                found = cur
                break
            if dist >= max_depth:
                continue
            for rel in conn.execute("SELECT dst, data FROM relation WHERE src = ?", (cur,)):
                if _is_advisory_data(rel["data"]):
                    continue
                nxt = str(rel["dst"] or "")
                if not nxt or nxt in prev:
                    continue
                prev[nxt] = cur
                queue.append((nxt, dist + 1))
        if not found:
            return []
        path = [found]
        while prev.get(path[-1]) is not None:
            path.append(prev[path[-1]] or "")
        path.reverse()
        return [p for p in path if p]

    def selected_kernel(self, key_name: str = "") -> list[dict[str, Any]]:
        with self._connect() as conn:
            if not key_name:
                return self._hits_from_rows(
                    conn, self._select_entities(conn, kinds=("KERNEL",), limit=40), with_snippet=True
                )
            start = self._entity_row(conn, key_name)
            if start is None:
                return []
            eid = str(start["id"])
            rows = conn.execute(
                """
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM relation r
                JOIN entity e ON e.id = r.dst
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE r.src = ? AND r.kind IN ('SELECTS', 'CONTROLS')
                LIMIT 40
                """,
                (eid,),
            ).fetchall()
            return self._hits_from_rows(conn, rows, with_snippet=True)

    def available_arch(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return self._hits_from_rows(
                conn, self._select_entities(conn, kinds=("ARCH",), limit=20), with_snippet=False
            )

    def aggregate_tiling_key(self, pattern: str = "", *, limit: int = 50) -> dict[str, Any]:
        needle = str(pattern or "").strip()
        keys = self.tiling_keys()
        if needle:
            low = needle.lower()
            keys = [
                k
                for k in keys
                if low in str(k.get("name") or "").lower() or low in str(k.get("id") or "").lower()
            ]
        keys = keys[: max(0, int(limit))]
        return _fit_payload(
            {
                "ok": True,
                "mode": "tiling_key",
                "pattern": needle,
                "keys": keys,
                "count": len(keys),
                "files": _group_by_file(keys),
            }
        )

    def aggregate_tiling_data(self, pattern: str = "", *, limit: int = 50) -> dict[str, Any]:
        needle = str(pattern or "").strip()
        if needle:
            fields = self._named_fields(
                needle, kinds=(EntityKind.TILING_FIELD.value, EntityKind.FIELD.value)
            )[: int(limit)]
            impact = self.field_impact(needle) if fields else {"ok": False}
            data = self.tiling_data(needle)
        else:
            fields = self.tiling_fields()[: int(limit)]
            impact = {}
            data = self.tiling_data()
        if needle:
            count = len(data) if data else len(fields)
        else:
            count = len(fields)
        return _fit_payload(
            {
                "ok": True,
                "mode": "tiling_data",
                "pattern": needle,
                "tiling_data": data[: int(limit)],
                "fields": fields,
                "impact": impact,
                "count": count,
            }
        )

    def aggregate_kernel_branch(self, pattern: str = "", *, limit: int = 50) -> dict[str, Any]:
        needle = str(pattern or "").strip()
        tokens = _TOKEN_RE.findall(needle) or ([needle] if needle else [])
        name_tok = tokens[0] if tokens else ""
        func_tok = tokens[1] if len(tokens) > 1 else ""
        with self._connect() as conn:
            params: list[Any] = []
            where = ["e.kind = 'BRANCH'", "IFNULL(e.file, '') != ''", "IFNULL(e.line_start, 0) > 0"]
            if name_tok:
                where.append(
                    """(
                    e.name = ?
                    OR e.name = ? COLLATE NOCASE
                    OR e.name LIKE '%::' || ? COLLATE NOCASE
                    OR e.name LIKE '%.' || ? COLLATE NOCASE
                    OR lower(IFNULL(json_extract(e.data, '$.condition'), '')) LIKE lower(?)
                    OR lower(IFNULL(json_extract(e.data, '$.predicate'), '')) LIKE lower(?)
                    OR lower(IFNULL(e.data, '')) LIKE lower(?)
                    )"""
                )
                params.extend(
                    [
                        name_tok,
                        name_tok,
                        name_tok,
                        name_tok,
                        f"%{name_tok}%",
                        f"%{name_tok}%",
                        f"%{name_tok}%",
                    ]
                )
            if func_tok:
                where.append(
                    "(json_extract(e.data, '$.function') = ? OR lower(IFNULL(json_extract(e.data, '$.function'), '')) LIKE lower(?))"
                )
                params.extend([func_tok, f"%{func_tok}%"])
            order_name = name_tok or ""
            rows = conn.execute(
                f"""
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE {' AND '.join(where)}
                ORDER BY
                  CASE WHEN e.name = ? THEN 0 ELSE 1 END,
                  e.id
                LIMIT ?
                """,
                tuple(params + [order_name, max(int(limit) * 24, 80)]),
            ).fetchall()
            branches = self._hits_from_rows(
                conn, rows, why="kernel_branch", with_snippet=True, with_rels=True
            )
            if name_tok:
                low = name_tok.lower()
                kept: list[dict[str, Any]] = []
                for hit in branches:
                    facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
                    name_ok = (
                        str(hit.get("name") or "").lower() == low
                        or low in str(hit.get("name") or "").lower()
                        or low in str(facts.get("condition") or "").lower()
                        or low in str(facts.get("predicate") or "").lower()
                    )
                    fn = str(facts.get("function") or "")
                    fn_ok = (not func_tok) or func_tok.lower() in fn.lower()
                    if name_ok and fn_ok:
                        kept.append(hit)
                branches = kept
        branches.sort(key=_branch_sort_key)
        total = len(branches)
        cap = _candidate_limit(limit)
        if func_tok:
            functions: dict[str, int] = {}
            for hit in branches:
                facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
                fn = str(facts.get("function") or "").strip() or "(unknown)"
                functions[fn] = functions.get(fn, 0) + 1
            exemplars = branches[:cap]
        else:
            exemplars, functions = _diversify_by_function(branches, limit=max(cap, int(limit)))
            exemplars = exemplars[:cap]
        coverage = _hits_coverage(exemplars, total=total)
        payload = {
                "ok": True,
                "mode": "kernel_branch",
                "pattern": needle,
                "coverage": coverage,
                "branches": exemplars,
                "count": total,
                "functions": functions,
                "files": _group_by_file(exemplars),
            }
        if total == 0:
            payload["empty_reason"] = "not_extracted"
            payload["hint"] = (
                "Kernel if reading this tiling field was not extracted as BRANCH. "
                "count=0 is not proof that the branch is absent."
            )
        return _fit_payload(payload)

    def _sql_template_cover(
        self, structured: dict[str, str], dim_only: str
    ) -> dict[str, Any] | None:
        """Cover query over template_block tables. None if the product has no accel."""
        from ascendc_codemap_mcp.engine.store.accel import has_template_block
        from ascendc_codemap_mcp.engine.tpl_dsl import bool_value_aliases

        with self._connect() as conn:
            if not has_template_block(conn):
                return None
            total = int(
                conn.execute("SELECT COUNT(*) FROM template_block").fetchone()[0] or 0
            )
            if total <= 0:
                return None
            if dim_only:
                values = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT DISTINCT value FROM template_block_dim WHERE dim = ? ORDER BY value",
                        (dim_only,),
                    )
                ]
                return {
                    "ok": True,
                    "dim_only": dim_only,
                    "dim_coverage": {dim_only: values},
                    "matching_block_count": 0,
                    "block_matches": [],
                    "nearby": [],
                    "sel_sites": [],
                }

            ids: set[int] | None = None
            if not structured:
                match_ids = [
                    int(row[0])
                    for row in conn.execute("SELECT id FROM template_block ORDER BY id")
                ]
            else:
                for dim, value in structured.items():
                    aliases = {str(value)}
                    aliases.update(bool_value_aliases(value))
                    placeholders = ",".join("?" for _ in aliases)
                    found = {
                        int(row[0])
                        for row in conn.execute(
                            f"SELECT DISTINCT block_id FROM template_block_dim "
                            f"WHERE dim = ? AND value IN ({placeholders})",
                            (dim, *sorted(aliases)),
                        )
                    }
                    ids = found if ids is None else ids & found
                match_ids = sorted(ids or [])
            block_matches: list[dict[str, Any]] = []
            sel_sites: list[dict[str, Any]] = []
            for bid in match_ids:
                row = conn.execute(
                    "SELECT name, file, line_start, line_end, data FROM template_block WHERE id = ?",
                    (bid,),
                ).fetchone()
                if row is None:
                    continue
                try:
                    data = json.loads(row[4] or "{}")
                except json.JSONDecodeError:
                    data = {}
                if isinstance(data, dict):
                    block_matches.append(data)
                file = str(row[1] or "")
                line = int(row[2] or 0)
                if file and line:
                    sel_sites.append(
                        {
                            "name": str(row[0] or ""),
                            "file": file,
                            "line": line,
                            "line_end": int(row[3] or line),
                        }
                    )
            dim_coverage: dict[str, list[str]] = {}
            if match_ids:
                placeholders = ",".join("?" for _ in match_ids)
                for dim, value in conn.execute(
                    f"SELECT dim, value FROM template_block_dim WHERE block_id IN ({placeholders})",
                    tuple(match_ids),
                ):
                    dim_coverage.setdefault(str(dim), [])
                    if str(value) not in dim_coverage[str(dim)]:
                        dim_coverage[str(dim)].append(str(value))
            # Same pin the YAML path applies: a queried dim is not a coverage
            # product. ``IsRope=1`` matching a block that also lists 0 would
            # otherwise report ``["0","1"]`` for the dim the caller already fixed.
            if structured and match_ids:
                for dim, value in structured.items():
                    aliases = {str(value)}
                    aliases.update(bool_value_aliases(value))
                    present = [v for v in dim_coverage.get(dim, []) if v in aliases]
                    canon = [v for v in present if v in {"0", "1"}]
                    dim_coverage[str(dim)] = (canon or present or [str(value)])[:1]
            nearby: list[dict[str, Any]] = []
            if structured and not match_ids:
                for drop in structured:
                    rest = {k: v for k, v in structured.items() if k != drop}
                    dropped = self._sql_template_cover(rest, "")
                    if dropped and dropped.get("matching_block_count"):
                        nearby.append(
                            {
                                "dropped": drop,
                                "matching_block_count": dropped["matching_block_count"],
                                "values": (dropped.get("dim_coverage") or {}).get(drop) or [],
                            }
                        )
            return {
                "ok": True,
                "dim_only": "",
                "dim_coverage": dim_coverage,
                "matching_block_count": len(block_matches),
                "block_matches": block_matches,
                "nearby": nearby,
                "sel_sites": sel_sites,
            }

    def _sel_sites_for_dim(self, dim: str, *, limit: int = 8) -> list[dict[str, Any]]:
        ident = _last_ident(str(dim or "").replace(".", "::"))
        if not ident:
            return []
        from ascendc_codemap_mcp.engine.store.accel import has_template_block

        with self._connect() as conn:
            if not has_template_block(conn):
                return []
            total = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT block_id) FROM template_block_dim WHERE dim = ?",
                    (ident,),
                ).fetchone()[0]
                or 0
            )
            rows = conn.execute(
                """
                SELECT b.name, b.file, b.line_start, b.line_end
                FROM template_block b
                JOIN template_block_dim d ON d.block_id = b.id
                WHERE d.dim = ? AND IFNULL(b.line_start, 0) > 0
                GROUP BY b.id
                ORDER BY b.line_start
                LIMIT ?
                """,
                (ident, max(int(limit), 1)),
            ).fetchall()
        sites = [
            {
                "name": str(row[0] or ""),
                "file": str(row[1] or ""),
                "line": int(row[2] or 0),
                "line_end": int(row[3] or 0),
            }
            for row in rows
            if int(row[2] or 0) > 0
        ]
        if sites:
            sites[0]["matching_block_count"] = total
            sites[0]["complete"] = len(sites) >= total
        return sites

    def aggregate_template_match(
        self,
        pattern: str = "",
        *,
        filters: dict[str, str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        needle = str(pattern or "").strip()
        rest, dim_only = normalize_cover_pattern(needle)
        structured = {
            str(k).strip(): str(v).strip()
            for k, v in dict(filters or {}).items()
            if str(k).strip() and str(v).strip()
        }
        if not structured:
            structured.update(_pattern_filters(rest if rest else needle))
        graph_pattern = "" if structured or dim_only else needle
        if graph_pattern:
            templates = self.templates_for_key(graph_pattern)
            macros = self.constant(graph_pattern)
        else:
            templates = []
            macros = []
        block_matches: list[dict[str, Any]] = []
        all_blocks: list[dict[str, Any]] = []
        block_status: dict[str, Any] = {"ok": True, "reason_code": "", "used": False}
        dim_coverage: dict[str, list[str]] = {}
        nearby: list[dict[str, Any]] = []
        matching_block_count = 0
        sel_sites: list[dict[str, Any]] = []
        sql_cover = self._sql_template_cover(structured, dim_only) if (structured or dim_only) else None
        if sql_cover is not None:
            block_status = {"ok": True, "reason_code": "", "used": True, "backend": "sql"}
            dim_coverage = sql_cover.get("dim_coverage") or {}
            matching_block_count = int(sql_cover.get("matching_block_count") or 0)
            block_matches = list(sql_cover.get("block_matches") or [])
            nearby = list(sql_cover.get("nearby") or [])
            sel_sites = list(sql_cover.get("sel_sites") or [])
        else:
            cached = _load_template_blocks_cached(self.product)
            block_status = {
                "ok": bool(cached.get("ok")),
                "reason_code": str(cached.get("reason_code") or ""),
                "used": bool(cached.get("ok")),
            }
            if cached.get("ok"):
                all_blocks = cached.get("rows") or []
                universe_coverage = _dim_coverage(all_blocks)
                if dim_only:
                    product = list(universe_coverage.get(dim_only) or [])
                    dim_coverage = {dim_only: product}
                    matching_block_count = 0
                    block_matches = []
                elif structured:
                    block_matches = [
                        row for row in all_blocks if _template_block_matches(row, structured)
                    ]
                    matching_block_count = len(block_matches)
                    dim_coverage = (
                        _dim_coverage_restricted(block_matches, structured)
                        if block_matches
                        else {}
                    )
                    if matching_block_count == 0:
                        nearby = _template_nearby(all_blocks, structured)
                else:
                    dim_coverage = universe_coverage
                    matching_block_count = len(all_blocks)
                    block_matches = all_blocks
        compact_blocks = [_compact_template_block(row) for row in block_matches]
        if not dim_only:
            compact_blocks = compact_blocks[:TEMPLATE_BLOCK_EXEMPLARS]
        else:
            compact_blocks = []
        fixed_coverage = _fixed_coverage(block_matches) if block_matches else {}
        universe_scanned = bool(block_status.get("ok") and block_status.get("used"))
        if dim_only:
            declared = self._declared_dim_values(dim_only)
            product = list(dim_coverage.get(dim_only) or [])
            completeness = "coverage_checked" if universe_scanned or declared else "first_hit"
            answerable = bool(product or declared)
        elif structured:
            completeness = "coverage_checked" if universe_scanned else "first_hit"
            answerable = bool(matching_block_count)
        else:
            completeness = "coverage_checked" if universe_scanned else "first_hit"
            answerable = bool(dim_coverage)
        coverage = {
            **_hits_coverage([], total=matching_block_count, dim_coverage=dim_coverage),
            "dim_coverage": dim_coverage,
            "fixed_coverage": fixed_coverage,
            "completeness": completeness,
            "answerable": answerable,
        }
        if nearby:
            coverage["nearby"] = nearby
        payload = {
            "ok": bool(block_status.get("ok")) if structured or dim_only else True,
            "mode": "template_match",
            "pattern": needle,
            "filters": structured,
            "coverage": coverage,
            "dim_coverage": dim_coverage,
            "matching_block_count": matching_block_count,
            "count": matching_block_count if structured or dim_only else len(templates),
            "templates": [] if structured or dim_only else templates[: int(limit)],
            "macros_compile_vars": [] if structured or dim_only else macros[: int(limit)],
            "template_blocks": compact_blocks,
            "template_projection": block_status,
            "fixed_coverage": fixed_coverage,
            "sel_sites": sel_sites[:8],
        }
        if dim_only:
            payload["dim_only"] = dim_only
            payload["cover_kind"] = "dim_list"
            payload["declared_coverage"] = {dim_only: self._declared_dim_values(dim_only)}
            payload["product_coverage"] = {dim_only: list(dim_coverage.get(dim_only) or [])}
            payload["count"] = len(payload["product_coverage"][dim_only])
        if nearby:
            payload["nearby"] = nearby
        attach_query_hints(payload, needle, count=int(payload["count"]))
        if structured and matching_block_count == 0:
            declared_hints: list[str] = []
            for dim_name, want in structured.items():
                domain = self._declared_dim_values(dim_name)
                if domain:
                    declared_hints.append(f"{dim_name} {{{', '.join(domain)}}}")
            if declared_hints:
                payload["hint"] = (
                    "declared "
                    + "; ".join(declared_hints)
                    + ", legal_key=0"
                )
        return _fit_payload(payload)

    def _declared_dim_values(self, name: str) -> list[str]:
        needle = str(name or "").strip()
        if not needle:
            return []
        found: list[str] = []
        seen: set[str] = set()

        def _add(raw: Any) -> None:
            if isinstance(raw, (list, tuple, set)):
                items = list(raw)
            elif raw is None:
                return
            else:
                items = [raw]
            for item in items:
                text = str(item)
                if text and text not in seen:
                    seen.add(text)
                    found.append(text)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT data FROM entity
                WHERE kind = 'TILING_KEY' AND IFNULL(name, '') = ?
                LIMIT 1
                """,
                (needle,),
            ).fetchone()
            if row:
                try:
                    data = json.loads(_row_get(row, "data") or row[0] or "{}")
                except (TypeError, json.JSONDecodeError, IndexError):
                    data = {}
                if isinstance(data, dict):
                    attrs = data.get("attrs") if isinstance(data.get("attrs"), dict) else data
                    for key in ("value_domain", "declared_values", "allowed_values", "domain"):
                        raw = None
                        if isinstance(attrs, dict):
                            raw = attrs.get(key)
                        if raw is None:
                            raw = data.get(key)
                        if raw is not None:
                            _add(raw)
                            break
            try:
                for extra in conn.execute(
                    "SELECT DISTINCT value FROM template_block_dim WHERE dim = ? ORDER BY value",
                    (needle,),
                ):
                    _add(extra[0])
            except sqlite3.OperationalError:
                pass
        return found

    def _legal_dim_value_counts(self, dim: str) -> dict[str, int]:
        needle = str(dim or "").strip()
        if not needle:
            return {}
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT value, COUNT(*) FROM legal_key_dim
                    WHERE dim = ?
                    GROUP BY value
                    ORDER BY value
                    """,
                    (needle,),
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
        return {str(row[0]): int(row[1] or 0) for row in rows if row[0] is not None}

    def _legal_cross_counts(
        self, dim: str, value: str, *, limit_dims: int = 12, only_dims: tuple[str, ...] | None = None
    ) -> dict[str, dict[str, int]]:
        """Sibling-dim counts under dim=value, including zeros for known values."""
        dname = str(dim or "").strip()
        dval = str(value or "").strip()
        if not dname or not dval:
            return {}
        cache_key = (dname, dval, tuple(only_dims or ()), int(limit_dims))
        cached = self._legal_cross_cache.get(cache_key)
        if cached is not None:
            return cached
        from ascendc_codemap_mcp.engine.tpl_dsl import bool_value_aliases

        aliases = [dval]
        for alt in bool_value_aliases(dval):
            if alt not in aliases:
                aliases.append(alt)
        with self._connect() as conn:
            try:
                marks = ",".join("?" for _ in aliases)
                sibling_filter = ""
                sibling_params: tuple[Any, ...] = (dname, *aliases, dname)
                want = [str(n).strip() for n in (only_dims or ()) if str(n).strip() and str(n).strip() != dname]
                if want:
                    marks_d = ",".join("?" for _ in want)
                    sibling_filter = f" AND s.dim IN ({marks_d})"
                    sibling_params = (dname, *aliases, dname, *want)
                sibling_rows = conn.execute(
                    f"""
                    SELECT s.dim, s.value, COUNT(*)
                    FROM legal_key_dim m
                    JOIN legal_key_dim s ON s.key_id = m.key_id
                    WHERE m.dim = ? AND m.value IN ({marks})
                      AND s.dim != ?{sibling_filter}
                    GROUP BY s.dim, s.value
                    """,
                    sibling_params,
                ).fetchall()
                if want:
                    universe = conn.execute(
                        f"""
                        SELECT dim, value FROM legal_key_dim
                        WHERE dim IN ({",".join("?" for _ in want)})
                        GROUP BY dim, value
                        """,
                        tuple(want),
                    ).fetchall()
                else:
                    universe = conn.execute(
                        """
                        SELECT dim, value FROM legal_key_dim
                        WHERE dim != ?
                        GROUP BY dim, value
                        """,
                        (dname,),
                    ).fetchall()
            except sqlite3.OperationalError:
                return {}
        present: dict[str, dict[str, int]] = {}
        for row in sibling_rows:
            present.setdefault(str(row[0]), {})[str(row[1])] = int(row[2] or 0)
        universe_map: dict[str, list[str]] = {}
        for row in universe:
            universe_map.setdefault(str(row[0]), []).append(str(row[1]))
        ranked: list[tuple[int, int, str]] = []
        for sdim, values in universe_map.items():
            declared = self._declared_dim_values(sdim)
            all_vals = list(dict.fromkeys(values))
            counts = {v: int(present.get(sdim, {}).get(v, 0)) for v in all_vals}
            for extra in declared:
                counts.setdefault(str(extra), int(present.get(sdim, {}).get(str(extra), 0)))
            has_pos = any(n > 0 for n in counts.values())
            has_zero = any(n == 0 for n in counts.values())
            prefix = dname[:4].lower()
            share = 0 if str(sdim).lower().startswith(prefix) and len(prefix) >= 3 else 1
            interesting = 0 if (has_pos and has_zero) else 1
            ranked.append((share, interesting, -sum(1 for n in counts.values() if n > 0), sdim))
            present[sdim] = counts
        ranked.sort()
        out: dict[str, dict[str, int]] = {}
        for _share, _interesting, _n, sdim in ranked[: max(1, int(limit_dims))]:
            out[sdim] = present.get(sdim) or {}
        self._legal_cross_cache[cache_key] = out
        return out

    def _legal_pair_cross(
        self,
        dim: str,
        value: str,
        dim_a: str,
        dim_b: str,
    ) -> dict[str, Any]:
        """Two-sibling grid under dim=value, including zero cells."""
        dname = str(dim or "").strip()
        dval = str(value or "").strip()
        a = str(dim_a or "").strip()
        b = str(dim_b or "").strip()
        if not dname or not dval or not a or not b or a == b:
            return {}
        from ascendc_codemap_mcp.engine.tpl_dsl import bool_value_aliases

        aliases = [dval]
        for alt in bool_value_aliases(dval):
            if alt not in aliases:
                aliases.append(alt)
        with self._connect() as conn:
            try:
                marks = ",".join("?" for _ in aliases)
                matched_sql = (
                    "SELECT key_id FROM legal_key_dim "
                    f"WHERE dim = ? AND value IN ({marks})"
                )
                matched_params = (dname, *aliases)
                vals_a = [
                    str(r[0])
                    for r in conn.execute(
                        "SELECT DISTINCT value FROM legal_key_dim WHERE dim = ? ORDER BY value",
                        (a,),
                    )
                ]
                vals_b = [
                    str(r[0])
                    for r in conn.execute(
                        "SELECT DISTINCT value FROM legal_key_dim WHERE dim = ? ORDER BY value",
                        (b,),
                    )
                ]
                hits = {
                    (str(r[0]), str(r[1])): int(r[2] or 0)
                    for r in conn.execute(
                        f"""
                        SELECT a.value, b.value, COUNT(*)
                        FROM legal_key_dim a
                        JOIN legal_key_dim b ON a.key_id = b.key_id
                        WHERE a.key_id IN ({matched_sql})
                          AND a.dim = ? AND b.dim = ?
                        GROUP BY a.value, b.value
                        """,
                        (*matched_params, a, b),
                    )
                }
            except sqlite3.OperationalError:
                return {}
        cells: list[dict[str, Any]] = []
        for va in vals_a:
            by_b: dict[str, int] = {}
            for vb in vals_b:
                by_b[vb] = int(hits.get((va, vb), 0))
            cells.append({"value": va, "counts": by_b})
        return {"dim_a": a, "dim_b": b, "cells": cells}

    def _legal_dim_exists(self, dim: str) -> bool:
        name = str(dim or "").strip()
        if not name:
            return False
        with self._connect() as conn:
            try:
                row = conn.execute(
                    "SELECT 1 FROM legal_key_dim WHERE dim = ? LIMIT 1",
                    (name,),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        return row is not None

    def _resolve_legal_dim(self, ident: str) -> str:
        """Map a host spelling onto a persisted legal_key dimension name."""
        for alias in _dim_aliases(ident):
            if self._legal_dim_exists(alias):
                return alias
        declared = {n.lower(): n for n in self._declared_dim_names()}
        for alias in _dim_aliases(ident):
            hit = declared.get(alias.lower())
            if hit:
                return hit
        return ""

    def compiled_support_for(
        self, ident: str, extras: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Host encoding ↔ persisted legal_key. Query never re-parses templates."""
        ident_key = str(ident or "").strip().lower()
        cached = self._compiled_support_cache.get(ident_key)
        if cached is not None:
            return dict(cached)
        dim = self._resolve_legal_dim(ident)
        if not dim:
            return None
        counts = self._legal_dim_value_counts(dim)
        variants = sum(int(n or 0) for n in counts.values())
        extra = extras if isinstance(extras, dict) else {}
        host: list[dict[str, Any]] = []
        expr = str(extra.get("value_expr") or extra.get("value") or extra.get("definition") or "")
        writers = list(extra.get("writers") or [])
        if not writers:
            try:
                field = self.field_impact(dim)
            except Exception:  # noqa: BLE001
                field = {}
            writers = list(field.get("writers") or []) if field.get("ok") else []
        for row in writers[:4]:
            if not isinstance(row, dict):
                continue
            file = str(row.get("file") or "")
            line = int(row.get("line") or row.get("line_start") or 0)
            if not file or line <= 0:
                continue
            item: dict[str, Any] = {
                "name": str(row.get("name") or ""),
                "file": file,
                "line": line,
            }
            if expr:
                item["expr"] = expr
            host.append(item)
        if expr and not host:
            host.append({"expr": expr})
        on_vals = [v for v in counts if str(v) not in {"0", "false", "False"}]
        on_val = "1" if "1" in counts else (on_vals[0] if on_vals else "")
        kernel: dict[str, Any] = {dim: dict(counts)}
        if on_val:
            want = (
                "DTemplate",
                "DTemplateNum",
                "IsDNoEqual",
                "dNoEqual",
                "IsRope",
                "hasRope",
            )
            cross = self._legal_cross_counts(dim, str(on_val), only_dims=want)
            for sdim in want:
                if sdim == dim:
                    continue
                cmap = cross.get(sdim)
                if isinstance(cmap, dict) and cmap:
                    kernel[sdim] = cmap
        support: dict[str, Any] = {
            "legal": variants > 0,
            "variants": variants,
            "dim": dim,
            "values": counts,
            "host_encoding": host,
            "kernel": kernel,
        }
        blob = f"{dim} {ident}".lower()
        if "rope" in blob:
            d_dim = ""
            for cand in ("DTemplate", "DTemplateNum"):
                if cand != dim and self._legal_dim_exists(cand):
                    d_dim = cand
                    break
            if d_dim:
                combo = self.legal_key_query(filters={d_dim: "128", dim: "1"}, limit=1)
                n = int(combo.get("total_matched") or combo.get("count") or 0)
                support["counterfactual"] = {
                    d_dim: "128",
                    dim: "1",
                    "legal": n > 0,
                    "variants": n,
                }
        self._compiled_support_cache[ident_key] = dict(support)
        return support

    def aggregate_locate(self, pattern: str = "", *, limit: int = 20) -> dict[str, Any]:
        needle = str(pattern or "").strip()
        tokens = search_needles(needle) or ([needle] if needle else [])
        fetch_limit = max(int(limit) * 4, 24)
        rows: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any, Any]] = set()
        for token in tokens:
            if not token:
                continue
            chunk = self.locate_dim(token, limit=fetch_limit) if token else []
            if not chunk:
                chunk = self.locate_field(token, limit=fetch_limit)
            if not chunk:
                chunk = self.locate(token, limit=fetch_limit)
            for loc in chunk:
                key = (loc.get("id"), loc.get("file"), loc.get("line_start"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(loc)
                if len(rows) >= fetch_limit:
                    break
            if len(rows) >= fetch_limit:
                break
        rows = _collapse_locate_hits(rows)
        rows.sort(
            key=lambda hit: _agent_sort_key(
                hit, needle, architecture=self._architecture
            )
        )
        page, meta = _page_by_exactness(rows, needle, limit=int(limit))
        coverage = _hits_coverage(
            page,
            total=int(meta["total"]),
            clipped=bool(meta["clipped"]),
            needle=needle,
            substring_only=bool(meta["substring_only"]),
        )
        payload = {
            "ok": True,
            "mode": "locate",
            "pattern": needle,
            "coverage": coverage,
            "locations": page,
            "count": int(meta["total"]),
            "files": _group_by_file(page),
        }
        if len(tokens) > 1:
            payload["pattern_tokens"] = tokens
        attach_query_hints(payload, needle, count=len(rows))
        return _fit_payload(payload)

    def _kernel_launch_entry(
        self, pattern: str, *, scope: str = "", winner_file: str = ""
    ) -> list[dict[str, Any]]:
        """Constructing function/KERNEL — not a compile-unit ``source_scope`` stub."""
        needle = str(pattern or "").strip()
        hits: list[dict[str, Any]] = []
        if needle:
            hits = list(self.locate(needle, limit=12) or [])
        if hits:
            hits = [
                hit
                for hit in hits
                if not _is_compile_unit_placeholder(str(hit.get("name") or ""))
            ]
            hits.sort(
                key=lambda hit: _agent_sort_key(
                    hit, needle, architecture=self._architecture
                )
            )
            return hits
        kinds = (
            EntityKind.KERNEL.value,
            EntityKind.FUNCTION.value,
            EntityKind.METHOD.value,
        )
        placeholders = ",".join("?" for _ in kinds)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE e.kind IN ({placeholders})
                  AND (
                    e.kind = 'KERNEL'
                    OR lower(IFNULL(e.file, '')) LIKE '%entry%'
                    OR e.name COLLATE NOCASE LIKE '%entry%'
                    OR (? != '' AND IFNULL(e.name, '') = ?)
                  )
                LIMIT 48
                """,
                (*kinds, str(scope or ""), str(scope or "")),
            ).fetchall()
            hits = self._hits_from_rows(
                conn, rows, why="kernel_launch_entry", with_snippet=True, with_rels=False
            )
        hits = [
            hit
            for hit in hits
            if not _is_compile_unit_placeholder(str(hit.get("name") or ""))
        ]
        want = str(scope or "").strip()
        win = str(winner_file or "").replace("\\", "/").lower()

        def _launch_entry_key(hit: dict[str, Any]) -> tuple[Any, ...]:
            kind = str(hit.get("kind") or "")
            name = str(hit.get("name") or "")
            file = str(hit.get("file") or "").replace("\\", "/").lower()
            same_file = 0 if win and (file == win or file.endswith("/" + win) or win.endswith("/" + file)) else 1
            scope_hit = 0 if want and name == want else 1
            kind_rank = {
                EntityKind.KERNEL.value: 0,
                EntityKind.FUNCTION.value: 1,
                EntityKind.METHOD.value: 2,
            }.get(kind, 3)
            return (
                scope_hit,
                same_file,
                kind_rank,
                _agent_sort_key(hit, want or "entry", architecture=self._architecture),
            )

        hits.sort(key=_launch_entry_key)
        return hits

    def _source_text(self, file: str) -> str:
        with self._connect() as conn:
            return _source_file_text(conn, file)

    def _destroy_ops_in_file(self, winner_file: str) -> list[tuple[int, str]]:
        want = str(winner_file or "").replace("\\", "/").lower()
        if not want:
            return []
        leaf = want.rsplit("/", 1)[-1]
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.file, e.line_start, e.data, e.name
                FROM entity e
                WHERE e.kind = 'OPERATION'
                  AND (
                    IFNULL(e.name, '') = 'Destroy'
                    OR IFNULL(json_extract(e.data, '$.callee'), '') = 'Destroy'
                  )
                LIMIT 200
                """
            ).fetchall()
        out: list[tuple[int, str]] = []
        for row in rows:
            file = str(_row_get(row, "file") or "").replace("\\", "/").lower()
            if file != want and not file.endswith("/" + leaf) and leaf not in file:
                continue
            data = _parse_data(_row_get(row, "data", "{}"))
            recv = receiver_leaf(str(data.get("receiver") or ""))
            if not recv:
                continue
            out.append((int(_row_get(row, "line_start") or 0), recv))
        return out

    def aggregate_kernel_launch(self, pattern: str = "", *, limit: int = 50) -> dict[str, Any]:
        """TPipe instances on the selected kernel entry, ordered by Destroy chain."""
        cache_key = (str(pattern or ""), max(int(limit), 1))
        cached = self._launch_cache.get(cache_key)
        if cached is not None:
            return cached
        with self._connect() as conn:
            pipe_rows = conn.execute(
                """
                SELECT e.id, e.kind, e.name, e.status, e.file, e.line_start, e.line_end, e.data,
                       IFNULL(s.snippet, '') AS snippet
                FROM entity e
                LEFT JOIN source_span s ON s.entity_id = e.id
                WHERE e.kind = 'PIPE'
                  AND IFNULL(json_extract(e.data, '$.catalog'), '') != 'ascendc'
                  AND IFNULL(json_extract(e.data, '$.pointer'), 0) != 1
                  AND IFNULL(e.file, '') != ''
                ORDER BY IFNULL(json_extract(e.data, '$.scope'), ''),
                         IFNULL(json_extract(e.data, '$.pipe_ordinal'), 0),
                         IFNULL(e.line_start, 0),
                         e.name
                LIMIT 200
                """
            ).fetchall()
            hits = self._hits_from_rows(
                conn, pipe_rows, why="kernel_launch_pipe", with_snippet=False, with_rels=False
            )
        selected, other_files = _select_launch_phases(
            hits, architecture=self._architecture, limit=max(int(limit), 1)
        )
        winner_file = str(selected[0].get("file") or "").replace("\\", "/") if selected else ""
        if selected:
            selected = _order_pipes_by_lifetime(
                selected,
                destroys=self._destroy_ops_in_file(winner_file),
                source_text=self._source_text(winner_file),
            )[: max(int(limit), 1)]
        group_total = sum(
            1
            for hit in hits
            if _is_launch_pipe_hit(hit)
            and str(hit.get("file") or "").replace("\\", "/").lower()
            == winner_file.replace("\\", "/").lower()
        ) if winner_file else 0
        clipped = bool(selected) and group_total > len(selected)
        phases: list[dict[str, Any]] = []
        for hit in selected:
            item = dict(hit)
            facts = dict(item.get("facts") or {}) if isinstance(item.get("facts"), dict) else {}
            ordinal = facts.get("pipe_ordinal") or (len(phases) + 1)
            facts["pipe_ordinal"] = ordinal
            item["facts"] = facts
            item["phase"] = str(ordinal)
            item["pipe"] = item.get("name")
            item["ok"] = True
            phases.append(item)
        first_facts = (
            (phases[0].get("facts") or {}) if phases and isinstance(phases[0].get("facts"), dict) else {}
        )
        scope = str(first_facts.get("scope") or "").strip()
        entry_hits = self._kernel_launch_entry(
            str(pattern or "").strip() or scope,
            scope=scope,
            winner_file=winner_file,
        )
        entry = entry_hits[0] if entry_hits else None
        coverage = _hits_coverage(
            [p for p in phases if p.get("ok")] + ([entry] if entry else []),
            total=group_total or sum(1 for p in phases if p.get("ok")),
            clipped=clipped,
        )
        coverage["kernel_phases"] = [
            str(p.get("facts", {}).get("pipe_ordinal") or p.get("phase") or "")
            for p in phases
            if p.get("ok")
        ]
        if other_files:
            coverage["other_kernel_files"] = other_files[:8]
        ok_n = sum(1 for p in phases if p.get("ok"))
        if clipped:
            coverage["completeness"] = "page_clipped"
            coverage["answerable"] = False
        elif ok_n >= 2:
            coverage["completeness"] = "siblings_checked"
            coverage["answerable"] = True
        elif ok_n == 1:
            coverage["completeness"] = "first_hit"
            coverage["answerable"] = True
        payload = {
            "ok": True,
            "mode": "kernel_launch",
            "pattern": str(pattern or "").strip(),
            "coverage": coverage,
            "phases": phases,
            "entry": entry,
            "count": ok_n,
            "files": _group_by_file(
                [p for p in phases if p.get("ok")] + ([entry] if entry else [])
            ),
        }
        if other_files:
            payload["other_kernels"] = other_files[:8]
        attach_query_hints(
            payload,
            pattern or "TPipe",
            count=int(payload["count"]),
            kinds=("PIPE",),
            mode="kernel_launch",
        )
        fitted = _fit_payload(payload)
        self._launch_cache[cache_key] = fitted
        return fitted

    def aggregate_gaps(self, pattern: str = "", *, limit: int = 50) -> dict[str, Any]:
        needle = str(pattern or "").strip().lower()
        rows = self.unresolved()
        total = len(rows)
        if needle:
            rows = [
                r
                for r in rows
                if needle in json.dumps(r, ensure_ascii=False, default=str).lower()
            ]
        rows = rows[: int(limit)]
        return _fit_payload(
            {
                "ok": True,
                "mode": "gaps",
                "pattern": needle,
                "gaps": rows,
                "count": len(rows),
                "total": total,
            }
        )

    def agent_query(
        self,
        *,
        pattern: str = "",
        file: str = "",
        line: int = 0,
        line_end: int = 0,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Agent-facing resolve helper: name card, dim cover, file:line, or index."""
        path = str(file or "").strip()
        line_n = int(line or 0)
        if path and line_n:
            span = int(line_end or line_n)
            if span > line_n:
                payload = self.query_around(path, line_n, line_end=span, limit=limit)
            else:
                payload = self.query_site_unit(path, line_n, limit=limit)
            from ascendc_codemap_mcp.engine.query.explore import attach_explore_fields

            return _fit_payload(attach_explore_fields(self, payload, pattern=f"{path}:{line_n}"))
        text = str(pattern or "").strip()
        if not text:
            return self.query_index(limit=limit)
        from ascendc_codemap_mcp.engine.query.explore import attach_explore_fields

        if "=" in text:
            payload = self.query_cover(text, limit=limit)
            return _fit_payload(attach_explore_fields(self, payload, pattern=text))
        payload = self.query_name_card(text, limit=limit)
        return _fit_payload(attach_explore_fields(self, payload, pattern=text))

    def _definition_spans(self, file: str) -> list[tuple[int, int]]:
        """Recorded FUNCTION/METHOD/KERNEL spans in one file, by start line."""
        key = str(file or "")
        hit = self._def_span_cache.get(key)
        if hit is not None:
            return hit
        kinds = tuple(_DEF_CARD_KINDS)
        ph = ",".join("?" for _ in kinds)
        leaf = key.replace("\\", "/").rsplit("/", 1)[-1]
        needle = key.replace("\\", "/")
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT line_start, line_end FROM entity
                WHERE kind IN ({ph})
                  AND IFNULL(line_start, 0) > 0
                  AND IFNULL(line_end, 0) > line_start
                  AND file = ?
                """,
                (*kinds, needle),
            ).fetchall()
            if not rows and leaf:
                rows = conn.execute(
                    f"""
                    SELECT line_start, line_end FROM entity
                    WHERE kind IN ({ph})
                      AND IFNULL(line_start, 0) > 0
                      AND IFNULL(line_end, 0) > line_start
                      AND IFNULL(file, '') LIKE '%/' || ?
                    """,
                    (*kinds, leaf),
                ).fetchall()
        spans = sorted({(int(a or 0), int(b or 0)) for a, b in rows})
        self._def_span_cache[key] = spans
        return spans

    def _clamp_definition_start(self, file: str, start: int, end: int) -> int:
        """Never let a definition's body begin inside its neighbour's.

        Extraction records `line_end` at the closing brace but starts some
        template member functions a few lines early, so a card would open with
        the tail of the function above it. Two recorded definitions cannot
        overlap, so the later one starts after the earlier one ends. The stored
        fact is left alone; this only bounds what gets read back.
        """
        lo, hi = int(start or 0), int(end or 0)
        if lo <= 0 or hi < lo:
            return lo
        floor = lo
        for other_start, other_end in self._definition_spans(file):
            if other_start >= lo or other_end < lo or other_end >= hi:
                continue
            floor = max(floor, other_end + 1)
        return min(floor, hi)

    def _definition_unit_snippet(self, hit: dict[str, Any]) -> tuple[str, bool]:
        """Whole body for a FUNCTION/METHOD/KERNEL card. ('', False) otherwise.

        `resolve(SomeFunction)` used to answer with the stored span snippet,
        which is a couple of lines, so the caller had to copy the file:line back
        in to read the body it had just asked for. Same projector as
        `resolve(file, line)`, so both spellings return the same source.
        """
        # A class body answers "how is this policy built" the same way a
        # function body answers "what does this compute"; both were clipped to
        # a few lines and forced a second call.
        if str(hit.get("kind") or "").upper() not in _UNIT_SNIPPET_KINDS:
            return "", False
        file = str(hit.get("file") or "")
        start = int(hit.get("line_start") or hit.get("line") or 0)
        end = int(hit.get("line_end") or 0)
        # A one-line span carries no body to rebuild; using it would replace the
        # stored snippet with the declaration line alone.
        if not file or start <= 0 or end <= start:
            return "", False
        start = self._clamp_definition_start(file, start, end)
        stop = min(end, start + _SITE_UNIT_HARD - 1)
        with self._connect() as conn:
            rows = _source_line_rows(conn, file, start, stop)
        if not rows:
            return "", False
        rows = _restore_blank_lines(rows, stop)
        return "\n".join(f"{ln}:{txt}" for ln, txt in rows), stop < end

    def query_name_card(self, pattern: str, *, limit: int = 8) -> dict[str, Any]:
        needle = str(pattern or "").strip()
        if not needle:
            payload = {"ok": False, "shape": "name", "pattern": needle, "cards": [], "count": 0}
            attach_query_hints(payload, needle, count=0, mode="name")
            return _fit_payload(payload)
        hits = self._exact_name_hits(needle, limit=max(int(limit) * 16, 64))
        kind_alias = catalog_kind_alias(needle)
        if not hits and kind_alias:
            hits = self._hits_by_kind(kind_alias, limit=max(int(limit) * 16, 64))
        if not hits:
            hits = self._prefix_name_hits(needle, limit=max(int(limit) * 16, 64))
        if not hits:
            located = self.aggregate_locate(needle, limit=max(int(limit), 8))
            hits = list(located.get("locations") or [])
        recall_samples: list[dict[str, Any]] = []
        recall_total = 0
        if not hits:
            hits, recall_samples, recall_total = self._recall_name_hits(
                needle, limit=max(int(limit), 8)
            )
        by_kind: dict[str, dict[str, Any]] = {}
        # An include guard only answers a query that spelled it out in full.
        if not is_include_guard("", needle):
            filtered = [
                hit
                for hit in hits
                if not is_include_guard(
                    str(hit.get("kind") or ""), str(hit.get("name") or ""), hit.get("data")
                )
            ]
            if filtered or not hits:
                hits = filtered
        ranked = sorted(
            hits,
            key=lambda hit: _agent_sort_key(hit, needle, architecture=self._architecture),
        )
        definition_sites, sites_complete = _definition_sites_from_hits(ranked, needle=needle)
        for hit in ranked:
            kind = str(hit.get("kind") or "")
            if kind and kind not in by_kind:
                by_kind[kind] = hit
        if (
            EntityKind.TILING_FIELD.value in by_kind
            and EntityKind.FIELD.value in by_kind
            and _last_ident(str(by_kind[EntityKind.TILING_FIELD.value].get("name") or "")).lower()
            == _last_ident(str(by_kind[EntityKind.FIELD.value].get("name") or "")).lower()
        ):
            by_kind.pop(EntityKind.FIELD.value, None)
        if (
            EntityKind.TILING_KEY.value in by_kind
            and EntityKind.TEMPLATE_ARG.value in by_kind
            and _last_ident(str(by_kind[EntityKind.TILING_KEY.value].get("name") or "")).lower()
            == _last_ident(str(by_kind[EntityKind.TEMPLATE_ARG.value].get("name") or "")).lower()
        ):
            by_kind.pop(EntityKind.TEMPLATE_ARG.value, None)
        key_ident = _last_ident(
            str((by_kind.get(EntityKind.TILING_KEY.value) or {}).get("name") or "")
        ).lower()
        if key_ident:
            for weaker in (
                EntityKind.FIELD.value,
                EntityKind.PREDICATE.value,
                EntityKind.VARIABLE.value,
            ):
                other = by_kind.get(weaker)
                if other and _last_ident(str(other.get("name") or "")).lower() == key_ident:
                    by_kind.pop(weaker, None)
        # Occupancy ranking already prefers located identities. Keep fileless
        # FUNCTION/METHOD API symbols when they carry CALLS / ROOTED_AT.
        ranked_kinds = list(by_kind.values())
        cards_src = ranked_kinds[:MAX_NAME_CARDS]
        primary_hits = cards_src[:1]
        extra_hits = cards_src[1:]
        field_kinds = {
            EntityKind.TILING_FIELD.value,
            EntityKind.FIELD.value,
            EntityKind.TILING_KEY.value,
        }
        field_names = [
            str(hit.get("name") or "")
            for hit in cards_src
            if str(hit.get("kind") or "") in field_kinds
        ]
        if len(field_names) > 1:
            self._prefetch_field_graph(field_names)
        cards: list[dict[str, Any]] = []
        next_names: list[str] = []
        seen_next: set[str] = set()
        self_names = {needle.lower()}
        for hit in primary_hits:
            kind = str(hit.get("kind") or "")
            eid = str(hit.get("id") or "")
            name = str(hit.get("name") or needle)
            self_names.add(name.lower())
            grouped = self._grouped_edges(eid, entity_kind=kind)
            if kind == EntityKind.TILING_KEY.value:
                derives = grouped.get("DERIVES") if isinstance(grouped.get("DERIVES"), dict) else None
                if derives:
                    neigh = list(derives.get("neighbors") or [])
                    if len(neigh) > DERIVES_PER_TILING_KEY:
                        for row in neigh[DERIVES_PER_TILING_KEY:]:
                            _append_next_name(
                                next_names,
                                seen_next,
                                self_names,
                                str(row.get("name") or ""),
                                limit=NEXT_HOP_LIMIT,
                            )
                        derives["neighbors"] = neigh[:DERIVES_PER_TILING_KEY]
                        derives["truncated"] = True
                keep = {"WRITES", "READS", "DERIVES"}
                grouped = {
                    rel: (
                        bucket
                        if rel in keep
                        else {"count": int(bucket.get("count") or 0)}
                    )
                    for rel, bucket in grouped.items()
                    if isinstance(bucket, dict)
                }
            snippet, snip_cut = _card_snippet_for_hit(hit)
            unit, unit_cut = self._definition_unit_snippet(hit)
            # Some declarations are recorded with a one-line span; rebuilding
            # from it would replace a good stored snippet with a worse one.
            if unit and unit.count("\n") >= snippet.count("\n"):
                snippet, snip_cut = unit, unit_cut
            card: dict[str, Any] = {
                "kind": kind,
                "name": name,
                "id": eid,
                "file": hit.get("file") or "",
                "line": int(hit.get("line_start") or 0),
                # Carried so the renderer can tell a full body from a window.
                "line_end": int(hit.get("line_end") or 0),
                "snippet": snippet,
                "edges": grouped,
            }
            if snip_cut:
                card["truncated"] = True
            omitted = hit.get("omitted")
            if omitted:
                card["omitted"] = omitted
            extras = self._card_extras(hit)
            if kind == EntityKind.TILING_KEY.value:
                extras["cover_followup"] = f"Dim={_last_ident(name)}"
            if len(definition_sites) > 1:
                extras["definition_sites"] = definition_sites
                extras["definition_sites_complete"] = sites_complete
            if extras.get("readers") and definition_sites:
                extras["readers"] = _prefer_statement_readers(
                    list(extras["readers"]),
                    definition_sites,
                    _last_ident(name),
                )
            line_end = int(hit.get("line_end") or hit.get("line_start") or 0)
            card["definition_span"] = {
                "file": hit.get("file") or "",
                "line_start": int(hit.get("line_start") or 0),
                "line_end": line_end,
            }
            returns = grouped.get("RETURNS") if isinstance(grouped.get("RETURNS"), dict) else {}
            return_hit = next(
                (
                    row
                    for row in list(returns.get("neighbors") or [])
                    if str(row.get("file") or "").strip() and int(row.get("line") or 0) > 0
                ),
                None,
            )
            if return_hit and not card["definition_span"]["file"]:
                card["file"] = str(return_hit.get("file") or "")
                card["line"] = int(return_hit.get("line") or 0)
                card["definition_span"] = {
                    "file": str(return_hit.get("file") or ""),
                    "line_start": int(return_hit.get("line") or 0),
                    "line_end": int(return_hit.get("line") or 0),
                }
                extras.setdefault("writers", []).insert(
                    0,
                    {
                        "name": str(return_hit.get("name") or ""),
                        "file": str(return_hit.get("file") or ""),
                        "line": int(return_hit.get("line") or 0),
                    },
                )
            for key in ("catalog", "role", "spelling", "wrapper", "tposition", "pipe_ordinal", "cpp_kind"):
                if extras.get(key) not in (None, "", []):
                    card[key] = extras[key]
            if hit.get("truncated") or extras.get("truncated"):
                card["truncated"] = True
            writers = extras.get("writers") if isinstance(extras.get("writers"), list) else []
            readers = extras.get("readers") if isinstance(extras.get("readers"), list) else []
            if writers and not (grouped.get("WRITES") or {}).get("neighbors"):
                grouped.setdefault("WRITES", {"count": len(writers), "neighbors": []})
                grouped["WRITES"]["neighbors"] = [
                    {
                        "name": str(row.get("name") or ""),
                        "kind": str(row.get("kind") or EntityKind.FUNCTION.value),
                        "file": str(row.get("file") or ""),
                        "line": int(row.get("line") or 0),
                    }
                    for row in writers[:EDGES_PER_KIND]
                ]
                grouped["WRITES"]["count"] = max(int(grouped["WRITES"].get("count") or 0), len(writers))
            if readers and not (grouped.get("READS") or {}).get("neighbors"):
                grouped.setdefault("READS", {"count": len(readers), "neighbors": []})
                grouped["READS"]["neighbors"] = [
                    {
                        "name": str(row.get("name") or ""),
                        "kind": str(row.get("kind") or "METHOD"),
                        "file": str(row.get("file") or ""),
                        "line": int(row.get("line") or 0),
                    }
                    for row in readers[:EDGES_PER_KIND]
                ]
                grouped["READS"]["count"] = max(int(grouped["READS"].get("count") or 0), len(readers))
            focus = _focus_value_write(extras)
            if focus:
                window = ""
                with self._connect() as conn:
                    window = _source_line_window(
                        conn,
                        str(focus.get("file") or ""),
                        int(focus.get("line") or 0),
                    )
                if window:
                    card["snippet"] = window
                    card["file"] = str(focus.get("file") or card.get("file") or "")
                    card["line"] = int(focus.get("line") or 0)
                    if _snippet_covers_line(window, int(focus.get("line") or 0)):
                        card.pop("truncated", None)
                grouped = _overlay_value_write(grouped, focus, needle)
                for ident in _focus_idents(str(focus.get("rhs") or ""), needle):
                    if _append_next_name(next_names, seen_next, self_names, ident):
                        break
            card["edges"] = grouped
            if extras:
                card["extras"] = extras
            covered = self._fill_semantic_card(card, extras, needle)
            self_names.update(covered)
            card["canonical"] = extras.get("canonical") or name
            support = self.compiled_support_for(name, extras) or self.compiled_support_for(
                needle, extras
            )
            if support:
                facets = card.get("facets") if isinstance(card.get("facets"), dict) else {}
                facets["compiled_support"] = support
                card["facets"] = facets
            cards.append(card)
            if kind in {EntityKind.COMPILE_VAR.value, EntityKind.MACRO.value}:
                facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
                expr = str(
                    facts.get("value_expr")
                    or facts.get("value")
                    or facts.get("definition")
                    or extras.get("value_expr")
                    or extras.get("value")
                    or extras.get("definition")
                    or ""
                )
                _extend_next_from_value_expr(
                    next_names, seen_next, self_names, expr, limit=NEXT_HOP_LIMIT
                )
            _extend_next_from_edges(
                next_names, seen_next, self_names, grouped, limit=NEXT_HOP_LIMIT
            )
        for hit in extra_hits:
            extra_kind = str(hit.get("kind") or "")
            extra_grouped = self._grouped_edges(str(hit.get("id") or ""), entity_kind=extra_kind)
            _extend_next_from_edges(
                next_names, seen_next, self_names, extra_grouped, limit=NEXT_HOP_LIMIT
            )
            cards.append(_compact_kind_card(hit))
        self._named_fields_cache = None
        self._edges_cache = None
        coverage = _hits_coverage(cards_src, total=len(hits), needle=needle)
        if definition_sites:
            # Count declarations, not rows: a FIELD and the BUFFER minted from
            # it share one line, and the list rendered under this count folds
            # them together. Counting rows made the count exceed the list.
            distinct_sites = {
                (str(s.get("file") or ""), int(s.get("line") or 0))
                for s in definition_sites
                if isinstance(s, dict) and s.get("file")
            }
            coverage["definition_sites_count"] = max(
                int(coverage.get("definition_sites_count") or 0),
                len(distinct_sites) or len(definition_sites),
            )
            coverage["definition_sites_complete"] = sites_complete
            if not sites_complete:
                coverage["completeness"] = "page_clipped"
                primary = cards[0] if cards else {}
                coverage["answerable"] = bool(
                    str(primary.get("file") or "").strip()
                    and int(primary.get("line") or 0) > 0
                )
            elif len(definition_sites) > 1:
                coverage["completeness"] = "siblings_checked"
                coverage["answerable"] = True
        payload = {
            "ok": bool(cards),
            "shape": "name",
            "pattern": needle,
            "cards": cards,
            "next": next_names,
            "count": len(cards),
            "coverage": _name_card_coverage(coverage),
        }
        matched = _matched_entity_rows(ranked)
        if len(matched) > 1:
            payload["matched_entities"] = matched
        primary = cards[0] if cards else {}
        primary_facets = primary.get("facets") if isinstance(primary.get("facets"), dict) else {}
        support = primary_facets.get("compiled_support")
        if not support:
            extras = primary.get("extras") if isinstance(primary.get("extras"), dict) else {}
            support = self.compiled_support_for(needle, extras)
        if support:
            payload["compiled_support"] = support
        if not cards:
            payload["dim_names"] = self._compiled_dim_names()
        if recall_total:
            payload["match"] = "recall_scan"
            payload["text_hits"] = recall_samples
            payload["text_hits_total"] = recall_total
            payload["text_hits_complete"] = len(recall_samples) >= recall_total
            payload["match_note"] = (
                "No graph identifier matched. These are source-text matches, "
                f"{len(recall_samples)} shown of {recall_total} total. Confirm the "
                "name at the cited line before citing it."
            )
        missed_owner = str((ranked[0] if ranked else {}).get("owner_mismatch") or "")
        if missed_owner:
            owners = sorted(
                {
                    self.owner_of(
                        str(hit.get("file") or ""),
                        int(hit.get("line_start") or 0),
                        hit.get("data"),
                    )
                    for hit in ranked
                }
                - {""}
            )
            payload["match_note"] = (
                f"No member of `{missed_owner}` is named `{_last_ident(needle)}`. "
                + (
                    f"Showing the same name under {', '.join(owners)} instead."
                    if owners
                    else "Showing same-named entities under other owners instead."
                )
            )
        if not payload.get("match_note"):
            payload["match_note"] = self._ambiguity_note(needle, ranked, primary)
        if not payload.get("match_note"):
            payload["match_note"] = _inexact_name_note(needle, ranked)
        attach_query_hints(payload, needle, count=len(cards), mode="name")
        return _fit_payload(payload)

    def _scope_of(self, hit: dict[str, Any]) -> str:
        """The class that declares this, or the function that does.

        `owner_of` answers about classes, and a register declared inside a VF
        function has none. That left the names with the most duplicates -- 35
        `vregSrc`, 42 `pregFullExe`, one per function -- looking unambiguous.
        """
        file = str(hit.get("file") or "")
        line = int(hit.get("line") or hit.get("line_start") or 0)
        owner = self.owner_of(file, line, hit.get("data"))
        if owner:
            return owner
        enclosing = self._enclosing_def(file, line)
        return _last_ident(str((enclosing or {}).get("name") or "").replace(".", "::"))

    def _ambiguity_note(
        self, needle: str, ranked: list[dict[str, Any]], primary: dict[str, Any]
    ) -> str:
        """Say when a bare name belongs to more than one scope.

        `ping_` is declared by four buffer policies. The card picked one and
        named its scope, which reads as the answer rather than as one of four,
        so agents cited the wrong class or spent calls discovering the rest.
        """
        want = str(needle or "").strip()
        if not want or "::" in want or "." in want:
            return ""
        # Only things that live inside a scope can be ambiguous by scope. A class
        # is not declared in itself, and a tiling field's identity spans layers
        # rather than owners, so both produced notes that named no real choice.
        if str(primary.get("kind") or "") not in _SCOPED_MEMBER_KINDS:
            return ""
        leaf = _last_ident(want).lower()
        owners: list[str] = []
        for hit in ranked:
            if _last_ident(str(hit.get("name") or "")).lower() != leaf:
                continue
            own = self._scope_of(hit)
            if own and own not in owners:
                owners.append(own)
        if len(owners) < 2:
            return ""
        chosen = self._scope_of(primary)
        shown = ", ".join(owners[:6]) + (" …" if len(owners) > 6 else "")
        note = f"`{_last_ident(want)}` is declared in {len(owners)} scopes: {shown}."
        if chosen:
            note += f" This card is {chosen}'s."
        return note + f" Resolve `Owner::{_last_ident(want)}` for a different one."

    def _is_ascendc_catalog_ident(self, needle: str) -> bool:
        ident = _last_ident(str(needle or "").replace(".", "::")).lower()
        if not ident:
            return False
        with self._connect() as conn:
            if self._accel_ready(conn):
                row = conn.execute(
                    "SELECT 1 FROM entity_name_leaf "
                    "WHERE leaf = ? AND is_ascendc = 1 LIMIT 1",
                    (ident,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT 1 FROM entity e
                    WHERE IFNULL(json_extract(e.data, '$.catalog'), '') = 'ascendc'
                      AND (
                        e.name COLLATE NOCASE = ?
                        OR lower(IFNULL(json_extract(e.data, '$.spelling'), '')) = ?
                        OR e.name COLLATE NOCASE LIKE ('%::' || ?)
                      )
                    LIMIT 1
                    """,
                    (ident, ident, ident),
                ).fetchone()
        return row is not None

    def _exact_name_hits(self, needle: str, *, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if self._accel_ready(conn):
                where, params = _leaf_name_where_indexed(needle)
            else:
                where, params = _leaf_name_where(needle)
            rows = self._select_entities(
                conn,
                extra_where=where,
                params=params,
                limit=max(int(limit), 8),
                order=(
                    "(CASE WHEN IFNULL(e.file,'') = '' OR IFNULL(e.line_start,0) = 0 "
                    "THEN 1 ELSE 0 END), e.kind, e.name, e.id"
                ),
            )
            hits = self._hits_from_rows(conn, rows, why="name_card", with_snippet=True)
        return self._rank_by_owner(hits, needle)

    def _rank_by_owner(
        self, hits: list[dict[str, Any]], needle: str
    ) -> list[dict[str, Any]]:
        """Honour the `Owner::` a caller wrote instead of dropping it.

        Members are stored under their leaf name, so `A::x` and `B::x` are the
        same row to a leaf lookup and the winner is whichever sorts first. A
        caller who took the trouble to qualify gets that qualifier applied, and
        a qualifier that matches nothing is reported rather than ignored.
        """
        want = str(needle or "").strip().replace(".", "::")
        if "::" not in want or not hits:
            return hits
        owner = _last_ident(want.rsplit("::", 1)[0]).lower()
        if not owner:
            return hits
        matched: list[dict[str, Any]] = []
        others: list[dict[str, Any]] = []
        for hit in hits:
            name = str(hit.get("name") or "").replace(".", "::")
            found = _last_ident(name.rsplit("::", 1)[0]).lower() if "::" in name else ""
            if not found:
                # A register declared in a VF function has no class, so a
                # class-only lookup answered `ReduceSinkVF::vregSrc` with some
                # other function's register and called the qualifier wrong.
                found = self._scope_of(hit).lower()
            (matched if found == owner else others).append(hit)
        if not matched:
            for hit in others:
                hit["owner_mismatch"] = want.rsplit("::", 1)[0]
            return others
        for hit in others:
            hit["owner_rank"] = 1
        return matched + others

    def _hits_by_kind(self, kind: str, *, limit: int) -> list[dict[str, Any]]:
        want = str(kind or "").strip()
        if not want:
            return []
        with self._connect() as conn:
            rows = self._select_entities(
                conn,
                kinds=[want],
                extra_where=_ASCENDC_CATALOG_SQL,
                limit=max(int(limit), 8),
            )
            return self._hits_from_rows(conn, rows, why="kind_alias", with_snippet=True)

    def _prefix_name_hits(self, needle: str, *, limit: int) -> list[dict[str, Any]]:
        ident = str(needle or "").strip()
        if not _IDENT_NAME_RE.match(ident):
            return []
        with self._connect() as conn:
            if self._accel_ready(conn):
                low = ident.lower()
                extra_where = (
                    "e.id IN (SELECT entity_id FROM entity_name_leaf "
                    "WHERE leaf >= ? AND leaf < ? AND is_ascendc = 0)"
                )
                params: tuple[Any, ...] = (low, low + "\uffff")
            else:
                extra_where = f"{_ASCENDC_CATALOG_SQL} AND e.name COLLATE NOCASE LIKE ?"
                params = (ident.lower() + "%",)
            rows = self._select_entities(
                conn,
                extra_where=extra_where,
                params=params,
                limit=max(int(limit), 8),
            )
            return self._hits_from_rows(conn, rows, why="name_card", with_snippet=True)

    def _recall_name_hits(
        self, needle: str, *, limit: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Last-resort recall over indexed source text.

        Name lookup needs the identifier up front. This scans `source_line` the
        way grep would, so a caller can still land on a site it could only
        describe by a fragment. Returns (entity hits, text hit samples, exact
        total) — the total is what lets a caller say "all N sites" instead of
        quoting a page.
        """
        text = str(needle or "").strip()
        if len(text) < 3:
            return [], [], 0
        with self._connect() as conn:
            from ascendc_codemap_mcp.engine.store.accel import has_source_fts, has_source_line

            samples: list[dict[str, Any]] = []
            total = 0
            sample_rows: list[Any] | None = None
            fts_q = _fts_match_query(text)
            if has_source_fts(conn) and fts_q:
                # `source_fts` is an external-content index over `source_line`
                # and stores no columns of its own, so path and line come back
                # through the rowid join. A quoted trigram MATCH asks the same
                # substring question as the LIKE branch below, from an index.
                try:
                    cap = max(int(limit), 8)
                    sample_rows = conn.execute(
                        """
                        SELECT sl.path, sl.line, sl.text
                        FROM source_fts f JOIN source_line sl ON sl.id = f.rowid
                        WHERE f.source_fts MATCH ?
                        ORDER BY sl.path, sl.line LIMIT ?
                        """,
                        (fts_q, cap),
                    ).fetchall()
                    total = len(sample_rows)
                    if total == cap:
                        total = cap + 1
                except sqlite3.OperationalError:
                    # A rejected MATCH expression or a half-built index must not
                    # read as "no such text" -- fall through to the scan, which
                    # needs no index and answers identically.
                    total = 0
                    sample_rows = None
            if sample_rows is None and has_source_line(conn):
                # `LIKE '%needle%'` cannot use an index, so this is a full scan of
                # `source_line`. Counting and sampling in two statements scanned it
                # twice for one answer -- 7-12ms each on a 40k-row table, and this
                # path is what the slowest name cards spend their time in. One scan
                # returns both: the count is exact because it *is* the row set, and
                # the sample is the same rows the LIMIT would have kept. The whole
                # table is ~5MB, which bounds the worst case a broad needle can pull
                # into memory.
                matched = conn.execute(
                    """
                    SELECT path, line, text FROM source_line
                    WHERE text LIKE '%' || ? || '%'
                    ORDER BY path, line
                    """,
                    (text,),
                ).fetchall()
                total = len(matched)
                sample_rows = matched[: max(int(limit), 8)]
            if not total or sample_rows is None:
                return [], [], 0
            samples = [
                {
                    "file": str(r[0] or ""),
                    "line": int(r[1] or 0),
                    "text": str(r[2] or "").strip()[:200],
                }
                for r in sample_rows
            ]
            hits: list[dict[str, Any]] = []
            for row in sample_rows[:PRIMARY_CANDIDATES]:
                ent_rows = self._select_entities(
                    conn,
                    extra_where=(
                        "(IFNULL(e.file,'') = ? OR IFNULL(e.file,'') LIKE '%' || ?) "
                        "AND IFNULL(e.line_start,0) <= ? "
                        "AND IFNULL(e.line_end, e.line_start) >= ?"
                    ),
                    params=(row[0], row[0], int(row[1] or 0), int(row[1] or 0)),
                    limit=4,
                )
                hits.extend(
                    self._hits_from_rows(
                        conn, ent_rows, why="recall_scan", with_snippet=True
                    )
                )
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for hit in hits:
            eid = str(hit.get("id") or "")
            if eid and eid in seen:
                continue
            seen.add(eid)
            hit["recall_scan"] = True
            unique.append(hit)
        return unique, samples, total

    def _grouped_edges(self, entity_id: str, *, entity_kind: str = "") -> dict[str, Any]:
        if not entity_id:
            return {}
        key = (str(entity_id), str(entity_kind or ""))
        cached = self._grouped_edges_cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
        grouped_out = self._grouped_edges_uncached(entity_id, entity_kind=entity_kind)
        if len(self._grouped_edges_cache) < 512:
            self._grouped_edges_cache[key] = copy.deepcopy(grouped_out)
        return grouped_out

    def _grouped_edges_uncached(
        self, entity_id: str, *, entity_kind: str = ""
    ) -> dict[str, Any]:
        placeholders = ",".join("?" for _ in CARD_EDGE_KINDS)
        neighbor_placeholders = ",".join("?" for _ in CARD_NEIGHBOR_RELS)
        skip_tpl = str(entity_kind or "").upper() == EntityKind.TILING_KEY.value
        grouped: dict[str, dict[str, Any]] = {}
        seed_file = ""
        with self._connect() as conn:
            seed_row = conn.execute(
                "SELECT file FROM entity WHERE id = ? LIMIT 1", (entity_id,)
            ).fetchone()
            if seed_row is not None:
                seed_file = str(seed_row[0] or "")
            counts = {
                str(row[0] or ""): int(row[1] or 0)
                for row in conn.execute(
                    f"""
                    SELECT kind, COUNT(*) FROM relation
                    WHERE (src = ? OR dst = ?) AND kind IN ({placeholders})
                    GROUP BY kind
                    """,
                    (entity_id, entity_id, *CARD_EDGE_KINDS),
                )
            }
            try:
                rows = conn.execute(
                    f"""
                    SELECT rel_kind, rel_data, other_kind, other_name, other_file, other_line
                    FROM (
                        SELECT r.kind AS rel_kind, r.data AS rel_data,
                               e.kind AS other_kind, e.name AS other_name,
                               e.file AS other_file, e.line_start AS other_line,
                               ROW_NUMBER() OVER (
                                   PARTITION BY r.kind ORDER BY
                                     CASE r.kind WHEN 'DECLARES' THEN {_DECLARES_KIND_SQL} ELSE 0 END,
                                     e.kind, e.name
                               ) AS rn
                        FROM relation r
                        JOIN entity e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
                        WHERE (r.src = ? OR r.dst = ?) AND r.kind IN ({neighbor_placeholders})
                    ) ranked
                    WHERE rn <= ?
                    """,
                    (entity_id, entity_id, entity_id, *CARD_NEIGHBOR_RELS, EDGES_PER_KIND * 8),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
                for kind in CARD_NEIGHBOR_RELS:
                    rows.extend(
                        conn.execute(
                            f"""
                            SELECT r.kind AS rel_kind, r.data AS rel_data,
                                   e.kind AS other_kind, e.name AS other_name,
                                   e.file AS other_file, e.line_start AS other_line
                            FROM relation r
                            JOIN entity e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
                            WHERE (r.src = ? OR r.dst = ?) AND r.kind = ?
                            ORDER BY CASE r.kind WHEN 'DECLARES' THEN {_DECLARES_KIND_SQL} ELSE 0 END, e.kind, e.name
                            LIMIT ?
                            """,
                            (entity_id, entity_id, entity_id, kind, EDGES_PER_KIND * 8),
                        ).fetchall()
                    )
        for rel_kind, n in counts.items():
            if rel_kind:
                grouped[rel_kind] = {"count": n, "neighbors": []}
        pending: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for row in rows:
            if _is_advisory_data(_row_get(row, "rel_data")):
                continue
            rel_kind = str(_row_get(row, "rel_kind") or "")
            rel_data = _parse_data(_row_get(row, "rel_data"))
            if rel_kind == "ALIASES" and str(rel_data.get("provenance") or "") == "source_compile_occupancy":
                continue
            other_kind = str(_row_get(row, "other_kind") or "")
            other_name = str(_row_get(row, "other_name") or "")
            other_file = _norm_file(str(_row_get(row, "other_file") or ""))
            if skip_tpl and rel_kind == "BINDS" and (
                other_kind == EntityKind.TEMPLATE.value or other_name.startswith("ARGS_SEL")
            ):
                continue
            if _skip_composition_neighbor(entity_kind, rel_kind, other_kind):
                continue
            grouped.setdefault(rel_kind, {"count": counts.get(rel_kind, 0), "neighbors": []})
            rank = _edge_evidence_rank(
                rel_kind,
                _row_get(row, "rel_data"),
                other_kind,
                other_file,
                seed_file,
            )
            pending.setdefault(rel_kind, []).append(
                (
                    rank,
                    {
                        "name": other_name,
                        "kind": other_kind,
                        "file": other_file,
                        "line": int(_row_get(row, "other_line") or 0),
                    },
                )
            )
        for rel_kind, scored in pending.items():
            kind_cap = (
                DERIVES_PER_TILING_KEY
                if skip_tpl and rel_kind == "DERIVES"
                else EDGES_PER_KIND
            )
            scored.sort(key=lambda item: (item[0], str(item[1].get("name") or "")))
            neighbors: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for _rank, neigh in scored:
                key = (str(neigh.get("kind") or ""), str(neigh.get("name") or ""))
                if key in seen:
                    continue
                seen.add(key)
                neighbors.append(neigh)
                if len(neighbors) >= kind_cap:
                    break
            bucket = grouped.setdefault(rel_kind, {"count": counts.get(rel_kind, 0), "neighbors": []})
            bucket["neighbors"] = neighbors
        if entity_id:
            callees = self._call_direction_neighbors(entity_id, outgoing=True)
            if callees or "CALLS" in grouped:
                grouped["CALLS"] = {
                    "count": len(callees),
                    "neighbors": callees,
                }
        return grouped

    def _call_direction_neighbors(self, entity_id: str, *, outgoing: bool) -> list[dict[str, Any]]:
        if not entity_id:
            return []
        sql = """
            SELECT e.kind AS other_kind, e.name AS other_name,
                   e.file AS other_file, e.line_start AS other_line
            FROM relation r
            JOIN entity e ON e.id = {other}
            WHERE r.kind = 'CALLS' AND {mine} = ?
            ORDER BY e.kind, e.name
            LIMIT ?
        """.format(
            other="r.dst" if outgoing else "r.src",
            mine="r.src" if outgoing else "r.dst",
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (entity_id, EDGES_PER_KIND)).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "name": str(_row_get(row, "other_name") or ""),
                    "kind": str(_row_get(row, "other_kind") or ""),
                    "file": _norm_file(str(_row_get(row, "other_file") or "")),
                    "line": int(_row_get(row, "other_line") or 0),
                }
            )
        return out

    def _write_timeline(self, needle: str, *, primary: dict[str, Any]) -> dict[str, Any]:
        ident = _last_ident(str(needle or "").replace(".", "::"))
        if not ident:
            return {}
        aliases = self._named_fields(
            ident,
            kinds=(
                EntityKind.TILING_FIELD.value,
                EntityKind.FIELD.value,
                EntityKind.TILING_KEY.value,
            ),
        )
        value_rows: list[dict[str, Any]] = []
        packing_rows: list[dict[str, Any]] = []
        seen_value: set[tuple[str, int]] = set()
        seen_pack: set[tuple[str, int]] = set()

        def _add(bucket: list[dict[str, Any]], seen: set[tuple[str, int]], row: dict[str, Any], role: str) -> None:
            item = _timeline_site(row, role=role)
            if item is None:
                return
            loc = (str(item["file"]), int(item["line"]))
            if loc in seen:
                return
            seen.add(loc)
            bucket.append(item)

        for hit in aliases:
            facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
            for site in list(facts.get("write_sites") or []):
                if isinstance(site, dict):
                    _add(value_rows, seen_value, site, "value")
            for site in list(facts.get("packing_value_sites") or []):
                if not isinstance(site, dict):
                    continue
                kind = str(site.get("kind") or "")
                if kind in {"assignment", "declaration"}:
                    _add(value_rows, seen_value, site, "value")
                else:
                    _add(packing_rows, seen_pack, site, "packing")
            if str(hit.get("kind") or "") != EntityKind.TILING_KEY.value:
                continue
            impact = self.field_impact(str(hit.get("id") or ident))
            if not impact.get("ok"):
                continue
            for writer in list(impact.get("writers") or []):
                row = {
                    "file": writer.get("file"),
                    "line": writer.get("line_start") or writer.get("line"),
                    "name": writer.get("name"),
                    "kind": writer.get("kind"),
                    "rhs": (writer.get("facts") or {}).get("rhs")
                    if isinstance(writer.get("facts"), dict)
                    else "",
                }
                if _site_loc(row) in seen_value:
                    continue
                _add(packing_rows, seen_pack, row, "packing")
        value_rows.sort(key=lambda row: (str(row.get("file") or ""), int(row.get("line") or 0)))
        packing_rows.sort(key=lambda row: (str(row.get("file") or ""), int(row.get("line") or 0)))
        if not value_rows and not packing_rows:
            return {}
        out: dict[str, Any] = {
            "write_sites_complete": len(value_rows) <= MAX_WRITE_TIMELINE
            and len(packing_rows) <= MAX_WRITE_TIMELINE,
        }
        if value_rows:
            out["value_writes"] = value_rows[:MAX_WRITE_TIMELINE]
        if packing_rows:
            out["packing_writes"] = packing_rows[:MAX_WRITE_TIMELINE]
        return out

    def _same_value_neighbors(self, hit: dict[str, Any]) -> list[dict[str, Any]]:
        facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
        expr = str(facts.get("value_expr") or facts.get("value") or "").strip()
        if not expr:
            return []
        self_id = str(hit.get("id") or "")
        self_name = str(hit.get("name") or "")
        grouped = self._grouped_edges(self_id, entity_kind=str(hit.get("kind") or ""))
        aliases = grouped.get("ALIASES") if isinstance(grouped.get("ALIASES"), dict) else {}
        from_graph: list[dict[str, Any]] = []
        for row in list(aliases.get("neighbors") or []):
            name = str(row.get("name") or "")
            if not name or name == self_name:
                continue
            from_graph.append(
                {
                    "name": name,
                    "file": str(row.get("file") or ""),
                    "line": int(row.get("line") or 0),
                }
            )
            if len(from_graph) >= MAX_SAME_VALUE:
                return from_graph
        return from_graph

    def _fill_semantic_card(
        self, card: dict[str, Any], extras: dict[str, Any], needle: str
    ) -> set[str]:
        """Fold writers/readers into a self-contained card. Returns names already covered."""
        writers = extras.get("writers") if isinstance(extras.get("writers"), list) else []
        readers = extras.get("readers") if isinstance(extras.get("readers"), list) else []
        guards = extras.get("guards") if isinstance(extras.get("guards"), list) else []

        def _sites(rows: Any, cap: int = 8) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                item = {
                    "name": str(row.get("name") or ""),
                    "kind": str(row.get("kind") or ""),
                    "file": str(row.get("file") or ""),
                    "line": int(row.get("line") or row.get("line_start") or 0),
                }
                out.append({k: v for k, v in item.items() if v not in (None, "")})
                if len(out) >= cap:
                    break
            return out

        card["definition"] = {
            "file": card.get("file") or "",
            "line": int(card.get("line") or 0),
            **(
                card.get("definition_span")
                if isinstance(card.get("definition_span"), dict)
                else {}
            ),
        }
        card["host"] = {"writers": _sites(writers), "guards": _sites(guards, 6)}
        card["kernel"] = {"readers": _sites(readers)}
        flow: dict[str, Any] = {}
        canonical = extras.get("canonical") or card.get("canonical") or card.get("name")
        if canonical:
            flow["tiling_field"] = canonical
        consumers = [str(row.get("name") or "") for row in readers[:8] if row.get("name")]
        if consumers:
            flow["kernel_consumers"] = consumers
        writers_named = [str(row.get("name") or "") for row in writers[:6] if row.get("name")]
        if writers_named:
            flow["host_writers"] = writers_named
        if flow:
            card["flow"] = flow
        related: list[str] = []
        for name in writers_named + consumers:
            if name and name.lower() != str(needle or "").lower() and name not in related:
                related.append(name)
        if related:
            card["related"] = related[:8]
        covered = {str(needle or "").lower()}
        covered.update(n.lower() for n in related)
        covered.update(n.lower() for n in writers_named)
        covered.update(n.lower() for n in consumers)
        return {n for n in covered if n}

    def _card_extras(self, hit: dict[str, Any]) -> dict[str, Any]:
        kind = str(hit.get("kind") or "")
        name = str(hit.get("name") or "")
        extras: dict[str, Any] = {}
        eid = str(hit.get("id") or "")
        facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
        if kind in {EntityKind.METHOD.value, EntityKind.FUNCTION.value, EntityKind.KERNEL.value}:
            extras["callers"] = self._call_direction_neighbors(eid, outgoing=False)
            extras["callees"] = self._call_direction_neighbors(eid, outgoing=True)
        write_sites = facts.get("write_sites")
        field_kinds = {
            EntityKind.TILING_FIELD.value,
            EntityKind.FIELD.value,
            EntityKind.TILING_KEY.value,
        }
        if kind not in field_kinds and isinstance(write_sites, list) and write_sites:
            extras["write_sites"] = write_sites[:4]
        if hit.get("truncated"):
            extras["truncated"] = True
        if kind in field_kinds:
            field = self.field_impact(eid or name)
            if field.get("ok"):
                extras["writers"] = [
                    {
                        "name": row.get("name"),
                        "kind": row.get("kind"),
                        "file": row.get("file"),
                        "line": row.get("line_start"),
                    }
                    for row in list(field.get("writers") or [])[:12]
                ]
                extras["readers"] = [
                    {
                        "name": row.get("name"),
                        "kind": row.get("kind"),
                        "file": row.get("file"),
                        "line": row.get("line_start"),
                    }
                    for row in list(field.get("readers") or [])[:12]
                ]
                if field.get("canonical"):
                    extras["canonical"] = field.get("canonical")
                if field.get("occupancy_axis"):
                    extras["occupancy_axis"] = field.get("occupancy_axis")
            if not extras.get("readers") and eid:
                grouped = self._grouped_edges(eid, entity_kind=kind)
                reads = grouped.get("READS") if isinstance(grouped.get("READS"), dict) else {}
                neigh = list(reads.get("neighbors") or [])
                if neigh:
                    extras["readers"] = [
                        {
                            "name": row.get("name"),
                            "kind": row.get("kind"),
                            "file": row.get("file"),
                            "line": row.get("line"),
                        }
                        for row in neigh[:8]
                    ]
            timeline = self._write_timeline(name, primary=hit)
            extras.update(timeline)
        if kind == EntityKind.KERNEL.value:
            launch = self.aggregate_kernel_launch(name, limit=8)
            extras["phases"] = [
                {
                    "pipe": row.get("pipe") or row.get("name"),
                    "file": row.get("file"),
                    "line": row.get("line_start"),
                    "phase": row.get("phase"),
                }
                for row in list(launch.get("phases") or [])
                if row.get("ok")
            ][:8]
            entry = launch.get("entry") if isinstance(launch.get("entry"), dict) else None
            if entry:
                extras["entry"] = {
                    "name": entry.get("name"),
                    "file": entry.get("file"),
                    "line": entry.get("line_start"),
                }
        facts = hit.get("facts") if isinstance(hit.get("facts"), dict) else {}
        packing = facts.get("packing_value_sites")
        if (
            isinstance(packing, list)
            and packing
            and kind != EntityKind.TILING_KEY.value
        ):
            extras["packing_value_sites"] = packing[:3]
        if kind in {EntityKind.COMPILE_VAR.value, EntityKind.MACRO.value}:
            for key in ("value", "value_expr", "origin", "definition"):
                if facts.get(key) not in (None, "", []):
                    extras[key] = facts[key]
            same = self._same_value_neighbors(hit)
            if same:
                extras["same_value"] = same
        if kind == EntityKind.TYPE.value:
            for key in ("alias_of", "cpp_kind", "type_name", "members", "catalog", "role", "spelling"):
                if facts.get(key) not in (None, "", []):
                    extras[key] = facts[key]
        if kind == EntityKind.OPERATION.value:
            for key in ("callee", "mechanism", "function", "receiver"):
                if facts.get(key) not in (None, "", []):
                    extras[key] = facts[key]
        extras.update(surface_facts(kind, facts))
        return extras

    def query_cover(self, pattern: str, *, limit: int = 8) -> dict[str, Any]:
        needle = str(pattern or "").strip()
        match = self.aggregate_template_match(needle, limit=limit)
        _rest, dim_only = normalize_cover_pattern(needle)
        structured = dict(match.get("filters") or {})
        if dim_only:
            counts = self._legal_dim_value_counts(str(dim_only))
            legal_n = sum(counts.values())
            legal = {"ok": True, "total_matched": legal_n, "count": legal_n}
        else:
            legal = self.legal_key_query(pattern=needle, limit=limit)
            counts = {}
            legal_n = int(legal.get("total_matched") or legal.get("count") or 0)
        matched = int(match.get("matching_block_count") or 0)
        coverage = dict(match.get("coverage") or {})
        dim_coverage = coverage.get("dim_coverage") or match.get("dim_coverage") or {}
        if dim_only and counts:
            dim_coverage = {str(dim_only): list(counts.keys())}
        coverage["legal_key_count"] = legal_n
        payload: dict[str, Any] = {
            "ok": bool(match.get("ok") or legal.get("ok") or dim_coverage or legal_n),
            "shape": "cover",
            "pattern": needle,
            "filters": structured,
            "dim_coverage": dim_coverage,
            "matching_block_count": matched,
            "nearby": list(match.get("nearby") or [])[: int(limit)],
            "total_matched": matched,
            "legal_key_count": legal_n,
            "coverage": coverage,
            "count": int(match.get("count") or matched or legal_n),
            "answerable": bool((match.get("coverage") or {}).get("answerable") or legal_n),
            "completeness": str((match.get("coverage") or {}).get("completeness") or ""),
        }
        if counts:
            payload["dim_value_counts"] = {str(dim_only): counts}
        if match.get("dim_only"):
            payload["dim_only"] = match.get("dim_only")
            payload["cover_kind"] = "dim_list"
            payload["ok"] = True
        if structured and not dim_only:
            if legal_n > 0:
                cross = {}
                if len(structured) == 1:
                    dim_name, dval = next(iter(structured.items()))
                    cross = self._legal_cross_counts(dim_name, dval)
                else:
                    # Combo filter: still cross remaining dims against the first.
                    dim_name, dval = next(iter(structured.items()))
                    cross = self._legal_cross_counts(dim_name, dval)
                    cross = {
                        k: v
                        for k, v in cross.items()
                        if k not in structured
                    }
                if cross:
                    payload["cross_counts"] = cross
                    prefix = str(dim_name)[:4].lower()
                    dim_sw = next(
                        (
                            k
                            for k in cross
                            if str(k).lower().startswith(prefix) and k != dim_name
                        ),
                        "",
                    )
                    dim_dt = (
                        "DeterType"
                        if dim_name != "DeterType" and self._legal_dim_exists("DeterType")
                        else ""
                    )
                    if not dim_sw:
                        dim_sw = next(
                            (k for k in cross if k not in {dim_name, dim_dt}),
                            "",
                        )
                    pair: dict[str, Any] = {}
                    if dim_dt and dim_sw:
                        pair = self._legal_pair_cross(dim_name, dval, dim_dt, dim_sw)
                    elif dim_sw:
                        other = next(
                            (k for k in cross if k not in {dim_name, dim_sw}),
                            "",
                        )
                        if other:
                            pair = self._legal_pair_cross(dim_name, dval, other, dim_sw)
                    if pair.get("cells"):
                        payload["cross_pair"] = pair
            else:
                declared_bits: list[str] = []
                for dim_name in structured:
                    domain = self._declared_dim_values(dim_name)
                    if domain:
                        declared_bits.append(f"{dim_name} {{{', '.join(domain)}}}")
                if declared_bits:
                    payload["hint"] = (
                        "declared " + "; ".join(declared_bits) + f", legal_key=0"
                    )
        if matched > 0:
            payload["template_blocks"] = list(match.get("template_blocks") or [])[:TEMPLATE_BLOCK_EXEMPLARS]
        else:
            payload["template_blocks"] = []
        sel_sites = list(match.get("sel_sites") or [])[:8]
        if sel_sites:
            payload["sel_sites"] = sel_sites
            first = sel_sites[0]
            window = ""
            with self._connect() as conn:
                window = _source_line_window(
                    conn,
                    str(first.get("file") or ""),
                    int(first.get("line") or 0),
                )
            if window:
                payload["snippet"] = window
                payload["file"] = first.get("file") or ""
                payload["line"] = int(first.get("line") or 0)
        compiled = self._compiled_dim_names()
        payload["dim_names"] = compiled
        dim_known = True
        if dim_only:
            known = {name.lower() for name in compiled}
            dim_known = dim_only.lower() in known
            if not dim_known:
                # A guessed field name is not an empty catalog of that dim.
                payload["ok"] = False
                payload["count"] = 0
                payload["dim_coverage"] = {}
                payload.pop("dim_only", None)
                payload.pop("cover_kind", None)
                payload.pop("dim_value_counts", None)
        legal_miss_hint = str(payload.get("hint") or "")
        hint_pattern = dim_only or needle
        attach_query_hints(
            payload, hint_pattern, count=int(payload.get("count") or 0), mode="cover"
        )
        if dim_known and legal_miss_hint:
            payload["hint"] = legal_miss_hint
        elif dim_known and match.get("hint"):
            payload["hint"] = match.get("hint")
        return _fit_payload(payload)

    def _around_seed_hits(
        self,
        file: str,
        start: int,
        end: int,
        *,
        limit: int,
        with_snippet: bool = True,
    ) -> list[dict[str, Any]]:
        needle = _strip_dot_slash(str(file or "").replace("\\", "/"))
        cap = max(1, int(limit or AROUND_SEED_LIMIT))
        order = (
            "ORDER BY (IFNULL(e.line_end, e.line_start) - IFNULL(e.line_start, 0)) ASC,"
            " e.kind, e.name"
        )
        with self._connect() as conn:
            seed_rows = _seed_rows_for_file(
                conn, needle, start, end, cap * 4, order=order
            )
            if not seed_rows:
                alt = _alternate_file_spelling(needle)
                if alt:
                    seed_rows = _seed_rows_for_file(
                        conn, alt, start, end, cap * 4, order=order
                    )
            hits = self._hits_from_rows(
                conn, seed_rows, why="around", with_snippet=with_snippet
            )
        return self._rank_around_seeds(hits, line=start, limit=cap)

    def _rank_around_seeds(
        self, hits: list[dict[str, Any]], *, line: int, limit: int
    ) -> list[dict[str, Any]]:
        enclosing = [hit for hit in hits if _encloses_line(hit, line)]
        enclosing.sort(
            key=lambda hit: (
                int(hit.get("line_end") or 0) - int(hit.get("line_start") or 0),
                str(hit.get("name") or ""),
            )
        )
        enclosed_ids = {str(hit.get("id") or "") for hit in enclosing}

        def _key(hit: dict[str, Any]) -> tuple[Any, ...]:
            kind = str(hit.get("kind") or "")
            name = str(hit.get("name") or "")
            start = int(hit.get("line_start") or 0)
            handler = 1 if "FieldHandler" in name else 0
            empty = 1 if not str(hit.get("file") or "").strip() else 0
            return (
                empty,
                handler,
                _AROUND_KIND_PRIORITY.get(kind, 8),
                abs(start - int(line or 0)),
                name,
            )

        others = [
            hit
            for hit in hits
            if str(hit.get("id") or "") not in enclosed_ids
        ]
        ranked = enclosing + sorted(others, key=_key)
        return ranked[: max(1, int(limit))]

    def _around_one_hop(self, seed_ids: list[str]) -> list[dict[str, Any]]:
        useful = tuple(USEFUL_EDGE_KINDS)
        if not seed_ids or not useful:
            return []
        placeholders = ",".join("?" for _ in useful)
        per_kind: dict[str, list[dict[str, Any]]] = {}
        seen: set[tuple[str, str, str]] = set()
        with self._connect() as conn:
            for eid in seed_ids:
                if not eid:
                    continue
                rows = conn.execute(
                    f"""
                    SELECT r.kind AS rel_kind, r.data AS rel_data,
                           e.kind AS other_kind, e.name AS other_name,
                           e.file AS other_file, e.line_start AS other_line, e.id AS other_id
                    FROM relation r
                    JOIN entity e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
                    WHERE (r.src = ? OR r.dst = ?) AND r.kind IN ({placeholders})
                    ORDER BY CASE r.kind WHEN 'DECLARES' THEN {_DECLARES_KIND_SQL} ELSE 0 END, e.kind, e.name
                    """,
                    (eid, eid, eid, *useful),
                ).fetchall()
                for row in rows:
                    if _is_advisory_data(_row_get(row, "rel_data")):
                        continue
                    rel = str(_row_get(row, "rel_kind") or "")
                    other_kind = str(_row_get(row, "other_kind") or "")
                    if _skip_composition_neighbor("", rel, other_kind):
                        continue
                    bucket = per_kind.setdefault(rel, [])
                    if len(bucket) >= AROUND_NEIGHBORS_PER_KIND:
                        continue
                    other_id = str(_row_get(row, "other_id") or "")
                    other_name = str(_row_get(row, "other_name") or "")
                    if _is_noise_name_sql(other_name):
                        continue
                    key = (rel, other_kind, other_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    bucket.append(
                        {
                            "rel": rel,
                            "name": other_name,
                            "kind": str(_row_get(row, "other_kind") or ""),
                            "file": _norm_file(str(_row_get(row, "other_file") or "")),
                            "line": int(_row_get(row, "other_line") or 0),
                            "id": other_id,
                        }
                    )
        out: list[dict[str, Any]] = []
        for rows in per_kind.values():
            out.extend(rows)
        return out

    def _enclosing_def(
        self, file: str, line: int
    ) -> dict[str, Any] | None:
        needle = _strip_dot_slash(str(file or "").replace("\\", "/"))
        loc = int(line or 0)
        if not needle or loc <= 0:
            return None
        key = (needle, loc)
        if key in self._enclosing_cache:
            return self._enclosing_cache[key]
        found = self._enclosing_def_uncached(needle, loc)
        if len(self._enclosing_cache) < 8192:
            self._enclosing_cache[key] = found
        return found

    def _enclosing_def_uncached(
        self, needle: str, loc: int
    ) -> dict[str, Any] | None:
        leaf = needle.rsplit("/", 1)[-1]
        kinds = tuple(_ENCLOSE_KINDS)
        ph = ",".join("?" for _ in kinds)
        # A callee recorded at its call site has line_end == line_start, and
        # the smallest-span tie-break made that one-line row beat the function
        # it sits inside. Only a span with a body can enclose anything.
        sql = f"""
            SELECT e.id, e.kind, e.name, e.file, e.line_start, e.line_end
            FROM entity e
            WHERE e.kind IN ({ph})
              AND IFNULL(e.line_start, 0) > 0
              AND IFNULL(e.line_end, 0) > e.line_start
              AND ? BETWEEN e.line_start AND e.line_end
              AND (
                    REPLACE(REPLACE(IFNULL(e.file, ''), '\\', '/'), '\\', '/') = ?
                 OR REPLACE(REPLACE(IFNULL(e.file, ''), '\\', '/'), '\\', '/') LIKE '%/' || ?
              )
            ORDER BY (e.line_end - e.line_start) ASC, e.name
            LIMIT 8
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (*kinds, loc, needle, leaf)).fetchall()
        hits: list[dict[str, Any]] = []
        for row in rows:
            path = _norm_file(str(_row_get(row, "e.file") or _row_get(row, "file") or ""))
            if path and not _file_same(path, needle):
                continue
            hits.append(
                {
                    "id": str(_row_get(row, "e.id") or _row_get(row, "id") or ""),
                    "kind": str(_row_get(row, "e.kind") or _row_get(row, "kind") or ""),
                    "name": str(_row_get(row, "e.name") or _row_get(row, "name") or ""),
                    "file": path or needle,
                    "line": int(_row_get(row, "e.line_start") or _row_get(row, "line_start") or 0),
                    "line_start": int(
                        _row_get(row, "e.line_start") or _row_get(row, "line_start") or 0
                    ),
                    "line_end": int(
                        _row_get(row, "e.line_end") or _row_get(row, "line_end") or 0
                    ),
                }
            )
        return hits[0] if hits else None

    def _defs_overlapping_span(
        self, file: str, start: int, end: int
    ) -> list[dict[str, Any]]:
        """FUNCTION/METHOD/KERNEL rows whose body overlaps ``[start, end]``.

        Ordered by overlap length descending so a search-style window that
        starts in the previous function still names the function that owns
        most of the requested lines.
        """
        needle = _strip_dot_slash(str(file or "").replace("\\", "/"))
        lo = int(start or 0)
        hi = int(end or 0)
        if not needle or lo <= 0 or hi < lo:
            return []
        leaf = needle.rsplit("/", 1)[-1]
        kinds = tuple(_ENCLOSE_KINDS)
        ph = ",".join("?" for _ in kinds)
        sql = f"""
            SELECT e.id, e.kind, e.name, e.file, e.line_start, e.line_end
            FROM entity e
            WHERE e.kind IN ({ph})
              AND IFNULL(e.line_start, 0) > 0
              AND IFNULL(e.line_end, 0) > e.line_start
              AND e.line_start <= ?
              AND e.line_end >= ?
              AND (
                    REPLACE(REPLACE(IFNULL(e.file, ''), '\\', '/'), '\\', '/') = ?
                 OR REPLACE(REPLACE(IFNULL(e.file, ''), '\\', '/'), '\\', '/') LIKE '%/' || ?
              )
            LIMIT 24
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (*kinds, hi, lo, needle, leaf)).fetchall()
        hits: list[dict[str, Any]] = []
        for row in rows:
            path = _norm_file(str(_row_get(row, "e.file") or _row_get(row, "file") or ""))
            if path and not _file_same(path, needle):
                continue
            fn_lo = int(_row_get(row, "e.line_start") or _row_get(row, "line_start") or 0)
            fn_hi = int(_row_get(row, "e.line_end") or _row_get(row, "line_end") or 0)
            overlap = max(0, min(fn_hi, hi) - max(fn_lo, lo) + 1)
            hits.append(
                {
                    "id": str(_row_get(row, "e.id") or _row_get(row, "id") or ""),
                    "kind": str(_row_get(row, "e.kind") or _row_get(row, "kind") or ""),
                    "name": str(_row_get(row, "e.name") or _row_get(row, "name") or ""),
                    "file": path or needle,
                    "line": fn_lo,
                    "line_start": fn_lo,
                    "line_end": fn_hi,
                    "overlap": overlap,
                }
            )
        hits.sort(
            key=lambda h: (
                -int(h.get("overlap") or 0),
                int(h.get("line_end") or 0) - int(h.get("line_start") or 0),
            )
        )
        return hits

    def _window_identity_def(
        self, file: str, line: int, line_end: int
    ) -> dict[str, Any] | None:
        """Identity for a source window: enclosing ``line``, unless the bulk is elsewhere.

        ``line_end`` that merely crops a few extra lines of the same function
        keeps that function (the 1657 SetSplitAxis baseline). A window whose
        start sits in the previous function's tail retitles to the function
        that covers most of ``[line, line_end]``.
        """
        loc = int(line or 0)
        asked_end = int(line_end or 0)
        start_enc = self._enclosing_def(file, loc)
        if asked_end <= loc:
            return start_enc
        window_len = asked_end - loc + 1
        start_overlap = 0
        if start_enc:
            a = max(loc, int(start_enc.get("line_start") or 0))
            b = min(asked_end, int(start_enc.get("line_end") or 0))
            start_overlap = max(0, b - a + 1)
        if start_overlap * 2 >= window_len:
            return start_enc
        overlapping = self._defs_overlapping_span(file, loc, asked_end)
        if overlapping:
            return overlapping[0]
        return self._enclosing_def(file, asked_end) or start_enc

    def _enclosing_type(
        self, file: str, line: int, *, span_start: int = 0
    ) -> dict[str, Any] | None:
        """The TYPE a header site belongs to.

        A struct is often recorded with a one-line span, so containment alone
        finds nothing; the source span that covers the site starts on the
        declaration line, which is the same type.
        """
        needle = _strip_dot_slash(str(file or "").replace("\\", "/"))
        loc = int(line or 0)
        if not needle or loc <= 0:
            return None
        leaf = needle.rsplit("/", 1)[-1]
        anchor = int(span_start or 0) or loc
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, name, file, line_start, line_end
                FROM entity
                WHERE kind = ?
                  AND IFNULL(line_start, 0) > 0
                  AND (
                        ? BETWEEN line_start AND IFNULL(line_end, line_start)
                     OR line_start = ?
                  )
                  AND (
                        REPLACE(IFNULL(file, ''), '\\', '/') = ?
                     OR REPLACE(IFNULL(file, ''), '\\', '/') LIKE '%/' || ?
                  )
                ORDER BY (IFNULL(line_end, line_start) - line_start) ASC
                LIMIT 4
                """,
                (EntityKind.TYPE.value, loc, anchor, needle, leaf),
            ).fetchall()
        for row in rows:
            path = _norm_file(str(row[3] or ""))
            if path and not _file_same(path, needle):
                continue
            return {
                "id": str(row[0] or ""),
                "kind": str(row[1] or ""),
                "name": str(row[2] or ""),
                "file": path or needle,
                "line": int(row[4] or 0),
                "line_start": int(row[4] or 0),
                "line_end": int(row[5] or 0),
            }
        return None

    def _covering_source_span(
        self, file: str, line: int
    ) -> tuple[int, int, str] | None:
        needle = _strip_dot_slash(str(file or "").replace("\\", "/"))
        loc = int(line or 0)
        if not needle or loc <= 0:
            return None
        leaf = needle.rsplit("/", 1)[-1]
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT file, line_start, line_end, snippet
                    FROM source_span
                    WHERE IFNULL(line_start, 0) > 0
                      AND IFNULL(line_end, 0) >= line_start
                      AND ? BETWEEN line_start AND line_end
                      AND (
                            REPLACE(REPLACE(IFNULL(file, ''), '\\', '/'), '\\', '/') = ?
                         OR REPLACE(REPLACE(IFNULL(file, ''), '\\', '/'), '\\', '/') LIKE '%/' || ?
                      )
                    ORDER BY (line_end - line_start) ASC
                    LIMIT 8
                    """,
                    (loc, needle, leaf),
                ).fetchall()
            except sqlite3.OperationalError:
                return None
        for row in rows:
            path = _norm_file(str(_row_get(row, "file") or ""))
            if path and not _file_same(path, needle):
                continue
            start = int(_row_get(row, "line_start") or 0)
            end = int(_row_get(row, "line_end") or 0)
            if start > 0 and end >= start:
                return start, end, str(_row_get(row, "snippet") or "")
        return None

    def _guard_by_relation(self) -> dict[str, str]:
        """relation.id → the guard that fired it. One scan per snapshot."""
        if self._guard_rel_cache is not None:
            return self._guard_rel_cache
        with self._connect() as conn:
            guards = conn.execute(
                """
                SELECT r.src, e.name AS guard_name
                FROM relation r
                JOIN entity e ON e.id = r.dst
                WHERE r.kind = 'GUARDED_BY'
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                """
            ).fetchall()
        by_rel: dict[str, str] = {}
        for row in guards:
            name = str(_row_get(row, "guard_name") or _row_get(row, "name") or "").strip()
            if not name or _is_noise_name_sql(name):
                continue
            src = str(_row_get(row, "r.src") or _row_get(row, "src") or "")
            if src and src not in by_rel:
                by_rel[src] = name
        self._guard_rel_cache = by_rel
        return by_rel

    def _branch_spans(
        self, file: str, *, start: int = 0, end: int = 0
    ) -> list[tuple[int, int, str]]:
        """BRANCH spans that can enclose a site, so a write can name its guard.

        Host def-use records the guard as a BRANCH entity sitting on the guarded
        line rather than as a GUARDED_BY edge on the write, so the write itself
        looks unconditional. Reading the spans back is what turns
        ``isBn2MultiBlk = false`` at 1693 into ``when dropMaskOuter``.

        The span is the guarded body where the build recorded one, not
        ``line_start``: a negated guard is written at the ``if`` but only holds
        inside the ``else``, so starting at the condition would report every
        site in the ``then`` block under the negation of its actual guard.

        A source window only needs branches overlapping that window. Loading
        every BRANCH in a host tiling file (thousands) dominated site latency.
        """
        path = str(file or "")
        if not path:
            return []
        lo, hi = int(start or 0), int(end or 0)
        windowed = lo > 0 and hi >= lo
        key = (path, lo, hi) if windowed else (path, 0, 0)
        cached = self._branch_index_cache.get(key)
        if cached is not None:
            return cached
        kind = EntityKind.BRANCH.value
        sql = (
            "SELECT name, IFNULL(json_extract(data, '$.guard_body_start'), line_start), "
            "line_end FROM entity WHERE kind = ? AND file = ? "
            "AND IFNULL(line_start, 0) > 0 AND IFNULL(line_end, 0) >= line_start"
        )
        args: list[Any] = [kind, path]
        if windowed:
            sql += " AND IFNULL(line_end, 0) >= ? AND IFNULL(line_start, 0) <= ?"
            args.extend([lo, hi])
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
            if not rows:
                leaf = path.replace("\\", "/").rsplit("/", 1)[-1]
                if leaf:
                    like_sql = sql.replace(
                        "AND file = ?", "AND IFNULL(file, '') LIKE '%/' || ?"
                    )
                    like_args = list(args)
                    like_args[1] = leaf
                    rows = conn.execute(like_sql, like_args).fetchall()
        spans: list[tuple[int, int, str]] = []
        for name, body_start, body_end in rows:
            text = str(name or "").strip()
            if not text or _is_noise_name_sql(text):
                continue
            span_lo, span_hi = int(body_start or 0), int(body_end or 0)
            if span_lo <= 0 or span_hi < span_lo:
                continue
            if windowed and (span_hi < lo or span_lo > hi):
                continue
            spans.append((span_lo, span_hi, text))
        spans.sort(key=lambda item: (item[0], item[1] - item[0]))
        self._branch_index_cache[key] = spans
        return spans

    _TYPE_SPAN_SQL = """
        SELECT name, line_start, line_end
        FROM entity
        WHERE kind = ?
          AND IFNULL(line_start, 0) > 0
          AND IFNULL(line_end, 0) >= line_start
          AND file {match}
    """

    def _type_spans(self, file: str) -> list[tuple[int, int, str]]:
        """TYPE spans in one file, innermost last, so a member can name its owner."""
        path = str(file or "")
        if not path:
            return []
        cached = self._type_span_cache.get(path)
        if cached is not None:
            return cached
        kind = EntityKind.TYPE.value
        with self._connect() as conn:
            rows = conn.execute(
                self._TYPE_SPAN_SQL.format(match="= ?"), (kind, path)
            ).fetchall()
            if not rows:
                leaf = path.replace("\\", "/").rsplit("/", 1)[-1]
                if leaf:
                    rows = conn.execute(
                        self._TYPE_SPAN_SQL.format(match="LIKE '%/' || ?"), (kind, leaf)
                    ).fetchall()
        spans: list[tuple[int, int, str]] = []
        for name, start, end in rows:
            text = str(name or "").strip()
            lo, hi = int(start or 0), int(end or 0)
            if text and lo > 0 and hi >= lo:
                spans.append((lo, hi, text))
        spans.sort(key=lambda item: (item[0], item[1] - item[0]))
        self._type_span_cache[path] = spans
        return spans

    def owner_of(self, file: str, line: int, data: Any = None) -> str:
        """Class or struct a member at this location belongs to. '' when free.

        `data.scope` is authoritative when the build filled it, but it is empty
        for whole headers, so fall back to the innermost enclosing TYPE span.
        Without an owner, two classes declaring the same member name are one
        entity to a caller, and `A::x` silently answers about `B::x`.
        """
        blob = data if isinstance(data, dict) else _parse_data(data)
        scope = str((blob or {}).get("scope") or "").strip()
        if scope:
            return _last_ident(scope.replace(".", "::"))
        loc = int(line or 0)
        if loc <= 0:
            return ""
        best = ""
        best_width = -1
        for start, end, name in self._type_spans(str(file or "")):
            if start > loc:
                break
            if loc > end:
                continue
            width = end - start
            if best_width < 0 or width < best_width:
                best, best_width = name, width
        if best:
            return best
        # An out-of-class definition sits outside every class body span, so the
        # only thing that still names its class is the qualifier on the member
        # being defined.
        enclosing = self._enclosing_def(str(file or ""), loc)
        owner = str((enclosing or {}).get("name") or "").replace(".", "::")
        if "::" in owner:
            return _last_ident(owner.rsplit("::", 1)[0])
        return ""

    def site_guard(self, file: str, line: int) -> str:
        """Condition under which the code at this location runs. '' when none.

        A cited location without its condition reads as unconditional. That is
        the difference between "the kernel reads this field" and "the kernel
        reads this field only on the IS_N_EQUAL path".
        """
        return _join_guards(self._branch_guards_at(file, line))

    def _branch_guards_at(self, file: str, line: int) -> list[str]:
        """Guards enclosing one line, innermost first.

        A branch written on the line itself is skipped: a read inside
        ``if (x > 0)`` is not conditional on ``x > 0``, it is what decides it.
        Reporting the site's own condition as its guard inverts the reading.
        """
        loc = int(line or 0)
        if loc <= 0:
            return []
        key = (str(file or ""), loc)
        hit = self._branch_guard_cache.get(key)
        if hit is not None:
            return hit
        spans = self._branch_spans(str(file or ""), start=loc, end=loc)
        out = _guards_from_spans(spans, loc)
        self._branch_guard_cache[key] = out
        return out

    def _state_index(self) -> dict[str, list[dict[str, Any]]]:
        """Confirmed WRITES/DERIVES sites grouped by file, ordered by line.

        The write site of every field in the product is one unfiltered join;
        the range predicate lives in ``relation.data`` JSON, so SQL cannot
        narrow it without decoding all 131k rows anyway. A committed ``.uo``
        never changes, so the decode happens once per snapshot and a site
        lookup becomes a bisect.
        """
        if self._state_index_cache is not None:
            return self._state_index_cache
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.id, r.data, dst.name AS dst_name, src.name AS src_name
                FROM relation r
                JOIN entity dst ON dst.id = r.dst
                LEFT JOIN entity src ON src.id = r.src
                WHERE r.kind IN ('WRITES', 'DERIVES')
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                """
            ).fetchall()
        index: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            name = str(_row_get(row, "dst_name") or "")
            if not name or _is_noise_name_sql(name):
                continue
            data = _parse_data(_row_get(row, "r.data") or _row_get(row, "data"))
            line = int(data.get("line") or data.get("line_start") or 0)
            if line <= 0:
                continue
            rid = str(_row_get(row, "r.id") or _row_get(row, "id") or "")
            file = str(data.get("file") or "")
            rhs = str(
                data.get("rhs")
                or data.get("expression")
                or _row_get(row, "src_name")
                or ""
            ).strip()
            index.setdefault(file, []).append(
                {
                    "name": name,
                    "leaf": _last_ident(name).lower(),
                    "line": line,
                    "rhs": rhs,
                    "when": "",
                    "writer": str(data.get("function") or ""),
                    "_rid": rid,
                }
            )
        for sites in index.values():
            sites.sort(key=lambda item: item["line"])
        self._state_index_cache = index
        return index

    def site_write_facts(self, file: str, line: int, name: str = "") -> dict[str, Any]:
        """What the graph records about the write at one location.

        Attribute-carried site lists (``producer_sites`` and friends) hold only
        file/line, so a symbol card could cite three writes without saying what
        any of them assigned. The relation index knows.
        """
        loc = int(line or 0)
        if loc <= 0:
            return {}
        want = _last_ident(str(name or "").replace(".", "::")).lower()
        needle = _norm_file(str(file or ""))
        rows = self._writes_named(want) if want else []
        for item in rows:
            if int(item.get("line") or 0) != loc:
                continue
            if needle and _norm_file(str(item.get("file") or "")) != needle:
                if not str(item.get("file") or "").replace("\\", "/").endswith(
                    needle.replace("\\", "/").rsplit("/", 1)[-1]
                ):
                    continue
            return {
                "rhs": str(item.get("rhs") or ""),
                "when": str(item.get("when") or "") or _join_guards(self._branch_guards_at(file, loc)),
                "function": str(item.get("writer") or "") or self._enclosing_name(file, loc),
            }
        enclosing = self._enclosing_name(file, loc)
        return {"function": enclosing} if enclosing else {}

    def _writes_named(self, leaf: str) -> list[dict[str, Any]]:
        """WRITES/DERIVES targeting one identifier, not every field in the product."""
        ident = str(leaf or "").strip().lower()
        if not ident:
            return []
        cached = getattr(self, "_writes_named_cache", None)
        if cached is None:
            self._writes_named_cache = {}
            cached = self._writes_named_cache
        hit = cached.get(ident)
        if hit is not None:
            return hit
        with self._connect() as conn:
            ids: list[str] = []
            if self._accel_ready(conn):
                ids = [
                    str(r[0])
                    for r in conn.execute(
                        "SELECT entity_id FROM entity_name_leaf WHERE leaf = ? AND is_ascendc = 0",
                        (ident,),
                    ).fetchall()
                    if r[0]
                ]
            if not ids:
                clause, params = _leaf_name_where(ident)
                ids = [
                    str(r[0])
                    for r in conn.execute(
                        f"SELECT e.id FROM entity e WHERE {clause}",
                        params,
                    ).fetchall()
                    if r[0]
                ]
            if not ids:
                cached[ident] = []
                return []
            ph = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"""
                SELECT r.data, dst.name AS dst_name, src.name AS src_name
                FROM relation r
                JOIN entity dst ON dst.id = r.dst
                LEFT JOIN entity src ON src.id = r.src
                WHERE r.kind IN ('WRITES', 'DERIVES')
                  AND r.dst IN ({ph})
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                """,
                ids,
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = _parse_data(_row_get(row, "r.data") or _row_get(row, "data"))
            line = int(data.get("line") or data.get("line_start") or 0)
            if line <= 0:
                continue
            file = str(data.get("file") or "")
            rhs = str(
                data.get("rhs")
                or data.get("expression")
                or _row_get(row, "src_name")
                or ""
            ).strip()
            out.append(
                {
                    "name": str(_row_get(row, "dst_name") or ""),
                    "leaf": ident,
                    "file": file,
                    "line": line,
                    "rhs": rhs,
                    "when": _join_guards(self._branch_guards_at(file, line)),
                    "writer": str(data.get("function") or ""),
                }
            )
        cached[ident] = out
        return out

    def _enclosing_name(self, file: str, line: int) -> str:
        """Function a cited line sits in. '' when the graph has no span for it."""
        enclosing = self._enclosing_def(file, line)
        return _last_ident(str((enclosing or {}).get("name") or ""))

    def _operation_index(self) -> dict[str, list[tuple[int, str, str]]]:
        """Classified kernel operations per file, ordered by line.

        Extraction labels every call site with what it does to the pipeline --
        ``sync_signal``, ``queue_dequeue``, ``memory_transfer``, ``reg_load``.
        That is the answer to "what does this block actually do", and nothing
        read it back.
        """
        if self._operation_index_cache is not None:
            return self._operation_index_cache
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file, line_start, name, data FROM entity "
                "WHERE kind = ? AND IFNULL(line_start, 0) > 0",
                (EntityKind.OPERATION.value,),
            ).fetchall()
        index: dict[str, list[tuple[int, str, str]]] = {}
        for file, line, name, raw in rows:
            data = _parse_data(raw)
            category = str(data.get("category") or "").strip()
            if not category or category.upper() == "UNKNOWN":
                continue
            callee = str(data.get("callee") or name or "")
            index.setdefault(str(file or ""), []).append(
                (int(line or 0), category, callee)
            )
        for rows_for_file in index.values():
            rows_for_file.sort(key=lambda item: item[0])
        self._operation_index_cache = index
        return index

    def _operations_in_span(
        self, file: str, start: int, end: int
    ) -> list[dict[str, Any]]:
        """Operation categories inside one source unit, most frequent first."""
        lo, hi = int(start or 0), int(end or 0)
        if lo <= 0 or hi < lo:
            return []
        index = self._operation_index()
        tally: Counter[str] = Counter()
        # A category count is a sum over unlike calls: `sync_signal x12` covered
        # eight SetFlag, three FetchEventID and one GetTPipePtr, and a reader
        # after the flag count had to derive it from the body. Keep the split.
        per_callee: dict[str, Counter[str]] = {}
        # Keyed by the operation index's own files: a callee can sit in a file
        # that records no writes at all, and matching against the write index
        # silently dropped every such callee.
        for key in self._index_files_for(index, file):
            rows = index.get(key) or []
            left = bisect_left(rows, lo, key=lambda item: item[0])
            right = bisect_right(rows, hi, key=lambda item: item[0])
            for _line, category, callee in rows[left:right]:
                tally[category] += 1
                if callee:
                    per_callee.setdefault(category, Counter())[callee] += 1
        out: list[dict[str, Any]] = []
        for category, count in tally.most_common(8):
            split = per_callee.get(category) or Counter()
            row: dict[str, Any] = {"category": category, "count": count}
            # `callees` is the same names without their counts. Carrying both
            # spends budget that the card pays for out of a neighbour list.
            if split:
                row["by_callee"] = [
                    {"name": name, "count": n} for name, n in split.most_common(4)
                ]
            out.append(row)
        return out

    def _delegated_operations(
        self, entity_id: str, *, limit: int = 12
    ) -> list[dict[str, Any]]:
        """Pipeline profile of what this function calls, one hop out.

        A dispatcher that delegates its real work reads as almost empty when
        only its own body is classified; the callees carry the operations, and
        the graph already knows who they are.
        """
        if not entity_id:
            return []
        kinds = tuple(_DEF_CARD_KINDS)
        ph = ",".join("?" for _ in kinds)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT dst.name, dst.file, dst.line_start, dst.line_end
                FROM relation r
                JOIN entity dst ON dst.id = r.dst
                WHERE r.src = ?
                  AND r.kind = 'CALLS'
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                  AND dst.kind IN ({ph})
                  AND IFNULL(dst.line_start, 0) > 0
                  AND IFNULL(dst.line_end, 0) > dst.line_start
                LIMIT ?
                """,
                (entity_id, *kinds, int(limit)),
            ).fetchall()
        tally: Counter[str] = Counter()
        via: dict[str, list[str]] = {}
        for name, file, start, end in rows:
            for row in self._operations_in_span(str(file or ""), int(start or 0), int(end or 0)):
                category = str(row.get("category") or "")
                tally[category] += int(row.get("count") or 0)
                bucket = via.setdefault(category, [])
                leaf = _last_ident(str(name or ""))
                if leaf and leaf not in bucket and len(bucket) < 3:
                    bucket.append(leaf)
        return [
            {"category": category, "count": count, "callees": via.get(category, [])}
            for category, count in tally.most_common(8)
        ]

    def _index_files_for(self, index: dict[str, Any], file: str) -> list[str]:
        """Keys of `index` that name the same file as `file`."""
        needle = str(file or "")
        cache_key = (id(index), needle)
        hit = self._index_files_cache.get(cache_key)
        if hit is not None:
            return hit
        keys = [key for key in index if not key or _file_same(key, needle)]
        self._index_files_cache[cache_key] = keys
        return keys

    def _state_files_for(self, file: str) -> list[str]:
        """Write-index keys that name the same file as `file`."""
        needle = str(file or "")
        hit = self._state_files_cache.get(needle)
        if hit is not None:
            return hit
        keys = self._index_files_for(self._state_index(), needle)
        self._state_files_cache[needle] = keys
        return keys

    def _state_changes_in_span(
        self,
        file: str,
        start: int,
        end: int,
        *,
        highlight: str = "",
    ) -> list[dict[str, Any]]:
        lo, hi = int(start or 0), int(end or 0)
        if lo <= 0 or hi < lo:
            return []
        want = _last_ident(str(highlight or "").replace(".", "::")).lower()
        index = self._state_index()
        writes: list[dict[str, Any]] = []
        for key in self._state_files_for(file):
            sites = index.get(key) or []
            left = bisect_left(sites, lo, key=lambda item: item["line"])
            right = bisect_right(sites, hi, key=lambda item: item["line"])
            for item in sites[left:right]:
                if want and item["leaf"] != want:
                    continue
                writes.append(item)
        writes.sort(key=lambda item: item["line"])
        spans = self._branch_spans(file, start=lo, end=hi) if writes else []
        grouped: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        seen: set[tuple[str, int, str]] = set()
        for item in writes:
            key3 = (item["name"], int(item["line"]), str(item.get("rhs") or ""))
            if key3 in seen:
                continue
            seen.add(key3)
            when = str(item.get("when") or "") or _join_guards(
                _guards_from_spans(spans, int(item["line"]))
            )
            if item["name"] not in grouped:
                grouped[item["name"]] = []
                order.append(item["name"])
            grouped[item["name"]].append(
                {
                    "name": item["name"],
                    "line": item["line"],
                    "rhs": item["rhs"],
                    "when": when,
                    "writer": item["writer"],
                }
            )
        return [{"name": name, "sites": grouped[name]} for name in order]

    def _tiling_data_fields(self, owner: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Per-field host→kernel map for one TilingData struct.

        A TilingData type is the ABI between host and kernel, so the question
        asked of it is always "who fills each field and who reads it". Resolving
        the struct returned only its declaration, which is the one thing the
        source already shows.
        """
        name = _last_ident(str(owner or ""))
        if not name:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, line_start, data FROM entity "
                "WHERE kind = ? ORDER BY line_start",
                (EntityKind.TILING_FIELD.value,),
            ).fetchall()
        def _cites(value: Any, cap: int) -> list[str]:
            out_cites: list[str] = []
            for site in value or []:
                if not isinstance(site, dict):
                    continue
                fn = _last_ident(str(site.get("function") or ""))
                site_line = int(site.get("line") or site.get("line_start") or 0)
                cite = f"{fn}:{site_line}" if fn and site_line else (fn or str(site_line or ""))
                if cite and cite not in out_cites:
                    out_cites.append(cite)
                if len(out_cites) >= cap:
                    break
            return out_cites

        out: list[dict[str, Any]] = []
        for eid, field, line, raw in rows:
            data = _parse_data(raw)
            if _last_ident(str(data.get("owner") or "")) != name:
                continue
            writes = _cites(data.get("value_defining_sites"), 3)
            transport = _cites(data.get("host_writer_sites"), 2)
            out.append(
                {
                    "name": str(field or ""),
                    "line": int(line or 0),
                    "writes": writes,
                    "transport": transport,
                    "readers": self._field_reader_names(str(eid or "")),
                }
            )
            if len(out) >= int(limit):
                break
        return out

    def _field_reader_names(self, entity_id: str, *, limit: int = 3) -> list[str]:
        if not entity_id:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT src.name FROM relation r
                JOIN entity src ON src.id = r.src
                WHERE r.dst = ? AND r.kind = 'READS'
                  AND src.kind IN ('METHOD', 'FUNCTION', 'KERNEL')
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                LIMIT ?
                """,
                (entity_id, int(limit) * 3),
            ).fetchall()
        out: list[str] = []
        for (name,) in rows:
            leaf = _last_ident(str(name or ""))
            if leaf and leaf not in out and not _is_noise_name_sql(str(name or "")):
                out.append(leaf)
            if len(out) >= int(limit):
                break
        return out

    def _allocation_index(self) -> dict[str, list[dict[str, Any]]]:
        """Where each buffer/queue is given its memory, keyed by the name passed.

        ``InitBuffer(inQueuePing, 1, SIZE)`` is the declaration of how much UB a
        queue owns; the card only cited the later ``AllocTensor``, which is
        where a tile is taken out of it. Those are different questions.
        """
        if self._alloc_index_cache is not None:
            return self._alloc_index_cache
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file, line_start, data FROM entity WHERE kind = ?",
                (EntityKind.OPERATION.value,),
            ).fetchall()
        index: dict[str, list[dict[str, Any]]] = {}
        for file, line, raw in rows:
            data = _parse_data(raw)
            if str(data.get("category") or "") not in {"buffer_init", "memory_init"}:
                continue
            args = data.get("args")
            if isinstance(args, str):
                args = [args]
            args = [str(a) for a in (args or []) if str(a or "").strip()]
            if not args:
                continue
            target = _last_ident(args[0])
            if not target:
                continue
            index.setdefault(target, []).append(
                {
                    "callee": str(data.get("callee") or ""),
                    "file": _norm_file(str(file or "")),
                    "line": int(line or 0),
                    "size": args[-1] if len(args) > 1 else "",
                }
            )
        for sites in index.values():
            sites.sort(key=lambda row: int(row.get("line") or 0))
        self._alloc_index_cache = index
        return index

    def allocation_sites(self, name: str) -> list[dict[str, Any]]:
        leaf = _last_ident(str(name or ""))
        if not leaf:
            return []
        return list(self._allocation_index().get(leaf) or [])[:4]

    def tier_template_parameter(self, file: str, line: int) -> tuple[str, str] | None:
        """The template parameter the declaration at *file:line* takes its tier from.

        A member of a class template gets its memory space from the enclosing
        ``template <...>``, so the declaration alone reads as tier-less. Both
        halves are in the snapshot, so this is a read of the source rather
        than an inference from it.
        """
        from ascendc_codemap_mcp.engine.semantics.ascendc_storage import (
            tier_template_parameter,
        )

        want = int(line or 0)
        if not file or want <= 0:
            return None
        with self._connect() as conn:
            path = _snapshot_path_for(conn, str(file))
            body = _snapshot_file_lines(conn, path)
        if not body:
            return None
        text = {ln: src for ln, src in body}
        decl = text.get(want, "")
        if "<" not in decl:
            return None
        header = _enclosing_template_header(text, want)
        return tier_template_parameter(decl, header) if header else None

    def _resource_facts(
        self, kind: str, data: dict[str, Any], file: str, line: int
    ) -> list[tuple[str, Any]]:
        """Identity facts, naming the tier parameter when the declaration has one."""
        from ascendc_codemap_mcp.engine.query.bundle import _resource_identity

        parametric = None
        if str(data.get("memory_space") or "").upper() == "UNKNOWN":
            parametric = self.tier_template_parameter(file, line)
        return _resource_identity(kind, data, parametric_tier=parametric)

    def _unit_resources(self, file: str, start: int, end: int) -> list[dict[str, Any]]:
        """Registers, buffers, queues and events declared inside this unit.

        The same entity answers with its class and memory space when asked by
        name, and with nothing but its kind when asked by file+line. Reading a
        VF body and being told only ``REGISTER vregSrc`` is the version that
        sends you back for another call.
        """
        lo, hi = int(start or 0), int(end or 0)
        if not file or lo <= 0 or hi < lo:
            return []
        kinds = (
            EntityKind.REGISTER.value,
            EntityKind.BUFFER.value,
            EntityKind.QUEUE.value,
            EntityKind.EVENT.value,
            EntityKind.PIPE.value,
        )
        ph = ",".join("?" for _ in kinds)
        leaf = str(file or "").replace("\\", "/").rsplit("/", 1)[-1]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT kind, name, line_start, data FROM entity
                WHERE kind IN ({ph})
                  AND line_start BETWEEN ? AND ?
                  AND (
                        REPLACE(IFNULL(file, ''), '\\', '/') = ?
                     OR REPLACE(IFNULL(file, ''), '\\', '/') LIKE '%/' || ?
                  )
                ORDER BY line_start
                """,
                (*kinds, lo, hi, str(file).replace("\\", "/"), leaf),
            ).fetchall()
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for kind, name, line, raw in rows:
            key = (str(kind or ""), str(name or ""))
            if key in seen:
                continue
            seen.add(key)
            facts = self._resource_facts(
                str(kind or ""), _parse_data(raw), file, int(line or 0)
            )
            if not facts:
                continue
            out.append(
                {
                    "kind": str(kind or ""),
                    "name": str(name or ""),
                    "line": int(line or 0),
                    "facts": facts,
                }
            )
        out.extend(self._resources_used_in(file, lo, hi, seen))
        out.sort(key=lambda r: int(r.get("line") or 0))
        return out[:24]

    def _resources_used_in(
        self, file: str, lo: int, hi: int, seen: set[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """Resources this span touches, wherever the entity happens to be filed.

        Two templates declaring the same event name share one entity, filed at
        whichever was indexed first. Listing only entities declared inside the
        span gave the second template its buffers and none of its events,
        though it sets and waits the same four.
        """
        from ascendc_codemap_mcp.engine.query.bundle import _resource_identity

        index = self._event_site_index()
        out: list[dict[str, Any]] = []
        for key in self._index_files_for(index, file):
            sites = index.get(key) or []
            left = bisect_left(sites, lo, key=lambda item: item[0])
            right = bisect_right(sites, hi, key=lambda item: item[0])
            for site, kind, name, raw in sites[left:right]:
                ident = (str(kind or ""), str(name or ""))
                if ident in seen:
                    continue
                facts = _resource_identity(str(kind or ""), _parse_data(raw))
                if not facts:
                    continue
                seen.add(ident)
                out.append(
                    {
                        "kind": str(kind or ""),
                        "name": str(name or ""),
                        "line": int(site or 0),
                        "facts": facts,
                    }
                )
        out.sort(key=lambda row: int(row.get("line") or 0))
        return out

    def _event_site_index(self) -> dict[str, list[tuple[int, str, str, str]]]:
        """Every place an event is set or waited, per file, ordered by line.

        The site's file and line live inside the edge's JSON, so selecting on
        them scanned all 131k relations to reach the 150 that are sync edges --
        80ms on a lookup that runs on every card. Read them once instead.
        """
        if self._event_site_index_cache is not None:
            return self._event_site_index_cache
        with self._connect() as conn:
            events = conn.execute(
                "SELECT id, kind, name, data FROM entity WHERE kind = 'EVENT'"
            ).fetchall()
            ids = [str(row["id"] or row[0]) for row in events]
            by_id = {
                str(row["id"] or row[0]): (
                    str(row["kind"] or ""),
                    str(row["name"] or ""),
                    row["data"],
                )
                for row in events
            }
            rel_rows: list[Any] = []
            if ids:
                ph = ",".join("?" for _ in ids)
                # Two indexed probes. `e.id IN (r.src, r.dst)` made SQLite
                # nested-loop the whole relation table (~6s on FAG).
                rel_rows = conn.execute(
                    f"""
                    SELECT r.data, r.src, r.dst FROM relation r
                    WHERE r.kind IN ('SIGNALS', 'AWAITS') AND r.src IN ({ph})
                    """,
                    ids,
                ).fetchall()
                rel_rows.extend(
                    conn.execute(
                        f"""
                        SELECT r.data, r.src, r.dst FROM relation r
                        WHERE r.kind IN ('SIGNALS', 'AWAITS') AND r.dst IN ({ph})
                        """,
                        ids,
                    ).fetchall()
                )
        index: dict[str, list[tuple[int, str, str, str]]] = {}
        seen: set[tuple[str, int, str]] = set()
        for raw_edge, src, dst in rel_rows:
            edge = _parse_data(raw_edge)
            line = int(edge.get("line") or 0)
            if line <= 0:
                continue
            for eid in (str(src or ""), str(dst or "")):
                ent = by_id.get(eid)
                if not ent:
                    continue
                kind, name, raw_entity = ent
                path = _strip_dot_slash(str(edge.get("file") or "").replace("\\", "/"))
                key = (path, line, name)
                if key in seen:
                    continue
                seen.add(key)
                index.setdefault(path, []).append((line, kind, name, raw_entity))
        for bucket in index.values():
            bucket.sort(key=lambda item: item[0])
        self._event_site_index_cache = index
        return index

    def _definition_starts_here(
        self, file: str, hit: dict[str, Any], line: int
    ) -> bool:
        """False when a definition only *looks* like it starts on this line.

        An over-long recorded start makes the next function claim a line that
        belongs to the previous one, which then shows up as an unrelated symbol
        sitting at the site.
        """
        if str(hit.get("kind") or "").upper() not in _DEF_CARD_KINDS:
            return True
        start = int(hit.get("line_start") or hit.get("line") or 0)
        end = int(hit.get("line_end") or 0)
        if start <= 0 or end <= start:
            return True
        return self._clamp_definition_start(file, start, end) <= int(line or 0)

    def query_site_unit(
        self,
        file: str,
        line: int,
        *,
        highlight: str = "",
        limit: int = 8,
        line_end: int = 0,
    ) -> dict[str, Any]:
        """Source-centric resolve(file,line): stable enclosing function or local window.

        ``line_end`` crops the snippet. Identity is the function enclosing
        ``line`` when that function still owns most of the window; a range
        that starts in the previous function's tail is titled after the
        function that covers most of ``[line, line_end]``.
        """
        path = str(file or "").strip()
        loc = int(line or 0)
        asked_end = int(line_end or 0)
        if not path or loc <= 0:
            return {
                "ok": False,
                "shape": "around",
                "resolve_mode": "site",
                "error": "around_needs_file_line",
                "hint": "Pass file and line from a previous card.",
            }
        enclosing = self._window_identity_def(path, loc, asked_end)
        fn_start = int((enclosing or {}).get("line_start") or 0)
        fn_end = int((enclosing or {}).get("line_end") or 0)
        if fn_start > 0 and fn_end >= fn_start:
            fn_start = max(fn_start, self._clamp_definition_start(path, fn_start, fn_end))
            if loc < fn_start:
                fn_start = int((enclosing or {}).get("line_start") or fn_start)
        missing_snapshot = False
        rows: list[tuple[int, str]] = []
        unit_start, unit_end = loc, loc
        if asked_end > loc:
            # Show exactly the asked lines. Identity was already chosen from
            # the window bulk, so a range that starts in the previous function
            # still titles the next one. Skip the enclosing-body fetch that
            # would be discarded.
            unit_start, unit_end = loc, asked_end
            with self._connect() as conn:
                rows = _source_line_rows(conn, path, unit_start, unit_end)
            if not rows:
                missing_snapshot = True
        elif enclosing and fn_start > 0 and fn_end >= fn_start:
            fn_len = fn_end - fn_start + 1
            if fn_len <= _SITE_UNIT_HARD:
                unit_start, unit_end = fn_start, fn_end
            else:
                tile_idx = max(0, (loc - fn_start) // _SITE_UNIT_HARD)
                unit_start = fn_start + tile_idx * _SITE_UNIT_HARD
                unit_end = min(fn_end, unit_start + _SITE_UNIT_HARD - 1)
            with self._connect() as conn:
                rows = _source_line_rows(conn, path, unit_start, unit_end)
        else:
            covered = self._covering_source_span(path, loc)
            span_snippet = ""
            if covered:
                unit_start, unit_end, span_snippet = covered
            else:
                unit_start = max(1, loc - _NO_ENCLOSE_RADIUS)
                unit_end = loc + _NO_ENCLOSE_RADIUS
            with self._connect() as conn:
                rows = _source_line_rows(conn, path, unit_start, unit_end)
            if not rows:
                if covered:
                    if span_snippet:
                        rows = [(unit_start, span_snippet)]
                else:
                    missing_snapshot = True
                    unit_start, unit_end = loc, loc
            else:
                compound = _smallest_compound_span(rows, loc, cap=_SITE_UNIT_HARD)
                if compound:
                    unit_start, unit_end = compound
                    rows = [(n, t) for n, t in rows if unit_start <= n <= unit_end]
                if rows and (unit_end - unit_start + 1) > _SITE_UNIT_HARD:
                    lo = max(unit_start, loc - _SITE_UNIT_HARD // 2)
                    hi = min(unit_end, lo + _SITE_UNIT_HARD - 1)
                    unit_start, unit_end = lo, hi
                    rows = [(n, t) for n, t in rows if lo <= n <= hi]
        snippet = "\n".join(f"{ln}:{txt}" for ln, txt in rows) if rows else ""
        seeds = self._around_seed_hits(
            path, loc, loc, limit=max(1, int(limit)), with_snippet=False
        )
        site_hits = [
            hit
            for hit in seeds
            if int(hit.get("line") or hit.get("line_start") or 0) == loc
            and self._definition_starts_here(path, hit, loc)
        ]
        compact = [_compact_around_hit(hit, snippet=False) for hit in (site_hits or seeds[:1])]
        # A line inside a struct has no enclosing *function*, and the card then
        # opened with a bare file:range. Naming the type it belongs to is the
        # same question answered one kind up.
        enc = enclosing or self._enclosing_type(path, loc, span_start=unit_start)
        if not enc or str(enc.get("kind") or "").upper() not in _DEF_CARD_KINDS:
            # A CONTRACT or BRANCH on the start line is a neighbour, not the
            # card's subject. Only fall back to it when nothing callable encloses
            # the line at all.
            if enclosing:
                enc = enclosing
            elif not enc:
                enc = compact[0] if compact else {}
        state = (
            []
            if missing_snapshot
            else self._state_changes_in_span(
                path, unit_start, unit_end, highlight=str(highlight or "")
            )
        )
        identity_start = fn_start or unit_start
        identity_end = fn_end or unit_end
        payload = {
            "ok": bool(snippet or enc) and not missing_snapshot,
            "shape": "around",
            "resolve_mode": "site",
            "file": path,
            "line": loc,
            "unit_start": unit_start,
            "unit_end": unit_end,
            "function_start": fn_start,
            "function_end": fn_end,
            "snippet": snippet,
            "seeds": compact,
            "cards": compact,
            "enclosing": {
                "id": enc.get("id"),
                "name": enc.get("name"),
                "kind": enc.get("kind") or "FUNCTION",
                "file": _norm_file(str(enc.get("file") or path)),
                "line": int(enc.get("line") or enc.get("line_start") or identity_start),
                "line_start": identity_start,
                "line_end": identity_end,
            },
            "state_changes": state,
            "hits": compact,
            "count": len(compact),
            "truncated": False,
            "highlight": str(highlight or ""),
        }
        payload["operations"] = self._operations_in_span(path, unit_start, unit_end)
        payload["unit_resources"] = self._unit_resources(path, unit_start, unit_end)
        payload["field_bundles"] = self._unit_field_bundles(payload)
        if missing_snapshot:
            with self._connect() as conn:
                span = _snapshot_line_range(conn, path)
            if span:
                # An indexed file asked for a line past its end was reported as
                # an absent file. A reader told the file is missing stops using
                # file+line; told the line is, they pass a different one.
                payload["hint"] = (
                    f"{path} is in the snapshot but ends at line {span[1]}; "
                    f"line {loc} is past it; check the file, or resolve a "
                    "search locator"
                )
                payload["error"] = "line_out_of_range"
            else:
                payload["hint"] = (
                    "this file is not in snapshot; resolve a search locator "
                    "or Read the workspace file"
                )
                payload["error"] = "not_in_snapshot"
        else:
            attach_query_hints(payload, path, count=int(payload.get("count") or 0), mode="around")
        return _fit_payload(payload)

    def query_around(
        self,
        file: str,
        line: int,
        *,
        line_end: int = 0,
        limit: int = 8,
    ) -> dict[str, Any]:
        path = str(file or "").strip()
        start = int(line or 0)
        end = int(line_end or start)
        if not path or start <= 0:
            return {
                "ok": False,
                "shape": "around",
                "error": "around_needs_file_line",
                "hint": "Pass file and line from a previous card.",
            }
        window = ""
        with self._connect() as conn:
            if end > start:
                rows = _source_line_rows(conn, path, start, end)
                window = "\n".join(f"{ln}:{txt}" for ln, txt in rows)
            else:
                window = _source_line_window(conn, path, start)
        seed_cap = AROUND_SEED_LIMIT
        seeds = self._around_seed_hits(path, start, end or start, limit=seed_cap)
        enclosing_hits = [hit for hit in seeds if _encloses_line(hit, start)]
        site_hits = [hit for hit in seeds if int(hit.get("line") or hit.get("line_start") or 0) == start]
        if not site_hits:
            site_hits = list(seeds)
        seeds = site_hits or enclosing_hits[:1]
        compact_seeds = [_compact_around_hit(hit, snippet=False) for hit in seeds]
        if window and compact_seeds:
            compact_seeds[0]["snippet"] = window
            compact_seeds[0]["line"] = start
            compact_seeds[0]["line_end"] = end
        seed_ids = [str(hit.get("id") or "") for hit in seeds if hit.get("id")]
        neighbors = self._around_one_hop(seed_ids)
        impact: dict[str, Any] = {}
        field_kinds = {
            EntityKind.TILING_FIELD.value,
            EntityKind.FIELD.value,
            EntityKind.TILING_KEY.value,
        }
        field_names = [
            str(hit.get("name") or "")
            for hit in seeds
            if str(hit.get("kind") or "") in field_kinds
        ]
        if field_names:
            packed = self.field_impact_many(field_names)
            primary = packed.get(field_names[0]) if packed else {}
            if isinstance(primary, dict) and primary.get("ok"):
                impact = {
                    "name": field_names[0],
                    "writers": [
                        {
                            "name": row.get("name"),
                            "file": row.get("file"),
                            "line": row.get("line_start") or row.get("line"),
                        }
                        for row in list(primary.get("writers") or [])[:8]
                        if isinstance(row, dict)
                    ],
                    "readers": [
                        {
                            "name": row.get("name"),
                            "file": row.get("file"),
                            "line": row.get("line_start") or row.get("line"),
                        }
                        for row in list(primary.get("readers") or [])[:8]
                        if isinstance(row, dict)
                    ],
                }
        enclosing = (
            _compact_around_hit(enclosing_hits[0], snippet=False)
            if enclosing_hits
            else (compact_seeds[0] if compact_seeds else {})
        )
        payload = {
            "ok": bool(window or seeds),
            "shape": "around",
            "resolve_mode": "site",
            "file": path,
            "line": start,
            "line_end": end or start,
            "snippet": window,
            "seeds": compact_seeds,
            "enclosing": enclosing,
            "neighbors": neighbors,
            "hits": compact_seeds,
            "impact": impact,
            "count": len(seeds),
            "truncated": False,
        }
        attach_query_hints(payload, path, count=int(payload.get("count") or 0), mode="around")
        return _fit_payload(payload)

    def count_call_sites(self, name: str) -> int:
        """How many OPERATION entities call this ident. 0 when it is not called.

        A resolve card shows the definition; this is what tells the caller that
        the definition is not the whole answer.
        """
        ident = str(name or "").rsplit("::", 1)[-1].strip()
        if not ident:
            return 0
        # `$.callee` lives in JSON, so this cannot be index-only; a session
        # asks the same idents repeatedly.
        cached = self._call_site_count_cache.get(ident)
        if cached is not None:
            return cached
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM entity e
                WHERE e.kind = ?
                  AND (e.name = ? COLLATE NOCASE
                       OR e.name LIKE '%::' || ? COLLATE NOCASE
                       OR IFNULL(json_extract(e.data, '$.callee'), '') = ? COLLATE NOCASE)
                """,
                (EntityKind.OPERATION.value, ident, ident, ident),
            ).fetchone()
        total = int(row[0] or 0) if row else 0
        if len(self._call_site_count_cache) < 4096:
            self._call_site_count_cache[ident] = total
        return total

    def list_call_sites(
        self, name: str, *, limit: int = 8
    ) -> tuple[list[dict[str, Any]], int]:
        """Located OPERATION rows for this ident, unique by file:line."""
        ident = str(name or "").rsplit("::", 1)[-1].strip()
        if not ident:
            return [], 0
        total = self.count_call_sites(ident)
        cap = max(int(limit), 8)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.file, e.line_start, e.name
                FROM entity e
                WHERE e.kind = ?
                  AND (e.name = ? COLLATE NOCASE
                       OR e.name LIKE '%::' || ? COLLATE NOCASE
                       OR IFNULL(json_extract(e.data, '$.callee'), '') = ? COLLATE NOCASE)
                  AND IFNULL(e.file, '') != ''
                  AND e.line_start > 0
                ORDER BY e.file, e.line_start, e.id
                LIMIT ?
                """,
                (EntityKind.OPERATION.value, ident, ident, ident, max(cap * 4, 32)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for row in rows:
            file = _norm_file(str(row[0] or ""))
            line = int(row[1] or 0)
            if not file or line <= 0:
                continue
            key = (file, line)
            if key in seen:
                continue
            seen.add(key)
            out.append({"file": file, "line": line, "name": str(row[2] or ident)})
            if len(out) >= cap:
                break
        return out, total

    def _hits_by_name_like(self, pattern: str, *, fetch: int) -> list[dict[str, Any]]:
        text = str(pattern or "").strip()
        if not text:
            return []
        with self._connect() as conn:
            rows = self._select_entities(
                conn,
                extra_where="e.name COLLATE NOCASE LIKE ?",
                params=[_name_pattern_to_like(text)],
                limit=max(int(fetch), 8),
                order="e.file, e.line_start, e.id",
            )
            return self._hits_from_rows(
                conn,
                rows,
                why="find",
                with_snippet=False,
                require_span_for_branch=False,
            )

    def _declared_dim_names(self) -> list[str]:
        dim_names: list[str] = []
        seen_dims: set[str] = set()
        with self._connect() as conn:
            dim_rows = conn.execute(
                """
                SELECT name, data FROM entity
                WHERE kind = 'TILING_KEY' AND IFNULL(name, '') != ''
                ORDER BY name
                """
            ).fetchall()
        for row in dim_rows:
            name = str(row[0] or "")
            if name in seen_dims or not _IDENT_NAME_RE.fullmatch(name):
                continue
            if name.isdigit():
                continue
            data = _parse_data(row[1] if len(row) > 1 else None)
            attrs = data.get("attrs") if isinstance(data.get("attrs"), dict) else data
            if not isinstance(attrs, dict):
                attrs = {}
            if attrs.get("source_declared") is not True:
                continue
            seen_dims.add(name)
            dim_names.append(name)
        return dim_names

    def _compiled_dim_names(self) -> list[str]:
        """Dim names the caller can actually query.

        Declared TILING_KEY rows are preferred; the legal_key table is the
        fallback when a product recorded keys but not `source_declared`.
        """
        declared = self._declared_dim_names()
        if declared:
            return declared
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT DISTINCT dim FROM legal_key_dim "
                    "WHERE IFNULL(dim, '') != '' ORDER BY dim"
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [str(row[0]) for row in rows if row[0]]

    def query_index(self, *, limit: int = 8) -> dict[str, Any]:
        launch = self.aggregate_kernel_launch(limit=max(int(limit), 8))
        dim_names = self._compiled_dim_names()
        dim_counts: dict[str, dict[str, int]] = {}
        for name in dim_names:
            counts = self._legal_dim_value_counts(name)
            if counts:
                dim_counts[name] = counts
        with self._connect() as conn:
            tiling_data = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT DISTINCT name FROM entity
                    WHERE kind = 'TILING_DATA' AND IFNULL(name, '') != ''
                    ORDER BY name
                    """
                ).fetchall()
                if row[0]
            ]
        phases = [
            {
                "pipe": row.get("pipe") or row.get("name"),
                "file": row.get("file"),
                "line": row.get("line_start"),
                "phase": row.get("phase"),
            }
            for row in list(launch.get("phases") or [])
            if row.get("ok")
        ]
        entry = launch.get("entry") if isinstance(launch.get("entry"), dict) else None
        payload = {
            "ok": True,
            "shape": "index",
            "entry": (
                {
                    "name": entry.get("name"),
                    "file": entry.get("file"),
                    "line": entry.get("line_start"),
                }
                if entry
                else None
            ),
            "phases": phases,
            "dim_names": dim_names,
            "dim_coverage": {name: list(counts) for name, counts in dim_counts.items()},
            "dim_value_counts": dim_counts,
            "legal_key_count": (
                sum(next(iter(dim_counts.values())).values()) if dim_counts else 0
            ),
            "tiling_data_names": tiling_data,
            "count": len(dim_names) or len(phases),
        }
        if dim_names:
            dim_example = dim_names[0]
            payload["hint"] = (
                f"trace dim={dim_example} lists that dim's built values. "
                f"trace dim=* listed {len(dim_names)} dims."
            )
        else:
            payload["hint"] = "this snapshot has no compiled dims to list."
        attach_query_hints(
            payload, "", count=len(dim_names) or len(phases), mode="index"
        )
        return _fit_payload(payload)

    def _search_source_lines(
        self,
        conn: sqlite3.Connection,
        phrase: str,
        *,
        file_filter: str = "",
        limit: int = 8,
        and_tokens: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        from ascendc_codemap_mcp.engine.store.accel import has_source_fts, has_source_line

        cap = max(1, int(limit))
        arch = str(self._architecture or "")
        file_tok = str(file_filter or "").replace("\\", "/").strip()
        file_leaf = file_tok.rsplit("/", 1)[-1] if file_tok else ""
        file_sql = ""
        file_params: list[Any] = []
        if file_leaf:
            file_sql = " AND REPLACE(sl.path, '\\', '/') LIKE '%' || ?"
            file_params.append(file_leaf)
        order_sql = (
            " ORDER BY CASE"
            " WHEN REPLACE(sl.path, '\\', '/') LIKE '%/common/%' THEN 3"
            " WHEN REPLACE(sl.path, '\\', '/') LIKE '%/op_host/%'"
            "   OR REPLACE(sl.path, '\\', '/') LIKE '%/op_kernel/%' THEN 0"
            " WHEN REPLACE(sl.path, '\\', '/') LIKE '%/' || ? || '/%' THEN 1"
            " ELSE 2 END, sl.path, sl.line"
        )
        order_params: list[Any] = [arch]
        fts_q = (
            _fts_and_query(and_tokens)
            if and_tokens
            else _fts_match_query(phrase)
        )
        sample_rows: list[Any] | None = None
        total = 0
        if has_source_fts(conn) and fts_q:
            try:
                where = "WHERE f.source_fts MATCH ?" + file_sql
                count_params = (fts_q, *file_params)
                total = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM source_fts f "
                        "JOIN source_line sl ON sl.id = f.rowid " + where,
                        count_params,
                    ).fetchone()[0]
                    or 0
                )
                sample_rows = conn.execute(
                    "SELECT sl.path, sl.line, sl.text "
                    "FROM source_fts f JOIN source_line sl ON sl.id = f.rowid "
                    + where
                    + order_sql
                    + " LIMIT ?",
                    (*count_params, *order_params, cap),
                ).fetchall()
            except sqlite3.OperationalError:
                total = 0
                sample_rows = None
        if sample_rows is None and has_source_line(conn):
            needles = list(and_tokens) if and_tokens else [phrase]
            like_sql = " AND ".join("sl.text LIKE '%' || ? || '%'" for _ in needles)
            where = "WHERE " + like_sql + file_sql
            like_params = (*needles, *file_params)
            total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM source_line sl " + where,
                    like_params,
                ).fetchone()[0]
                or 0
            )
            sample_rows = conn.execute(
                "SELECT sl.path, sl.line, sl.text FROM source_line sl "
                + where
                + order_sql
                + " LIMIT ?",
                (*like_params, *order_params, cap),
            ).fetchall()
        if not total or sample_rows is None:
            return [], 0
        cards = [
            {
                "file": str(r[0] or ""),
                "line": int(r[1] or 0),
                "text": str(r[2] or "").rstrip(),
            }
            for r in sample_rows
        ]
        return cards, total

    def _is_template_hit(self, path: str, text: str) -> bool:
        from ascendc_codemap_mcp.engine.query.explore import _is_tpl_boilerplate
        from ascendc_codemap_mcp.engine.query.rg import ROLE_TPL, line_role

        blob = str(path or "").replace("\\", "/")
        if "template_tiling_key" in blob.rsplit("/", 1)[-1] or "/template_tiling_key" in blob:
            return True
        if _is_tpl_boilerplate(text):
            return True
        return line_role(text) == ROLE_TPL

    def _group_search_hits(
        self,
        conn: sqlite3.Connection,
        matched: list[tuple[str, int, str]],
        phrase: str,
    ) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.query.rg import rank_hit

        tpl: list[tuple[str, int, str]] = []
        real: list[tuple[str, int, str]] = []
        for row in matched:
            if self._is_template_hit(row[0], row[2]):
                tpl.append(row)
            else:
                real.append(row)
        kinds = tuple(_ENCLOSE_KINDS)
        ph = ",".join("?" for _ in kinds)
        spans = conn.execute(
            f"""
            SELECT e.name, e.file, e.line_start, e.line_end, e.kind
            FROM entity e
            WHERE e.kind IN ({ph})
              AND IFNULL(e.line_start, 0) > 0
              AND IFNULL(e.line_end, 0) >= e.line_start
            -- def_span_preload
            """,
            kinds,
        ).fetchall()
        indexed: dict[str, list[tuple[str, str, int, int]]] = {}
        for name, file, start, end, _kind in spans:
            path = str(file or "").replace("\\", "/")
            if not path:
                continue
            rec = (str(name or ""), path, int(start or 0), int(end or 0))
            indexed.setdefault(_rel_key(path), []).append(rec)
            indexed.setdefault(path.rsplit("/", 1)[-1], []).append(rec)
        units: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
        leftover: list[dict[str, Any]] = []
        arch = str(self._architecture or "")
        for path, line, text in real:
            hit = {"file": path, "line": line, "text": text}
            leaf = path.replace("\\", "/").rsplit("/", 1)[-1]
            candidates = indexed.get(_rel_key(path), []) + indexed.get(leaf, [])
            best: tuple[int, str, str, int, int] | None = None
            for name, fpath, start, end in candidates:
                if not _file_same(fpath, path):
                    continue
                if start <= line <= end:
                    span = end - start
                    if best is None or span < best[0]:
                        best = (span, name, fpath, start, end)
            if best is None:
                leftover.append(hit)
                continue
            key = (best[1], best[2], best[3], best[4])
            units.setdefault(key, []).append(hit)
        unit_rows: list[dict[str, Any]] = []
        for key, hits in units.items():
            name, file, start, end = key
            hits.sort(key=lambda h: rank_hit(h["file"], h["line"], h["text"], phrase, arch))
            best_rank = rank_hit(hits[0]["file"], hits[0]["line"], hits[0]["text"], phrase, arch)
            unit_rows.append(
                {
                    "name": name,
                    "file": file,
                    "line_start": start,
                    "line_end": end,
                    "hits": hits,
                    "rank": best_rank,
                }
            )
        unit_rows.sort(key=lambda u: u["rank"])
        leftover.sort(key=lambda h: rank_hit(h["file"], h["line"], h["text"], phrase, arch))
        return {
            "template_lines": len(tpl),
            "real_total": len(real),
            "source_units": len(unit_rows),
            "units": [{k: v for k, v in u.items() if k != "rank"} for u in unit_rows],
            "leftover": leftover,
            "real": real,
        }

    def _search_regex_lines(
        self,
        conn: sqlite3.Connection,
        phrase: str,
        *,
        file_filter: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        from ascendc_codemap_mcp.engine.query.rg import (
            compile_search,
            fts_prefilter,
            line_matches,
            path_matches,
            rank_hit,
        )
        from ascendc_codemap_mcp.engine.store.accel import has_source_fts, has_source_line

        cre = compile_search(phrase)
        cap = max(1, int(limit))
        start = max(0, int(offset or 0))
        arch = str(self._architecture or "")
        rows: list[tuple[Any, ...]] = []
        # The trigram index only stands in for the scan when it provably returns
        # a superset; `fts_prefilter` returns '' whenever it cannot prove that,
        # and the regex below is still what decides a hit either way.
        fts_q = fts_prefilter(phrase)
        scanned = False
        if fts_q and has_source_fts(conn):
            try:
                rows = conn.execute(
                    "SELECT sl.path, sl.line, sl.text "
                    "FROM source_fts f JOIN source_line sl ON sl.id = f.rowid "
                    "WHERE f.source_fts MATCH ?",
                    (fts_q,),
                ).fetchall()
                scanned = True
            except sqlite3.OperationalError:
                rows = []
                scanned = False
        if not scanned and has_source_line(conn):
            rows = conn.execute("SELECT path, line, text FROM source_line").fetchall()
        matched: list[tuple[str, int, str]] = []
        for path, line, text in rows:
            if not path_matches(str(path or ""), file_filter):
                continue
            if not line_matches(cre, str(text or "")):
                continue
            matched.append((str(path or ""), int(line or 0), str(text or "").rstrip()))
        matched.sort(key=lambda r: rank_hit(r[0], r[1], r[2], phrase, arch))
        extra = self._group_search_hits(conn, matched, phrase)
        real = extra.get("real") or matched
        total = int(extra.get("real_total") or len(real))
        page = real[start : start + cap]
        cards = [{"file": p, "line": ln, "text": tx} for p, ln, tx in page]
        return cards, total, extra

    def _field_ids_named(self, ident: str) -> dict[str, list[str]]:
        leaf = _last_ident(str(ident or "").replace(".", "::"))
        if not leaf:
            return {}
        cached = self._field_ids_cache.get(leaf)
        if cached is not None:
            return {kind: list(ids) for kind, ids in cached.items()}
        kinds = (
            EntityKind.TILING_FIELD.value,
            EntityKind.FIELD.value,
            EntityKind.TILING_KEY.value,
            EntityKind.COMPILE_VAR.value,
        )
        ph = ",".join("?" for _ in kinds)
        name_clause, name_params = _leaf_name_where(leaf)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.id, e.kind, e.file, e.line_start, e.data
                FROM entity e
                WHERE e.kind IN ({ph})
                  AND {name_clause}
                ORDER BY CASE
                    WHEN IFNULL(e.file, '') = '' OR IFNULL(e.line_start, 0) = 0 THEN 1
                    ELSE 0
                END,
                CASE
                    WHEN instr(IFNULL(e.data, ''), '"value_defining_sites"') > 0
                      OR instr(IFNULL(e.data, ''), '"host_writer_sites"') > 0
                      OR instr(IFNULL(e.data, ''), '"producer_sites"') > 0
                      OR instr(IFNULL(e.data, ''), '"packing_value_sites"') > 0
                    THEN 0
                    ELSE 1
                END,
                e.kind, e.id
                """,
                (*kinds, *name_params),
            ).fetchall()
        from ascendc_codemap_mcp.engine.query.bundle import data_has_partition_sites

        grouped: dict[str, list[tuple[str, bool]]] = {}
        seen: set[str] = set()
        for row in rows:
            eid = str(row[0] or "")
            kind = str(row[1] or "")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            grouped.setdefault(kind, []).append((eid, data_has_partition_sites(row[4])))
        buckets: dict[str, list[str]] = {}
        for kind, items in grouped.items():
            attributed = [eid for eid, ok in items if ok]
            buckets[kind] = attributed or [eid for eid, _ in items]
        self._field_ids_cache[leaf] = {kind: list(ids) for kind, ids in buckets.items()}
        return buckets

    def _flatten_field_ids(self, ident: str) -> list[str]:
        buckets = self._field_ids_named(ident)
        order = (
            EntityKind.TILING_FIELD.value,
            EntityKind.FIELD.value,
            EntityKind.TILING_KEY.value,
            EntityKind.COMPILE_VAR.value,
        )
        out: list[str] = []
        for kind in order:
            out.extend(buckets.get(kind) or [])
        for kind, ids in buckets.items():
            if kind not in order:
                out.extend(ids)
        return out

    def assignments_for(self, ident: str) -> dict[str, Any] | None:
        fids = self._flatten_field_ids(ident)
        if not fids:
            return None
        ph = ",".join("?" for _ in fids)
        sites: list[dict[str, Any]] = []
        unresolved = 0
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT r.id, r.src, r.data, src.name, src.file, src.line_start, src.kind
                FROM relation r
                LEFT JOIN entity src ON src.id = r.src
                WHERE r.dst IN ({ph})
                  AND r.kind IN ('WRITES', 'DERIVES')
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                """,
                fids,
            ).fetchall()
        by_rel = self._guard_by_relation()
        seen: set[tuple[str, int, str]] = set()
        for row in rows:
            data = _parse_data(_row_get(row, "r.data") or _row_get(row, "data"))
            # A tiling key depends on the operator inputs transitively, and the
            # graph records that as a direct edge from every INPUT root. Listing
            # them as assignments claims `query` and `head_num` are written into
            # the key, which is not what the source says.
            if str(_row_get(row, "src.kind") or _row_get(row, "kind") or "").upper() in {
                EntityKind.INPUT.value,
                EntityKind.OUTPUT.value,
            }:
                continue
            writer = str(_row_get(row, "src.name") or _row_get(row, "name") or "")
            line = int(data.get("line") or _row_get(row, "src.line_start") or _row_get(row, "line_start") or 0)
            file = str(data.get("file") or _row_get(row, "src.file") or _row_get(row, "file") or "")
            rhs = str(data.get("rhs") or data.get("expression") or "").strip()
            rid = str(_row_get(row, "r.id") or _row_get(row, "id") or "")
            if not writer or _is_noise_name_sql(writer):
                if not line:
                    unresolved += 1
                continue
            key = (writer, line, rhs)
            if key in seen:
                continue
            seen.add(key)
            if not file or line <= 0:
                unresolved += 1
            sites.append(
                {
                    "writer": writer,
                    "file": _norm_file(file),
                    "line": line,
                    "rhs": rhs,
                    "when": by_rel.get(rid) or "",
                }
            )
        if not sites and not unresolved:
            return None
        groups: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for site in sites:
            writer = str(site["writer"])
            if writer not in groups:
                groups[writer] = []
                order.append(writer)
            groups[writer].append(site)
        consumed = self._consumed_by_names(fids)
        confirmed = len(sites)
        return {
            "confirmed": confirmed,
            "unresolved": unresolved,
            "exhaustive": False,
            "total": confirmed + unresolved,
            "groups": [{"writer": w, "sites": groups[w]} for w in order],
            "consumed_by": consumed,
        }

    def _consumed_by_names(self, field_ids: list[str]) -> list[str]:
        if not field_ids:
            return []
        ph = ",".join("?" for _ in field_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT src.name
                FROM relation r
                JOIN entity src ON src.id = r.src
                WHERE r.dst IN ({ph})
                  AND r.kind IN ('READS', 'CALLS', 'BINDS')
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                """,
                field_ids,
            ).fetchall()
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row[0] or "")
            if not name or _is_noise_name_sql(name) or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def host_kernel_for(self, ident: str) -> dict[str, Any] | None:
        fids = self._flatten_field_ids(ident)
        if not fids:
            return None
        ph = ",".join("?" for _ in fids)
        producers: list[dict[str, Any]] = []
        consumers: list[dict[str, Any]] = []
        transport: list[str] = []
        unresolved_p = 0
        unresolved_c = 0
        with self._connect() as conn:
            writes = conn.execute(
                f"""
                SELECT src.name, src.file, src.line_start
                FROM relation r
                JOIN entity src ON src.id = r.src
                WHERE r.dst IN ({ph})
                  AND r.kind IN ('WRITES', 'DERIVES')
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                  AND src.kind NOT IN ('INPUT', 'OUTPUT')
                """,
                fids,
            ).fetchall()
            reads = conn.execute(
                f"""
                SELECT src.name, src.file, src.line_start, src.kind
                FROM relation r
                JOIN entity src ON src.id = r.src
                WHERE r.dst IN ({ph})
                  AND r.kind IN ('READS', 'CALLS_UNDER_GUARD')
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                """,
                fids,
            ).fetchall()
            binds = conn.execute(
                f"""
                SELECT e.name
                FROM relation r
                JOIN entity e ON e.id = CASE WHEN r.src IN ({ph}) THEN r.dst ELSE r.src END
                WHERE (r.src IN ({ph}) OR r.dst IN ({ph}))
                  AND r.kind IN ('BINDS', 'MATERIALIZES_AS')
                  AND LOWER(IFNULL(r.status, '')) IN ('confirmed', 'extracted', 'verified')
                """,
                (*fids, *fids, *fids),
            ).fetchall()
        seen_p: set[str] = set()
        seen_c: set[str] = set()
        for row in writes:
            name = str(row[0] or "")
            file = str(row[1] or "")
            if not name or _is_noise_name_sql(name) or name in seen_p:
                continue
            blob = file.replace("\\", "/")
            if "/op_kernel/" in blob or blob.startswith("op_kernel/"):
                continue
            seen_p.add(name)
            if not file:
                unresolved_p += 1
            producers.append({"name": name, "file": _norm_file(file), "line": int(row[2] or 0)})
        for row in reads:
            name = str(row[0] or "")
            file = str(row[1] or "")
            if not name or _is_noise_name_sql(name) or name in seen_c:
                continue
            blob = file.replace("\\", "/")
            if not ("/op_kernel/" in blob or blob.startswith("op_kernel/")):
                continue
            seen_c.add(name)
            if not file:
                unresolved_c += 1
            consumers.append({"name": name, "file": _norm_file(file), "line": int(row[2] or 0)})
        seen_t: set[str] = set()
        for row in binds:
            name = str(row[0] or "")
            if not name or _is_noise_name_sql(name) or name in seen_t:
                continue
            seen_t.add(name)
            transport.append(name)
        if not producers and not consumers and not transport:
            return None
        return {
            "producers": producers,
            "consumers": consumers,
            "transport": transport,
            "coverage": {
                "producers_confirmed": len(producers),
                "producers_unresolved": unresolved_p,
                "consumers_confirmed": len(consumers),
                "consumers_unresolved": unresolved_c,
                "exhaustive": False,
            },
        }

    def calls_for(self, entity_id: str) -> dict[str, Any] | None:
        """Confirmed Calls / Called by, plus partial incoming as Possible callers.

        External (fileless) neighbors are dropped; they are API shadows, not
        operator source. Unresolved-call stubs stay off the card.
        """
        eid = str(entity_id or "")
        if not eid:
            return None
        if eid in self._calls_for_cache:
            return self._calls_for_cache[eid]
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.status, r.src, r.dst, r.data, r.kind AS r_kind,
                       s.kind AS s_kind, s.name AS s_name, s.file AS s_file,
                       s.line_start AS s_line, s.line_end AS s_end, s.data AS s_data,
                       d.kind AS d_kind, d.name AS d_name, d.file AS d_file,
                       d.line_start AS d_line, d.data AS d_data
                FROM relation r
                JOIN entity s ON s.id = r.src
                JOIN entity d ON d.id = r.dst
                WHERE r.kind IN ('CALLS', 'CALLS_UNDER_GUARD') AND (r.src = ? OR r.dst = ?)
                """,
                (eid, eid),
            ).fetchall()

        def _located(file: Any, line: Any) -> bool:
            return bool(str(file or "").strip()) and int(line or 0) > 0

        def _unresolved_stub(data: Any) -> bool:
            parsed = _parse_data(data) if not isinstance(data, dict) else data
            return bool(parsed.get("internal_unresolved"))

        def _site_rows(rel_data: dict[str, Any], file: str, line: int) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            raw_sites = rel_data.get("sites") if isinstance(rel_data.get("sites"), list) else []
            for site in raw_sites:
                if not isinstance(site, dict):
                    continue
                sfile = _norm_file(str(site.get("file") or file or ""))
                sline = int(site.get("line") or 0)
                if sline <= 0:
                    continue
                item = {"file": sfile, "line": sline}
                guard = str(site.get("guard") or "").strip()
                if guard:
                    item["guard"] = guard
                out.append(item)
            if out:
                return out
            # The relation records where the call is written; the entity records
            # where the caller is declared. Falling straight through to the
            # entity printed "Called by Entry::Run @50" for a call on line 61,
            # and sent the reader to the signature instead of the site.
            rel_file = str(rel_data.get("file") or "")
            rel_line = int(rel_data.get("line") or 0)
            if _located(rel_file, rel_line):
                out.append({"file": _norm_file(rel_file), "line": rel_line})
            elif _located(file, line):
                out.append({"file": _norm_file(file), "line": int(line or 0)})
            return out

        outgoing: dict[str, dict[str, Any]] = {}
        incoming: dict[str, dict[str, Any]] = {}
        possible: dict[str, dict[str, Any]] = {}

        def _accumulate(
            store: dict[str, dict[str, Any]],
            *,
            nid: str,
            kind: str,
            name: str,
            file: str,
            line: int,
            rel_data: dict[str, Any],
            end: int = 0,
        ) -> None:
            if not name or _is_noise_name_sql(name):
                return
            leaf = _last_ident(name)
            item = store.get(leaf)
            if item is None:
                item = {
                    "id": nid,
                    "kind": kind,
                    "name": name,
                    "file": _norm_file(file),
                    "line": int(line or 0),
                    "sites": [],
                    "when": [],
                }
                store[leaf] = item
            sites = item["sites"]
            for site in _site_rows(rel_data, file, line):
                key = (str(site.get("file") or ""), int(site.get("line") or 0))
                if any((str(x.get("file") or ""), int(x.get("line") or 0)) == key for x in sites):
                    continue
                if (
                    end > int(line or 0) > 0
                    and kind != EntityKind.MACRO.value
                    and _file_same(str(site.get("file") or ""), file)
                    and not (int(line or 0) <= key[1] <= end)
                ):
                    # The site has to lie inside the body of the function being
                    # credited with it. A call on line 773 attributed to a
                    # function that ends on 763 names a caller that cannot be
                    # one, and the reader who opens the file finds a stranger.
                    #
                    # Only a real span can say that, so an entity whose end
                    # equals its start is treated as unmeasured rather than as
                    # one line long. A macro is exempt outright: its sites are
                    # the expansions, which is the whole point of listing them.
                    item["rejected"] = int(item.get("rejected") or 0) + 1
                    continue
                sites.append(site)
                guard = str(site.get("guard") or "").strip()
                if guard and guard not in item["when"]:
                    item["when"].append(guard)
            if not item["line"] and sites:
                item["line"] = int(sites[0].get("line") or 0)
                item["file"] = str(sites[0].get("file") or item["file"])

        for row in rows:
            status = str(_row_get(row, "r.status") or _row_get(row, "status") or "").lower()
            src = str(_row_get(row, "r.src") or _row_get(row, "src") or "")
            dst = str(_row_get(row, "r.dst") or _row_get(row, "dst") or "")
            rel_data = _parse_data(_row_get(row, "r.data") or _row_get(row, "data"))
            s_kind = str(_row_get(row, "s_kind") or "")
            s_name = str(_row_get(row, "s_name") or "")
            s_file = str(_row_get(row, "s_file") or "")
            s_line = int(_row_get(row, "s_line") or 0)
            s_end = int(_row_get(row, "s_end") or 0)
            d_kind = str(_row_get(row, "d_kind") or "")
            d_name = str(_row_get(row, "d_name") or "")
            d_file = str(_row_get(row, "d_file") or "")
            d_line = int(_row_get(row, "d_line") or 0)
            s_data = _row_get(row, "s_data")
            d_data = _row_get(row, "d_data")
            confirmed = status in {"confirmed", "extracted", "verified"}
            dst_sites = _site_rows(rel_data, d_file, d_line)
            src_sites = _site_rows(rel_data, s_file, s_line)
            if src == eid:
                if _unresolved_stub(d_data):
                    continue
                if not _located(d_file, d_line) and not dst_sites:
                    continue
                if confirmed:
                    _accumulate(
                        outgoing,
                        nid=dst,
                        kind=d_kind,
                        name=d_name,
                        file=d_file,
                        line=d_line,
                        rel_data=rel_data,
                    )
            elif dst == eid:
                if _unresolved_stub(s_data):
                    continue
                if not _located(s_file, s_line) and not src_sites:
                    continue
                if s_kind == EntityKind.BRANCH.value:
                    # A branch is a condition on a call, not the thing making
                    # it. Its guard reaches the callers through the per-site
                    # lookup below, which knows which sites it actually encloses.
                    continue
                if not confirmed and s_kind not in _CALLER_KINDS:
                    # A queue member or a TPipe cannot call anything. Offering
                    # one as a candidate caller sends the reader to a line that
                    # holds a declaration, and the guess costs more than the
                    # empty list it replaced.
                    continue
                target = incoming if confirmed else possible
                _accumulate(
                    target,
                    nid=src,
                    kind=s_kind,
                    name=s_name,
                    file=s_file,
                    line=s_line,
                    end=s_end,
                    rel_data=rel_data,
                )

        # A caller whose every site was rejected as out-of-body has nothing
        # left to point at, and keeping the row would fall back to printing its
        # declaration line — which is how the bad site got in front of a reader
        # in the first place, just with a different number on it.
        for store in (outgoing, incoming, possible):
            for leaf in [
                key
                for key, item in store.items()
                if item.get("rejected") and not item.get("sites")
            ]:
                store.pop(leaf, None)
            for item in store.values():
                item.pop("rejected", None)

        # Every call site sits under whatever branch encloses that site, so the
        # condition has to be resolved one site at a time. Appending each guard
        # seen on any incoming edge to every caller put the `then` condition on
        # a call living in the matching `else` and stated the opposite of what
        # the code does. An ambiguous dispatch gets the same treatment: listing
        # candidates without their condition reads as "any of these, always",
        # which is the reverse of what a template fork means.
        for store in (incoming, possible):
            for item in store.values():
                for site in item.get("sites") or []:
                    if not isinstance(site, dict):
                        continue
                    found = [str(site.get("guard") or "").strip()]
                    found.extend(
                        self._branch_guards_at(
                            str(site.get("file") or ""), int(site.get("line") or 0)
                        )
                    )
                    kept: list[str] = []
                    for guard in found:
                        if guard and guard not in kept:
                            kept.append(guard)
                    if not kept:
                        continue
                    site["guard"] = _join_guards(kept[:3])
                    for guard in kept:
                        if guard not in item["when"]:
                            item["when"].append(guard)

        family = self._attach_virtual_dispatch(eid, outgoing, incoming, possible)
        result: dict[str, Any] | None = None
        if outgoing or incoming or possible:
            result = {
                "calls": list(outgoing.values())[:12],
                "called_by": list(incoming.values())[:12],
                "possible_callers": list(possible.values())[:12],
                "calls_total": len(outgoing),
                "called_by_total": len(incoming),
                "possible_callers_total": len(possible),
            }
            if family:
                result["virtual_dispatch"] = family
        if len(self._calls_for_cache) < 2048:
            self._calls_for_cache[eid] = result
        return result

    _REACH_CALLERS_MAX = 2

    def virtual_family(self, leaf: str) -> dict[str, Any] | None:
        """Empty virtual base + override for this short name, if both exist."""
        key = _last_ident(leaf).lower()
        if not key:
            return None
        if key in self._virtual_family_cache:
            return self._virtual_family_cache[key]
        family = self._virtual_family_uncached(key)
        self._virtual_family_cache[key] = family
        return family

    def _virtual_family_uncached(self, leaf: str) -> dict[str, Any] | None:
        from ascendc_codemap_mcp.engine.query.virtual_dispatch import build_family

        with self._connect() as conn:
            rows = self._callable_rows_for_leaf(conn, leaf)
            if len(rows) < 2:
                return None
            owners = self._type_owner_by_file(conn)
            members: list[dict[str, Any]] = []
            texts: dict[str, str] = {}
            for row in rows:
                file = _norm_file(str(row["file"] or ""))
                line = int(row["line_start"] or 0)
                data = _parse_data(row["data"])
                window = _source_line_rows(conn, file, line, line + 2)
                text = "\n".join(str(t) for _n, t in window)
                texts[f"{file}:{line}"] = text
                stored = data.get("virtual_dispatch") if isinstance(data.get("virtual_dispatch"), dict) else None
                if stored and stored.get("base") and stored.get("overrides"):
                    return stored
                members.append(
                    {
                        "id": str(row["id"] or ""),
                        "kind": str(row["kind"] or ""),
                        "name": str(row["name"] or ""),
                        "file": file,
                        "line": line,
                        "line_start": line,
                        "line_end": int(row["line_end"] or 0),
                        "attrs": data,
                        "text": text,
                    }
                )
        return build_family(members, texts=texts, owners_by_file=owners)

    def _callable_rows_for_leaf(self, conn: sqlite3.Connection, leaf: str) -> list[sqlite3.Row]:
        token = str(leaf or "").strip()
        if not token:
            return []
        if self._accel_ready(conn):
            rows = conn.execute(
                """
                SELECT e.id, e.kind, e.name, e.file, e.line_start, e.line_end, e.data
                FROM entity_name_leaf n
                JOIN entity e ON e.id = n.entity_id
                WHERE n.leaf = ? AND n.is_ascendc = 0
                  AND e.kind IN ('FUNCTION', 'METHOD')
                """,
                (token,),
            ).fetchall()
            if rows:
                return rows
        clause, params = _leaf_name_where(token)
        return conn.execute(
            f"""
            SELECT e.id, e.kind, e.name, e.file, e.line_start, e.line_end, e.data
            FROM entity e
            WHERE e.kind IN ('FUNCTION', 'METHOD') AND {clause}
            """,
            params,
        ).fetchall()

    def _type_owner_by_file(self, conn: sqlite3.Connection) -> dict[str, str]:
        rows = conn.execute(
            "SELECT name, file, data FROM entity WHERE kind = 'TYPE' AND file != ''"
        ).fetchall()
        by_file: dict[str, list[str]] = {}
        for row in rows:
            name = _last_ident(str(row["name"] or ""))
            file = _norm_file(str(row["file"] or ""))
            if not name or not file:
                continue
            data = _parse_data(row["data"])
            kind = str(data.get("cpp_kind") or "").lower()
            if kind and kind not in {"class", "struct"}:
                continue
            by_file.setdefault(file, []).append(name)
        return {file: names[0] for file, names in by_file.items() if len(names) == 1}

    def _attach_virtual_dispatch(
        self,
        entity_id: str,
        outgoing: dict[str, dict[str, Any]],
        incoming: dict[str, dict[str, Any]],
        possible: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        subject_leaf = ""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM entity WHERE id = ?", (entity_id,)
            ).fetchone()
        if row is not None:
            subject_leaf = _last_ident(str(row["name"] or ""))
        subject_family = self.virtual_family(subject_leaf) if subject_leaf else None
        if subject_family:
            self._merge_virtual_sibling_callers(
                entity_id, subject_family, incoming, possible
            )
        for item in outgoing.values():
            fam = self.virtual_family(str(item.get("name") or ""))
            if fam:
                item["virtual_dispatch"] = fam
        return subject_family

    def _merge_virtual_sibling_callers(
        self,
        entity_id: str,
        family: dict[str, Any],
        incoming: dict[str, dict[str, Any]],
        possible: dict[str, dict[str, Any]],
    ) -> None:
        """A call bound to the override is also a caller of the empty virtual."""
        sibling_ids = [
            str(row.get("id") or "")
            for row in list(family.get("members") or [])
            + list(family.get("base") or [])
            + list(family.get("overrides") or [])
            if str(row.get("id") or "") and str(row.get("id") or "") != entity_id
        ]
        sibling_ids = list(dict.fromkeys(sibling_ids))
        if not sibling_ids:
            return
        ph = ",".join("?" for _ in sibling_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT r.status, r.src, r.data,
                       s.kind AS s_kind, s.name AS s_name, s.file AS s_file,
                       s.line_start AS s_line, s.line_end AS s_end, s.data AS s_data
                FROM relation r
                JOIN entity s ON s.id = r.src
                WHERE r.kind IN ('CALLS', 'CALLS_UNDER_GUARD')
                  AND r.dst IN ({ph})
                  AND s.kind IN ('FUNCTION', 'METHOD')
                """,
                sibling_ids,
            ).fetchall()
        for row in rows:
            status = str(row["status"] or "").lower()
            src = str(row["src"] or "")
            if src == entity_id:
                continue
            rel_data = _parse_data(row["data"])
            s_kind = str(row["s_kind"] or "")
            s_name = str(row["s_name"] or "")
            s_file = str(row["s_file"] or "")
            s_line = int(row["s_line"] or 0)
            s_end = int(row["s_end"] or 0)
            if _is_noise_name_sql(s_name):
                continue
            if not str(s_file or "").strip() or s_line <= 0:
                continue
            target = incoming if status in {"confirmed", "extracted", "verified"} else possible
            leaf = _last_ident(s_name)
            item = target.get(leaf)
            if item is None:
                item = {
                    "id": src,
                    "kind": s_kind,
                    "name": s_name,
                    "file": _norm_file(s_file),
                    "line": s_line,
                    "sites": [],
                    "when": [],
                }
                target[leaf] = item
            for site in rel_data.get("sites") or []:
                if not isinstance(site, dict):
                    continue
                sfile = _norm_file(str(site.get("file") or s_file))
                sline = int(site.get("line") or 0)
                if sline <= 0:
                    continue
                if any(
                    str(x.get("file") or "") == sfile and int(x.get("line") or 0) == sline
                    for x in item["sites"]
                ):
                    continue
                if s_end > s_line > 0 and not (s_line <= sline <= s_end):
                    continue
                entry = {"file": sfile, "line": sline}
                guard = str(site.get("guard") or "").strip()
                if guard:
                    entry["guard"] = guard
                    if guard not in item["when"]:
                        item["when"].append(guard)
                item["sites"].append(entry)
            if not item["sites"]:
                rel_file = _norm_file(str(rel_data.get("file") or s_file))
                rel_line = int(rel_data.get("line") or s_line or 0)
                if rel_line > 0:
                    item["sites"].append({"file": rel_file, "line": rel_line})

    def writer_reach(self, names: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
        """Who calls each of these functions, and on which line.

        Knowing that ``SetSplitAxis:1693`` writes false is half an answer: what
        settles who overrides whom is that it runs after ``SetSplitAxis:1673``
        and before ``DoSparse:1099``, and that ordering lives on the call sites,
        not on the writes. Recovering it cost the agent several extra round
        trips per flag, so the bundle carries it now.
        """
        wanted = {_last_ident(str(n or "")) for n in names or ()}
        wanted.discard("")
        if not wanted:
            return {}
        cache_key = frozenset(w.lower() for w in wanted)
        with self._reach_lock:
            hit = self._reach_cache.get(cache_key)
            if hit is not None:
                return hit
        out = self._writer_reach_uncached(wanted)
        with self._reach_lock:
            self._reach_cache[cache_key] = out
        return out

    def _writer_reach_uncached(self, wanted: set[str]) -> dict[str, list[dict[str, Any]]]:
        """Incoming CALLS for a handful of callees — never the whole call graph.

        The first version selected every CALLS row and filtered in Python. On
        FAG that is thousands of edges and ~70ms, paid once per bundle, three
        times on a source card. The dst index exists specifically for this.
        """
        with self._connect() as conn:
            ids = self._function_ids_for_leaves(conn, wanted)
            if not ids:
                return {}
            ph = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"""
                SELECT d.name AS callee, s.name AS caller,
                       s.line_start AS s_line, s.line_end AS s_end, r.data AS r_data
                FROM relation r
                JOIN entity s ON s.id = r.src
                JOIN entity d ON d.id = r.dst
                WHERE r.kind IN ('CALLS', 'CALLS_UNDER_GUARD')
                  AND r.dst IN ({ph})
                  AND s.kind IN ('FUNCTION', 'METHOD')
                """,
                ids,
            ).fetchall()
        out: dict[str, list[dict[str, Any]]] = {}
        seen: set[tuple[str, str, int]] = set()
        wanted_l = {w.lower() for w in wanted}
        for row in rows:
            callee = _last_ident(str(row["callee"] or ""))
            if callee.lower() not in wanted_l:
                continue
            caller = _last_ident(str(row["caller"] or ""))
            if not caller or caller == callee or _is_noise_name_sql(caller):
                continue
            data = _parse_data(row["r_data"])
            start = int(row["s_line"] or 0)
            end = int(row["s_end"] or 0)
            for site in self._call_site_lines(data):
                if end > start > 0 and not (start <= site <= end):
                    continue
                key = (callee, caller, site)
                if key in seen:
                    continue
                seen.add(key)
                out.setdefault(callee, []).append({"caller": caller, "line": site})
        for callee, hits in out.items():
            hits.sort(key=lambda h: int(h.get("line") or 0))
            out[callee] = hits[: self._REACH_CALLERS_MAX]
        return out

    def _function_ids_for_leaves(self, conn: sqlite3.Connection, leaves: set[str]) -> list[str]:
        tokens = [x.lower() for x in leaves if x]
        if not tokens:
            return []
        if self._accel_ready(conn):
            ph = ",".join("?" for _ in tokens)
            rows = conn.execute(
                f"""
                SELECT DISTINCT n.entity_id
                FROM entity_name_leaf n
                JOIN entity e ON e.id = n.entity_id
                WHERE n.leaf IN ({ph}) AND n.is_ascendc = 0
                  AND e.kind IN ('FUNCTION', 'METHOD')
                """,
                tokens,
            ).fetchall()
            return [str(r[0]) for r in rows if r[0]]
        ids: list[str] = []
        for leaf in tokens:
            clause, params = _leaf_name_where(leaf)
            rows = conn.execute(
                f"""
                SELECT e.id FROM entity e
                WHERE e.kind IN ('FUNCTION', 'METHOD') AND {clause}
                """,
                params,
            ).fetchall()
            ids.extend(str(r[0]) for r in rows if r[0])
        return list(dict.fromkeys(ids))

    @staticmethod
    def _call_site_lines(data: dict[str, Any]) -> list[int]:
        sites = data.get("sites") if isinstance(data.get("sites"), list) else []
        lines = [
            int(s.get("line") or 0)
            for s in sites
            if isinstance(s, dict) and int(s.get("line") or 0) > 0
        ]
        if lines:
            return lines
        one = int(data.get("line") or 0)
        return [one] if one > 0 else []

    _UNIT_FIELD_CAP = 3

    def _field_leaves_among(self, names: list[str]) -> list[str]:
        wanted = {_last_ident(str(n)) for n in names if n}
        wanted.discard("")
        if not wanted:
            return []
        kinds = (EntityKind.TILING_FIELD.value, EntityKind.FIELD.value)
        found: list[str] = []
        seen: set[str] = set()
        tokens = sorted(wanted)
        with self._connect() as conn:
            if self._accel_ready(conn):
                ph = ",".join("?" for _ in tokens)
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT n.leaf
                    FROM entity_name_leaf n
                    JOIN entity e ON e.id = n.entity_id
                    WHERE n.leaf IN ({ph}) AND e.kind IN (?, ?)
                    """,
                    (*tokens, *kinds),
                ).fetchall()
                for row in rows:
                    leaf = str(row[0] or "")
                    if leaf and leaf not in seen:
                        seen.add(leaf)
                        found.append(leaf)
                return found
            for leaf in tokens:
                clause, params = _leaf_name_where(leaf)
                row = conn.execute(
                    f"SELECT 1 FROM entity e WHERE e.kind IN (?, ?) AND {clause} LIMIT 1",
                    (*kinds, *params),
                ).fetchone()
                if row and leaf not in seen:
                    seen.add(leaf)
                    found.append(leaf)
        return found

    def _unit_field_relevance(self, payload: dict[str, Any]) -> dict[str, tuple[int, int]]:
        """Rank fields by what this site does to them, not by how often the
        enclosing function happens to name them.

        `resolve(file, line)` means "I am looking at this line". Token frequency
        answers a different question and reliably loses to whatever the function
        mentions most, so the line's own WRITES / DERIVES targets come first and
        frequency is only the tie-break of last resort.
        """
        loc = int(payload.get("line") or 0)
        rank: dict[str, tuple[int, int]] = {}

        def offer(name: str, tier: int, distance: int) -> None:
            leaf = _last_ident(str(name or "").replace(".", "::"))
            if not leaf or leaf in _FOCUS_SKIP_IDENTS or len(leaf) < 3:
                return
            key = (tier, distance)
            if leaf not in rank or key < rank[leaf]:
                rank[leaf] = key

        # Tier 0/2: what the graph records as written here, nearest line first.
        for group in payload.get("state_changes") or []:
            if not isinstance(group, dict):
                continue
            name = str(group.get("name") or "")
            for site in group.get("sites") or []:
                if not isinstance(site, dict):
                    continue
                line = int(site.get("line") or 0)
                distance = abs(line - loc) if line and loc else 9999
                offer(name, 0 if distance == 0 else 2, distance)

        # Tier 1: entities the seed scan anchored to this exact line.
        for row in payload.get("cards") or payload.get("seeds") or []:
            if isinstance(row, dict):
                offer(str(row.get("name") or ""), 1, 0)

        # Tier 1: an explicit highlight is the caller saying what it came for.
        offer(
            str(payload.get("highlight") or payload.get("explore_pattern") or ""),
            1,
            0,
        )
        return rank

    def _site_window_field_bundles(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """In-window writes only — no kernel-consumer expansion."""
        file = str(payload.get("file") or "")
        out: list[dict[str, Any]] = []
        for group in payload.get("state_changes") or []:
            if not isinstance(group, dict):
                continue
            name = str(group.get("name") or "")
            if not name:
                continue
            sites: list[dict[str, Any]] = []
            for site in group.get("sites") or []:
                if not isinstance(site, dict):
                    continue
                sites.append(
                    {
                        "file": file,
                        "line": int(site.get("line") or 0),
                        "rhs": str(site.get("rhs") or ""),
                        "function": str(site.get("writer") or ""),
                    }
                )
            if not sites:
                continue
            out.append({"name": name, "bundle": {"host_value_definitions": sites}})
            if len(out) >= self._UNIT_FIELD_CAP:
                break
        return out

    def _unit_field_bundles(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if str(payload.get("resolve_mode") or "") == "site":
            # In-window WRITES first. A toy snapshot may have the field entity
            # and an assignment in the snippet without a WRITES edge; still
            # name those leaves, without walking kernel consumers.
            out = self._site_window_field_bundles(payload)
            have = {_last_ident(str(row.get("name") or "")) for row in out}
            snippet = str(payload.get("snippet") or "")
            counts: Counter[str] = Counter()
            for tok in _TOKEN_RE.findall(snippet):
                if tok in _FOCUS_SKIP_IDENTS or len(tok) < 3:
                    continue
                counts[tok] += 1
            relevance = self._unit_field_relevance(payload)
            extra = [
                name
                for name in self._field_leaves_among(
                    sorted(set(counts) | set(relevance))
                )
                if name not in have
            ]
            loc = int(payload.get("line") or 0)
            file = str(payload.get("file") or "")
            for name in extra:
                out.append(
                    {
                        "name": name,
                        "bundle": {
                            "host_value_definitions": [
                                {"file": file, "line": loc}
                            ]
                        },
                    }
                )
                if len(out) >= self._UNIT_FIELD_CAP:
                    break
            return out
        snippet = str(payload.get("snippet") or "")
        counts: Counter[str] = Counter()
        for tok in _TOKEN_RE.findall(snippet):
            if tok in _FOCUS_SKIP_IDENTS or len(tok) < 3:
                continue
            counts[tok] += 1
        relevance = self._unit_field_relevance(payload)
        candidates = set(counts) | set(relevance)
        if not candidates:
            return []
        leaves = self._field_leaves_among(sorted(candidates))
        ranked = sorted(
            leaves,
            key=lambda n: (
                relevance.get(n, (4, 9999)),
                -counts.get(n, 0),
                n.lower(),
            ),
        )
        from ascendc_codemap_mcp.engine.query.bundle import build_symbol_bundle

        out: list[dict[str, Any]] = []
        for name in ranked:
            bundle = build_symbol_bundle(self, name)
            if not isinstance(bundle, dict):
                continue
            if not (
                bundle.get("host_value_definitions")
                or bundle.get("kernel_consumers")
                or bundle.get("assignments")
                or bundle.get("transport")
            ):
                continue
            out.append({"name": name, "bundle": bundle})
            if len(out) >= self._UNIT_FIELD_CAP:
                break
        return out

    def attach_card_facets(self, payload: dict[str, Any]) -> dict[str, Any]:
        cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
        enclosing = payload.get("enclosing")
        site = str(payload.get("resolve_mode") or "") == "site"
        if site:
            # Call facts belong to the enclosing function, never to the
            # CONTRACT/BRANCH that happens to start on the requested line.
            enc = enclosing if isinstance(enclosing, dict) else {}
            eid = str(enc.get("id") or "")
            kind = str(enc.get("kind") or "").upper()
            if eid and kind in _DEF_CARD_KINDS:
                calls = self.calls_for(eid)
                payload["calls_computed"] = True
                if calls:
                    payload["calls"] = calls
                    facets = enc.get("facets") if isinstance(enc.get("facets"), dict) else {}
                    facets["calls"] = calls
                    enc["facets"] = facets
                    payload["enclosing"] = enc
            else:
                payload["calls_computed"] = False
            if not payload.get("field_bundles"):
                payload["field_bundles"] = self._unit_field_bundles(payload)
            return payload
        if isinstance(enclosing, dict):
            cards = [*cards, enclosing]
        for card in cards:
            eid = str(card.get("id") or card.get("_entity_id") or "")
            if not eid:
                continue
            calls = self.calls_for(eid)
            if calls:
                facets = card.get("facets") if isinstance(card.get("facets"), dict) else {}
                facets["calls"] = calls
                card["facets"] = facets
                if payload.get("calls") is None:
                    payload["calls"] = calls
        for card in cards:
            eid = str(card.get("id") or card.get("_entity_id") or "")
            if not eid:
                continue
            facets = card.get("facets") if isinstance(card.get("facets"), dict) else {}
            storage = self._storage_facet(eid)
            if storage:
                facets["storage"] = storage
            controls = self._controls_facet(eid)
            if controls:
                facets["controls"] = controls
            memory = self._memory_facet(eid)
            if memory:
                facets["memory"] = memory
            used = self._used_by_facet(eid)
            if used:
                facets["used_by"] = used
            if facets:
                card["facets"] = facets
            extra = None
            try:
                from ascendc_codemap_mcp.engine.query.bundle import attach_entity_projections

                extra = attach_entity_projections(self, card)
            except Exception:  # noqa: BLE001
                extra = None
            if extra:
                facets = card.get("facets") if isinstance(card.get("facets"), dict) else {}
                if extra.get("resource"):
                    facets["resource"] = extra["resource"]
                if extra.get("controls_proj"):
                    facets["controls_proj"] = extra["controls_proj"]
                card["facets"] = facets
        seed = ""
        for key in ("pattern", "explore_pattern"):
            seed = str(payload.get(key) or "").strip()
            if seed:
                break
        if not seed:
            primary = cards[0] if cards else {}
            seed = str(primary.get("name") or "")
        if ":" in seed and any(ch.isdigit() for ch in seed):
            seed = str((cards[0] if cards else {}).get("name") or "")
        # A definition card and a site card answer the same question about the
        # same lines; only the site card carried the pipeline profile and the
        # resources declared there, so asking by name forced a second call by
        # file+line to get either one.
        primary_card = cards[0] if cards else {}
        primary_kind = str(primary_card.get("kind") or "").upper()
        if primary_kind in _UNIT_SNIPPET_KINDS:
            span_start = int(primary_card.get("line") or 0)
            span_end = int(primary_card.get("line_end") or 0)
            if span_start > 0 and span_end >= span_start:
                span_file = str(primary_card.get("file") or "")
                # A class declares its buffers; a function runs a pipeline. Both
                # are read by name, so neither should need a second call.
                resources = self._unit_resources(span_file, span_start, span_end)
                if resources:
                    payload["unit_resources"] = resources
                if primary_kind in _DEF_CARD_KINDS:
                    operations = self._operations_in_span(span_file, span_start, span_end)
                    if operations:
                        payload["operations"] = operations
                    delegated = self._delegated_operations(
                        str(primary_card.get("id") or primary_card.get("_entity_id") or "")
                    )
                    if delegated:
                        payload["delegated_operations"] = delegated
        # A TilingData name usually also exists as a plain TYPE; whichever card
        # ranks first, the ABI question is the same.
        for card in cards:
            if str(card.get("kind") or "") in {
                EntityKind.TILING_DATA.value,
                EntityKind.TYPE.value,
            }:
                fields = self._tiling_data_fields(str(card.get("name") or ""))
                if fields:
                    payload["tiling_data_fields"] = fields
                    break
        if seed:
            from ascendc_codemap_mcp.engine.query.bundle import (
                attach_entity_projections,
                build_symbol_bundle,
            )

            bundle = build_symbol_bundle(self, seed)
            primary = cards[0] if cards else None
            if bundle is None and isinstance(primary, dict):
                extra = attach_entity_projections(self, primary)
                resource = extra.get("resource") if extra else None
                controls = extra.get("controls_proj") if extra else None
                layout = (resource or {}).get("workspace_layout") or []
                if resource or controls or layout:
                    bundle = {
                        "host_value_definitions": [],
                        "transport": [],
                        "kernel_consumers": [],
                        "assignments": [],
                        "consumed_by": [],
                        "workspace_layout": layout,
                        "resource": resource or {},
                        "controls": controls or {},
                    }
            if bundle:
                payload["bundle"] = bundle
                if isinstance(primary, dict):
                    facets = primary.get("facets") if isinstance(primary.get("facets"), dict) else {}
                    facets["bundle"] = bundle
                    primary["facets"] = facets
            assignments = self.assignments_for(seed)
            if assignments:
                payload["assignments"] = assignments
                if isinstance(primary, dict):
                    facets = primary.get("facets") if isinstance(primary.get("facets"), dict) else {}
                    facets["assignments"] = assignments
                    primary["facets"] = facets
            hk = self.host_kernel_for(seed)
            if hk:
                payload["host_kernel"] = hk
                if isinstance(primary, dict):
                    facets = primary.get("facets") if isinstance(primary.get("facets"), dict) else {}
                    facets["host_kernel"] = hk
                    primary["facets"] = facets
            if not payload.get("compiled_support"):
                support = self.compiled_support_for(seed)
                if support:
                    payload["compiled_support"] = support
        # Everything above lands after the card was already fitted, so the body
        # had been cut to make room for a budget these blocks were never charged
        # against. Re-fit with the whole card present; the order sheds these
        # first and gives a large function its body back.
        fitted = _fit_payload(payload)
        if fitted is not payload:
            payload.clear()
            payload.update(fitted)
        return payload

    def _confirmed_neighbors(
        self, entity_id: str, kinds: tuple[str, ...], *, direction: str = "any"
    ) -> list[tuple[str, str, str, str, int, dict[str, Any], dict[str, Any]]]:
        """Neighbours across ``kinds``. ``direction`` picks the edge end.

        ``incoming`` keeps edges pointing at the seed, ``outgoing`` those
        leaving it. A section that names a direction has to filter on one, or
        it prints callees under a heading that promises callers.
        """
        if not entity_id or not kinds:
            return []
        ph = ",".join("?" for _ in kinds)
        if direction == "incoming":
            where, args = "r.dst = ?", (entity_id, entity_id)
        elif direction == "outgoing":
            where, args = "r.src = ?", (entity_id, entity_id)
        else:
            where, args = "(r.src = ? OR r.dst = ?)", (entity_id, entity_id, entity_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT r.kind, e.kind, e.name, e.file, e.line_start, e.data, r.data
                FROM relation r
                JOIN entity e ON e.id = CASE WHEN r.src = ? THEN r.dst ELSE r.src END
                WHERE {where} AND r.kind IN ({ph})
                  AND LOWER(IFNULL(r.status,'')) IN ('confirmed','extracted','verified')
                """,
                (*args, *kinds),
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                (
                    str(row[0] or ""),
                    str(row[1] or ""),
                    str(row[2] or ""),
                    str(row[3] or ""),
                    int(row[4] or 0),
                    _parse_data(row[5]),
                    _parse_data(row[6]),
                )
            )
        return out

    def _storage_facet(self, entity_id: str) -> dict[str, Any] | None:
        backed: list[dict[str, Any]] = []
        types: list[str] = []
        for rkind, ekind, name, file, line, edata, _rdata in self._confirmed_neighbors(
            entity_id, ("BACKED_BY", "INSTANCE_OF")
        ):
            if rkind == "BACKED_BY":
                space = str(
                    edata.get("physical_space") or edata.get("memory_space") or ""
                )
                if not space:
                    continue
                backed.append(
                    {
                        "name": name,
                        "kind": ekind,
                        "file": file,
                        "line": line,
                        "physical_space": space,
                    }
                )
            elif rkind == "INSTANCE_OF" and name:
                types.append(name)
        if not backed and not types:
            return None
        return {"backed_by": backed, "instance_of": types}

    def _controls_facet(self, entity_id: str) -> dict[str, Any] | None:
        controls: list[str] = []
        materializes: list[str] = []
        for rkind, _ekind, name, _file, _line, _edata, _rdata in self._confirmed_neighbors(
            entity_id, ("CONTROLS", "MATERIALIZES_AS")
        ):
            if not name or _is_noise_name_sql(name):
                continue
            if rkind == "CONTROLS":
                if name not in controls:
                    controls.append(name)
            elif rkind == "MATERIALIZES_AS":
                if name not in materializes:
                    materializes.append(name)
        if not controls and not materializes:
            return None
        return {"controls": controls, "materializes_as": materializes}

    def _memory_facet(self, entity_id: str) -> dict[str, Any] | None:
        rows = self._confirmed_neighbors(entity_id, ("FLOWS_TO",))
        transfers: list[tuple[str, str]] = []
        unresolved = 0
        total = 0
        for rkind, _ekind, name, _file, _line, edata, rdata in rows:
            if rkind != "FLOWS_TO":
                continue
            if str(rdata.get("via") or "") != "MemoryTransfer":
                continue
            total += 1
            src_space = str(rdata.get("src_space") or rdata.get("from_space") or "")
            dst_space = str(rdata.get("dst_space") or rdata.get("to_space") or "")
            if not src_space:
                src_space = str(edata.get("memory_space") or edata.get("physical_space") or "")
            if not src_space or not dst_space:
                unresolved += 1
                continue
            transfers.append((src_space, dst_space))
        if total == 0:
            return None
        counts: dict[str, int] = {}
        for src, dst in transfers:
            key = f"{src} → {dst}"
            counts[key] = counts.get(key, 0) + 1
        return {
            "resolved": total - unresolved,
            "total": total,
            "unresolved": unresolved,
            "confirmed": total - unresolved,
            "exhaustive": False,
            "flows": counts,
        }

    def _used_by_facet(self, entity_id: str) -> dict[str, int] | None:
        """Entities that consume this one, by name.

        Walking these edges undirected put the seed's own callees in the list,
        so the section read as a caller list with the arrows reversed. Each
        relation is followed the way it points instead: a caller and a reader
        arrive at the seed, while data flows away from it to reach its consumer.
        """
        counts: dict[str, int] = {}
        rows = [
            *(
                (r[0], r[2])
                for r in self._confirmed_neighbors(
                    entity_id, ("CALLS", "READS"), direction="incoming"
                )
            ),
            *(
                (r[0], r[2])
                for r in self._confirmed_neighbors(
                    entity_id, ("FLOWS_TO",), direction="outgoing"
                )
            ),
        ]
        for rkind, name in rows:
            if rkind not in {"CALLS", "READS", "FLOWS_TO"} or not name:
                continue
            if _is_noise_name_sql(name):
                continue
            counts[name] = counts.get(name, 0) + 1
        return counts or None

    def _search_entity_rows(
        self,
        conn: sqlite3.Connection,
        phrase: str,
        *,
        kind: str,
        file_filter: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        from ascendc_codemap_mcp.engine.query.lexicon import entity_haystack
        from ascendc_codemap_mcp.engine.query.rg import (
            compile_search,
            line_matches,
            path_matches,
            rank_hit,
        )

        cre = compile_search(phrase)
        cap = max(1, int(limit))
        start = max(0, int(offset or 0))
        arch = str(self._architecture or "")
        rows = conn.execute(
            "SELECT id, kind, name, file, line_start, data FROM entity WHERE kind = ?",
            (kind,),
        ).fetchall()
        matched: list[dict[str, Any]] = []
        for eid, ekind, name, file, line, data in rows:
            blob = _parse_data(data)
            hay = entity_haystack(str(ekind or ""), str(name or ""), blob)
            if not line_matches(cre, hay):
                continue
            path = str(file or "")
            if not path_matches(path, file_filter):
                continue
            matched.append(
                {
                    "file": path,
                    "line": int(line or 0),
                    "name": str(name or ""),
                    "kind": str(ekind or ""),
                    "text": "",
                    "id": str(eid or ""),
                }
            )
        matched.sort(
            key=lambda r: rank_hit(
                str(r.get("file") or ""),
                int(r.get("line") or 0),
                str(r.get("text") or r.get("name") or ""),
                phrase,
                arch,
            )
        )
        total = len(matched)
        return matched[start : start + cap], total

    def _suggest_lexicon_symbols(
        self,
        conn: sqlite3.Connection,
        phrase: str,
        *,
        file_filter: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        from ascendc_codemap_mcp.engine.query.lexicon import lexicon_tags
        from ascendc_codemap_mcp.engine.query.rg import path_matches

        tags = lexicon_tags(phrase)
        if not tags:
            return []
        from ascendc_codemap_mcp.engine.query.lexicon import entity_haystack

        rows = conn.execute(
            "SELECT kind, name, file, line_start, data FROM entity"
        ).fetchall()
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()
        for alias, tag in tags:
            for ekind, name, file, line, data in rows:
                nm = str(name or "")
                hay = entity_haystack(str(ekind or ""), nm, _parse_data(data))
                if tag not in nm and tag not in hay:
                    continue
                path = str(file or "")
                if not path_matches(path, file_filter):
                    continue
                key = (nm, path, int(line or 0))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "name": nm,
                        "kind": str(ekind or ""),
                        "file": path,
                        "line": int(line or 0),
                        "matched_alias": alias,
                    }
                )
                if len(out) >= max(1, int(limit)):
                    return out
        return out

    def _related_search_tokens(
        self,
        conn: sqlite3.Connection,
        phrase: str,
        *,
        file_filter: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Narrower patterns a zero-hit search could try, with their own counts.

        These are suggestions, never results: each carries the pattern it
        belongs to so a caller can never mistake one for a hit on `phrase`.
        """
        out: list[dict[str, Any]] = []
        for token in _recovery_tokens(phrase):
            try:
                _cards, total, _extra = self._search_regex_lines(
                    conn, token, file_filter=file_filter, limit=1, offset=0
                )
            except Exception:  # noqa: BLE001
                continue
            if total > 0:
                out.append({"pattern": token, "matches": int(total)})
            if len(out) >= max(1, int(limit)):
                break
        out.sort(key=lambda row: (row["matches"], row["pattern"]))
        return out

    _RESOLVABLE_KINDS = (
        EntityKind.TILING_FIELD.value,
        EntityKind.FIELD.value,
        EntityKind.TILING_KEY.value,
        EntityKind.TILING_DATA.value,
        EntityKind.EVENT.value,
        EntityKind.BUFFER.value,
        EntityKind.QUEUE.value,
        EntityKind.REGISTER.value,
        EntityKind.METHOD.value,
        EntityKind.FUNCTION.value,
        EntityKind.KERNEL.value,
    )

    def _resolvable_symbols(
        self, phrase: str, cards: list[dict[str, Any]], *, limit: int = 4
    ) -> list[dict[str, Any]]:
        """Names in this result that already have a semantic card.

        Search answers "where does this text appear"; the expensive part of a
        review is then reconstructing the field's writers, guards and consumers
        by reading each hit. When one of those names is already an entity, the
        answer is one resolve away -- but nothing in the result said so, so the
        reading happened anyway.

        A name reached only through a use site -- a base class whose header is
        outside the tree, a registry method declared elsewhere -- has an entity
        row but no body to show, so resolving it returns callers and nothing
        else. Offering it here reads as a promise of the full picture and one
        reader spent six searches finding out it was not, so require a
        definition before suggesting the call.

        The body may sit under a different kind than the row matched here:
        `CrossCoreSetFlag` suggests off a bodyless FUNCTION but resolves to an
        OPERATION with 122 sites. Ask whether the name has a definition
        anywhere, not whether this particular row carries one.
        """
        seeds: list[str] = []
        ident = _last_ident(str(phrase or ""))
        if ident and len(ident) >= 3:
            seeds.append(ident)
        for card in cards[:12]:
            for token in _TOKEN_RE.findall(str(card.get("text") or ""))[:24]:
                if len(token) >= 4 and token not in _FOCUS_SKIP_IDENTS and token not in seeds:
                    seeds.append(token)
        if not seeds:
            return []
        wanted = seeds[:40]
        # Field entities are named by their access path, so an exact-name join
        # misses the very symbol the caller typed; the leaf index is what makes
        # `fBaseParams.isBn2MultiBlk` findable from `isBn2MultiBlk`.
        leaves = [name.lower() for name in wanted]
        ph_names = ",".join("?" for _ in leaves)
        ph_kinds = ",".join("?" for _ in self._RESOLVABLE_KINDS)
        with self._connect() as conn:
            if self._accel_ready(conn):
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT e.name, e.kind
                    FROM entity_name_leaf l
                    JOIN entity e ON e.id = l.entity_id
                    WHERE l.leaf IN ({ph_names})
                      AND e.kind IN ({ph_kinds})
                    """,
                    (*leaves, *self._RESOLVABLE_KINDS),
                ).fetchall()
                bodied = {
                    str(row[0] or "").lower()
                    for row in conn.execute(
                        f"""
                        SELECT DISTINCT l.leaf
                        FROM entity_name_leaf l
                        JOIN entity e ON e.id = l.entity_id
                        WHERE l.leaf IN ({ph_names})
                          AND IFNULL(e.file, '') <> ''
                        """,
                        leaves,
                    ).fetchall()
                }
            else:
                rows = conn.execute(
                    f"""
                    SELECT name, kind FROM entity
                    WHERE kind IN ({ph_kinds})
                      AND name COLLATE NOCASE IN ({ph_names})
                    """,
                    (*self._RESOLVABLE_KINDS, *wanted),
                ).fetchall()
                bodied = {
                    _last_ident(str(row[0] or "")).lower()
                    for row in conn.execute(
                        f"""
                        SELECT DISTINCT name FROM entity
                        WHERE name COLLATE NOCASE IN ({ph_names})
                          AND IFNULL(file, '') <> ''
                        """,
                        wanted,
                    ).fetchall()
                }
        order = {name.lower(): index for index, name in enumerate(wanted)}
        best: dict[str, set[str]] = {}
        for name, kind in rows:
            leaf = _last_ident(str(name or ""))
            if not leaf or _is_noise_name_sql(str(name or "")):
                continue
            if leaf.lower() not in bodied:
                continue
            best.setdefault(leaf, set()).add(str(kind or ""))
        # A suggestion is only worth a call if it is about what was asked and
        # carries cross-layer facts; a std helper that happened to appear in a
        # hit line is neither.
        core = "".join(ch for ch in str(phrase or "").lower() if ch.isalnum())
        rank = {
            EntityKind.TILING_FIELD.value: 0,
            EntityKind.TILING_KEY.value: 0,
            EntityKind.TILING_DATA.value: 0,
            EntityKind.FIELD.value: 1,
            EntityKind.EVENT.value: 1,
            EntityKind.BUFFER.value: 1,
            EntityKind.QUEUE.value: 1,
            EntityKind.REGISTER.value: 1,
        }
        out = [
            {"symbol": leaf, "kinds": sorted(kinds)}
            for leaf, kinds in best.items()
        ]
        out.sort(
            key=lambda row: (
                0 if core and core in str(row["symbol"]).lower() else 1,
                min((rank.get(k, 2) for k in row["kinds"]), default=2),
                order.get(str(row["symbol"]).lower(), 999),
            )
        )
        return out[: max(1, int(limit))]

    def query_search(
        self, plan: Any, *, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.query.rg import InvalidRegex, compile_search

        phrase = str(getattr(plan, "pattern", "") or getattr(plan, "name", "") or "").strip()
        file_filter = str(getattr(plan, "file", "") or "").strip()
        cap = max(1, int(limit))
        start = max(0, int(offset or 0))
        empty = {
            "ok": True,
            "shape": "search",
            "operation": "search",
            "cards": [],
            "count": 0,
            "returned": 0,
            "total": 0,
            "truncated": False,
            "exhaustive": True,
            "engine_paged": True,
            "next_offset": None,
            "completeness": "",
            "unresolved_reason": "",
            "hint": "",
        }
        if not phrase:
            return empty
        try:
            compile_search(phrase)
        except InvalidRegex as exc:
            return {
                **empty,
                "ok": False,
                "error_code": "INVALID_REGEX",
                "error": str(exc),
                "hint": f"invalid regex: {exc}",
            }
        kind = str(getattr(plan, "kind", "") or "").strip().upper()
        extra: dict[str, Any] = {}
        related: list[dict[str, Any]] = []
        file_miss: list[str] | None = None
        with self._connect() as conn:
            symbols: list[dict[str, Any]] = []
            if kind:
                cards, total = self._search_entity_rows(
                    conn,
                    phrase,
                    kind=kind,
                    file_filter=file_filter,
                    limit=cap,
                    offset=start,
                )
            else:
                cards, total, extra = self._search_regex_lines(
                    conn, phrase, file_filter=file_filter, limit=cap, offset=start
                )
                if total == 0:
                    # A pattern that matched nothing matched nothing. Widening it
                    # here would report someone else's hits under the caller's
                    # pattern, and the caller cannot tell the two apart.
                    related = self._related_search_tokens(
                        conn, phrase, file_filter=file_filter
                    )
                    symbols = self._suggest_lexicon_symbols(
                        conn, phrase, file_filter=file_filter, limit=cap
                    )
            if total == 0 and file_filter:
                file_miss = _glob_nearest(conn, file_filter)
        returned = len(cards)
        nxt = start + returned if start + returned < total else None
        # The rendered answer is the unit grouping, and that is built from every
        # match rather than from this page. Offering a cursor next to it hands
        # back the same view under a new offset, which reads as a stalled pager.
        if nxt is not None and _units_cover_all(extra, total):
            nxt = None
        hint = ""
        resolvable = self._resolvable_symbols(phrase, cards)
        payload = {
            "ok": True,
            "shape": "search",
            "operation": "search",
            "cards": cards,
            "count": returned,
            "returned": returned,
            "total": total,
            "truncated": nxt is not None,
            "exhaustive": nxt is None,
            "engine_paged": True,
            "next_offset": nxt,
            "completeness": "",
            "unresolved_reason": "",
            "hint": hint,
            "symbols": symbols,
        }
        if file_miss is not None:
            payload["file_filter_miss"] = {"glob": file_filter, "nearest": file_miss}
        if related:
            payload["related_patterns"] = related
        if resolvable:
            payload["resolvable_symbols"] = resolvable
        if extra:
            payload["template_lines"] = int(extra.get("template_lines") or 0)
            payload["source_units"] = int(extra.get("source_units") or 0)
            payload["units"] = extra.get("units") or []
            payload["leftover"] = extra.get("leftover") or []
        return payload

    def query_find(self, plan: Any, *, limit: int = 8) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.query.completeness import COMPLETE, UNKNOWN
        from ascendc_codemap_mcp.engine.query.predicate_ast import (
            ast_matches_literal,
            ast_matches_operator,
            ast_matches_symbol,
            ast_matches_value,
        )

        kind = str(getattr(plan, "kind", "") or "")
        name_pattern = str(getattr(plan, "name", "") or "")
        # Name discovery ranks after fetch. A small page ordered by file never
        # sees operator idents that sit behind common headers.
        fetch = max(int(limit) * 8, 64)
        if name_pattern and not kind:
            fetch = max(fetch, 400)
        where: list[str] = []
        params: list[Any] = []
        if name_pattern:
            where.append("e.name COLLATE NOCASE LIKE ?")
            params.append(_name_pattern_to_like(name_pattern))
        layer = str(getattr(plan, "layer", "") or "")
        if layer:
            where.append("IFNULL(json_extract(e.data, '$.layer'), '') = ?")
            params.append(layer)
        function = str(getattr(plan, "function", "") or "")
        if function:
            where.append("IFNULL(json_extract(e.data, '$.function'), '') = ? COLLATE NOCASE")
            params.append(function)
        callee = str(getattr(plan, "callee", "") or "")
        if callee:
            where.append(
                "(e.name = ? COLLATE NOCASE OR IFNULL(json_extract(e.data, '$.callee'), '') = ? COLLATE NOCASE"
                " OR e.name LIKE '%::' || ? COLLATE NOCASE)"
            )
            params.extend([callee, callee, callee])
        dim = str(getattr(plan, "dim", "") or "")
        if dim:
            where.append("e.name = ? COLLATE NOCASE")
            params.append(dim)
        referenced_value = str(getattr(plan, "referenced_value", "") or "")
        if referenced_value:
            where.append(
                "("
                "EXISTS (SELECT 1 FROM json_each(IFNULL(json_extract(e.data, '$.enum_values'), '[]'))"
                "        WHERE value = ? OR value LIKE '%_' || ?)"
                " OR EXISTS (SELECT 1 FROM json_each(IFNULL(json_extract(e.data, '$.literals'), '[]'))"
                "            WHERE CAST(value AS TEXT) = ?)"
                ")"
            )
            params.extend([referenced_value, referenced_value, referenced_value])
        literal = str(getattr(plan, "literal", "") or "")
        if literal:
            lit_norm = literal
            if literal.lstrip("-").isdigit():
                from ascendc_codemap_mcp.engine.query.predicate_ast import _norm_int

                lit_norm = _norm_int(literal)
            where.append(
                "EXISTS (SELECT 1 FROM json_each(IFNULL(json_extract(e.data, '$.literals'), '[]'))"
                "        WHERE CAST(value AS TEXT) IN (?, ?))"
            )
            params.extend([literal, lit_norm])
        operator = str(getattr(plan, "operator", "") or "").upper()
        if operator:
            where.append(
                "EXISTS (SELECT 1 FROM json_each(IFNULL(json_extract(e.data, '$.operators'), '[]'))"
                "        WHERE CAST(value AS TEXT) = ?)"
            )
            params.append(operator)
        referenced_symbol = str(getattr(plan, "referenced_symbol", "") or "")
        if referenced_symbol:
            leaf = referenced_symbol.replace("::", ".").rsplit(".", 1)[-1]
            where.append(
                "EXISTS (SELECT 1 FROM json_each(IFNULL(json_extract(e.data, '$.references'), '[]'))"
                "        WHERE CAST(value AS TEXT) IN (?, ?))"
            )
            params.extend([referenced_symbol, leaf])
        extra = " AND ".join(where) if where else ""
        ast_filters = bool(
            getattr(plan, "referenced_symbol", "")
            or getattr(plan, "referenced_value", "")
            or getattr(plan, "literal", "")
            or getattr(plan, "operator", "")
            or getattr(plan, "entry_role", "")
            or getattr(plan, "consumer_role", "")
        )
        with self._connect() as conn:
            sql_total = (
                self._count_entities(
                    conn, kinds=[kind], extra_where=extra, params=params
                )
                if kind
                else 0
            )
            fetch_n = fetch
            if kind and not ast_filters:
                fetch_n = min(max(int(sql_total), fetch), 2048)
            rows = self._select_entities(
                conn,
                kinds=[kind],
                extra_where=extra,
                params=params,
                limit=fetch_n,
                order="e.file, e.line_start, e.id",
            )
            kept: list[sqlite3.Row] = []
            for row in rows:
                data = _parse_data(_row_get(row, "data", "{}"))
                if not isinstance(data, dict):
                    data = {}
                if getattr(plan, "referenced_symbol", "") and not ast_matches_symbol(
                    data, str(plan.referenced_symbol)
                ):
                    continue
                if getattr(plan, "referenced_value", "") and not ast_matches_value(
                    data, str(plan.referenced_value)
                ):
                    continue
                if getattr(plan, "literal", "") and not ast_matches_literal(data, str(plan.literal)):
                    continue
                if getattr(plan, "operator", "") and not ast_matches_operator(data, str(plan.operator)):
                    continue
                role = str(getattr(plan, "entry_role", "") or "")
                if role and str(data.get("entry_role") or "") != role:
                    continue
                consumer = str(getattr(plan, "consumer_role", "") or "")
                if consumer and str(data.get("consumer_role") or "") != consumer:
                    continue
                kept.append(row)
            # Name discovery answers "what is this called"; it needs no source.
            discovery = bool(name_pattern) and not kind
            hits = self._hits_from_rows(
                conn,
                kept,
                why="find",
                with_snippet=not discovery,
                require_span_for_branch=False,
            )
        if discovery:
            hits = _collapse_by_name(hits)
            rank_pattern = name_pattern
            total_ents, freq = self._ident_frequencies()
            if not _is_glob_pattern(rank_pattern):
                best_q = max(
                    (_match_quality(rank_pattern, str(hit.get("name") or "")) for hit in hits),
                    default=0,
                )
                if best_q >= 100:
                    hits = [
                        hit
                        for hit in hits
                        if _match_quality(rank_pattern, str(hit.get("name") or "")) >= 75
                    ]
            for hit in hits:
                quality = _match_quality(rank_pattern, str(hit.get("name") or ""))
                hit["match"] = _MATCH_LABEL.get(quality, "substring")
            hits.sort(
                key=lambda hit: _name_discovery_key(
                    hit,
                    rank_pattern,
                    architecture=self._architecture,
                    total=total_ents,
                    freq=freq,
                )
            )
            page = _round_robin_by_file(hits, limit=max(len(hits), 1))
            total = len(hits)
            kind_counts = Counter(str(hit.get("kind") or "") or "OTHER" for hit in hits)
        else:
            total = sql_total if (kind and not ast_filters) else len(hits)
            if kind == EntityKind.OPERATION.value:
                page = _round_robin_by_file(hits, limit=max(len(hits), 1))
            else:
                page = hits
        returned = len(page)
        exhaustive = returned >= total and total >= 0
        payload: dict[str, Any] = {
            "ok": bool(page),
            "shape": "find",
            "cards": page,
            "count": returned,
            "returned": returned,
            "total": total,
            "truncated": not exhaustive,
            "exhaustive": exhaustive,
            "completeness": COMPLETE if page else UNKNOWN,
            "unresolved_reason": "" if page else "NO_SEED",
        }
        if kind == EntityKind.OPERATION.value:
            payload["projection"] = "locations"
        if discovery:
            payload["projection"] = "locations"
            payload["kind_groups"] = [
                {"kind": gkind, "count": n}
                for gkind, n in kind_counts.most_common()
                if gkind
            ]
            if page:
                hint = (
                    "name discovery: resolve one of these idents, "
                    "or find kind=... to enumerate its sites"
                )
                leaf = _needle_core(name_pattern)
                exact_op = any(
                    str(hit.get("kind") or "") == EntityKind.OPERATION.value
                    and _last_ident(str(hit.get("name") or "")).lower() == leaf
                    for hit in hits
                )
                if exact_op and leaf:
                    site_total = self.count_call_sites(leaf)
                    sites, site_total = self.list_call_sites(
                        leaf, limit=max(int(site_total), 1)
                    )
                    if sites:
                        payload["cards"] = [
                            {
                                "kind": EntityKind.OPERATION.value,
                                "name": str(row.get("name") or leaf),
                                "file": str(row.get("file") or ""),
                                "line": int(row.get("line") or 0),
                            }
                            for row in sites
                        ]
                        payload["count"] = len(payload["cards"])
                        payload["returned"] = len(payload["cards"])
                        payload["total"] = site_total
                        payload["exhaustive"] = len(payload["cards"]) >= site_total
                        payload["truncated"] = not payload["exhaustive"]
                        payload["operation_sites"] = sites
                        payload["operation_sites_total"] = site_total
                        payload["operation_sites_truncated"] = site_total > len(sites)
                        hint = f"call sites: {site_total}"
                if total > len(page) and not exact_op:
                    hint = "tighten name= or raise limit. " + hint
                payload["hint"] = hint
            else:
                payload["hint"] = (
                    f"no ident {name_pattern}; use search name={name_pattern}"
                )
        if not page:
            if dim:
                payload["dim_names"] = self._compiled_dim_names()
            elif callee:
                payload["hint"] = (
                    f"no OPERATION callee={callee}. "
                    f"If that ident is a function, resolve it. "
                    f"To list kernel API calls, find kind=OPERATION callee=<apiName>. "
                    f"To discover names, find name={callee}"
                )
            elif not discovery:
                payload["dim_names"] = self._compiled_dim_names()
                attach_query_hints(
                    payload, name_pattern or callee or dim or kind, count=0, mode="name"
                )
        return payload

    def query_entry(self, plan: Any, *, limit: int = 8) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.query.completeness import COMPLETE, UNKNOWN

        fetch = max(int(limit) * 8, 64)
        where = ["IFNULL(json_extract(e.data, '$.predicate_role'), '') = 'entry_path'"]
        params: list[Any] = []
        role = str(getattr(plan, "entry_role", "") or "")
        if role:
            where.append("IFNULL(json_extract(e.data, '$.entry_role'), '') = ?")
            params.append(role)
        function = str(getattr(plan, "function", "") or "")
        if function:
            where.append("IFNULL(json_extract(e.data, '$.function'), '') = ? COLLATE NOCASE")
            params.append(function)
        layer = str(getattr(plan, "layer", "") or "")
        if layer:
            where.append("IFNULL(json_extract(e.data, '$.layer'), '') = ?")
            params.append(layer)
        with self._connect() as conn:
            rows = self._select_entities(
                conn,
                kinds=[EntityKind.PREDICATE.value],
                extra_where=" AND ".join(where),
                params=params,
                limit=fetch,
                order="e.file, e.line_start, e.id",
            )
            hits = self._hits_from_rows(conn, rows, why="entry", with_snippet=True)
        if getattr(plan, "referenced_symbol", ""):
            from ascendc_codemap_mcp.engine.query.predicate_ast import ast_matches_symbol

            hits = [
                h
                for h in hits
                if ast_matches_symbol(h.get("facts") or {}, str(plan.referenced_symbol))
            ]
        total = len(hits)
        page = hits[: max(1, int(limit))]
        return {
            "ok": bool(page),
            "shape": "entry",
            "cards": page,
            "count": len(page),
            "total": total,
            "truncated": total > len(page),
            "completeness": COMPLETE if page else UNKNOWN,
            "unresolved_reason": "" if page else "NO_SEED",
        }

    def _split_relation_kinds(self, raw: str) -> tuple[str, ...]:
        """``plan.relation`` is comma-joined kinds; never pass that string to IN ()."""
        parts: list[str] = []
        seen: set[str] = set()
        for token in str(raw or "").replace("|", ",").split(","):
            kind = token.strip().upper()
            if not kind or kind in seen or "," in kind:
                continue
            seen.add(kind)
            parts.append(kind)
        return tuple(parts)

    def _trace_name_is_ident(self, name: str) -> bool:
        leaf = _last_ident(name)
        return bool(_IDENT_NAME_RE.fullmatch(leaf))

    def _directed_trace_walk(
        self,
        conn: Any,
        src_ids: set[str],
        dst_ids: set[str],
        kinds: tuple[str, ...],
        *,
        node_budget: int = _TRACE_FAMILY_NODE_BUDGET,
        max_depth: int = _TRACE_FAMILY_MAX_DEPTH,
        fanout: int = _TRACE_FAMILY_FANOUT,
    ) -> tuple[list[dict[str, Any]], int, bool, bool]:
        """Outgoing BFS. Skip nodes whose name is not a pasteable identifier."""
        if not kinds or not src_ids or not dst_ids:
            return [], len(src_ids), True, False
        placeholders = ",".join("?" for _ in kinds)
        parent: dict[str, tuple[str, str]] = {}
        found = ""
        queue: deque[tuple[str, int]] = deque((sid, 0) for sid in src_ids)
        seen: set[str] = set(src_ids)
        while queue and len(seen) < node_budget:
            cur, depth = queue.popleft()
            if cur in dst_ids and cur not in src_ids:
                found = cur
                break
            if depth >= max_depth:
                continue
            for row in conn.execute(
                f"""
                SELECT r.kind AS kind, r.dst AS dst, e.name AS name
                FROM relation r
                JOIN entity e ON e.id = r.dst
                WHERE r.src = ? AND r.kind IN ({placeholders})
                LIMIT ?
                """,
                (cur, *kinds, fanout),
            ):
                nxt = str(_row_get(row, "dst") or "")
                if not nxt or nxt in seen:
                    continue
                name = str(_row_get(row, "name") or "")
                if nxt not in dst_ids and not self._trace_name_is_ident(name):
                    continue
                seen.add(nxt)
                parent[nxt] = (cur, str(_row_get(row, "kind") or ""))
                queue.append((nxt, depth + 1))
                if nxt in dst_ids and nxt not in src_ids:
                    found = nxt
                    queue.clear()
                    break
        explored = len(seen)
        exhausted = explored < node_budget
        depth_bound = not exhausted or bool(queue)
        if not found:
            return [], explored, exhausted, depth_bound
        steps: list[dict[str, Any]] = []
        cur = found
        while cur not in src_ids and cur in parent:
            prev, kind = parent[cur]
            steps.append({"from": prev, "to": cur, "kind": kind})
            cur = prev
        steps.reverse()
        return steps, explored, exhausted, depth_bound

    def _trace_family_jobs(self, plan: Any) -> list[tuple[str, str, tuple[str, ...]]]:
        """(family, role, kinds) walks for this to_symbol query."""
        from ascendc_codemap_mcp.engine.query.contract import (
            RELATION_FAMILIES,
            RELATION_FAMILY_NAMES,
        )

        families = [str(f) for f in (getattr(plan, "relation_families", None) or ())]
        if not families and not str(getattr(plan, "relation", "") or ""):
            families = list(RELATION_FAMILY_NAMES)
        jobs: list[tuple[str, str, tuple[str, ...]]] = []
        if families:
            for fam in families:
                kinds = tuple(RELATION_FAMILIES.get(fam) or ())
                if fam == "data":
                    write_kinds = tuple(k for k in kinds if k != "READS")
                    if write_kinds:
                        jobs.append((fam, "write", write_kinds))
                    jobs.append((fam, "read", ("READS",)))
                else:
                    jobs.append((fam, "weak" if fam == "control" else "", kinds))
            return jobs
        kinds = self._split_relation_kinds(str(getattr(plan, "relation", "") or ""))
        if kinds:
            jobs.append(("path", "", kinds))
        return jobs

    def query_trace(self, plan: Any, *, limit: int = 8) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.query.completeness import COMPLETE, UNKNOWN

        src_name = str(
            getattr(plan, "from_symbol", "") or getattr(plan, "symbol", "") or ""
        )
        dst_name = str(getattr(plan, "to_symbol", "") or "")
        src_hits = self._exact_name_hits(src_name, limit=8)
        dst_hits = self._exact_name_hits(dst_name, limit=8)
        if not src_hits or not dst_hits:
            missing = [
                n for n, h in ((src_name, src_hits), (dst_name, dst_hits)) if n and not h
            ]
            return {
                "ok": False,
                "shape": "trace",
                "cards": [],
                "count": 0,
                "completeness": UNKNOWN,
                "unresolved_reason": "NO_SEED",
                "trace_from": src_name,
                "trace_to": dst_name,
                "unknown_endpoints": missing,
                "path": [],
                "family_paths": [],
                "hint": (
                    "not in this CodeMap: "
                    + ", ".join(missing)
                    + " — search for the name first"
                )
                if missing
                else "",
            }
        src_ids = {str(h.get("id") or "") for h in src_hits if h.get("id")}
        dst_ids = {str(h.get("id") or "") for h in dst_hits if h.get("id")}
        jobs = self._trace_family_jobs(plan)
        family_paths: list[dict[str, Any]] = []
        explored = 0
        any_budget = False
        all_exhausted = True
        all_kinds: list[str] = []
        hop_cap = max(1, int(limit)) * 4
        with self._connect() as conn:
            for family, role, kinds in jobs:
                if not kinds:
                    family_paths.append(
                        {
                            "family": family,
                            "role": role,
                            "steps": [],
                            "found": False,
                        }
                    )
                    continue
                all_kinds.extend(k for k in kinds if k not in all_kinds)
                steps, seen_n, exhausted, depth_bound = self._directed_trace_walk(
                    conn,
                    src_ids,
                    dst_ids,
                    kinds,
                    node_budget=(
                        _TRACE_COMPILE_NODE_BUDGET
                        if family == "compile"
                        else _TRACE_FAMILY_NODE_BUDGET
                    ),
                    max_depth=(
                        _TRACE_COMPILE_MAX_DEPTH
                        if family == "compile"
                        else _TRACE_FAMILY_MAX_DEPTH
                    ),
                )
                explored += seen_n
                all_exhausted = all_exhausted and exhausted
                if depth_bound and not steps:
                    any_budget = True
                if steps:
                    steps = steps[:hop_cap]
                    self._name_trace_steps(steps)
                    dirty = any(
                        not self._trace_name_is_ident(str(s.get("from_name") or ""))
                        or not self._trace_name_is_ident(str(s.get("to_name") or ""))
                        for s in steps
                    )
                    if dirty:
                        steps = []
                family_paths.append(
                    {
                        "family": family,
                        "role": role,
                        "steps": steps,
                        "found": bool(steps),
                    }
                )
        # Prefer call, then data write, then data read, then anything else.
        rank = {
            ("call", ""): 0,
            ("data", "write"): 1,
            ("data", "read"): 2,
            ("path", ""): 3,
            ("control", "weak"): 4,
            ("compile", ""): 5,
        }
        found_paths = [p for p in family_paths if p.get("found") and p.get("steps")]
        found_paths.sort(
            key=lambda p: rank.get((str(p.get("family")), str(p.get("role"))), 9)
        )
        primary = found_paths[0]["steps"] if found_paths else []
        hop_ok = bool(primary) and all(
            s.get("from") and s.get("to") and s.get("kind") for s in primary
        )
        if not found_paths:
            return {
                "ok": False,
                "shape": "trace",
                "cards": [src_hits[0], dst_hits[0]],
                "count": 2,
                "completeness": UNKNOWN,
                "unresolved_reason": "SEARCH_BUDGET" if any_budget else "NO_PATH",
                "trace_from": src_name,
                "trace_to": dst_name,
                "trace_relations": list(all_kinds),
                "explored": explored,
                "node_budget": _TRACE_FAMILY_NODE_BUDGET,
                "max_depth": _TRACE_FAMILY_MAX_DEPTH,
                "exhausted": all_exhausted and not any_budget,
                "path": [],
                "family_paths": family_paths,
                "truncated": False,
            }
        dst_found = str(primary[-1].get("to") or "") if primary else ""
        return {
            "ok": True,
            "shape": "trace",
            "cards": [
                src_hits[0],
                next((h for h in dst_hits if h.get("id") == dst_found), dst_hits[0]),
            ],
            "count": len(primary),
            "completeness": COMPLETE if hop_ok else UNKNOWN,
            "unresolved_reason": "" if hop_ok else "NO_PATH",
            "trace_from": src_name,
            "trace_to": dst_name,
            "trace_relations": list(all_kinds),
            "explored": explored,
            "node_budget": _TRACE_FAMILY_NODE_BUDGET,
            "max_depth": _TRACE_FAMILY_MAX_DEPTH,
            "exhausted": all_exhausted,
            "path": primary,
            "family_paths": family_paths,
            "truncated": False,
        }

    def _name_trace_steps(self, steps: list[dict[str, Any]]) -> None:
        """Replace raw entity ids on each hop with name / kind / location.

        A path printed as ``E_FUNCTION_4d3f8d7886ef -> E_BRANCH_...`` is a fact
        the reader has to run more queries to read.
        """
        ids = {str(s.get(k) or "") for s in steps for k in ("from", "to")}
        ids.discard("")
        if not ids:
            return
        brief: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            for chunk in _chunks(sorted(ids)):
                marks = ",".join("?" for _ in chunk)
                for row in conn.execute(
                    f"SELECT id, kind, name, file, line_start FROM entity WHERE id IN ({marks})",
                    tuple(chunk),
                ):
                    brief[str(row["id"])] = {
                        "name": str(row["name"] or ""),
                        "kind": str(row["kind"] or ""),
                        "file": _norm_file(str(row["file"] or "")),
                        "line": int(row["line_start"] or 0),
                    }
        for step in steps:
            for end in ("from", "to"):
                info = brief.get(str(step.get(end) or ""))
                if not info:
                    continue
                step[f"{end}_name"] = info["name"]
                step[f"{end}_kind"] = info["kind"]
                step[f"{end}_file"] = info["file"]
                step[f"{end}_line"] = info["line"]

    def legal_key_query(
        self,
        *,
        pattern: str = "",
        dim: str = "",
        value: str = "",
        filters: dict[str, str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.query.legal_key_cache import query_legal_keys

        return query_legal_keys(
            self.product,
            pattern=pattern,
            dim=dim,
            value=value,
            filters=filters,
            limit=limit,
            offset=offset,
        )
