# -*- coding: utf-8 -*-
"""Operation-specific agent cards. One semantic fact, one representation.

find → candidates; resolve → definition + references; contract → host /
tiling-key / kernel; impact → affected locations. Completeness is independent
of how the seed was found. No human asides.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.query.closure import semantic_impact_closure_sql
from ascendc_codemap_mcp.engine.query.virtual_dispatch import (
    annotate_call_line,
    family_sites,
    render_virtual_dispatch,
)
from ascendc_codemap_mcp.engine.query.completeness import (
    AMBIGUOUS,
    COMPLETE,
    INCOMPLETE,
    UNKNOWN,
    fence_contract,
    pick_transport,
)

MAX_EXPLORE_CHARS = 25_000
#: Headroom the search layout leaves for the header rewrite and any trailing note.
_SEARCH_TAIL_RESERVE = 400
FILE_SECTION_PREFIX = "### "
_INTERNAL_ID_RE = re.compile(
    r"(?:SRCPOL(?:COND)?|SRCMACRO|SRCFRONTIER|E_TYPE_|REL::|entity_id=)[^\s]*"
)
_PROOF_LINES = 16
_MAX_RANGE_LINES = 40
#: Hits previewed per source unit before the rest are counted rather than listed.
_UNIT_HIT_PREVIEW = 3
#: Whole-card source budget. A set answer with many hits must not spend it all
#: on one file.
_MAX_SOURCE_LINES = 120
#: A single definition may use more: its span is the answer, and the indexer has
#: already shaped long function bodies.
_MAX_DEFINITION_LINES = 240
#: Lines kept around each hit when the answer is a site list.
_TIGHT_RADIUS = 2
_PRODUCER_KINDS = {
    RelationKind.WRITES.value,
    RelationKind.DERIVES.value,
}
_PRODUCER_SKIP_KINDS = {
    EntityKind.INPUT.value,
    EntityKind.OUTPUT.value,
}
_CONSUMER_KINDS = {
    RelationKind.READS.value,
    RelationKind.CALLS_UNDER_GUARD.value,
    RelationKind.BINDS.value,
    RelationKind.MATERIALIZES_AS.value,
    RelationKind.SELECTS.value,
}
_LOC_KEEP = ("name", "kind", "file", "line", "role", "consumer_role")
_VALIDATION_NAMES = frozenset(
    {
        "CheckLogLevel",
        "DlogRecord",
        "GetTid",
        "GetSafeStr",
        "GetOpInfo",
        "ReportInnerErrMsg",
        "GetOpLogSuffix",
    }
)
_TPL_BOILER = ("ASCENDC_TPL_ARGS_DECL", "ASCENDC_TPL_SEL", "ASCENDC_TPL_ARGS_SEL")
_KIND_MERGE_PREF = [
    EntityKind.TILING_FIELD.value,
    EntityKind.TILING_KEY.value,
    EntityKind.CONTRACT.value,
    EntityKind.TYPE.value,
    EntityKind.KERNEL.value,
    EntityKind.FUNCTION.value,
    EntityKind.BUFFER.value,
    EntityKind.METHOD.value,
    EntityKind.VARIABLE.value,
]
_CONTRACT_SEED_PREF = [
    EntityKind.TILING_FIELD.value,
    EntityKind.FIELD.value,
    EntityKind.TILING_KEY.value,
    EntityKind.COMPILE_VAR.value,
    EntityKind.MACRO.value,
    EntityKind.CONTRACT.value,
]


@lru_cache(maxsize=32768)
def _is_validation_name(name: str) -> bool:
    leaf = str(name or "").replace("::", ".").rsplit(".", 1)[-1]
    if leaf in _VALIDATION_NAMES:
        return True
    return leaf.startswith(("OP_LOG", "OP_CHECK", "OPS_CHECK", "OP_TILING_CHECK", "DLOG_"))


_FLOW_NOISE_LEAVES = frozenset(
    {
        "min",
        "max",
        "abs",
        "swap",
        "move",
        "sizeof",
        "decltype",
        "ceil",
        "floor",
        "to_string",
        "c_str",
        "make_pair",
        "basic_string",
    }
)
_USED_AT_LINES = 5
_DEF_BODY_KINDS = frozenset(
    {
        EntityKind.TYPE.value,
        EntityKind.FUNCTION.value,
        EntityKind.METHOD.value,
        EntityKind.KERNEL.value,
    }
)


@lru_cache(maxsize=32768)
def _is_noise_name(name: str) -> bool:
    raw = str(name or "")
    qualified = raw.replace(".", "::")
    if qualified.startswith("std::") or "::std::" in qualified:
        return True
    leaf = raw.replace("::", ".").rsplit(".", 1)[-1]
    if leaf.startswith("operator"):
        return True
    if leaf.lower() in _FLOW_NOISE_LEAVES:
        return True
    return _is_validation_name(name)


def _is_tpl_boilerplate(snippet: str) -> bool:
    text = str(snippet or "")
    return any(tok in text for tok in _TPL_BOILER)


def _is_tpl_machinery(name: str) -> bool:
    leaf = _last_ident(name)
    return leaf.startswith("ASCENDC_TPL_") or leaf == "GET_TPL_TILING_KEY"


def _is_unrelated_type_neighbor(seed_name: str, row: dict[str, Any]) -> bool:
    seed_leaf = _last_ident(seed_name).lower()
    leaf = _last_ident(str(row.get("name") or "")).lower()
    if leaf and leaf == seed_leaf:
        return False
    kind = str(row.get("kind") or "").upper()
    if kind == EntityKind.TYPE.value and leaf.endswith("type"):
        return True
    return _is_tpl_machinery(str(row.get("name") or ""))


def _merge_canonical_identities(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same leaf at the same span is one identity, not AMBIGUOUS."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, int]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        file, line, name, _ = _site_line(card)
        leaf = _last_ident(name).lower()
        file_n = file.replace("\\", "/")
        key = (leaf, file_n, line) if file and line > 0 else (leaf, "", 0)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(card)
    located_leaves = {key[0] for key in groups if key[1]}
    out: list[dict[str, Any]] = []
    for key in order:
        leaf, file, _line = key
        if not file and leaf in located_leaves:
            continue
        members = groups[key]
        members.sort(
            key=lambda card: (
                _KIND_MERGE_PREF.index(str(card.get("kind") or ""))
                if str(card.get("kind") or "") in _KIND_MERGE_PREF
                else 50,
                str(card.get("kind") or ""),
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
    return out or cards


def _clip_source(
    query: Any,
    file: str,
    line: int,
    *,
    line_end: int = 0,
    max_lines: int = _PROOF_LINES,
) -> str:
    """Snippet from `.uo` source_line only. Never reads the working tree."""
    if not file or int(line or 0) <= 0 or query is None:
        return ""
    from ascendc_codemap_mcp.engine.query.sql import (
        _restore_blank_lines,
        _source_line_rows,
        _source_line_window,
    )

    # One card renders the same location from several facets, and the enclosing
    # unit is re-derived every time.
    cache = getattr(query, "_clip_source_cache", None)
    key = (str(file), int(line), int(line_end or 0), int(max_lines))
    if cache is not None and key in cache:
        return cache[key]

    def _remember(value: str) -> str:
        if cache is not None and len(cache) < 4096:
            cache[key] = value
        return value

    try:
        with query._connect() as conn:
            if int(line_end or 0) >= int(line):
                rows = _source_line_rows(conn, file, int(line), int(line_end))
                if rows:
                    rows = _restore_blank_lines(rows, int(line_end))
                    return _remember("\n".join(f"{ln}:{txt}" for ln, txt in rows))
            half = max(0, int(max_lines) // 2)
            from ascendc_codemap_mcp.engine.query.sql import STATEMENT_AFTER, STATEMENT_BEFORE

            window_rows = _source_line_rows(
                conn,
                file,
                max(1, int(line) - max(half, STATEMENT_BEFORE)),
                int(line) + max(half, STATEMENT_AFTER, int(max_lines)),
            )
            if window_rows:
                from ascendc_codemap_mcp.engine.query.agent_card import clip_logical_unit

                unit = clip_logical_unit(window_rows, int(line), max_lines=max(int(max_lines), 16))
                if unit:
                    return _remember("\n".join(f"{ln}:{txt}" for ln, txt in unit))
            window = _source_line_window(
                conn, file, int(line), before=half, after=max(half, int(max_lines) - half)
            )
            if window:
                return _remember(window)
    except Exception:  # noqa: BLE001
        return ""
    return _remember("")


def _loc_row(ent: dict[str, Any], *, role: str, snippet: str = "") -> dict[str, Any]:
    line = int(ent.get("line_start") or ent.get("line") or 0)
    row = {
        "id": ent.get("id") or "",
        "name": ent.get("name") or "",
        "kind": ent.get("kind") or "",
        "file": ent.get("file") or "",
        "line": line,
        "role": role,
        "consumer_role": ent.get("consumer_role") or "",
    }
    line_end = int(ent.get("line_end") or 0)
    if line_end > line:
        row["line_end"] = line_end
    if snippet:
        row["snippet"] = snippet
    return row


def _last_ident(name: str) -> str:
    return str(name or "").replace(".", "::").split("::")[-1].strip()


_PLAIN_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_plain_ident(text: str) -> bool:
    """Whether this can be pasted after ``symbol=`` and still parse.

    Cards whose subject was an expression printed follow-ups like
    ``symbol=isSparse)`` and ``symbol=context_->GetAttrs()->GetAttrNum()>IDX``.
    Both are calls the reader cannot make, and one of them was tried anyway.
    """
    return bool(_PLAIN_IDENT_RE.fullmatch(str(text or "").strip()))


def _norm_path(path: str) -> str:
    return str(path or "").replace("\\", "/")


def _is_kernel_path(path: str) -> bool:
    blob = _norm_path(path)
    return "/op_kernel/" in blob or blob.startswith("op_kernel/")


def _is_host_path(path: str) -> bool:
    blob = _norm_path(path)
    return "/op_host/" in blob or blob.startswith("op_host/")


def _kernel_accessor_rows(query: Any, seed: dict[str, Any]) -> list[dict[str, Any]]:
    """Tiling-data getter/setter in op_kernel when the graph has no kernel READS."""
    leaf = _last_ident(str(seed.get("name") or ""))
    if not leaf:
        return []
    names = [f"get_{leaf}", f"Get{leaf[:1].upper() + leaf[1:]}" if len(leaf) > 1 else ""]
    names = [n for n in names if n]
    op_root = getattr(query, "_op_root", None)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with query._connect() as conn:
        for name in names:
            try:
                rows = conn.execute(
                    """
                    SELECT id, kind, name, file, line_start FROM entity
                    WHERE name = ? COLLATE NOCASE
                    LIMIT 8
                    """,
                    (name,),
                ).fetchall()
            except Exception:  # noqa: BLE001
                rows = []
            for row in rows:
                file = str(row[3] or "")
                line = int(row[4] or 0)
                if not _is_kernel_path(file) or not line:
                    continue
                key = (_norm_path(file), line)
                if key in seen:
                    continue
                seen.add(key)
                hit = {
                    "id": str(row[0] or ""),
                    "name": str(row[2] or name),
                    "kind": str(row[1] or ""),
                    "file": file,
                    "line_start": line,
                }
                snippet = _clip_source(query, file, line)
                out.append(_loc_row(hit, role="consumer", snippet=snippet))
        if not out:
            try:
                lines = conn.execute(
                    """
                    SELECT path, line, text FROM source_line
                    WHERE text LIKE '%' || ? || '%'
                      AND REPLACE(path, '\\', '/') LIKE '%/op_kernel/%'
                    LIMIT 8
                    """,
                    (f"get_{leaf}",),
                ).fetchall()
            except Exception:  # noqa: BLE001
                lines = []
            for row in lines:
                file = str(row[0] or "")
                line = int(row[1] or 0)
                key = (_norm_path(file), line)
                if not file or line <= 0 or key in seen:
                    continue
                seen.add(key)
                hit = {
                    "id": "",
                    "name": f"get_{leaf}",
                    "kind": EntityKind.METHOD.value,
                    "file": file,
                    "line_start": line,
                }
                snippet = str(row[2] or "").strip() or _clip_source(query, file, line)
                out.append(_loc_row(hit, role="consumer", snippet=snippet))
    return out


def _prefer_tiling_key_seed(
    cards: list[dict[str, Any]], pattern: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    needle = _last_ident(pattern).lower()
    if not needle:
        return cards, []
    tks = [
        card
        for card in cards
        if str(card.get("kind") or "") == EntityKind.TILING_KEY.value
        and _last_ident(str(card.get("name") or "")).lower() == needle
        and "_def.cpp" not in str(card.get("file") or "").replace("\\", "/")
    ]
    if not tks:
        return cards, []
    primary = tks[0]
    aliases = [card for card in cards if str(card.get("id") or "") != str(primary.get("id") or "")]
    return [primary], aliases


def _leaf_of(card: dict[str, Any]) -> str:
    return _last_ident(str(card.get("name") or "")).lower()


def _seed_pref_key(card: dict[str, Any]) -> tuple[int, str]:
    kind = str(card.get("kind") or "")
    return (
        _CONTRACT_SEED_PREF.index(kind) if kind in _CONTRACT_SEED_PREF else 50,
        kind,
    )


def _prefer_contract_seeds(cards: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Same leaf → one contract; seed prefers the WRITES target (TILING_FIELD)."""
    located = [c for c in cards if isinstance(c, dict) and c.get("id")]
    if not located:
        return None, []
    ranked = sorted(located, key=_seed_pref_key)
    chosen_leaf = _leaf_of(ranked[0])
    group = [c for c in located if _leaf_of(c) == chosen_leaf] or ranked
    group.sort(key=_seed_pref_key)
    return group[0], group[1:]


def _distinct_leaves(cards: list[dict[str, Any]]) -> set[str]:
    return {leaf for c in cards if isinstance(c, dict) and (leaf := _leaf_of(c))}


def _expand_statement_snippet(
    query: Any, card: dict[str, Any], pool: list[dict[str, Any]]
) -> dict[str, Any]:
    """COMPILE_VAR / MACRO / TILING_KEY Definition uses a ~16-line statement window."""
    from ascendc_codemap_mcp.engine.query.sql import _source_line_window

    out = dict(card)
    kind = str(out.get("kind") or "").upper()
    candidates = [out]
    leaf = _leaf_of(out)
    for other in pool:
        if not isinstance(other, dict):
            continue
        if _leaf_of(other) != leaf:
            continue
        other_kind = str(other.get("kind") or "").upper()
        if other_kind in {
            EntityKind.COMPILE_VAR.value,
            EntityKind.MACRO.value,
            EntityKind.TILING_KEY.value,
            EntityKind.TYPE.value,
            EntityKind.FIELD.value,
        }:
            candidates.append(other)
    preferred = sorted(
        candidates,
        key=lambda row: (
            0
            if str(row.get("kind") or "").upper() == EntityKind.COMPILE_VAR.value
            else 1
            if str(row.get("kind") or "").upper() == EntityKind.MACRO.value
            else 2,
        ),
    )
    source = preferred[0] if preferred else out
    file = str(source.get("file") or "")
    line = int(source.get("line") or source.get("line_start") or 0)
    if not file or line <= 0:
        return out
    window = ""
    try:
        with query._connect() as conn:
            window = _source_line_window(conn, file, line)
    except Exception:  # noqa: BLE001
        window = ""
    if window:
        out["file"] = file
        out["line"] = line
        out["snippet"] = window
        if str(source.get("kind") or ""):
            out["kind"] = source.get("kind")
    return out


def build_contract_card(
    query: Any,
    seed: dict[str, Any],
    *,
    extra_seeds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One seed → producers / transport / consumers / sinks / entry / fence."""
    sid = str(seed.get("id") or "")
    kind = str(seed.get("kind") or "")
    op_root = getattr(query, "_op_root", None)
    arch = str(getattr(query, "_architecture", "") or "")
    producers: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    binds: list[dict[str, Any]] = []
    kernel_repr: list[dict[str, Any]] = []
    entry: list[dict[str, Any]] = []
    has_branch = False
    import json

    with query._connect() as conn:
        seed_ids: list[str] = []
        for extra in [seed, *(extra_seeds or [])]:
            if not isinstance(extra, dict):
                continue
            eid = str(extra.get("id") or "")
            if eid and eid not in seed_ids:
                seed_ids.append(eid)
        if sid and sid not in seed_ids:
            seed_ids.insert(0, sid)
        for cur_id in seed_ids:
            rel_rows = conn.execute(
                """
                SELECT kind, src, dst, data FROM relation
                WHERE src = ? OR dst = ?
                LIMIT 400
                """,
                (cur_id, cur_id),
            ).fetchall()
            for rel in rel_rows:
                rkind = str(rel["kind"] or "")
                src_id = str(rel["src"] or "")
                dst_id = str(rel["dst"] or "")
                try:
                    rdata = json.loads(rel["data"] or "{}")
                except json.JSONDecodeError:
                    rdata = {}
                if not isinstance(rdata, dict):
                    rdata = {}
                other_id = dst_id if src_id == cur_id else src_id
                row = query._entity_row(conn, other_id)
                if row is None:
                    continue
                try:
                    edata = json.loads(row["data"] or "{}")
                except json.JSONDecodeError:
                    edata = {}
                if not isinstance(edata, dict):
                    edata = {}
                hit = {
                    "id": str(row["id"]),
                    "name": str(row["name"] or ""),
                    "kind": str(row["kind"] or ""),
                    "file": str(rdata.get("file") or row["file"] or ""),
                    "line_start": int(rdata.get("line") or row["line_start"] or 0),
                    "consumer_role": rdata.get("consumer_role") or "",
                }
                if str(row["kind"] or "") == EntityKind.BRANCH.value:
                    has_branch = True
                if (
                    rkind in _PRODUCER_KINDS
                    and dst_id == cur_id
                    and str(hit["kind"] or "") not in _PRODUCER_SKIP_KINDS
                    and not _is_validation_name(str(hit.get("name") or ""))
                ):
                    snippet = _clip_source(query, hit["file"], hit["line_start"])
                    producers.append(_loc_row(hit, role="producer", snippet=snippet))
                if (
                    rkind in _CONSUMER_KINDS
                    and (dst_id == cur_id or src_id == cur_id)
                    and not _is_validation_name(str(hit.get("name") or ""))
                ):
                    snippet = _clip_source(query, hit["file"], hit["line_start"])
                    consumers.append(_loc_row(hit, role="consumer", snippet=snippet))
                if rkind == RelationKind.BINDS.value:
                    binds.append(_loc_row(hit, role="bind"))
                if rkind == RelationKind.MATERIALIZES_AS.value:
                    kernel_repr.append(
                        _loc_row(
                            hit,
                            role="kernel_repr",
                            snippet=_clip_source(query, hit["file"], hit["line_start"]),
                        )
                    )
                if (
                    rkind == RelationKind.CONTROLS.value
                    and str(row["kind"] or "") == EntityKind.PREDICATE.value
                    and str(edata.get("predicate_role") or "") == "entry_path"
                ):
                    entry.append(
                        _loc_row(
                            hit,
                            role="entry",
                            snippet=_clip_source(query, hit["file"], hit["line_start"]),
                        )
                    )
        closure = semantic_impact_closure_sql(
            conn, seed_ids or [sid], entity_row=query._entity_row
        )
    producers = _dedup_rows(producers)
    consumers = _dedup_rows(consumers)
    seed_file = _norm_path(str(seed.get("file") or ""))
    seed_line = int(seed.get("line_start") or seed.get("line") or 0)
    have_kernel = any(
        _is_kernel_path(str(row.get("file") or ""))
        and (
            _norm_path(str(row.get("file") or "")),
            int(row.get("line") or row.get("line_start") or 0),
        )
        != (seed_file, seed_line)
        for row in consumers
    )
    if not have_kernel:
        consumers = _dedup_rows(consumers + _kernel_accessor_rows(query, seed))
    sinks = []
    for row in closure.get("sinks") or []:
        if _is_validation_name(str(row.get("name") or "")):
            continue
        snippet = _clip_source(query, str(row.get("file") or ""), int(row.get("line") or 0))
        sinks.append({**row, "snippet": snippet} if snippet else dict(row))
    transport = pick_transport(seed_kind=kind, has_branch_reader=has_branch, has_bind=bool(binds))
    seed_snippet = str(seed.get("snippet") or "") or _clip_source(
        query,
        str(seed.get("file") or ""),
        int(seed.get("line_start") or seed.get("line") or 0),
        line_end=int(seed.get("line_end") or 0),
    )
    seed_row = _loc_row(seed, role="seed", snippet=seed_snippet)
    fence = fence_contract(
        seeds=[seed_row],
        producers=producers,
        consumers=consumers,
        sinks=sinks,
        transport=transport,
        binds=binds,
        kernel_repr=kernel_repr or binds,
        architecture=arch,
        seed_arch=str(seed.get("architecture") or ""),
    )
    return {
        "seed": seed_row,
        "transport": transport,
        "producers": producers[:12],
        "consumers": consumers[:12],
        "impact_sinks": sinks[:12],
        "binds": binds[:8],
        "kernel_repr": kernel_repr[:8],
        "entry": entry[:8],
        "completeness": fence["completeness"],
        "unresolved_reason": fence.get("unresolved_reason") or "",
        "unresolved_reasons": fence.get("unresolved_reasons") or [],
        "checks": fence.get("checks") or {},
        "windows": fence.get("windows") or [],
        "impact_nodes": len(closure.get("nodes") or []),
    }


def _loc_only(row: dict[str, Any]) -> dict[str, Any]:
    out = {key: row.get(key) for key in _LOC_KEEP if row.get(key) not in (None, "", [])}
    return out


def sanitize_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """One sanitizer for resolve / evidence / around: dedup, noise, no triple source."""
    out = dict(payload)
    snippet = str(out.get("snippet") or "")
    if snippet:
        for key in ("seeds", "hits", "cards"):
            rows = out.get(key)
            if not isinstance(rows, list):
                continue
            cleaned = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                item.pop("snippet", None)
                cleaned.append(item)
            out[key] = cleaned
    used = out.get("used_at")
    if isinstance(used, list):
        out["used_at"] = [
            row
            for row in used
            if isinstance(row, dict) and not _is_noise_name(str(row.get("name") or ""))
        ]
    neighbors = out.get("neighbors")
    if isinstance(neighbors, list):
        out["neighbors"] = [
            row
            for row in neighbors
            if isinstance(row, dict) and not _is_noise_name(str(row.get("name") or ""))
        ]
    return out


def slim_explore_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """One copy of each fact. Snippets live in markdown, not in JSON."""
    out = sanitize_agent_payload(payload)
    out.pop("proof", None)
    contract = out.get("contract")
    if isinstance(contract, dict):
        slim = dict(contract)
        slim.pop("proof", None)
        for key in ("producers", "consumers", "impact_sinks", "binds", "kernel_repr", "entry"):
            rows = slim.get(key)
            if isinstance(rows, list):
                slim[key] = [_loc_only(row) if isinstance(row, dict) else row for row in rows]
        if isinstance(slim.get("seed"), dict):
            slim["seed"] = _loc_only(slim["seed"])
        out["contract"] = slim
    cards = out.get("cards")
    if isinstance(cards, list):
        slim_cards = []
        for card in cards:
            if not isinstance(card, dict):
                continue
            item = {
                key: card[key]
                for key in (
                    "kind",
                    "name",
                    "file",
                    "line",
                    "line_start",
                    "line_end",
                    "summary",
                    "source",
                    "text",
                    "snippet",
                    "match",
                )
                if key in card and card[key] not in (None, "")
            }
            if "line" not in item:
                start = int(card.get("line_start") or card.get("line") or 0)
                if start:
                    item["line"] = start
            hidden = str(card.get("id") or card.get("entity_id") or "")
            if hidden:
                item["_entity_id"] = hidden
            extras = card.get("extras") if isinstance(card.get("extras"), dict) else {}
            writers = extras.get("writers") or card.get("writers")
            readers = extras.get("readers") or card.get("readers")
            if writers:
                item["writers"] = writers
            if readers:
                item["readers"] = readers
            if isinstance(card.get("facets"), dict):
                item["facets"] = card["facets"]
            if isinstance(card.get("counts"), dict):
                item["counts"] = card["counts"]
            slim_cards.append(item)
        out["cards"] = slim_cards
    out.pop("declared_coverage", None)
    out.pop("product_coverage", None)
    cov = out.get("coverage")
    if isinstance(cov, dict) and out.get("dim_coverage"):
        cov = dict(cov)
        cov.pop("dim_coverage", None)
        out["coverage"] = cov
    return out


def _site_line(row: dict[str, Any]) -> tuple[str, int, str, str]:
    file = str(row.get("file") or "").replace("\\", "/")
    line = int(row.get("line") or row.get("line_start") or 0)
    name = str(row.get("name") or "")
    snippet = str(row.get("snippet") or "")
    return file, line, name, snippet


def _snippet_rows(snippet: str, fallback_line: int) -> list[tuple[int, str]]:
    """Numbered source rows. Unnumbered text is contiguous from the anchor.

    Callers key rows by line number, so giving every unnumbered line the same
    anchor collapsed a stored span snippet down to its first line and dropped
    the rest without saying so. A span snippet is the entity's own contiguous
    text, so counting on from the anchor is what it already means.
    """
    rows: list[tuple[int, str]] = []
    offset = 0
    for raw in str(snippet or "").splitlines():
        if ":" in raw[:8]:
            num, _, rest = raw.partition(":")
            if num.strip().isdigit():
                rows.append((int(num), rest))
                offset = 0
                continue
        rows.append((fallback_line + offset, raw))
        offset += 1
    return rows


def _has_loc(row: dict[str, Any]) -> bool:
    file, line, _, _ = _site_line(row)
    return bool(file) and line > 0


def _dedup_rows(rows: Any) -> list[dict[str, Any]]:
    """Same name at the same location is one fact, however many edges found it."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        file, line, name, _ = _site_line(row)
        key = (file, line, name)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _is_name_list(payload: dict[str, Any]) -> bool:
    """Name discovery: the answer is which idents exist, not their bodies."""
    if str(payload.get("shape") or "") != "find":
        return False
    if str(payload.get("projection") or "") != "locations":
        return False
    return bool([c for c in (payload.get("cards") or []) if isinstance(c, dict)])


def _is_site_list(payload: dict[str, Any]) -> bool:
    """A multi-hit set answer: the caller wants where, not the whole body."""
    if str(payload.get("operation") or "") not in {"find", "entry"}:
        return False
    return len([c for c in (payload.get("cards") or []) if isinstance(c, dict)]) > 1


def _render_source(sites: list[dict[str, Any]], *, tight: bool) -> list[str]:
    """One numbered line per source line, merged across overlapping windows.

    Sites in the same function used to each print their own window, so a set
    answer repeated the same block once per hit. Keying by line number collapses
    that and keeps line numbers monotonic per file.

    A site list is budgeted per file, because many hits must share the card. A
    single definition is not: dropping lines out of the middle of one span hides
    the decisive line while looking complete, so its span is emitted whole and
    only the whole-card budget can cut it — from the tail, visibly.
    """
    max_lines_per_file = _MAX_RANGE_LINES if tight else _MAX_SOURCE_LINES
    max_lines_total = _MAX_SOURCE_LINES if tight else _MAX_DEFINITION_LINES
    by_file: dict[str, dict[int, str]] = {}
    anchors: dict[str, set[int]] = {}
    order: list[str] = []
    tpl_shown = False
    for site in sites:
        file = site["file"]
        snippet = site["snippet"]
        if _is_tpl_boilerplate(snippet):
            if tpl_shown:
                continue
            tpl_shown = True
        anchor = int(site["line"] or 0)
        rows = _snippet_rows(snippet, anchor)
        if not rows and anchor:
            rows = [(anchor, site["name"])]
        if not rows:
            continue
        if tight and anchor:
            rows = [(no, text) for no, text in rows if abs(no - anchor) <= _TIGHT_RADIUS]
        if file not in by_file:
            by_file[file] = {}
            anchors[file] = set()
            order.append(file)
        bucket = by_file[file]
        for no, text in rows:
            bucket.setdefault(no, text.rstrip())
        if anchor:
            anchors[file].add(anchor)

    out: list[str] = ["**Source**"]
    budget = max_lines_total
    # The per-file cap exists so one file cannot eat a multi-hit card. A card
    # showing a single definition has no one to share with, and capping it there
    # cut function bodies at 120 lines while most of the budget went unspent.
    if len(order) == 1:
        max_lines_per_file = max_lines_total
    for file in order:
        bucket = by_file[file]
        if not bucket or budget <= 0:
            continue
        numbers = sorted(bucket)
        keep = _budget_lines(
            numbers, anchors[file], min(max_lines_per_file, budget), contiguous=not tight
        )
        if not keep:
            continue
        if len(keep) < len(numbers):
            out.append(f"(showing {len(keep)} of {len(numbers)} lines)")
        out.append(f"{FILE_SECTION_PREFIX}{file}")
        prev: int | None = None
        for no in keep:
            if prev is not None and no > prev + 1:
                out.append("...")
            out.append(f"{no}|  {bucket[no]}")
            prev = no
        budget -= len(keep)
    if len(out) == 1:
        return []
    out.append("")
    return out


def _budget_lines(
    numbers: list[int], anchors: set[int], budget: int, *, contiguous: bool
) -> list[int]:
    """Trim a line list to budget.

    `contiguous` keeps one unbroken run starting at the anchor, so a definition
    is never shown with its middle missing. Otherwise keep the lines nearest an
    anchor, which is what a scattered site list wants.
    """
    if budget <= 0:
        return []
    if len(numbers) <= budget:
        return numbers
    if not anchors:
        return numbers[:budget]
    if contiguous:
        start = min(anchors)
        head = [no for no in numbers if no >= start][:budget]
        if len(head) < budget:
            lead = [no for no in numbers if no < start]
            head = lead[-(budget - len(head)) :] + head
        return head
    ranked = sorted(numbers, key=lambda no: min(abs(no - a) for a in anchors))
    return sorted(ranked[:budget])


def cut_explore_text(text: str) -> str:
    text = _INTERNAL_ID_RE.sub("", str(text or ""))
    if len(text) <= MAX_EXPLORE_CHARS:
        return text
    cut = text[:MAX_EXPLORE_CHARS]
    last = cut.rfind("\n" + FILE_SECTION_PREFIX)
    if last > MAX_EXPLORE_CHARS * 0.5:
        cut = cut[:last]
    note = "\n\ntruncated; retry with a narrower ident"
    return cut.rstrip() + note


def _card_kinds(row: dict[str, Any]) -> str:
    kinds = row.get("kinds")
    if isinstance(kinds, list) and kinds:
        return " · ".join(str(k) for k in kinds if k)
    return str(row.get("kind") or "")


def _loc(row: dict[str, Any]) -> str:
    file, line, name, _ = _site_line(row)
    if file and line:
        return f"{name} ({file}:{line})" if name else f"{file}:{line}"
    return name or file or "?"


def _useful_rows(rows: Any, seed_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if _is_validation_name(name) or _is_noise_name(name) or _is_tpl_machinery(name):
            continue
        if _is_unrelated_type_neighbor(seed_name, row):
            continue
        if not _has_loc(row):
            continue
        out.append(row)
    return _dedup_rows(out)


def _render_glob_miss(payload: dict[str, Any]) -> list[str]:
    """Say when the zero came from `file=`, not from the pattern.

    A glob naming no file and a glob naming files that hold no match both print
    zero, but only the second is a statement about the pattern. Readers who
    cannot tell them apart go on to widen a pattern that was never the problem.
    """
    miss = payload.get("file_filter_miss")
    if not isinstance(miss, dict):
        return []
    glob = str(miss.get("glob") or "")
    lines = [f"file={glob} matched no file in the snapshot, so the pattern never ran"]
    nearest = [str(p) for p in (miss.get("nearest") or []) if p]
    if nearest:
        lines.append("  files with that name, spelled as they are stored")
        lines.extend(f"    {path}" for path in nearest)
    else:
        lines.append("  drop file= to search the whole snapshot")
    lines.append("")
    return lines


def _render_related_patterns(payload: dict[str, Any]) -> list[str]:
    """Suggestions for a zero-hit search. Each keeps its own pattern and count."""
    rows = [r for r in (payload.get("related_patterns") or []) if isinstance(r, dict)]
    if not rows:
        return []
    # Offering substitutes right after a zero is what made readers suspect the
    # pattern had been rejected or quietly rewritten. A `\.` or a character
    # class that costs a retry to re-confirm costs more than this line.
    pattern = str(payload.get("pattern") or payload.get("explore_pattern") or "").strip()
    ran = f"`{pattern}` ran as written and matched nothing." if pattern else (
        "The pattern ran as written and matched nothing."
    )
    lines = [f"{ran} These are other patterns, not retries of it:"]
    for row in rows:
        pattern = str(row.get("pattern") or "")
        matches = int(row.get("matches") or 0)
        if pattern:
            lines.append(f"  {pattern}  {matches} matches")
    lines.append("")
    return lines


def _render_resolvable(payload: dict[str, Any]) -> list[str]:
    """Point at the semantic cards this text already has.

    This goes above the hit list, not below it. Its whole purpose is to stop the
    caller reading hits one by one, and a broad pattern truncates its own list --
    so placed underneath, the advice arrived only for the searches that did not
    need it.
    """
    rows = [r for r in (payload.get("resolvable_symbols") or []) if isinstance(r, dict)]
    if not rows:
        return []
    lines = ["Trace for the full picture (writers, guards, consumers)"]
    for row in rows:
        symbol = str(row.get("symbol") or "")
        kinds = ", ".join(str(k) for k in (row.get("kinds") or []) if k)
        if symbol and _is_plain_ident(_last_ident(symbol)):
            lines.append(f"  trace symbol={symbol}" + (f"   {kinds}" if kinds else ""))
    lines.append("")
    return lines


def _render_search_markdown(payload: dict[str, Any]) -> list[str]:
    prefix: list[str] = []
    units = [u for u in (payload.get("units") or []) if isinstance(u, dict)]
    leftover = [r for r in (payload.get("leftover") or []) if isinstance(r, dict)]
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    tpl_n = int(payload.get("template_lines") or 0)
    unit_n = int(payload.get("source_units") or len(units))
    if units or tpl_n:
        real_total = int(payload.get("total") or 0)
        if real_total <= 0:
            real_total = sum(len(u.get("hits") or []) for u in units) + len(leftover)
        header = f"{real_total} matches · {unit_n} source units"
        lines: list[str] = [header]
        if tpl_n:
            lines.append(f"+{tpl_n} template lines (collapsed)")
        lines.extend(_render_resolvable(payload))
        shown = 0
        units_shown = 0
        # Laying out every unit and letting the payload clipper cut the tail
        # produced a header counting lines the reader never received, over a
        # body that ended mid-statement. Stop at a whole unit instead.
        budget = MAX_EXPLORE_CHARS - sum(len(row) + 1 for row in lines) - _SEARCH_TAIL_RESERVE
        for unit in units:
            name = str(unit.get("name") or "")
            file = str(unit.get("file") or "")
            start = int(unit.get("line_start") or 0)
            end = int(unit.get("line_end") or 0)
            loc = f"{file}:{start}-{end}" if file and start and end else file
            block = [f"{name}  {loc}".rstrip()]
            hits = [h for h in (unit.get("hits") or []) if isinstance(h, dict)]
            kept = 0
            for hit in hits[:_UNIT_HIT_PREVIEW]:
                hline = int(hit.get("line") or 0)
                text = str(hit.get("text") or "").rstrip()
                if hline and text:
                    block.append(f"  {hline}:{text}")
                    kept += 1
            # A unit that shows three of nine reads exactly like a unit that has
            # three, and the reader has no way to tell which they are looking at.
            rest = len(hits) - _UNIT_HIT_PREVIEW
            if rest > 0:
                block.append(f"  … {rest} more in this unit")
            cost = sum(len(row) + 1 for row in block)
            if units_shown and cost > budget:
                break
            lines.extend(block)
            budget -= cost
            shown += kept
            units_shown += 1
        for row in leftover:
            file = str(row.get("file") or "")
            line = int(row.get("line") or 0)
            text = str(row.get("text") or "").rstrip()
            loc = f"{file}:{line}" if file and line else (file or "?")
            entry = f"{loc}:{text}" if text else loc
            if len(entry) + 1 > budget:
                break
            lines.append(entry)
            budget -= len(entry) + 1
            shown += 1
        # "51 matches" above a list of twenty reads as a complete list of
        # fifty-one. Say which of the two this is, and how to get the rest.
        if units_shown < unit_n:
            lines[0] = (
                f"{header} — showing {shown} lines from {units_shown} of {unit_n} units; "
                f"narrow the pattern or pass file= to reach the rest"
            )
        elif shown < real_total:
            # Every unit being present is not the same as every line being
            # present, and the per-unit "3 more" notes put the arithmetic on
            # the reader: one pattern folded a hundred lines across twenty-four
            # units and the decisive one was inside the fold.
            hidden = real_total - shown
            lines[0] = (
                f"{header} — every unit listed, showing {shown} lines; "
                f"{hidden} more folded into the units below, "
                f"resolve a unit to read them"
            )
        else:
            lines[0] = f"{header} — complete"
        return prefix + lines
    total = int(payload.get("total") or len(cards))
    showing = len(cards)
    if total == 0 and not cards:
        lines = ["0 matches", ""]
        lines.extend(_render_glob_miss(payload))
        lines.extend(_render_related_patterns(payload))
        symbols = [s for s in (payload.get("symbols") or []) if isinstance(s, dict)]
        if symbols:
            lines.append("Symbols")
            for row in symbols:
                loc = _loc(row)
                kind = str(row.get("kind") or "")
                name = str(row.get("name") or "")
                alias = str(row.get("matched_alias") or "")
                lines.append(f"{loc}  {name}  {kind}".rstrip())
                if alias:
                    lines.append(f"  matched alias: {alias}")
            lines.append("")
        return prefix + lines
    lines = [f"{total} matches", ""]
    for row in cards:
        file = str(row.get("file") or "")
        line = int(row.get("line") or row.get("line_start") or 0)
        text = str(row.get("text") or row.get("snippet") or "").rstrip()
        loc = f"{file}:{line}" if file and line else (file or "?")
        kind = str(row.get("kind") or "")
        name = str(row.get("name") or "")
        if kind or name:
            extra = " ".join(p for p in (kind, name) if p)
            lines.append(f"{loc}:{text}" if text else f"{loc}  {extra}")
            if extra and text:
                lines.append(f"  {extra}")
        else:
            lines.append(f"{loc}:{text}" if text else loc)
    if showing < total:
        lines.append("")
        lines.append(f"showing {showing}/{total}")
    cursor = str(payload.get("next_cursor") or "")
    if cursor:
        lines.append(f"next_cursor={cursor}")
    return prefix + lines


def _render_find_markdown(payload: dict[str, Any]) -> list[str]:
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    total = int(payload.get("total") or len(cards))
    returned = int(payload.get("returned") or len(cards))
    exhaustive = payload.get("exhaustive")
    if exhaustive is None:
        exhaustive = returned >= total and not payload.get("truncated")
    lines = [
        f"Matches: {total}",
        f"returned {returned} of {total} · exhaustive={'yes' if exhaustive else 'no'}",
        "",
        "Top candidates:",
    ]
    for index, row in enumerate(cards, 1):
        file, line, name, _ = _site_line(row)
        kind_s = _card_kinds(row)
        where = f"{file}:{line}" if file and line else file
        lines.append(f"{index}. {name}")
        bits = [part for part in (kind_s, where) if part]
        if bits:
            lines.append("   " + " · ".join(bits))
        match = str(row.get("match") or "")
        if match:
            lines.append(f"   match: {match}")
    lines.append("")
    groups = [g for g in (payload.get("kind_groups") or []) if isinstance(g, dict)]
    if not groups:
        counts: dict[str, int] = {}
        for row in cards:
            kind = str(row.get("kind") or "OTHER")
            counts[kind] = counts.get(kind, 0) + 1
        groups = [
            {"kind": kind, "count": n}
            for kind, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    if groups:
        lines.append("Groups:")
        lines.append(
            " · ".join(f"{g.get('kind')} {g.get('count')}" for g in groups if g.get("kind"))
        )
        lines.append("")
    if total > len(cards) and str(payload.get("projection") or "") != "locations":
        lines.append("Refine with kind= / file= or a tighter name=")
        lines.append("")
    op_sites = [r for r in (payload.get("operation_sites") or []) if isinstance(r, dict)]
    if op_sites:
        lines.append("**Call sites**")
        for row in op_sites:
            file, line, _, _ = _site_line(row)
            if file and line:
                lines.append(f"- {file}:{line}")
        site_total = int(payload.get("operation_sites_total") or len(op_sites))
        if payload.get("operation_sites_truncated"):
            lines.append(
                f"returned {len(op_sites)} of {site_total} · exhaustive=no"
            )
        lines.append("")
    elif not _is_name_list(payload) and cards:
        lines.append("**Call sites**")
        seen: set[tuple[str, int]] = set()
        for row in cards:
            file, line, _, _ = _site_line(row)
            if not file or line <= 0 or (file, line) in seen:
                continue
            seen.add((file, line))
            lines.append(f"- {file}:{line}")
        lines.append("")
    hint = str(payload.get("hint") or "")
    if hint and (
        "no ident" in hint.lower()
        or "showing" in hint.lower()
        or "already listed" in hint.lower()
    ):
        lines.extend([hint, ""])
    return lines


def _render_proof(confirmed: int, unresolved: int, exhaustive: Any, *, total: int | None = None) -> str:
    shown = total if total is not None else confirmed + unresolved
    flag = "yes" if exhaustive else "no"
    if unresolved:
        return f"{confirmed}/{shown} · unresolved {unresolved} · exhaustive={flag}"
    return f"{confirmed}/{shown} · exhaustive={flag}"


def _render_site_cite(site: dict[str, Any]) -> str:
    line = int(site.get("line") or 0)
    rhs = _clip_expr(site.get("rhs") or site.get("expression"))
    when = _clip_guard(site.get("when"))
    bit = f"  {line}" if line else "  ?"
    if rhs:
        bit += f" = {rhs}"
    if when:
        bit += f" when {when}"
    return bit


def _render_write_site(site: dict[str, Any]) -> str:
    """One site, one line: ``<function>:<line> = <value> when <guard>``.

    Splitting the function name onto its own row made a three-site field read
    as six unrelated lines, and dropped the pairing that makes it an answer.
    """
    fn = str(site.get("function") or site.get("writer") or "").strip()
    line = int(site.get("line") or 0)
    head = f"{fn}:{line}" if fn and line else (fn or (str(line) if line else "?"))
    rhs = _clip_expr(site.get("rhs") or site.get("expression"), _REMOTE_EXPR_MAX)
    when = _clip_guard(site.get("when"))
    bit = f"  {head}"
    if rhs:
        bit += f" = {rhs}"
    if when:
        bit += f" when {when}"
    return bit


def _render_reach(site: dict[str, Any]) -> list[str]:
    """The call that reaches this write, indented under it.

    Three writes to one flag are three candidate answers until you know the
    order they run in, and the order is a property of the calls, not the
    writes. Pinning it took the agent its own round trips per flag.
    """
    hits = [h for h in (site.get("reached_by") or []) if isinstance(h, dict)]
    if not hits:
        return []
    parts = []
    for hit in hits:
        caller = str(hit.get("caller") or "").strip()
        if not caller:
            continue
        line = int(hit.get("line") or 0)
        parts.append(f"{caller}:{line}" if line else caller)
    return [f"      reached by {', '.join(parts)}"] if parts else []


def _render_write_tree(sites: Sequence[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for site in sites:
        out.append(_render_write_site(site))
        out.extend(_render_reach(site))
    return out


def _section_head(label: str, shown: int, total: int, *, computed: bool = True) -> str:
    """``<label>  N of M, complete`` — never a bare count.

    A section that prints three writes and stops is indistinguishable from a
    symbol that has three writes, and the reader has no way to ask which. The
    same collapse of "not computed" into "none" is what produced seventeen
    false `no resolved caller` verdicts, so every section states its own
    standing.
    """
    if not computed:
        return f"{label}  not computed here"
    if total <= shown:
        return f"{label}  {shown} of {shown}, complete"
    return f"{label}  {shown} of {total} shown"


def _render_bundle(bundle: dict[str, Any]) -> list[str]:
    if not isinstance(bundle, dict):
        return []
    lines: list[str] = []
    host = [s for s in (bundle.get("host_value_definitions") or []) if isinstance(s, dict)]
    if host:
        lines.append(_section_head("Host value definitions", len(host), len(host)))
        lines.extend(_render_write_tree(host))
    transport = [s for s in (bundle.get("transport") or []) if isinstance(s, dict)]
    if transport:
        lines.append("Transport")
        for site in transport:
            line = int(site.get("line") or 0)
            fn = str(site.get("function") or "").strip()
            recv = str(site.get("receiver") or "")
            head = f"{fn}:{line}" if fn and line else (str(line) if line else "?")
            bit = f"  {head}"
            if recv:
                bit += f"  {recv}"
            lines.append(bit)
    accessors = [s for s in (bundle.get("accessors") or []) if isinstance(s, dict)]
    if accessors:
        lines.append("Accessors")
        for row in accessors[:6]:
            role = str(row.get("role") or "")
            name = str(row.get("name") or "")
            line = int(row.get("line") or 0)
            bit = f"  {role} {name}" if role else f"  {name}"
            if line:
                bit += f"  {line}"
            lines.append(bit.rstrip())
    consumers = [s for s in (bundle.get("kernel_consumers") or []) if isinstance(s, dict)]
    if consumers:
        lines.append(_section_head("Kernel consumers", len(consumers), len(consumers)))
        for row in consumers:
            name = str(row.get("name") or "")
            line = int(row.get("line") or 0)
            bit = f"  {name}  {line}".rstrip()
            when = _clip_guard(row.get("when"))
            if when:
                bit += f"  when {when}"
            lines.append(bit)
            lines.extend(_render_reach(row))
    assignments = [s for s in (bundle.get("assignments") or []) if isinstance(s, dict)]
    # A name that is both a TILING_FIELD and a FIELD carries two site lists that
    # describe the same writes; printing both reads as two findings.
    host_sites = {(str(s.get("file") or ""), int(s.get("line") or 0)) for s in host}
    assignments = [
        s
        for s in assignments
        if (str(s.get("file") or ""), int(s.get("line") or 0)) not in host_sites
    ]
    if assignments:
        lines.append(_section_head("Writes", len(assignments), len(assignments)))
        lines.extend(_render_write_tree(assignments))
    consumed = _consumer_names(bundle.get("consumed_by"))
    if consumed:
        lines.append("Consumed by")
        lines.append("  " + "  ".join(consumed[:8]))
    layout = [s for s in (bundle.get("workspace_layout") or []) if isinstance(s, dict)]
    if layout:
        lines.append("Workspace layout")
        for row in layout:
            label = str(row.get("label") or row.get("name") or "")
            line = int(row.get("line") or 0)
            if label and line:
                lines.append(f"  {label}:{line}")
    resource = bundle.get("resource") if isinstance(bundle.get("resource"), dict) else {}
    lines.extend(_render_resource(resource))
    controls = bundle.get("controls") if isinstance(bundle.get("controls"), dict) else {}
    cited = " ".join(
        str(site.get("when") or "")
        for site in (*host, *assignments)
        if isinstance(site, dict)
    )
    lines.extend(_render_control_proj(controls, already_cited=cited))
    if lines:
        lines.append("")
    return lines


def _sync_totals(sync: dict[str, Any]) -> list[str]:
    """What the recorded sites support, and no more.

    The two numbers count guarded sites, not executed calls: one site can issue
    a second flag at an offset, and an `isReuse` pair is two sites only one of
    which ever compiles. Reading a difference between them as a defect made the
    tool call a balanced cross-core handshake UNBALANCED, so a difference is now
    left to the reader and only a missing side is called out.
    """
    signals = sync.get("signal_count")
    awaits = sync.get("await_count")
    if signals is None and awaits is None:
        return []
    sig = signals if isinstance(signals, int) else None
    awa = awaits if isinstance(awaits, int) else None
    head = f"  set sites={sig if sig is not None else '?'}"
    head += f"  wait sites={awa if awa is not None else '?'}"
    if sig and not awa:
        head += "  — set with no wait recorded"
    elif awa and not sig:
        head += "  — wait with no set recorded"
    out = [head]
    guarded = _guarded_sync_sites(sync)
    if guarded and sig and awa:
        # Saying only "not call counts" left the reader to guess how far off
        # they were. Naming how many sites are conditional says which way, and
        # by how much at most: two sites under `X` and `!X` are two lines and
        # one call, so a difference between the totals is not yet a defect.
        out.append(
            f"    counted at source; {guarded} of these sit under a condition, "
            f"so a run issues fewer than the totals"
        )
    return out


def _guarded_sync_sites(sync: dict[str, Any]) -> int:
    scopes = [s for s in (sync.get("scopes") or []) if isinstance(s, dict)]
    pools = [s.get("edges") for s in scopes] + [sync.get("edges")]
    seen: set[tuple[str, int]] = set()
    for edges in pools:
        for edge in edges or []:
            if not isinstance(edge, dict) or not str(edge.get("when") or "").strip():
                continue
            seen.add((str(edge.get("file") or ""), int(edge.get("line") or 0)))
    return len(seen)


def _sync_edge_lines(edges: Any, *, indent: str) -> list[str]:
    """Every set and every wait, with the branch it sits under."""
    rows = [e for e in (edges or []) if isinstance(e, dict)]
    # One name can be the event of two templates. Bare line numbers then read as
    # one implementation setting twice rather than two setting once each.
    files = {str(e.get("file") or "") for e in rows}
    show_file = len(files) > 1
    out: list[str] = []
    for rel, label in (("SIGNALS", "set at"), ("AWAITS", "wait at")):
        picked = [e for e in rows if str(e.get("rel") or "") == rel]
        for row in sorted(
            picked, key=lambda e: (str(e.get("file") or ""), int(e.get("line") or 0))
        ):
            where = f"{row.get('file')}:" if show_file else ""
            bit = f"{indent}{label} {where}{int(row.get('line') or 0)}"
            when = _when_clauses([row.get("when")], limit=1)
            if when:
                bit += f"  when {when[0]}"
            out.append(bit)
    return out


def _render_resource(resource: dict[str, Any]) -> list[str]:
    if not resource:
        return []
    lines: list[str] = []
    identity = [
        row
        for row in (resource.get("identity") or [])
        if isinstance(row, (list, tuple)) and len(row) == 2
    ]
    if identity:
        lines.append("Resource")
        lines.append("  " + "  ".join(f"{key}={value}" for key, value in identity))
    allocated = [s for s in (resource.get("allocated_at") or []) if isinstance(s, dict)]
    if allocated:
        lines.append("Allocated at")
        for row in allocated:
            callee = str(row.get("callee") or "")
            line = int(row.get("line") or 0)
            size = str(row.get("size") or "")
            bit = f"  {callee}:{line}" if callee and line else f"  {line or '?'}"
            if size:
                bit += f"  size {size}"
            lines.append(bit)
    backing = [s for s in (resource.get("backing") or []) if isinstance(s, dict)]
    if backing:
        shown = backing[:6]
        rest = len(backing) - len(shown)
        head = "Backing"
        if rest > 0:
            head += f" ({len(shown)} of {len(backing)} sites)"
        lines.append(head)
        # Two sites in different files read as two lines of one file when only
        # the line number is printed, so name the file once it stops being one.
        multifile = len({str(r.get("file") or "") for r in backing}) > 1
        for row in shown:
            name = str(row.get("name") or "")
            space = str(row.get("physical_space") or "")
            via = str(row.get("via") or "")
            line = int(row.get("line") or 0)
            file = str(row.get("file") or "")
            bit = f"  {name}"
            if space:
                bit += f"  {space}"
            if via:
                bit += f"  {via}"
            if line:
                bit += f"  {file}:{line}" if multifile and file else f"  {line}"
            lines.append(bit)
    sync = resource.get("sync") if isinstance(resource.get("sync"), dict) else {}
    if sync and any(
        sync.get(k) not in (None, "", [])
        for k in ("paired", "signal_count", "await_count", "mechanism", "edges", "scopes")
    ):
        lines.append("Sync pairing")
        if sync.get("mechanism") and not identity:
            lines.append(f"  mechanism={sync.get('mechanism')}")
        lines.extend(_sync_totals(sync))
        scopes = [s for s in (sync.get("scopes") or []) if isinstance(s, dict)]
        # The ratio above is the pairing question; a setter scope and a waiter
        # scope each hold one half of it, so the halves are named separately.
        for scope in scopes:
            head = str(scope.get("scope") or "?")
            line_no = int(scope.get("line") or 0)
            if line_no:
                head += f":{line_no}"
            counts = [
                f"{label}={scope.get(key)}"
                for key, label in (
                    ("signal_count", "set sites"),
                    ("await_count", "wait sites"),
                )
                if int(scope.get(key) or 0) > 0
            ]
            lines.append(f"    {head}" + ("  " + "  ".join(counts) if counts else ""))
            calls = [c for c in (scope.get("calls") or []) if isinstance(c, dict)]
            if calls:
                # Sites are guarded places; this is what the function executes.
                named = ", ".join(
                    f"{c.get('name')} ×{int(c.get('count') or 0)}" for c in calls[:4]
                )
                lines.append(f"      this scope issues {named} in all (every id)")
            lines.extend(_sync_edge_lines(scope.get("edges"), indent="      "))
        if not scopes:
            lines.extend(_sync_edge_lines(sync.get("edges"), indent="    "))
    order = [s for s in (resource.get("order") or []) if isinstance(s, dict)]
    if order:
        lines.append("Order")
        for row in order[:6]:
            lines.append(f"  {row.get('name') or ''}  {row.get('via') or ''}".rstrip())
    layout = [s for s in (resource.get("workspace_layout") or []) if isinstance(s, dict)]
    if layout and "Workspace layout" not in lines:
        lines.append("Workspace layout")
        for row in layout:
            label = str(row.get("label") or row.get("name") or "")
            line = int(row.get("line") or 0)
            if label and line:
                lines.append(f"  {label}:{line}")
    return lines


def _render_control_proj(
    controls: dict[str, Any], *, already_cited: str = ""
) -> list[str]:
    if not controls:
        return []
    from ascendc_codemap_mcp.engine.query.sql import _simplify_negation

    lines: list[str] = []
    guarded = [s for s in (controls.get("guarded_by") or []) if isinstance(s, dict)]
    shown: set[str] = set()
    # A guard printed beside the write it gates already says more than the same
    # expression repeated under its own heading.
    rows = [
        row
        for row in guarded
        if _simplify_negation(str(row.get("name") or "")) not in already_cited
    ]
    if rows:
        lines.append("Guarded by")
        for row in rows[:12]:
            name = _simplify_negation(str(row.get("name") or ""))
            count = int(row.get("count") or 1)
            line_nos = [int(n) for n in (row.get("lines") or []) if int(n or 0) > 0]
            shown.add(name)
            bit = f"  {_clip_expr(name)}"
            if count > 1:
                bit += f"  x{count}"
            if line_nos:
                bit += f"  lines {min(line_nos)}-{max(line_nos)}" if len(line_nos) > 1 else f"  {line_nos[0]}"
            lines.append(bit)
    for row in guarded:
        shown.add(_simplify_negation(str(row.get("name") or "")))
    # `Controls` reprinted the same expressions one section lower, which reads
    # as a second, independent fact.
    names = [
        _simplify_negation(str(n))
        for n in (controls.get("controls") or [])
        if n and _simplify_negation(str(n)) not in shown
    ]
    folded = [
        s
        for s in (controls.get("controls_folded") or [])
        if isinstance(s, dict)
        and _simplify_negation(str(s.get("name") or "")) not in shown
    ]
    if names or folded:
        lines.append("Controls")
        for row in folded[:8]:
            name = _simplify_negation(str(row.get("name") or ""))
            count = int(row.get("count") or 1)
            bit = f"  {_clip_expr(name)}"
            if count > 1:
                bit += f"  x{count}"
            lines.append(bit)
        for name in names[:8]:
            lines.append(f"  {_clip_expr(name)}")
    return lines


def _render_assignments(facet: dict[str, Any]) -> list[str]:
    groups = [g for g in (facet.get("groups") or []) if isinstance(g, dict)]
    if not groups:
        return []
    confirmed = int(facet.get("confirmed") or 0)
    unresolved = int(facet.get("unresolved") or 0)
    total = int(facet.get("total") or (confirmed + unresolved))
    lines = [
        "Assignments "
        + _render_proof(confirmed, unresolved, facet.get("exhaustive"), total=total)
    ]
    for group in groups:
        writer = str(group.get("writer") or "")
        if writer:
            lines.append(f"  {writer}")
        for site in group.get("sites") or []:
            if not isinstance(site, dict):
                continue
            line = int(site.get("line") or 0)
            rhs = str(site.get("rhs") or "").strip()
            when = str(site.get("when") or "").strip()
            bit = f"    {line}" if line else "    ?"
            if rhs:
                bit += f" = {rhs}"
            if when:
                bit += f"   when {when}"
            lines.append(bit)
    consumed = _consumer_names(facet.get("consumed_by"))
    if consumed:
        lines.append("Consumed by")
        lines.append("  " + "  ".join(consumed[:8]))
    lines.append("")
    return lines


def _render_host_kernel(facet: dict[str, Any]) -> list[str]:
    producers = [r for r in (facet.get("producers") or []) if isinstance(r, dict)]
    consumers = [r for r in (facet.get("consumers") or []) if isinstance(r, dict)]
    transport = [str(n) for n in (facet.get("transport") or []) if n]
    if not producers and not consumers and not transport:
        return []
    cov = facet.get("coverage") if isinstance(facet.get("coverage"), dict) else {}
    lines: list[str] = []
    if producers:
        lines.append("Host producers")
        for row in producers[:8]:
            lines.append(f"  {row.get('name') or ''}".rstrip())
    if transport:
        lines.append("Transport")
        for name in transport[:6]:
            lines.append(f"  {name}")
    if consumers:
        lines.append("Kernel consumers")
        for row in consumers[:8]:
            lines.append(f"  {row.get('name') or ''}".rstrip())
    pc = int(cov.get("producers_confirmed") or len(producers))
    pu = int(cov.get("producers_unresolved") or 0)
    cc = int(cov.get("consumers_confirmed") or len(consumers))
    cu = int(cov.get("consumers_unresolved") or 0)
    exhaustive = cov.get("exhaustive")
    lines.append("Coverage")
    lines.append(f"  producers: {pc}/{pc + pu}")
    lines.append(f"  consumers: {cc}/{cc + cu}")
    lines.append(f"  exhaustive: {'true' if exhaustive else 'false'}")
    lines.append("")
    return lines


def _render_facets(
    facets: dict[str, Any], *, projection: str = "summary", seed_name: str = ""
) -> list[str]:
    lines: list[str] = []
    resource_facet = facets.get("resource") if isinstance(facets.get("resource"), dict) else {}
    already_backed = {
        str(row.get("name") or "")
        for row in (resource_facet.get("backing") or [])
        if isinstance(row, dict)
    }
    lines.extend(_render_assignments(facets.get("assignments") if isinstance(facets.get("assignments"), dict) else {}))
    lines.extend(_render_host_kernel(facets.get("host_kernel") if isinstance(facets.get("host_kernel"), dict) else {}))
    storage = facets.get("storage") if isinstance(facets.get("storage"), dict) else None
    if storage:
        rows: list[str] = []
        seen_backing: set[str] = set()
        for row in storage.get("backed_by") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            # The resource projection already cites this backing with its real
            # space and the call that allocated it; repeating it here as
            # "UNKNOWN", twice, reads as two weaker findings.
            if not name or name in seen_backing or name in already_backed:
                continue
            seen_backing.add(name)
            space = str(row.get("physical_space") or "")
            if space.upper() == "UNKNOWN":
                space = ""
            rows.append(f"  backed by {name}" + (f"  {space}" if space else ""))
        for typ in storage.get("instance_of") or []:
            rows.append(f"  INSTANCE_OF {typ}")
        if rows:
            lines.append("Storage")
            lines.extend(rows)
            lines.append("")
    controls = facets.get("controls") if isinstance(facets.get("controls"), dict) else None
    if controls:
        lines.append("Controls")
        for name in controls.get("controls") or []:
            lines.append(f"  {name}")
        for name in controls.get("materializes_as") or []:
            lines.append(f"  → {name}")
        lines.append("")
    memory = facets.get("memory") if isinstance(facets.get("memory"), dict) else None
    if memory:
        total = int(memory.get("total") or 0)
        resolved = int(memory.get("resolved") or 0)
        lines.append("Memory")
        if total:
            lines.append(f"{resolved}/{total} transfers resolved")
        flows = memory.get("flows") if isinstance(memory.get("flows"), dict) else {}
        for key, n in list(flows.items())[:8]:
            lines.append(f"  {key}   ×{n}")
        unresolved = int(memory.get("unresolved") or 0)
        if unresolved:
            lines.append(f"{unresolved} unresolved endpoints")
        if memory.get("exhaustive") is not None:
            lines.append(f"exhaustive={'yes' if memory.get('exhaustive') else 'no'}")
        lines.append("")
    used = facets.get("used_by") if isinstance(facets.get("used_by"), dict) else None
    if used:
        keep = set(_consumer_names(used))
        seed_leaf = _last_ident(str(seed_name or "")).lower()
        items = [
            (name, n)
            for name, n in used.items()
            if name in keep and _last_ident(str(name)).lower() != seed_leaf
        ]
        if items:
            cap = 20 if str(projection or "") == "locations" else 8
            head = "Used by"
            if len(items) > cap:
                head += f" ({cap} of {len(items)})"
            lines.append(head)
            for name, n in items[:cap]:
                lines.append(f"  {name} ×{n}")
            lines.append("")
    resource = facets.get("resource") if isinstance(facets.get("resource"), dict) else {}
    lines.extend(_render_resource(resource))
    controls_proj = facets.get("controls_proj") if isinstance(facets.get("controls_proj"), dict) else {}
    lines.extend(_render_control_proj(controls_proj))
    return lines


def _collapse_runs(numbers: list[int]) -> list[str]:
    """``[1381,1382,1383,1385,1386]`` → ``['1381-1383', '1385-1386']``."""
    out: list[str] = []
    start = prev = None
    for value in numbers:
        if start is None:
            start = prev = value
            continue
        if value == prev + 1:
            prev = value
            continue
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = value
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return out


#: An assigned value is echoed from a line the card already prints, so cutting
#: it costs nothing. A guard can live far above the write and be the only place
#: the condition appears, so it gets room to arrive whole.
_EXPR_ECHO_MAX = 96
_GUARD_ECHO_MAX = 240
#: A field's write sites sit in files this card never prints. Nothing echoes
#: them, so an elided right-hand side is the only copy the reader gets, and one
#: cut mid-call sent a reader back for the line it came from.
_REMOTE_EXPR_MAX = 220


def _clip_expr(text: str, limit: int = _EXPR_ECHO_MAX) -> str:
    """Shorten an expression that the Source block above already prints in full."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _clip_guard(text: str) -> str:
    return _clip_expr(text, _GUARD_ECHO_MAX)


# A loop or switch header is not a condition on the site: `for (i < n)` says the
# site runs on some iteration, and `while (true)` says nothing at all. Echoing
# either after "when" reads as a claim about when the code runs that is not true.
_LOOP_GUARD_RE = re.compile(r"^(?:for|while|do|switch|cxx_for_range)\s*\(")


def _when_clauses(values: Any, limit: int = 3) -> list[str]:
    """Guard texts fit to print after "when", conditions only."""
    from ascendc_codemap_mcp.engine.query.sql import _simplify_negation

    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or _LOOP_GUARD_RE.match(text):
            continue
        text = _clip_guard(_simplify_negation(text))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


_NOT_A_CONSUMER = re.compile(r'[()"\']|&&|\|\||==|!=|>=|<=')


def _consumer_names(values: Any) -> list[str]:
    """Consumers, as names. A log call and a predicate are neither.

    "Consumed by" is a list of places to go read next, so an entry has to be
    something a caller can resolve; expression text mixed into the same run-on
    line makes the resolvable names harder to find, not easier.
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or _NOT_A_CONSUMER.search(text) or len(text) > 80:
            continue
        if _is_validation_name(text) or _is_noise_name(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


_IDENTITY_RHS_RE = re.compile(r"^(?:static_cast\s*<[^>]*>\s*)?\(?\s*([A-Za-z_][\w.:>-]*)\s*\)?$")


def _is_identity_rhs(name: str, rhs: str) -> bool:
    """Whether the right-hand side only restates the field being written.

    `IsNEqual ← static_cast<uint8_t>(isNEqual)` carries one fact — the write
    happened here — and spends a line saying the value came from the field of
    the same name. Forty of them in a row buried the two lines that named a
    different source, and a reader nearly missed them.
    """
    match = _IDENTITY_RHS_RE.match(str(rhs or "").strip())
    if match is None:
        return False
    return _last_ident(match.group(1)).lower() == _last_ident(str(name or "")).lower()


def _render_state_changes(changes: Any) -> list[str]:
    rows = [c for c in (changes or []) if isinstance(c, dict) and c.get("name")]
    if not rows:
        return []
    lines = ["State changes"]
    for group in rows:
        name = str(group.get("name") or "")
        lines.append(f"  {name}")
        sites = [s for s in (group.get("sites") or []) if isinstance(s, dict)]
        for site in sites:
            line = int(site.get("line") or 0)
            raw_rhs = str(site.get("rhs") or "")
            rhs = "" if _is_identity_rhs(name, raw_rhs) else _clip_expr(raw_rhs)
            when = _clip_guard(site.get("when"))
            bit = f"    {line}" if line else "    ?"
            if rhs:
                bit += f" ← {rhs}"
            if when:
                bit += f" when {when}"
            lines.append(bit)
    lines.append("")
    return lines


def _base_name(path: str) -> str:
    return str(path or "").replace("\\", "/").rsplit("/", 1)[-1]


def _site_text(pairs: list[tuple[str, int]], home: str) -> str:
    """Where the sites are. A bare line number only when the file is the card's.

    ``Called by X @773`` reads as line 773 of the file on screen, so a caller
    living in another file has to say which, or the reader opens the wrong one.
    """
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for file, line in pairs:
        base = _base_name(file)
        tag = "" if base and base == home else base
        if tag not in grouped:
            grouped[tag] = []
            order.append(tag)
        if line not in grouped[tag]:
            grouped[tag].append(line)
    if order == [""] and len(grouped[""]) == 1:
        return f"@{grouped[''][0]}"
    chunks = []
    for tag in order:
        nums = ", ".join(str(n) for n in grouped[tag])
        chunks.append(f"{tag}:{nums}" if tag else nums)
    return "— " + "; ".join(chunks)


def _call_loc(row: dict[str, Any], *, home: str = "") -> list[str]:
    """One caller or callee, as the lines to print for it.

    A row becomes several lines when its sites disagree about the condition:
    two calls in the two halves of an ``if constexpr`` are one name and two
    answers, and folding them onto one line has to pick a winner, which states
    the wrong polarity for whichever site loses.
    """
    name = str(row.get("name") or "").strip()
    parts = name.replace(".", "::").split("::")
    display = parts[-1] if parts else name
    if display in {"Init", "Set", "Get"} and len(parts) >= 2:
        display = "::".join(parts[-2:])
    home_base = _base_name(home)
    groups: dict[str, list[tuple[str, int]]] = {}
    order: list[str] = []
    for site in row.get("sites") or []:
        if not isinstance(site, dict):
            continue
        line = int(site.get("line") or 0)
        if line <= 0:
            continue
        guard = str(site.get("guard") or "")
        if guard not in groups:
            groups[guard] = []
            order.append(guard)
        pair = (str(site.get("file") or row.get("file") or ""), line)
        if pair not in groups[guard]:
            groups[guard].append(pair)

    def _tail(text: str, guards: Any) -> str:
        when = _when_clauses(guards)
        return text + "  " + "  ".join(f"when {w}" for w in when) if when else text

    if not order:
        line = int(row.get("line") or 0)
        head = f"- {display} @{line}" if line else f"- {display}"
        lines = [_tail(head, row.get("when"))]
    else:
        lines = [
            _tail(f"- {display} {_site_text(groups[guard], home_base)}", [guard])
            for guard in order
        ]
    family = row.get("virtual_dispatch") if isinstance(row.get("virtual_dispatch"), dict) else None
    if family:
        lines = [annotate_call_line(line) for line in lines]
        lines.extend(render_virtual_dispatch(family, home=home, indent="    "))
    return lines


def _render_call_graph(
    calls: dict[str, Any] | None,
    *,
    name: str = "",
    has_body: bool = False,
    home: str = "",
    computed: bool = True,
) -> list[str]:
    """Call edges, and which of four states the empty case is in.

    ``computed=False`` means no caller search was run for the subject of this
    card -- true of a source card, whose call facts belong to whichever entity
    encloses the line, not to anything the reader named. Printing that as "no
    resolved caller" is how ``DoSparse`` came back with no callers on a card
    that had just printed its body, while the graph held
    ``DoOpTiling -> DoSparse @819`` the whole time.
    """
    if not isinstance(calls, dict):
        return []
    lines: list[str] = []
    called_by = [r for r in (calls.get("called_by") or []) if isinstance(r, dict)]
    outgoing = [r for r in (calls.get("calls") or []) if isinstance(r, dict)]
    possible = [r for r in (calls.get("possible_callers") or []) if isinstance(r, dict)]

    def _head(label: str, shown: list[dict[str, Any]], key: str) -> str:
        total = int(calls.get(key) or len(shown))
        return f"**{label}**" if total <= len(shown) else (
            f"**{label} ({len(shown)} of {total})**"
        )

    if called_by:
        lines.append(_head("Called by", called_by, "called_by_total"))
        for row in called_by:
            lines.extend(_call_loc(row, home=home))
        lines.append("")
    elif not computed:
        # A source card that did not search callers used to print
        # "not computed for BN2_MAX_D" using the start-line entity as the
        # subject. That line is noise: omit the section.
        pass
    else:
        # A tiling template's hooks are invoked by the framework base class,
        # which is not in the tree, so their caller list is empty. Dropping the
        # section made that look like a question the card had not been asked,
        # and one reader searched eleven times for the call order before
        # concluding it was unavailable. Printing it only when a body was found
        # left the same gap one step further along: two cards, one saying
        # "nothing calls this" and one silent, with no way to tell whether the
        # silent one meant zero or meant unchecked.
        leaf = _last_ident(name) or name
        subject = f"{leaf} to" if leaf else "it to"
        lines.append("**Called by**")
        # "Never called" is a claim about the whole tree, and an unresolved
        # call site is indistinguishable here from an absent one. State the
        # search that was run instead, so a reader who suspects a caller knows
        # what has not been ruled out.
        note = (
            f"- no resolved caller. No indexed call site binds {subject} a "
            f"definition; that fits a framework hook called from outside the "
            f"snapshot, and equally a call this CodeMap failed to resolve"
            if has_body
            else
            f"- no resolved caller, and no definition either. This CodeMap "
            f"indexes one operator, so a shared header's other users are out "
            f"of range and 'no callers' means none within it"
        )
        lines.append(note)
        lines.append("")
    if outgoing:
        lines.append(_head("Calls", outgoing, "calls_total"))
        for row in outgoing:
            lines.extend(_call_loc(row, home=home))
        lines.append("")
    if possible:
        total = int(calls.get("possible_callers_total") or len(possible))
        seen = f"{len(possible)} of {total} candidates" if total > len(possible) else (
            f"{len(possible)} candidates"
        )
        lines.append(f"**Possible callers ({seen})**")
        for row in possible:
            lines.extend(_call_loc(row, home=home))
        lines.append("")
    family = calls.get("virtual_dispatch") if isinstance(calls.get("virtual_dispatch"), dict) else None
    outgoing_has_family = any(
        isinstance(row.get("virtual_dispatch"), dict) for row in outgoing
    )
    if family and not outgoing_has_family:
        # Callee card: the family is the subject, not one of its callees.
        lines.extend(render_virtual_dispatch(family, home=home, heading=True))
    return lines


def _render_tiling_data_fields(rows: Any) -> list[str]:
    """Each field of a TilingData struct with who fills it and who reads it."""
    items = [r for r in (rows or []) if isinstance(r, dict) and r.get("name")]
    if not items:
        return []
    lines = ["TilingData fields  (host write → transport → kernel read)"]
    for row in items:
        name = str(row.get("name") or "")
        line = int(row.get("line") or 0)
        head = f"  {name}"
        if line:
            head += f"  {line}"
        writes = [str(x) for x in (row.get("writes") or []) if x]
        transport = [str(x) for x in (row.get("transport") or []) if x]
        readers = [str(x) for x in (row.get("readers") or []) if x]
        parts: list[str] = []
        if writes:
            parts.append("host " + ", ".join(writes))
        if transport:
            parts.append("via " + ", ".join(transport))
        if readers:
            parts.append("kernel " + ", ".join(readers))
        if parts:
            head += "  " + "  ".join(parts)
        elif not line:
            continue
        lines.append(head)
    lines.append("")
    return lines


def _render_unit_resources(rows: Any) -> list[str]:
    """Resources declared in this unit, each with the facts its own card shows."""
    items = [r for r in (rows or []) if isinstance(r, dict) and r.get("name")]
    if not items:
        return []
    lines = ["Resources in this unit"]
    for row in items:
        kind = str(row.get("kind") or "")
        name = str(row.get("name") or "")
        line = int(row.get("line") or 0)
        facts = [
            f"{key}={value}"
            for key, value in (row.get("facts") or [])
            if key not in {"scope"}
        ]
        head = f"  {kind} {name}"
        if line:
            head += f"  {line}"
        if facts:
            head += "  " + "  ".join(facts)
        lines.append(head)
    lines.append("")
    return lines


def _render_operations(rows: Any, *, title: str = "Pipeline operations") -> list[str]:
    """What this unit does to the pipeline, by classified call site."""
    items = [r for r in (rows or []) if isinstance(r, dict) and r.get("category")]
    if not items:
        return []
    lines = [title]
    for row in items:
        category = str(row.get("category") or "")
        count = int(row.get("count") or 0)
        split = [c for c in (row.get("by_callee") or []) if isinstance(c, dict)]
        bit = f"  {category} ×{count}"
        if split:
            # The category total answers "how busy"; only the split answers
            # "how many flags", which is the question that gets asked.
            named = ", ".join(
                f"{c.get('name')} ×{int(c.get('count') or 0)}" for c in split[:4]
            )
            rest = len(split) - 4
            bit += "  " + named + (f", +{rest} more" if rest > 0 else "")
        else:
            callees = [str(c) for c in (row.get("callees") or []) if c]
            if callees:
                bit += "  " + ", ".join(callees[:3])
        lines.append(bit)
    lines.append("")
    return lines


def _keep_site_extra(row: dict[str, Any]) -> bool:
    """Whether a neighbour at this line is worth naming.

    CONTRACT constants, log-guard BRANCHes, and predicates that are not an
    identifier were the bulk of "At this site" on source cards, and they
    competed with the enclosing function for the reader's attention.
    """
    kind = str(row.get("kind") or "").upper()
    name = str(row.get("name") or "")
    if kind == EntityKind.CONTRACT.value:
        return False
    if kind == EntityKind.PREDICATE.value:
        return _is_plain_ident(_last_ident(name))
    if kind == EntityKind.BRANCH.value and _is_validation_name(name):
        return False
    return bool(kind or name)


def _render_unit_fields(
    rows: Any,
    *,
    window_start: int = 0,
    window_end: int = 0,
    file: str = "",
) -> list[str]:
    items = [r for r in (rows or []) if isinstance(r, dict)]
    if not items:
        return []
    host_file = _is_host_path(file)
    lo, hi = int(window_start or 0), int(window_end or 0)

    def _in_window(site: dict[str, Any]) -> bool:
        ln = int(site.get("line") or 0)
        if lo > 0 and hi >= lo and ln > 0:
            return lo <= ln <= hi
        return True

    lines = ["Fields in this unit"]
    shown = 0
    for item in items:
        name = str(item.get("name") or "")
        bundle = item.get("bundle") if isinstance(item.get("bundle"), dict) else {}
        host = [
            s
            for s in (bundle.get("host_value_definitions") or [])
            if isinstance(s, dict) and _in_window(s)
        ]
        assignments = [
            s
            for s in (bundle.get("assignments") or [])
            if isinstance(s, dict) and _in_window(s)
        ]
        writes = host or assignments
        consumers = []
        if not host_file:
            consumers = [
                s
                for s in (bundle.get("kernel_consumers") or [])
                if isinstance(s, dict) and _in_window(s)
            ]
        if not writes and not consumers:
            continue
        if name:
            lines.append(f"  {name}")
        for site in writes[:2]:
            cite = _render_site_cite(site).strip()
            fn = str(site.get("function") or "")
            bit = f"    host {cite}"
            if fn and fn not in bit:
                bit += f"  {fn}"
            lines.append(bit)
        for row in consumers[:2]:
            cname = str(row.get("name") or "")
            line = int(row.get("line") or 0)
            lines.append(f"    kernel {cname}  {line}".rstrip())
        shown += 1
    if shown == 0:
        return []
    lines.append("")
    return lines


def _site_coverage_line(
    payload: dict[str, Any], *, ident: str, file: str, line: int
) -> str:
    """Say what a site view covers, and name the half it does not.

    Addressing a target by file and line and addressing it by name return
    different sections of the same subject: the site view carries the state
    changes and the fields, the symbol card carries the definition sites, the
    callers and the guards. Neither contains the other, and only the symbol
    card ever said how complete it was -- so a reader driven here by a clipped
    body or an ambiguous name lost the completeness signal without being told
    it had a second half to ask for.
    """
    if not (file and line):
        return ""
    where = f"the unit around {file}:{line}"
    if not ident:
        return f"Coverage: {where} · lists complete for this unit"
    leaf = _last_ident(ident) or ident
    follow = f" — trace symbol={leaf}" if _is_plain_ident(leaf) else ""
    tail = (
        f"state changes and fields below are every one in this window"
    )
    if ident and _is_plain_ident(leaf):
        tail += (
            f"; name matches and guards for {leaf} are on its symbol card"
            f"{follow}"
        )
    return f"Coverage: {where} · {tail}"


def _render_site_markdown(payload: dict[str, Any], *, projection: str) -> list[str]:
    file = str(payload.get("file") or "")
    line = int(payload.get("line") or 0)
    snippet = str(payload.get("snippet") or "")
    enclosing = payload.get("enclosing") if isinstance(payload.get("enclosing"), dict) else {}
    unit_start = int(payload.get("unit_start") or enclosing.get("line_start") or line or 0)
    unit_end = int(payload.get("unit_end") or enclosing.get("line_end") or line or 0)
    fn_start = int(payload.get("function_start") or enclosing.get("line_start") or 0)
    fn_end = int(payload.get("function_end") or enclosing.get("line_end") or 0)
    ident = _last_ident(str(enclosing.get("name") or ""))
    # CONTRACT / BRANCH / PREDICATE at the requested line are neighbours, never
    # the card's title. The enclosing function name is the identity; a leftover
    # expression in `enclosing.name` is dropped rather than printed as a heading.
    if ident and not _is_plain_ident(ident):
        ident = ""
    lines: list[str] = []
    if ident:
        lines.append(ident)
    identity_start = fn_start or unit_start
    identity_end = fn_end or unit_end
    if file and identity_start and identity_end and identity_end != identity_start:
        lines.append(f"{file}:{identity_start}-{identity_end}")
    elif file and line:
        lines.append(f"{file}:{line}")
    if fn_start and fn_end and (unit_start != fn_start or unit_end != fn_end):
        lines.append(f"Showing {unit_start}-{unit_end}")
    cover = _site_coverage_line(payload, ident=ident, file=file, line=line)
    if cover:
        lines.append(cover)
    if lines:
        lines.append("")
    if snippet:
        lines.append("Source")
        for no, text in _snippet_rows(snippet, line):
            lines.append(f"{no}|  {text}")
        lines.append("")
    lines.extend(_render_state_changes(payload.get("state_changes")))
    site_rows = [c for c in (payload.get("cards") or payload.get("seeds") or []) if isinstance(c, dict)]
    extras: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    ident_leaf = _last_ident(ident).lower()
    for row in site_rows:
        kind = str(row.get("kind") or "")
        name = str(row.get("name") or "")
        key = (kind, name)
        if key in seen or not (kind or name):
            continue
        if ident_leaf and _last_ident(name).lower() == ident_leaf:
            continue
        if not _keep_site_extra(row):
            continue
        seen.add(key)
        extras.append(row)
    if extras:
        lines.append("At this site")
        for row in extras:
            lines.append(f"  {row.get('kind') or ''} {row.get('name') or ''}".rstrip())
        lines.append("")
    calls = payload.get("calls") if isinstance(payload.get("calls"), dict) else None
    if not calls:
        enc = payload.get("enclosing") if isinstance(payload.get("enclosing"), dict) else {}
        facets = enc.get("facets") if isinstance(enc.get("facets"), dict) else {}
        calls = facets.get("calls") if isinstance(facets.get("calls"), dict) else None
    lines.extend(
        _render_call_graph(
            calls,
            name=ident,
            has_body=bool(snippet and fn_start and fn_end >= fn_start),
            home=str(payload.get("file") or ""),
            computed=bool(payload.get("calls_computed")),
        )
    )
    lines.extend(_render_operations(payload.get("operations")))
    lines.extend(_render_unit_resources(payload.get("unit_resources")))
    lines.extend(
        _render_unit_fields(
            payload.get("field_bundles"),
            window_start=unit_start,
            window_end=unit_end,
            file=file,
        )
    )
    hint = str(payload.get("hint") or "").strip()
    if hint:
        lines.append(hint)
    return lines


def _rival_owners(query: Any, primary: dict[str, Any]) -> set[str]:
    """Classes that declare their own member of the same name as the seed.

    References are found by leaf name, so a call to `NzPost::ProcessSink` was
    listed under `S1S2PostRegbase::ProcessSink` -- a reader could only conclude
    the two are connected, which they are not. A site sitting inside a class
    that has its own version is referring to that one.

    Only classes that declare the name are excluded. A caller in an unrelated
    class is a real reference and stays.
    """
    owner_of = getattr(query, "owner_of", None)
    hits_for = getattr(query, "_exact_name_hits", None)
    if not callable(owner_of) or not callable(hits_for):
        return set()
    name = str(primary.get("name") or "")
    leaf = name.replace(".", "::").rsplit("::", 1)[-1]
    if not leaf:
        return set()
    seed = (
        name.replace(".", "::").rsplit("::", 1)[0].rsplit("::", 1)[-1]
        if "::" in name
        else owner_of(str(primary.get("file") or ""), int(primary.get("line") or 0))
    )
    if not seed:
        return set()
    owners: set[str] = set()
    try:
        for hit in hits_for(leaf, limit=32):
            found = owner_of(
                str(hit.get("file") or ""),
                int(hit.get("line_start") or hit.get("line") or 0),
                hit.get("data"),
            )
            if found and found != seed:
                owners.add(found)
    except Exception:
        return set()
    return owners


def _mark_rival_owner_uses(
    query: Any, primary: dict[str, Any], uses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Name the class a reference sits in when that class has its own version.

    References are found by leaf name, so a call to `NzPost::ProcessSink` was
    listed plainly under `S1S2PostRegbase::ProcessSink` and read as a link
    between two unrelated templates.

    Dropping them is the wrong trade. `KernelBase::Init` calling `preSfmg.Init()`
    sits in a class with its own `Init` and is still a real reference, and losing
    a true caller costs more than showing a suspect one. Say which class the
    line is in and let the reader judge.
    """
    rivals = _rival_owners(query, primary)
    owner_at = getattr(query, "owner_of", None)
    if not rivals or not callable(owner_at):
        return uses
    for use in uses:
        owner = owner_at(str(use.get("file") or ""), int(use.get("line") or 0))
        if owner in rivals:
            use["rival_owner"] = owner
    return uses


def _sibling_definition_rows(
    payload: dict[str, Any], card: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every declaration site of this name, one row per file:line.

    The sites hang off whichever card carried the cross-reference extras, and
    that is not always the card that gets rendered: `ping_` resolves to the
    BUFFER, while the sites were attached to the FIELD it was minted from. The
    result was a card claiming three definition sites and listing none. Fall
    back to the matched-entity rows, which carry file and line for every hit.

    Cached on the payload because the card is rendered once with its extras
    and again from the slimmed copy.
    """
    rows = payload.get("other_definitions")
    if isinstance(rows, list):
        return rows
    extras = card.get("extras") if isinstance(card.get("extras"), dict) else {}
    sites = [s for s in (extras.get("definition_sites") or []) if isinstance(s, dict)]
    if not sites:
        sites = [s for s in (payload.get("matched_entities") or []) if isinstance(s, dict)]
    rows = []
    seen: set[tuple[str, int]] = set()
    for site in sites:
        file = str(site.get("file") or "")
        line = int(site.get("line") or 0)
        if not file or (file, line) in seen:
            continue
        seen.add((file, line))
        rows.append({"file": file, "line": line, "name": str(site.get("name") or "")})
    payload["other_definitions"] = rows
    return rows


def _render_sibling_definitions(
    payload: dict[str, Any],
    card: dict[str, Any],
    *,
    shown_file: str,
    shown_line: int,
) -> list[str]:
    """The definitions of this name that the card did not open.

    A name defined on both sides of the boundary resolves to whichever site
    ranked first, and the coverage line counted the others without saying
    where they were. A reader after the kernel copy of `CalTNDDenseIndex` got
    the host one, with nothing to indicate a kernel one existed.

    The card is rendered once with its cross-reference extras and again from
    the slimmed payload, so the list is cached where slimming can keep it.
    """
    rows = _sibling_definition_rows(payload, card)
    leaf = _last_ident(str(card.get("name") or ""))
    here = (str(shown_file or ""), int(shown_line or 0))
    calls = payload.get("calls") if isinstance(payload.get("calls"), dict) else {}
    family = calls.get("virtual_dispatch") if isinstance(calls.get("virtual_dispatch"), dict) else None
    if family is None:
        facets = card.get("facets") if isinstance(card.get("facets"), dict) else {}
        faceted = facets.get("calls") if isinstance(facets.get("calls"), dict) else {}
        family = faceted.get("virtual_dispatch") if isinstance(faceted.get("virtual_dispatch"), dict) else None
    skip = family_sites(family)
    rest = [
        r for r in rows
        if isinstance(r, dict)
        and (_norm_path(str(r.get("file") or "")), int(r.get("line") or 0)) != (
            _norm_path(here[0]),
            here[1],
        )
        and (_norm_path(str(r.get("file") or "")), int(r.get("line") or 0)) not in skip
    ]
    if not rest:
        return []
    lines = ["**Other definitions of this name**"]
    for row, extra in _fold_by_file(rest):
        file = str(row.get("file") or "")
        line = int(row.get("line") or 0)
        name = str(row.get("name") or "")
        qualifier = f"  {name}" if name and _last_ident(name) != leaf else ""
        more = f"  (+{extra} more in this file)" if extra else ""
        lines.append(f"- {file}:{line}{qualifier}{more}   source file={file} line={line}")
    lines.append("")
    return lines


def _fold_by_file(rows: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    """One row per file, carrying how many more it stands for.

    A flag written on eight consecutive lines of one translation unit is one
    fact about that unit, not eight findings. Listing each separately pushed the
    write tree below eleven rows of the same file and cost the reader the part
    of the card that answers the question.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_norm_path(str(row.get("file") or "")), []).append(row)
    out: list[tuple[dict[str, Any], int]] = []
    for group in groups.values():
        head = min(group, key=lambda r: int(r.get("line") or 0))
        out.append((head, len(group) - 1))
    return out


def _matched_lookalikes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Entities the name reached that are neither the card nor its definitions.

    The coverage count is only checkable against the list behind it, so the
    count and the list have to be decided together: a card that says "listed
    below" and then lists nothing is the same broken promise in a new place.
    """
    rows = [r for r in (payload.get("matched_entities") or []) if isinstance(r, dict)]
    if len(rows) < 2:
        return []
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    card = cards[0] if cards else {}
    leaf = _last_ident(str(card.get("name") or "")).lower()
    here = (
        str(card.get("name") or ""),
        _norm_path(str(card.get("file") or "")),
        int(card.get("line") or card.get("line_start") or 0),
    )
    # Where the definition list will carry these rows, this heading would
    # repeat them and say something weaker about them. Ask the list, not the
    # count: the count came from a different card than the one being rendered,
    # so trusting it suppressed this heading on cards that listed nothing.
    here_site = (here[1], here[2])
    if [
        r
        for r in _sibling_definition_rows(payload, card)
        if (_norm_path(str(r.get("file") or "")), int(r.get("line") or 0)) != here_site
    ]:
        return []
    rest: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name") or "")
        file = _norm_path(str(row.get("file") or ""))
        line = int(row.get("line") or 0)
        if (name, file, line) == here:
            continue
        # A second row for the same name with nowhere to go is the primary
        # seen through another index, not a look-alike worth reporting.
        if not file and _last_ident(name).lower() == leaf:
            continue
        rest.append(row)
    return rest


def _render_matched_entities(payload: dict[str, Any]) -> list[str]:
    """Name the look-alikes, so "did I get the right one" needs no search."""
    rest = _matched_lookalikes(payload)
    if not rest:
        return []
    lines = ["**Other entities matching this name**"]
    for row in rest:
        name = str(row.get("name") or "")
        kind = str(row.get("kind") or "")
        file = str(row.get("file") or "")
        line = int(row.get("line") or 0)
        loc = f"  {file}:{line}" if file and line else (f"  {file}" if file else "")
        lines.append(f"- {name}  {kind}{loc}".rstrip())
    lines.append("")
    return lines


def _payload_has_definition(payload: dict[str, Any]) -> bool:
    """Whether this card can show a body, as opposed to only use sites.

    An entity minted from a call or a base-class clause carries a name and a
    kind but no file, because the declaration it refers to was never in the
    snapshot. The card rendered identically to a resolved one either way.
    """
    if [c for c in (payload.get("candidate_sources") or []) if isinstance(c, dict)]:
        return True
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    if not cards:
        return True
    return bool(str(cards[0].get("file") or "").strip())


def _coverage_line(payload: dict[str, Any]) -> str:
    """State whether the lists below are all of them.

    Reviewers cannot cite "these are the only writers" from a list that might
    be a first page, so they either re-derive it with a text search or hedge the
    conclusion. The card already knows which case it is -- `coverage` records
    whether siblings were checked, and the fitting pass records what it shed --
    but none of it was rendered, so every answer looked equally provisional.
    """
    cov = payload.get("coverage")
    cov = cov if isinstance(cov, dict) else {}
    sites = int(cov.get("definition_sites_count") or 0)
    sites_complete = bool(cov.get("definition_sites_complete"))
    state = str(cov.get("completeness") or "")
    parts: list[str] = []
    # "lists complete" over a card that never had a body reads as "this is all
    # there is to know about the name", when what happened is that the tree
    # holds no definition at all. Say which of the two it is first.
    if not _payload_has_definition(payload):
        return (
            "Coverage: no definition in this CodeMap — the name is reached only "
            "through the use sites below, so its body is outside the indexed "
            "tree. Nothing further to resolve; read the header in the source "
            "repository if the body matters."
        )
    lookalikes = _matched_lookalikes(payload)
    if sites <= 1 and lookalikes:
        # One definition plus look-alikes is not the same fact as several
        # definitions, and the reader cannot tell which they have without the
        # names: `IsEmptyOutput` read as three definitions of itself until two
        # of them turned out to be branches on the call.
        parts.append(
            f"1 definition · {len(lookalikes)} other entities matched the name, "
            f"listed below"
        )
    elif sites > 1:
        if sites_complete:
            parts.append(f"{sites} name matches (all listed)")
        else:
            # Saying a list is partial without saying how to finish it just
            # moves the guesswork. Search pages to completion and now declares
            # when it has reached it.
            seed = str(payload.get("explore_pattern") or payload.get("pattern") or "").strip()
            hint = f"; search pattern={seed} for all" if seed else ""
            parts.append(f"{sites} name matches (first page only{hint})")
    elif sites == 1 and sites_complete:
        # `first_hit` on a name with one definition means unique, not unchecked.
        parts.append("the only definition of this name")
    elif state == "first_hit":
        parts.append("first match only; siblings not checked")
    elif state == "siblings_checked":
        parts.append("siblings checked")
    # Naming the list that was cut is the whole value of admitting the cut. An
    # unnamed admission reads as "distrust everything here", and one reader
    # rebuilt a caller list that had arrived complete.
    shed = [str(x) for x in (payload.get("trimmed_lists") or []) if str(x).strip()]
    if shed:
        parts.append(f"{', '.join(shed[:3])} trimmed to fit; the rest complete")
    elif payload.get("context_trimmed"):
        parts.append("cross-reference lists trimmed to fit")
    elif payload.get("truncated"):
        parts.append("some lists trimmed to fit")
    else:
        # "Lists complete" was read as a warranty over every section on the
        # card, and it is not one: it means nothing was dropped to fit a size
        # budget. A section with its own cap says so in its own heading, so
        # name the scope here and let those headings speak for themselves.
        parts.append("nothing dropped to fit; a capped section says so in its heading")
    return "Coverage: " + " · ".join(parts) if parts else ""


def _render_resolve_markdown(payload: dict[str, Any], *, projection: str) -> list[str]:
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    primary = cards[0] if cards else {}
    seed_name = str(primary.get("name") or "")
    lines: list[str] = []
    file, line, name, snippet = _site_line(primary)
    kind = str(primary.get("kind") or "")
    if seed_name:
        lines.append(seed_name)
        if kind:
            lines.append(kind)
        if file and line:
            lines.append(f"{file}:{line}")
        lines.append("")
    # Whatever made this answer approximate belongs above the answer, not in a
    # structured field the reader never sees.
    note = str(payload.get("match_note") or "").strip()
    if note:
        lines.append(f"> {note}")
        lines.append("")
    coverage = _coverage_line(payload)
    if coverage:
        lines.append(coverage)
        lines.append("")
    candidates = [c for c in (payload.get("candidates") or []) if isinstance(c, dict)]
    cand_src = [c for c in (payload.get("candidate_sources") or []) if isinstance(c, dict)]
    if candidates:
        lines.append("**Candidates** (resolve one)")
        for cand in candidates:
            ck = str(cand.get("kind") or "")
            cfile = str(cand.get("file") or "")
            suffix = f"  {ck}" + (f"  {cfile}" if cfile else "")
            lines.append(f"- {cand.get('name')}{suffix}")
        lines.append("")
    if cand_src:
        lines.append("**Definition**")
        for row in cand_src:
            cfile = str(row.get("file") or "")
            if cfile:
                lines.append(f"{FILE_SECTION_PREFIX}{cfile}")
            for no, text in _snippet_rows(str(row.get("snippet") or ""), int(row.get("line") or 0)):
                lines.append(f"{no}|  {text}")
        lines.append("")
    elif primary and str(projection or payload.get("projection") or "summary") != "locations":
        if file:
            # Tightening around the anchor is right for a hit inside a big span,
            # but a definition card *is* the read: clipping it to a few lines is
            # what forced a second resolve(file, line) to see the body.
            is_definition = kind in {
                EntityKind.FUNCTION.value,
                EntityKind.METHOD.value,
                EntityKind.KERNEL.value,
                EntityKind.TYPE.value,
            }
            src = _render_source(
                [{"file": file, "line": line, "name": name, "snippet": snippet}],
                tight=str(projection or "") == "summary" and not is_definition,
            )
            if is_definition:
                src.extend(_definition_coverage_note(primary, src))
            lines.extend(["**Definition**" if ln == "**Source**" else ln for ln in src])
    siblings = _render_sibling_definitions(
        payload, primary, shown_file=file, shown_line=line
    )
    lines.extend(siblings)
    if not siblings:
        lines.extend(_render_matched_entities(payload))
    calls = payload.get("calls") if isinstance(payload.get("calls"), dict) else None
    if not calls:
        facets = primary.get("facets") if isinstance(primary.get("facets"), dict) else {}
        calls = facets.get("calls") if isinstance(facets.get("calls"), dict) else None
    lines.extend(
        _render_call_graph(
            calls,
            name=seed_name,
            has_body=_payload_has_definition(payload),
            home=str(primary.get("file") or ""),
        )
    )
    extras = primary.get("extras") if isinstance(primary.get("extras"), dict) else {}
    writers = extras.get("writers") or primary.get("writers") or []
    readers = extras.get("readers") or primary.get("readers") or []
    support = payload.get("compiled_support")
    if not isinstance(support, dict):
        support = (primary.get("facets") or {}).get("compiled_support") if isinstance(primary.get("facets"), dict) else None
    assignments = payload.get("assignments") if isinstance(payload.get("assignments"), dict) else None
    if not assignments:
        assignments = (primary.get("facets") or {}).get("assignments") if isinstance(primary.get("facets"), dict) else None
    host_kernel = payload.get("host_kernel") if isinstance(payload.get("host_kernel"), dict) else None
    if not host_kernel:
        host_kernel = (primary.get("facets") or {}).get("host_kernel") if isinstance(primary.get("facets"), dict) else None
    bundle = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else None
    if not bundle:
        bundle = (primary.get("facets") or {}).get("bundle") if isinstance(primary.get("facets"), dict) else None
    if isinstance(bundle, dict) and bundle:
        lines.extend(_render_bundle(bundle))
    else:
        if isinstance(assignments, dict):
            lines.extend(_render_assignments(assignments))
        if isinstance(host_kernel, dict):
            lines.extend(_render_host_kernel(host_kernel))
    host_from_compiled = bool(
        isinstance(support, dict) and support.get("host_encoding")
    )
    if writers and not host_from_compiled and not host_kernel:
        lines.append("Host")
        for row in _dedup_rows([r for r in writers if isinstance(r, dict)])[:3]:
            lines.append(_loc(row))
        lines.append("")
    if readers and not host_kernel:
        lines.append("Kernel")
        for row in _dedup_rows([r for r in readers if isinstance(r, dict)])[:3]:
            lines.append(_loc(row))
        lines.append("")
    used = _dedup_rows(r for r in (payload.get("used_at") or []) if isinstance(r, dict))
    used = [
        row
        for row in used
        if not _is_unrelated_type_neighbor(seed_name, row)
        and not _is_tpl_machinery(str(row.get("name") or ""))
        and not _is_noise_name(str(row.get("name") or ""))
    ]
    if used:
        lines.append("**References**")
        # Eight rows that were really one call site spread over eight argument
        # lines read as eight independent places to go look.
        by_file: dict[str, list[int]] = {}
        rival_by_file: dict[str, set[str]] = {}
        for row in used:
            file, line, _, _ = _site_line(row)
            if file and line:
                by_file.setdefault(file, []).append(int(line))
                rival = str(row.get("rival_owner") or "")
                if rival:
                    rival_by_file.setdefault(file, set()).add(rival)
        total = sum(len(v) for v in by_file.values())
        leaf = seed_name.replace(".", "::").rsplit("::", 1)[-1]
        for file, line_nos in list(by_file.items())[:8]:
            spans = _collapse_runs(sorted(set(line_nos)))
            rivals = sorted(rival_by_file.get(file) or ())
            verb = "declare their own" if len(rivals) > 1 else "declares its own"
            note = f"  (in {', '.join(rivals)}, which {verb} {leaf})" if rivals else ""
            lines.append(f"- {file}:{', '.join(spans)}{note}")
        if len(by_file) > 8:
            lines.append(f"{total} references across {len(by_file)} files")
        lines.append("")
    dim_names = [str(n).strip() for n in (payload.get("dim_names") or []) if str(n).strip()]
    if isinstance(support, dict) and support:
        lines.append("Compiled")
        legal = "yes" if support.get("legal") else "no"
        dim = str(support.get("dim") or "")
        variants = support.get("variants")
        bits = [f"legal={legal}"]
        if variants not in (None, ""):
            bits.append(f"variants={variants}")
        if dim:
            bits.append(f"dim={dim}")
        checked = support.get("checked") or support.get("legal_key_count") or variants
        if checked not in (None, ""):
            bits.append(f"{checked} legal keys checked")
        lines.append("- " + " · ".join(bits))
        values = support.get("values") if isinstance(support.get("values"), dict) else {}
        if values:
            parts = [f"{k}: {v}" for k, v in list(values.items())[:12]]
            lines.append(f"- values: {{{', '.join(parts)}}}")
        host_rows = [r for r in (support.get("host_encoding") or []) if isinstance(r, dict)]
        if host_rows:
            lines.append("Host encoding")
            for row in host_rows[:4]:
                loc = _loc(row)
                expr = str(row.get("expr") or "")
                lines.append(f"- {loc}" + (f"  {expr}" if expr else ""))
        kernel = support.get("kernel") if isinstance(support.get("kernel"), dict) else {}
        if kernel:
            lines.append("Kernel specialization")
            for kdim, cmap in list(kernel.items())[:8]:
                if isinstance(cmap, dict):
                    parts = [f"{k}: {v}" for k, v in list(cmap.items())[:8]]
                    lines.append(f"- {kdim}: {{{', '.join(parts)}}}")
                elif isinstance(cmap, list):
                    lines.append(f"- {kdim}: {{{', '.join(str(v) for v in cmap[:8])}}}")
        cf = support.get("counterfactual") if isinstance(support.get("counterfactual"), dict) else {}
        if cf:
            cf_legal = "yes" if cf.get("legal") else "no"
            conds = [
                f"{k}={v}"
                for k, v in cf.items()
                if k not in {"legal", "variants"}
            ]
            lines.append(f"Counterfactual: {', '.join(conds)} → legal={cf_legal}")
        lines.append("")
    if dim_names and not seed_name:
        lines.append("**Dims**")
        lines.append("- " + ", ".join(dim_names))
        lines.append("")
    facets = dict(primary.get("facets") or {}) if isinstance(primary.get("facets"), dict) else {}
    facets.pop("assignments", None)
    facets.pop("host_kernel", None)
    facets.pop("bundle", None)
    facets.pop("calls", None)
    if bundle:
        facets.pop("resource", None)
        facets.pop("controls_proj", None)
    lines.extend(_render_operations(payload.get("operations")))
    lines.extend(
        _render_operations(
            payload.get("delegated_operations"),
            title="Pipeline operations via callees (depth 1)",
        )
    )
    lines.extend(_render_unit_resources(payload.get("unit_resources")))
    lines.extend(_render_tiling_data_fields(payload.get("tiling_data_fields")))
    lines.extend(_render_facets(facets, projection=projection, seed_name=seed_name))
    lines.extend(_render_other_identities(cards[1:]))
    hint = str(payload.get("hint") or "")
    if hint:
        lines.extend([hint, ""])
    return lines


def _definition_coverage_note(card: dict[str, Any], rendered: list[str]) -> list[str]:
    """Say so when the body shown is a window, not the definition.

    Fitting the card to a size budget can cut the body down to a handful of
    lines. Without this the reader sees a short function and moves on, which is
    a worse failure than returning nothing.
    """
    start = int(card.get("line_start") or card.get("line") or 0)
    end = int(card.get("line_end") or 0)
    if start <= 0 or end <= start:
        return []
    span = end - start + 1
    numbers = [
        int(m.group(1))
        for m in (_NUMBERED_LINE_RE.match(row) for row in rendered)
        if m
    ]
    shown = len(numbers)
    if shown <= 0 or shown >= span:
        return []
    # "any part of 2379-2663" invites a line the reader already has, and a site
    # resolve answers with the same window it just returned. One reader spent
    # eight calls that way; another stopped at the cut. Name the line that
    # begins what is missing.
    resume = max(numbers) + 1 if numbers else start
    tail = end - resume + 1
    where = f"source file={card.get('file') or ''} line={resume}"
    rest = (
        # Naming the end as well takes the tail in one call. Without it the
        # reader re-enters at the resume line and gets another window, which is
        # why one of them gave up on the last forty lines rather than page.
        f"{where} line_end={end} for the remaining {tail} lines ({resume}-{end})"
        if resume <= end
        else f"{where} for the rest"
    )
    return [f"(showing {shown} of {span} lines; {rest})", ""]


_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\|")


def _render_other_identities(cards: list[dict[str, Any]]) -> list[str]:
    """Resource and sync facts carried by the same name under another kind.

    One name is often both a plain field declaration and the event that field
    identifies. Whichever card ranks first, the semantics live on the other one,
    so picking a winner loses them; both are stated instead.
    """
    blocks: list[str] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        facets = card.get("facets") if isinstance(card.get("facets"), dict) else {}
        resource = facets.get("resource") if isinstance(facets.get("resource"), dict) else {}
        rendered = _render_resource(resource)
        if not rendered:
            continue
        kind = str(card.get("kind") or "")
        file = str(card.get("file") or "")
        line = int(card.get("line") or 0)
        head = f"Also {kind}" if kind else "Also"
        if file and line:
            head += f"  {file}:{line}"
        blocks.append(head)
        blocks.extend(("  " + row) if row.strip() else row for row in rendered)
    if blocks:
        blocks.append("")
    return blocks


def _render_contract_markdown(payload: dict[str, Any]) -> list[str]:
    contract = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    seed = contract.get("seed") if isinstance(contract.get("seed"), dict) else (cards[0] if cards else {})
    seed_name = str(seed.get("name") or "")
    producers = _useful_rows(contract.get("producers") or [], seed_name)
    consumers = _useful_rows(contract.get("consumers") or [], seed_name)
    if not producers:
        producers = _dedup_rows(
            [row for row in (contract.get("producers") or []) if isinstance(row, dict) and _has_loc(row)]
        )
    if not consumers:
        consumers = _dedup_rows(
            [row for row in (contract.get("consumers") or []) if isinstance(row, dict) and _has_loc(row)]
        )
    if not producers:
        for card in cards:
            extras = card.get("extras") if isinstance(card.get("extras"), dict) else {}
            producers.extend(
                _useful_rows(extras.get("writers") or card.get("writers") or [], seed_name)
            )
        producers = _dedup_rows(producers)
    if not consumers:
        for card in cards:
            extras = card.get("extras") if isinstance(card.get("extras"), dict) else {}
            consumers.extend(
                _useful_rows(extras.get("readers") or card.get("readers") or [], seed_name)
            )
        consumers = _dedup_rows(consumers)
    lines = ["**Contract**", ""]
    if producers:
        lines.append("Host")
        for row in producers[:3]:
            file, line, name, snippet = _site_line(row)
            lines.append(f"{name} ({file}:{line})" if file and line else name)
            if snippet:
                rows = _snippet_rows(snippet, line)
                if rows:
                    pick = next(
                        (pair for pair in rows if seed_name and seed_name in pair[1]),
                        rows[0],
                    )
                    lines.append(f"{pick[0]}|  {pick[1]}")
        lines.append("        │")
        lines.append("        ▼")
    file, line, name, _ = _site_line(seed)
    lines.append("TilingKey")
    lines.append(f"{name} · {file}:{line}" if file and line else name)
    seed_file, seed_line = file, line
    kernel_rows: list[dict[str, Any]] = []
    for row in consumers:
        cfile, cline, cname, _ = _site_line(row)
        if cfile and cline and (cfile, cline) == (seed_file, seed_line):
            continue
        if _is_host_path(cfile):
            continue
        kernel_rows.append(row)
    if kernel_rows:
        lines.append("        │")
        lines.append("        ▼")
        lines.append("Kernel")
        for row in kernel_rows[:8]:
            cfile, cline, cname, _ = _site_line(row)
            if cfile and cline:
                lines.append(f"  {cfile}:{cline}")
            elif cname:
                lines.append(f"  {cname}")
    lines.append("")
    hint = str(payload.get("hint") or "")
    if hint:
        lines.extend([hint, ""])
    return lines


def _render_impact_markdown(payload: dict[str, Any]) -> list[str]:
    contract = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    seed_name = str((cards[0] if cards else {}).get("name") or "")
    producers = _useful_rows(contract.get("producers") or [], seed_name)
    consumers = _useful_rows(contract.get("consumers") or [], seed_name)
    lines = ["**Impact**"]
    if producers:
        lines.append("- writes: " + ", ".join(_loc(r) for r in producers[:8]))
    if consumers:
        lines.append("- reads: " + ", ".join(_loc(r) for r in consumers[:8]))
    if not producers and not consumers:
        lines.append("(no located writers/readers)")
    lines.append("")
    return lines


def _rel_summary(rels: list[str]) -> str:
    """Name the relation set without pasting fourteen enum members."""
    if not rels:
        return "all relations"
    if len(rels) == 1 and "," not in rels[0]:
        return rels[0]
    if any("," in r for r in rels):
        return "relation kinds"
    return f"{len(rels)} relation kinds including CALLS, WRITES, READS"


def _trace_endpoint(step: dict[str, Any], end: str) -> str:
    name = _last_ident(str(step.get(f"{end}_name") or "")) or str(step.get(end) or "?")
    file = str(step.get(f"{end}_file") or "")
    line = int(step.get(f"{end}_line") or 0)
    where = f"  {file}:{line}" if file and line else (f"  {file}" if file else "")
    return f"{name}{where}"


def _render_trace_hops(steps: list[dict[str, Any]]) -> list[str]:
    lines = [f"**{len(steps)} hop{'s' if len(steps) != 1 else ''}**"]
    for i, step in enumerate(steps, start=1):
        kind = str(step.get("kind") or "?")
        lines.append(
            f"{i}. {_trace_endpoint(step, 'from')}  —{kind}→  {_trace_endpoint(step, 'to')}"
        )
    return lines


def _family_heading(entry: dict[str, Any]) -> str:
    family = str(entry.get("family") or "path")
    role = str(entry.get("role") or "")
    if family == "control" or role == "weak":
        if role and role not in {"weak", ""}:
            return f"{family} ({role}, weak)"
        return f"{family}  weak"
    if role:
        return f"{family} ({role})"
    return family


def _render_trace_markdown(payload: dict[str, Any]) -> list[str]:
    """A path between two names, plus what the walk did and did not cover."""
    src = str(payload.get("trace_from") or "")
    dst = str(payload.get("trace_to") or "")
    family_paths = [s for s in (payload.get("family_paths") or []) if isinstance(s, dict)]
    steps = [s for s in (payload.get("path") or []) if isinstance(s, dict)]
    reason = str(payload.get("unresolved_reason") or "")
    explored = int(payload.get("explored") or 0)
    budget = int(payload.get("node_budget") or 0)
    rels = [str(k) for k in (payload.get("trace_relations") or []) if k]
    lines: list[str] = [f"Path  {src} → {dst}".rstrip()]
    lines.append("")
    if reason == "NO_SEED":
        unknown = [str(x) for x in (payload.get("unknown_endpoints") or []) if x]
        lines.append(
            "**No path** — not a name in this CodeMap: " + ", ".join(unknown)
            if unknown
            else "**No path** — endpoint not in this CodeMap"
        )
        lines.append("")
        lines.append(f"Search this snapshot for the name first: search pattern={unknown[0]}"
                     if unknown else "")
        return [ln for ln in lines if ln is not None]
    depth = int(payload.get("max_depth") or 0)
    if family_paths:
        by_fam: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for entry in family_paths:
            fam = str(entry.get("family") or "path")
            if fam not in by_fam:
                order.append(fam)
                by_fam[fam] = []
            by_fam[fam].append(entry)
        for fam in order:
            entries = by_fam[fam]
            found = [e for e in entries if e.get("found") and e.get("steps")]
            if not found:
                lines.append(fam)
                lines.append("  no path")
                lines.append("")
                continue
            for entry in found:
                lines.append(_family_heading(entry))
                lines.extend(_render_trace_hops(list(entry.get("steps") or [])))
                lines.append("")
        lines.append(
            f"Coverage: directed family walk, not all simple paths; "
            f"{explored} nodes enumerated"
            + ("" if payload.get("exhausted") else f", bounded at {budget} nodes / depth {depth}")
            + ". Pick a hop name and trace symbol= for its card."
        )
        lines.append("")
        return lines
    if not steps:
        if reason == "SEARCH_BUDGET":
            lines.append(
                f"**Not proven** — the walk stopped at {explored} nodes / depth "
                f"{depth} without reaching {dst}. That is where the search ended, "
                f"not a proof that no path exists. Narrow it with relation=, or "
                f"trace to an intermediate name."
            )
        else:
            lines.append(
                f"**No path** — every node reachable from {src} over "
                f"{_rel_summary(rels)} was enumerated ({explored} of them) and "
                f"none is {dst}."
            )
        lines.append("")
        return lines
    lines.extend(_render_trace_hops(steps))
    lines.append("")
    lines.append(
        f"Coverage: directed family walk, not all simple paths; "
        f"{explored} nodes enumerated"
        + ("" if payload.get("exhausted") else f", bounded at {budget} nodes / depth {depth}")
        + ". Pick a hop name and trace symbol= for its card."
    )
    lines.append("")
    return lines


def render_explore_markdown(
    payload: dict[str, Any],
    *,
    verdict: str = "",
    layer: str = "",
    projection: str = "summary",
) -> str:
    completeness = str(payload.get("completeness") or "")
    header = "  ".join(
        part
        for part in (
            f"verdict: {verdict}" if verdict else "",
            f"layer: {layer}" if layer else "",
            f"completeness: {completeness}" if completeness else "",
        )
        if part
    )
    op = str(payload.get("operation") or "")
    shape = str(payload.get("shape") or "")
    quiet_success = not payload.get("error_code") and completeness not in {
        "AMBIGUOUS",
        "INCOMPLETE",
    }
    if quiet_success:
        header = ""
    if op == "search" or shape == "search":
        body = _render_search_markdown(payload)
    elif shape == "trace":
        body = _render_trace_markdown(payload)
    elif payload.get("resolve_mode") == "site" or shape == "around":
        body = _render_site_markdown(payload, projection=projection)
    elif _is_name_list(payload) or op == "find":
        body = _render_find_markdown(payload)
    elif op == "contract":
        body = _render_contract_markdown(payload)
    elif op == "impact":
        body = _render_impact_markdown(payload)
    else:
        body = _render_resolve_markdown(payload, projection=projection)
    extra: list[str] = []
    dim_cov = payload.get("dim_coverage")
    if isinstance(dim_cov, dict) and dim_cov:
        extra.append("**Dim**")
        counts_by_dim = (
            payload.get("dim_value_counts")
            if isinstance(payload.get("dim_value_counts"), dict)
            else {}
        )
        for dim, values in dim_cov.items():
            shown = list(values) if isinstance(values, list) else [values]
            dim_counts = (
                counts_by_dim.get(dim) if isinstance(counts_by_dim.get(dim), dict) else {}
            )
            if not shown and dim_counts:
                shown = list(dim_counts.keys())
            if dim_counts:
                parts = []
                for v in shown:
                    n = dim_counts.get(str(v), dim_counts.get(v))
                    parts.append(f"{v}: {n}" if n is not None else str(v))
                extra.append(f"- {dim}: {{{', '.join(parts)}}}")
            else:
                extra.append(f"- {dim}: {{{', '.join(str(v) for v in shown)}}}")
        if payload.get("legal_key_count") not in (None, ""):
            extra.append(f"- legal_key_count: {payload.get('legal_key_count')}")
        extra.append("")
    cross = payload.get("cross_counts")
    if isinstance(cross, dict) and cross:
        extra.append("**Cross**")
        for sdim, cmap in cross.items():
            if not isinstance(cmap, dict):
                continue
            parts = [f"{v}: {n}" for v, n in list(cmap.items())[:12]]
            extra.append(f"- {sdim}: {{{', '.join(parts)}}}")
        pair = payload.get("cross_pair") if isinstance(payload.get("cross_pair"), dict) else {}
        cells = pair.get("cells") if isinstance(pair.get("cells"), list) else []
        if cells:
            da = str(pair.get("dim_a") or "")
            db = str(pair.get("dim_b") or "")
            extra.append(f"- {da} × {db}")
            for cell in cells[:8]:
                if not isinstance(cell, dict):
                    continue
                va = cell.get("value")
                cmap = cell.get("counts") if isinstance(cell.get("counts"), dict) else {}
                parts = [f"{vb}: {n}" for vb, n in list(cmap.items())[:8]]
                extra.append(f"  {va} × {{{', '.join(parts)}}}")
        extra.append("")
    lines = ([header, ""] if header else []) + extra + body
    text = "\n".join(lines).rstrip() + "\n"
    families = payload.get("relation_families")
    if families:
        text = _narrow_to_families(text, frozenset(str(f) for f in families))
    return cut_explore_text(text)


#: Which family each section of a card belongs to. A section absent from this
#: map is identity (name, location, coverage) and survives every narrowing:
#: a filter is allowed to drop evidence, never to drop what the card is about.
_SECTION_FAMILY: dict[str, str] = {
    "Writes": "data",
    "Host value definitions": "data",
    "Transport": "data",
    "Accessors": "data",
    "Kernel consumers": "data",
    "Consumed by": "data",
    "Reads": "data",
    "Workspace layout": "data",
    "Calls": "call",
    "Called by": "call",
    "Possible callers": "call",
    "Call graph": "call",
    "Guarded by": "control",
    "Controls": "control",
    "Active under": "control",
    "Compiled": "compile",
    "Host encoding": "compile",
    "Kernel specialization": "compile",
    "Dim": "compile",
    "Cross": "compile",
}


def _narrow_to_families(text: str, families: frozenset[str]) -> str:
    """Drop sections outside the named families, and say which were dropped.

    A narrowed answer that looks identical to a complete one is the same trap as
    rendering "not computed" as "none": the reader cannot tell a symbol with no
    writes from a question that never asked about writes. So the sections come
    off and the names of the families holding them go on.
    """
    out: list[str] = []
    withheld: set[str] = set()
    skipping = False
    for line in text.split("\n"):
        label = line.strip().strip("*# ").split("  ")[0].strip()
        family = _SECTION_FAMILY.get(label)
        if family is not None:
            skipping = family not in families
            if skipping:
                withheld.add(family)
                continue
        elif skipping:
            # A section runs until the next heading; a blank line inside one
            # does not end it, but an unindented non-heading line does.
            if line and not line.startswith((" ", "-", "|", "\t")):
                skipping = False
            else:
                continue
        if not skipping:
            out.append(line)
    if withheld:
        out.append("")
        out.append(
            f"Withheld by relation filter: {', '.join(sorted(withheld))}. "
            f"Drop relation= to see every family."
        )
    return "\n".join(out).rstrip() + "\n"


#: Kinds that actually carry a host → TilingKey → kernel contract. Only these
#: can be graded by the C1–C8 fence. A bare FIELD / TYPE alias is not transport.
_CONTRACT_KINDS = frozenset(
    {
        EntityKind.TILING_FIELD.value,
        EntityKind.TILING_KEY.value,
        EntityKind.TEMPLATE_ARG.value,
        EntityKind.COMPILE_VAR.value,
        EntityKind.INPUT.value,
        EntityKind.OUTPUT.value,
    }
)
_TRANSPORT_REL_HINTS = frozenset(
    {
        RelationKind.BINDS.value,
        RelationKind.MATERIALIZES_AS.value,
        RelationKind.SELECTS.value,
        RelationKind.LAUNCHES.value,
    }
)


def _prefer_located_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A fileless API shadow of a located ident is not a second seed."""
    located_leaves = {
        _last_ident(str(card.get("name") or "")).lower()
        for card in cards
        if _has_loc(card)
    }
    if not located_leaves:
        return cards
    kept = [
        card
        for card in cards
        if _has_loc(card)
        or _last_ident(str(card.get("name") or "")).lower() not in located_leaves
    ]
    return kept or cards


def _definition_site_candidates(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for card in cards:
        extras = card.get("extras") if isinstance(card.get("extras"), dict) else {}
        for site in extras.get("definition_sites") or []:
            if not isinstance(site, dict):
                continue
            file = str(site.get("file") or "")
            line = int(site.get("line") or 0)
            kind = str(site.get("kind") or "")
            key = (file, line, kind)
            if not file or key in seen:
                continue
            seen.add(key)
            out.append(site)
    return out


def _has_transport_edges(card: dict[str, Any]) -> bool:
    extras = card.get("extras") if isinstance(card.get("extras"), dict) else {}
    for key in ("binds", "transport", "materializes"):
        rows = extras.get(key) or card.get(key)
        if isinstance(rows, list) and rows:
            return True
    for rel in extras.get("relations") or []:
        if isinstance(rel, dict) and str(rel.get("kind") or "") in _TRANSPORT_REL_HINTS:
            return True
    return False


def _contract_applicable(primary: dict[str, Any], *, seed_kind: str = "") -> bool:
    kind = str(primary.get("kind") or seed_kind or "")
    if kind == EntityKind.TILING_FIELD.value:
        return True
    if kind in _CONTRACT_KINDS and _has_transport_edges(primary):
        return True
    if kind == EntityKind.TILING_KEY.value:
        return True
    return False


def _has_definition(card: dict[str, Any]) -> bool:
    file, line, _, snippet = _site_line(card)
    return bool(file) and line > 0 and bool(str(snippet or "").strip())


def _file_matches(file: str, needle: str) -> bool:
    a = str(file or "").replace("\\", "/").lower().rstrip("/")
    b = str(needle or "").replace("\\", "/").lower().rstrip("/")
    if not a or not b:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a) or a.endswith(b) or b.endswith(a)


def _looks_like_definition(site: dict[str, Any]) -> bool:
    kind = str(site.get("kind") or "").upper()
    name = _last_ident(str(site.get("name") or ""))
    snippet = str(site.get("snippet") or "")
    start = int(site.get("line") or site.get("line_start") or 0)
    end = int(site.get("line_end") or 0)
    span = max(0, end - start) if end else 0
    cpp = str(site.get("cpp_kind") or "").lower()
    if kind == EntityKind.TYPE.value:
        if cpp in {"class", "struct", "enum", "union"}:
            return True
        if name and re.search(rf"\b(class|struct|enum|union)\s+{re.escape(name)}\b", snippet):
            return True
        return span >= 8
    if kind in _DEF_BODY_KINDS:
        if span >= 2:
            return True
        return bool(name and "{" in snippet and name in snippet)
    return False


def _overlay_site(card: dict[str, Any], site: dict[str, Any], query: Any) -> dict[str, Any]:
    out = dict(card)
    file = str(site.get("file") or "")
    line = int(site.get("line") or 0)
    line_end = int(site.get("line_end") or 0)
    kind = str(site.get("kind") or card.get("kind") or "").upper()
    snippet = _clip_source(
        query,
        file,
        line,
        line_end=line_end or 0,
        max_lines=_MAX_RANGE_LINES if _looks_like_definition(site) else _USED_AT_LINES,
    )
    out["file"] = file
    out["line"] = line
    if line_end:
        out["line_end"] = line_end
    # The overlay re-reads the site with a use-site budget. When the card
    # already carries a recorded span, that rebuild is narrower than what it
    # replaces, and a declaration whose span is a single line loses its body.
    existing = str(card.get("snippet") or "") if _file_matches(
        str(card.get("file") or ""), file
    ) else ""
    if snippet and snippet.count("\n") >= existing.count("\n"):
        out["snippet"] = snippet
    elif existing:
        out["snippet"] = existing
    elif snippet:
        out["snippet"] = snippet
    elif site.get("snippet"):
        out["snippet"] = str(site.get("snippet") or "")
    if site.get("cpp_kind"):
        out["cpp_kind"] = site.get("cpp_kind")
    return out


def _prefer_definition_card(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Class/struct definition outranks a using/alias mention of the same ident."""
    if len(cards) < 2:
        return cards
    defs = [c for c in cards if isinstance(c, dict) and _looks_like_definition(c)]
    if not defs:
        return cards
    head = defs[0]
    rest = [c for c in cards if c is not head]
    return [head, *rest]


def _sibling_use_sites(
    primary: dict[str, Any], cards: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pfile, pline, _, _ = _site_line(primary)
    seen: set[tuple[str, int]] = {(pfile.replace("\\", "/"), pline)}
    out: list[dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        file, line, name, snippet = _site_line(card)
        key = (file.replace("\\", "/"), line)
        if not file or line <= 0 or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "file": file,
                "line": line,
                "name": name or str(card.get("name") or ""),
                "kind": str(card.get("kind") or ""),
                "snippet": snippet,
            }
        )
    return out


def attach_explore_fields(
    query: Any,
    payload: dict[str, Any],
    *,
    pattern: str,
    unique_seed: bool = True,
    projection: str = "summary",
    operation: str = "",
    seed_kind: str = "",
    file_filter: str = "",
) -> dict[str, Any]:
    """Enrich a payload with a contract card. Set operations skip the unique-seed fence."""
    # query_find pins `locations` for name discovery; the plan default must not
    # overwrite a projection the shape already chose.
    payload["projection"] = str(payload.get("projection") or projection or "summary")
    payload["pattern"] = str(pattern or payload.get("pattern") or "")
    payload["explore_pattern"] = payload["pattern"]
    if operation:
        payload["operation"] = operation
    cards = list(payload.get("cards") or [])
    if not cards and payload.get("seeds"):
        cards = list(payload.get("seeds") or [])
    cards = [c for c in cards if isinstance(c, dict)]
    def _finish() -> dict[str, Any]:
        attach = getattr(query, "attach_card_facets", None)
        if callable(attach):
            attach(payload)
        payload["text"] = render_explore_markdown(payload, projection=projection)
        return slim_explore_payload(payload)

    if str(payload.get("shape") or "") == "search" or operation == "search":
        payload["completeness"] = ""
        payload["text"] = render_explore_markdown(payload, projection=projection)
        return slim_explore_payload(payload)
    raw_cards = list(cards)
    extra_seeds: list[dict[str, Any]] = []
    if unique_seed:
        cards = _prefer_located_cards(cards)
    if unique_seed and operation == "contract":
        primary_c, extra_seeds = _prefer_contract_seeds(cards)
        if primary_c is not None:
            cards = [primary_c, *extra_seeds] if extra_seeds else [primary_c]
    elif str(payload.get("shape") or "") == "name" and unique_seed:
        cards, aliases = _prefer_tiling_key_seed(cards, pattern)
        if aliases:
            payload["aliases"] = [
                {"name": a.get("name"), "kind": a.get("kind")}
                for a in aliases
                if isinstance(a, dict)
            ]
        extra_seeds = [
            a for a in (aliases or []) if isinstance(a, dict) and a.get("id")
        ]
    cards = _merge_canonical_identities(cards)
    if unique_seed:
        cards = _prefer_definition_card(cards)
    payload["cards"] = cards
    if not cards:
        payload.setdefault("completeness", UNKNOWN)
        payload["text"] = render_explore_markdown(payload, projection=projection)
        return slim_explore_payload(payload)
    primary = cards[0]
    if not isinstance(primary, dict) or not primary.get("id"):
        payload.setdefault("completeness", UNKNOWN)
        payload["text"] = render_explore_markdown(payload, projection=projection)
        return slim_explore_payload(payload)
    if str(payload.get("shape") or "") == "around" and payload.get("snippet"):
        primary = dict(primary)
        primary["snippet"] = payload.get("snippet")
        primary["line"] = payload.get("line") or primary.get("line") or primary.get("line_start")
        primary["line_end"] = payload.get("line_end") or primary.get("line_end")
        cards[0] = primary
        payload["cards"] = cards
    if unique_seed:
        payload["count"] = len(cards)
        site_has_window = (
            str(payload.get("resolve_mode") or "") == "site"
            and bool(payload.get("snippet"))
        )
        if not site_has_window:
            primary = _expand_statement_snippet(query, primary, raw_cards + extra_seeds)
            cards[0] = primary
            payload["cards"] = cards
    sites = _definition_site_candidates(cards)
    def_sites = [s for s in sites if _looks_like_definition(s)]
    if file_filter:
        match = next(
            (s for s in sites if _file_matches(str(s.get("file") or ""), file_filter)),
            None,
        )
        if match is not None:
            primary = _overlay_site(primary, match, query)
            primary = _expand_statement_snippet(query, primary, raw_cards)
            cards[0] = primary
            payload["cards"] = cards
    if not unique_seed:
        payload.setdefault("completeness", COMPLETE if cards else UNKNOWN)
        payload.setdefault("total", payload.get("total") or len(cards))
        return _finish()
    apply_contract = _contract_applicable(primary, seed_kind=seed_kind)
    if not apply_contract:
        if _has_definition(primary):
            payload["completeness"] = COMPLETE
            payload["unresolved_reason"] = ""
        else:
            payload["completeness"] = UNKNOWN
            payload["unresolved_reason"] = "DEFINITION_MISSING"
        payload.pop("contract", None)
        payload.setdefault("explore_pattern", pattern)
        sites = _definition_site_candidates(cards)
        if not sites:
            sites = [
                {
                    "file": str(c.get("file") or ""),
                    "line": int(c.get("line") or c.get("line_start") or 0),
                    "line_end": int(c.get("line_end") or 0),
                    "kind": str(c.get("kind") or ""),
                    "name": str(c.get("name") or ""),
                    "snippet": str(c.get("snippet") or ""),
                    "cpp_kind": str(c.get("cpp_kind") or ""),
                }
                for c in cards
                if isinstance(c, dict)
            ]
        def_like = [s for s in sites if _looks_like_definition(s)]
        if file_filter:
            match = next(
                (s for s in sites if _file_matches(str(s.get("file") or ""), file_filter)),
                None,
            )
            if match is not None:
                primary = _overlay_site(primary, match, query)
                if match.get("snippet"):
                    primary["snippet"] = str(match.get("snippet") or "")
                cards[0] = primary
                payload["cards"] = cards
        elif def_like:
            pick = def_like[0]
            primary = _overlay_site(primary, pick, query)
            if pick.get("snippet") and not _looks_like_definition(primary):
                primary["snippet"] = str(pick.get("snippet") or "")
            cards[0] = primary
            payload["cards"] = cards
        uses = _sibling_use_sites(primary, cards)
        pfile = str(primary.get("file") or "")
        seen = {(str(u.get("file") or "").replace("\\", "/"), int(u.get("line") or 0)) for u in uses}
        for site in sites:
            file = str(site.get("file") or "")
            line = int(site.get("line") or 0)
            key = (file.replace("\\", "/"), line)
            if not file or line <= 0 or key in seen or _file_matches(file, pfile):
                continue
            seen.add(key)
            uses.append(
                {
                    "file": file,
                    "line": line,
                    "name": site.get("name") or primary.get("name"),
                    "kind": site.get("kind"),
                    "snippet": site.get("snippet") or "",
                }
            )
        if uses:
            payload["used_at"] = _mark_rival_owner_uses(query, primary, uses)
        _attach_call_site_fanout(query, payload, primary)
        return _finish()
    try:
        card = build_contract_card(
            query,
            primary,
            extra_seeds=extra_seeds or None,
        )
    except Exception:  # noqa: BLE001
        payload.setdefault("completeness", INCOMPLETE)
        payload.setdefault("unresolved_reason", "CONTRACT_BUILD_FAILED")
        return _finish()
    payload["contract"] = card
    payload["impact_sinks"] = card.get("impact_sinks") or []
    payload["entry"] = card.get("entry") or []
    multi_leaf = len(_distinct_leaves(cards if operation == "contract" else raw_cards)) > 1
    if multi_leaf and str(payload.get("shape") or "") == "name":
        payload["completeness"] = AMBIGUOUS
        payload["unresolved_reason"] = "MULTIPLE_SEEDS"
        payload["candidates"] = [
            {"name": c.get("name"), "kind": c.get("kind"), "file": c.get("file")}
            for c in (raw_cards or cards)[:8]
            if isinstance(c, dict)
        ]
    elif len(def_sites) > 1 and operation != "contract":
        op_root = getattr(query, "_op_root", None)
        payload["completeness"] = AMBIGUOUS
        payload["unresolved_reason"] = "MULTIPLE_SEEDS"
        payload["candidates"] = [
            {
                "name": s.get("name") or primary.get("name"),
                "kind": s.get("kind"),
                "file": s.get("file"),
                "line": s.get("line"),
            }
            for s in def_sites[:8]
        ]
        sources: list[dict[str, Any]] = []
        for site in def_sites[:8]:
            file = str(site.get("file") or "")
            line = int(site.get("line") or 0)
            snippet = _clip_source(
                query,
                file,
                line,
                line_end=int(site.get("line_end") or 0),
                max_lines=_MAX_RANGE_LINES,
            )
            sources.append(
                {
                    "name": site.get("name") or primary.get("name"),
                    "kind": site.get("kind"),
                    "file": file,
                    "line": line,
                    "snippet": snippet,
                }
            )
        payload["candidate_sources"] = sources
    else:
        primary_file = str(primary.get("file") or "")
        op_root = getattr(query, "_op_root", None)
        used: list[dict[str, Any]] = []
        for site in sites:
            file = str(site.get("file") or "")
            line = int(site.get("line") or 0)
            if not file or not line:
                continue
            if _file_matches(file, primary_file):
                continue
            snippet = _clip_source(query, file, line, max_lines=_USED_AT_LINES)
            used.append(
                {
                    "file": file,
                    "line": line,
                    "name": site.get("name") or primary.get("name"),
                    "kind": site.get("kind"),
                    "snippet": snippet,
                }
            )
        if used:
            payload["used_at"] = _mark_rival_owner_uses(query, primary, used)
        if str(primary.get("kind") or seed_kind or "") not in _CONTRACT_KINDS:
            payload["completeness"] = COMPLETE if _has_definition(primary) else INCOMPLETE
            payload["unresolved_reason"] = (
                "" if payload["completeness"] == COMPLETE else "DEFINITION_MISSING"
            )
        else:
            payload["completeness"] = card.get("completeness") or INCOMPLETE
            payload["unresolved_reason"] = card.get("unresolved_reason") or ""
    payload["unresolved_reasons"] = card.get("unresolved_reasons") or []
    payload["checks"] = card.get("checks") or {}
    if payload.get("completeness") == COMPLETE:
        payload["ok"] = True
    payload.setdefault("explore_pattern", pattern)
    _attach_call_site_fanout(query, payload, primary)
    return _finish()


def _attach_call_site_fanout(
    query: Any, payload: dict[str, Any], primary: dict[str, Any]
) -> None:
    """A definition card for a called ident says how many call sites exist."""
    if str(payload.get("resolve_mode") or "") == "site":
        return
    name = str(primary.get("name") or "")
    if not name:
        return
    try:
        count = int(query.count_call_sites(name))
    except Exception:  # noqa: BLE001
        return
    if count < 2:
        return
    leaf = name.rsplit("::", 1)[-1]
    payload["call_sites"] = count
    note = f"call sites: {count} — find kind=OPERATION callee={leaf}"
    existing = str(payload.get("hint") or "")
    payload["hint"] = f"{existing} {note}".strip() if existing else note
