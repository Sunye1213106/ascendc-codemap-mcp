# -*- coding: utf-8 -*-
"""Operation-specific agent cards. One semantic fact, one representation.

find → candidates; resolve → definition + references; contract → host /
tiling-key / kernel; impact → affected locations. Completeness is independent
of how the seed was found. No human asides.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.query.closure import semantic_impact_closure_sql
from ascendc_codemap_mcp.engine.query.completeness import (
    AMBIGUOUS,
    COMPLETE,
    INCOMPLETE,
    UNKNOWN,
    fence_contract,
    pick_transport,
)

MAX_EXPLORE_CHARS = 25_000
FILE_SECTION_PREFIX = "### "
_PROOF_LINES = 3
_MAX_RANGE_LINES = 40
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
_LOC_KEEP = ("id", "name", "kind", "file", "line", "role", "consumer_role")
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
    op_root: Path | None,
    file: str,
    line: int,
    *,
    line_end: int = 0,
    max_lines: int = _PROOF_LINES,
) -> str:
    if not file or int(line or 0) <= 0:
        return ""
    path = Path(str(file).replace("\\", "/"))
    if not path.is_file() and op_root is not None:
        from ascendc_codemap_mcp.engine.paths import resolve_operator_file

        resolved = resolve_operator_file(Path(op_root), str(file))
        if resolved is not None:
            path = resolved
        else:
            path = Path(op_root) / str(file).replace("\\", "/")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    rows = text.splitlines()
    start_i = int(line) - 1
    if start_i < 0 or start_i >= len(rows):
        return ""
    if int(line_end or 0) >= int(line):
        end_i = min(len(rows), int(line_end))
        if end_i - start_i > _MAX_RANGE_LINES:
            end_i = start_i + _MAX_RANGE_LINES
        out = [f"{i}:{rows[i - 1]}" for i in range(start_i + 1, end_i + 1)]
        return "\n".join(out)
    half = max(0, int(max_lines) // 2)
    start = max(0, start_i - half)
    end = min(len(rows), start + int(max_lines))
    if end - start < int(max_lines):
        start = max(0, end - int(max_lines))
    out = []
    for i, raw in enumerate(rows[start:end], start + 1):
        out.append(f"{i}:{raw}")
    return "\n".join(out)


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
    ]
    if not tks:
        return cards, []
    primary = tks[0]
    aliases = [card for card in cards if str(card.get("id") or "") != str(primary.get("id") or "")]
    return [primary], aliases


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
        rel_rows = conn.execute(
            """
            SELECT kind, src, dst, data FROM relation
            WHERE src = ? OR dst = ?
            LIMIT 400
            """,
            (sid, sid),
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
            other_id = dst_id if src_id == sid else src_id
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
                and dst_id == sid
                and str(hit["kind"] or "") not in _PRODUCER_SKIP_KINDS
                and not _is_validation_name(str(hit.get("name") or ""))
            ):
                snippet = _clip_source(op_root, hit["file"], hit["line_start"])
                producers.append(_loc_row(hit, role="producer", snippet=snippet))
            if (
                rkind in _CONSUMER_KINDS
                and (dst_id == sid or src_id == sid)
                and not _is_validation_name(str(hit.get("name") or ""))
            ):
                snippet = _clip_source(op_root, hit["file"], hit["line_start"])
                consumers.append(_loc_row(hit, role="consumer", snippet=snippet))
            if rkind == RelationKind.BINDS.value:
                binds.append(_loc_row(hit, role="bind"))
            if rkind == RelationKind.MATERIALIZES_AS.value:
                kernel_repr.append(
                    _loc_row(
                        hit,
                        role="kernel_repr",
                        snippet=_clip_source(op_root, hit["file"], hit["line_start"]),
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
                        snippet=_clip_source(op_root, hit["file"], hit["line_start"]),
                    )
                )
        closure = semantic_impact_closure_sql(
            conn, [sid], entity_row=query._entity_row
        )
    sinks = []
    for row in closure.get("sinks") or []:
        if _is_validation_name(str(row.get("name") or "")):
            continue
        snippet = _clip_source(op_root, str(row.get("file") or ""), int(row.get("line") or 0))
        sinks.append({**row, "snippet": snippet} if snippet else dict(row))
    transport = pick_transport(seed_kind=kind, has_branch_reader=has_branch, has_bind=bool(binds))
    seed_snippet = str(seed.get("snippet") or "") or _clip_source(
        op_root,
        str(seed.get("file") or ""),
        int(seed.get("line_start") or seed.get("line") or 0),
        line_end=int(seed.get("line_end") or 0),
    )
    seed_row = _loc_row(seed, role="seed", snippet=seed_snippet)
    fence = fence_contract(
        seeds=[seed_row] + list(extra_seeds or []),
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


def slim_explore_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """One copy of each fact. Snippets live in markdown, not in JSON."""
    out = dict(payload)
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
                for key in ("kind", "name", "id", "file", "line")
                if key in card and card[key] not in (None, "")
            }
            extras = card.get("extras") if isinstance(card.get("extras"), dict) else {}
            writers = extras.get("writers") or card.get("writers")
            readers = extras.get("readers") or card.get("readers")
            if writers:
                item["writers"] = writers
            if readers:
                item["readers"] = readers
            slim_cards.append(item)
        out["cards"] = slim_cards
    out.pop("declared_coverage", None)
    out.pop("product_coverage", None)
    out.pop("truncated", None)
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
    rows: list[tuple[int, str]] = []
    for raw in str(snippet or "").splitlines():
        if ":" in raw[:8]:
            num, _, rest = raw.partition(":")
            if num.strip().isdigit():
                rows.append((int(num), rest))
                continue
        rows.append((fallback_line, raw))
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


def _render_find_markdown(payload: dict[str, Any]) -> list[str]:
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    total = int(payload.get("total") or len(cards))
    lines = [f"Matches: {total}", "", "Top candidates:"]
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
    if total > len(cards):
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
            lines.append(f"(showing {len(op_sites)} of {site_total})")
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


def _render_resolve_markdown(payload: dict[str, Any], *, projection: str) -> list[str]:
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    primary = cards[0] if cards else {}
    seed_name = str(primary.get("name") or "")
    lines: list[str] = []
    candidates = [c for c in (payload.get("candidates") or []) if isinstance(c, dict)]
    cand_src = [c for c in (payload.get("candidate_sources") or []) if isinstance(c, dict)]
    if candidates:
        lines.append("**Candidates** (resolve one)")
        for cand in candidates:
            kind = str(cand.get("kind") or "")
            file = str(cand.get("file") or "")
            suffix = f"  {kind}" + (f"  {file}" if file else "")
            lines.append(f"- {cand.get('name')}{suffix}")
        lines.append("")
    if cand_src:
        lines.append("**Definition**")
        for row in cand_src:
            file = str(row.get("file") or "")
            if file:
                lines.append(f"{FILE_SECTION_PREFIX}{file}")
            for no, text in _snippet_rows(str(row.get("snippet") or ""), int(row.get("line") or 0)):
                lines.append(f"{no}|  {text}")
        lines.append("")
    elif primary and str(projection or payload.get("projection") or "summary") != "locations":
        file, line, name, snippet = _site_line(primary)
        if file:
            src = _render_source(
                [{"file": file, "line": line, "name": name, "snippet": snippet}],
                tight=False,
            )
            lines.extend(["**Definition**" if ln == "**Source**" else ln for ln in src])
    used = _dedup_rows(r for r in (payload.get("used_at") or []) if isinstance(r, dict))
    used = [
        row
        for row in used
        if not _is_unrelated_type_neighbor(seed_name, row) and not _is_tpl_machinery(str(row.get("name") or ""))
    ]
    if used:
        lines.append("**References**")
        counts: dict[str, int] = {}
        for row in used:
            file, _line, _, _ = _site_line(row)
            if file:
                counts[file] = counts.get(file, 0) + 1
        if len(used) <= 8:
            for row in used:
                file, line, _, _ = _site_line(row)
                if file and line:
                    lines.append(f"- {file}:{line}")
        else:
            for file, n in list(counts.items())[:8]:
                lines.append(f"- {file} · {n} refs")
            lines.append(f"{len(used)} references across {len(counts)} files")
        lines.append("")
    dim_names = [str(n).strip() for n in (payload.get("dim_names") or []) if str(n).strip()]
    if dim_names and not seed_name:
        lines.append("**Dims**")
        lines.append("- " + ", ".join(dim_names))
        lines.append("")
    hint = str(payload.get("hint") or "")
    if hint:
        lines.extend([hint, ""])
    return lines


def _render_contract_markdown(payload: dict[str, Any]) -> list[str]:
    contract = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
    cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
    seed = contract.get("seed") if isinstance(contract.get("seed"), dict) else (cards[0] if cards else {})
    seed_name = str(seed.get("name") or "")
    producers = _useful_rows(contract.get("producers") or [], seed_name)
    consumers = _useful_rows(contract.get("consumers") or [], seed_name)
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
    if consumers:
        lines.append("        │")
        lines.append("        ▼")
        lines.append("Kernel")
        seed_file, seed_line, _, _ = _site_line(seed if isinstance(seed, dict) else {})
        for row in consumers[:8]:
            cfile, cline, cname, _ = _site_line(row)
            if cfile and cline and (cfile, cline) == (seed_file, seed_line):
                continue
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
    if _is_name_list(payload) or op == "find":
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
        for dim, values in dim_cov.items():
            shown = values if isinstance(values, list) else [values]
            extra.append(f"- {dim}: {{{', '.join(str(v) for v in shown)}}}")
        if payload.get("legal_key_count") not in (None, ""):
            extra.append(f"- legal_key_count: {payload.get('legal_key_count')}")
        extra.append("")
    lines = ([header, ""] if header else []) + extra + body
    return cut_explore_text("\n".join(lines).rstrip() + "\n")


#: Kinds that actually carry a host → TilingKey → kernel contract. Only these
#: can be graded by the C1–C8 fence; a plain function or struct has no producer
#: to miss, so grading it there reported INCOMPLETE for a complete answer.
_CONTRACT_KINDS = frozenset(
    {
        EntityKind.TILING_FIELD.value,
        EntityKind.FIELD.value,
        EntityKind.TILING_KEY.value,
        EntityKind.TEMPLATE_ARG.value,
        EntityKind.COMPILE_VAR.value,
        EntityKind.INPUT.value,
        EntityKind.OUTPUT.value,
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
    op_root = getattr(query, "_op_root", None)
    if _looks_like_definition(site):
        snippet = _clip_source(
            op_root, file, line, line_end=line_end or 0, max_lines=_MAX_RANGE_LINES
        )
    else:
        snippet = _clip_source(op_root, file, line, max_lines=_USED_AT_LINES)
    out["file"] = file
    out["line"] = line
    if line_end:
        out["line_end"] = line_end
    if snippet:
        out["snippet"] = snippet
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
    if operation:
        payload["operation"] = operation
    cards = list(payload.get("cards") or [])
    if not cards and payload.get("seeds"):
        cards = list(payload.get("seeds") or [])
    cards = [c for c in cards if isinstance(c, dict)]
    if unique_seed:
        cards = _prefer_located_cards(cards)
    if str(payload.get("shape") or "") == "name" and unique_seed:
        cards, aliases = _prefer_tiling_key_seed(cards, pattern)
        if aliases:
            payload["aliases"] = [
                {"name": a.get("name"), "kind": a.get("kind")}
                for a in aliases
                if isinstance(a, dict)
            ]
    cards = _merge_canonical_identities(cards)
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
    sites = _definition_site_candidates(cards)
    def_sites = [s for s in sites if _looks_like_definition(s)]
    if file_filter:
        match = next(
            (s for s in sites if _file_matches(str(s.get("file") or ""), file_filter)),
            None,
        )
        if match is not None:
            primary = _overlay_site(primary, match, query)
            cards[0] = primary
            payload["cards"] = cards
    if not unique_seed:
        payload.setdefault("completeness", COMPLETE if cards else UNKNOWN)
        payload.setdefault("total", payload.get("total") or len(cards))
        payload["text"] = render_explore_markdown(payload, projection=projection)
        return slim_explore_payload(payload)
    try:
        card = build_contract_card(query, primary, extra_seeds=None)
    except Exception:  # noqa: BLE001
        payload.setdefault("completeness", INCOMPLETE)
        payload.setdefault("unresolved_reason", "CONTRACT_BUILD_FAILED")
        payload["text"] = render_explore_markdown(payload, projection=projection)
        return slim_explore_payload(payload)
    payload["contract"] = card
    payload["impact_sinks"] = card.get("impact_sinks") or []
    payload["entry"] = card.get("entry") or []
    if len(cards) > 1 and str(payload.get("shape") or "") == "name":
        payload["completeness"] = AMBIGUOUS
        payload["unresolved_reason"] = "MULTIPLE_SEEDS"
        payload["candidates"] = [
            {"name": c.get("name"), "kind": c.get("kind"), "file": c.get("file")}
            for c in cards[:8]
            if isinstance(c, dict)
        ]
    elif len(def_sites) > 1:
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
                op_root,
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
            snippet = _clip_source(op_root, file, line, max_lines=_USED_AT_LINES)
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
            payload["used_at"] = used
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
    payload["text"] = render_explore_markdown(payload, projection=projection)
    return slim_explore_payload(payload)


def _attach_call_site_fanout(
    query: Any, payload: dict[str, Any], primary: dict[str, Any]
) -> None:
    """A definition card for a called ident says how many call sites exist."""
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
