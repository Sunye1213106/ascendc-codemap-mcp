# -*- coding: utf-8 -*-
"""Resolve remaining selected-source Kernel calls through source type facts.

Handles two common AscendC patterns that a direct lexical call binder cannot
resolve alone: inherited CRTP methods and template member dispatch through
aliases such as ``CubeBlockType``/``VecBlockType``.  Multiple source-supported
implementations are retained as an explicit dispatch set rather than collapsed
to one guessed target.  Calls with no local source target are reclassified as
external instead of polluting the internal unresolved count.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind

_WORD_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_]\w*)\s*(?::(?P<bases>[^\{]+))?\{", re.S)
_ALIAS_RE = re.compile(r"\busing\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", re.S)
_BOUND = {
    "source_kernel_call_bound_v2",
    "source_kernel_macro_call_bound_v2",
    "source_kernel_call_bound_v3",
    "source_kernel_call_dispatch_set_v3",
}
_MEMBER_RE_CACHE: dict[str, re.Pattern[str]] = {}


def resolve_kernel_call_frontiers(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    selected = list((codemap.meta.get("kernel_tiling_closure") or {}).get("selected_kernel_files") or [])
    texts = _load(root, selected)
    _MEMBER_RE_CACHE.clear()

    definitions = [
        e for e in codemap.entities.values()
        if str(e.attrs.get("provenance") or "") == "source_kernel_definition_v2"
    ]
    known_classes = {str(e.attrs.get("owner") or "") for e in definitions if e.attrs.get("owner")}
    known_classes.discard("")
    by_method: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    free: dict[str, list[Entity]] = defaultdict(list)
    for ent in definitions:
        owner = str(ent.attrs.get("owner") or "")
        short = ent.name.split("::")[-1]
        if owner:
            by_method[(owner, short)].append(ent)
        else:
            free[short].append(ent)

    inheritance, members, aliases = _type_model(texts, known_classes)
    source_lines = {file: raw.splitlines() for file, raw in texts.items()}

    unresolved = [
        rel for rel in list(codemap.relations.values())
        if rel.kind_name() == RelationKind.CALLS.value
        and str(rel.attrs.get("provenance") or "") == "source_kernel_call_unresolved_v2"
    ]
    resolved_sites = 0
    dispatch_sites = 0
    externalized = 0
    still_unresolved = 0
    remove_rel: set[str] = set()
    remove_ent: set[str] = set()

    for rel in unresolved:
        caller = codemap.entities.get(rel.src)
        ref = codemap.entities.get(rel.dst)
        if caller is None or ref is None:
            continue
        call = str(ref.attrs.get("call_target") or rel.attrs.get("call") or "")
        receiver = str(ref.attrs.get("receiver") or rel.attrs.get("receiver") or "")
        file = str(rel.attrs.get("file") or ref.file or caller.file or "")
        line = int(rel.attrs.get("line") or ref.line_start or 0)
        caller_owner = str(caller.attrs.get("owner") or "")
        line_text = _line_text(source_lines.get(file) or [], line)

        owners: set[str] = set()
        member_name = receiver
        if receiver in {"", "this"} and caller_owner:
            owners.update(_owner_closure(caller_owner, inheritance))
        if not receiver:
            pat = _MEMBER_RE_CACHE.get(call)
            if pat is None:
                pat = re.compile(
                    rf"(?:this\s*->\s*)?([A-Za-z_]\w*)\s*(?:\.|->)\s*(?:template\s+)?{re.escape(call)}\b"
                )
                _MEMBER_RE_CACHE[call] = pat
            member_match = pat.search(line_text)
            if member_match:
                member_name = member_match.group(1)
        if member_name and caller_owner:
            for owner in _owner_closure(caller_owner, inheritance):
                owners.update(members.get((owner, member_name)) or ())
        if receiver and receiver != "this":
            owners.update(_receiver_types_from_context(source_lines.get(file) or [], line, receiver, known_classes, aliases))

        candidates: list[Entity] = []
        for owner in owners:
            for actual_owner in _owner_closure(owner, inheritance):
                candidates.extend(by_method.get((actual_owner, call), ()))

        # Unqualified calls can also be inherited methods or real free-function
        # overloads.  A source overload set is retained as a dispatch set.
        if not receiver:
            if caller_owner:
                for owner in _owner_closure(caller_owner, inheritance):
                    candidates.extend(by_method.get((owner, call), ()))
            candidates.extend(free.get(call, ()))

        candidates = _unique(candidates)
        if candidates:
            provenance = "source_kernel_call_bound_v3" if len(candidates) == 1 else "source_kernel_call_dispatch_set_v3"
            site = {
                "provenance": provenance,
                "file": file,
                "line": line,
                "receiver": receiver,
                "call": call,
                "dispatch_candidates": [e.id for e in candidates],
            }
            for target in candidates:
                new_rel = codemap.mint_candidate_relation(
                    RelationKind.CALLS,
                    caller.id,
                    target.id,
                    provenance=provenance,
                    extra=site,
                    status="confirmed",
                )
                new_rel.attrs.update(site)
                new_rel.attrs.setdefault("sites", []).append({"file": file, "line": line, "receiver": receiver, "call": call})
            remove_rel.add(rel.id)
            remove_ent.add(ref.id)
            if len(candidates) == 1:
                resolved_sites += 1
            else:
                dispatch_sites += 1
            continue

        # If no selected-source definition can be reached through the caller's
        # own type/inheritance/member model, this is an external API call, not an
        # internal CodeMap hole.  Keep true receiver-typed internal uncertainty.
        locally_named = bool(free.get(call)) or any(name == call for (_owner, name) in by_method)
        typed_internal = bool(owners)
        if not typed_internal and (not locally_named or member_name != receiver):
            remove_rel.add(rel.id)
            remove_ent.add(ref.id)
            externalized += 1
        elif not typed_internal and receiver == "" and not caller_owner:
            remove_rel.add(rel.id)
            remove_ent.add(ref.id)
            externalized += 1
        else:
            still_unresolved += 1

    for rid in remove_rel:
        codemap.relations.pop(rid, None)
    for eid in remove_ent:
        if not codemap.has_incident(eid):
            codemap.entities.pop(eid, None)

    reachable = _reachable(codemap)
    reachable_unresolved = sum(
        1 for rel in codemap.relations.values()
        if rel.kind_name() == RelationKind.CALLS.value
        and str(rel.attrs.get("provenance") or "") == "source_kernel_call_unresolved_v2"
        and rel.src in reachable
    )
    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})
    closure.update(
        {
            "kernel_inheritance_edges": sum(len(v) for v in inheritance.values()),
            "kernel_member_type_bindings": sum(len(v) for v in members.values()),
            "kernel_resolved_frontier_call_sites": resolved_sites,
            "kernel_dispatch_set_call_sites": dispatch_sites,
            "kernel_externalized_call_sites": externalized,
            "kernel_unresolved_internal_call_sites": still_unresolved,
            "kernel_reachable_unresolved_internal_call_sites": reachable_unresolved,
            "kernel_reachable_scopes": len(reachable),
            "call_frontier_policy": "inheritance-member-dispatch/v1",
        }
    )
    codemap.meta["kernel_tiling_closure"] = closure
    return codemap


def _type_model(
    texts: dict[str, str], known_classes: set[str]
) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[str]], dict[str, set[str]]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    raw_aliases: list[tuple[str, str]] = []
    for raw in texts.values():
        raw_aliases.extend((m.group(1), m.group(2)) for m in _ALIAS_RE.finditer(raw))
    for _ in range(3):
        for alias, expr in raw_aliases:
            tokens = set(_WORD_RE.findall(expr))
            aliases[alias].update(tokens & known_classes)
            for token in tokens:
                aliases[alias].update(aliases.get(token) or ())

    inheritance: dict[str, set[str]] = defaultdict(set)
    members: dict[tuple[str, str], set[str]] = defaultdict(set)
    from ascendc_codemap_mcp.engine.passes.source_text_cache import mask_cached

    for raw in texts.values():
        masked = mask_cached(raw)
        for match in _CLASS_RE.finditer(masked):
            owner = match.group(1)
            open_pos = masked.find("{", match.start(), match.end())
            close = _matching(masked, open_pos, "{", "}")
            if close < 0:
                continue
            base_expr = match.group("bases") or ""
            base_tokens = set(_WORD_RE.findall(base_expr))
            inheritance[owner].update(base_tokens & known_classes)
            body = masked[open_pos + 1:close]
            depth = 0
            for raw_line in body.splitlines():
                line = raw_line.strip().rstrip("\\")
                if depth == 0 and line.endswith(";") and "(" not in line:
                    left = line[:-1].split("=", 1)[0].strip()
                    var_match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", left)
                    if var_match:
                        name = var_match.group(1)
                        prefix = left[:var_match.start(1)]
                        tokens = set(_WORD_RE.findall(prefix))
                        targets = tokens & known_classes
                        for token in tokens:
                            targets.update(aliases.get(token) or ())
                        if targets:
                            members[(owner, name)].update(targets)
                depth += raw_line.count("{") - raw_line.count("}")
                depth = max(0, depth)
    return inheritance, members, aliases


def _receiver_types_from_context(
    lines: list[str], line: int, receiver: str, known: set[str], aliases: dict[str, set[str]]
) -> set[str]:
    if not lines or line <= 0 or not receiver:
        return set()
    start = max(0, line - 120)
    snippet = "\n".join(lines[start:line])
    prefix = ""
    pos = len(snippet)
    token = receiver
    while pos > 0:
        found = snippet.rfind(token, 0, pos)
        if found < 0:
            break
        before = snippet[found - 1] if found else ""
        after_i = found + len(token)
        rest = snippet[after_i:].lstrip()
        ident_before = bool(before) and (before.isalnum() or before == "_")
        if not ident_before and rest[:1] in ";=":
            stmt = snippet.rfind(";", 0, found)
            brace = max(snippet.rfind("{", 0, found), snippet.rfind("}", 0, found))
            prefix = snippet[max(stmt, brace) + 1 : found]
            break
        pos = found
        if pos == 0:
            break
        pos -= 1
    if not prefix:
        return set()
    candidates: set[str] = set()
    tokens = set(_WORD_RE.findall(prefix))
    candidates.update(tokens & known)
    for token_name in tokens:
        candidates.update(aliases.get(token_name) or ())
    return candidates


def _owner_closure(owner: str, inheritance: dict[str, set[str]]) -> set[str]:
    seen = {owner}
    q = deque([owner])
    while q:
        cur = q.popleft()
        for base in inheritance.get(cur, ()):
            if base not in seen:
                seen.add(base)
                q.append(base)
    return seen


def _reachable(codemap: CodeMap) -> set[str]:
    starts = {e.id for e in codemap.by_kind(EntityKind.KERNEL) if e.attrs.get("source_signature")}
    adj: dict[str, set[str]] = defaultdict(set)
    for rel in codemap.relations.values():
        if rel.kind_name() == RelationKind.CALLS.value and str(rel.attrs.get("provenance") or "") in _BOUND:
            adj[rel.src].add(rel.dst)
    seen = set(starts)
    q = deque(starts)
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _load(root: Path, selected: list[str]) -> dict[str, str]:
    from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

    out: dict[str, str] = {}
    for raw in selected:
        path = _resolve_file(root, raw)
        if path is None:
            continue
        out[raw.replace("\\", "/").lstrip("./")] = read_text(path)
    return out


def _unique(items: Iterable[Entity]) -> list[Entity]:
    out: list[Entity] = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        out.append(item)
    return out


def _line_text(lines: list[str], line: int) -> str:
    if not lines or line <= 0:
        return ""
    lo = max(0, line - 2)
    hi = min(len(lines), line + 1)
    return " ".join(lines[lo:hi]).replace("\\", " ")


def _matching(text: str, pos: int, opener: str, closer: str) -> int:
    if pos < 0:
        return -1
    depth = 0
    for idx in range(pos, len(text)):
        if text[idx] == opener:
            depth += 1
        elif text[idx] == closer:
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _mask_non_code(text: str) -> str:
    out = list(text)
    i = 0
    state = "code"
    quote = ""
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/": out[i] = out[i + 1] = " "; i += 2; state = "line"; continue
            if ch == "/" and nxt == "*": out[i] = out[i + 1] = " "; i += 2; state = "block"; continue
            if ch in {'\"', "'"}: quote = ch; out[i] = " "; i += 1; state = "string"; continue
            i += 1; continue
        if state == "line":
            if ch == "\n": state = "code"
            else: out[i] = " "
            i += 1; continue
        if state == "block":
            if ch == "*" and nxt == "/": out[i] = out[i + 1] = " "; i += 2; state = "code"
            else:
                if ch != "\n": out[i] = " "
                i += 1
            continue
        if ch == "\\" and i + 1 < len(text): out[i] = out[i + 1] = " "; i += 2; continue
        if ch == quote: out[i] = " "; i += 1; state = "code"
        else:
            if ch != "\n": out[i] = " "
            i += 1
    return "".join(out)


def _resolve_file(root: Path, raw: str) -> Path | None:
    from ascendc_codemap_mcp.engine.paths import resolve_operator_file

    return resolve_operator_file(root, raw)
