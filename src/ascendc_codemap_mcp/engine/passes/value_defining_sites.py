# -*- coding: utf-8 -*-
"""Backtrace TilingData host writers to the sites that decide their values.

``host_writer_sites`` often stop at the final ABI copy
(``set_flag(params.flag)`` / ``td->flag = params.flag``). Lemmas and construct
need the earlier assignments that actually choose the value
(``params.flag = (x % 8 == 0)``). This pass walks host sources for those
defining writes and stores them as ``value_defining_sites`` on each
TILING_FIELD — operator-agnostic, keyed only by the writer expression path.
"""
from __future__ import annotations

import re
from pathlib import Path

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.passes.tiling_host_writes import _line, _mask_non_code, _selected_host_files

_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_PATH_RE = re.compile(
    r"^(?P<path>[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)+)\s*$"
)
_CALL_RE = re.compile(
    r"^(?:(?:this\.)?[A-Za-z_]\w*(?:\s*(?:\.|->)\s*))*([A-Za-z_]\w*)\s*\("
)


def enrich_value_defining_sites(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    fields = list(codemap.by_kind(EntityKind.TILING_FIELD))
    if not fields:
        return codemap

    paths = _selected_host_files(root, architecture)
    texts: list[tuple[str, str, str]] = []
    for path in paths:
        raw = read_text(path)
        texts.append((_rel(root, path), raw, _mask_non_code(raw)))

    # Collect every direct assignment lhs -> (file, line, rhs) once.
    assigns: dict[str, list[dict]] = {}
    #: Plain local declarations (`auto dropValue = ...`). A TilingKey dimension
    #: is usually packed from one of these, so resolving them is what connects a
    #: key dimension to the host state a TilingData guard also reads.
    locals_: dict[str, list[dict]] = {}
    #: function name -> guards its call sites sit under. A field written inside
    #: `ProcessX` is only written when `ProcessX` is called, so the call-site
    #: condition is part of the field's guard set.
    call_guards: dict[str, list[dict]] = {}
    assign_re = re.compile(
        r"(?P<lhs>[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)+)\s*"
        r"(?<![=!<>])=(?!=)\s*(?P<rhs>[^;]+);",
        re.S,
    )
    decl_re = re.compile(
        r"\b(?:auto|bool|u?int\d*_t|uint8_t|size_t|float|double)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*(?<![=!<>])=(?!=)\s*(?P<rhs>[^;]+);",
        re.S,
    )
    call_re = re.compile(r"\b(?P<fn>[A-Z][A-Za-z_]\w*)\s*\(")
    for file, raw, masked in texts:
        regions = _guard_regions(masked)
        functions = _function_spans(masked)
        for match in decl_re.finditer(masked):
            guards = _guards_at(regions, match.start(), raw)
            locals_.setdefault(_norm_path(match.group("name")), []).append({
                "file": file,
                "line": _line(raw, match.start()),
                "lhs": match.group("name"),
                "rhs": raw[match.start("rhs"):match.end("rhs")].strip()[:400],
                "kind": "declaration",
                "guards": guards,
                "unconditional": not guards,
                "function": _enclosing_function(functions, match.start()),
            })
        for match in call_re.finditer(masked):
            guards = _guards_at(regions, match.start(), raw)
            if not guards:
                continue
            call_guards.setdefault(match.group("fn"), []).append({
                "file": file,
                "line": _line(raw, match.start()),
                "guards": guards,
            })
        for match in assign_re.finditer(masked):
            lhs = _norm_path(match.group("lhs"))
            rhs = raw[match.start("rhs"):match.end("rhs")].strip()
            guards = _guards_at(regions, match.start(), raw)
            site = {
                "file": file,
                "line": _line(raw, match.start()),
                "lhs": lhs,
                "rhs": rhs[:400],
                "kind": "assignment",
                # A write nobody guards is the field's default: the value it
                # keeps whenever every guarded write is skipped. That is what
                # lets a key dimension pin the field without a replay witness.
                "guards": guards,
                "unconditional": not guards,
                "function": _enclosing_function(functions, match.start()),
            }
            assigns.setdefault(lhs, []).append(site)
            # Also index by leaf so ``flag`` matches ``params.flag`` writers
            # when the intermediate aggregate name differs across TUss.
            leaf = lhs.rsplit(".", 1)[-1]
            if leaf != lhs:
                assigns.setdefault(leaf, []).append(site)

    annotated = 0
    for field in fields:
        writers = list(field.attrs.get("host_writer_sites") or [])
        if not writers:
            field.attrs.setdefault("value_defining_sites", [])
            continue
        found: list[dict] = []
        seen: set[tuple] = set()
        for w in writers:
            if not isinstance(w, dict):
                continue
            expr = str(w.get("expression") or "").strip()
            if not expr:
                continue
            for site in _sites_for_expression(expr, assigns, texts):
                key = (site.get("file"), site.get("line"), site.get("rhs") or site.get("expression"))
                if key in seen:
                    continue
                seen.add(key)
                fn = str(site.get("function") or "")
                if fn and fn in call_guards:
                    site = dict(site)
                    site["caller_guards"] = call_guards[fn][:4]
                found.append(site)
        field.attrs["value_defining_sites"] = found
        if found:
            annotated += 1
        aliases = _local_aliases_for_field(field, assigns, locals_)
        if aliases:
            field.attrs["local_aliases"] = aliases
            field.attrs["fused_outer_candidates"] = aliases

    keyed = _annotate_key_packing_roots(codemap, assigns, locals_)

    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})
    closure["tiling_value_defining_fields"] = annotated
    closure["tiling_key_packing_rooted_dims"] = keyed
    codemap.meta["kernel_tiling_closure"] = closure
    return codemap


def _annotate_key_packing_roots(
    codemap: CodeMap,
    assigns: dict[str, list[dict]],
    locals_: dict[str, list[dict]],
) -> int:
    """Resolve each TilingKey dimension's packing symbol to its host definition.

    ``IsDrop`` packs ``static_cast<uint8_t>(dropValue)``; the dimension only
    becomes comparable with a TilingData guard once ``dropValue`` is resolved to
    the host state it was computed from. Without this the two surfaces share no
    vocabulary and every cross-surface implication has to be found by hand.
    """
    count = 0
    for ent in codemap.by_kind(EntityKind.TILING_KEY):
        exprs = [str(x) for x in (ent.attrs.get("host_packing_expressions") or [])]
        if not exprs:
            continue
        sites: list[dict] = []
        seen: set[tuple] = set()
        for expr in exprs:
            for name in _identifiers(expr):
                for site in (locals_.get(name) or []) + (assigns.get(name) or []):
                    key = (site.get("file"), site.get("line"))
                    if key in seen:
                        continue
                    seen.add(key)
                    sites.append({**site, "packing_symbol": name})
        if sites:
            ent.attrs["packing_value_sites"] = sites[:12]
            count += 1
    return count


_OCCUPANCY_TOKENS = ("aicnum", "corenum", "aivnum")


def _is_occupancy_expr(rhs: str) -> bool:
    low = str(rhs or "").lower()
    return any(tok in low for tok in _OCCUPANCY_TOKENS)


def _plain_local_defs(
    ident: str,
    assigns: dict[str, list[dict]],
    locals_: dict[str, list[dict]],
) -> list[dict]:
    return list(locals_.get(ident) or []) + [
        s for s in (assigns.get(ident) or []) if "." not in str(s.get("lhs") or "")
    ]


def _local_aliases_for_field(
    field,
    assigns: dict[str, list[dict]],
    locals_: dict[str, list[dict]],
) -> list[dict]:
    """Locals copied into this field, including occupancy multi-hop formulas.

    Host occupancy often writes ``blockOuter`` from a local of the same name
    whose RHS mentions another local (``fusedOuter``) together with ``aicNum``.
    One-hop RHS-of-field-write is not enough for that shape.
    """
    leaf = str(field.name or "").rsplit(".", 1)[-1]
    if not leaf:
        return []
    sites = list(assigns.get(leaf) or [])
    name = str(field.name or "")
    if name and name != leaf:
        sites.extend(assigns.get(_norm_path(name)) or [])
    sites.extend(locals_.get(leaf) or [])
    out: list[dict] = []
    seen: set[tuple] = set()
    visited: set[str] = {leaf.lower()}

    def _emit(ident: str, item: dict, hops: int) -> bool:
        marker = (ident, item.get("file"), item.get("line"), item.get("rhs"))
        if marker in seen:
            return False
        seen.add(marker)
        out.append(
            {
                "name": ident,
                "function": item.get("function") or "",
                "file": item.get("file") or "",
                "line": item.get("line"),
                "rhs": item.get("rhs") or "",
                "guard": item.get("guards") or [],
                "hops": hops,
            }
        )
        return len(out) >= 12

    frontier: list[str] = [leaf]
    for hop in range(3):
        nxt: list[str] = []
        for current in frontier:
            seed = list(_plain_local_defs(current, assigns, locals_))
            if current.lower() == leaf.lower():
                seed.extend(sites)
            for site in seed:
                rhs = str(site.get("rhs") or "")
                occupancy = _is_occupancy_expr(rhs)
                for ident in _identifiers(rhs):
                    if len(ident) < 3 or ident.lower() in visited:
                        continue
                    defs = _plain_local_defs(ident, assigns, locals_)
                    if not defs and not occupancy:
                        continue
                    visited.add(ident.lower())
                    if defs:
                        nxt.append(ident)
                    for item in defs:
                        if _emit(ident, item, hop + 1):
                            return out
        frontier = nxt
        if not frontier:
            break
    return out


_CPP_NOISE = frozenset({
    "static_cast", "reinterpret_cast", "uint8_t", "uint16_t", "uint32_t",
    "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t", "size_t", "bool",
    "float", "double", "auto", "true", "false", "return", "if", "else",
    "std", "ge", "sizeof", "const", "void", "nullptr",
})


def _identifiers(text: str) -> list[str]:
    out: list[str] = []
    for tok in re.findall(r"\b[A-Za-z_]\w*\b", text or ""):
        if tok in _CPP_NOISE or tok in out:
            continue
        out.append(tok)
    return out


def _function_spans(masked: str) -> list[tuple[int, int, str]]:
    """(start, end, name) for definitions, so a write knows its function."""
    braces = _match_pairs(masked, "{", "}")
    head = re.compile(
        r"(?:^|\n)[^\n;{}]*?\b(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{"
    )
    out: list[tuple[int, int, str]] = []
    for m in head.finditer(masked):
        name = m.group("name")
        if name in _CONTROL_HEADS:
            # `if (...) {` matches the same shape as a definition; treating it
            # as one made every guarded write report `function="if"` and lost
            # the call-site guard that decides whether the write runs at all.
            continue
        open_brace = masked.find("{", m.end() - 1)
        if open_brace < 0:
            continue
        end = braces.get(open_brace)
        if end is None:
            continue
        out.append((open_brace, end, name))
    return out


_CONTROL_HEADS = frozenset({
    "if", "else", "while", "for", "switch", "catch", "do", "return",
})


def _enclosing_function(spans: list[tuple[int, int, str]], offset: int) -> str:
    best = ""
    best_len = None
    for start, end, name in spans:
        if start <= offset <= end:
            length = end - start
            if best_len is None or length < best_len:
                best, best_len = name, length
    return best


def _sites_for_expression(
    expr: str,
    assigns: dict[str, list[dict]],
    texts: list[tuple[str, str, str]],
) -> list[dict]:
    path_m = _PATH_RE.match(expr)
    if path_m:
        path = _norm_path(path_m.group("path"))
        out = list(assigns.get(path) or [])
        leaf = path.rsplit(".", 1)[-1]
        # Prefer exact path; fall back to leaf only when exact miss.
        if not out and leaf != path:
            out = [s for s in (assigns.get(leaf) or []) if s.get("lhs", "").endswith("." + leaf)]
        return out

    call_m = _CALL_RE.match(expr)
    if call_m:
        callee = call_m.group(1)
        # Record the call itself as a defining site anchor, plus return-shaped
        # assigns inside functions named like the callee when visible as text.
        anchors = [{
            "file": "",
            "line": None,
            "lhs": expr[:200],
            "rhs": expr[:400],
            "kind": "call",
            "callee": callee,
        }]
        # Best-effort: assignments whose enclosing function name contains callee
        # are expensive to recover without HostIR; keep the call anchor.
        return anchors

    return []


_GUARD_HEAD = re.compile(r"\b(if|while|for)\s*\(")


def _match_pairs(text: str, opener: str, closer: str) -> dict[int, int]:
    """open index -> close index, computed once per file."""
    out: dict[int, int] = {}
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == opener:
            stack.append(i)
        elif ch == closer and stack:
            out[stack.pop()] = i
    return out


def _guard_regions(masked: str) -> list[tuple[int, int, str, int, int]]:
    """(start, end, keyword, cond_start, cond_end) for each guarded region.

    Guards decide whether a defining write runs at all, so a field's value under
    a given key is a statement about these conditions. Both braced blocks and
    single-statement bodies are covered; an unmatched head is skipped rather
    than guessed at.

    The condition is returned as an offset pair rather than text: masking blanks
    string literals to keep quotes from breaking the brace matcher, so slicing
    the condition here would render `strcmp(layout, "TND")` as `strcmp(layout,
    )` and hide the very value the guard turns on. Offsets are the same in both
    buffers, so the caller slices the raw source.
    """
    parens = _match_pairs(masked, "(", ")")
    braces = _match_pairs(masked, "{", "}")
    regions: list[tuple[int, int, str, int, int]] = []
    for m in _GUARD_HEAD.finditer(masked):
        open_paren = m.end() - 1
        close_paren = parens.get(open_paren)
        if close_paren is None:
            continue
        j = close_paren + 1
        while j < len(masked) and masked[j].isspace():
            j += 1
        if j < len(masked) and masked[j] == "{":
            end = braces.get(j)
            if end is None:
                continue
            start = j
        else:
            end = masked.find(";", j)
            if end < 0:
                continue
            start = close_paren
        regions.append((start, end, m.group(1), open_paren + 1, close_paren))
    return regions


def _guards_at(regions: list[tuple[int, int, str, int, int]], offset: int,
               raw: str) -> list[dict]:
    """Conditions whose region encloses ``offset``, outermost first."""
    hits = [r for r in regions if r[0] <= offset <= r[1]]
    hits.sort(key=lambda r: r[0])
    out: list[dict] = []
    for start, _end, kw, cond_start, cond_end in hits:
        if kw != "if":
            # Loop headers bound iteration, not value choice; keep them out of
            # the pin argument so a lemma never rests on a loop bound.
            continue
        cond = " ".join(raw[cond_start:cond_end].split())
        out.append({"keyword": kw, "condition": cond[:300], "line": _line(raw, start)})
    return out


def _norm_path(s: str) -> str:
    return re.sub(r"\s*(?:\.|->)\s*", ".", s.strip())


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
