# -*- coding: utf-8 -*-
"""Resolve current-source Host writes to qualified TilingData fields.

Architecture-local Host sources and shared top-level Host sources that explicitly
reference the requested architecture are scanned.  A write is accepted only
when receiver type/member identity resolves to one concrete TilingData owner;
short field names alone never select a target.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.ir.type_identity import ir_var_types, short_type_name, type_tokens
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.passes.symbol_identity import normalize_symbol
from ascendc_codemap_mcp.engine.passes.tiling_gaps import record_unresolved_tiling
from ascendc_codemap_mcp.engine.source_layout import selected_host_files as _layout_host_files

_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_WORD_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_SETTER_RE = re.compile(
    r"(?P<receiver>[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)*)\s*"
    r"(?:\.|->)\s*set_(?P<field>[A-Za-z_]\w*)\s*\("
)
_DIRECT_RE = re.compile(
    r"(?P<lhs>[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)+)\s*"
    r"(?<![=!<>])=(?!=)\s*(?P<rhs>[^;]+);", re.S,
)
_CLASS_HEAD_RE = re.compile(r"\b(?:class|struct)\s+(?P<name>[A-Za-z_]\w*)\b[^;{]*\{")
_METHOD_HEAD_RE = re.compile(
    r"\b(?P<cls>[A-Za-z_]\w*)::(?P<fn>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
    r"(?:const\s*)?(?:override\s*)?\{"
)


def enrich_tiling_host_writes(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    types = {e.name: e for e in codemap.by_kind(EntityKind.TILING_DATA)}
    if not types:
        return codemap
    known = set(types)
    fields: dict[tuple[str, str], Entity] = {}
    nested: dict[str, set[str]] = defaultdict(set)
    for field in codemap.by_kind(EntityKind.TILING_FIELD):
        owner = str(field.attrs.get("owner") or "")
        fields[(owner, field.name)] = field
        nested[field.name].update(_referenced_types(str(field.attrs.get("cpp_type") or ""), known))

    paths = _selected_host_files(root, architecture)
    texts: list[tuple[Path, str, str]] = []
    receiver_types: dict[str, set[str]] = defaultdict(set)
    file_types: dict[Path, dict[str, set[str]]] = {}
    class_members: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    method_spans: dict[Path, list[tuple[int, int, str]]] = {}
    type_alt = "|".join(re.escape(name) for name in sorted(known, key=len, reverse=True))
    decl_re = re.compile(
        rf"\b(?P<type>{type_alt})\b(?:\s*<[^;{{}}]*>)?"
        rf"(?:\s*(?:const|volatile|mutable|__restrict(?:__)?|restrict|[*&]))*"
        rf"\s*(?P<name>[A-Za-z_]\w*)\b"
    ) if type_alt else None
    for path in paths:
        raw = read_text(path)
        masked = _mask_non_code(raw)
        texts.append((path, raw, masked))
        local: dict[str, set[str]] = defaultdict(set)
        if decl_re is not None:
            for match in decl_re.finditer(masked):
                name = normalize_symbol(match.group("name"))
                if name in {
                    "const", "volatile", "mutable", "restrict", "__restrict", "__restrict__",
                }:
                    continue
                local[name].add(match.group("type"))
                receiver_types[name].add(match.group("type"))
        file_types[path] = local
        for cls, members in _class_members(masked, known).items():
            for name, type_names in members.items():
                class_members[cls][name].update(type_names)
        method_spans[path] = _method_spans(masked)

    _purge(codemap)
    sites = resolved = ambiguous = 0
    written: set[str] = set()
    clang_written: set[tuple[str, int, str]] = set()
    clang_types = dict(receiver_types)
    for name, types in ir_var_types(host_ir, known).items():
        clang_types.setdefault(name, set()).update(types)
    if host_ir is not None:
        events = list(getattr(host_ir, "writes", None) or [])
        events.extend(getattr(host_ir, "local_writes", None) or [])
        for ev in events:
            kind = str(getattr(ev, "kind", "assign") or "assign")
            if kind not in {"assign", "replace", ""}:
                continue
            path_text = normalize_symbol(str(getattr(ev, "path", "") or "").replace("->", "."))
            parts = [p for p in path_text.split(".") if p]
            if len(parts) < 2:
                continue
            receiver, field_name = ".".join(parts[:-1]), parts[-1]
            targets = _targets(
                receiver, field_name, clang_types, clang_types, fields, nested, known
            )
            if not targets:
                type_hits = type_tokens(path_text) & known
                if len(type_hits) == 1:
                    owner = next(iter(type_hits))
                    field = fields.get((owner, field_name)) or fields.get(
                        (short_type_name(owner), field_name)
                    )
                    if field is not None:
                        targets = [field]
            if len(targets) != 1:
                continue
            file = str(getattr(ev, "file", "") or "").replace("\\", "/")
            try:
                file = _rel(root, Path(file)) if file else file
            except Exception:
                pass
            line = int(getattr(ev, "line", 0) or 0)
            expr = str(getattr(ev, "rhs", "") or "").strip()
            sites += 1
            _write(
                codemap,
                targets[0],
                file,
                line,
                receiver,
                expr,
                "clang_assign",
                provenance="clang_host_write",
            )
            written.add(targets[0].id)
            clang_written.add((file, line, targets[0].id))
            resolved += 1

    for path, raw, masked in texts:
        file = _rel(root, path)
        local_types = dict(file_types.get(path) or {})
        spans = method_spans.get(path) or []
        for match in _SETTER_RE.finditer(masked):
            close = _matching_paren(masked, match.end() - 1)
            if close < 0:
                continue
            sites += 1
            receiver = normalize_symbol(match.group("receiver"))
            field_name = match.group("field")
            lookup = _with_class_members(local_types, class_members, spans, match.start())
            targets = _targets(
                receiver, field_name, lookup, receiver_types, fields, nested, known
            )
            line = _line(raw, match.start())
            expr = raw[match.end():close].strip()
            if len(targets) == 1:
                if (file, line, targets[0].id) in clang_written:
                    continue
                _write(codemap, targets[0], file, line, receiver, expr, "setter")
                written.add(targets[0].id); resolved += 1
            elif len(targets) > 1:
                _unresolved(codemap, file, line, receiver, field_name, expr, targets)
                ambiguous += 1

        for match in _DIRECT_RE.finditer(masked):
            lhs = normalize_symbol(match.group("lhs"))
            parts = [p for p in lhs.split(".") if p]
            if len(parts) < 2:
                continue
            receiver, field_name = ".".join(parts[:-1]), parts[-1]
            lookup = _with_class_members(local_types, class_members, spans, match.start())
            targets = _targets(
                receiver, field_name, lookup, receiver_types, fields, nested, known
            )
            if not targets:
                continue
            sites += 1
            line = _line(raw, match.start())
            expr = raw[match.start("rhs"):match.end("rhs")].strip()
            if len(targets) == 1:
                if (file, line, targets[0].id) in clang_written:
                    continue
                _write(codemap, targets[0], file, line, receiver, expr, "assignment")
                written.add(targets[0].id); resolved += 1
            else:
                _unresolved(codemap, file, line, receiver, field_name, expr, targets)
                ambiguous += 1

    _attach_defaults(codemap, root, fields)
    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})
    closure.update({
        "selected_host_writer_files": [_rel(root, p) for p in paths],
        "tiling_host_writer_sites": sites,
        "tiling_resolved_host_writer_sites": resolved,
        "tiling_host_writer_fields": len(written),
        "tiling_ambiguous_writer_sites": ambiguous,
        "tiling_host_writer_policy": "qualified-receiver-arch-shared/v2",
    })
    codemap.meta["kernel_tiling_closure"] = closure
    return codemap


def _selected_host_files(root: Path, architecture: str) -> list[Path]:
    return [p.resolve() for p in _layout_host_files(root, architecture)]


def _with_class_members(
    local_types: dict[str, set[str]],
    class_members: dict[str, dict[str, set[str]]],
    spans: list[tuple[int, int, str]],
    pos: int,
) -> dict[str, set[str]]:
    """Prefer the enclosing class's member type over a globally ambiguous name."""
    cls = _class_at(spans, pos)
    if not cls or cls not in class_members:
        return local_types
    merged: dict[str, set[str]] = {k: set(v) for k, v in local_types.items()}
    for name, types in class_members[cls].items():
        merged[name] = set(types)
    return merged


def _class_members(text: str, known: set[str]) -> dict[str, dict[str, set[str]]]:
    type_alt = "|".join(re.escape(name) for name in sorted(known, key=len, reverse=True))
    if not type_alt:
        return {}
    member_re = re.compile(
        rf"\b(?P<type>{type_alt})\b(?:\s*<[^;{{}}]*>)?"
        rf"(?:\s*(?:const|volatile|mutable|__restrict(?:__)?|restrict|[*&]))*"
        rf"\s*(?P<name>[A-Za-z_]\w*)\b"
    )
    out: dict[str, dict[str, set[str]]] = {}
    for match in _CLASS_HEAD_RE.finditer(text):
        open_b = match.end() - 1
        if open_b < 0 or open_b >= len(text) or text[open_b] != "{":
            continue
        close_b = _matching_brace(text, open_b)
        if close_b < 0:
            continue
        members: dict[str, set[str]] = defaultdict(set)
        for mm in member_re.finditer(text[open_b + 1 : close_b]):
            name = normalize_symbol(mm.group("name"))
            if name in {
                "const", "volatile", "mutable", "restrict", "__restrict", "__restrict__",
            }:
                continue
            members[name].add(mm.group("type"))
        if members:
            out[match.group("name")] = members
    return out


def _method_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _METHOD_HEAD_RE.finditer(text):
        open_b = match.end() - 1
        close_b = _matching_brace(text, open_b)
        if close_b < 0:
            continue
        spans.append((match.start(), close_b, match.group("cls")))
    return spans


def _class_at(spans: list[tuple[int, int, str]], pos: int) -> str:
    for start, end, cls in spans:
        if start <= pos <= end:
            return cls
    return ""


def _matching_brace(text: str, open_pos: int) -> int:
    if open_pos < 0 or open_pos >= len(text) or text[open_pos] != "{":
        return -1
    depth = 0
    quote = ""
    escape = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {'"', "'"}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _targets(receiver, field_name, file_types, global_types, fields, nested, known) -> list[Entity]:
    owners = _receiver_owners(receiver, file_types, fields, nested, known)
    if not owners:
        owners = _receiver_owners(receiver, global_types, fields, nested, known)
    hits = _unique([fields[(o, field_name)] for o in owners if (o, field_name) in fields])
    if len(hits) > 1:
        file_owners: set[str] = set()
        for types in (file_types or {}).values():
            file_owners.update(types)
        narrowed = [h for h in hits if str(h.attrs.get("owner") or "") in file_owners]
        if len(narrowed) == 1:
            return narrowed
    return hits


def _receiver_owners(receiver, receiver_types, fields, nested, known) -> set[str]:
    parts = [p for p in normalize_symbol(receiver).split(".") if p]
    if not parts:
        return set()
    owners = set(receiver_types.get(parts[0]) or ())
    if not owners:
        owners.update(nested.get(parts[0]) or ())
    for segment in parts[1:]:
        nxt: set[str] = set()
        for owner in owners:
            field = fields.get((owner, segment))
            if field is not None:
                nxt.update(_referenced_types(str(field.attrs.get("cpp_type") or ""), known))
        if not nxt:
            nxt.update(nested.get(segment) or ())
        owners = nxt
    return owners


def _purge(codemap: CodeMap) -> None:
    provs = {
        "source_tilingdata_host_write_verified",
        "source_tilingdata_host_write_unresolved",
        "clang_host_write",
    }
    remove_ent = {eid for eid,e in codemap.entities.items() if str(e.attrs.get("provenance") or "") in provs}
    remove_rel = {
        rid for rid,r in codemap.relations.items()
        if str(r.attrs.get("provenance") or "") in {
            "source_tilingdata_host_write_verified",
            "clang_host_write",
        }
    }
    for rid,r in list(codemap.relations.items()):
        if r.src in remove_ent or r.dst in remove_ent: remove_rel.add(rid)
    for rid in remove_rel: codemap.relations.pop(rid, None)
    for eid in remove_ent: codemap.entities.pop(eid, None)


def _write(codemap, field, file, line, receiver, expr, mode, provenance="source_tilingdata_host_write_verified") -> None:
    owner = str(field.attrs.get("owner") or "")
    prefix = "TDWRITECL" if provenance.startswith("clang") else "TDWRITEV"
    node = codemap.upsert(
        EntityKind.PREDICATE, f"{owner}::{field.name} <- {expr[:120]}",
        eid=f"{prefix}::{file}::{line}::{owner}::{field.name}",
        attrs={"predicate_role":"tilingdata_writer","owner":owner,"field":field.name,"receiver":receiver,
               "expression":expr[:600],"write_mode":mode,"provenance":provenance},
        file=file,line=line,status="confirmed",
    )
    attrs = {"file": file, "line": line, "mode": mode, "provenance": provenance}
    if provenance.startswith("clang"):
        codemap.link(RelationKind.WRITES, node.id, field.id, attrs=attrs, status="confirmed")
    else:
        codemap.mint_candidate_relation(
            RelationKind.WRITES,
            node.id,
            field.id,
            provenance=provenance,
            extra={"file": file, "line": line, "mode": mode},
            status="confirmed",
        )
    site={"file":file,"line":line,"receiver":receiver,"expression":expr[:300],"mode":mode}
    if site not in field.attrs.setdefault("host_writer_sites",[]): field.attrs["host_writer_sites"].append(site)
    field.attrs["host_writer_site_count"]=len(field.attrs["host_writer_sites"])


def _unresolved(codemap,file,line,receiver,field_name,expr,candidates) -> None:
    record_unresolved_tiling(
        codemap,
        None,
        role="tilingdata_writer_unresolved",
        file=file,
        line=line,
        expression=expr,
        extra={
            "reason": "field_owner_ambiguous" if candidates else "field_owner_unknown",
            "receiver": receiver,
            "field": field_name,
            "candidate_fields": [f.attrs.get("qualified_name") for f in candidates],
            "provenance": "source_tilingdata_host_write_unresolved",
        },
    )


def _attach_defaults(codemap, root, fields) -> None:
    cache: dict[str,list[str]]={}
    for field in fields.values():
        if not field.file or not field.line_start: continue
        if field.file not in cache:
            p=_resolve_file(root,field.file); cache[field.file]=p.read_text(encoding="utf-8",errors="replace").splitlines() if p else []
        lines=cache[field.file]; n=int(field.line_start or 0)
        if 1<=n<=len(lines):
            m=re.search(rf"\b{re.escape(field.name)}\b\s*=\s*([^;]+);",lines[n-1])
            if m:
                field.attrs["default_initializer"]=m.group(1).strip(); field.attrs["default_initializer_site"]={"file":field.file,"line":n}


def _referenced_types(raw: str, known: set[str]) -> set[str]:
    return set(_WORD_RE.findall(raw or "")) & known


def _unique(items) -> list[Entity]:
    out=[]; seen=set()
    for item in items:
        if item.id not in seen: seen.add(item.id); out.append(item)
    return out


def _matching_paren(text,open_pos):
    if open_pos<0 or open_pos>=len(text) or text[open_pos]!="(": return -1
    d=0
    for i in range(open_pos,len(text)):
        if text[i]=="(": d+=1
        elif text[i]==")":
            d-=1
            if d==0:return i
    return -1


def _mask_non_code(text: str) -> str:
    out=list(text); i=0; state="code"; quote=""
    while i<len(text):
        ch=text[i]; nxt=text[i+1] if i+1<len(text) else ""
        if state=="code":
            if ch=="/" and nxt=="/": out[i]=out[i+1]=" "; i+=2; state="line"; continue
            if ch=="/" and nxt=="*": out[i]=out[i+1]=" "; i+=2; state="block"; continue
            if ch in {'\"',"'"}: quote=ch; out[i]=" "; i+=1; state="string"; continue
            i+=1; continue
        if state=="line":
            if ch=="\n":state="code"
            else:out[i]=" "
            i+=1;continue
        if state=="block":
            if ch=="*" and nxt=="/":out[i]=out[i+1]=" ";i+=2;state="code"
            else:
                if ch!="\n":out[i]=" "
                i+=1
            continue
        if ch=="\\" and i+1<len(text):out[i]=out[i+1]=" ";i+=2;continue
        if ch==quote:out[i]=" ";i+=1;state="code"
        else:
            if ch!="\n":out[i]=" "
            i+=1
    return "".join(out)


def _resolve_file(root: Path, raw: str) -> Path | None:
    from ascendc_codemap_mcp.engine.paths import resolve_operator_file

    return resolve_operator_file(root, raw)


def _rel(root: Path,path: Path)->str:
    try:return path.relative_to(root.parent).as_posix()
    except ValueError:return path.as_posix()


def _line(text: str,offset: int)->int:return text.count("\n",0,max(0,offset))+1
