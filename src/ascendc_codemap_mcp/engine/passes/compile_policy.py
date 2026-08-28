# -*- coding: utf-8 -*-
"""Compile-time policy aliases: ``using T = std::conditional_t<…>`` and ``GET_*``.

Links the alias to Then/Else types and to BUFFER/QUEUE objects whose declared
type mentions the alias. No operator-specific policy class names.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.source_layout import selected_host_files, selected_kernel_files

_USING_RE = re.compile(
    r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[^;]+);",
    re.S,
)
_GET_MACRO_RE = re.compile(r"\b(GET_[A-Z][A-Z0-9_]*)\b")
_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
_SKIP = frozenset(
    {
        "std",
        "conditional",
        "conditional_t",
        "type",
        "typename",
        "template",
        "const",
        "constexpr",
        "static",
        "using",
        "true",
        "false",
        "bool",
        "void",
        "int",
        "uint32_t",
        "uint16_t",
        "uint64_t",
        "int32_t",
        "size_t",
        "nullptr_t",
        "auto",
    }
)
_COND_KEYS = ("conditional_t", "conditional")
_SOURCE_KINDS = (
    EntityKind.TEMPLATE_ARG,
    EntityKind.TILING_KEY,
    EntityKind.COMPILE_VAR,
    EntityKind.MACRO,
    EntityKind.FIELD,
    EntityKind.VARIABLE,
    EntityKind.TYPE,
)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _split_args(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    angle = 0
    for ch in str(text or ""):
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0 and angle == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _conditional_parts(rhs: str) -> tuple[str, str, str] | None:
    text = str(rhs or "")
    for key_word in _COND_KEYS:
        key = text.find(key_word)
        if key < 0:
            continue
        lt = text.find("<", key)
        if lt < 0:
            continue
        depth = 0
        for i, ch in enumerate(text[lt:], lt):
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
                if depth == 0:
                    args = _split_args(text[lt + 1 : i])
                    if len(args) >= 3:
                        return args[0].strip(), args[1].strip(), args[2].strip()
                    return None
    return None


def _base_ident(text: str) -> str:
    raw = re.sub(r"\b(?:const|volatile|typename|template|static)\b", " ", str(text or ""))
    raw = raw.replace("&", " ").replace("*", " ").strip()
    token = raw.split("<", 1)[0].strip().split("::")[-1].strip()
    return token if token.isidentifier() and token not in _SKIP else ""


def _unique(codemap: CodeMap, name: str, kinds: tuple[EntityKind, ...] = _SOURCE_KINDS) -> Entity | None:
    leaf = str(name or "").replace("::", ".").rsplit(".", 1)[-1]
    if not leaf or leaf in _SKIP:
        return None
    hits: dict[str, Entity] = {}
    for kind in kinds:
        for ent in codemap.by_name(leaf, kind=kind):
            hits[ent.id] = ent
        if leaf != name:
            for ent in codemap.by_name(name, kind=kind):
                hits[ent.id] = ent
    if len(hits) == 1:
        return next(iter(hits.values()))
    return None


def _upsert_type(codemap: CodeMap, name: str, *, file: str, line: int, arch: str, provenance: str) -> Entity:
    leaf = _base_ident(name) or str(name or "").split("::")[-1]
    return codemap.upsert(
        EntityKind.TYPE,
        leaf,
        eid=f"SRCPOL::{file}::{leaf}",
        attrs={"provenance": provenance, "architecture": arch, "policy_alias": True},
        file=file,
        line=line,
        status="confirmed",
    )


def _link_buffers(codemap: CodeMap, alias: Entity, names: set[str], *, file: str, line: int) -> None:
    needles = {n.lower() for n in names if n}
    if not needles:
        return
    for kind in (EntityKind.BUFFER, EntityKind.QUEUE):
        for buf in codemap.by_kind(kind):
            blob = " ".join(
                [
                    str(buf.name or ""),
                    str(buf.attrs.get("type_name") or ""),
                    " ".join(str(x) for x in (buf.attrs.get("trace") or [])),
                    str(buf.attrs.get("root") or ""),
                ]
            ).lower()
            if not any(n in blob for n in needles):
                continue
            codemap.mint_candidate_relation(
                RelationKind.BINDS,
                alias.id,
                buf.id,
                provenance="source_compile_policy_buffer",
                extra={"file": file, "line": line, "role": "buffer_policy"},
            )


def enrich_compile_policy(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    files = list(selected_kernel_files(root, architecture=arch)) + list(
        selected_host_files(root, architecture=arch)
    )
    for path in files:
        try:
            text = read_text(path)
        except OSError:
            continue
        file = _rel(root, path)
        for match in _USING_RE.finditer(text):
            alias_name = str(match.group("alias") or "").strip()
            rhs = str(match.group("rhs") or "").strip()
            if not alias_name or not rhs:
                continue
            line = text.count("\n", 0, match.start()) + 1
            alias = _upsert_type(
                codemap, alias_name, file=file, line=line, arch=arch, provenance="source_compile_policy"
            )
            parts = _conditional_parts(rhs)
            names: set[str] = {alias_name}
            if parts is not None:
                cond, then, els = parts
                then_name = _base_ident(then)
                else_name = _base_ident(els)
                for branch, label in ((then_name, "then"), (else_name, "else")):
                    if not branch:
                        continue
                    names.add(branch)
                    dst = _upsert_type(
                        codemap,
                        branch,
                        file=file,
                        line=line,
                        arch=arch,
                        provenance="source_compile_policy_branch",
                    )
                    codemap.mint_candidate_relation(
                        RelationKind.MATERIALIZES_AS,
                        alias.id,
                        dst.id,
                        provenance="source_compile_policy",
                        extra={"file": file, "line": line, "polarity": label},
                    )
                cond_idents = [
                    ident
                    for ident in _IDENT_RE.findall(cond)
                    if ident not in _SKIP and ident != alias_name
                ]
                linked = 0
                for ident in cond_idents:
                    src = _unique(codemap, ident)
                    if src is None or src.id == alias.id:
                        continue
                    codemap.mint_candidate_relation(
                        RelationKind.CONTROLS,
                        src.id,
                        alias.id,
                        provenance="source_compile_policy_cond",
                        extra={"file": file, "line": line, "role": "policy_predicate"},
                    )
                    linked += 1
                if linked == 0:
                    for ident in cond_idents[:1]:
                        src = codemap.upsert(
                            EntityKind.COMPILE_VAR,
                            ident,
                            eid=f"SRCPOLCOND::{file}::{ident}",
                            attrs={"provenance": "source_compile_policy_cond", "architecture": arch},
                            file=file,
                            line=line,
                            status="extracted",
                        )
                        codemap.mint_candidate_relation(
                            RelationKind.CONTROLS,
                            src.id,
                            alias.id,
                            provenance="source_compile_policy_cond",
                            extra={"file": file, "line": line, "role": "policy_predicate"},
                        )
            for macro_name in _GET_MACRO_RE.findall(rhs):
                names.add(macro_name)
                macro = codemap.upsert(
                    EntityKind.MACRO,
                    macro_name,
                    eid=f"SRCMACRO::{file}::{macro_name}",
                    attrs={"provenance": "source_compile_policy_get", "architecture": arch},
                    file=file,
                    line=line,
                    status="confirmed",
                )
                codemap.mint_candidate_relation(
                    RelationKind.EXPANDS_TO,
                    macro.id,
                    alias.id,
                    provenance="source_compile_policy_get",
                    extra={"file": file, "line": line},
                )
            _link_buffers(codemap, alias, names, file=file, line=line)
    return codemap
