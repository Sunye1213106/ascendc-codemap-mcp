# -*- coding: utf-8 -*-
"""High-confidence call-site identity: declaration root + receiver type.

Lexical C++ scope only. Never proves an AscendC catalog root. A unique
project method / free function plus a unique receiver binding is enough to
name the callee; overload resolution and CANN headers are out of scope.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.passes import kernel_scan as kscan
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

_CLASS_RE = re.compile(r"\b(?:class|struct)\s+(?P<name>[A-Za-z_]\w*)\b")
_NS_RE = re.compile(r"\bnamespace\s+(?P<name>[A-Za-z_]\w*)\b")
_METHOD_OWNER_RE = re.compile(
    r"\b(?P<owner>[A-Za-z_]\w*)(?:\s*<[^;{}()<>]{0,200}>)?\s*::\s*"
    r"(?P<name>[A-Za-z_~]\w*)\s*\("
)
_NAME_PAREN_RE = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\s*(?:<[^;{}()=]{0,240}>)?\s*\(")
_LOCAL_DECL_RE = re.compile(
    r"(?P<type>(?:[\w:<>,\s*&]+?))\s+(?P<name>[A-Za-z_]\w*)\s*(?:=|;)"
)
_AUTO_FROM_CALL_RE = re.compile(
    r"\bauto\s*(?:&&|&|\*)?\s*(?P<name>[A-Za-z_]\w*)\s*=\s*"
    r"(?P<recv>[A-Za-z_]\w*)\s*(?:\.|->)\s*(?:template\s+)?"
    r"(?P<callee>[A-Za-z_]\w*)"
)
_MEMBER_RE = re.compile(
    r"(?P<type>(?:[\w:<>,\s*&!]+?))\s+(?P<name>[A-Za-z_]\w*)\s*(?:=[^;]*)?;"
)
_USING_IN_CLASS_RE = re.compile(
    r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*(?P<target>[^;{]+)\s*;"
)
_DECL_SPEC_RE = re.compile(r"\b(__aicore__|inline|constexpr|virtual|explicit|static|__host__)\b")
_STMT_RE = re.compile(
    r"\b(if|while|for|switch|catch|return|else|static_assert|sizeof|alignof|decltype)\b"
)
_DECL_CARRY_RE = re.compile(
    r"\b(template|__aicore__|inline|constexpr|virtual|explicit|static|void|bool|auto|typename)\b"
)
_METHOD_DECL_RE = re.compile(r"\b[A-Za-z_]\w*\s*\([^;]*\)\s*(?:const\s*)?;")
_SKIP_OWNERS = frozenset(
    {
        "conditional",
        "conditional_t",
        "type",
        "nullptr_t",
        "nullptr",
        "using",
        "typedef",
        "auto",
        "void",
        "bool",
        "int",
        "float",
        "double",
        "char",
        "unsigned",
        "long",
        "short",
    }
)

_SKIP_NAMES = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "catch",
        "return",
        "sizeof",
        "alignof",
        "decltype",
        "static_assert",
        "new",
        "delete",
        "static_cast",
        "reinterpret_cast",
        "const_cast",
        "dynamic_cast",
        "public",
        "private",
        "protected",
        "operator",
        "namespace",
        "class",
        "struct",
        "enum",
    }
)


def _conditional_branch_types(type_text: str) -> list[str]:
    """THEN/ELSE types from ``std::conditional`` / ``std::conditional_t``."""
    text = str(type_text or "")
    for key_word in ("conditional_t", "conditional"):
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
                        return [args[1].strip(), args[2].strip()]
                    return []
    return []


def _nested_qual(type_text: str) -> tuple[str, str] | None:
    text = re.sub(r"\b(?:typename|template)\b", " ", str(type_text or ""))
    text = re.sub(r"\s+", " ", text).strip().rstrip("&").rstrip("*").strip()
    m = re.search(r"\b([A-Za-z_]\w*)(?:\s*<[^;]*>)?\s*::\s*([A-Za-z_]\w+)\s*$", text)
    if not m or m.group(1) in {"std", "AscendC"}:
        return None
    return m.group(1), m.group(2)


def _owner_candidates(
    type_text: str,
    index: SourceSymbolIndex | None = None,
    *,
    scope_owner: str = "",
    _seen: set[str] | None = None,
) -> list[str]:
    text = str(type_text or "").strip()
    if not text:
        return []
    seen = _seen if _seen is not None else set()
    sig = f"{scope_owner}|{text}"
    if sig in seen:
        return []
    seen.add(sig)
    owners: list[str] = []
    branches = _conditional_branch_types(text)
    if branches:
        for branch in branches:
            for owner in _owner_candidates(branch, index, scope_owner=scope_owner, _seen=seen):
                if owner not in owners:
                    owners.append(owner)
        return owners
    if index is not None:
        nested = _nested_qual(text)
        if nested:
            target = index.aliases.get(nested)
            if target:
                return _owner_candidates(target, index, scope_owner=nested[0], _seen=seen)
        base_now = _base_type(text)
        if base_now and scope_owner:
            target = index.aliases.get((scope_owner, base_now))
            if target:
                return _owner_candidates(target, index, scope_owner=scope_owner, _seen=seen)
    base = _base_type(text)
    if base and base not in _SKIP_OWNERS:
        owners.append(base)
    return owners


def _base_type(type_text: str) -> str:
    text = str(type_text or "").strip()
    text = re.sub(r"\b(?:const|volatile|static|mutable|typename|template)\b", " ", text)
    text = text.replace("&", " ").replace("*", " ")
    token = text.split("<", 1)[0].strip().split("::")[-1].strip()
    return token if token.isidentifier() else ""


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


def _param_bindings(sig: str) -> dict[str, str]:
    open_at = sig.find("(")
    close_at = sig.rfind(")")
    if open_at < 0 or close_at <= open_at:
        return {}
    out: dict[str, str] = {}
    for raw in _split_args(sig[open_at + 1 : close_at]):
        item = re.sub(r"=\s*.*$", "", raw).strip()
        if not item or item == "void" or item.startswith("..."):
            continue
        m = re.search(r"(?P<name>[A-Za-z_]\w*)\s*$", item)
        if not m:
            continue
        name = m.group("name")
        typ = item[: m.start()].strip()
        if name in _SKIP_NAMES or not typ:
            continue
        out[name] = typ
    return out


def _return_type(sig: str, name: str) -> str:
    open_at = sig.find("(")
    if open_at < 0:
        return ""
    head = sig[:open_at]
    idx = head.rfind(name)
    if idx < 0:
        return ""
    ret = head[:idx]
    ret = re.sub(r"\b(?:__aicore__|inline|constexpr|virtual|explicit|static|__host__)\b", " ", ret)
    ret = re.sub(r"\btemplate\s*<[^;{}]*>", " ", ret)
    return re.sub(r"\s+", " ", ret).strip()


def _has_assignment(prefix: str) -> bool:
    """True when ``=`` is an initializer, ignoring template default args."""
    depth = 0
    for ch in prefix:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            return True
    return False


def _looks_like_definition(prefix: str, name: str, *, in_class: bool) -> bool:
    name_pat = rf"{re.escape(name)}\s*$" if name.startswith("~") else rf"\b{re.escape(name)}\s*$"
    if not re.search(name_pat, prefix):
        return False
    head = prefix[: prefix.rfind(name)].rstrip() if name in prefix else prefix.rstrip()
    if head.endswith((".", "->")):
        return False
    if _has_assignment(prefix):
        return False
    if _STMT_RE.search(prefix):
        return False
    if re.search(r"[+\-*/,]\s*$", head.rstrip(":").rstrip()):
        return False
    spec = bool(_DECL_SPEC_RE.search(prefix))
    if head.endswith("::"):
        # Out-of-line `Ret Owner::Method(` is a definition; bare `Owner::Method(` is a call.
        before = re.sub(r"(?:[A-Za-z_]\w*(?:\s*<[^;{}()]*>)?\s*::\s*)+$", "", head).rstrip()
        return bool(spec or (before and re.search(r"[\w:>&]$", before)))
    if spec:
        return True
    if not in_class:
        return False
    if head.endswith("&"):
        return True
    if head.endswith("*"):
        before = head[:-1].rstrip()
        return bool(re.search(r"(?:void|bool|auto|int\d*|uint\d*|float|double|char|>)$", before))
    return bool(re.search(rf"[\w:>&]\s+{re.escape(name)}\s*$", prefix))


@dataclass
class SymbolDecl:
    name: str
    owner: str
    qualified: str
    return_type: str
    file: str
    line: int
    kind: str


@dataclass
class FuncScope:
    file: str
    start: int
    end: int
    owner: str
    name: str
    bindings: dict[str, str] = field(default_factory=dict)
    enter_brace: int = 0


@dataclass
class SourceSymbolIndex:
    methods: dict[tuple[str, str], SymbolDecl] = field(default_factory=dict)
    free: dict[str, list[SymbolDecl]] = field(default_factory=lambda: defaultdict(list))
    members: dict[tuple[str, str], str] = field(default_factory=dict)
    aliases: dict[tuple[str, str], str] = field(default_factory=dict)
    scopes: dict[str, list[FuncScope]] = field(default_factory=lambda: defaultdict(list))
    files: dict[str, list[str]] = field(default_factory=dict)

    def unique_method(self, owner: str, name: str) -> SymbolDecl | None:
        if not owner or not name:
            return None
        return self.methods.get((owner, name))

    def unique_free(self, name: str) -> SymbolDecl | None:
        defs = self.free.get(name) or []
        if not defs:
            return None
        files = {d.file for d in defs}
        if len(files) == 1:
            return defs[0]
        return None

    def scope_at(self, file: str, line: int) -> FuncScope | None:
        hit: FuncScope | None = None
        keys = [file, file.replace("\\", "/")]
        base = file.replace("\\", "/").split("/")[-1]
        for key, rows in self.scopes.items():
            if key in keys or key.endswith(base) or file.endswith(key):
                for sc in rows:
                    if sc.start <= line <= sc.end:
                        if hit is None or sc.start >= hit.start:
                            hit = sc
        return hit


def _record_method(index: SourceSymbolIndex, *, owner: str, name: str, sig: str, file: str, line: int) -> None:
    if not owner or not name or name in _SKIP_NAMES or name == owner or name.startswith("~"):
        return
    decl = SymbolDecl(
        name=name,
        owner=owner,
        qualified=f"{owner}::{name}",
        return_type=_return_type(sig, name),
        file=file,
        line=line,
        kind="method",
    )
    key = (owner, name)
    prev = index.methods.get(key)
    if prev is None or line < prev.line:
        index.methods[key] = decl


def _record_member(index: SourceSymbolIndex, *, owner: str, typ: str, name: str) -> None:
    if not owner or not name or name in _SKIP_NAMES or not name.isidentifier():
        return
    text = str(typ or "").strip()
    if not text or text.startswith(("using", "typedef", "enum", "public", "private", "protected")):
        return
    base = _base_type(text)
    if base in _SKIP_OWNERS and "conditional" not in text:
        return
    index.members[(owner, name)] = text


def _finish_member_blob(index: SourceSymbolIndex, *, owner: str, blob: str) -> None:
    text = " ".join(str(blob or "").split())
    if not text or not owner:
        return
    um = _USING_IN_CLASS_RE.search(text)
    if um:
        index.aliases[(owner, um.group("alias"))] = um.group("target").strip()
        return
    head, _sep, _rest = text.partition("(")
    if _METHOD_DECL_RE.search(text) and "=" not in head:
        return
    mm = _MEMBER_RE.search(text)
    if mm:
        _record_member(index, owner=owner, typ=mm.group("type").strip(), name=mm.group("name"))


def _record_free(index: SourceSymbolIndex, *, name: str, sig: str, file: str, line: int, ns: str) -> None:
    if not name or name in _SKIP_NAMES:
        return
    index.free[name].append(
        SymbolDecl(
            name=name,
            owner=ns,
            qualified=f"{ns}::{name}" if ns else name,
            return_type=_return_type(sig, name),
            file=file,
            line=line,
            kind="free",
        )
    )


def _patch_member_bindings(index: SourceSymbolIndex) -> None:
    for rows in index.scopes.values():
        for sc in rows:
            if not sc.owner:
                continue
            for (ow, field_name), typ in index.members.items():
                if ow == sc.owner and field_name not in sc.bindings:
                    sc.bindings[field_name] = typ


def _merge_symbol_index(dst: SourceSymbolIndex, src: SourceSymbolIndex) -> None:
    dst.methods.update(src.methods)
    for key, rows in src.free.items():
        dst.free[key].extend(rows)
    dst.members.update(src.members)
    dst.aliases.update(src.aliases)
    for key, rows in src.scopes.items():
        dst.scopes[key].extend(rows)
    dst.files.update(src.files)


def _index_one_source_file(
    path: Path,
    *,
    root: str,
    deadline: float,
) -> SourceSymbolIndex | None:
    if time.perf_counter() > deadline:
        return None
    index = SourceSymbolIndex()
    try:
        text = read_text(path)
    except OSError:
        return None
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    nfile = kscan.norm_file(str(path), root)
    lines = text.splitlines()
    index.files[nfile] = lines
    class_stack: list[tuple[str, int]] = []
    ns_stack: list[tuple[str, int]] = []
    brace = 0
    pending_sig = ""
    pending_line = 0
    pending_kind = ""
    pending_owner = ""
    pending_name = ""
    pending_paren = 0
    pending_type = ""
    decl_carry = ""
    func_stack: list[FuncScope] = []

    def current_class() -> str:
        return class_stack[-1][0] if class_stack else ""

    def current_ns() -> str:
        return ns_stack[-1][0] if ns_stack else ""

    def start_or_decl(*, body: bool) -> None:
        nonlocal pending_sig, pending_line, pending_kind, pending_owner, pending_name, pending_paren, decl_carry
        decl_carry = ""
        if not pending_name:
            return
        if pending_kind == "method":
            _record_method(
                index,
                owner=pending_owner,
                name=pending_name,
                sig=pending_sig,
                file=nfile,
                line=pending_line,
            )
        elif pending_kind == "free":
            _record_free(
                index,
                name=pending_name,
                sig=pending_sig,
                file=nfile,
                line=pending_line,
                ns=current_ns(),
            )
        if body:
            func_stack.append(
                FuncScope(
                    file=nfile,
                    start=pending_line,
                    end=10**9,
                    owner=pending_owner,
                    name=pending_name,
                    bindings=_param_bindings(pending_sig),
                    enter_brace=0,
                )
            )
        pending_sig = ""
        pending_line = 0
        pending_kind = ""
        pending_owner = ""
        pending_name = ""
        pending_paren = 0

    for i, raw in enumerate(lines, start=1):
        if i % 400 == 0 and time.perf_counter() > deadline:
            break
        line = kscan._strip_line_noise(raw)
        stripped = line.strip()
        opened_func = False

        if stripped:
            class_m = _CLASS_RE.search(line)
            if class_m and ";" not in line:
                class_stack.append((class_m.group("name"), brace))
            ns = _NS_RE.search(line)
            if ns and ";" not in line and "using" not in line:
                ns_stack.append((ns.group("name"), brace))

            if class_stack and not func_stack and pending_name == "":
                um = _USING_IN_CLASS_RE.search(line)
                if um:
                    index.aliases[(current_class(), um.group("alias"))] = um.group("target").strip()
                    pending_type = ""
                elif pending_type:
                    pending_type = pending_type + " " + stripped
                    if ";" in stripped:
                        _finish_member_blob(index, owner=current_class(), blob=pending_type)
                        pending_type = ""
                elif "(" not in line:
                    mm = _MEMBER_RE.search(line)
                    if mm and ";" in line:
                        _record_member(
                            index,
                            owner=current_class(),
                            typ=mm.group("type").strip(),
                            name=mm.group("name"),
                        )
                    elif (
                        stripped
                        and not stripped.startswith(("public:", "private:", "protected:", "#", "~"))
                        and ";" not in stripped
                        and "{" not in stripped
                        and not stripped.endswith(")")
                    ):
                        pending_type = stripped

            if pending_name:
                pending_sig += " " + stripped
                pending_paren += line.count("(") - line.count(")")
                if pending_paren <= 0:
                    if "{" in line:
                        start_or_decl(body=True)
                        opened_func = True
                    elif ";" in line:
                        start_or_decl(body=False)
            elif not func_stack:
                if (
                    stripped
                    and "(" not in stripped
                    and "{" not in stripped
                    and ";" not in stripped
                    and not stripped.startswith(("#", "//"))
                    and stripped not in {"public:", "private:", "protected:", "{", "}"}
                    and _DECL_CARRY_RE.search(stripped)
                ):
                    decl_carry = f"{decl_carry} {stripped}".strip()
                owner_matches = list(_METHOD_OWNER_RE.finditer(line))
                owner_m = owner_matches[-1] if owner_matches else None
                name_m = _NAME_PAREN_RE.search(line)
                owner = ""
                name = ""
                prefix = line
                if owner_m:
                    owner = owner_m.group("owner")
                    name = owner_m.group("name")
                    prefix = line[: owner_m.end()]
                elif name_m and name_m.group("name") not in _SKIP_NAMES:
                    name = name_m.group("name")
                    owner = current_class()
                    prefix = line[: name_m.end()]
                check_prefix = f"{decl_carry} {prefix}".strip() if decl_carry else prefix
                if name and _looks_like_definition(
                    check_prefix[: check_prefix.rfind("(")] if "(" in check_prefix else check_prefix,
                    name,
                    in_class=bool(owner),
                ):
                    pending_sig = f"{decl_carry} {stripped}".strip() if decl_carry else stripped
                    pending_line = i
                    pending_name = name
                    pending_owner = owner
                    pending_kind = "method" if owner else "free"
                    pending_paren = line.count("(") - line.count(")")
                    if pending_paren <= 0 and "{" in line:
                        start_or_decl(body=True)
                        opened_func = True
                    elif pending_paren <= 0 and ";" in line:
                        start_or_decl(body=False)
                elif "(" in line:
                    decl_carry = ""

            if func_stack and not pending_name:
                head = line.split("=", 1)[0]
                if "(" not in head:
                    for dm in _LOCAL_DECL_RE.finditer(line):
                        nm = dm.group("name")
                        typ = dm.group("type").strip()
                        if nm in _SKIP_NAMES or not typ or _base_type(typ) in {"auto", ""}:
                            continue
                        func_stack[-1].bindings[nm] = typ

        delta = raw.count("{") - raw.count("}")
        brace += delta
        if opened_func and func_stack:
            func_stack[-1].enter_brace = brace
            body = line[line.find("{") :] if "{" in line else ""
            if body.count("{") and body.count("{") <= body.count("}"):
                sc = func_stack.pop()
                sc.end = i
                index.scopes[nfile].append(sc)
        while class_stack and brace <= class_stack[-1][1]:
            class_stack.pop()
            pending_type = ""
        while ns_stack and brace <= ns_stack[-1][1]:
            ns_stack.pop()
        while func_stack and func_stack[-1].enter_brace and brace < func_stack[-1].enter_brace:
            sc = func_stack.pop()
            sc.end = i
            index.scopes[nfile].append(sc)

    for sc in func_stack:
        sc.end = len(lines)
        index.scopes[nfile].append(sc)
    return index


def build_source_symbol_index(
    files: list[Path],
    *,
    root: str,
    deadline: float,
) -> SourceSymbolIndex:
    from ascendc_codemap_mcp.engine.parallel import map_files

    parts = map_files(
        list(files),
        lambda path: _index_one_source_file(path, root=root, deadline=deadline),
    )
    index = SourceSymbolIndex()
    for part in parts:
        if part is not None:
            _merge_symbol_index(index, part)
    _patch_member_bindings(index)
    return index


def _receiver_type(index: SourceSymbolIndex, site: dict[str, Any], *, root: str) -> str:
    existing = str(site.get("receiver_type") or site.get("receiver_canonical_type") or "")
    if existing:
        return existing
    recv = str(site.get("receiver") or "")
    nfile = kscan.norm_file(str(site.get("file") or ""), root)
    line = int(site.get("line") or 0)
    sc = index.scope_at(nfile, line)
    if recv == "this" and sc is not None:
        return sc.owner
    if sc is None:
        return ""
    if not recv:
        return sc.owner
    if recv in sc.bindings:
        return sc.bindings[recv]
    if sc.owner:
        return index.members.get((sc.owner, recv), "")
    return ""


def _apply_decl(site: dict[str, Any], decl: SymbolDecl, *, receiver_type: str) -> None:
    if site.get("callee_usr"):
        return
    if site.get("callee_qualified") and site.get("callee_decl_file"):
        return
    site["callee_qualified"] = decl.qualified
    site["callee_decl_file"] = decl.file
    site["callee_decl_line"] = decl.line
    site["identity_kind"] = decl.kind
    site["callee_return_type"] = decl.return_type
    if receiver_type and not site.get("receiver_type"):
        site["receiver_type"] = receiver_type
        site["receiver_canonical_type"] = _base_type(receiver_type) or receiver_type


def enrich_call_sites(
    calls: list[Any],
    index: SourceSymbolIndex,
    *,
    root: str = "",
) -> int:
    """Fill callee_qualified / decl file / receiver_type when uniquely bound.

    Does not overwrite Clang USR/qualified identity. Returns how many sites
    gained a project declaration root.
    """
    dicts: list[dict[str, Any]] = []
    for site in calls:
        dicts.append(site if isinstance(site, dict) else kscan.site_as_dict(site))

    def resolve_one(d: dict[str, Any]) -> bool:
        if d.get("callee_usr") or (d.get("callee_qualified") and d.get("callee_decl_file")):
            already = True
        else:
            already = False
        nfile = kscan.norm_file(str(d.get("file") or ""), root)
        line = int(d.get("line") or 0)
        sc = index.scope_at(nfile, line)
        if sc is not None:
            if sc.owner and not d.get("caller_qualified"):
                d["caller_qualified"] = f"{sc.owner}::{sc.name}"
            if sc.name and not d.get("caller"):
                d["caller"] = sc.name
        if already:
            return False
        callee = str(d.get("callee") or "").split("::")[-1]
        if not callee or callee in _SKIP_NAMES:
            return False
        raw_line = ""
        lines = index.files.get(nfile) or []
        if 0 < line <= len(lines):
            raw_line = lines[line - 1]
        # `AscendC::Mutex::Lock<pipe>(…)` is a qualified call, not MutexBuffer::Lock.
        if re.search(
            rf"::\s*(?:template\s+)?{re.escape(callee)}\s*(?:<|\()",
            raw_line,
        ):
            m = re.search(
                rf"((?:[A-Za-z_]\w*::)+)(?:template\s+)?{re.escape(callee)}\s*(?:<|\()",
                raw_line,
            )
            if m and not d.get("callee_qualified"):
                d["callee_qualified"] = m.group(1) + callee
            return False
        recv = str(d.get("receiver") or "")
        rtype = _receiver_type(index, d, root=root)
        if recv and rtype and not d.get("receiver_type"):
            d["receiver_type"] = rtype
            d["receiver_canonical_type"] = _base_type(rtype) or rtype
        if recv:
            hits = [
                index.unique_method(owner, callee)
                for owner in _owner_candidates(rtype, index, scope_owner=sc.owner if sc else "")
            ]
            found = [h for h in hits if h is not None]
            uniq = {h.qualified: h for h in found}
            if len(uniq) != 1:
                return False
            decl = next(iter(uniq.values()))
            _apply_decl(d, decl, receiver_type=rtype)
            return True
        owner = _base_type(rtype)
        if owner:
            decl = index.unique_method(owner, callee)
            if decl is not None:
                _apply_decl(d, decl, receiver_type=rtype or owner)
                return True
        decl = index.unique_free(callee)
        if decl is not None:
            _apply_decl(d, decl, receiver_type="")
            return True
        return False

    filled = sum(1 for d in dicts if resolve_one(d))

    for key, lines in index.files.items():
        for sc in index.scopes.get(key) or []:
            last = min(sc.end, len(lines))
            for lineno in range(sc.start, last + 1):
                if lineno <= 0 or lineno > len(lines):
                    continue
                m = _AUTO_FROM_CALL_RE.search(lines[lineno - 1])
                if not m:
                    continue
                recv = m.group("recv")
                callee = m.group("callee")
                lhs = m.group("name")
                rtype = sc.bindings.get(recv) or index.members.get((sc.owner, recv), "")
                hits = [
                    index.unique_method(owner, callee)
                    for owner in _owner_candidates(rtype, index, scope_owner=sc.owner)
                ]
                found = [h for h in hits if h is not None]
                uniq = {h.qualified: h for h in found}
                if len(uniq) != 1:
                    continue
                decl = next(iter(uniq.values()))
                if _base_type(decl.return_type) in {"", "auto", "void"}:
                    continue
                sc.bindings[lhs] = decl.return_type

    extra = sum(
        1
        for d in dicts
        if not (d.get("callee_qualified") and d.get("callee_decl_file")) and resolve_one(d)
    )

    for i, site in enumerate(calls):
        if isinstance(site, dict):
            site.update(dicts[i])
        else:
            calls[i] = dicts[i]
    return filled + extra
