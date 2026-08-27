# -*- coding: utf-8 -*-
"""Complete current-source TilingData declarations, including array/conditional members.

The original lightweight member regex accepted scalar declarations only and
rejected types containing template predicates such as ``!isNewDeter``.  Those
members are real ABI fields and must exist before read/write closure is built.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.ir.type_identity import iter_unique_field_decls, short_type_name
from ascendc_codemap_mcp.engine.passes.source_contract import _kernel_candidates, _rel as _src_rel
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.source_layout import selected_tiling_headers

_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_CLASS_RE = re.compile(
    r"(?:template\s*<.*?>\s*)?(?:class|struct)\s+([A-Za-z_]\w*)[^\{;]*\{",
    re.S,
)
# Type text is intentionally permissive.  The field declarator at the end of a
# top-level class line is the stable anchor; restricting the type grammar loses
# valid std::conditional/decltype/template spellings.
_MEMBER_RE = re.compile(
    r"^\s*(?P<type>.+?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"(?P<arrays>(?:\[[^\]]+\]\s*)*)"
    r"(?:=\s*(?P<init>[^;]+))?;\s*$"
)


def complete_tiling_fields(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    kernel_root = root / "op_kernel"
    if not kernel_root.is_dir():
        return codemap

    owners = {e.name: e for e in codemap.by_kind(EntityKind.TILING_DATA)}
    added = 0
    arrays = 0
    initializers = 0
    created_owners = 0
    clang_fields = 0
    paths: list[Path] = []
    seen: set[Path] = set()
    for path in list(_kernel_candidates(root, architecture)) + list(
        selected_tiling_headers(root, architecture)
    ):
        if not path.is_file() or path.suffix.lower() not in _SUFFIXES:
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)

    # Clang members first so lexical complete cannot mint the same (kind,src,dst)
    # at advisory trust and freeze out the authoritative record.
    for owner_name, field_name, cpp_type, file, line in iter_unique_field_decls(
        host_ir, kernel_ir
    ):
        owner = owners.get(owner_name) or owners.get(short_type_name(owner_name))
        if owner is None:
            continue
        eid = f"TDF::{owner.name}::{field_name}"
        existing = codemap.entities.get(eid)
        if existing is None:
            field = codemap.upsert(
                EntityKind.TILING_FIELD,
                field_name,
                eid=eid,
                attrs={
                    "owner": owner.name,
                    "qualified_name": f"{owner.name}::{field_name}",
                    "cpp_type": cpp_type,
                    "provenance": "clang_field_decl",
                },
                file=file,
                line=line or 0,
                status="confirmed",
            )
            codemap.link(
                RelationKind.DECLARES,
                owner.id,
                field.id,
                attrs={"provenance": "clang_field_decl"},
                status="confirmed",
            )
            added += 1
            clang_fields += 1
        else:
            if cpp_type:
                existing.attrs.setdefault("cpp_type", cpp_type)
        array_suffix = "".join(re.findall(r"\[[^\]]*\]", cpp_type or ""))
        if array_suffix:
            field = existing or codemap.entities[eid]
            field.attrs["array_extent"] = re.sub(r"\s+", "", array_suffix)
            field.attrs["is_array"] = True
            arrays += 1

    for path in paths:
        text = read_text(path)
        for match in _CLASS_RE.finditer(text):
            if re.search(r"\benum\s+$", text[: match.start()]):
                continue
            owner_name = match.group(1)
            owner = owners.get(owner_name)
            # REGISTER / GET_TILING_DATA name the type. Filename globs must not.
            if owner is None:
                continue
            open_pos = text.find("{", match.start(), match.end())
            close_pos = _matching_brace(text, open_pos)
            if close_pos < 0:
                continue
            body = text[open_pos + 1:close_pos]
            depth = 0
            abs_off = 0
            for raw_line in body.splitlines(keepends=True):
                line = raw_line.rstrip("\r\n")
                field_line = _line(text, open_pos + 1 + abs_off)
                stripped = re.sub(r"//.*", "", line).strip()
                if (
                    depth == 0
                    and stripped
                    and "(" not in stripped
                    and not stripped.endswith(":")
                    and not stripped.startswith(("using ", "typedef ", "template ", "static_assert"))
                ):
                    mm = _MEMBER_RE.match(stripped)
                    if mm:
                        cpp_type = " ".join(mm.group("type").split())
                        # Access labels and other non-declarations are already
                        # excluded above; still fail closed on obviously empty
                        # or preprocessor-like type text.
                        if not cpp_type or cpp_type.startswith("#"):
                            mm = None
                    if mm:
                        field_name = mm.group("name")
                        array_suffix = re.sub(r"\s+", "", mm.group("arrays") or "")
                        initializer = (mm.group("init") or "").strip()
                        eid = f"TDF::{owner_name}::{field_name}"
                        existing = codemap.entities.get(eid)
                        if existing is None:
                            field = codemap.upsert(
                                EntityKind.TILING_FIELD,
                                field_name,
                                eid=eid,
                                attrs={
                                    "owner": owner_name,
                                    "qualified_name": f"{owner_name}::{field_name}",
                                    "cpp_type": cpp_type,
                                    "provenance": "source_tiling_data_member_complete",
                                },
                                file=_src_rel(root, path),
                                line=field_line,
                                status="confirmed",
                            )
                            codemap.link(
                                RelationKind.DECLARES,
                                owner.id,
                                field.id,
                                attrs={"provenance": "source_tiling_data_class"},
                                status="confirmed",
                            )
                            added += 1
                        else:
                            field = existing
                            field.attrs.setdefault("owner", owner_name)
                            field.attrs.setdefault("qualified_name", f"{owner_name}::{field_name}")
                            if str(field.attrs.get("provenance") or "").startswith("clang"):
                                field.attrs.setdefault("cpp_type", cpp_type)
                            else:
                                field.attrs["cpp_type"] = cpp_type
                        if array_suffix:
                            field.attrs["array_extent"] = array_suffix
                            field.attrs["is_array"] = True
                            arrays += 1
                        if initializer:
                            field.attrs["default_initializer"] = initializer
                            field.attrs["default_initializer_site"] = {
                                "file": _src_rel(root, path),
                                "line": field_line,
                            }
                            initializers += 1
                depth += line.count("{") - line.count("}")
                depth = max(0, depth)
                abs_off += len(raw_line)

    codemap.meta["source_tiling_data_complete"] = {
        "added_fields": added,
        "created_owners": created_owners,
        "clang_fields": clang_fields,
        "array_fields": arrays,
        "default_initializers": initializers,
        "total_fields": len(codemap.by_kind(EntityKind.TILING_FIELD)),
        "policy": "source-member-array-conditional/v3",
    }
    return codemap


def _matching_brace(text: str, open_pos: int) -> int:
    if open_pos < 0:
        return -1
    depth = 0
    quote = ""
    escape = False
    for idx in range(open_pos, len(text)):
        ch = text[idx]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {'\"', "'"}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1
