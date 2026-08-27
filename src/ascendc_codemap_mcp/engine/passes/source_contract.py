# -*- coding: utf-8 -*-
"""Operator-agnostic current-source enrichment for an AscendC CodeMap.

This pass is the deterministic fallback when the complete CANN translation unit
is unavailable.  It records only source-verifiable contracts shared by AscendC
operators: REG_OP API declarations, InputIndex/AttrIndex aliases, template
TilingKey declarations, TilingData classes and members, Host setter writes,
__aicore__ kernel templates, ABI positions and GET_TILING_DATA_WITH_STRUCT.

No operator name, repository macro name or free-text derivation is special-cased.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.ir.tiling_binding import (
    KEY_CHAIN_CONSUMER,
    KEY_CHAIN_DISPATCH,
    kernels_for_use_site,
    link_tiling_data_binding,
    summarize_tiling_data_bindings,
)
from ascendc_codemap_mcp.engine.ir.type_identity import (
    clang_type_short_counts,
    iter_unique_field_decls,
    named_type_is_unique,
)
from ascendc_codemap_mcp.engine.source_layout import (
    GLOBAL_KERNEL_RE,
    _path_is_under,
    follow_repo_includes,
    quoted_include_basenames,
    selected_host_files,
    selected_kernel_files,
    selected_tiling_headers,
    tpl_decl_files,
)
from ascendc_codemap_mcp.engine.tpl_dsl import _uint_bit_width, expand_tpl_source, load_quoted_include_texts, parse_args_decl

_CPP_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"}
_ENUM_RE = re.compile(r"enum\s+class\s+(InputIndex|AttrIndex)\s*:[^{]+\{(.*?)\};", re.S)
_ALIAS_INPUT_RE = re.compile(
    r"\b(?:auto|const\s+auto|[A-Za-z_:<>\s\*&]+)\s+([A-Za-z_]\w*)\s*=.{0,320}?InputIndex::([A-Za-z_]\w*)",
    re.S,
)
_ALIAS_ATTR_RE = re.compile(
    r"\b(?:auto|const\s+auto|[A-Za-z_:<>\s\*&]+)\s+([A-Za-z_]\w*)\s*=.{0,420}?AttrIndex::([A-Za-z_]\w*)",
    re.S,
)
_GET_TILING_RE = re.compile(
    r"GET_TILING_DATA_WITH_STRUCT\s*\(\s*([A-Za-z_:]\w*(?:::\w+)*)\s*,",
    re.S,
)
_GET_TILING_MEMBER_RE = re.compile(
    r"GET_TILING_DATA_MEMBER\s*\(\s*([A-Za-z_:]\w*(?:::\w+)*)\s*,",
    re.S,
)
_GET_TILING_BARE_CALL_RE = re.compile(r"\bGET_TILING_DATA\s*\(")
# Kernel TUs that skip GET_TILING_DATA* still consume the registered POD via
# ``reinterpret_cast<T*>(tiling)`` / C-style cast / GetRawTilingData.
_CAST_TILING_DATA_RE = re.compile(
    r"(?:reinterpret_cast|static_cast)\s*<\s*[^>]*?\b([A-Za-z_]\w*TilingData)\b[^>]*>"
    r"|\(\s*(?:const\s+)?(?:__\w+__\s+)*([A-Za-z_]\w*TilingData)\s*\*"
)
_CAST_TILING_USE_SITE_RE = re.compile(
    r"(?:reinterpret_cast|static_cast)\s*<\s*(?:const\s+)?(?:struct\s+|class\s+)?"
    r"([A-Za-z_:]\w*)\s*\*\s*>\s*\(\s*"
    r"(?:[^;]{0,120}?\b(?:GetRawTilingData|tiling_data|tilingData|rawTiling|tiling)\b)"
)
_API_HEAD_RE = re.compile(
    r"\.(?P<op>INPUT|OPTIONAL_INPUT|DYNAMIC_INPUT|OUTPUT|DYNAMIC_OUTPUT|ATTR|REQUIRED_ATTR)\s*\("
)
_DEF_MEMBER_RE = re.compile(
    r'this->(?P<kind>Input|Output|Attr|DynamicInput|DynamicOutput)\s*\(\s*"(?P<name>[^"]+)"\s*\)'
)
_DATATYPE_ALIAS_RE = re.compile(r"\.DATATYPE\s*\(")
_NAMED_DTYPE_VEC_RE = re.compile(
    r"std::vector\s*<\s*(?:ge::)?DataType\s*>\s+([A-Za-z_]\w*)\s*=\s*\{",
)
_DATA_TYPE_NAME_RE = re.compile(r"\.DataType\s*\(\s*([A-Za-z_]\w*)\s*\)")
_TENSOR_TYPE_NAMED_RE = re.compile(r"TensorType\s*::\s*([A-Za-z_]\w*)\s*\(\s*\)")
_REGISTER_TILING_KEY_RE = re.compile(
    r"REGISTER_TILING_FOR_TILINGKEY\s*\(\s*\"[^\"]+\"\s*,\s*([A-Za-z_:][A-Za-z0-9_:]*)\s*\)",
    re.S,
)
_REGISTER_TILING_DEFAULT_RE = re.compile(
    r"REGISTER_TILING_DEFAULT\s*\(\s*([A-Za-z_:][A-Za-z0-9_:]*)\s*\)",
    re.S,
)
# Positional packed-key helpers (TPL or decimal packers). Not GetTilingKey().
# Bare GET_TILINGKEY / GET_TILING_KEY is a real host packer; the prefix form
# covers OP_GET_TILINGKEY-style wrappers.
_PACKING_HELPER_CALL_RE = re.compile(
    r"\b(?P<name>GET_TPL_TILING_KEY|(?:[A-Z][A-Z0-9_]*)?GET_TILING_?KEY)\s*\("
)
_PACKING_CAST_WORDS = {
    "static_cast",
    "reinterpret_cast",
    "const_cast",
    "dynamic_cast",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "int",
    "unsigned",
    "long",
    "bool",
    "true",
    "false",
}
_MEMBER_PACK_ARG_RE = re.compile(
    r"^[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)+$"
)
_PRIMITIVE_TYPES = {
    "bool", "char", "short", "int", "long", "float", "double", "void",
    "unsigned", "signed", "size_t", "ptrdiff_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
}
_GLOBAL_KERNEL_RE = GLOBAL_KERNEL_RE
_PARAM_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$")
_TEMPLATE_PARAM_RE = re.compile(
    r"(?:bool|u?int(?:8|16|32|64)_t|int(?:8|16|32|64)_t|size_t|int|unsigned(?:\s+int)?)\s+([A-Za-z_]\w*)"
)
_CLASS_RE = re.compile(
    r"(?:template\s*<.*?>\s*)?(?:class|struct)\s+([A-Za-z_]\w*)[^\{;]*\{",
    re.S,
)
_MEMBER_RE = re.compile(
    # ``!isNewDeter && isTnd`` is a real std::conditional predicate (FAG TndParam).
    r"^\s*(?P<type>[A-Za-z_][\w:\s<>,*&!]*?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"(?P<arrays>(?:\[[^\]]+\]\s*)*)"
    r"(?:(?:=\s*(?P<init>[^;]+))|(?P<brace>\{[^;]*\}))?"
    r";\s*$"
)


def enrich_codemap_from_operator_source(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    stats: dict[str, Any] = {}
    api = _parse_api(codemap, root)
    stats.update({k: v for k, v in api.items() if not k.startswith("_")})
    enum_maps = _parse_host_enums(root, architecture, api)
    _link_api_to_historical_variables(codemap, root, enum_maps)

    stats.update(_parse_tiling_keys(codemap, root, architecture))
    stats.update(_parse_tiling_data(codemap, root, architecture, host_ir=host_ir, kernel_ir=kernel_ir))

    stats.update(_parse_kernel_contract(codemap, root, architecture, api, kernel_ir=kernel_ir))
    _link_tiling_data_reads(codemap, root, architecture, host_ir=host_ir, kernel_ir=kernel_ir)
    _link_tiling_key_kernel_selects(codemap, root, architecture)
    _link_nested_tiling_data_types(codemap)
    reconcile_source_declared_tiling_keys(codemap)
    codemap.meta["tiling_data_bindings"] = summarize_tiling_data_bindings(codemap)

    codemap.meta["source_contract"] = "ascendc-source-contract/v2"
    codemap.meta["source_contract_architecture"] = architecture
    codemap.meta["source_contract_stats"] = stats
    return codemap


def reconcile_source_declared_tiling_keys(codemap: CodeMap) -> None:
    """Keep ``source_declared`` aligned with the source-contract schema.

    Later TPL / clang passes may mint extra TilingKey entities (inactive
    ``#if`` branches, ``TILING_KEY_IS`` catalogs). Those are selection facts,
    not additional packing dimensions.
    """
    names = [
        str(n).strip()
        for n in (codemap.meta.get("source_declared_tiling_keys") or [])
        if str(n).strip()
    ]
    if not names:
        return
    allowed = set(names)
    for ent in codemap.by_kind(EntityKind.TILING_KEY):
        ent.attrs["source_declared"] = ent.name in allowed


def _cpp_files(path: Path, *, recursive: bool = True) -> list[Path]:
    if not path.is_dir():
        return []
    it = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in _CPP_SUFFIXES)


def _read(path: Path) -> str:
    from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

    return read_text(path)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _split_args(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    escape = False
    for ch in text:
        if quote:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {'"', "'"}:
            quote = ch
            buf.append(ch)
        elif ch in "(<[{":
            depth += 1
            buf.append(ch)
        elif ch in ")>]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return out


def _apply_tensor_dtype(ent: Entity, dtypes: list[str] | str | None) -> None:
    if not dtypes:
        return
    if not ent.attrs.get("dtype"):
        ent.attrs["dtype"] = dtypes
    facts = ent.attrs.get("facts")
    if not isinstance(facts, dict):
        facts = {}
        ent.attrs["facts"] = facts
    if not facts.get("dtype"):
        facts["dtype"] = ent.attrs.get("dtype") or dtypes


def _unique_dt(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        name = str(raw or "").strip().split("::")[-1]
        if not name.startswith("DT_") or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _dt_list_from_braces(inner: str) -> list[str]:
    return _unique_dt(tok.strip() for tok in str(inner or "").split(","))


def _parse_datatype_aliases(text: str) -> dict[str, list[str]]:
    """REG_OP ``.DATATYPE(T, TensorType({DT_...}))`` type parameters."""
    aliases: dict[str, list[str]] = {}
    for m in _DATATYPE_ALIAS_RE.finditer(text):
        close = _matching_paren(text, m.end() - 1)
        if close < 0:
            continue
        args = _split_args(text[m.end() : close])
        if len(args) < 2:
            continue
        name = args[0].strip()
        if not re.match(r"^[A-Za-z_]\w*$", name):
            continue
        dts = _tensor_dtypes(args[1])
        if dts:
            aliases[name] = dts
    return aliases


def _parse_named_dtype_vectors(text: str) -> dict[str, list[str]]:
    """``std::vector<ge::DataType> foo = { ge::DT_... }`` tables used by OpDef."""
    named: dict[str, list[str]] = {}
    for m in _NAMED_DTYPE_VEC_RE.finditer(text):
        close = text.find("}", m.end())
        if close < 0:
            continue
        dts = _dt_list_from_braces(text[m.end() : close])
        if dts:
            named[m.group(1)] = dts
    return named


def _fill_tensor_dtype_facts(codemap: CodeMap, root: Path) -> None:
    """Copy dtype onto facts and fill INPUT/OUTPUT that REG_OP created without DataType."""
    aliases: dict[str, list[str]] = {}
    named: dict[str, list[str]] = {}
    for path in _cpp_files(root / "op_graph"):
        text = _read(path)
        if "REG_OP(" in text:
            aliases.update(_parse_datatype_aliases(text))
    for path in _def_cpp_files(root):
        named.update(_parse_named_dtype_vectors(_read(path)))
    by_name: dict[str, list[Entity]] = {}
    for kind in (EntityKind.INPUT, EntityKind.OUTPUT):
        for ent in codemap.by_kind(kind):
            by_name.setdefault(ent.name, []).append(ent)
            cur = ent.attrs.get("dtype")
            if cur:
                _apply_tensor_dtype(ent, cur)
            else:
                decl = str(ent.attrs.get("declaration") or "")
                parsed = _tensor_dtypes(decl, aliases=aliases, named=named)
                if parsed:
                    _apply_tensor_dtype(ent, parsed)
    for path in _def_cpp_files(root):
        text = _read(path)
        if "this->" not in text:
            continue
        file_named = _parse_named_dtype_vectors(text)
        file_named.update(named)
        for m in _DEF_MEMBER_RE.finditer(text):
            name = m.group("name")
            semi = text.find(";", m.end())
            payload = text[m.end() : semi] if semi >= 0 else ""
            dtypes = _tensor_dtypes(payload, aliases=aliases, named=file_named)
            if not dtypes:
                continue
            for ent in by_name.get(name) or []:
                _apply_tensor_dtype(ent, dtypes)


def _tensor_dtypes(
    payload: str,
    *,
    aliases: dict[str, list[str]] | None = None,
    named: dict[str, list[str]] | None = None,
) -> list[str]:
    """Parse REG_OP TensorType / DATATYPE aliases / OpDef DataType lists or named vectors."""
    text = str(payload or "")
    m = re.search(r"TensorType\s*\(\s*\{([^}]*)\}", text)
    if m:
        return _dt_list_from_braces(m.group(1))
    m = re.search(r"DataType(?:List)?\s*\(\s*\{([^}]*)\}", text)
    if m:
        return _dt_list_from_braces(m.group(1))
    call = _DATA_TYPE_NAME_RE.search(text)
    if call and named:
        got = named.get(call.group(1))
        if got:
            return list(got)
    named_tt = _TENSOR_TYPE_NAMED_RE.search(text)
    if named_tt and aliases:
        got = aliases.get(named_tt.group(1))
        if got:
            return list(got)
    ident = text.strip().strip("\"'")
    if aliases and re.fullmatch(r"[A-Za-z_]\w*", ident):
        got = aliases.get(ident)
        if got:
            return list(got)
    return []


def _matching_paren(text: str, open_pos: int) -> int:
    depth = 0
    quote = ""
    escape = False
    for i in range(max(0, open_pos), len(text)):
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
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _upsert_api_tensor(
    codemap: CodeMap,
    *,
    kind: str,
    name: str,
    payload: str,
    file: str,
    line: int,
    bucket: list[Entity],
    required: bool,
    dynamic: bool = False,
    aliases: dict[str, list[str]] | None = None,
    named: dict[str, list[str]] | None = None,
) -> None:
    dtypes = _tensor_dtypes(payload, aliases=aliases, named=named)
    k = kind.replace("_", "").upper()
    is_output = k in {"OUTPUT", "DYNAMICOUTPUT"}
    facts: dict[str, Any] = {}
    if dtypes:
        facts["dtype"] = dtypes
    ent = codemap.upsert(
        EntityKind.OUTPUT if is_output else EntityKind.INPUT,
        name,
        attrs={
            "api_kind": "tensor",
            "required": required,
            "declaration": payload.strip(),
            "dtype": dtypes,
            "facts": facts,
            "api_index": len(bucket),
            "dynamic": dynamic,
            "provenance": "source_reg_op" if kind.isupper() else "source_op_def",
        },
        file=file,
        line=line,
        status="confirmed",
    )
    _apply_tensor_dtype(ent, dtypes)
    bucket.append(ent)


def _upsert_api_attr(
    codemap: CodeMap,
    *,
    name: str,
    payload: str,
    file: str,
    line: int,
    bucket: list[Entity],
    required: bool,
    provenance: str,
) -> None:
    parts = _split_args(payload)
    ent = codemap.upsert(
        EntityKind.INPUT,
        name,
        attrs={
            "api_kind": "attribute",
            "required": required,
            "attr_type": parts[0] if parts else payload.strip(),
            "default": parts[1] if len(parts) > 1 else None,
            "api_attr_index": len(bucket),
            "provenance": provenance,
        },
        file=file,
        line=line,
        status="confirmed",
    )
    bucket.append(ent)


def _ingest_reg_op_text(
    codemap: CodeMap,
    root: Path,
    path: Path,
    text: str,
    tensor_inputs: list[Entity],
    attrs: list[Entity],
    outputs: list[Entity],
) -> None:
    file = _rel(root, path)
    aliases = _parse_datatype_aliases(text)
    seen: set[tuple[str, str]] = set()
    for m in _API_HEAD_RE.finditer(text):
        close = _matching_paren(text, m.end() - 1)
        if close < 0:
            continue
        inner = text[m.end() : close]
        args = _split_args(inner)
        if not args:
            continue
        name = args[0].strip()
        if not re.match(r"^[A-Za-z_]\w*$", name):
            continue
        payload = ", ".join(args[1:]) if len(args) > 1 else ""
        op = m.group("op")
        key = (op, name)
        if key in seen:
            continue
        seen.add(key)
        line = _line_of(text, m.start())
        if op in {"INPUT", "OPTIONAL_INPUT", "DYNAMIC_INPUT"}:
            _upsert_api_tensor(
                codemap,
                kind=op,
                name=name,
                payload=payload,
                file=file,
                line=line,
                bucket=tensor_inputs,
                required=op == "INPUT",
                dynamic=op == "DYNAMIC_INPUT",
                aliases=aliases,
            )
        elif op in {"OUTPUT", "DYNAMIC_OUTPUT"}:
            _upsert_api_tensor(
                codemap,
                kind=op,
                name=name,
                payload=payload,
                file=file,
                line=line,
                bucket=outputs,
                required=True,
                dynamic=op == "DYNAMIC_OUTPUT",
                aliases=aliases,
            )
        else:
            _upsert_api_attr(
                codemap,
                name=name,
                payload=payload,
                file=file,
                line=line,
                bucket=attrs,
                required=op == "REQUIRED_ATTR",
                provenance="source_reg_op",
            )
    _link_api_file(codemap, file, tensor_inputs, attrs, outputs)


def _link_api_file(
    codemap: CodeMap,
    file: str,
    tensor_inputs: list,
    attrs: list,
    outputs: list,
) -> None:
    """REG_OP is one declaration. The proto FILE contains every port it names.

    Tensor INPUTs pick up host-defuse edges and become the walk from an
    anchor; attribute INPUTs (scale_value, pse_type, ...) were minted from
    the same macro and then left with degree 0. CONTAINS from the proto
    file is the compiler's grouping, and because INPUT is an anchor the
    file and the attributes become reachable together.
    """
    if not file:
        return
    file_ent = codemap.upsert(
        EntityKind.FILE,
        file,
        attrs={"role": "api", "provenance": "source_reg_op"},
        file=file,
        line=1,
        status="confirmed",
    )
    for ent in list(tensor_inputs) + list(attrs) + list(outputs):
        if str(ent.file or "").replace("\\", "/") != file.replace("\\", "/"):
            continue
        codemap.link(
            RelationKind.CONTAINS,
            file_ent.id,
            ent.id,
            attrs={"provenance": "source_reg_op", "via": "reg_op_decl"},
            status="confirmed",
        )


def _def_cpp_files(root: Path) -> list[Path]:
    host = root / "op_host"
    if not host.is_dir():
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    for path in list(host.glob("*_def.cpp")) + list(host.glob("*_def.cc")) + list(
        host.rglob("*_def.cpp")
    ):
        key = path.resolve()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        found.append(path)
    return found


def _ingest_def_cpp(
    codemap: CodeMap,
    root: Path,
    path: Path,
    tensor_inputs: list[Entity],
    attrs: list[Entity],
    outputs: list[Entity],
) -> None:
    text = _read(path)
    if "this->" not in text:
        return
    file = _rel(root, path)
    named = _parse_named_dtype_vectors(text)
    seen: set[tuple[str, str]] = {(e.kind_name(), e.name) for e in tensor_inputs + attrs + outputs}
    for m in _DEF_MEMBER_RE.finditer(text):
        kind = m.group("kind")
        name = m.group("name")
        key = (kind, name)
        line = _line_of(text, m.start())
        semi = text.find(";", m.end())
        payload = text[m.end() : semi] if semi >= 0 else ""
        dtypes = _tensor_dtypes(payload, named=named)
        if kind in {"Input", "DynamicInput", "Output", "DynamicOutput"}:
            ek = EntityKind.OUTPUT if kind in {"Output", "DynamicOutput"} else EntityKind.INPUT
            hits = list(codemap.by_name(name, kind=ek))
            if hits:
                for ent in hits:
                    _apply_tensor_dtype(ent, dtypes)
                continue
        if key in seen:
            continue
        seen.add(key)
        if kind in {"Input", "DynamicInput"}:
            _upsert_api_tensor(
                codemap,
                kind=kind,
                name=name,
                payload=payload,
                file=file,
                line=line,
                bucket=tensor_inputs,
                required=kind == "Input",
                dynamic=kind == "DynamicInput",
                named=named,
            )
        elif kind in {"Output", "DynamicOutput"}:
            _upsert_api_tensor(
                codemap,
                kind=kind,
                name=name,
                payload=payload,
                file=file,
                line=line,
                bucket=outputs,
                required=True,
                dynamic=kind == "DynamicOutput",
                named=named,
            )
        else:
            _upsert_api_attr(
                codemap,
                name=name,
                payload="",
                file=file,
                line=line,
                bucket=attrs,
                required=False,
                provenance="source_op_def",
            )


def _parse_api(codemap: CodeMap, root: Path) -> dict[str, Any]:
    tensor_inputs: list[Entity] = []
    attrs: list[Entity] = []
    outputs: list[Entity] = []
    source_files = 0
    for path in _cpp_files(root / "op_graph"):
        text = _read(path)
        if "REG_OP(" not in text:
            continue
        source_files += 1
        _ingest_reg_op_text(codemap, root, path, text, tensor_inputs, attrs, outputs)
    if not tensor_inputs and not outputs:
        for path in _def_cpp_files(root):
            source_files += 1
            _ingest_def_cpp(codemap, root, path, tensor_inputs, attrs, outputs)
    else:
        for path in _def_cpp_files(root):
            _ingest_def_cpp(codemap, root, path, tensor_inputs, attrs, outputs)
    _fill_tensor_dtype_facts(codemap, root)
    return {
        "api_source_files": source_files,
        "api_tensor_inputs": len(tensor_inputs),
        "api_attributes": len(attrs),
        "api_outputs": len(outputs),
        "_api_tensor_input_names": [e.name for e in tensor_inputs],
        "_api_attribute_names": [e.name for e in attrs],
        "_api_output_names": [e.name for e in outputs],
    }


def _parse_enum_values(body: str) -> list[str]:
    names: list[str] = []
    for raw in body.split(","):
        item = re.sub(r"//.*", "", raw).strip()
        if not item:
            continue
        name = item.split("=", 1)[0].strip()
        if re.match(r"^[A-Za-z_]\w*$", name):
            names.append(name)
    return names


def _parse_host_enums(root: Path, architecture: str, api: dict[str, Any]) -> dict[str, dict[str, str]]:
    tokens: dict[str, list[str]] = {"InputIndex": [], "AttrIndex": []}
    for path in selected_host_files(root, architecture):
        text = _read(path)
        for m in _ENUM_RE.finditer(text):
            tokens[m.group(1)] = _parse_enum_values(m.group(2))
    tensor_names = list(api.get("_api_tensor_input_names") or [])
    attr_names = list(api.get("_api_attribute_names") or [])
    return {
        "InputIndex": {token: tensor_names[i] for i, token in enumerate(tokens["InputIndex"]) if i < len(tensor_names)},
        "AttrIndex": {token: attr_names[i] for i, token in enumerate(tokens["AttrIndex"]) if i < len(attr_names)},
    }


def _runtime_source_name(ent: Entity) -> str:
    norm = ((ent.attrs.get("identity") or {}).get("normalized") or {})
    return str(norm.get("source_name") or ent.attrs.get("source_name") or "").strip()


def _source_spans(ent: Entity) -> list[dict[str, Any]]:
    spans = [src for src in (ent.attrs.get("sources") or []) if isinstance(src, dict) and src.get("file")]
    if ent.file:
        spans.append({"file": ent.file, "span": {"start_line": ent.line_start, "end_line": ent.line_end}})
    return spans


def _resolve_source_file(root: Path, raw: str) -> Path | None:
    from ascendc_codemap_mcp.engine.paths import resolve_operator_file

    return resolve_operator_file(root, raw)


def _link_api_to_historical_variables(
    codemap: CodeMap,
    root: Path,
    enum_maps: dict[str, dict[str, str]],
) -> None:
    cache: dict[Path, tuple[list[str], dict[str, str], dict[str, str]]] = {}
    variables = codemap.by_kind(EntityKind.VARIABLE)
    var_by_source = {name: e for e in variables if (name := _runtime_source_name(e))}

    for var in variables:
        for src in _source_spans(var):
            candidate = _resolve_source_file(root, str(src.get("file") or ""))
            if candidate is None:
                continue
            if candidate not in cache:
                text = _read(candidate)
                lines = text.splitlines()
                input_alias: dict[str, str] = {}
                attr_alias: dict[str, str] = {}
                for m in _ALIAS_INPUT_RE.finditer(text):
                    api_name = enum_maps.get("InputIndex", {}).get(m.group(2))
                    if api_name:
                        input_alias[m.group(1)] = api_name
                for m in _ALIAS_ATTR_RE.finditer(text):
                    api_name = enum_maps.get("AttrIndex", {}).get(m.group(2))
                    if api_name:
                        attr_alias[m.group(1)] = api_name
                cache[candidate] = (lines, input_alias, attr_alias)
            lines, input_alias, attr_alias = cache[candidate]
            span = src.get("span") or {}
            start = max(1, int(span.get("start_line") or var.line_start or 1))
            end = max(start, int(span.get("end_line") or start))
            snippet = "\n".join(lines[start - 1 : min(len(lines), end)])
            for alias, api_name in {**input_alias, **attr_alias}.items():
                if not re.search(rf"\b{re.escape(alias)}\b", snippet):
                    continue
                for inp in codemap.by_name(api_name, kind=EntityKind.INPUT):
                    codemap.link(
                        RelationKind.DERIVES,
                        inp.id,
                        var.id,
                        attrs={
                            "provenance": "source_host_alias",
                            "file": _rel(root, candidate),
                            "line_start": start,
                            "line_end": end,
                            "alias": alias,
                        },
                        status="confirmed",
                    )
            for source_name, source_ent in var_by_source.items():
                if source_ent.id == var.id:
                    continue
                if re.search(rf"\b{re.escape(source_name)}\b", snippet):
                    codemap.link(
                        RelationKind.DERIVES,
                        source_ent.id,
                        var.id,
                        attrs={
                            "provenance": "source_host_assignment",
                            "file": _rel(root, candidate),
                            "line_start": start,
                            "line_end": end,
                        },
                        status="confirmed",
                    )


def _macro_ints(text: str) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in re.findall(r"^\s*#define\s+([A-Za-z_]\w*)\s+([0-9]+)\s*$", text, re.M)
    }


def _parse_allowed_values(tail: str, ints: dict[str, int] | None = None) -> list[int | str]:
    marker = ""
    if "ASCENDC_TPL_UI_LIST" in tail:
        marker = "ASCENDC_TPL_UI_LIST"
    elif "ASCENDC_TPL_UI_RANGE" in tail:
        marker = "ASCENDC_TPL_UI_RANGE"
    if not marker:
        return []
    after = tail.split(marker, 1)[1]
    out: list[int | str] = []
    table = ints or {}
    for token in _split_args(after.lstrip(", \n\t")):
        token = token.strip().rstrip(")")
        if not token:
            continue
        try:
            out.append(int(token, 0))
        except ValueError:
            if token in table:
                out.append(table[token])
            else:
                out.append(token)
    return out


def _decl_bit_width(decl_kind: str, width_token: str, ints: dict[str, int]) -> int:
    if decl_kind == "BOOL":
        return 1
    token = (width_token or "").strip()
    if token in ints:
        return ints[token]
    try:
        return int(token, 0)
    except ValueError:
        return _uint_bit_width(token)


def _dim_decl_line(text: str, decl_kind: str, name: str) -> int:
    match = re.search(
        rf"ASCENDC_TPL_{re.escape(decl_kind)}_DECL\s*\(\s*{re.escape(name)}\b",
        text,
    )
    return _line_of(text, match.start()) if match else 1


def _parse_tiling_keys(codemap: CodeMap, root: Path, architecture: str) -> dict[str, Any]:
    declared: list[str] = []
    bit_shift = 0
    for path in tpl_decl_files(root, architecture):
        raw = _read(path)
        if "ASCENDC_TPL_ARGS_DECL" not in raw:
            continue
        extras = load_quoted_include_texts(path)
        text = expand_tpl_source(raw, extras)
        ints = _macro_ints(text)
        for extra in extras:
            ints.update(_macro_ints(extra))
        schema = parse_args_decl(text)
        for dim in schema.dims:
            name = str(dim.name or "").strip()
            if not name or name in declared:
                continue
            decl_kind = str(dim.kind or "UINT").upper()
            order = len(declared)
            declared.append(name)
            tail = ",".join(str(v) for v in (dim.vals or []))
            if decl_kind == "BOOL":
                allowed = sorted({int(v) for v in re.findall(r"\b[01]\b", tail)})
                for tok in dim.vals or []:
                    token = str(tok).strip()
                    if token in ints and ints[token] in (0, 1):
                        allowed = sorted(set(allowed) | {ints[token]})
                if not allowed:
                    allowed = [0, 1]
            else:
                allowed = _parse_allowed_values(tail, ints)
                if not allowed and decl_kind in {"DTYPE", "FORMAT"}:
                    allowed = [str(v).strip() for v in (dim.vals or []) if str(v).strip()]
            width = int(dim.bw) if dim.bw else _decl_bit_width(decl_kind, tail.split(",")[0] if tail else "", ints)
            line = _dim_decl_line(text, decl_kind, name)
            bit_lo = bit_shift
            bit_hi = bit_lo + max(int(width), 1) - 1
            bit_shift = bit_hi + 1
            attrs = {
                "bit_width": width,
                "bw": width,
                "bit_lo": bit_lo,
                "bit_hi": bit_hi,
                "bit_offset": bit_lo,
                "allowed_values": allowed,
                "value_domain": [str(v) for v in allowed],
                "decl_kind": decl_kind.lower(),
                "source_declared": True,
                "provenance": "source_tpl_args_decl",
                "decl_order": order,
            }
            existing = codemap.by_name(name, kind=EntityKind.TILING_KEY)
            if existing:
                ent = existing[0]
                ent.attrs.update(
                    {
                        k: v
                        for k, v in attrs.items()
                        if v is not None and not (k in {"allowed_values", "value_domain"} and not v)
                    }
                )
                ent.file = _rel(root, path)
                ent.line_start = line
                ent.line_end = line
                ent.status = "confirmed"
                ent.confidence = 1.0
            else:
                codemap.upsert(
                    EntityKind.TILING_KEY,
                    name,
                    attrs=attrs,
                    file=_rel(root, path),
                    line=line,
                    status="confirmed",
                )
    if not declared:
        declared = _parse_fallback_tiling_keys(codemap, root, architecture)
    packed = _collect_packed_key_catalog(root, architecture)
    if packed:
        codemap.meta["source_packed_legal_keys"] = packed
        codemap.meta["source_packed_legal_key_count"] = len(packed)
    codemap.meta["source_declared_tiling_keys"] = declared
    codemap.meta["source_declared_tiling_key_count"] = len(declared)
    if declared:
        allowed = set(declared)
        for ent in codemap.by_kind(EntityKind.TILING_KEY):
            if ent.name in allowed:
                ent.attrs["source_declared"] = True
            elif ent.attrs.get("source_declared"):
                ent.attrs["source_declared"] = False
                ent.attrs["unselected_tpl_schema"] = True
    return {
        "source_declared_tiling_keys": len(declared),
        "source_has_tiling_key_sites": bool(declared),
    }


_TILING_KEY_IS_INT = r"(?:0[xX][0-9A-Fa-f]+|\d+)[uUlL]*"
_TILING_KEY_IS_RE = re.compile(
    rf"\bTILING_KEY_IS\s*\(\s*([A-Za-z_]\w*|{_TILING_KEY_IS_INT})\s*\)"
)
_DEFINE_TILING_KEY_RE = re.compile(
    r"^\s*#\s*define\s+((?:T(?:I)?LING_KEY_|[A-Z0-9_]*TILING_KEY)[A-Za-z0-9_]*)\b"
    r"(?:\s+(-?\d+))?",
    re.MULTILINE,
)
_CONSTEXPR_TILING_KEY_RE = re.compile(
    r"\b(?:const\s+static|static\s+const|static\s+constexpr|constexpr\s+static|constexpr)\s+"
    r"(?:const\s+)?(?:u?int(?:32|64)_t|int)\s+"
    r"((?:T(?:I)?LING_KEY_|[A-Z0-9_]*TILING_KEY)[A-Za-z0-9_]*)\s*"
    r"(?:=\s*(-?\d+))?"
)
_SET_TILING_KEY_IDENT_RE = re.compile(
    r"\b(?:SetTilingKey|set_tiling_key)\s*\(\s*([A-Za-z_]\w*)"
)
_NOT_TILING_KEY_IDENTS = {
    "GetTilingKey",
    "get_tiling_key",
    "SetTilingKey",
    "set_tiling_key",
}
_GENERIC_TILING_OBJECTS = {
    "tiling",
    "tilingData",
    "tiling_data",
    "tilingKey",
    "tiling_key",
    "tilingKey_",
    "tiling_key_",
    "td",
    "rawTiling",
    "raw_tiling",
}


def _normalize_tiling_key_is_token(raw: str) -> str:
    """Strip C integer suffixes so ``TILING_KEY_IS(24UL)`` is the catalog value 24."""
    token = str(raw or "").strip()
    stripped = re.sub(r"[uUlL]+$", "", token)
    if re.fullmatch(r"(?:0[xX][0-9A-Fa-f]+|\d+)", stripped):
        return stripped
    return token


def _is_include_guard_ident(name: str) -> bool:
    token = str(name or "")
    return token.startswith("__") and token.endswith("__") or bool(re.search(r"_H__?$", token))


def _looks_like_tiling_key_ident(name: str) -> bool:
    """Name-spelling score only. Not identity. Sinks use ``_is_tiling_key_sink_token``."""
    token = str(name or "")
    if not token or token in _NOT_TILING_KEY_IDENTS or _is_include_guard_ident(token):
        return False
    norm = _normalize_tiling_key_is_token(token)
    if re.fullmatch(r"(?:0[xX][0-9A-Fa-f]+|\d+)", norm):
        return True
    upper = token.upper()
    return "TILING_KEY" in upper or upper.endswith("TILINGKEY")


def _is_tiling_key_sink_token(name: str) -> bool:
    """Identity from SetTilingKey / TILING_KEY_IS / GET_TILING_KEY arguments.

    Catalog macros need not contain ``TILING_KEY``. Generic tiling objects
    (the POD pointer, packed ``tilingKey_``) are not keys.
    """
    token = str(name or "")
    if not _is_tiling_key_is_catalog_token(token):
        return False
    if token in _GENERIC_TILING_OBJECTS:
        return False
    if _looks_like_tiling_key_ident(token):
        return True
    norm = _normalize_tiling_key_is_token(token)
    if re.fullmatch(r"(?:0[xX][0-9A-Fa-f]+|\d+)", norm):
        return True
    return bool(re.fullmatch(r"[A-Z][A-Z0-9]*(_[A-Z0-9]+)+", token))


def _is_tiling_key_is_catalog_token(name: str) -> bool:
    """``TILING_KEY_IS(x)`` is the dispatch catalog; ``x`` need not contain TILING_KEY.

    Host/kernel macros such as ``NORMAL_INT32_FULLY_LOAD`` are still keys. Only
    include guards and Get/SetTilingKey identifiers are rejected.
    """
    token = str(name or "")
    if not token or token in _NOT_TILING_KEY_IDENTS or _is_include_guard_ident(token):
        return False
    return True


def _function_like_defines(text: str) -> list[tuple[str, list[str], str]]:
    """``#define NAME(a, b)`` bodies with backslash continuations joined."""
    out: list[tuple[str, list[str], str]] = []
    for match in re.finditer(
        r"^\s*#\s*define\s+([A-Za-z_]\w*)\s*\(([^)]*)\)",
        text,
        re.MULTILINE,
    ):
        params = [p.strip() for p in match.group(2).split(",") if p.strip() and p.strip() != "..."]
        i = match.end()
        chunks: list[str] = []
        while True:
            nl = text.find("\n", i)
            if nl < 0:
                chunks.append(text[i:])
                break
            line = text[i:nl]
            if line.rstrip().endswith("\\"):
                chunks.append(line.rstrip()[:-1] + "\n")
                i = nl + 1
                continue
            chunks.append(line)
            break
        out.append((match.group(1), params, "".join(chunks)))
    return out


def _tiling_key_is_wrapper_macros(text: str) -> dict[str, int]:
    """Function-like macros whose body is ``TILING_KEY_IS(param)`` → arg index."""
    wrappers: dict[str, int] = {}
    for name, params, body in _function_like_defines(text):
        for match in _TILING_KEY_IS_RE.finditer(body):
            token = _normalize_tiling_key_is_token(match.group(1))
            if token in params:
                wrappers[name] = params.index(token)
                break
    return wrappers


def _tiling_key_is_wrapper_params(text: str) -> set[str]:
    params: set[str] = set()
    for _name, macro_params, body in _function_like_defines(text):
        for match in _TILING_KEY_IS_RE.finditer(body):
            token = _normalize_tiling_key_is_token(match.group(1))
            if token in macro_params:
                params.add(token)
    return params


def iter_tiling_key_is_catalog(text: str) -> Iterable[tuple[str, int]]:
    """Yield catalog tokens from ``TILING_KEY_IS`` and wrapper-macro invocations.

    ``#define BRANCH(tilingKey, ...) { if (TILING_KEY_IS(tilingKey)) ... }``
    plus ``BRANCH(TILING_KEY_1111, ...)`` mints ``TILING_KEY_1111``, not the
    parameter name.
    """
    wrapper_params = _tiling_key_is_wrapper_params(text)
    seen: set[str] = set()
    for match in _TILING_KEY_IS_RE.finditer(text):
        name = _normalize_tiling_key_is_token(match.group(1))
        if name in seen or name in wrapper_params or not _is_tiling_key_is_catalog_token(name):
            continue
        seen.add(name)
        yield name, match.start()
    for macro, index in _tiling_key_is_wrapper_macros(text).items():
        for match in re.finditer(rf"\b{re.escape(macro)}\s*\(", text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            if re.search(r"#\s*define\b", text[line_start : match.start()]):
                continue
            open_pos = match.end() - 1
            close = _matching_paren(text, open_pos)
            if close < 0:
                continue
            args = [a.strip() for a in _split_args(text[open_pos + 1 : close]) if a.strip()]
            if index >= len(args):
                continue
            name = _normalize_tiling_key_is_token(args[index])
            if name in seen or not _is_tiling_key_is_catalog_token(name):
                continue
            seen.add(name)
            yield name, match.start()


def iter_packing_helper_calls(text: str) -> Iterable[tuple[int, int, list[str], str]]:
    """Yield positional packed-key helper calls: GET_TPL_TILING_KEY / *_GET_TILINGKEY.

    Definitions (parameter packs / typenames) are skipped. ``GET_TPL_TILING_KEY``
    packs one or more TilingKey dimensions. Other helpers must pack at least
    two arguments — a single argument is the already-packed catalog integer
    ``TILING_KEY_IS`` enumerates.
    """
    for match in _PACKING_HELPER_CALL_RE.finditer(text):
        name = match.group("name")
        open_pos = match.end() - 1
        close_pos = _matching_paren(text, open_pos)
        if close_pos < 0:
            continue
        args_text = text[open_pos + 1 : close_pos]
        if "..." in args_text or re.search(r"\btypename\b", args_text):
            continue
        args = [a.strip() for a in _split_args(args_text) if a.strip()]
        # GET_TPL_TILING_KEY(oneDim) is still a packing formula. Other
        # GET_TILINGKEY(packedInt) helpers with a single argument are the
        # already-packed catalog integer, not dimensions.
        if not args:
            continue
        if len(args) < 2 and name != "GET_TPL_TILING_KEY":
            continue
        yield match.start(), close_pos + 1, args, name


def _packing_dim_name(expr: str, index: int) -> str:
    expr = expr.strip()
    if _MEMBER_PACK_ARG_RE.match(expr):
        return re.findall(r"[A-Za-z_]\w*", expr)[-1]
    idents = [tok for tok in re.findall(r"[A-Za-z_]\w*", expr) if tok not in _PACKING_CAST_WORDS]
    if len(idents) == 1:
        return idents[0]
    return ""


_BITPACK_ACC = r"(?:tilingKey_|tiling_key_|tilingKey)"
_SHIFT_PACK_RE = re.compile(
    rf"\b(?P<acc>{_BITPACK_ACC})\s*=\s*\(\s*(?P=acc)\s*<<\s*(?P<bw>\d+)\s*\)\s*\+\s*(?P<rhs>[^;]+);"
)
_TYPED_INIT_RE = re.compile(
    rf"\b(?:(?:u?int(?:32|64)_t)|auto)\s+(?P<acc>{_BITPACK_ACC})\s*=\s*(?P<rhs>[^;]+);"
)
_PLUS_EQ_LIT_RE = re.compile(
    rf"\b(?P<acc>{_BITPACK_ACC})\s*\+=\s*(?P<lit>\d+)[uUlL]*\s*;"
)
_PLUS_EQ_SCALE_RE = re.compile(
    rf"\b(?P<acc>{_BITPACK_ACC})\s*\+=\s*(?P<rhs>[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*)\s*;"
)
_MUL_PLACE_RE = re.compile(
    rf"\b(?P<acc>{_BITPACK_ACC})\s*(?:\*=\s*10[uUlL]*|=\s*(?P=acc)\s*\*\s*10[uUlL]*)\s*;"
)
_CAST_WRAP_RE = re.compile(r"^static_cast\s*<[^>]+>\s*\((.*)\)\s*$", re.S)
_CMP_LHS_RE = re.compile(r"^(.+?)\s*(?:==|!=)\s*.+$")
_SCALE_LHS_RE = re.compile(r"^([A-Za-z_]\w*)\s*\*")
_IF_IDENT_RE = re.compile(r"if\s*\(\s*!?\s*([A-Za-z_]\w*)\s*\)")
_ACC_ASSIGN_LIT_RE = re.compile(
    rf"\b(?P<acc>{_BITPACK_ACC})\s*=\s*(?P<lit>\d+)[uUlL]*\s*;"
)
_ACC_ASSIGN_MACRO_RE = re.compile(
    rf"\b(?P<acc>{_BITPACK_ACC})\s*=\s*(?P<rhs>[A-Z][A-Z0-9_]*)\s*;"
)
_PLUS_EQ_SCALE_LIT_RE = re.compile(
    rf"\b(?P<acc>{_BITPACK_ACC})\s*\+=\s*\(?\s*(?P<ident>[A-Za-z_]\w*)\s*\*\s*(?P<lit>\d+)[uUlL]*\s*\)?\s*;"
)
_HOST_FN_RE = re.compile(
    r"(?:inline\s+|static\s+|virtual\s+|constexpr\s+)*"
    r"[A-Za-z_][\w:<>,\s*&~]*?\s+"
    r"(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*"
    r"\([^;{}]*\)\s*(?:const\s*)?(?:override\s*)?\{",
    re.S,
)
_IF_OPEN_RE = re.compile(r"\bif\s*\(")
_COND_PACK_SKIP = _PACKING_CAST_WORDS | {
    "true",
    "false",
    "nullptr",
    "NULL",
    "ge",
    "std",
    "this",
    "if",
    "else",
    "return",
}


def _unwrap_cast_expr(expr: str) -> str:
    expr = expr.strip()
    match = _CAST_WRAP_RE.match(expr)
    if match:
        return match.group(1).strip()
    return expr


def _bitpack_dim_name(expr: str, index: int) -> str:
    inner = _unwrap_cast_expr(expr)
    cmp_match = _CMP_LHS_RE.match(inner)
    if cmp_match:
        return _packing_dim_name(cmp_match.group(1).strip(), index)
    scale_match = _SCALE_LHS_RE.match(inner)
    if scale_match:
        return scale_match.group(1)
    return _packing_dim_name(inner, index)


def _unique_dim_name(name: str, used: set[str], index: int) -> str:
    if name not in used:
        used.add(name)
        return name
    alt = f"{name}_{index}"
    used.add(alt)
    return alt


def _decimal_place(value: int) -> int:
    value = abs(int(value))
    if value == 0:
        return -1
    place = 0
    while value % 10 == 0:
        value //= 10
        place += 1
    return place


def _cond_pack_ident(cond: str) -> str | None:
    cond = cond or ""
    chain = re.findall(r"[A-Za-z_]\w*(?:\s*(?:\.|->)\s*[A-Za-z_]\w*)+", cond)
    if chain:
        toks = [
            tok
            for tok in re.findall(r"[A-Za-z_]\w*", chain[0])
            if tok not in _COND_PACK_SKIP
        ]
        if toks:
            return toks[-1]
    for tok in re.findall(r"[A-Za-z_]\w*", cond):
        if tok not in _COND_PACK_SKIP:
            return tok
    return None


_FN_NAME_SKIP = {"if", "else", "while", "for", "switch", "catch"}


def _iter_fn_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in _HOST_FN_RE.finditer(text):
        name = match.group("name").split("::")[-1]
        if name in _FN_NAME_SKIP:
            continue
        open_brace = match.end() - 1
        close = _matching_brace(text, open_brace)
        if close > open_brace:
            spans.append((open_brace, close))
    return spans or [(0, len(text))]


def _if_bodies(text: str, start: int, end: int) -> list[tuple[int, int, str]]:
    """Innermost-friendly list of ``if`` bodies in ``text[start:end]``."""
    out: list[tuple[int, int, str]] = []
    pos = start
    while pos < end:
        match = _IF_OPEN_RE.search(text, pos, end)
        if not match:
            break
        open_par = match.end() - 1
        close_par = _matching_paren(text, open_par)
        if close_par < 0 or close_par >= end:
            break
        cond = text[open_par + 1 : close_par]
        cursor = close_par + 1
        while cursor < end and text[cursor] in " \t\r\n":
            cursor += 1
        if cursor < end and text[cursor] == "{":
            close_brace = _matching_brace(text, cursor)
            if close_brace < 0 or close_brace > end:
                break
            out.append((cursor, close_brace, cond))
        else:
            semi = text.find(";", cursor, end)
            if semi < 0:
                break
            out.append((cursor, semi, cond))
        pos = close_par + 1
    return out


def _innermost_if_cond(bodies: list[tuple[int, int, str]], offset: int) -> str | None:
    covering = [item for item in bodies if item[0] <= offset <= item[1]]
    if not covering:
        return None
    covering.sort(key=lambda item: item[1] - item[0])
    return covering[0][2]


def _iter_literal_pack_dims(text: str) -> list[dict[str, Any]]:
    """Host ``if (axis) tilingKey += LITERAL`` decimal-field packing axes.

    Distinct controlling identifiers become dimensions only when the literals
    occupy at least two decimal places. A named-macro catalog plus ``+= 1``
    is not a multi-axis pack.
    """
    best: list[dict[str, Any]] = []
    for start, end in _iter_fn_spans(text):
        body = text[start : end + 1]
        abs_start = start
        if _ACC_ASSIGN_MACRO_RE.search(body):
            continue
        if_bodies = _if_bodies(text, start, end)
        grouped: dict[str, dict[str, Any]] = {}
        places: set[int] = set()
        shift_offs = [m.start() for m in _MUL_PLACE_RE.finditer(body)]

        def _effective_place(lit: int, body_off: int) -> int:
            digit = _decimal_place(lit)
            shifted = sum(1 for pos in shift_offs if pos < body_off)
            if shifted and digit <= 0:
                return shifted
            return digit

        def _note(name: str, offset: int, expr: str, lit: int) -> None:
            if not name:
                return
            place = _effective_place(lit, offset - abs_start)
            if place >= 0:
                places.add(place)
            slot = grouped.setdefault(
                name,
                {"name": name, "expr": expr, "offset": offset, "lits": []},
            )
            if offset < int(slot["offset"]):
                slot["offset"] = offset
                slot["expr"] = expr
            if lit:
                slot["lits"].append(lit)

        for match in _PLUS_EQ_LIT_RE.finditer(body):
            lit = int(match.group("lit"))
            if lit == 0:
                continue
            abs_off = abs_start + match.start()
            ident = _cond_pack_ident(_innermost_if_cond(if_bodies, abs_off) or "")
            if ident:
                _note(ident, abs_off, ident, lit)
        for match in _ACC_ASSIGN_LIT_RE.finditer(body):
            lit = int(match.group("lit"))
            if lit == 0:
                continue
            abs_off = abs_start + match.start()
            ident = _cond_pack_ident(_innermost_if_cond(if_bodies, abs_off) or "")
            if ident:
                _note(ident, abs_off, ident, lit)
        for match in _PLUS_EQ_SCALE_LIT_RE.finditer(body):
            ident = match.group("ident")
            lit = int(match.group("lit"))
            abs_off = abs_start + match.start()
            cond_ident = _cond_pack_ident(_innermost_if_cond(if_bodies, abs_off) or "")
            _note(cond_ident or ident, abs_off, ident, lit)

        names = [name for name, slot in grouped.items() if slot["lits"]]
        if len(names) < 2 or len(places) < 2:
            continue
        names.sort(key=lambda name: int(grouped[name]["offset"]))
        dims = [
            {
                "name": name,
                "expr": grouped[name]["expr"],
                "bw": 0,
                "offset": grouped[name]["offset"],
                "bit_lo": None,
                "bit_hi": None,
            }
            for name in names
        ]
        if len(dims) > len(best):
            best = dims
    return best


def iter_bitpack_dims(text: str) -> list[dict[str, Any]]:
    """Host packing *axes*: shift-chain, weighted-add, or if-gated decimal fields.

    These are TilingKey *dimensions*, not the expanded ``TILING_KEY_IS`` catalog.
    A lone ``tilingKey_ = NAMED_KEY; tilingKey_ += 1`` is not a pack chain.
    """
    shifts = list(_SHIFT_PACK_RE.finditer(text))
    if shifts:
        acc = shifts[0].group("acc")
        chain = [m for m in shifts if m.group("acc") == acc]
        if not chain:
            return []
        first = chain[0].start()
        init_rhs = None
        init_off = 0
        for init in _TYPED_INIT_RE.finditer(text[:first]):
            if init.group("acc") != acc:
                continue
            rhs = init.group("rhs").strip()
            if re.fullmatch(r"\d+[uUlL]*", rhs):
                continue
            if "<<" in rhs:
                continue
            init_rhs = rhs
            init_off = init.start()
        dims: list[dict[str, Any]] = []
        used: set[str] = set()
        if init_rhs:
            name = _unique_dim_name(_bitpack_dim_name(init_rhs, 0), used, 0)
            dims.append({"name": name, "expr": init_rhs, "bw": 1, "offset": init_off})
        for index, match in enumerate(chain, start=len(dims)):
            rhs = match.group("rhs").strip()
            name = _unique_dim_name(_bitpack_dim_name(rhs, index), used, index)
            dims.append(
                {
                    "name": name,
                    "expr": rhs,
                    "bw": int(match.group("bw")),
                    "offset": match.start(),
                }
            )
        last = chain[-1].end()
        window = text[last : last + 400]
        plus = _PLUS_EQ_LIT_RE.search(window)
        if plus and plus.group("acc") == acc:
            prefix = text[max(0, last + plus.start() - 120) : last + plus.start()]
            ident = _IF_IDENT_RE.findall(prefix)
            name = ident[-1] if ident else f"pack_flag_{plus.group('lit')}"
            name = _unique_dim_name(name, used, len(dims))
            dims.append(
                {
                    "name": name,
                    "expr": plus.group(0).strip(),
                    "bw": 0,
                    "offset": last + plus.start(),
                }
            )
        if len(dims) < 2:
            return []
        cursor = 0
        for dim in reversed(dims):
            bw = max(int(dim["bw"]), 1) if dim["bw"] else 0
            if bw:
                dim["bit_lo"] = cursor
                dim["bit_hi"] = cursor + bw - 1
                cursor = dim["bit_hi"] + 1
            else:
                dim["bit_lo"] = None
                dim["bit_hi"] = None
        return dims

    scales = list(_PLUS_EQ_SCALE_RE.finditer(text))
    if len(scales) >= 2:
        acc = scales[0].group("acc")
        chain = [m for m in scales if m.group("acc") == acc]
        raw_names = [_bitpack_dim_name(m.group("rhs").strip(), i) for i, m in enumerate(chain)]
        if len(chain) >= 2 and len(set(raw_names)) >= 2:
            dims = []
            used: set[str] = set()
            for index, match in enumerate(chain):
                rhs = match.group("rhs").strip()
                name = _unique_dim_name(raw_names[index], used, index)
                dims.append(
                    {
                        "name": name,
                        "expr": rhs,
                        "bw": 0,
                        "offset": match.start(),
                        "bit_lo": None,
                        "bit_hi": None,
                    }
                )
            return dims

    return _iter_literal_pack_dims(text)


def _parse_bitpack_tiling_keys(
    codemap: CodeMap,
    root: Path,
    architecture: str,
    *,
    paths: list[Path] | None = None,
) -> list[str]:
    """Mint packing *dimensions* from host bit-pack / weighted-add / decimal-field chains."""
    best: list[dict[str, Any]] | None = None
    site: Path | None = None
    host_paths = list(paths) if paths is not None else list(selected_host_files(root, architecture))
    for path in host_paths:
        try:
            text = _read(path)
        except OSError:
            continue
        dims = iter_bitpack_dims(text)
        if not dims:
            continue
        if best is None or len(dims) > len(best):
            best = dims
            site = path
    if not best or site is None:
        return []
    text = _read(site)
    declared: list[str] = []
    for order, dim in enumerate(best):
        name = dim["name"]
        declared.append(name)
        line = _line_of(text, int(dim["offset"]))
        bw = int(dim.get("bw") or 0)
        attrs: dict[str, Any] = {
            "source_declared": True,
            "provenance": "source_bitpack_dim",
            "decl_order": order,
            "decl_kind": "uint",
            "host_packing_expressions": [dim["expr"]],
        }
        if bw:
            attrs["bit_width"] = bw
            attrs["bw"] = bw
        if dim.get("bit_lo") is not None:
            attrs["bit_lo"] = dim["bit_lo"]
            attrs["bit_hi"] = dim["bit_hi"]
            attrs["bit_offset"] = dim["bit_lo"]
        existing = codemap.by_name(name, kind=EntityKind.TILING_KEY)
        if existing:
            ent = existing[0]
            ent.attrs.update(attrs)
            ent.file = _rel(root, site)
            ent.line_start = line
            ent.line_end = line
            ent.status = "confirmed"
            ent.confidence = 1.0
        else:
            codemap.upsert(
                EntityKind.TILING_KEY,
                name,
                attrs=attrs,
                file=_rel(root, site),
                line=line,
                status="confirmed",
            )
    return declared


def _collect_packed_key_catalog(root: Path, architecture: str) -> list[str]:
    """``TILING_KEY_IS`` packed values are the legal-key set, not dimensions."""
    names: list[str] = []
    seen: set[str] = set()
    for path in selected_kernel_files(root, architecture):
        try:
            text = _read(path)
        except OSError:
            continue
        for name, _offset in iter_tiling_key_is_catalog(text):
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def _local_source_files(root: Path, files: list[Path]) -> list[Path]:
    return [path for path in files if _path_is_under(path, root)]


def _foreign_source_files(root: Path, files: list[Path]) -> list[Path]:
    return [path for path in files if not _path_is_under(path, root)]


def _parse_packing_helper_keys(
    codemap: CodeMap,
    root: Path,
    architecture: str,
    *,
    paths: list[Path] | None = None,
) -> list[str]:
    """Recover TilingKey dimensions from host packed-key helper call sites."""
    schema: list[str] | None = None
    schema_args: list[str] | None = None
    site: tuple[Path, int] | None = None
    host_paths = list(paths) if paths is not None else list(selected_host_files(root, architecture))
    for path in host_paths:
        try:
            text = _read(path)
        except OSError:
            continue
        for start, _end, args, _name in iter_packing_helper_calls(text):
            names: list[str] = []
            named_exprs: list[str] = []
            used: set[str] = set()
            for index, expr in enumerate(args):
                name = _packing_dim_name(expr, index)
                if not name or name.startswith("pack_arg_"):
                    continue
                if name in used:
                    name = f"{name}_{index}"
                used.add(name)
                names.append(name)
                named_exprs.append(expr)
            if not names:
                continue
            if schema is None or len(names) > len(schema):
                schema = names
                schema_args = named_exprs
                site = (path, _line_of(text, start))
    if not schema or site is None:
        return []
    path, line = site
    declared: list[str] = []
    for order, name in enumerate(schema):
        declared.append(name)
        expr = (schema_args or [""] * len(schema))[order]
        codemap.upsert(
            EntityKind.TILING_KEY,
            name,
            attrs={
                "source_declared": True,
                "provenance": "source_packing_helper_arg",
                "decl_order": order,
                "decl_kind": "uint",
                "host_packing_expressions": [expr] if expr else [],
                "packing_value_sites": [
                    {"file": _rel(root, path), "line": line, "expression": expr[:300]}
                ]
                if expr
                else [],
            },
            file=_rel(root, path),
            line=line,
            status="confirmed",
        )
    return declared


def _parse_fallback_tiling_keys(
    codemap: CodeMap, root: Path, architecture: str
) -> list[str]:
    """Integer/macro TilingKeys used when there is no ASCENDC_TPL_*_DECL."""
    host_files = list(selected_host_files(root, architecture))
    local_host = _local_source_files(root, host_files)
    foreign_host = _foreign_source_files(root, host_files)
    helper_keys = _parse_packing_helper_keys(
        codemap, root, architecture, paths=local_host
    )
    if helper_keys:
        return helper_keys
    bitpack_keys = _parse_bitpack_tiling_keys(
        codemap, root, architecture, paths=local_host
    )
    if bitpack_keys:
        return bitpack_keys
    found: list[tuple[str, Path, int, str, int | None]] = []
    seen: set[str] = set()
    kernel_files = list(selected_kernel_files(root, architecture))

    def _collect(files: list[Path], patterns: list[tuple[re.Pattern[str], str]]) -> None:
        for path in files:
            try:
                text = _read(path)
            except OSError:
                continue
            defines: dict[str, int] = {}
            for dm in re.finditer(
                r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(-?\d+)[uUlL]*",
                text,
                re.MULTILINE,
            ):
                defines[dm.group(1)] = int(dm.group(2))
            for regex, prov in patterns:
                if regex is _TILING_KEY_IS_RE and prov == "source_tiling_key_is":
                    for name, offset in iter_tiling_key_is_catalog(text):
                        if name in seen:
                            continue
                        seen.add(name)
                        found.append(
                            (
                                name,
                                path,
                                _line_of(text, offset),
                                prov,
                                defines.get(name),
                            )
                        )
                    continue
                for m in regex.finditer(text):
                    name = _normalize_tiling_key_is_token(m.group(1))
                    if name in seen:
                        continue
                    if regex is _SET_TILING_KEY_IDENT_RE:
                        if not _is_tiling_key_sink_token(name):
                            continue
                    elif not _is_tiling_key_is_catalog_token(name):
                        continue
                    seen.add(name)
                    value = None
                    if m.lastindex and m.lastindex >= 2 and m.group(2):
                        raw = str(m.group(2))
                        if raw.lstrip("-").isdigit():
                            value = int(raw)
                    if value is None:
                        value = defines.get(name)
                    found.append((name, path, _line_of(text, m.start()), prov, value))

    _collect(kernel_files, [(_TILING_KEY_IS_RE, "source_tiling_key_is")])
    # Kernel ``TILING_KEY_IS(10000)`` is the dispatch contract. Host
    # ``TLING_KEY_*`` / ``FULL_LOAD_*TILING_KEY`` literals pack onto those
    # values; they are not extra keys. Sibling / 3rd-party GET_TPL must not
    # replace that catalog just because Clang confirmed those files.
    if not found:
        helper_keys = _parse_packing_helper_keys(
            codemap, root, architecture, paths=foreign_host
        )
        if helper_keys:
            return helper_keys
        bitpack_keys = _parse_bitpack_tiling_keys(
            codemap, root, architecture, paths=foreign_host
        )
        if bitpack_keys:
            return bitpack_keys
        _collect(
            kernel_files + host_files,
            [
                (_DEFINE_TILING_KEY_RE, "source_tiling_key_define"),
                (_CONSTEXPR_TILING_KEY_RE, "source_tiling_key_constexpr"),
                (_SET_TILING_KEY_IDENT_RE, "source_set_tiling_key"),
            ],
        )
    declared: list[str] = []
    for name, path, line, prov, value in found:
        declared.append(name)
        attrs = {
            "source_declared": True,
            "provenance": prov,
            "decl_order": len(declared) - 1,
            "decl_kind": "uint",
            "candidate_score": 2 if _looks_like_tiling_key_ident(name) else 1,
            "key_chain_role": KEY_CHAIN_DISPATCH if prov == "source_tiling_key_is" else KEY_CHAIN_CONSUMER,
        }
        if value is None and re.fullmatch(r"\d+", name):
            value = int(name)
        elif value is None and re.fullmatch(r"0[xX][0-9A-Fa-f]+", name):
            value = int(name, 16)
        if value is not None:
            attrs["value"] = value
            attrs["allowed_values"] = [value]
            attrs["value_domain"] = [str(value)]
        codemap.upsert(
            EntityKind.TILING_KEY,
            name,
            attrs=attrs,
            file=_rel(root, path),
            line=line,
            status="confirmed",
        )
    return declared


def _matching_brace(text: str, open_pos: int) -> int:
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


def _class_members(body: str, body_abs_start: int, text: str) -> Iterable[tuple[str, str, int]]:
    depth = 0
    abs_off = 0
    for raw in body.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        line_no = _line_of(text, body_abs_start + abs_off)
        stripped = re.sub(r"//.*", "", line).strip()
        if (
            depth == 0
            and stripped
            and "(" not in stripped
            and not stripped.endswith(":")
            and not stripped.startswith(("using ", "typedef ", "template ", "static_assert"))
        ):
            m = _MEMBER_RE.match(stripped)
            if m:
                cpp_type = " ".join(m.group("type").split())
                name = m.group("name")
                arrays = re.sub(r"\s+", "", m.group("arrays") or "")
                if arrays:
                    cpp_type = f"{cpp_type}{arrays}"
                if name not in {"public", "private", "protected"} and cpp_type:
                    yield cpp_type, name, line_no
        depth += line.count("{") - line.count("}")
        depth = max(0, depth)
        abs_off += len(raw)


def wanted_tiling_data_names(codemap: CodeMap, root: Path, architecture: str) -> set[str]:
    """TilingData types from registration / GET_TILING_DATA contracts."""
    names = {e.name.split("::")[-1] for e in codemap.by_kind(EntityKind.TILING_DATA) if e.name}
    for path in _kernel_candidates(root, architecture):
        text = _read(path)
        names.update(n.split("::")[-1] for n in _GET_TILING_RE.findall(text))
        names.update(n.split("::")[-1] for n in _GET_TILING_MEMBER_RE.findall(text))
        names.update(n.split("::")[-1] for n in _REGISTER_TILING_KEY_RE.findall(text))
        names.update(n.split("::")[-1] for n in _REGISTER_TILING_DEFAULT_RE.findall(text))
        names.update(_cast_tiling_data_names(text))
    names.discard("")
    return names


def _cast_tiling_data_names(text: str) -> set[str]:
    out: set[str] = set()
    for match in _CAST_TILING_DATA_RE.finditer(text or ""):
        name = match.group(1) or match.group(2)
        if name:
            out.add(name.split("::")[-1])
    for match in _CAST_TILING_USE_SITE_RE.finditer(text or ""):
        name = (match.group(1) or "").split("::")[-1]
        if name and name not in _PRIMITIVE_TYPES:
            out.add(name)
    return out


def _class_index(files: list[Path]) -> dict[str, tuple[Path, str, int, int, int]]:
    """Map class name → (path, text, match_start, body_open, body_close)."""
    index: dict[str, tuple[Path, str, int, int, int]] = {}
    for path in files:
        text = _read(path)
        for m in _CLASS_RE.finditer(text):
            if re.search(r"\benum\s+$", text[: m.start()]):
                continue
            name = m.group(1)
            open_pos = text.find("{", m.start(), m.end())
            close_pos = _matching_brace(text, open_pos)
            if close_pos < 0:
                continue
            index.setdefault(name, (path, text, m.start(), open_pos, close_pos))
    return index


def _cpp_type_name(cpp_type: str) -> str:
    cleaned = re.sub(r"\b(?:const|volatile|mutable|static|inline|typename|class|struct)\b", " ", cpp_type)
    cleaned = re.sub(r"<.*", "", cleaned)
    cleaned = cleaned.replace("*", " ").replace("&", " ").strip()
    if not cleaned:
        return ""
    return cleaned.split()[-1].split("::")[-1]


_WORD_TYPE_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_USING_ALIAS_RE = re.compile(
    r"\busing\s+([A-Za-z_]\w*)\s*=\s*([^;]+);",
    re.S,
)


def _referenced_type_names(cpp_type: str, known: set[str]) -> set[str]:
    """All known type tokens inside a member type (incl. std::conditional arms)."""
    return {t for t in _WORD_TYPE_RE.findall(cpp_type or "") if t in known}


def _resolve_tiling_aliases(text: str, known_classes: set[str]) -> dict[str, str]:
    """Map ``using Alias = ConcreteType<...>`` onto class-index names."""
    out: dict[str, str] = {}
    for match in _USING_ALIAS_RE.finditer(text):
        alias = match.group(1)
        targets = [
            t for t in _WORD_TYPE_RE.findall(match.group(2) or "") if t in known_classes
        ]
        if len(targets) == 1:
            out[alias] = targets[0]
    return out


def _parse_tiling_data(
    codemap: CodeMap,
    root: Path,
    architecture: str,
    *,
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> dict[str, Any]:
    class_count = 0
    field_count = 0
    files = list(_kernel_candidates(root, architecture))
    seen_files: set[Path] = {p.resolve() for p in files}
    for path in list(selected_tiling_headers(root, architecture)) + list(
        selected_host_files(root, architecture)
    ):
        key = path.resolve()
        if key in seen_files:
            continue
        seen_files.add(key)
        files.append(path)
    try:
        from ascendc_codemap_mcp.engine.paths import ops_root as _ops_root

        repo = _ops_root()
    except Exception:
        repo = None
    if repo is None:
        repo = root.parent.parent
    for extra in follow_repo_includes(files, repo_root=Path(repo), architecture=architecture):
        key = extra.resolve()
        if key in seen_files:
            continue
        seen_files.add(key)
        files.append(extra)
    index = _class_index(files)
    known_classes = set(index)
    aliases: dict[str, str] = {}
    for path in files:
        try:
            aliases.update(_resolve_tiling_aliases(_read(path), known_classes))
        except OSError:
            continue
    wanted = wanted_tiling_data_names(codemap, root, architecture)
    clang_fields = {(o, m) for o, m, *_ in iter_unique_field_decls(host_ir, kernel_ir)}
    # Collapse ``using Alias = ConcreteType<...>`` onto the class index name.
    for alias, concrete in aliases.items():
        if alias in wanted:
            wanted.add(concrete)
    seen: set[str] = set()
    queue = list(wanted)
    while queue:
        owner = queue.pop()
        if not owner or owner in seen:
            continue
        owner = aliases.get(owner, owner)
        if owner in seen:
            continue
        loc = index.get(owner)
        if loc is None:
            if owner in wanted:
                existing = codemap.by_name(owner, kind=EntityKind.TILING_DATA)
                if not existing:
                    codemap.upsert(
                        EntityKind.TILING_DATA,
                        owner,
                        attrs={"provenance": "source_tiling_data_type_identity", "architecture": architecture},
                        status="partial",
                    )
            # Leave ``seen`` unset so BEGIN_TILING_DATA_DEF can still emit fields.
            continue
        seen.add(owner)
        path, text, start, open_pos, close_pos = loc
        line = _line_of(text, start)
        owner_ent = codemap.upsert(
            EntityKind.TILING_DATA,
            owner,
            attrs={"provenance": "source_tiling_data_class", "architecture": architecture},
            file=_rel(root, path),
            line=line,
            status="confirmed",
        )
        class_count += 1
        body = text[open_pos + 1 : close_pos]
        for cpp_type, field_name, field_line in _class_members(body, open_pos + 1, text):
            nested_hits = _referenced_type_names(cpp_type, known_classes)
            nested = _cpp_type_name(cpp_type)
            if nested and nested not in _PRIMITIVE_TYPES and nested in index:
                nested_hits.add(nested)
            for nested_name in nested_hits:
                if nested_name not in _PRIMITIVE_TYPES:
                    queue.append(nested_name)
                    wanted.add(nested_name)
            if (owner, field_name) in clang_fields:
                continue
            field = codemap.upsert(
                EntityKind.TILING_FIELD,
                field_name,
                eid=f"TDF::{owner}::{field_name}",
                attrs={
                    "owner": owner,
                    "qualified_name": f"{owner}::{field_name}",
                    "cpp_type": cpp_type,
                    "provenance": "source_tiling_data_member",
                },
                file=_rel(root, path),
                line=field_line,
                status="confirmed",
            )
            codemap.link(
                RelationKind.DECLARES,
                owner_ent.id,
                field.id,
                attrs={"provenance": "source_tiling_data_class"},
                status="confirmed",
            )
            field_count += 1
    from ascendc_codemap_mcp.engine.tiling_data_ir import parse_macro_structs

    for path in files:
        try:
            text = _read(path)
        except OSError:
            continue
        if "BEGIN_TILING_DATA_DEF" not in text:
            continue
        for st in parse_macro_structs(text, file=_rel(root, path)):
            if st.name in seen:
                continue
            seen.add(st.name)
            owner_ent = codemap.upsert(
                EntityKind.TILING_DATA,
                st.name,
                attrs={"provenance": "source_tiling_data_macro", "architecture": architecture},
                file=_rel(root, path),
                line=st.line,
                status="confirmed",
            )
            class_count += 1
            for field in st.fields:
                field_ent = codemap.upsert(
                    EntityKind.TILING_FIELD,
                    field.name,
                    eid=f"TDF::{st.name}::{field.name}",
                    attrs={
                        "owner": st.name,
                        "qualified_name": f"{st.name}::{field.name}",
                        "cpp_type": field.ctype,
                        "provenance": "source_tiling_data_macro_field",
                    },
                    file=_rel(root, path),
                    line=field.line,
                    status="confirmed",
                )
                codemap.link(
                    RelationKind.DECLARES,
                    owner_ent.id,
                    field_ent.id,
                    attrs={"provenance": "source_tiling_data_macro"},
                    status="confirmed",
                )
                field_count += 1
    codemap.meta["source_tiling_data_class_count"] = class_count
    codemap.meta["source_tiling_data_field_count"] = field_count
    return {"source_tiling_data_classes": class_count, "source_tiling_data_fields": field_count}


def _link_nested_tiling_data_types(codemap: CodeMap) -> None:
    owners = {e.name: e for e in codemap.by_kind(EntityKind.TILING_DATA)}
    known = set(owners)
    for field in codemap.by_kind(EntityKind.TILING_FIELD):
        cpp_type = str(field.attrs.get("cpp_type") or "")
        targets = _referenced_type_names(cpp_type, known)
        simple = _cpp_type_name(cpp_type)
        if simple in known:
            targets.add(simple)
        for name in targets:
            target = owners.get(name)
            if target is None:
                continue
            codemap.link(
                RelationKind.REFERENCES,
                field.id,
                target.id,
                attrs={"provenance": "source_tiling_data_member_type"},
                status="confirmed",
            )


def _kernel_candidates(root: Path, architecture: str) -> list[Path]:
    return selected_kernel_files(root, architecture)


def _param_name(raw: str) -> str:
    raw = raw.split("=", 1)[0].strip()
    m = _PARAM_NAME_RE.search(raw)
    return m.group(1) if m else ""


_ABI_SKIP_NAMES = frozenset(
    {
        "tiling",
        "tiling_data",
        "tilingdata",
        "tiling_arg",
        "tilingarg",
        "workspace",
        "usrworkspace",
        "aicore_sync",
        "aicoresync",
        "sync",
    }
)


def is_abi_skip_param(name: str) -> bool:
    compact = str(name or "").replace("_", "").lower()
    if compact in _ABI_SKIP_NAMES:
        return True
    return "tiling" in compact and compact.endswith("data")


def _squash_ident(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _abi_squash_keys(param: str) -> set[str]:
    """Kernel ``queryRope`` / ``deqScaleQ`` vs REG_OP ``query_rope`` / ``d_scale_q``."""
    keys: set[str] = set()
    squashed = _squash_ident(param)
    if squashed:
        keys.add(squashed)
    if squashed.startswith("deq") and len(squashed) > 3:
        keys.add("d" + squashed[3:])
    return keys


def _unique_io_for_kernel_param(codemap: CodeMap, param: str) -> tuple[Any | None, Any | None]:
    """Bind by exact name, then by unique squashed spelling. Never zip by position."""
    inputs = list(codemap.by_name(param, kind=EntityKind.INPUT))
    outputs = list(codemap.by_name(param, kind=EntityKind.OUTPUT))
    if len(inputs) == 1 and not outputs:
        return inputs[0], None
    if len(outputs) == 1 and not inputs:
        return None, outputs[0]
    if inputs or outputs:
        return None, None
    keys = _abi_squash_keys(param)
    if not keys:
        return None, None
    in_hits = [
        entity
        for entity in codemap.by_kind(EntityKind.INPUT)
        if _squash_ident(entity.name) in keys
    ]
    out_hits = [
        entity
        for entity in codemap.by_kind(EntityKind.OUTPUT)
        if _squash_ident(entity.name) in keys
    ]
    if len(in_hits) == 1 and not out_hits:
        return in_hits[0], None
    if len(out_hits) == 1 and not in_hits:
        return None, out_hits[0]
    return None, None


def kernel_params_from_ir(kernel_ir: Any, kernel_name: str) -> list[str]:
    if kernel_ir is None:
        return []
    rec = (getattr(kernel_ir, "functions", None) or {}).get(kernel_name) or {}
    params = [str(p) for p in (rec.get("params") or []) if str(p)]
    if params:
        return params
    short = kernel_name.split("::")[-1]
    for name, row in (getattr(kernel_ir, "functions", None) or {}).items():
        if str(name).split("::")[-1] == short:
            return [str(p) for p in (row.get("params") or []) if str(p)]
    return []


def link_kernel_abi_by_param_name(
    codemap: CodeMap,
    kernel: Entity,
    params: list[str],
    *,
    provenance: str,
    file: str,
    line: int,
) -> int:
    """Bind INPUT/OUTPUT by kernel parameter name. Never zip by position."""
    linked = 0
    seen_params: set[str] = set()
    for param in params:
        if not param or param in seen_params or is_abi_skip_param(param):
            continue
        seen_params.add(param)
        inp, out = _unique_io_for_kernel_param(codemap, param)
        if inp is not None and out is None:
            codemap.link(
                RelationKind.FLOWS_TO,
                inp.id,
                kernel.id,
                attrs={
                    "provenance": provenance,
                    "kernel_param": param,
                    "file": file,
                    "line": line,
                },
                status="confirmed",
            )
            linked += 1
        elif out is not None and inp is None:
            codemap.link(
                RelationKind.FLOWS_TO,
                kernel.id,
                out.id,
                attrs={
                    "provenance": provenance,
                    "kernel_param": param,
                    "file": file,
                    "line": line,
                },
                status="confirmed",
            )
            linked += 1
    return linked


def _parse_kernel_contract(
    codemap: CodeMap,
    root: Path,
    architecture: str,
    api: dict[str, Any],
    *,
    kernel_ir: Any = None,
) -> dict[str, Any]:
    input_names = list(api.get("_api_tensor_input_names") or [])
    output_names = list(api.get("_api_output_names") or [])
    kernel_count = 0
    tpl_args_bound = 0
    abi_links = 0
    seen_kernel_ids: set[str] = set()
    for path in _kernel_candidates(root, architecture):
        text = _read(path)
        for m in _GLOBAL_KERNEL_RE.finditer(text):
            name = m.group("name")
            line = _line_of(text, m.start())
            kernels = codemap.by_name(name, kind=EntityKind.KERNEL)
            if kernels:
                kernel = kernels[0]
                kernel.attrs.update({"source_signature": True, "architecture": architecture, "provenance": "source_kernel_signature"})
                kernel.file = _rel(root, path)
                kernel.line_start = line
                kernel.line_end = line
                kernel.status = "confirmed"
                kernel.confidence = 1.0
            else:
                kernel = codemap.upsert(
                    EntityKind.KERNEL,
                    name,
                    attrs={"source_signature": True, "architecture": architecture, "provenance": "source_kernel_signature"},
                    file=_rel(root, path),
                    line=line,
                    status="confirmed",
                )
            if kernel.id not in seen_kernel_ids:
                kernel_count += 1
                seen_kernel_ids.add(kernel.id)

            template = codemap.upsert(
                EntityKind.TEMPLATE,
                f"{name}<template>",
                attrs={"target": name, "architecture": architecture, "provenance": "source_kernel_template"},
                file=_rel(root, path),
                line=line,
                status="confirmed",
            )
            codemap.link(RelationKind.DEFINES, template.id, kernel.id, attrs={"provenance": "source_kernel_template"}, status="confirmed")
            for order, arg_name in enumerate(_TEMPLATE_PARAM_RE.findall(m.group("tpl") or "")):
                arg = codemap.upsert(
                    EntityKind.TEMPLATE_ARG,
                    arg_name,
                    eid=f"TPLARG::{name}::{arg_name}",
                    attrs={"owner": name, "order": order, "provenance": "source_kernel_template"},
                    file=_rel(root, path),
                    line=line,
                    status="confirmed",
                )
                codemap.link(RelationKind.DECLARES, template.id, arg.id, attrs={"provenance": "source_kernel_template"}, status="confirmed")
                for key in codemap.by_name(arg_name, kind=EntityKind.TILING_KEY):
                    codemap.mint_candidate_relation(
                        RelationKind.BINDS,
                        key.id,
                        arg.id,
                        provenance="source_tpl_name_match",
                    )
                    codemap.link(RelationKind.CONTROLS, arg.id, kernel.id, attrs={"provenance": "source_kernel_template_param"}, status="confirmed")
                    tpl_args_bound += 1

            params = [_param_name(x) for x in _split_args(m.group("params"))]
            params = [p for p in params if p]
            clang_params = kernel_params_from_ir(kernel_ir, name)
            if clang_params:
                abi_links += link_kernel_abi_by_param_name(
                    codemap,
                    kernel,
                    clang_params,
                    provenance="clang_kernel_abi",
                    file=_rel(root, path),
                    line=line,
                )
            elif params:
                abi_links += link_kernel_abi_by_param_name(
                    codemap,
                    kernel,
                    params,
                    provenance="source_kernel_abi_position",
                    file=_rel(root, path),
                    line=line,
                )
    return {
        "source_kernel_entries": kernel_count,
        "source_template_args_bound": tpl_args_bound,
        "source_kernel_abi_links": abi_links,
    }


def _link_tiling_data_reads(
    codemap: CodeMap,
    root: Path,
    architecture: str,
    *,
    host_ir: Any = None,
    kernel_ir: Any = None,
) -> None:
    owners = {e.name: e for e in codemap.by_kind(EntityKind.TILING_DATA)}
    clang_counts = clang_type_short_counts(host_ir, kernel_ir)
    macro_names = {
        e.name
        for e in owners.values()
        if "macro" in str(e.attrs.get("provenance") or "").lower()
        or str(e.attrs.get("provenance") or "") == "source_tiling_data_macro"
    }
    for path in _kernel_candidates(root, architecture):
        text = _read(path)
        typed = {name.split("::")[-1] for name in _GET_TILING_RE.findall(text)}
        typed.update(name.split("::")[-1] for name in _GET_TILING_MEMBER_RE.findall(text))
        typed.update(_cast_tiling_data_names(text))
        # Untyped GET_TILING_DATA is only a contract when REGISTER_TILING_DEFAULT
        # names the struct. Never spray every TilingData identifier in the file.
        if _GET_TILING_BARE_CALL_RE.search(text):
            typed.update(
                name.split("::")[-1] for name in _REGISTER_TILING_DEFAULT_RE.findall(text)
            )
        if not typed:
            continue
        target_kernels = kernels_for_use_site(codemap, path, text, root)
        if not target_kernels:
            continue
        for type_name in typed:
            if not named_type_is_unique(
                type_name, clang_counts=clang_counts, macro_names=macro_names
            ):
                continue
            tdata = owners.get(type_name)
            if tdata is None:
                continue
            for kernel in target_kernels:
                link_tiling_data_binding(
                    codemap,
                    tdata,
                    kernel,
                    provenance="source_get_tiling_data",
                    file=_rel(root, path),
                )


def _kernels_for_packed_key_file(
    codemap: CodeMap, path: Path, text: str, root: Path
) -> list[Entity]:
    """Pick kernels that consume a packed-key catalog in this file."""
    return kernels_for_use_site(codemap, path, text, root)


def _link_tiling_key_kernel_selects(codemap: CodeMap, root: Path, architecture: str) -> None:
    """``TILING_KEY_IS`` selects a kernel. Packed catalogs bind dimensions, not 1:1 names."""
    keys = {e.name: e for e in codemap.by_kind(EntityKind.TILING_KEY)}
    if not keys:
        return
    packing_dims = [
        e
        for e in keys.values()
        if str(e.attrs.get("provenance") or "")
        in {"source_packing_helper_arg", "source_bitpack_dim"}
    ]
    for path in _kernel_candidates(root, architecture):
        text = _read(path)
        names = [name for name, _offset in iter_tiling_key_is_catalog(text)]
        if not names:
            continue
        file_rel = _rel(root, path)
        target_kernels = _kernels_for_packed_key_file(codemap, path, text, root)
        if not target_kernels:
            continue
        if packing_dims:
            for key in packing_dims:
                for kernel in target_kernels:
                    codemap.link(
                        RelationKind.SELECTS,
                        key.id,
                        kernel.id,
                        attrs={
                            "provenance": "source_packing_helper_selects",
                            "file": file_rel,
                            "key_chain_role": KEY_CHAIN_DISPATCH,
                        },
                        status="confirmed",
                    )
            continue
        matched = False
        for name in names:
            key = keys.get(name)
            if key is None:
                continue
            matched = True
            for kernel in target_kernels:
                codemap.link(
                    RelationKind.SELECTS,
                    key.id,
                    kernel.id,
                    attrs={
                        "provenance": "source_tiling_key_is",
                        "file": file_rel,
                        "key_chain_role": KEY_CHAIN_DISPATCH,
                    },
                    status="confirmed",
                )
        if matched:
            continue
        # Packed integer catalog: host produces one key (``tilingKey`` /
        # SetTilingKey) and the kernel enumerates legal values with
        # TILING_KEY_IS(QF16_..._TILING). Those spellings are not dimensions.
        packed = [
            e
            for e in keys.values()
            if e.attrs.get("source_declared")
            and str(e.attrs.get("provenance") or "") != "source_tpl_args_decl"
        ]
        for key in packed:
            for kernel in target_kernels:
                codemap.link(
                    RelationKind.SELECTS,
                    key.id,
                    kernel.id,
                    attrs={
                        "provenance": "source_packed_key_is_selects",
                        "file": file_rel,
                        "key_chain_role": KEY_CHAIN_DISPATCH,
                    },
                    status="confirmed",
                )

    declared_tpl = [
        e
        for e in keys.values()
        if e.attrs.get("source_declared")
        and str(e.attrs.get("provenance") or "") == "source_tpl_args_decl"
    ]
    schema_headers = {
        Path(str(e.file or "")).name.lower()
        for e in declared_tpl
        if e.file
    }
    if declared_tpl and schema_headers:
        for path in _kernel_candidates(root, architecture):
            text = _read(path)
            if not GLOBAL_KERNEL_RE.search(text):
                continue
            included = {name.lower() for name in quoted_include_basenames(path)}
            if not (included & schema_headers):
                continue
            file_rel = _rel(root, path)
            target_kernels = _kernels_for_packed_key_file(codemap, path, text, root)
            for key in declared_tpl:
                for kernel in target_kernels:
                    codemap.link(
                        RelationKind.SELECTS,
                        key.id,
                        kernel.id,
                        attrs={
                            "provenance": "source_tpl_header_selects",
                            "file": file_rel,
                            "key_chain_role": KEY_CHAIN_DISPATCH,
                        },
                        status="confirmed",
                    )

    # A single KERNEL is not evidence that every declared TilingKey selects it.


def source_contract_stats(codemap: CodeMap) -> dict[str, int]:
    inputs = codemap.by_kind(EntityKind.INPUT)
    return {
        "api_tensor_inputs": sum(1 for e in inputs if e.attrs.get("api_kind") == "tensor"),
        "api_attributes": sum(1 for e in inputs if e.attrs.get("api_kind") == "attribute"),
        "outputs": len(codemap.by_kind(EntityKind.OUTPUT)),
        "tiling_keys": len(codemap.by_kind(EntityKind.TILING_KEY)),
        "tiling_data": len(codemap.by_kind(EntityKind.TILING_DATA)),
        "tiling_fields": len(codemap.by_kind(EntityKind.TILING_FIELD)),
        "kernels": len(codemap.by_kind(EntityKind.KERNEL)),
    }
