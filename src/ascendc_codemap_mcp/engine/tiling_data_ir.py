# -*- coding: utf-8 -*-
"""TilingData struct inventory + host writers + kernel readers.

Two source forms are accepted:

* arch22-style macros: ``BEGIN_TILING_DATA_DEF`` / ``TILING_DATA_FIELD_DEF``
* arch35 regbase: plain ``class`` / ``struct`` with typed members (+ get_/set_)

Writers come from HostIR ``WriteEvent`` path tails. Readers are a text scan of
kernel sources for ``get_<field>()``, ``->field``, ``.field``. Nothing here
guesses: unmatched names stay unmatched and show up as ``no_writer`` /
``no_reader`` defects in the view.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ascendc_codemap_mcp.engine.ids import named_id, rel_posix

# ---------------------------------------------------------------------------
# Regex inventory
# ---------------------------------------------------------------------------

MACRO_BEGIN_RE = re.compile(
    r"\bBEGIN_TILING_DATA_DEF\s*\(\s*([A-Za-z_]\w*)\s*\)"
)
MACRO_FIELD_RE = re.compile(
    r"\bTILING_DATA_FIELD_DEF(?:_ARR|_STRUCT|_STRUCT_ARR)?\s*\(\s*"
    r"([^,]+?)\s*,\s*([A-Za-z_]\w*)"
)
MACRO_END_RE = re.compile(r"\bEND_TILING_DATA_DEF\b")

# `class Foo {` / `struct Foo {` — keep going until brace depth returns to 0.
CLASS_RE = re.compile(
    r"\b(?:class|struct)\s+([A-Za-z_]\w*)\s*(?::[^{]*)?\{",
    re.MULTILINE,
)

# Member: `int64_t s1;` / `uint32_t layout = 0;` / `int64_t n{0};`
# Skip access-specifiers, methods, nested types, using-aliases.
MEMBER_RE = re.compile(
    r"^\s*"
    r"((?:(?:unsigned|signed|const|volatile|static|mutable|constexpr)\s+)*"
    r"(?:u?int(?:8|16|32|64)_t|size_t|float|double|bool|"
    r"unsigned(?:\s+int|\s+long|\s+char)?|int|long|char|short|"
    r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*))"
    r"\s+"
    r"([A-Za-z_]\w*)"
    r"(?:\s*(?:=\s*([^;]+)|\{[^;]*\}))?"
    r"\s*;"
    r"\s*$",
    re.MULTILINE,
)

METHOD_OR_NESTED_RE = re.compile(
    r"^\s*(?:(?:inline|virtual|explicit|constexpr|static|friend)\s+)*"
    r"(?:void|[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s+[A-Za-z_]\w*\s*\(",
    re.MULTILINE,
)

CONSTEXPR_RE = re.compile(
    r"\b(?:static\s+)?constexpr\s+(?:static\s+)?"
    r"(?:const\s+)?"
    r"(?:u?int(?:8|16|32|64)_t|size_t|unsigned(?:\s+int)?|int|float|double|bool)\s+"
    r"([A-Za-z_]\w*)\s*=\s*([^;]+)\s*;"
)

DEFINE_INT_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(-?\d+[uUlL]?)\s*(?://.*)?$",
    re.MULTILINE,
)

ACCESSOR_GET_RE = re.compile(r"\bget_([A-Za-z_]\w*)\s*\(")
ACCESSOR_SET_RE = re.compile(r"\bset_([A-Za-z_]\w*)\s*\(")

# Pointer / member access of a known field. The field list is injected later.
ARROW_FIELD_TMPL = r"(?:->|\.)({fields})\b"


@dataclass
class TilingField:
    name: str
    ctype: str
    default: str | None = None
    struct: str = ""
    form: str = ""  # regbase_class | macro_def
    file: str = ""
    line: int = 0

    @property
    def qualified(self) -> str:
        return f"{self.struct}.{self.name}" if self.struct else self.name

    def node_id(self) -> str:
        return named_id("TilingDataField", self.qualified)


@dataclass
class FieldSite:
    file: str
    line: int
    form: str = ""  # assign | accessor | member
    function: str = ""
    expr: str = ""
    guard: str = ""
    path: str = ""
    via: str = ""


@dataclass
class TilingStruct:
    name: str
    form: str
    file: str
    line: int
    fields: list[TilingField] = field(default_factory=list)
    candidate_score: int = 0


@dataclass
class NamedConstant:
    name: str
    value: str
    kind: str  # constexpr | define | enum | platform
    file: str = ""
    line: int = 0


@dataclass
class TilingDataIR:
    structs: list[TilingStruct] = field(default_factory=list)
    constants: list[NamedConstant] = field(default_factory=list)
    writers: dict[str, list[FieldSite]] = field(default_factory=dict)  # field -> sites
    readers: dict[str, list[FieldSite]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def all_fields(self) -> list[TilingField]:
        out: list[TilingField] = []
        for st in self.structs:
            out.extend(st.fields)
        return out

    def field_names(self) -> set[str]:
        return {f.name for f in self.all_fields()}

    def by_name(self, name: str) -> list[TilingField]:
        return [f for f in self.all_fields() if f.name == name]

    def to_view(
        self,
        *,
        graph_fingerprint: str = "",
        writer_edges: list[dict[str, Any]] | None = None,
        reader_edges: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Projection consumed by AI / TG: structs → writers → readers → defects."""
        del writer_edges, reader_edges  # reserved; data already in writers/readers
        structs_out: list[dict[str, Any]] = []
        no_writer: list[str] = []
        no_reader: list[str] = []
        for st in self.structs:
            fields_out: list[dict[str, Any]] = []
            for f in st.fields:
                w = list(self.writers.get(f.name, []))
                # Prefer writers that mention the owning struct in the path.
                w_struct = [
                    s
                    for s in w
                    if st.name and st.name.lower() in (s.path or "").lower()
                ] or w
                r = list(self.readers.get(f.name, []))
                defect = None
                if r and not w_struct:
                    defect = "no_writer"
                    no_writer.append(f.qualified)
                elif w_struct and not r:
                    defect = "no_reader"
                    no_reader.append(f.qualified)
                fields_out.append(
                    {
                        "id": f.node_id(),
                        "name": f.name,
                        "type": f.ctype,
                        "default": f.default,
                        "writers": [_site_dict(s) for s in w_struct[:40]],
                        "readers": [_site_dict(s) for s in r[:40]],
                        "closure": {
                            "declared": max(len(w_struct), 1 if w_struct or r else 0),
                            "writer_count": len(w_struct),
                            "reader_count": len(r),
                            "status": "open",
                            "defect": defect,
                        },
                    }
                )
            structs_out.append(
                {
                    "name": st.name,
                    "form": st.form,
                    "source": {"file": st.file, "line": st.line},
                    "fields": fields_out,
                }
            )
        return {
            "schema": "uo-view-tilingdata/v1",
            "version": 1,
            "status": "extracted" if structs_out else "not_extracted",
            "graph_fingerprint": graph_fingerprint,
            "structs": structs_out,
            "constants": [
                {
                    "name": c.name,
                    "value": c.value,
                    "kind": c.kind,
                    "source": {"file": c.file, "line": c.line},
                }
                for c in self.constants
            ],
            "defects": {
                "no_writer": sorted(set(no_writer)),
                "no_reader": sorted(set(no_reader)),
            },
            "notes": list(self.notes),
        }


def _site_dict(s: FieldSite) -> dict[str, Any]:
    out: dict[str, Any] = {
        "file": s.file,
        "line": s.line,
        "form": s.form,
    }
    if s.function:
        out["function"] = s.function
    if s.expr:
        out["expr"] = s.expr
    if s.guard:
        out["guard"] = s.guard
    if s.path:
        out["path"] = s.path
    if s.via:
        out["via"] = s.via
    return out


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _strip_comments(text: str) -> str:
    """Remove // and /* */ comments without disturbing string contents much."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            if j < 0:
                break
            out.append("\n")
            i = j + 1
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                break
            chunk = text[i : j + 2]
            out.append("\n" * chunk.count("\n"))
            i = j + 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def parse_macro_structs(text: str, *, file: str = "") -> list[TilingStruct]:
    clean = _strip_comments(text)
    structs: list[TilingStruct] = []
    current: TilingStruct | None = None
    for line_no, line in enumerate(clean.splitlines(), 1):
        m_begin = MACRO_BEGIN_RE.search(line)
        if m_begin:
            current = TilingStruct(
                name=m_begin.group(1),
                form="macro_def",
                file=file,
                line=line_no,
            )
            continue
        if current is None:
            continue
        if MACRO_END_RE.search(line):
            if current.fields:
                structs.append(current)
            current = None
            continue
        m_field = MACRO_FIELD_RE.search(line)
        if m_field:
            current.fields.append(
                TilingField(
                    name=m_field.group(2),
                    ctype=m_field.group(1).strip(),
                    struct=current.name,
                    form="macro_def",
                    file=file,
                    line=line_no,
                )
            )
    return structs


def parse_class_structs(text: str, *, file: str = "") -> list[TilingStruct]:
    """Extract public data members from class/struct bodies.

    Methods (anything with ``(`` after the name) and nested class/struct
    bodies are skipped. Nested types are still discovered by a second CLASS_RE
    pass over the whole file — nesting is flattened into sibling structs.
    """
    clean = _strip_comments(text)
    structs: list[TilingStruct] = []
    for m in CLASS_RE.finditer(clean):
        name = m.group(1)
        # Skip obvious non-tiling helpers.
        if name.endswith("Helper") or name.endswith("Utils"):
            continue
        body_start = m.end()
        depth = 1
        i = body_start
        while i < len(clean) and depth:
            ch = clean[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        body = clean[body_start : i - 1]
        # Drop nested class/struct bodies so their members are not attributed here.
        body = _drop_nested_types(body)
        fields: list[TilingField] = []
        for fm in MEMBER_RE.finditer(body):
            ctype, fname, default = fm.group(1).strip(), fm.group(2), fm.group(3)
            if fname in {"public", "private", "protected"}:
                continue
            # Skip if this lookalike is a method return type line that MEMBER_RE
            # somehow matched — methods should fail because of `(`.
            line_text = body[fm.start() : body.find("\n", fm.start())]
            if "(" in line_text:
                continue
            fields.append(
                TilingField(
                    name=fname,
                    ctype=ctype,
                    default=(default.strip() if default else None),
                    struct=name,
                    form="regbase_class",
                    file=file,
                    line=_line_of(clean, m.start() + fm.start()),
                )
            )
        if not fields:
            continue
        structs.append(
            TilingStruct(
                name=name,
                form="regbase_class",
                file=file,
                line=_line_of(clean, m.start()),
                fields=fields,
                candidate_score=_tiling_struct_candidate_score(name, file, fields),
            )
        )
    return structs


def _tiling_struct_candidate_score(name: str, file: str, fields: list[TilingField]) -> int:
    """Name/path hints only. Never used to drop a parsed struct."""
    score = 0
    if "Tiling" in name or "Params" in name:
        score += 2
    path = file.replace("\\", "/").lower()
    if "tiling" in path:
        score += 2
    if any(f.name in {"b", "s1", "s2", "coreNum", "batch"} for f in fields):
        score += 1
    return score


def _drop_nested_types(body: str) -> str:
    """Replace nested class/struct bodies with whitespace (preserve newlines)."""
    out: list[str] = []
    i = 0
    while i < len(body):
        m = CLASS_RE.match(body, i)
        if m is None:
            # Also catch `class Foo` mid-line.
            m2 = re.search(r"\b(?:class|struct)\s+[A-Za-z_]\w*\s*(?::[^{]*)?\{", body[i:])
            if m2 is None:
                out.append(body[i:])
                break
            abs_start = i + m2.start()
            out.append(body[i:abs_start])
            brace = i + m2.end()
            depth = 1
            j = brace
            while j < len(body) and depth:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            out.append("\n" * body[abs_start:j].count("\n"))
            i = j
            continue
        # CLASS_RE.match from i — uncommon; fall through
        out.append(body[i])
        i += 1
    return "".join(out)


def parse_constants(text: str, *, file: str = "") -> list[NamedConstant]:
    clean = _strip_comments(text)
    out: list[NamedConstant] = []
    for m in CONSTEXPR_RE.finditer(clean):
        out.append(
            NamedConstant(
                name=m.group(1),
                value=m.group(2).strip(),
                kind="constexpr",
                file=file,
                line=_line_of(clean, m.start()),
            )
        )
    for m in DEFINE_INT_RE.finditer(text):  # keep raw for #define line numbers
        out.append(
            NamedConstant(
                name=m.group(1),
                value=m.group(2).rstrip("uUlL"),
                kind="define",
                file=file,
                line=_line_of(text, m.start()),
            )
        )
    return out


def parse_tiling_data_file(path: str | Path, *, op_root: str = "") -> TilingDataIR:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = rel_posix(str(path), op_root)
    macros = parse_macro_structs(text, file=rel)
    classes = parse_class_structs(text, file=rel)
    # Prefer macros when both fire on the same file (macro expansion noise).
    structs = macros or classes
    constants = parse_constants(text, file=rel)
    ir = TilingDataIR(structs=structs, constants=constants)
    if not structs:
        ir.notes.append(f"no_tiling_structs: {rel}")
    return ir


def parse_tiling_data_paths(
    paths: Iterable[str | Path], *, op_root: str = ""
) -> TilingDataIR:
    merged = TilingDataIR()
    seen_struct: set[str] = set()
    seen_const: set[str] = set()
    for p in paths:
        if not p or not Path(p).is_file():
            continue
        part = parse_tiling_data_file(p, op_root=op_root)
        for st in part.structs:
            key = f"{st.form}:{st.name}"
            if key in seen_struct:
                continue
            seen_struct.add(key)
            merged.structs.append(st)
        for c in part.constants:
            if c.name in seen_const:
                continue
            seen_const.add(c.name)
            merged.constants.append(c)
        merged.notes.extend(part.notes)
    return merged


def join_host_writers(ir: TilingDataIR, host_ir, *, op_root: str = "") -> None:
    """Attach HostIR write events whose path tail matches a declared field."""
    names = ir.field_names()
    if not names or host_ir is None:
        return
    writes = list(getattr(host_ir, "writes", []) or [])
    writes += list(getattr(host_ir, "local_writes", []) or [])
    expand = getattr(host_ir, "expand_callee_writers", None)
    if callable(expand):
        try:
            writes = list(expand())
        except Exception:  # noqa: BLE001
            pass
    for w in writes:
        path = str(getattr(w, "path", "") or "")
        if not path:
            continue
        # `this.fBaseParams.s1` / `tiling->s1` / `s1`
        tail = path.replace("->", ".").rstrip(".").split(".")[-1]
        if tail not in names:
            continue
        guards = []
        if hasattr(w, "guards"):
            try:
                guards = list(w.guards() or [])
            except Exception:  # noqa: BLE001
                guards = []
        ir.writers.setdefault(tail, []).append(
            FieldSite(
                file=rel_posix(str(getattr(w, "file", "") or ""), op_root),
                line=int(getattr(w, "line", 0) or 0),
                form=str(getattr(w, "kind", "") or "assign"),
                function=str(getattr(w, "function", "") or ""),
                expr=str(getattr(w, "rhs", "") or "")[:200],
                guard="; ".join(guards)[:300],
                path=path,
                via=str(getattr(w, "via", "") or ""),
            )
        )


def scan_kernel_readers(
    ir: TilingDataIR,
    kernel_files: Iterable[str | Path],
    *,
    op_root: str = "",
    max_hits_per_field: int = 40,
) -> None:
    """Text-scan kernel sources for field reads (accessor / member / arrow)."""
    names = sorted(ir.field_names())
    if not names:
        return
    # Longest-first so `singleCoreDqNum` wins over `Num`.
    names_sorted = sorted(names, key=len, reverse=True)
    field_alt = "|".join(re.escape(n) for n in names_sorted)
    member_re = re.compile(rf"(?:->|\.)({field_alt})\b")
    get_re = re.compile(rf"\bget_({field_alt})\s*\(")

    for path in kernel_files:
        p = Path(path)
        if not p.is_file():
            continue
        # Skip the tiling-data header itself — its get_/set_ are declarations.
        if "tiling_data" in p.name.lower():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = rel_posix(str(p), op_root)
        for m in get_re.finditer(text):
            name = m.group(1)
            bucket = ir.readers.setdefault(name, [])
            if len(bucket) >= max_hits_per_field:
                continue
            bucket.append(
                FieldSite(
                    file=rel,
                    line=_line_of(text, m.start()),
                    form="accessor",
                    expr=m.group(0)[:80],
                )
            )
        for m in member_re.finditer(text):
            name = m.group(1)
            bucket = ir.readers.setdefault(name, [])
            if len(bucket) >= max_hits_per_field:
                continue
            # Skip set_ LHS on same token patterns already covered by accessor.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.start())
            line = text[line_start : line_end if line_end >= 0 else len(text)]
            if re.search(rf"\bset_{name}\s*\(", line):
                continue
            bucket.append(
                FieldSite(
                    file=rel,
                    line=_line_of(text, m.start()),
                    form="member",
                    expr=line.strip()[:120],
                )
            )


def scan_host_setters(
    ir: TilingDataIR,
    host_files: Iterable[str | Path],
    *,
    op_root: str = "",
    max_hits_per_field: int = 40,
) -> None:
    """Text-scan host sources for ``set_<field>(`` when HostIR writers are thin.

    Prefer clang ``WriteEvent`` joins; this pass only fills fields that still
    have zero writers so AI can still land on a source line.
    """
    names = ir.field_names()
    if not names:
        return
    names_sorted = sorted(names, key=len, reverse=True)
    field_alt = "|".join(re.escape(n) for n in names_sorted)
    set_re = re.compile(rf"\bset_({field_alt})\s*\(")
    # `xxx.s1 =` / `xxx->s1 =` — require a real assignment, not `==` / `!=` / `<=`.
    assign_re = re.compile(rf"(?:->|\.)({field_alt})\s*=(?!=)")

    for path in host_files:
        p = Path(path)
        if not p.is_file():
            continue
        if "tiling_data" in p.name.lower():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = rel_posix(str(p), op_root)
        for m in set_re.finditer(text):
            name = m.group(1)
            bucket = ir.writers.setdefault(name, [])
            # Clang WriteEvents already cover this field — do not dilute them.
            if any(s.form not in {"text_setter", "text_assign"} for s in bucket):
                continue
            if len(bucket) >= max_hits_per_field:
                continue
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.start())
            line = text[line_start : line_end if line_end >= 0 else len(text)]
            stripped = line.strip()
            # Skip pure declarations: `void set_s1(int64_t);`
            if (
                stripped.endswith(";")
                and "{" not in stripped
                and re.match(r"(?:inline\s+)?(?:void|[A-Za-z_]\w*)\s+set_", stripped)
            ):
                continue
            bucket.append(
                FieldSite(
                    file=rel,
                    line=_line_of(text, m.start()),
                    form="text_setter",
                    expr=stripped[:120],
                )
            )
        for m in assign_re.finditer(text):
            name = m.group(1)
            bucket = ir.writers.setdefault(name, [])
            if any(s.form not in {"text_setter", "text_assign"} for s in bucket):
                continue
            if len(bucket) >= max_hits_per_field:
                continue
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.start())
            line = text[line_start : line_end if line_end >= 0 else len(text)]
            if "get_" in line:
                continue
            bucket.append(
                FieldSite(
                    file=rel,
                    line=_line_of(text, m.start()),
                    form="text_assign",
                    expr=line.strip()[:120],
                )
            )


def discover_tiling_data_headers(spec) -> list[Path]:
    """All tiling-data headers for this op (arch-scoped first, then shared)."""
    from ascendc_codemap_mcp.engine.source_layout import selected_tiling_headers

    hits: list[Path] = []
    seen: set[Path] = set()
    primary = getattr(spec, "tiling_data_header", None)
    if primary and Path(primary).is_file():
        hits.append(Path(primary))
        seen.add(Path(primary).resolve())
    op_dir = getattr(spec, "op_dir", None)
    arch_dir = getattr(spec, "arch_dir", None) or ""
    if op_dir:
        for p in selected_tiling_headers(Path(op_dir), arch_dir):
            key = p.resolve()
            if key in seen:
                continue
            seen.add(key)
            hits.append(p)
    return hits


def discover_kernel_sources(spec) -> list[Path]:
    kernel_root = getattr(spec, "kernel_root", None)
    arch_dir = getattr(spec, "arch_dir", None) or ""
    if not kernel_root:
        return []
    roots = [Path(kernel_root) / arch_dir, Path(kernel_root)]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in {".h", ".hpp", ".cpp", ".cc", ".c"}:
                continue
            key = p.resolve()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def discover_host_sources(spec) -> list[Path]:
    host_root = getattr(spec, "host_root", None)
    arch_dir = getattr(spec, "arch_dir", None) or ""
    if not host_root:
        return []
    roots = [Path(host_root) / arch_dir, Path(host_root)]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in {".h", ".hpp", ".cpp", ".cc", ".c"}:
                continue
            key = p.resolve()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def build_tiling_data_ir(spec, host_ir=None, *, op_root: str = "") -> TilingDataIR:
    root = op_root or str(getattr(spec, "op_dir", "") or "")
    headers = discover_tiling_data_headers(spec)
    ir = parse_tiling_data_paths(headers, op_root=root)
    join_host_writers(ir, host_ir, op_root=root)
    # Fill gaps clang did not cover (or when host_ir was skipped).
    scan_host_setters(ir, discover_host_sources(spec), op_root=root)
    scan_kernel_readers(ir, discover_kernel_sources(spec), op_root=root)
    if not headers:
        ir.notes.append("tiling_data_header_not_found")
    return ir


def iter_unique_fields(ir: TilingDataIR) -> Iterator[TilingField]:
    seen: set[str] = set()
    for f in ir.all_fields():
        if f.qualified in seen:
            continue
        seen.add(f.qualified)
        yield f
