# -*- coding: utf-8 -*-
"""ASCENDC_TPL_* DSL textual parser (schema invisible in normal clang AST)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


BW_RE = re.compile(r"ASCENDC_TPL_(\d+)_BW")

_BOOL_TRUE = frozenset({"1", "true", "True", "TRUE", "yes", "Yes", "YES"})
_BOOL_FALSE = frozenset({"0", "false", "False", "FALSE", "no", "No", "NO"})


def canonicalize_bool_token(value: int | bool | str) -> str:
    """Store BOOL TPL values as ``0`` / ``1`` (query still aliases true/false)."""
    if value is True or value == 1:
        return "1"
    if value is False or value == 0:
        return "0"
    text = str(value).strip()
    if text in _BOOL_TRUE:
        return "1"
    if text in _BOOL_FALSE:
        return "0"
    return "1" if text.lower() in {"true", "1"} else "0"


def bool_value_aliases(value: int | bool | str) -> tuple[str, ...]:
    """Expand true/false/0/1 only. Other tokens stay themselves."""
    if value is True:
        return ("1", "true", "True", "TRUE")
    if value is False:
        return ("0", "false", "False", "FALSE")
    text = str(value).strip()
    if text in _BOOL_TRUE or text.lower() == "true":
        return ("1", "true", "True", "TRUE")
    if text in _BOOL_FALSE or text.lower() == "false":
        return ("0", "false", "False", "FALSE")
    return (text,)


def is_tiling_struct_sel(sel: dict) -> bool:
    """ARGS_SEL layout pick, not a TilingKey dim."""
    kind = str((sel or {}).get("kind") or "").upper()
    return kind in {"TILING_STRUCT", "TILING_STRUCT_SEL"} or bool((sel or {}).get("struct"))


def canonicalize_sel_vals(kind: str, vals: list[str]) -> list[str]:
    if str(kind).upper() != "BOOL":
        return [str(v) for v in vals]
    return [canonicalize_bool_token(v) for v in vals]


DECL_KIND_RE = re.compile(
    r"ASCENDC_TPL_(UINT|BOOL|DTYPE|FORMAT|KERNEL_TYPE)_DECL\s*\("
)
SEL_KIND_RE = re.compile(
    r"ASCENDC_TPL_(UINT|BOOL|DTYPE|FORMAT)_SEL\s*\(|"
    r"ASCENDC_TPL_TILING_STRUCT_SEL\s*\("
)


@dataclass
class TplDim:
    name: str
    kind: str
    bw: int
    vals: list[str]
    bit_lo: int = 0
    bit_hi: int = 0
    #: Original width token from ARGS_DECL, e.g. ``ASCENDC_TPL_4_BW``. Empty
    #: when the width was a literal or a BOOL/KERNEL_TYPE dim.
    bw_token: str = ""

    @property
    def value_domain(self) -> list[str]:
        """Values excluding UI_LIST/UI_RANGE marker at vals[0] for UINT."""
        if self.kind == "UINT" and self.vals:
            marker = self.vals[0]
            if "UI_LIST" in marker or "UI_RANGE" in marker:
                return self.vals[1:]
        return list(self.vals)


@dataclass
class TplSchema:
    op_tag: str
    dims: list[TplDim] = field(default_factory=list)
    selections: list[list[dict]] = field(default_factory=list)

    @property
    def total_bits(self) -> int:
        return sum(d.bw for d in self.dims)

    def encode_uint(self, dim: TplDim, value: int | str) -> int:
        """UINT encodes index in the declared value domain (UI_LIST marker stripped).

        Matches CANN ``FastEncodeTilingKeyDirect``: find value in domain, pack index.
        """
        if dim.kind != "UINT":
            raise ValueError(f"{dim.name} is not UINT")
        domain = dim.value_domain
        sval = str(value)
        try:
            return domain.index(sval)
        except ValueError as e:
            raise ValueError(f"{dim.name} value {value!r} not in {domain}") from e

    def encode_bool(self, value: int | bool | str) -> int:
        return 1 if canonicalize_bool_token(value) == "1" else 0

    def encode_tiling_key(self, inst: dict[str, str | int | bool]) -> int:
        """Pack one concrete ARGS_SEL instance into a uint64 tiling key."""
        key = 0
        shift = 0
        for dim in self.dims:
            raw = inst.get(dim.name)
            if raw is None:
                domain = dim.value_domain
                if not domain:
                    raise ValueError(f"missing value for {dim.name}")
                raw = domain[0]
            if dim.kind == "UINT":
                encode_val = self.encode_uint(dim, raw)
            elif dim.kind == "BOOL":
                encode_val = self.encode_bool(raw)
            else:
                encode_val = int(raw)
            mask = (1 << dim.bw) - 1
            key |= (encode_val & mask) << shift
            shift += dim.bw
        if shift > 64:
            raise ValueError(f"tiling key bits {shift} exceed 64")
        return key

    def decode_tiling_key(self, key: int) -> dict[str, str]:
        """Inverse of :meth:`encode_tiling_key` (best-effort for UINT/BOOL)."""
        out: dict[str, str] = {}
        shift = 0
        for dim in self.dims:
            mask = (1 << dim.bw) - 1
            encode_val = (int(key) >> shift) & mask
            shift += dim.bw
            if dim.kind == "UINT":
                domain = dim.value_domain
                out[dim.name] = domain[encode_val] if encode_val < len(domain) else str(encode_val)
            else:
                out[dim.name] = str(encode_val)
        return out


def encode_tiling_key(schema: TplSchema, inst: dict[str, str | int | bool]) -> int:
    return schema.encode_tiling_key(inst)


def schema_construct_macros(schema: TplSchema) -> frozenset[str]:
    """CANN TPL construct names implied by a parsed schema, not a source scan."""
    names = {"ASCENDC_TPL_ARGS_DECL", "GET_TPL_TILING_KEY"}
    has_sel = bool(schema.selections and any(schema.selections))
    if has_sel:
        names.add("ASCENDC_TPL_ARGS_SEL")
        names.add("ASCENDC_TPL_SEL")
    for dim in schema.dims:
        kind = str(dim.kind or "").upper()
        if kind:
            names.add(f"ASCENDC_TPL_{kind}_DECL")
            if has_sel and kind != "KERNEL_TYPE":
                names.add(f"ASCENDC_TPL_{kind}_SEL")
        token = str(dim.bw_token or "").strip()
        if token:
            names.add(token)
        if kind == "UINT" and dim.vals:
            marker = str(dim.vals[0])
            if "UI_LIST" in marker:
                names.add("ASCENDC_TPL_UI_LIST")
            if "UI_RANGE" in marker:
                names.add("ASCENDC_TPL_UI_RANGE")
    if has_sel:
        for group in schema.selections:
            if any(is_tiling_struct_sel(sel) for sel in group):
                names.add("ASCENDC_TPL_TILING_STRUCT_SEL")
                break
    return frozenset(names)


def _join_continuations(src: str) -> str:
    return re.sub(r"\\\r?\n", " ", src)


def strip_cpp_comments(src: str) -> str:
    """Drop ``//`` and ``/* */`` so TPL macro args are not swallowed by comments.

    Clustered DECL/SEL headers put a ``//`` note immediately after ``(``.
    Without this, the first argument becomes the comment plus the dim name
    and canonical TPL rebuild cannot match ARGS_SEL fields to TILING_KEY
    entities.
    """
    text = src or ""
    if "//" not in text and "/*" not in text:
        return text
    n = len(text)
    out: list[str] = []
    i = 0
    start = 0
    while i < n:
        ch = text[i]
        if ch in {'"', "'"}:
            i += 1
            while i < n:
                cur = text[i]
                if cur == "\\" and i + 1 < n:
                    i += 2
                    continue
                i += 1
                if cur == ch:
                    break
            continue
        if ch == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                out.append(text[start:i])
                i += 2
                while i < n and text[i] != "\n":
                    i += 1
                start = i
                continue
            if nxt == "*":
                out.append(text[start:i])
                end = text.find("*/", i + 2)
                if end < 0:
                    out.append(" ")
                    return "".join(out)
                out.append(" ")
                i = end + 2
                start = i
                continue
        i += 1
    out.append(text[start:])
    return "".join(out)


def _balanced_paren_body(src: str, open_paren_idx: int) -> str:
    """open_paren_idx points at '('; return inside excluding outer parens."""
    depth = 0
    for j in range(open_paren_idx, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[open_paren_idx + 1 : j]
    raise ValueError("unbalanced parenthesis")


def _split_args(inner: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in inner:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p != ""]


def _uint_bit_width(token: str) -> int:
    """Width from ``ASCENDC_TPL_N_BW``, a decimal literal, or a named macro.

    Missing / unknown tokens keep a conservative 8-bit width so extract does
    not crash; callers still see the dim.
    """
    raw = (token or "").strip()
    if not raw:
        return 8
    m = BW_RE.fullmatch(raw) or BW_RE.search(raw)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"\d+", raw):
        return int(raw)
    return 8


_QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
_DEFINE_LINE_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_]\w*)(?:\(([^)]*)\))?\s*(.*?)\s*$",
    re.M,
)
_GET_TPL_HINT = "GET_TPL_TILING_KEY"
_TPL_HINT = "ASCENDC_TPL_"


def cann_include_search_roots() -> list[Path]:
    try:
        from ascendc_codemap_mcp.engine.paths import cann_root, resolve_cann_relative
    except Exception:
        return []
    root = cann_root()
    if root is None:
        return []
    rels = (
        "cann-opbase/x86_64-linux/pkg_inc",
        "cann-opbase/x86_64-linux/pkg_inc/op_common",
        "cann-opbase/x86_64-linux/include",
        "cann-opbase/x86_64-linux/include/op_common",
        "cann-asc-devkit/x86_64-linux/asc/include",
        "cann-asc-devkit/x86_64-linux/ascendc/include/highlevel_api",
    )
    return [resolve_cann_relative(root, rel) for rel in rels if resolve_cann_relative(root, rel).is_dir()]


def collect_defines(text: str) -> dict[str, tuple[list[str] | None, str]]:
    """``#define NAME`` / ``#define NAME(a,...)`` after line-continuation join."""
    src = _join_continuations(text or "")
    out: dict[str, tuple[list[str] | None, str]] = {}
    for match in _DEFINE_LINE_RE.finditer(src):
        name, params, body = match.group(1), match.group(2), (match.group(3) or "").strip()
        if not body or body.startswith("#"):
            continue
        if params is None:
            out[name] = (None, body)
            continue
        args = [p.strip() for p in params.split(",") if p.strip()]
        out[name] = (args, body)
    return out


def _subst_macro(body: str, params: list[str], args: list[str]) -> str:
    named = list(params)
    va: list[str] = []
    if named and named[-1] in {"...", "__VA_ARGS__"}:
        named = named[:-1]
        va = args[len(named) :]
        body = body.replace("__VA_ARGS__", ", ".join(va))
    for param, value in zip(named, args):
        body = re.sub(rf"##\s*{re.escape(param)}\b", value, body)
        body = re.sub(rf"\b{re.escape(param)}\s*##", value, body)
        body = re.sub(rf"\b{re.escape(param)}\b", value, body)
    return body


def expand_interesting_macros(text: str, defines: dict[str, tuple[list[str] | None, str]]) -> str:
    """Inline macros whose body is a TPL / GET_TPL packing helper."""
    interesting = {
        name: spec
        for name, spec in defines.items()
        if name != "GET_TPL_TILING_KEY"
        and (_TPL_HINT in spec[1] or _GET_TPL_HINT in spec[1])
    }
    if not interesting:
        return text
    src = _join_continuations(text or "")
    for _ in range(64):
        changed = False
        for name, (params, body) in interesting.items():
            if params is None:
                nxt = re.sub(rf"\b{re.escape(name)}\b", body, src)
                if nxt != src:
                    src = nxt
                    changed = True
                continue
            if not params:
                nxt = re.sub(rf"\b{re.escape(name)}\s*\(\s*\)", body, src)
                if nxt != src:
                    src = nxt
                    changed = True
                continue
            # Drain every call this pass. One-at-a-time with a 24-step cap
            # left ARGS_SEL(helper(...)) wrappers unexpanded on large schemas.
            expansions = 0
            while expansions < 512:
                match = re.search(rf"\b{re.escape(name)}\s*\(", src)
                if not match:
                    break
                open_pos = src.find("(", match.start())
                try:
                    inner = _balanced_paren_body(src, open_pos)
                except ValueError:
                    break
                close = open_pos + 1 + len(inner)
                args = _split_args(inner)
                repl = _subst_macro(body, params, args)
                src = src[: match.start()] + repl + src[close + 1 :]
                changed = True
                expansions += 1
        if not changed:
            break
    return src


def load_quoted_include_texts(path: Path, *, extra_roots: list[Path] | None = None) -> list[str]:
    """Quoted includes from ``path`` (a few levels), plus CANN search roots."""
    parent = Path(path).parent
    try:
        from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

        src = read_text(path)
    except OSError:
        return []
    walk: list[Path] = []
    cur = parent.resolve()
    for _ in range(8):
        walk.append(cur)
        if (cur / "op_host").is_dir() or (cur / "op_kernel").is_dir():
            walk.append(cur.parent)
            walk.append(cur.parent.parent)
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    try:
        from ascendc_codemap_mcp.engine.paths import ops_root

        repo = ops_root()
        if repo is not None:
            walk.append(Path(repo))
    except Exception:
        pass
    roots = [parent, *walk, *(extra_roots or []), *cann_include_search_roots()]
    texts: list[str] = []
    seen: set[Path] = set()
    pending: list[tuple[str, Path]] = [(src, parent)]
    depth = 0
    while pending and depth < 4:
        nxt: list[tuple[str, Path]] = []
        for text, base in pending:
            for inc in _QUOTED_INCLUDE_RE.findall(text):
                low = inc.replace("\\", "/").lower()
                if not any(tok in low for tok in ("tiling", "tpl", "template_argument", "atvoss")):
                    continue
                search = [base, *roots]
                for root in search:
                    cand = (root / inc.replace("\\", "/")).resolve()
                    if not cand.is_file() or cand in seen:
                        continue
                    seen.add(cand)
                    try:
                        from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

                        body = read_text(cand)
                    except OSError:
                        break
                    texts.append(body)
                    nxt.append((body, cand.parent))
                    break
        pending = nxt
        depth += 1
    return texts


def expand_get_tpl_arg_macros(text: str, defines: dict[str, tuple[list[str] | None, str]]) -> str:
    """Expand object-like macros used as ``GET_TPL_TILING_KEY`` arguments.

    Clustered TPL schemas pass unused-dimension placeholders such as
    ``SET_NOT_USE_QUANT_MM_TILING`` → ``0UL, 0UL, 0``. Those bodies have no
    ``ASCENDC_TPL_`` / ``GET_TPL`` hint, so ``expand_interesting_macros``
    leaves them intact and arity never matches the ARGS_DECL.
    """
    src = _join_continuations(text or "")
    object_macros = {
        name: body
        for name, (params, body) in defines.items()
        if params is None and name != "GET_TPL_TILING_KEY" and body
    }
    if not object_macros:
        return src
    ident_re = re.compile(r"\b[A-Za-z_]\w*\b")
    for _ in range(16):
        changed = False
        for match in re.finditer(r"\bGET_TPL_TILING_KEY\s*\(", src):
            open_pos = src.find("(", match.start())
            try:
                inner = _balanced_paren_body(src, open_pos)
            except ValueError:
                continue

            def _repl(token: re.Match[str]) -> str:
                name = token.group(0)
                return object_macros.get(name, name)

            nxt = ident_re.sub(_repl, inner)
            if nxt == inner:
                continue
            close = open_pos + 1 + len(inner)
            src = src[: open_pos + 1] + nxt + src[close:]
            changed = True
            break
        if not changed:
            break
    return src


def expand_tpl_source(text: str, extra_texts: Iterable[str] | None = None) -> str:
    defines = collect_defines(text)
    for extra in extra_texts or ():
        defines.update(collect_defines(extra))
    expanded = expand_interesting_macros(text, defines)
    return expand_get_tpl_arg_macros(expanded, defines)


def parse_args_decl(src: str) -> TplSchema:
    src = _join_continuations(strip_cpp_comments(src))
    m = re.search(r"ASCENDC_TPL_ARGS_DECL\s*\(", src)
    if not m:
        return TplSchema(op_tag="")
    body = _balanced_paren_body(src, m.end() - 1)
    top = _split_args(body)
    op_tag = top[0] if top else ""
    # re-scan DECL macros inside body (after op tag)
    dims: list[TplDim] = []
    off = 0
    for dm in DECL_KIND_RE.finditer(body):
        kind = dm.group(1)
        inner = _balanced_paren_body(body, dm.end() - 1)
        parts = _split_args(inner)
        name = parts[0] if parts else ""
        if not name:
            continue
        if kind == "UINT":
            bw_token = parts[1] if len(parts) > 1 else ""
            bw = _uint_bit_width(bw_token)
            vals = parts[2:]
        elif kind == "BOOL":
            bw, vals, bw_token = 1, canonicalize_sel_vals("BOOL", parts[1:]), ""
        elif kind == "KERNEL_TYPE":
            # Host GET_TPL_TILING_KEY still passes this dim (cvMode / mix ratio).
            bw, vals, bw_token = 6, parts[1:], ""
        else:
            bw, vals, bw_token = 8, parts[1:], ""
        dims.append(TplDim(name=name, kind=kind, bw=bw, vals=vals, bw_token=bw_token))
    # assign bit ranges
    bit = 0
    for d in dims:
        d.bit_lo = bit
        d.bit_hi = bit + d.bw - 1
        bit += d.bw
    return TplSchema(op_tag=op_tag, dims=dims)


def parse_args_sel(src: str) -> list[list[dict]]:
    """Return list of ARGS_SEL groups; each group is list of {name,kind,vals}."""
    src = _join_continuations(strip_cpp_comments(src))
    groups: list[list[dict]] = []
    for m in re.finditer(r"ASCENDC_TPL_ARGS_SEL\s*\(", src):
        line = src.count("\n", 0, m.start()) + 1
        body = _balanced_paren_body(src, m.end() - 1)
        sels: list[dict] = []
        for sm in re.finditer(
            r"ASCENDC_TPL_(UINT|BOOL|DTYPE|FORMAT)_SEL\s*\(|"
            r"ASCENDC_TPL_TILING_STRUCT_SEL\s*\(",
            body,
        ):
            inner = _balanced_paren_body(body, sm.end() - 1)
            parts = _split_args(inner)
            token = sm.group(0)
            if "TILING_STRUCT_SEL" in token:
                struct = (parts[0].split("::")[-1] if parts else "").strip()
                if not struct:
                    continue
                sels.append(
                    {
                        "name": struct,
                        "kind": "TILING_STRUCT",
                        "vals": [struct],
                        "struct": struct,
                        "line": line,
                    }
                )
                continue
            kind = sm.group(1)
            sels.append(
                {
                    "name": parts[0],
                    "kind": kind,
                    "vals": canonicalize_sel_vals(kind, parts[1:]),
                    "line": line,
                }
            )
        if sels:
            groups.append(sels)
    return groups


def parse_tpl_corpus(paths: Iterable[str | Path]) -> TplSchema:
    """Parse ARGS_DECL + ARGS_SEL from one or more TPL headers.

    DECL and SEL often live in different files (``*_tiling_key_decl.h`` vs
    ``archNN/*_tiling_key.h``). Quoted includes that themselves contain TPL
    usages are merged; CANN ``template_argument.h`` is used only for macro
    expansion so its ``#define ASCENDC_TPL_ARGS_DECL`` is not parsed as a
    schema.
    """
    chunks: list[str] = []
    extras: list[str] = []
    seen: set[Path] = set()
    for raw in paths:
        header = Path(raw)
        try:
            key = header.resolve()
        except OSError:
            continue
        if key in seen or not header.is_file():
            continue
        seen.add(key)
        try:
            chunks.append(header.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        extras.extend(load_quoted_include_texts(header))
    tpl_extras = [
        body
        for body in extras
        if "ASCENDC_TPL_ARGS_DECL" in body or "ASCENDC_TPL_ARGS_SEL" in body
    ]
    corpus = expand_tpl_source("\n".join(chunks + tpl_extras), extras)
    schema = parse_args_decl(corpus)
    schema.selections = parse_args_sel(corpus)
    return schema


def parse_file(path: str | Path) -> TplSchema:
    return parse_tpl_corpus([path])


def expand_legal_instances(schema: TplSchema) -> list[dict[str, str]]:
    """Expand ARGS_SEL groups into concrete dim->value maps (cartesian per group)."""
    import itertools

    out: list[dict[str, str]] = []
    for group in schema.selections:
        axes: list[tuple[str, list[str]]] = []
        for sel in group:
            if is_tiling_struct_sel(sel):
                continue
            name = sel["name"]
            vals = sel["vals"]
            # UINT_SEL often: name, UI_LIST, v1, v2... or name, UI_RANGE, lo, hi
            if vals and ("UI_LIST" in vals[0] or "UI_RANGE" in vals[0]):
                domain = vals[1:]
            else:
                domain = vals
            axes.append((name, domain))
        if not axes:
            continue
        names = [a[0] for a in axes]
        for combo in itertools.product(*[a[1] for a in axes]):
            out.append(dict(zip(names, combo)))
    return out


def bit_comment_ranges(src: str) -> dict[str, tuple[int, int]]:
    """Parse `// bit: hi-lo` or `// bit: n` comments near DECL lines if present."""
    src = _join_continuations(src)
    found: dict[str, tuple[int, int]] = {}
    for m in re.finditer(
        r"ASCENDC_TPL_(?:UINT|BOOL|DTYPE|FORMAT)_DECL\s*\(\s*(\w+)[^)]*\)[^\n]*//\s*bit:\s*(\d+)(?:-(\d+))?",
        src,
    ):
        name = m.group(1)
        a, b = int(m.group(2)), m.group(3)
        if b is None:
            found[name] = (a, a)
        else:
            lo, hi = sorted((a, int(b)))
            found[name] = (lo, hi)
    return found
