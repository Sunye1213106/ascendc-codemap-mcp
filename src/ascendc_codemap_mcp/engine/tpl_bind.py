# -*- coding: utf-8 -*-
"""Positional binding: TPL DECL ↔ host encode args ↔ kernel NTTPs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from ascendc_codemap_mcp.engine.tpl_dsl import TplDim, TplSchema, parse_file


@dataclass
class Binding:
    index: int
    decl: TplDim
    host_expr: str
    nttp_name: str
    nttp_type: str


@dataclass
class BindingResult:
    bindings: list[Binding]
    site: "EncodeSite | None" = None
    rejected_sites: list["EncodeSite"] = dataclass_field(default_factory=list)

    def check(self) -> None:
        names_decl = [b.decl.name for b in self.bindings]
        names_nttp = [b.nttp_name for b in self.bindings]
        if names_decl != names_nttp:
            raise ValueError(
                f"name mismatch decl={names_decl} nttp={names_nttp}"
            )

    @property
    def derived_count(self) -> int:
        """Bindings whose host expression is not a bare constant."""
        return sum(
            1 for b in self.bindings if not _LITERAL_ARG_RE.match(b.host_expr.strip())
        )

    def to_dict(self) -> dict:
        return {
            "site": self.site.to_dict() if self.site else None,
            "rejected_sites": [s.to_dict() for s in self.rejected_sites],
            "derived_count": self.derived_count,
            "bindings": [
                {
                    "index": b.index,
                    "dim": b.decl.name,
                    "host_expr": b.host_expr,
                    "nttp": b.nttp_name,
                    "nttp_type": b.nttp_type,
                }
                for b in self.bindings
            ],
        }


def parse_kernel_nttps(src: str, entry_name: str | None = None) -> list[tuple[str, str]]:
    """Parse `template <type Name, ...>` off the kernel entry definition.

    Without `entry_name` the entry is the templated `__global__ __aicore__`
    function in the TU, which is how every AscendC operator declares it.
    """
    src = re.sub(r"\\\r?\n", " ", src)
    if entry_name:
        pattern = (
            r"template\s*<([^>]+)>\s*(?:__\w+__\s*)*void\s+"
            + re.escape(entry_name)
            + r"\s*\("
        )
    else:
        pattern = r"template\s*<([^>]+)>\s*(?:__\w+__\s+)+void\s+\w+\s*\("
    m = re.search(pattern, src, re.DOTALL)
    if not m:
        raise ValueError(
            f"kernel entry template not found (entry_name={entry_name or '<auto>'})"
        )
    inner = m.group(1)
    params: list[tuple[str, str]] = []
    for part in inner.split(","):
        part = " ".join(part.split())
        if not part:
            continue
        toks = part.split()
        if len(toks) < 2:
            continue
        name = toks[-1]
        typ = " ".join(toks[:-1])
        params.append((typ, name))
    return params


ENCODE_MACROS = ("GET_TPL_TILING_KEY", "ASCENDC_TPL_SEL_PARAM")

_MACRO_RE = re.compile(r"\b(" + "|".join(ENCODE_MACROS) + r")\s*\(")
_LITERAL_ARG_RE = re.compile(
    r"^(?:-?\d+[uUlL]*|0[xX][0-9a-fA-F]+[uUlL]*|true|false"
    r"|[A-Z][A-Z0-9_]*"  # TILING_KEY_1 and friends
    r"|static_cast<[^>]+>\s*\(\s*(?:-?\d+|true|false|[A-Z][A-Z0-9_]*)\s*\))$"
)


def _same_path(a: str, b: str) -> bool:
    return a.replace("\\", "/").lower() == b.replace("\\", "/").lower()


def _name_tokens(name: str) -> set[str]:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)
    return {p.lower() for p in parts if len(p) >= 4}


def _enclosing_function(host_ir, file: str, line: int) -> str:
    near = [
        w
        for w in host_ir.local_writes
        if w.function and w.line <= line and _same_path(w.file, file)
    ]
    return max(near, key=lambda w: w.line).function if near else ""


def _infer_literal_site_selector(host_ir, site: "EncodeSite") -> str | None:
    """A bool predicate that decides whether a literal-only encode site runs.

    Literal-only sites live in early-return helpers (`RunEmptyTilingRegbase`).
    The caller of that helper typically gates the call on a bool predicate
    (`IsEmptyOutput`). Recover it by: site's enclosing function → its callers →
    bool-returning callees of those callers whose name shares a token with the
    enclosing function. Exactly one match is a selector; anything else is not.
    """
    fn = _enclosing_function(host_ir, site.file, site.line)
    if not fn:
        return None
    callers = [
        name
        for name, summary in host_ir.summaries.items()
        if any(callee == fn for callee, _ in (summary.calls or ()))
    ]
    preferred: list[str] = []
    fallback: list[str] = []
    fn_tok = _name_tokens(fn)
    for caller in callers:
        for callee, _args in host_ir.summaries[caller].calls or ():
            summary = host_ir.summaries.get(callee)
            if summary is None or not summary.returns:
                continue
            rets = {r.strip() for r in summary.returns}
            if not rets or not rets <= {"true", "false"}:
                continue
            if _name_tokens(callee) & fn_tok:
                preferred.append(callee)
            else:
                fallback.append(callee)
    chosen = list(dict.fromkeys(preferred)) or list(dict.fromkeys(fallback))
    if len(chosen) != 1:
        return None
    return f"{chosen[0]}(context)"


def merge_literal_encode_alts(binding: BindingResult, host_ir) -> BindingResult:
    """Fold rejected literal-only encode sites into dimensions they disagree on.

    The primary site may hard-code a dimension (FAG's main encode writes
    `IsEmptyTensor=0`), while a rejected literal-only site encodes a different
    fixed key on an early-return path (`IsEmptyTensor=1`). Leaving the primary
    alone makes that dimension look constant and silently drops the alternate
    key. When a selector predicate can be inferred, the host expression becomes
    `selector ? alt : primary`; otherwise a soft site-tag keeps both values
    visible as an over-approximation.
    """
    if not binding.rejected_sites:
        return binding
    new_bindings: list[Binding] = []
    for b in binding.bindings:
        primary = b.host_expr.strip()
        if not _LITERAL_ARG_RE.match(primary):
            new_bindings.append(b)
            continue
        alts: list[tuple[EncodeSite, str]] = []
        for site in binding.rejected_sites:
            if not site.literal_only or b.index >= len(site.args):
                continue
            alt = site.args[b.index].strip()
            if alt != primary and _LITERAL_ARG_RE.match(alt):
                alts.append((site, alt))
        if not alts:
            new_bindings.append(b)
            continue
        site, alt = alts[0]
        selector = _infer_literal_site_selector(host_ir, site)
        if selector is None:
            selector = f"__literal_encode_site_{site.line}"
        new_bindings.append(
            Binding(
                index=b.index,
                decl=b.decl,
                host_expr=f"(({selector}) ? ({alt}) : ({primary}))",
                nttp_name=b.nttp_name,
                nttp_type=b.nttp_type,
            )
        )
    return BindingResult(
        bindings=new_bindings,
        site=binding.site,
        rejected_sites=list(binding.rejected_sites),
    )


@dataclass
class EncodeSite:
    """One textual `GET_TPL_TILING_KEY(...)` call with its argument list."""

    file: str
    line: int
    macro: str
    args: list[str]

    @property
    def literal_only(self) -> bool:
        """A site whose every argument is a constant encodes one fixed key.

        FAG's empty-tensor early return is exactly this shape and has the same
        arity as the real site, so arity alone cannot tell them apart.
        """
        return bool(self.args) and all(
            _LITERAL_ARG_RE.match(a.strip()) for a in self.args
        )

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "macro": self.macro,
            "args": list(self.args),
            "literal_only": self.literal_only,
        }


def _match_paren(src: str, open_idx: int) -> int:
    """Index of the `)` closing the `(` at `open_idx`, or -1.

    Skips string and character literals so a `,` or `)` inside `"..."` cannot
    end the argument list early.
    """
    depth = 0
    i = open_idx
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in "\"'":
            quote = ch
            i += 1
            while i < n and src[i] != quote:
                i += 2 if src[i] == "\\" else 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_args(inner: str) -> list[str]:
    args: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in inner:
        if ch in "([{<":
            depth += 1 if ch != "<" else 0
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            args.append(" ".join("".join(buf).split()))
            buf = []
        else:
            buf.append(ch)
    if buf:
        args.append(" ".join("".join(buf).split()))
    return [a for a in args if a]


def find_encode_sites(src: str, *, file: str = "") -> list[EncodeSite]:
    """Every encode-macro call in `src`, with true parenthesis matching."""
    sites: list[EncodeSite] = []
    for m in _MACRO_RE.finditer(src):
        open_idx = m.end() - 1
        close_idx = _match_paren(src, open_idx)
        if close_idx < 0:
            continue
        inner = src[open_idx + 1 : close_idx]
        args = _split_args(inner)
        if m.group(1) == "ASCENDC_TPL_SEL_PARAM" and args:
            args = args[1:]  # first arg is the selection group name
        if not args:
            continue
        sites.append(
            EncodeSite(
                file=file,
                line=src[: m.start()].count("\n") + 1,
                macro=m.group(1),
                args=args,
            )
        )
    return sites


def select_encode_site(sites: list[EncodeSite], *, arity: int) -> EncodeSite:
    """Pick the site that actually derives the key from host state."""
    if not sites:
        raise ValueError("GET_TPL_TILING_KEY call not found")
    right_arity = [s for s in sites if len(s.args) == arity]
    if not right_arity:
        found = sorted({len(s.args) for s in sites})
        raise ValueError(
            f"no encode site with arity {arity} (found {found})"
        )
    derived = [s for s in right_arity if not s.literal_only]
    if not derived:
        raise ValueError(
            "every encode site is literal-only; the key is not host-derived at "
            + ", ".join(f"{s.file}:{s.line}" for s in right_arity[:4])
        )
    return sorted(derived, key=lambda s: (s.file, s.line))[0]


def collect_encode_sites(paths) -> list[EncodeSite]:
    out: list[EncodeSite] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        out.extend(
            find_encode_sites(
                path.read_text(encoding="utf-8", errors="replace"),
                file=path.as_posix(),
            )
        )
    return out


def parse_host_encode_args_text(src: str, *, arity: int | None = None) -> list[str]:
    """Argument list of the primary encode site in `src`."""
    sites = find_encode_sites(src)
    if arity is not None:
        return select_encode_site(sites, arity=arity).args
    derived = [s for s in sites if not s.literal_only]
    if not derived:
        raise ValueError("GET_TPL_TILING_KEY call not found")
    return sorted(derived, key=lambda s: s.line)[0].args


def bind(
    schema: TplSchema,
    host_args: list[str],
    nttps: list[tuple[str, str]],
    *,
    site: EncodeSite | None = None,
    rejected_sites: list[EncodeSite] | None = None,
) -> BindingResult:
    if not (len(schema.dims) == len(host_args) == len(nttps)):
        raise ValueError(
            f"arity mismatch decl={len(schema.dims)} "
            f"host={len(host_args)} nttp={len(nttps)}"
        )
    bindings = []
    for i, (dim, expr, (typ, name)) in enumerate(
        zip(schema.dims, host_args, nttps)
    ):
        bindings.append(
            Binding(index=i, decl=dim, host_expr=expr, nttp_name=name, nttp_type=typ)
        )
    res = BindingResult(
        bindings=bindings, site=site, rejected_sites=list(rejected_sites or [])
    )
    res.check()
    return res


def bind_sources(
    key_hdr: str | Path,
    host_cpp,
    apt_cpp: str | Path,
    *,
    entry_name: str | None = None,
) -> BindingResult:
    """Three-way bind for one operator: TPL DECL ↔ host encode args ↔ kernel NTTPs.

    ``host_cpp`` may be a single path or an iterable of candidate host sources;
    the encode site is chosen by arity against the DECL, preferring sites whose
    arguments are host-derived rather than constants.
    """
    schema = parse_file(key_hdr)
    hosts = [host_cpp] if isinstance(host_cpp, (str, Path)) else list(host_cpp)
    sites = collect_encode_sites(hosts)
    chosen = select_encode_site(sites, arity=len(schema.dims))
    rejected = [s for s in sites if s is not chosen]
    apt_src = Path(apt_cpp).read_text(encoding="utf-8", errors="replace")
    nttps = parse_kernel_nttps(apt_src, entry_name)
    return bind(schema, chosen.args, nttps, site=chosen, rejected_sites=rejected)


def bind_from_spec(spec, host_cpp=None) -> BindingResult:
    """Bind using an `op_spec.OpSpec` for the header and entry locations.

    Defaults to every host target of the spec so the encode site is picked on
    evidence rather than on which file happened to be passed in.
    """
    if spec.tiling_key_header is None or spec.kernel_entry is None:
        raise ValueError("op spec lacks tiling_key_header or kernel_entry")
    if host_cpp is None:
        host_cpp = [p for p in spec.host_targets if Path(p).is_file()]
    return bind_sources(
        spec.tiling_key_header,
        host_cpp,
        spec.kernel_entry,
        entry_name=spec.op_snake,
    )
