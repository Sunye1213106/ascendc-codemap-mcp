# -*- coding: utf-8 -*-
"""Compile-time gates that otherwise leave a kernel TU empty.

Packed-key kernels (and dtype-gated kernels) put the real instantiations
behind ``#if ORIG_DTYPE_*`` / ``#if TILING_KEY_VAR == IDENT``. Clang only
sees those bodies when the macros are defined. Discovery walks the quoted
include closure under the operator/ops tree, honoring ``#if`` conditions
that can be evaluated from architecture ``-D`` macros.

A single preferred dtype (``DT_FLOAT16``) is enough when every ``ORIG_DTYPE_*``
in a taken ``#if`` equals that dtype. When one ``#if`` assigns *different*
``DT_*`` values to different ``ORIG_DTYPE_*`` macros, a second walk with that
per-macro assignment is required — otherwise the preferred-dtype TU never
includes the TQue bodies (EnQue/DeQue) that live on the mixed-dtype path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

_MAX_FILES = 80
_MAX_BYTES = 2_000_000

_DIRECTIVE_RE = re.compile(
    r"^[ \t]*#[ \t]*(include|if|ifdef|ifndef|elif|else|endif)\b(.*)$",
    re.M,
)
_QUOTED_INCLUDE_RE = re.compile(r'^\s*"([^"]+)"')
_ORIG_DTYPE_RE = re.compile(r"\b(ORIG_DTYPE_[A-Z0-9_]+)\b")
_ORIG_EQ_RE = re.compile(r"\b(ORIG_DTYPE_[A-Z0-9_]+)\s*==\s*(DT_[A-Z0-9_]+)")
_TILING_KEY_EQ_RE = re.compile(r"\bTILING_KEY_VAR\s*==\s*([A-Za-z_]\w*|\d+)")
_DTYPE_ALIAS_RE = re.compile(r"\b((?:MM_)?DTYPE_[A-Z0-9_]+)\b")
_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_DEFINED_RE = re.compile(
    r"\bdefined\s*\(\s*([A-Za-z_]\w*)\s*\)|\bdefined\s+([A-Za-z_]\w*)"
)

_DT_TO_CPP = {
    "DT_FLOAT": "float",
    "DT_FLOAT16": "half",
    "DT_BF16": "bfloat16_t",
    "DT_INT8": "int8_t",
    "DT_INT16": "int16_t",
    "DT_INT32": "int32_t",
    "DT_INT64": "int64_t",
    "DT_UINT8": "uint8_t",
    "DT_UINT64": "uint64_t",
    "DT_BOOL": "bool",
    "DT_INT4": "int4b_t",
    "DT_HIFLOAT8": "hifloat8_t",
    "DT_FLOAT8_E4M3FN": "fp8_e4m3fn_t",
    "DT_FLOAT8_E5M2": "fp8_e5m2_t",
}
_ORIG_MACRO_RE = re.compile(r"^ORIG_DTYPE_[A-Z0-9_]+$")
_GE_DT_MACRO_RE = re.compile(r"^DT_[A-Z0-9_]+$")


@dataclass(frozen=True)
class KernelCompileGates:
    """Preprocessor macros that must be injected so one kernel path instantiates."""

    orig_dtypes: tuple[str, ...] = ()
    dtype_aliases: tuple[str, ...] = ()
    tiling_key_choices: tuple[tuple[tuple[tuple[str, str], ...], str], ...] = ()
    orig_equality_groups: tuple[tuple[tuple[str, str], ...], ...] = ()

    @property
    def needed(self) -> bool:
        return bool(self.orig_dtypes or self.dtype_aliases or self.tiling_key_choices)

    def pick_tiling_key_var(self, dtype_variant: str) -> str | None:
        """One packed-key ident/number compatible with ``dtype_variant``."""
        variant = str(dtype_variant or "")
        for reqs, ident in self.tiling_key_choices:
            if reqs and all(dt == variant for _, dt in reqs):
                return ident
        for reqs, ident in self.tiling_key_choices:
            if not reqs:
                return ident
        if self.tiling_key_choices:
            return self.tiling_key_choices[0][1]
        return None

    def pick_mixed_orig_assignment(self, dtype_variant: str) -> dict[str, str] | None:
        """One per-macro ``ORIG_DTYPE_*`` map that a uniform dtype cannot open.

        Prefers the conjunction that names the most distinct ORIG macros.
        """
        best: dict[str, str] | None = None
        for group in self.orig_equality_groups:
            collapsed = _collapse_orig_pairs(group)
            if len(collapsed) < 2 or len(set(collapsed.values())) < 2:
                continue
            if best is None or len(collapsed) > len(best):
                best = collapsed
        _ = dtype_variant
        return best

    def clang_defines(
        self,
        dtype_variant: str,
        dt_enum_defines: Mapping[str, object] | None = None,
        *,
        orig_assignment: Mapping[str, str] | None = None,
    ) -> list[str]:
        if not self.needed:
            return []
        variant = str(dtype_variant or "")
        assignment = {str(k): str(v) for k, v in dict(orig_assignment or {}).items()}
        args: list[str] = []
        for name in self.orig_dtypes:
            args.append(f"-D{name}={assignment.get(name, variant)}")
        enum_defs = _dt_enum_for_clang(
            dt_enum_defines,
            assignment,
            variant,
            extra_dts=(dt for group in self.orig_equality_groups for _, dt in group),
            include_discovered=bool(assignment),
        )
        if self.orig_dtypes or self.tiling_key_choices:
            for name, val in enum_defs.items():
                args.append(f"-D{name}={val}")
        for alias in self.dtype_aliases:
            args.append(f"-D{alias}={_cpp_for_alias(alias, assignment, variant)}")
        key = self.pick_tiling_key_var(variant)
        if key is not None:
            args.append(f"-DTILING_KEY_VAR={key}")
        return args


def discover_kernel_gates(
    source_path: str | Path | None,
    *,
    op_dir: str | Path | None = None,
    ops_root: str | Path | None = None,
    macros: Mapping[str, str] | None = None,
) -> KernelCompileGates:
    if not source_path:
        return KernelCompileGates()
    path = Path(source_path)
    if not path.is_file():
        return KernelCompileGates()
    frozen = tuple(sorted((str(k), str(v)) for k, v in (macros or {}).items()))
    return _discover_cached(
        str(path),
        str(op_dir or ""),
        str(ops_root or ""),
        frozen,
    )


def source_uses_kernel_gates(
    source_path: str | Path | None,
    *,
    op_dir: str | Path | None = None,
    ops_root: str | Path | None = None,
    macros: Mapping[str, str] | None = None,
) -> bool:
    return discover_kernel_gates(
        source_path, op_dir=op_dir, ops_root=ops_root, macros=macros
    ).needed


@lru_cache(maxsize=256)
def _discover_cached(
    source_path: str,
    op_dir: str,
    ops_root: str,
    macro_items: tuple[tuple[str, str], ...],
) -> KernelCompileGates:
    macros = dict(macro_items)
    roots = _roots(source_path, op_dir, ops_root)
    orig: list[str] = []
    orig_seen: set[str] = set()
    aliases: list[str] = []
    alias_seen: set[str] = set()
    keys: list[tuple[tuple[tuple[str, str], ...], str]] = []
    key_seen: set[tuple[tuple[tuple[str, str], ...], str]] = set()
    eq_groups: list[tuple[tuple[str, str], ...]] = []
    eq_seen: set[tuple[tuple[str, str], ...]] = set()

    pending = [Path(source_path)]
    seen_files: set[str] = set()
    while pending and len(seen_files) < _MAX_FILES:
        current = pending.pop(0)
        try:
            resolved = current.resolve()
        except OSError:
            continue
        key = str(resolved).replace("\\", "/").lower()
        if key in seen_files or not resolved.is_file():
            continue
        if not _under_any(resolved, roots):
            continue
        seen_files.add(key)
        try:
            if resolved.stat().st_size > _MAX_BYTES:
                continue
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for name in _ORIG_DTYPE_RE.findall(text):
            if name not in orig_seen:
                orig_seen.add(name)
                orig.append(name)
        for name in _DTYPE_ALIAS_RE.findall(text):
            if name.startswith("ORIG_"):
                continue
            if name not in alias_seen:
                alias_seen.add(name)
                aliases.append(name)

        for kind, taking, cond, rest in _iter_directives(text, macros):
            # Equalities live on the #if line, including branches whose helper
            # ``defined(FOO)`` is false until clang injects ORIG/DT macros.
            if kind in {"if", "elif"}:
                eqs = _orig_equalities(cond)
                if eqs and eqs not in eq_seen:
                    eq_seen.add(eqs)
                    eq_groups.append(eqs)
            if kind != "include" or not taking:
                continue
            inc = _quoted_include(rest)
            if not inc:
                continue
            nxt = _resolve_include(resolved, inc, roots)
            if nxt is not None:
                pending.append(nxt)

        for reqs, ident in _tiling_keys_with_constraints(text, macros):
            item = (reqs, ident)
            if item not in key_seen:
                key_seen.add(item)
                keys.append(item)

    return KernelCompileGates(
        orig_dtypes=tuple(orig),
        dtype_aliases=tuple(aliases),
        tiling_key_choices=tuple(keys),
        orig_equality_groups=tuple(eq_groups),
    )


def _roots(source_path: str, op_dir: str, ops_root: str) -> list[Path]:
    out: list[Path] = []
    for raw in (ops_root, op_dir, str(Path(source_path).parent)):
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            out.append(p)
    return out


def _under_any(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def _quoted_include(rest: str) -> str | None:
    m = _QUOTED_INCLUDE_RE.match(rest.strip())
    return m.group(1) if m else None


def _resolve_include(current: Path, inc: str, roots: list[Path]) -> Path | None:
    candidates = [current.parent / inc]
    for root in roots:
        candidates.append(root / inc)
        candidates.append(root / "op_kernel" / inc)
    for cand in candidates:
        try:
            p = cand.resolve()
        except OSError:
            continue
        if p.is_file() and _under_any(p, roots):
            return p
    return None


def _orig_equalities(cond: str) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({(m.group(1), m.group(2)) for m in _ORIG_EQ_RE.finditer(cond)}))


def _strip_pp_comment(text: str) -> str:
    text = re.sub(r"//.*", "", text)
    return text.strip()


def _collapse_orig_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """First ``DT_*`` per ORIG macro. Source order is lost after sorting."""
    out: dict[str, str] = {}
    for name, dt in pairs:
        if name not in out:
            out[name] = dt
    return out


def _alias_orig_name(alias: str) -> str | None:
    core = alias[3:] if alias.startswith("MM_") else alias
    if core.startswith("DTYPE_"):
        return "ORIG_" + core
    return None


def _cpp_for_alias(
    alias: str, assignment: Mapping[str, str], dtype_variant: str
) -> str:
    orig = _alias_orig_name(alias)
    dt = assignment.get(orig, dtype_variant) if orig else dtype_variant
    return _DT_TO_CPP.get(str(dt), _DT_TO_CPP.get(str(dtype_variant), "half"))


def _dt_enum_for_clang(
    dt_enum_defines: Mapping[str, object] | None,
    assignment: Mapping[str, str],
    dtype_variant: str,
    *,
    extra_dts: Iterable[str] = (),
    include_discovered: bool = False,
) -> dict[str, object]:
    """Yaml DT enums, plus GE values needed to open mixed-ORIG ``defined(DT_*)``."""
    out = dict(dt_enum_defines or {})
    if not include_discovered:
        return out
    from ascendc_codemap_mcp.engine.variable_model import GE_DATA_TYPE

    needed = {str(dtype_variant)}
    needed.update(str(v) for v in assignment.values())
    needed.update(str(dt) for dt in extra_dts)
    for name in needed:
        if name in GE_DATA_TYPE and name not in out:
            out[name] = GE_DATA_TYPE[name]
    return out


def _is_injected_pp_macro(name: str) -> bool:
    """Macros clang_defines will inject; ``defined()`` must not close those #ifs."""
    return bool(_ORIG_MACRO_RE.fullmatch(name) or _GE_DT_MACRO_RE.fullmatch(name))


def try_eval_pp(expr: str, macros: Mapping[str, str]) -> bool | None:
    """Evaluate a restricted ``#if`` expression. ``None`` = unknown."""
    text = _strip_pp_comment(expr)
    if not text:
        return None

    def _defined(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(2)
        if name in macros or _is_injected_pp_macro(name):
            return "1"
        return "0"

    text = _DEFINED_RE.sub(_defined, text)

    def _ident(m: re.Match[str]) -> str:
        name = m.group(0)
        if name not in macros:
            return name
        val = macros[name]
        if val == "":
            return "1"
        if re.fullmatch(r"-?\d+", str(val)):
            return str(val)
        return "1"

    text = _IDENT_RE.sub(_ident, text)
    if re.search(r"[A-Za-z_]", text):
        return None
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)
    try:
        return bool(eval(text, {"__builtins__": {}}, {}))  # noqa: S307 — digits/ops only
    except Exception:  # noqa: BLE001
        return None


@dataclass
class _Frame:
    parent_on: bool
    branch_on: bool
    any_taken: bool
    any_unknown: bool
    equalities: tuple[tuple[str, str], ...] = ()

    @property
    def on(self) -> bool:
        return self.parent_on and self.branch_on


def _push_if(stack: list[_Frame], ev: bool | None, eqs: tuple[tuple[str, str], ...]) -> None:
    parent = stack[-1].on if stack else True
    if not parent:
        stack.append(_Frame(False, False, True, False, ()))
        return
    if ev is True:
        stack.append(_Frame(True, True, True, False, eqs))
    elif ev is False:
        stack.append(_Frame(True, False, False, False, ()))
    else:
        stack.append(_Frame(True, True, False, True, eqs))


def _apply_elif(fr: _Frame, ev: bool | None, eqs: tuple[tuple[str, str], ...]) -> None:
    if not fr.parent_on:
        fr.branch_on = False
        fr.equalities = ()
        return
    if fr.any_taken and not fr.any_unknown:
        fr.branch_on = False
        fr.equalities = ()
        return
    if ev is True:
        fr.branch_on = True
        fr.any_taken = True
        fr.any_unknown = False
        fr.equalities = eqs
    elif ev is False:
        # Previous unknown branch: still take this body so both sides are scanned.
        fr.branch_on = fr.any_unknown
        fr.equalities = eqs if fr.branch_on else ()
    else:
        fr.branch_on = True
        fr.any_unknown = True
        fr.equalities = eqs


def _apply_else(fr: _Frame) -> None:
    if not fr.parent_on:
        fr.branch_on = False
        fr.equalities = ()
        return
    if fr.any_taken and not fr.any_unknown:
        fr.branch_on = False
        fr.equalities = ()
        return
    fr.branch_on = True
    fr.equalities = ()


def _iter_directives(text: str, macros: Mapping[str, str]):
    """Yield ``(kind, taking, cond, rest)`` for include/if family directives.

    Kept for tests; tiling-key collection uses ``_tiling_keys_with_constraints``.
    """
    stack: list[_Frame] = []
    for m in _DIRECTIVE_RE.finditer(text):
        kind = m.group(1)
        rest = m.group(2) or ""
        cond = _strip_pp_comment(rest)
        taking = stack[-1].on if stack else True
        if kind == "include":
            yield kind, taking, cond, rest
            continue
        if kind == "if":
            _push_if(stack, try_eval_pp(cond, macros), _orig_equalities(cond))
            yield kind, stack[-1].on, cond, rest
        elif kind == "ifdef":
            name = cond.split()[0] if cond.split() else ""
            _push_if(stack, try_eval_pp(f"defined({name})", macros), ())
            yield kind, stack[-1].on, cond, rest
        elif kind == "ifndef":
            name = cond.split()[0] if cond.split() else ""
            _push_if(stack, try_eval_pp(f"!defined({name})", macros), ())
            yield kind, stack[-1].on, cond, rest
        elif kind == "elif":
            if stack:
                _apply_elif(stack[-1], try_eval_pp(cond, macros), _orig_equalities(cond))
                yield kind, stack[-1].on, cond, rest
        elif kind == "else":
            if stack:
                _apply_else(stack[-1])
                yield kind, stack[-1].on, cond, rest
        elif kind == "endif":
            if stack:
                stack.pop()
            yield kind, (stack[-1].on if stack else True), cond, rest


def _stacked_equalities(stack: list[_Frame]) -> tuple[tuple[str, str], ...]:
    merged: dict[str, str] = {}
    for fr in stack:
        if not fr.on:
            continue
        for name, dt in fr.equalities:
            merged[name] = dt
    return tuple(sorted(merged.items()))


def _tiling_keys_with_constraints(
    text: str, macros: Mapping[str, str]
) -> list[tuple[tuple[tuple[str, str], ...], str]]:
    out: list[tuple[tuple[tuple[str, str], ...], str]] = []
    stack: list[_Frame] = []
    for m in _DIRECTIVE_RE.finditer(text):
        kind = m.group(1)
        rest = m.group(2) or ""
        cond = _strip_pp_comment(rest)
        if kind == "include":
            continue
        if kind == "if":
            _push_if(stack, try_eval_pp(cond, macros), _orig_equalities(cond))
        elif kind == "ifdef":
            name = cond.split()[0] if cond.split() else ""
            _push_if(stack, try_eval_pp(f"defined({name})", macros), ())
        elif kind == "ifndef":
            name = cond.split()[0] if cond.split() else ""
            _push_if(stack, try_eval_pp(f"!defined({name})", macros), ())
        elif kind == "elif":
            if stack:
                _apply_elif(stack[-1], try_eval_pp(cond, macros), _orig_equalities(cond))
        elif kind == "else":
            if stack:
                _apply_else(stack[-1])
        elif kind == "endif":
            if stack:
                stack.pop()
            continue
        if not (stack and stack[-1].on):
            continue
        if kind not in ("if", "elif"):
            continue
        reqs = _stacked_equalities(stack)
        for km in _TILING_KEY_EQ_RE.finditer(cond):
            out.append((reqs, km.group(1)))
    return out
