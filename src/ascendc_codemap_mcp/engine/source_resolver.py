# -*- coding: utf-8 -*-
"""Resolve control-node conditions to root Sources (the closure engine).

Every atom in a guard expression is either mapped to a legal root Source or
kept as an explicit `Unknown` with a stable reason code. Field paths that are
host-computed state are chased one hop at a time through the Host IR SSA, so
`fBaseParams.isNzOut` resolves through its defining assignment rather than
being reported as an opaque symbol.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ascendc_codemap_mcp.engine.cpp_expr import parse_expr
from ascendc_codemap_mcp.engine.expr_ir import Bin, Call, Const, Expr, Ite, Ref, Select, Un, Unknown

# Root kinds accepted by the closure gates (mirrors docs §1.1).
LEGAL_ROOTS = {
    "INPUT_SHAPE",
    "INPUT_DTYPE",
    "INPUT_FORMAT",
    "INPUT_VALUE",
    "OPTIONAL_INPUT_PRESENCE",
    "ATTRIBUTE",
    "PLATFORM_ARCH",
    "PLATFORM_CORE_COUNT",
    "PLATFORM_MEMORY_SIZE",
    "PLATFORM_L2_SIZE",
    "PLATFORM_AIV_COUNT",
    "SESSION_OPTION",
    "COMPILE_INFO",
    "COMPILE_DEFINE",
    "TILING_KEY",
    "TILING_DATA",
    "TEMPLATE_LITERAL",
    "KERNEL_BUILTIN",
    "EXECUTION_ROLE",
    "LOOP_INDUCTION",
    "LOOP_DERIVED",
    "CONSTANT",
    "EXTERNAL",
}

REASON_UNMAPPED_CALL = "UNMAPPED_CALL"
REASON_UNMAPPED_SYMBOL = "UNMAPPED_SYMBOL"
REASON_PARSE_FAILED = "PARSE_FAILED"
REASON_NO_CONDITION = "NO_CONDITION_TEXT"
REASON_DEPTH_EXCEEDED = "DERIVATION_DEPTH_EXCEEDED"
REASON_RESOLVE_BUDGET = "RESOLVE_BUDGET"

# Per top-level resolve(): walk / helper-chase steps. Large merged helpers
# (same unqualified name across TUs) can otherwise loop for minutes.
RESOLVE_STEP_BUDGET = 2048
# Helpers bigger than this are performance/tiling methods, not one-hop wrappers.
HELPER_BODY_WEIGHT_LIMIT = 24

# Default chase depth — enough for deep helper/field chains without false DEPTH.
DEFAULT_MAX_DEPTH = 24

_SUBSCRIPT_RE = re.compile(r"\[[^\]]*\]")
_INPUT_ACCESSOR_RE = re.compile(
    r"\bGetOptionalInput\w*|\bGetInput\w*|\bGetData\b|\bGetValue\b|\bGetTensorData\b|\bGetInputPointer\b"
)
_EXPR_SPACE = (
    (re.compile(r"\s*\.\s*"), "."),
    (re.compile(r"\s*->\s*"), "->"),
    (re.compile(r"\s*::\s*"), "::"),
    # Close a bracket up against what it belongs to, and no further. Eating the
    # space on both sides turned `strcmp(a, b) == 0` into `strcmp(a, b)== 0`
    # and `x.size() > 0` into `x.size()>0`. These strings get matched against
    # source text and against each other, so the spacing has to survive.
    #
    # `<` and `>` are left alone: in a condition they are nearly always
    # comparisons, and nothing distinguishes `a < b` from `Cast < T >` by
    # spacing alone. Keeping the space is what agrees with the source.
    (re.compile(r"(?<=[\w\]>])\s+([(\[])"), r"\1"),
    (re.compile(r"([(\[])\s+"), r"\1"),
    (re.compile(r"\s+([)\]])"), r"\1"),
    (re.compile(r"\s*,\s*"), ", "),
    (re.compile(r"\s+"), " "),
)


def _strip_subscripts(path: str) -> str:
    """`foo.bar[i].baz` → `foo.bar.baz` for field-write matching."""
    return _SUBSCRIPT_RE.sub("", path)


def _norm_expr(text: str) -> str:
    """Defensive collapse of clang token spacing in stored expressions."""
    t = (text or "").strip()
    for pat, repl in _EXPR_SPACE:
        t = pat.sub(repl, t)
    return t.strip()

# --- call-name → root -------------------------------------------------------
CALL_ROOTS: list[tuple[str, str]] = [
    (r"^GetOptionalInput", "OPTIONAL_INPUT_PRESENCE"),
    (r"^GetShapeSize$|^GetStorageShape$|^GetOriginShape$|^GetDim$|^GetDimNum$", "INPUT_SHAPE"),
    (r"^GetInputShape$|^GetInputDesc$|^GetInputTensor$", "INPUT_SHAPE"),
    # output descriptors are shaped by the same inputs the kernel is given
    (r"^GetOutputShape$|^GetOutputDesc$|^GetOptionalOutputShape$", "INPUT_SHAPE"),
    (r"^GetDataType$|^GetOriginDataType$", "INPUT_DTYPE"),
    (r"^GetFormat$|^GetStorageFormat$|^GetOriginFormat$", "INPUT_FORMAT"),
    (r"^GetAttrPointer$|^GetAttrNum$|^GetAttrs$|^GetBool$|^GetInt$|^GetFloat$|^GetStr$", "ATTRIBUTE"),
    (r"^GetData$|^GetValue$", "INPUT_VALUE"),
    (r"^GetCoreNumAiv$|^GetCoreNumAic$|^GetCoreNum", "PLATFORM_CORE_COUNT"),
    (r"^GetUbSize$|^GetL1Size$|^GetL0[ABC]Size$", "PLATFORM_MEMORY_SIZE"),
    (r"^GetL2Size$|^GetLibApiWorkSpaceSize$|^GetWorkspaceSizes$|^GetL0CSize$", "PLATFORM_MEMORY_SIZE"),
    (r"^GetCurNpuArch$|^GetSocVersion$|^GetShortSocName$", "PLATFORM_ARCH"),
    (r"^GetDeterministic$", "SESSION_OPTION"),
    (r"^GetCompileInfo", "COMPILE_INFO"),
    (r"^GetBlockIdx$|^GetBlockNum$|^GetSubBlockIdx$", "KERNEL_BUILTIN"),
    (r"^GetTilingKey$", "TILING_KEY"),
    # remaining CANN tiling-context accessors
    (r"^GetOutputShape$|^GetOutputDesc$|^GetDimNum$|^GetShape$", "INPUT_SHAPE"),
    (r"^GetPlatformInfo$|^GetPlatformInfoPtr$|^GetAscendcPlatform$", "PLATFORM_ARCH"),
    (r"^GetInputPointer$|^GetTensorData$", "INPUT_VALUE"),
]

# --- bare identifier → root -------------------------------------------------
# Host-side context / tiling-context pointers are external framework state.
SYMBOL_ROOTS: list[tuple[str, str]] = [
    (r"^npuArch$|^NpuArch::", "PLATFORM_ARCH"),
    (r"^socVersion$|^SocVersion::", "PLATFORM_ARCH"),
    (r"^aivNum$|^aicNum$|^coreNum$|^blockDim$", "PLATFORM_CORE_COUNT"),
    (r"^ubSize$|^l1Size$|^l0CSize$", "PLATFORM_MEMORY_SIZE"),
    (r"^l2Size$|^l2CacheSize$|^l2_size$", "PLATFORM_L2_SIZE"),
    (r"^isDeterministic$", "SESSION_OPTION"),
    (r"^compileInfo", "COMPILE_INFO"),
    (r"^TILING_KEY_VAR$|^tilingKey$", "TILING_KEY"),
    (r"^block_idx$|^blockIdx$|^subBlockIdx$", "KERNEL_BUILTIN"),
    # host-side per-core / per-iteration indices are loop-derived, not inputs
    (r"^cBlockIdx$|^coreIdx$|^blockIndex$|^loopIdx$|^curIdx$", "LOOP_DERIVED"),
    # framework context handles — not operator-authored tiling state
    (r"^(?:this\.)?context_?$", "EXTERNAL"),
    (r"^gert::TilingContext\b|^TilingContext\b", "EXTERNAL"),
    # Any scoped name whose member is SCREAMING_CASE is an enum literal or a
    # named constant: SparseMode::BAND, ge::DT_FLOAT16, AttrIndex::HEAD_NUM.
    (r"^[A-Za-z_]\w*::[A-Z][A-Z0-9_]*$", "CONSTANT"),
    (r"^[A-Z][A-Z0-9_]{2,}$", "CONSTANT"),  # SCREAMING_CASE macro constants
    (r"^ge::|^std::numeric_limits", "CONSTANT"),
]

#: Fallback only, for a resolver built without a HostIR — see
#: `SourceResolver.tiling_derived`, which decides this structurally instead.
#: This pattern spells one operator's names and will misclassify any operator
#: that chose others.
_PARAMS_DERIVED_RE = re.compile(
    r"\b(?:\w*(?:Params|TilingData)|tilingData|deterPrefixData)\.",
)
#: An identifier immediately followed by a member access.
_DOTTED_HEAD_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\.")
_TUPLE_ACCESSORS = frozenset({"first", "second", "third", "data", "value", "get"})
_LOCAL_TRANSPARENT_ACCESSORS = _TUPLE_ACCESSORS | frozenset(
    {
        "size",
        "empty",
        "find",
        "count",
        "contains",
        "begin",
        "end",
        "cbegin",
        "cend",
        "front",
        "back",
        "at",
    }
)
_CONTAINER_TYPE_RE = re.compile(
    r"\b(?:std::)?(?:vector|deque|list|set|multiset|map|multimap|unordered_(?:set|map|multiset|multimap)|array|string)\b"
)
_LITERAL_OR_CONSTANT_RE = re.compile(
    r"^\s*(?:[-+]?\d+(?:\.\d+)?(?:[uUlLfF]*)?|true|false|nullptr|NULL|[A-Z][A-Z0-9_]*(?:::[A-Z][A-Z0-9_]*)?)\s*$"
)
_TUPLE_ELEM_RE = re.compile(r"\b(?:__tuple_elem|std::get|get)\s*(?:<|\()")


def _mentions_sym(var: str, rhs: str) -> bool:
    return bool(re.search(rf"\b{re.escape(var)}\b", rhs or ""))


_TUPLE_CALL_RE = re.compile(
    r"^(?:std::)?(?:make_tuple|tie|forward_as_tuple)\((.*)\)$", re.DOTALL
)


def _split_top_level_args(inner: str) -> list[str]:
    """Split `a, b, f(c, d)` on top-level commas."""
    args: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            piece = inner[start:i].strip()
            if piece:
                args.append(piece)
            start = i + 1
    tail = inner[start:].strip()
    if tail:
        args.append(tail)
    return args


def _nth_tuple_arg(text: str, index: int) -> str | None:
    """Pull the i-th argument out of `make_tuple(...)` / `tie(...)` text."""
    t = (text or "").strip()
    m = _TUPLE_CALL_RE.match(t)
    if not m:
        return None
    args = _split_top_level_args(m.group(1))
    if 0 <= index < len(args):
        return args[index]
    return None

# Host-computed state: not a root, must be chased through the Host IR.
DERIVED_FIELD_RE = re.compile(r"^(?:this\.)?\w*(?:BaseParams|Params|tilingData|TilingData)\w*\.")

# Pure helpers that carry no source of their own: resolve their arguments.
PASSTHROUGH_CALLS = {
    "strcmp",
    "strncmp",
    "memcmp",
    "strlen",
    "abs",
    "labs",
    "max",
    "min",
    "std::max",
    "std::min",
    "ceil",
    "floor",
    "CeilDiv",
    "AlignUp",
    # container queries and arithmetic helpers are transparent: whatever root
    # the argument has is the root of the result
    "size",
    "length",
    "empty",
    "begin",
    "end",
    "back",
    "front",
    "at",
    "data",
    "AlignTo",
    "AlignDown",
    "std::get",
    "std::tie",
    "std::make_tuple",
    "std::make_pair",
    "std::abs",
    "std::round",
    "static_cast",
    "find",
    "std::find",
    "std::find_if",
    "count",
    "std::count",
    "distance",
    "std::distance",
    "CeilDivideBy",
    "AbsCeil",
    "Gcd",
    "Lcm",
    "AlignDown",
    # functional-style casts: `int64_t(x)` carries the source of x
    "int",
    "uint",
    "bool",
    "float",
    "double",
    "size_t",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
}

REASON_FUNCTION_PARAMETER = "FUNCTION_PARAMETER"
REASON_TILING_DATA_NO_WRITER = "TILING_DATA_NO_WRITER"


def _aggregate_head(path: str) -> str:
    """`fBaseParams.foo.bar` -> `fBaseParams`; bare names have no head."""
    return path.split(".", 1)[0] if "." in path else ""


def _same_aggregate(candidate: str, wanted: str) -> bool:
    head = _aggregate_head(wanted)
    return bool(head) and _aggregate_head(candidate) == head


_GENERATED_ACCESSOR_RE = re.compile(r"^(?:set|Set|get|Get)_?(?P<field>\w+)$")


def _is_generated_accessor(function: str, tail: str) -> bool:
    """`set_keepProb(keepProb_val)` in a generated tiling-data header is not a
    derivation of `fBaseParams.keepProb`; it is the struct's own mutator."""
    m = _GENERATED_ACCESSOR_RE.match((function or "").split("::")[-1])
    if not m:
        return False
    return m.group("field").lower() == tail.lower()


def _unproven_field(path: str, reason: str) -> Atom:
    """A tiling-data field whose deciding write we could not locate.

    It keeps the `TILING_DATA` root because that is what it is, but is flagged
    `partial` so closure metrics do not count a dead end as an answer.
    """
    return Atom(text=path, root="TILING_DATA", symbol=path, reason=reason, partial=True)


@dataclass
class Atom:
    text: str
    root: str | None = None
    symbol: str = ""
    reason: str | None = None
    via: tuple[str, ...] = ()
    # Positional index carried by the accessor, e.g. the `2` in `GetDim(2)`.
    # Lets the variable layer name `VAR_SHAPE_QUERY_D2` instead of collapsing
    # every dimension of an input onto one variable.
    index: int | None = None
    # Which quantity of the shape was read, when it was read whole. A rank and
    # an element count are both "the shape without an axis" and used to share
    # one variable, so `GetDimNum() != 4` and `GetShapeSize() != 0` constrained
    # the same unknown -- and a premise stating the rank refused every input
    # whose element count was not 4.
    reads: str | None = None
    # A root was assigned by fallback rather than proven. `TILING_DATA` with no
    # locatable writer is the usual case: the field really is tiling state, but
    # nothing here shows what decides it, so it must not count as closed.
    partial: bool = False


def _constant_values(atoms: Iterable[Atom]) -> set[str]:
    """The distinct values a group of constant atoms folded to."""
    return {str(a.symbol) for a in atoms}


def _selects(e: Expr) -> bool:
    """Whether an expression picks one of several values rather than combining them."""
    if isinstance(e, (Ite, Select)):
        return True
    if isinstance(e, Un):
        return _selects(e.arg)
    if isinstance(e, Bin):
        return _selects(e.left) or _selects(e.right)
    if isinstance(e, Call):
        return any(_selects(a) for a in e.args)
    return False


def _picks_between_constants(res: Resolution) -> bool:
    """A resolution that is all-constant but still holds more than one value.

    A ternary over shapes is the usual shape of this: `resolve_value` drops the
    condition on purpose, so all that survives are the arms, and they are all
    constants. Calling the whole thing a constant keeps whichever arm was seen
    first and folds away the branches that would have produced the others --
    which narrows what the analysis believes possible, the direction that
    invents false "unreachable" answers. Arithmetic over constants is excluded:
    `kBlockSize * 2` mentions two values but only ever produces one.
    """
    return (
        res.expr is not None
        and _selects(res.expr)
        and len(_constant_values(res.atoms)) > 1
    )


def _literal_or_empty_ctor(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _LITERAL_OR_CONSTANT_RE.match(t):
        return True
    return t in {"{}", "[]"} or bool(
        re.match(r"^(?:std::)?\w+(?:<[^>]*>)?\s*\(\s*\)$", t)
    )


def _self_derived_local(var: str, candidates: Sequence[str]) -> bool:
    """Whether a local's value is a recurrence over its previous value.

    Only locals whose non-self definitions are literal/default values are
    classified this way.  A local seeded from an accessor (`aicNum =
    GetCoreNumAic(); aicNum = min(aicNum, ...)`) must still chase the accessor.
    """
    cleaned = [_norm_expr(c) for c in candidates if str(c or "").strip()]
    if not cleaned:
        return False
    self_refs = [c for c in cleaned if _mentions_sym(var, c)]
    if not self_refs:
        return False
    independent = [c for c in cleaned if not _mentions_sym(var, c)]
    if not independent:
        return True
    return all(_literal_or_empty_ctor(c) for c in independent)


def _multi_def_local_state(
    var: str, candidates: Sequence[str], local_names: set[str]
) -> bool:
    """State updated through other locals, e.g. binary-search left/right/mid."""
    cleaned = [_norm_expr(c) for c in candidates if str(c or "").strip()]
    if len(cleaned) < 2:
        return False
    if not any(_literal_or_empty_ctor(c) for c in cleaned):
        return False
    for rhs in cleaned:
        if _literal_or_empty_ctor(rhs):
            continue
        if any(name != var and _mentions_sym(name, rhs) for name in local_names):
            return True
    return False


def inferred_function_local_roots(host_ir: Any, function: str) -> dict[str, str]:
    """Infer cheap local roots from HostIR without solving loop semantics.

    The result is intentionally conservative: these locals are host-side
    loop/container state, not user-controllable inputs.  Marking them as
    LOOP_* removes fake unresolved-symbol blockers without inventing a source
    lemma.
    """
    if host_ir is None or not function:
        return {}
    cache = getattr(host_ir, "_inferred_function_local_roots", None)
    if cache is None:
        cache = {}
        setattr(host_ir, "_inferred_function_local_roots", cache)
    if function in cache:
        return dict(cache[function])
    roots: dict[str, str] = {}

    for node in getattr(host_ir, "controls", ()) or ():
        if getattr(node, "function", "") != function:
            continue
        for name in getattr(node, "induction_vars", ()) or ():
            if name:
                roots.setdefault(str(name), "LOOP_INDUCTION")

    try:
        defs = host_ir.defs_by_function().get(function, {})
    except Exception:
        defs = {}
    for var, candidates in defs.items():
        if _self_derived_local(str(var), candidates):
            roots.setdefault(str(var), "LOOP_DERIVED")

    local_names = {str(v) for v in defs}
    for var, candidates in defs.items():
        if str(var) in roots:
            continue
        if _multi_def_local_state(str(var), candidates, local_names):
            roots.setdefault(str(var), "LOOP_DERIVED")

    for var, candidates in defs.items():
        if str(var) in roots:
            continue
        if any(_TUPLE_ELEM_RE.search(_norm_expr(rhs)) for rhs in candidates):
            roots.setdefault(str(var), "LOOP_DERIVED")

    for decl in getattr(host_ir, "local_decls", ()) or ():
        if getattr(decl, "function", "") != function:
            continue
        if _CONTAINER_TYPE_RE.search(getattr(decl, "type_text", "") or ""):
            roots.setdefault(str(getattr(decl, "name", "") or ""), "LOOP_DERIVED")

    # Cheap fixed point: locals computed from loop/container locals are also
    # host-derived.  Keep the root non-steerable even when the RHS also reads
    # input/tiling data; the loop state is what prevents direct TG control.
    for _ in range(4):
        changed = False
        for var, candidates in defs.items():
            var = str(var)
            if var in roots:
                continue
            if any(
                _mentions_sym(src, rhs)
                for rhs in candidates
                for src in roots
                if src and src != var
            ):
                roots[var] = "LOOP_DERIVED"
                changed = True
        if not changed:
            break
    clean = {k: v for k, v in roots.items() if k}
    cache[function] = dict(clean)
    return clean


def inferred_parameter_roots(host_ir: Any, function: str) -> dict[str, str]:
    """Propagate caller-local LOOP_* roots through call-site formals.

    This covers same-name by-reference containers (`foo(v)` where `foo` also
    names the parameter `v`) without teaching the closed-vocabulary layer an
    invented input binding.
    """
    if host_ir is None or not function:
        return {}
    cache = getattr(host_ir, "_inferred_parameter_roots", None)
    if cache is None:
        cache = {}
        setattr(host_ir, "_inferred_parameter_roots", cache)
    if function in cache:
        return dict(cache[function])
    summary = getattr(host_ir, "summaries", {}).get(function)
    if summary is None:
        cache[function] = {}
        return {}
    candidates: dict[str, set[str]] = {p: set() for p in getattr(summary, "params", ()) or ()}
    call_edges: list[tuple[str, str, tuple[str, ...]]] = []
    for site in getattr(host_ir, "call_sites", ()) or ():
        call_edges.append(
            (
                getattr(site, "caller", "") or "",
                getattr(site, "callee", "") or "",
                tuple(getattr(site, "args", ()) or ()),
            )
        )
    if not call_edges:
        for caller_summary in getattr(host_ir, "summaries", {}).values():
            caller_name = getattr(caller_summary, "name", "") or ""
            for callee, args in getattr(caller_summary, "calls", ()) or ():
                call_edges.append((caller_name, callee, tuple(args or ())))

    for caller, callee, args in call_edges:
        if callee != function:
            continue
        caller_roots = inferred_function_local_roots(host_ir, caller)
        for i, actual in enumerate(args):
            if i >= len(summary.params):
                continue
            text = (actual or "").strip().lstrip("&*").strip()
            if re.fullmatch(r"[A-Za-z_]\w*", text or "") and text in caller_roots:
                candidates[summary.params[i]].add(caller_roots[text])
                continue
            if any(_mentions_sym(src, text) for src in caller_roots):
                candidates[summary.params[i]].add("LOOP_DERIVED")
    out: dict[str, str] = {}
    for param, roots in candidates.items():
        if not roots or not roots <= {"LOOP_INDUCTION", "LOOP_DERIVED"}:
            continue
        if roots == {"LOOP_INDUCTION"}:
            out[param] = "LOOP_INDUCTION"
        else:
            # Mixed induction/derived call sites are still non-steerable host
            # loop state.  Use the less precise root instead of rejecting.
            out[param] = "LOOP_DERIVED"
    cache[function] = dict(out)
    return out


@dataclass
class Resolution:
    condition: str
    atoms: list[Atom] = field(default_factory=list)
    expr: Expr | None = None

    @property
    def closed(self) -> bool:
        return bool(self.atoms) and all(
            a.root in LEGAL_ROOTS and not a.partial for a in self.atoms
        )

    @property
    def partial(self) -> bool:
        """Every atom landed on a root but at least one was a fallback."""
        return bool(self.atoms) and all(a.root in LEGAL_ROOTS for a in self.atoms) and any(
            a.partial for a in self.atoms
        )

    @property
    def roots(self) -> list[str]:
        seen: list[str] = []
        for a in self.atoms:
            if a.root and a.root not in seen:
                seen.append(a.root)
        return seen

    @property
    def reasons(self) -> list[str]:
        return sorted({a.reason for a in self.atoms if a.reason})


def _index_name(e: Expr) -> str | None:
    """Pull `actual_seq_q_len` out of `InputIndex::ACTUAL_SEQ_Q_LEN`."""
    if isinstance(e, Ref) and "::" in e.symbol:
        head, tail = e.symbol.split("::", 1)
        if head.endswith("Index"):
            return tail.lower()
    return None


def _const_int(e: Expr, constants: Mapping[str, int] | None = None) -> int | None:
    """The integer an argument denotes, literal or named.

    Operators index their inputs and axes through named constants far more
    often than through literals (`GetDim(DIM_2)`, `GetInputShape(
    QUERY_INPUT_INDEX)`). Reading only literals loses which axis and which
    tensor was meant, and everything then collapses onto one variable.
    """
    if isinstance(e, Const) and isinstance(e.value, int) and not isinstance(e.value, bool):
        return e.value
    if isinstance(e, Ref) and constants:
        got = constants.get(e.symbol)
        if got is None and "::" in e.symbol:
            got = constants.get(e.symbol.split("::")[-1])
        if isinstance(got, int) and not isinstance(got, bool):
            return got
    return None


# `GetDim(2)` names one axis; `GetDimNum()` names the rank, which is not an axis.
_DIM_ACCESSOR_RE = re.compile(r"^GetDim$|^GetShapeDim$|^GetOriginDim$")

#: Reads the rank rather than the extent. Kept apart from the element count so
#: the two do not share a variable -- see `Atom.reads`.
_RANK_ACCESSOR_RE = re.compile(r"^GetDimNum$|^GetShapeDimNum$|^GetOriginDimNum$")
READS_RANK = "rank"

#: Accessors that select one operand by position, and which list it indexes.
#: The position is what tells `query`'s shape from `key`'s; without it every
#: tensor in the operator shares a single variable.
_OPERAND_ACCESSORS: list[tuple[str, str]] = [
    (r"^GetOptionalOutputShape$|^GetOutputShape$|^GetOutputDesc$|^GetOutputTensor$", "output"),
    (r"^GetOptionalInputShape$|^GetOptionalInputDesc$|^GetOptionalInputTensor$", "input"),
    (r"^GetInputShape$|^GetInputDesc$|^GetInputTensor$|^GetInputPointer$", "input"),
    (r"^GetAttrPointer$|^GetBool$|^GetInt$|^GetFloat$|^GetStr$", "attr"),
]


def _operand_list(name: str) -> str | None:
    for pat, kind in _OPERAND_ACCESSORS:
        if re.match(pat, name):
            return kind
    return None


#: Suffixes a by-name layer adds to a declared operand. `attenMaskOptional` is
#: `atten_mask`; `dqOut` is the output `dq`. Only tried once the plain name has
#: missed, so an operand really ending this way still wins.
#: `shape` and `dtype` are here because a by-name layer keeps what it read in
#: a local named after both the tensor and the thing read: `auto queryShape =
#: query->GetViewShape()`, then `queryShape.GetDimNum()`. The outer accessor
#: already says which of the two it wants, so the operand is all that is left
#: to recover.
_OPERAND_SUFFIXES = (
    "optional", "out", "tensor", "input", "in", "shape", "dtype", "type", "desc"
)


def _squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _declared_operand(ident: str, operands: Mapping[str, Sequence[str]]) -> str | None:
    """The declared operand a bare name refers to, if any.

    Tiling reaches a tensor by position -- `GetInputDesc(0)` -- but every layer
    written for people reaches it by name: `query->GetDataType()`, or a helper
    taking `const aclTensor *query`. Both name the same operand, and only the
    positional form was being recognised, so every by-name read fell back to
    the accessor's own name and the whole operator shared one dtype variable.

    Spellings are compared with separators and case removed, since the two
    sides disagree on both (`queryRope` against `query_rope`).
    """
    want = _squash(ident)
    if not want:
        return None
    known = {_squash(n): n for names in operands.values() for n in names}
    hit = known.get(want)
    if hit:
        return hit
    trimmed = want
    while True:
        for suffix in _OPERAND_SUFFIXES:
            if trimmed.endswith(suffix) and len(trimmed) > len(suffix):
                trimmed = trimmed[: -len(suffix)]
                break
        else:
            return None
        hit = known.get(trimmed)
        if hit:
            return hit


def _dim_index(name: str, args, constants: Mapping[str, int] | None = None) -> int | None:
    if not _DIM_ACCESSOR_RE.match(name):
        return None
    for a in args:
        got = _const_int(a, constants)
        if got is not None:
            return got
    return None


def _match(patterns: list[tuple[str, str]], text: str) -> str | None:
    for pat, root in patterns:
        if re.search(pat, text):
            return root
    return None


def _call_name(e: Call) -> str:
    return e.func[len("field:") :] if e.func.startswith("field:") else e.func


def dotted_path(e: Expr) -> str | None:
    """Rebuild `a.b.c` from the nested field-access Calls the parser produces."""
    parts: list[str] = []
    cur: Expr | None = e
    while isinstance(cur, Call) and cur.func.startswith("field:"):
        parts.append(cur.func[len("field:") :])
        cur = cur.args[0] if cur.args else None
    if isinstance(cur, Ref):
        parts.append(cur.symbol)
    elif cur is not None:
        return None
    return ".".join(reversed(parts)) if len(parts) >= 2 else None


class SourceResolver:
    def __init__(
        self,
        host_ir=None,
        max_depth: int = 16,
        bindings: dict[str, str] | None = None,
        local_roots: dict[str, str] | None = None,
        parameters: set[str] | None = None,
        param_actuals: dict[str, list[str]] | None = None,
        def_lists: dict[str, list[str]] | None = None,
        constants: Mapping[str, int] | None = None,
        operands: Mapping[str, Sequence[str]] | None = None,
    ):
        self.host_ir = host_ir
        self.max_depth = max_depth
        # Enum members and constexpr ints, so `GetDim(DIM_2)` names axis 2.
        self.constants = dict(constants or {})
        # "input"/"output"/"attr" -> operand names in declaration order, so a
        # positional accessor resolves to the operand the opdef declared rather
        # than to the name of the accessor that read it.
        self.operands = {k: list(v) for k, v in (operands or {}).items()}
        # names bound by the enclosing signature: resolving them needs the caller
        self.parameters = set(parameters or ())
        # parameter name -> actual argument sources observed at call sites
        self.param_actuals = dict(param_actuals or {})
        # local variable name -> initialiser source, so guards phrased in terms
        # of locals resolve through to the accessor that produced them
        self.bindings = dict(bindings or {})
        # local -> every known RHS (for cycle-breaking multi-def chase)
        self.def_lists = {k: list(v) for k, v in (def_lists or {}).items()}
        # symbol -> root, for things with no initialiser to chase (loop vars)
        self.local_roots = dict(local_roots or {})
        # names currently being chased, so `a = b; b = a` terminates
        self._chasing: set[str] = set()
        # Mutable cell so scoped() children share one top-level budget.
        self._budget = [0]

    def _charge(self) -> bool:
        """True while this top-level resolve still has walk budget."""
        cell = getattr(self, "_budget", None)
        if not isinstance(cell, list) or not cell:
            self._budget = [0]
            cell = self._budget
        cell[0] = int(cell[0] or 0) + 1
        return cell[0] <= RESOLVE_STEP_BUDGET

    def adopt(self, var_model: Any) -> None:
        """Take the constant table and operand order from the variable model.

        The resolver is built before the model exists, but it cannot tell
        `GetInputShape(QUERY_INPUT_INDEX)` from `GetInputShape(KEY_INPUT_INDEX)`
        without both. Handing them over afterwards keeps the construction order
        and still gives every later resolution the operand's real identity.
        """
        if var_model is None:
            return
        self.constants = dict(getattr(var_model, "named_constants", {}) or {})
        getter = getattr(var_model, "operand_names", None)
        if callable(getter):
            self.operands = {k: list(v) for k, v in getter().items()}

    def scoped(
        self,
        *,
        bindings=None,
        local_roots=None,
        parameters=None,
        param_actuals=None,
        def_lists=None,
    ) -> "SourceResolver":
        """A view with extra per-node bindings (function locals, loop variables)."""
        child = SourceResolver(host_ir=self.host_ir, max_depth=self.max_depth)
        child.constants = self.constants
        child.operands = self.operands
        child.bindings = {**self.bindings, **(bindings or {})}
        child.local_roots = {**self.local_roots, **(local_roots or {})}
        child.parameters = self.parameters | set(parameters or ())
        child.param_actuals = {**self.param_actuals, **(param_actuals or {})}
        child.def_lists = {**self.def_lists, **(def_lists or {})}
        child._chasing = set(self._chasing)
        child._budget = self._budget
        return child

    # -- atoms -------------------------------------------------------------
    def resolve_call(self, e: Call, depth: int) -> Atom:
        # Member access keeps `field:` on e.func; _call_name strips it for display.
        if e.func.startswith("field:"):
            field = e.func[len("field:") :]
            if e.args:
                base_txt = self._expr_text(e.args[0])
                sub = self.resolve(base_txt, depth + 1) if base_txt else None
                if sub is not None and sub.closed and sub.atoms:
                    first = sub.atoms[0]
                    return Atom(
                        text=f"{base_txt}.{field}" if base_txt else field,
                        root=first.root,
                        symbol=first.symbol,
                        index=first.index,
                        reads=first.reads,
                        via=(f".{field}",) + first.via,
                        partial=first.partial,
                    )
                if field in _TUPLE_ACCESSORS and sub is not None and sub.atoms:
                    # even partial base: prefer reporting the base blocker
                    return sub.atoms[0]
            return Atom(text=field, reason=REASON_UNMAPPED_SYMBOL)

        name = _call_name(e)
        if name == "__tuple_elem" and len(e.args) >= 2:
            return self._resolve_tuple_elem(e.args[0], e.args[1], depth)
        if name in ("std::get", "get") and len(e.args) == 1:
            # index was stripped by the template eraser; cannot recover — treat
            # as passthrough of the tuple (caller should use __tuple_elem).
            return self._inherit_arg_roots(e, depth) or Atom(
                text=name, reason=REASON_UNMAPPED_CALL
            )

        root = _match(CALL_ROOTS, name)
        if root is None:
            if "." in name or "->" in name:
                return self._chase_field(name.replace("->", "."), depth)
            through = self._chase_helper_body(name, e, depth)
            if through is not None:
                return through
            inherited = self._inherit_arg_roots(e, depth)
            if inherited is not None:
                return inherited
            # C++ keywords mis-parsed out of truncated macro text
            if name in {"for", "while", "if", "switch", "return", "sizeof"}:
                return Atom(text=name, reason=REASON_NO_CONDITION, root=None)
            return Atom(text=name, reason=REASON_UNMAPPED_CALL, root=None)
        dim_index = _dim_index(name, e.args, self.constants)
        symbol = self._operand_symbol(name, e.args)
        if symbol is None:
            inherited = self._inherit_operand(e, root, depth)
            if inherited is not None:
                symbol, inner_index = inherited
                if dim_index is None:
                    dim_index = inner_index
        if symbol is None:
            # Nothing said which operand this reads, so the accessor's own name
            # has to stand for all of them. `identity_merged` marks the result.
            symbol = name
        reads = READS_RANK if _RANK_ACCESSOR_RE.match(name) else None
        return Atom(
            text=name, root=root, symbol=symbol, index=dim_index, reads=reads
        )

    def _operand_symbol(self, name: str, args: Sequence[Expr]) -> str | None:
        """Which operand a positional accessor selects: `GetInputShape(1)` → key."""
        for a in args:
            named = _index_name(a)
            if named:
                return named
        kind = _operand_list(name)
        names = self.operands.get(kind or "") or []
        if not names:
            return None
        for a in args:
            pos = _const_int(a, self.constants)
            if pos is not None and 0 <= pos < len(names):
                return names[pos]
        return None

    def _inherit_operand(self, e: Call, root: str, depth: int) -> tuple[str, int | None] | None:
        """Carry the operand down an accessor chain.

        `GetInputShape(0)->GetStorageShape().GetDim(2)` parses as nested calls
        with the receiver first, so which tensor is read is only known at the
        innermost call. What is read off it — shape, dtype, format — is decided
        by the outer accessor, which is why the receiver's root is not required
        to match: `GetInputDesc(0)->GetDataType()` is query's dtype.
        """
        if depth >= self.max_depth or not e.args:
            return None
        got = self._operand_of(e.args[0], depth)
        if got is not None:
            return got[0], got[1]
        for a in e.args[1:]:
            other = self._operand_of(a, depth)
            if other is not None and other[2] == root:
                return other[0], other[1]
        return None

    def _operand_of(self, a: Expr, depth: int) -> tuple[str, int | None, str | None] | None:
        """The operand an argument denotes, chasing locals that alias a chain."""
        inner: Atom | None = None
        if isinstance(a, Call):
            inner = self.resolve_call(a, depth + 1)
        elif isinstance(a, Ref) and (
            a.symbol in self.bindings
            or a.symbol in self.def_lists
            # A helper's formal names the tensor its callers passed:
            # `IsSameShape(dyShape, queryShape)` decides what `aShape` is. Left
            # out, the accessor read off a formal had no operand and fell back
            # to its own name, merging every helper's tensors into one.
            or a.symbol in self.parameters
            or a.symbol in self.param_actuals
        ):
            # `auto &queryShape = context->GetInputShape(...)->GetStorageShape();`
            sub = self.resolve(a.symbol, depth + 1)
            inner = sub.atoms[0] if sub.atoms else None
        if inner is None or not inner.symbol or inner.reason:
            return self._named_operand(a)
        if _match(CALL_ROOTS, inner.symbol) is not None:
            # The receiver did not resolve to an operand either; propagating its
            # accessor name would just spread the collapse one level further.
            return self._named_operand(a)
        return inner.symbol, inner.index, inner.root

    def _named_operand(self, a: Expr) -> tuple[str, None, None] | None:
        """Last resort: the receiver is simply the operand's declared name.

        True of the outermost function of a by-name layer, whose parameters no
        caller in scope supplies, so there is nothing to chase them to.
        """
        if not isinstance(a, Ref):
            return None
        hit = _declared_operand(a.symbol, self.operands)
        return (hit, None, None) if hit else None

    def _resolve_tuple_elem(self, idx_expr: Expr, tup_expr: Expr, depth: int) -> Atom:
        """Resolve `__tuple_elem(i, tup)` through make_tuple / tie actuals."""
        from ascendc_codemap_mcp.engine.expr_ir import Const

        if not isinstance(idx_expr, Const):
            return Atom(text="__tuple_elem", reason=REASON_UNMAPPED_CALL)
        try:
            index = int(idx_expr.value)
        except (TypeError, ValueError):
            return Atom(text="__tuple_elem", reason=REASON_UNMAPPED_CALL)
        tup_txt = _norm_expr(self._expr_text(tup_expr))
        # Direct make_tuple / tie in the expression
        elem = _nth_tuple_arg(tup_txt, index)
        if elem:
            sub = self.resolve(elem, depth + 1)
            if sub.atoms:
                first = sub.atoms[0]
                return Atom(
                    text=f"__tuple_elem({index})",
                    root=first.root,
                    symbol=first.symbol,
                    reason=first.reason,
                    via=(f"get<{index}>({tup_txt[:30]})",) + first.via,
                    partial=first.partial,
                )
        # Parameter whose call-site actual is make_tuple(...)
        if tup_txt in self.parameters or tup_txt in self.param_actuals:
            for actual in self.param_actuals.get(tup_txt, []):
                elem = _nth_tuple_arg(_norm_expr(actual), index)
                if not elem:
                    continue
                # Same-name cycle: make_tuple(m,...,) where callee also binds
                # m via __tuple_elem back into this parameter.
                if elem in self._chasing or elem == tup_txt:
                    return Atom(
                        text=f"__tuple_elem({index})",
                        root="TILING_DATA",
                        symbol=elem,
                        via=(f"get<{index}>(pack)",),
                    )
                sub = self.resolve(elem, depth + 1)
                if sub.closed and sub.atoms:
                    first = sub.atoms[0]
                    return Atom(
                        text=f"__tuple_elem({index})",
                        root=first.root,
                        symbol=first.symbol,
                        via=(f"get<{index}>({actual[:30]})",) + first.via,
                    )
                # Bare pack element with no independent def: host coordinate
                # packs are tiling-derived scratch state.
                if re.fullmatch(r"[A-Za-z_]\w*", elem) and (
                    not sub.closed or not sub.atoms
                ):
                    return Atom(
                        text=f"__tuple_elem({index})",
                        root="TILING_DATA",
                        symbol=elem,
                        via=(f"get<{index}>({actual[:30]})",),
                    )
            # fall through: resolve the tuple binding itself
        sub = self.resolve(tup_txt, depth + 1)
        if sub.closed and sub.atoms:
            first = sub.atoms[0]
            # If the tuple closed via a make_tuple in via/bindings, try again
            for actual in self.param_actuals.get(tup_txt, []):
                elem = _nth_tuple_arg(_norm_expr(actual), index)
                if elem:
                    inner = self.resolve(elem, depth + 1)
                    if inner.closed and inner.atoms:
                        return Atom(
                            text=f"__tuple_elem({index})",
                            root=inner.atoms[0].root,
                            symbol=inner.atoms[0].symbol,
                            via=(f"get<{index}>",) + inner.atoms[0].via,
                        )
            return Atom(
                text=f"__tuple_elem({index})",
                root=first.root,
                symbol=first.symbol,
                via=(f"get<{index}>({tup_txt[:30]})",) + first.via,
                partial=first.partial,
            )
        return Atom(text=f"__tuple_elem({index},{tup_txt})", reason=REASON_UNMAPPED_SYMBOL)

    def _inherit_arg_roots(self, e: Call, depth: int) -> Atom | None:
        if not e.args or depth >= self.max_depth:
            return None
        roots: list[Atom] = []
        for a in e.args:
            sub = self.resolve(self._expr_text(a), depth + 1)
            if not sub.closed or not sub.atoms:
                return None
            meaningful = [x for x in sub.atoms if x.root not in (None, "CONSTANT")]
            roots.extend(meaningful or list(sub.atoms))
        if not roots:
            return None
        first = roots[0]
        return Atom(
            text=_call_name(e),
            root=first.root,
            symbol=first.symbol,
            index=first.index,
            reads=first.reads,
            via=(f"{_call_name(e)}(...)",) + first.via,
            partial=first.partial,
        )

    def resolve_symbol(self, sym: str, depth: int) -> Atom:
        if "[" in sym:
            sym = _strip_subscripts(sym)
        if sym in self.local_roots:
            return Atom(text=sym, root=self.local_roots[sym], symbol=sym)

        # `best.first` where `best` is a local/param: chase the head, do not
        # look for a tiling write to a field literally named `first`.
        if "." in sym:
            head, _, rest = sym.partition(".")
            head_known = (
                head in self.bindings
                or head in self.def_lists
                or head in self.local_roots
                or head in self.parameters
            )
            if head_known and depth < self.max_depth:
                base = self.resolve_symbol(head, depth + 1)
                accessor = rest.split(".", 1)[0]
                if base.root in LEGAL_ROOTS and not base.partial and not base.reason:
                    if (
                        base.root == "TILING_DATA"
                        or accessor in _LOCAL_TRANSPARENT_ACCESSORS
                        or self.tiling_derived(f"x.{rest}")
                    ):
                        return Atom(
                            text=sym,
                            root=base.root,
                            symbol=base.symbol,
                            index=base.index,
                            reads=base.reads,
                            via=(f"{head}->{rest}",) + base.via,
                        )
                if (
                    (
                        base.root == "TILING_DATA"
                        or accessor in _LOCAL_TRANSPARENT_ACCESSORS
                    )
                    and base.root in LEGAL_ROOTS
                    and not base.reason
                ):
                    return Atom(
                        text=sym,
                        root=base.root,
                        symbol=base.symbol,
                        index=base.index,
                        reads=base.reads,
                        via=(f"{head}->{rest}",) + base.via,
                        partial=base.partial,
                    )

        # Collect defining RHSs; prefer ones that do not re-mention `sym`.
        raw: list[str] = []
        for rhs in self.def_lists.get(sym, ()):
            n = _norm_expr(rhs)
            if n and n not in raw:
                raw.append(n)
        if sym in self.bindings:
            n = _norm_expr(self.bindings[sym])
            if n and n not in raw:
                raw.insert(0, n)
        others = [c for c in raw if c != sym]
        independent = [c for c in others if not _mentions_sym(sym, c)]
        if _self_derived_local(sym, others):
            return Atom(
                text=sym,
                root="LOOP_DERIVED",
                symbol=sym,
                via=(f"{sym}=self-derived",),
            )
        candidates = (independent or others)[:4]
        # Dropping the self-mentioning definitions is how `p = CeilDiv(...);
        # p = p + q` still reaches the CeilDiv, but for a counter it leaves
        # only the initialiser: `coreIdx = 0` beside `coreIdx += 1`. That is
        # the value before the loop and never after it, so accepting it as
        # the constant pins everything downstream to the empty case and rules
        # out keys that are reachable.
        accumulates = bool(independent) and len(independent) < len(others)
        if candidates and depth < self.max_depth and sym not in self._chasing:
            self._chasing.add(sym)
            try:
                fallback: Atom | None = None
                constant: Atom | None = None
                for rhs in candidates:
                    sub = self.resolve_value(rhs, depth + 1)
                    meaningful = [a for a in sub.atoms if a.root != "CONSTANT"]
                    if meaningful:
                        first = meaningful[0]
                        atom = Atom(
                            text=sym if first.reason is None else f"{sym}<-{first.text}",
                            root=first.root,
                            symbol=first.symbol,
                            index=first.index,
                            reads=first.reads,
                            reason=first.reason,
                            via=(f"{sym}={rhs[:40]}",) + first.via,
                            partial=first.partial,
                        )
                        if sub.closed:
                            return atom
                        if fallback is None:
                            fallback = atom
                        continue
                    # `bool isExceed = false;` ahead of the write that decides
                    # it. Remember the constant, but let a definition carrying
                    # actual provenance win over declaration order.
                    if sub.closed and sub.atoms and constant is None:
                        if _picks_between_constants(sub):
                            continue
                        constant = Atom(
                            text=sym,
                            root="CONSTANT",
                            symbol=sub.atoms[0].symbol or sym,
                            via=(f"{sym}={rhs[:40]}",) + sub.atoms[0].via,
                        )
                if fallback is not None:
                    return fallback
                # Last resort, not a shortcut: a local we could not reduce, whose
                # RHS is visibly tiling-derived, really is host tiling state.
                # Taking this branch *before* the chase above is what mislabels
                # reducible locals such as `isExceed` as a TILING_DATA field that
                # no variable corresponds to.
                for rhs in candidates:
                    if self.tiling_derived(rhs) and not _INPUT_ACCESSOR_RE.search(rhs):
                        return Atom(
                            text=sym,
                            root="TILING_DATA",
                            symbol=sym,
                            via=(f"{sym}={rhs[:40]}",),
                        )
                if constant is not None and not accumulates:
                    return constant
            finally:
                self._chasing.discard(sym)
        root = _match(SYMBOL_ROOTS, sym)
        if root:
            return Atom(text=sym, root=root, symbol=sym)
        if DERIVED_FIELD_RE.match(sym) or "." in sym:
            return self._chase_field(sym, depth)
        if sym in self.parameters:
            return self._chase_parameter(sym, depth)
        if self.host_ir is not None and sym in self.host_ir.class_fields:
            # A class field the host fills member by member is tiling state as a
            # whole; chasing a write to the aggregate itself would find nothing.
            if sym in self.host_ir.aggregate_heads():
                return Atom(text=sym, root="TILING_DATA", symbol=sym)
            return self._chase_field(f"this.{sym}", depth)
        return Atom(text=sym, reason=REASON_UNMAPPED_SYMBOL)

    def tiling_derived(self, text: str) -> bool:
        """Does `text` read a member of a host tiling aggregate?

        Decided structurally: an aggregate is a symbol whose fields the host
        writes (`HostIR.aggregate_heads`). Falls back to the name patterns only
        when there is no IR to ask, since those hardcode one operator's names.
        """
        if not text:
            return False
        ir = self.host_ir
        if ir is None:
            return bool(_PARAMS_DERIVED_RE.search(text))
        heads = ir.aggregate_heads()
        return any(m.group(1) in heads for m in _DOTTED_HEAD_RE.finditer(text))

    def _in_function(self, fn: str) -> "SourceResolver":
        """Re-scope onto another function: its locals, parameters and call sites.

        Needed because a write like `fBaseParams.queryType = queryType` names a
        local of the *writing* function, which is invisible from the guard site.
        """
        if not fn or self.host_ir is None:
            return self
        summary = self.host_ir.summaries.get(fn)
        if summary is None:
            return self
        bindings = dict(self.host_ir.locals_by_function().get(fn, {}))
        bindings.update(self.host_ir.output_bindings_by_function().get(fn, {}))
        local_roots = inferred_function_local_roots(self.host_ir, fn)
        local_roots.update(inferred_parameter_roots(self.host_ir, fn))
        return self.scoped(
            bindings=bindings,
            local_roots=local_roots,
            def_lists=self.host_ir.defs_by_function().get(fn, {}),
            parameters=set(summary.params),
            param_actuals=self.host_ir.param_bindings().get(fn, {}),
        )

    def _expr_text(self, e: Expr) -> str:
        if isinstance(e, Ref):
            return e.symbol
        if isinstance(e, Const):
            return repr(e.value)
        if isinstance(e, Call):
            name = _call_name(e)
            args = ", ".join(self._expr_text(a) for a in e.args)
            return f"{name}({args})"
        if isinstance(e, Un):
            return f"{e.op}{self._expr_text(e.arg)}"
        if isinstance(e, Bin):
            return f"{self._expr_text(e.left)} {e.op} {self._expr_text(e.right)}"
        if isinstance(e, Select):
            return f"{self._expr_text(e.array)}[{self._expr_text(e.index)}]"
        return ""

    def _chase_helper_body(self, name: str, call: Call, depth: int) -> Atom | None:
        """Chase a known FuncRecord through returns / out-param assigns (generic)."""
        if self.host_ir is None or depth >= self.max_depth:
            return None
        short = name.split("::")[-1]
        if short in self._chasing:
            return None
        summary = self.host_ir.summaries.get(short)
        if summary is None:
            return None
        helper_weight = (
            len(summary.calls)
            + len(summary.locals)
            + len(summary.guards)
            + len(summary.returns)
        )
        if helper_weight > HELPER_BODY_WEIGHT_LIMIT:
            return None
        # Skip huge / non-helper bodies: chasing DoOpTiling-scale functions
        # through every call site explodes. Prefer small helpers with returns
        # or explicit out-params.
        if not summary.returns and not summary.out_params:
            if len(summary.guards) > 8:
                return None
        extra: dict[str, str] = {}
        for p, a in zip(summary.params, call.args):
            text = self._expr_text(a).lstrip("&").strip()
            if text:
                extra[p] = text
        self._chasing.add(short)
        try:
            callee = self._in_function(short).scoped(bindings=extra)

            for expr in summary.returns:
                sub = callee.resolve(expr, depth + 1)
                meaningful = [a for a in sub.atoms if a.root != "CONSTANT"]
                if not meaningful:
                    if sub.closed and sub.atoms and not _picks_between_constants(sub):
                        return Atom(
                            text=name,
                            root="CONSTANT",
                            symbol=sub.atoms[0].symbol,
                            via=(f"{name}()->{expr[:40]}",),
                        )
                    continue
                if not sub.closed:
                    blocked = next((a for a in sub.atoms if a.reason), None)
                    return Atom(
                        text=name,
                        reason=blocked.reason if blocked else REASON_UNMAPPED_CALL,
                        via=(f"{name}()->{expr[:40]}",),
                    )
                return Atom(
                    text=name,
                    root=meaningful[0].root,
                    symbol=meaningful[0].symbol,
                    index=meaningful[0].index,
                    reads=meaningful[0].reads,
                    via=(f"{name}()->{expr[:40]}",) + meaningful[0].via,
                )

            for p in summary.out_params or ():
                rhs = summary.assigns.get(p)
                if not rhs:
                    continue
                sub = callee.resolve(rhs, depth + 1)
                if sub.closed and sub.atoms:
                    first = sub.atoms[0]
                    return Atom(
                        text=name,
                        root=first.root,
                        symbol=first.symbol,
                        index=first.index,
                        reads=first.reads,
                        via=(f"{name}(&{p})={rhs[:40]}",) + first.via,
                        partial=first.partial,
                    )

            if summary.returns or summary.out_params:
                roots, symbols, via = self._roots_of_guards(callee, summary, depth)
                if roots:
                    return Atom(text=name, root=roots[0], symbol=symbols[0], via=(via,))
            return None
        finally:
            self._chasing.discard(short)

    def _chase_return(self, name: str, depth: int) -> Atom | None:
        """Compatibility wrapper: body chase without a Call node."""
        if self.host_ir is None or depth >= self.max_depth:
            return None
        short = name.split("::")[-1]
        summary = self.host_ir.summaries.get(short)
        if summary is None:
            return None
        callee = self._in_function(short)
        for expr in summary.returns:
            sub = callee.resolve(expr, depth + 1)
            meaningful = [a for a in sub.atoms if a.root != "CONSTANT"]
            if not meaningful:
                continue
            if not sub.closed:
                blocked = next((a for a in sub.atoms if a.reason), None)
                return Atom(
                    text=name,
                    reason=blocked.reason if blocked else REASON_UNMAPPED_CALL,
                    via=(f"{name}()->{expr[:40]}",),
                )
            return Atom(
                text=name,
                root=meaningful[0].root,
                symbol=meaningful[0].symbol,
                via=(f"{name}()->{expr[:40]}",),
            )
        roots, symbols, via = self._roots_of_guards(callee, summary, depth)
        if not roots:
            return None
        return Atom(text=name, root=roots[0], symbol=symbols[0], via=(via,))

    def _roots_of_guards(self, callee, summary, depth):
        roots: list[str] = []
        symbols: list[str] = []
        for guard in summary.guards:
            sub = callee.resolve(guard, depth + 1)
            meaningful = [a for a in sub.atoms if a.root not in (None, "CONSTANT")]
            if not sub.closed or not meaningful:
                return [], [], ""
            roots.append(meaningful[0].root or "UNKNOWN")
            symbols.append(meaningful[0].symbol)
        via = f"{summary.name}() guarded by {summary.guards[0][:40]}" if roots else ""
        return roots, symbols, via

    def _chase_parameter(self, sym: str, depth: int) -> Atom:
        """Resolve a formal parameter through the actual arguments its callers pass."""
        actuals = self.param_actuals.get(sym, [])
        if not actuals or depth >= self.max_depth:
            return Atom(text=sym, reason=REASON_FUNCTION_PARAMETER)
        roots: list[str] = []
        symbols: list[str] = []
        for actual in actuals:
            resolved = actual.lstrip("&").strip()
            if resolved == sym:
                # same-name actual: try the local binding already in this scope
                if sym in self.bindings and self.bindings[sym] != sym:
                    resolved = self.bindings[sym]
                else:
                    continue
            if self.tiling_derived(resolved) and not _INPUT_ACCESSOR_RE.search(resolved):
                roots.append("TILING_DATA")
                symbols.append(sym)
                continue
            sub = self.resolve(resolved, depth + 1)
            if not sub.closed or not sub.roots:
                return Atom(text=sym, reason=REASON_FUNCTION_PARAMETER)
            roots.append(sub.roots[0])
            symbols.append(sub.atoms[0].symbol)
        if not roots:
            return Atom(text=sym, reason=REASON_FUNCTION_PARAMETER)
        return Atom(
            text=sym,
            root=roots[0],
            symbol=symbols[0],
            via=(f"{sym}<-{'|'.join(a[:24] for a in actuals[:3])}",),
        )

    def _chase_field(self, path: str, depth: int) -> Atom:
        """Follow a host-computed field back to the RHS of its defining writes.

        A field is usually assigned from several places under different guards,
        so every defining write is tried; the field closes as soon as one of
        them reaches a root, and the winning assignment is recorded in `via`.
        """
        if self.host_ir is None:
            return _unproven_field(path, "NO_HOST_IR")
        if depth >= self.max_depth:
            return Atom(text=path, reason=REASON_DEPTH_EXCEEDED)
        path = _strip_subscripts(path)
        if path in self._chasing:
            return _unproven_field(path, "CYCLIC_FIELD_DEPENDENCY")
        tail = path.split(".")[-1]
        by_tail = (
            self.host_ir.writes_by_tail().get(tail, [])
            if hasattr(self.host_ir, "writes_by_tail")
            else [w for w in self.host_ir.writes if w.rhs.strip()]
        )
        cleaned_writes = [(_strip_subscripts(w.path), w) for w in by_tail]
        # Prefer longest / most specific path match.
        exact = [w for p, w in cleaned_writes if p == path or p.endswith("." + path)]
        if exact:
            writes = exact
        else:
            loose = [(p, w) for p, w in cleaned_writes if p.endswith("." + tail)]
            same_head = [(p, w) for p, w in loose if _same_aggregate(p, path)]
            pool = same_head or [
                (p, w) for p, w in loose if not _is_generated_accessor(w.function, tail)
            ]
            writes = [w for _, w in (pool or loose)]
            writes = sorted(writes, key=lambda w: (w.file, w.line), reverse=True)[:8]
        if not writes:
            return _unproven_field(path, REASON_TILING_DATA_NO_WRITER)

        self._chasing.add(path)
        try:
            return self._chase_writes(path, writes, depth)
        finally:
            self._chasing.discard(path)

    def _chase_writes(self, path: str, writes, depth: int) -> Atom:
        """Resolve a field through its defining writes.

        The first closed write is *not* always enough: an enum-valued host field
        (layoutType, splitAxis, deterSparseType, …) has several constant writes
        under different guards. Picking the first collapses it to CONSTANT and
        makes `layoutType == INPUT_FORMAT_TND` look like a tautology over a
        fixed literal — which is exactly how IsTnd lost its input root.
        """
        fallback: Atom | None = None
        closed: list[Atom] = []
        for i, w in enumerate(writes):
            if i >= 8 or not self._charge():
                break
            sub = self._in_function(w.function).resolve_value(w.rhs, depth + 1)
            if not sub.atoms:
                continue
            first = sub.atoms[0]
            via = (f"{w.ssa_name}={w.rhs[:40]}",) + first.via
            root = first.root
            # A5: TDF written from an input accessor → controllable INPUT_VALUE
            if sub.closed and root in (None, "TILING_DATA"):
                if _INPUT_ACCESSOR_RE.search(w.rhs) or any(
                    _INPUT_ACCESSOR_RE.search(v) for v in via
                ):
                    root = "INPUT_VALUE"
            if sub.closed:
                closed.append(
                    Atom(
                        text=path,
                        root=root or first.root,
                        symbol=first.symbol,
                        index=first.index,
                        reads=first.reads,
                        via=via,
                    )
                )
                continue
            if fallback is None:
                fallback = Atom(
                    text=f"{path}<-{first.text}",
                    reason=first.reason or REASON_UNMAPPED_SYMBOL,
                    via=via,
                )
        if not closed:
            return fallback or _unproven_field(path, REASON_TILING_DATA_NO_WRITER)
        non_const = [a for a in closed if a.root != "CONSTANT"]
        if non_const:
            return non_const[0]
        const_syms = _constant_values(closed)
        if len(const_syms) > 1:
            # Distinct enum/constexpr assignments: the field is host state that
            # *holds* a constant, not a constant itself.
            return Atom(
                text=path,
                root="TILING_DATA",
                symbol=path.split(".")[-1],
                via=closed[0].via,
            )
        return closed[0]

    # -- expression tree ---------------------------------------------------
    def _walk(self, e: Expr, out: list[Atom], depth: int, *, as_value: bool = False) -> None:
        if not self._charge():
            out.append(Atom(text="?", reason=REASON_RESOLVE_BUDGET))
            return
        if isinstance(e, Const):
            out.append(Atom(text=repr(e.value), root="CONSTANT", symbol=repr(e.value)))
            return
        if isinstance(e, Unknown):
            out.append(Atom(text="?", reason=e.reason))
            return
        if isinstance(e, Ref):
            out.append(self.resolve_symbol(e.symbol, depth))
            return
        if isinstance(e, Call):
            path = dotted_path(e)
            if path is not None:
                out.append(self.resolve_symbol(path, depth))
                return
            if _call_name(e) in PASSTHROUGH_CALLS:
                for a in e.args:
                    self._walk(a, out, depth, as_value=as_value)
                return
            atom = self.resolve_call(e, depth)
            if atom.root is None:
                # unmapped wrapper: still try to close its arguments
                inner: list[Atom] = []
                for a in e.args:
                    self._walk(a, inner, depth + 1, as_value=as_value)
                meaningful = [a for a in inner if a.root not in (None, "CONSTANT")]
                if meaningful and all(a.root in LEGAL_ROOTS for a in meaningful):
                    out.extend(meaningful)
                    return
            out.append(atom)
            return
        if isinstance(e, Un):
            self._walk(e.arg, out, depth, as_value=as_value)
            return
        if isinstance(e, Bin):
            self._walk(e.left, out, depth, as_value=as_value)
            self._walk(e.right, out, depth, as_value=as_value)
            return
        if isinstance(e, Ite):
            # Value position: `x = c ? a : b` is decided by `a`/`b`. Walking
            # `c` first made numeric fields like `fBaseParams.d` collapse to
            # the optional-input presence test inside `c` (hasRope).
            if not as_value:
                self._walk(e.cond, out, depth, as_value=False)
                self._walk(e.then, out, depth, as_value=True)
                self._walk(e.else_, out, depth, as_value=True)
                return
            branch_atoms: list[Atom] = []
            self._walk(e.then, branch_atoms, depth, as_value=True)
            self._walk(e.else_, branch_atoms, depth, as_value=True)
            if any(a.root != "CONSTANT" for a in branch_atoms):
                out.extend(branch_atoms)
                return
            # If every arm is a literal, the selector is the provenance of the
            # chosen enum/mode.  Without this, `cond ? 0 : 1` is neither a safe
            # constant nor traced back to the inputs deciding `cond`.
            self._walk(e.cond, out, depth, as_value=False)
            out.extend(branch_atoms)
            return
        if isinstance(e, Select):
            # Prefer the array's root; the index is often a loop induction var
            # that would otherwise keep a closed container access "open".
            before = len(out)
            self._walk(e.array, out, depth, as_value=as_value)
            arr = out[before:]
            if arr and all(
                a.root in LEGAL_ROOTS and not a.partial and not a.reason for a in arr
            ):
                return
            self._walk(e.index, out, depth, as_value=False)
            return
        out.append(Atom(text=str(e), reason=REASON_UNMAPPED_SYMBOL))

    def resolve_value(self, expression: str, depth: int = 0) -> Resolution:
        """Resolve an assignment RHS: ternary *conditions* do not contribute roots.

        Guards still use `resolve()` (condition + both branches). Definition
        chase must not let `hasRope` steal provenance from
        `hasRope ? ROPE_D_192 : queryDim`.
        """
        if not expression or not expression.strip():
            return Resolution(
                condition=expression,
                atoms=[Atom(text="", reason=REASON_NO_CONDITION)],
            )
        expression = _norm_expr(expression)
        try:
            expr = parse_expr(expression)
        except Exception as exc:  # pragma: no cover
            return Resolution(
                condition=expression,
                atoms=[
                    Atom(
                        text=expression[:40],
                        reason=f"{REASON_PARSE_FAILED}:{exc}",
                    )
                ],
            )
        atoms: list[Atom] = []
        self._walk(expr, atoms, depth, as_value=True)
        non_const = [a for a in atoms if a.root != "CONSTANT"]
        if not non_const and atoms:
            return Resolution(condition=expression, atoms=list(atoms), expr=expr)
        return Resolution(
            condition=expression,
            atoms=non_const or list(atoms) or [Atom(text=expression[:40], reason=REASON_UNMAPPED_SYMBOL)],
            expr=expr,
        )

    def resolve(self, condition: str, depth: int = 0) -> Resolution:
        if not condition or not condition.strip():
            return Resolution(
                condition=condition,
                atoms=[Atom(text="", reason=REASON_NO_CONDITION)],
            )
        condition = _norm_expr(condition)
        if depth == 0:
            if not isinstance(getattr(self, "_budget", None), list):
                self._budget = [0]
            self._budget[0] = 0
        cache = getattr(self, "_resolve_cache", None)
        if cache is None:
            self._resolve_cache = {}
            cache = self._resolve_cache
        # Cache at every depth. Controllability re-resolves the same leaf /
        # guard texts across hundreds of sibling branches; the old depth<=2
        # cutoff threw away most of those hits. `_chasing` stays in the key so
        # recursive frames remain correct.
        cache_key = (condition, frozenset(self._chasing), depth)
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
        try:
            expr = parse_expr(condition)
        except Exception as exc:  # pragma: no cover - parser is total in practice
            return Resolution(
                condition=condition,
                atoms=[Atom(text=condition[:40], reason=f"{REASON_PARSE_FAILED}:{exc}")],
            )
        atoms: list[Atom] = []
        self._walk(expr, atoms, depth)
        # A guard made only of literals carries no source dependency.
        non_const = [a for a in atoms if a.root != "CONSTANT"]
        res = Resolution(condition=condition, atoms=non_const or atoms, expr=expr)
        cache[cache_key] = res
        return res


def resolve_node(node, resolver: SourceResolver) -> Resolution:
    """Resolve a control node's own condition (path conditions handled separately)."""
    return resolver.resolve(node.condition)


def resolve_with_path(node, resolver: SourceResolver) -> tuple[Resolution, list[Resolution]]:
    """Resolve the node condition plus every enclosing guard on its path."""
    own = resolver.resolve(node.condition)
    guards = [resolver.resolve(pc.text) for pc in node.path_conditions if not pc.is_opaque]
    return own, guards
