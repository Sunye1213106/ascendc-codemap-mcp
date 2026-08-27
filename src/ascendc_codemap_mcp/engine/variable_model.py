# -*- coding: utf-8 -*-
"""Named variables with value domains — the layer between source text and constraint IR.

The resolver answers "which root does this guard come from"; constraint consumers need
"which named variable, of which type, ranging over which values". This module
mints those variables from evidence only:

    opdef            input/output dtype and format lists, optional presence,
                     attribute types and defaults
    TilingKey DSL    dimension domains (exact, closed)
    enum class       scoped enumerations used by attributes
    check branches   integer bounds asserted by validation code

Anything the source does not prove is marked `completeness: open` rather than
invented, because the missing half is a test-strategy decision that belongs to
TG, not a fact about the operator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.ids import named_id, slug
from ascendc_codemap_mcp.engine.kb_model import Domain, Evidence

# `this->Input("query")` opens a block that runs until the next `this->`.
MEMBER_RE = re.compile(r'this->(Input|Output|Attr)\s*\(\s*"([^"]+)"\s*\)')
PARAM_TYPE_RE = re.compile(r"\.(?:Param|Attr)Type\s*\(\s*(\w+)\s*\)")
DATATYPE_RE = re.compile(r"\.DataType\s*\(\s*\{([^}]*)\}\s*\)")
FORMAT_RE = re.compile(r"\.Format\s*\(\s*\{([^}]*)\}\s*\)")
ATTR_VALUE_RE = re.compile(r"\.(Int|Float|Bool|String|ListInt|ListFloat)\s*\(([^)]*)\)")

# Attribute C++ declaration type -> constraint variable type.
ATTR_TYPE_MAP = {
    "Int": "int",
    "Float": "int",  # numeric attrs are bucketed by TG
    "Bool": "bool",
    "String": "enum",
    "ListInt": "int",
    "ListFloat": "int",
}

PLATFORM_VARS = {
    "PLATFORM_ARCH": "VAR_PLATFORM_ARCH",
    "PLATFORM_CORE_COUNT": "VAR_PLATFORM_CORE_COUNT",
    "PLATFORM_MEMORY_SIZE": "VAR_PLATFORM_MEMORY_SIZE",
    "PLATFORM_L2_SIZE": "VAR_PLATFORM_L2_SIZE",
    "PLATFORM_AIV_COUNT": "VAR_PLATFORM_AIV_COUNT",
}

# Session / graph runtime options (not op inputs, but controllable for TG).
SESSION_VARS = {
    "SESSION_OPTION": "VAR_SESSION_DETERMINISTIC",
}

# Functions whose branches assert legality rather than select an implementation.
VALIDATION_FN_RE = re.compile(r"check|valid|verify|assert", re.IGNORECASE)
FAILURE_RE = re.compile(r"GRAPH_FAILED|ge::FAILED|return\s+false|OP_LOGE|FAILED")
# `x > 65535`, `dimNum != 4`
SIMPLE_CMP_RE = re.compile(
    r"^\s*(?P<lhs>[\w.:>\-\[\]()]+)\s*(?P<op>[<>]=?|==|!=)\s*(?P<rhs>-?\d+)\s*$"
)


@dataclass
class ParamDecl:
    """One `Input` / `Output` / `Attr` entry of the operator definition."""

    kind: str  # input | output | attr
    name: str
    #: Position among entries of the same kind. This is what host tiling reads
    #: an input by -- `GetInputDesc(4)` names nothing -- so it is the only thing
    #: tying host code to the API layer, which reads the same input by name.
    index: int = -1
    param_type: str = ""  # REQUIRED | OPTIONAL | DYNAMIC
    #: Which dtypes this parameter may take, deduplicated.
    dtypes: list[str] = field(default_factory=list)
    #: The same list as written, in order and with repeats. Entry `i` of every
    #: parameter's row belongs to one supported combination, so deduplicating
    #: breaks the alignment -- `dtypes` cannot say that a FLOAT8 query goes
    #: with a FLOAT8 key.
    dtype_row: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    value_type: str = ""  # Int / Float / Bool / String for attrs
    default: str = ""
    file: str = ""
    line: int = 0

    @property
    def is_optional(self) -> bool:
        return self.param_type.upper() == "OPTIONAL"

    def evidence(self) -> Evidence:
        return Evidence.at(self.file, self.line, snippet=f'{self.kind} "{self.name}"')


@dataclass
class VarSpec:
    var_id: str
    name: str
    value_type: str
    domain: Domain
    origin: str
    evidence: list[Evidence] = field(default_factory=list)
    description: str = ""
    #: True when one id stands for several distinct values, because what
    #: distinguishes them is not in the id. Two cases produce it: a variable
    #: named after the accessor that read it (`VAR_SHAPE_GETSTORAGESHAPE` is
    #: every shape dimension in the operator), and an element of a container at
    #: an index we could not resolve.
    #:
    #: Within one expression the merge is a harmless over-approximation — the
    #: variable is free, so it can take whichever value that occurrence needs.
    #: It stops being harmless when two expressions are solved together, where
    #: sharing the id asserts an equality nobody proved. Anything conjoining
    #: expressions must isolate these per expression first.
    identity_merged: bool = False

    def to_tg_entry(self) -> dict[str, Any]:
        """Shape used by TG constraint and replay planning."""
        entry: dict[str, Any] = {"type": self.value_type}
        d = self.domain
        if d.values:
            entry["domain"] = list(d.values)
        if d.lo is not None:
            entry["lo"] = d.lo
        if d.hi is not None:
            entry["hi"] = d.hi
        entry["completeness"] = d.completeness
        entry["origin"] = self.origin
        entry["evidence_refs"] = [e.id for e in self.evidence]
        return entry


#: A variable whose name came from a getter call rather than from the thing it
#: reads. The resolver reaches `GetStorageShape` / `GetDimNum` / `GetAttrs` for
#: every tensor and every axis, so the slug collapses all of them into one id.
_ACCESSOR_NAMED = re.compile(r"_GET[A-Z]")


def names_an_accessor(var_id: str) -> bool:
    """Is this id named after the call that read it, not after what was read?"""
    return bool(_ACCESSOR_NAMED.search(str(var_id)))


def _uniq(seq: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for x in seq:
        x = x.strip()
        if x and x not in seen:
            seen.append(x)
    return seen


def _strip_ns(token: str) -> str:
    return token.split("::")[-1].strip()


def parse_opdef(path: str | Path, *, text_cache: dict[str, str] | None = None) -> list[ParamDecl]:
    """Structured read of the operator definition.

    Each `this->Input("x")` opens a chained-call block terminated by the next
    `this->`, so the block text is sliced between successive matches.
    """
    path = Path(path)
    key = str(path.resolve()).replace("\\", "/")
    if text_cache is not None and key in text_cache:
        text = text_cache[key]
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(MEMBER_RE.finditer(text))
    out: list[ParamDecl] = []
    seen_of_kind: dict[str, int] = {}
    for i, m in enumerate(matches):
        kind = m.group(1).lower()
        name = m.group(2)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end() : end]
        decl = ParamDecl(
            kind=kind,
            name=name,
            index=seen_of_kind.get(kind, 0),
            file=str(path).replace("\\", "/"),
            line=text[: m.start()].count("\n") + 1,
        )
        pt = PARAM_TYPE_RE.search(block)
        decl.param_type = pt.group(1) if pt else ""
        dt = DATATYPE_RE.search(block)
        if dt:
            row = [_strip_ns(t) for t in dt.group(1).split(",")]
            decl.dtype_row = [t for t in row if t]
            decl.dtypes = _uniq(row)
        fm = FORMAT_RE.search(block)
        if fm:
            decl.formats = _uniq(_strip_ns(t) for t in fm.group(1).split(","))
        if kind == "attr":
            av = ATTR_VALUE_RE.search(block)
            if av:
                decl.value_type = av.group(1)
                decl.default = av.group(2).strip()
        # Repeated names (an operator may re-declare a param) keep the first,
        # and do not advance the position: a re-declaration is the same slot.
        if not any(d.kind == kind and d.name == name for d in out):
            seen_of_kind[kind] = decl.index + 1
            out.append(decl)
    return out


class VariableModel:
    """All named variables for one operator, plus the atom → variable mapping."""

    def __init__(self) -> None:
        self.variables: dict[str, VarSpec] = {}
        self.params: dict[str, ParamDecl] = {}
        self.unmapped_atoms: list[str] = []
        # Enum members and `constexpr` names → integer literals. Used when a
        # leaf is a named constant (`ConstAxisTemplateNum::NUM128`, `ROPE_D_192`)
        # so comparisons can be decided rather than left as opaque strings.
        self.named_constants: dict[str, int] = {}
        # Locked platform profile (CANN INI); used to fold GetCoreNumAic → Const.
        self.platform_profile: Any = None

    def lookup_constant(self, symbol: str) -> int | None:
        if not symbol:
            return None
        got = self.named_constants.get(symbol)
        if got is not None:
            return got
        return self.named_constants.get(symbol.split("::")[-1])

    # -- construction ------------------------------------------------------
    def add(self, spec: VarSpec) -> VarSpec:
        existing = self.variables.get(spec.var_id)
        if existing is None:
            self.variables[spec.var_id] = spec
            return spec
        # Later evidence narrows the domain; it never widens a proven one.
        if spec.domain.lo is not None and existing.domain.lo is None:
            existing.domain.lo = spec.domain.lo
        if spec.domain.hi is not None and existing.domain.hi is None:
            existing.domain.hi = spec.domain.hi
        if spec.domain.values and not existing.domain.values:
            existing.domain.values = list(spec.domain.values)
            existing.domain.completeness = spec.domain.completeness
        for ev in spec.evidence:
            if ev.id not in {e.id for e in existing.evidence}:
                existing.evidence.append(ev)
        # A merge never un-merges: one occurrence that cannot say which value it
        # means is enough to make the id ambiguous for every reader.
        existing.identity_merged = existing.identity_merged or spec.identity_merged
        return existing

    def get(self, var_id: str) -> VarSpec | None:
        return self.variables.get(var_id)

    def mark_identity_merged(self, var_id: str) -> None:
        """Record that this id stands for more than one value.

        Separate from `add` because the caller often finds the variable already
        declared and skips construction entirely.
        """
        spec = self.variables.get(var_id)
        if spec is not None:
            spec.identity_merged = True

    def operand_names(self) -> dict[str, list[str]]:
        """Operand names per kind, in declaration order.

        Position is how the host picks one: `GetInputShape(1)` is the second
        declared input. Without this mapping a positional accessor cannot say
        which tensor it read, and every tensor collapses onto one variable.
        """
        out: dict[str, list[str]] = {}
        for key in self.params:
            kind, _, name = key.partition(":")
            out.setdefault(kind, []).append(name)
        return out

    # -- atom mapping ------------------------------------------------------
    def var_id_for(
        self,
        root: str,
        symbol: str,
        index: int | None = None,
        reads: str | None = None,
    ) -> str | None:
        """Map a resolved atom onto a declared variable id.

        Returns None for roots that carry no solver variable (constants, loop
        induction, kernel builtins), which the predicate layer treats as
        非可控 leaves rather than as failures.
        """
        if root in PLATFORM_VARS:
            return PLATFORM_VARS[root]
        if root in SESSION_VARS:
            return SESSION_VARS[root]
        sym = slug(symbol or "")
        if not sym:
            return None
        if root == "INPUT_SHAPE":
            if index is not None:
                return f"VAR_SHAPE_{sym}_D{index}"
            # A rank and an element count are both read off the whole shape,
            # and sharing one variable made `GetDimNum() != 4` and
            # `GetShapeSize() == 0` speak about the same unknown.
            return f"VAR_RANK_{sym}" if reads == "rank" else f"VAR_SHAPE_{sym}"
        if root == "INPUT_DTYPE":
            return f"VAR_DTYPE_{sym}"
        if root == "INPUT_FORMAT":
            return f"VAR_FORMAT_{sym}"
        if root == "INPUT_VALUE":
            return f"VAR_VALUE_{sym}"
        if root == "OPTIONAL_INPUT_PRESENCE":
            return f"VAR_OPT_{sym}"
        if root == "ATTRIBUTE":
            return f"VAR_ATTR_{sym}"
        if root == "SESSION_OPTION":
            return f"VAR_SESSION_{sym}" if sym else SESSION_VARS["SESSION_OPTION"]
        if root == "TILING_KEY":
            return f"VAR_KEY_{sym}"
        if root == "COMPILE_INFO":
            return f"VAR_COMPILE_{sym}"
        # Host tiling-data / params fields that hold an enum across several
        # guarded writes (layoutType, deterSparseType, pseOptional, …).
        if root == "TILING_DATA":
            return f"VAR_TDF_{sym}"
        return None

    def declare_on_demand(
        self, var_id: str, root: str, index: int | None = None
    ) -> VarSpec:
        """Mint a variable referenced by a guard but absent from the opdef.

        Shape dimensions are the common case: the definition lists no rank, so
        `VAR_SHAPE_QUERY_D2` only becomes known when a guard reads it. The
        domain stays open, which is the honest statement.

        `index` is the axis the accessor named, and it decides the lower bound:
        an axis of a tensor that is there has a length of at least one, but a
        shape read without an axis is the shape as a whole — its rank, or its
        element count — and zero is a value both take. Zero is in fact the
        value the source looks for: the standard way to ask whether an optional
        input was passed is `GetStorageShape().GetDimNum() == 0`, so a lower
        bound of one on those variables makes the test false by construction
        and deletes the absent-tensor branch it selects.
        """
        existing = self.variables.get(var_id)
        if existing is not None:
            return existing
        if root in ("INPUT_DTYPE", "INPUT_FORMAT", "TILING_DATA"):
            value_type = "enum"
        elif root in ("OPTIONAL_INPUT_PRESENCE", "SESSION_OPTION"):
            value_type = "bool"
        else:
            value_type = "int"
        lo = (1 if index is not None else 0) if root == "INPUT_SHAPE" else None
        return self.add(
            VarSpec(
                var_id=var_id,
                name=var_id,
                value_type=value_type,
                domain=Domain(
                    var_id=var_id,
                    value_type=value_type,
                    lo=lo,
                    completeness="open",
                    source="guard_reference",
                ),
                origin="guard_reference",
                description="referenced by a guard; domain not proven by the definition",
                identity_merged=names_an_accessor(var_id),
            )
        )

    # -- export ------------------------------------------------------------
    def to_tg_variables(self) -> dict[str, dict[str, Any]]:
        return {vid: spec.to_tg_entry() for vid, spec in sorted(self.variables.items())}

    def domains(self) -> dict[str, Domain]:
        return {vid: spec.domain for vid, spec in self.variables.items()}


def _dtype_domain(decl: ParamDecl) -> Domain:
    var_id = f"VAR_DTYPE_{slug(decl.name)}"
    return Domain(
        var_id=var_id,
        value_type="enum",
        values=list(decl.dtypes),
        completeness="closed" if decl.dtypes else "open",
        source="opdef.DataType",
    )


def _format_domain(decl: ParamDecl) -> Domain:
    var_id = f"VAR_FORMAT_{slug(decl.name)}"
    return Domain(
        var_id=var_id,
        value_type="enum",
        values=list(decl.formats),
        completeness="closed" if decl.formats else "open",
        source="opdef.Format",
    )


def _attr_spec(decl: ParamDecl, enums: dict[str, dict[str, int]]) -> VarSpec:
    var_id = f"VAR_ATTR_{slug(decl.name)}"
    value_type = ATTR_TYPE_MAP.get(decl.value_type, "int")
    values: list[Any] = []
    completeness = "open"
    source = f"opdef.Attr.{decl.value_type or 'unknown'}"

    # `sparse_mode` <-> `enum class SparseMode`: the enum proves the domain.
    enum_name = "".join(p.title() for p in decl.name.split("_"))
    members = enums.get(enum_name)
    if members:
        values = sorted(members.values())
        value_type = "int"
        completeness = "closed"
        source = f"enum class {enum_name}"
    elif value_type == "bool":
        values = [False, True]
        completeness = "closed"

    domain = Domain(
        var_id=var_id,
        value_type=value_type,
        values=values,
        completeness=completeness,
        source=source,
    )
    return VarSpec(
        var_id=var_id,
        name=decl.name,
        value_type=value_type,
        domain=domain,
        origin="opdef_attr",
        evidence=[decl.evidence()],
        description=f"attribute default={decl.default}" if decl.default else "",
    )


def _key_spec(dim, header: str) -> VarSpec:
    var_id = named_id("Variable", f"KEY_{dim.name}")
    raw = dim.value_domain
    values: list[Any] = []
    for v in raw:
        try:
            values.append(int(v, 0))
        except (TypeError, ValueError):
            values.append(str(v))
    value_type = "bool" if dim.kind == "BOOL" else ("int" if dim.kind == "UINT" else "enum")
    if value_type == "bool":
        values = [False, True]
    domain = Domain(
        var_id=var_id,
        value_type=value_type,
        values=values,
        # The DSL enumerates every legal encoding, so the domain is exhaustive.
        completeness="closed" if values else "open",
        source="ASCENDC_TPL_*_DECL",
    )
    return VarSpec(
        var_id=var_id,
        name=dim.name,
        value_type=value_type,
        domain=domain,
        origin="tpl_dim",
        evidence=[Evidence.at(header, 1, snippet=f"ASCENDC_TPL {dim.kind} {dim.name}")],
        description=f"TilingKey bits {dim.bit_lo}-{dim.bit_hi}",
    )


def infer_bounds_from_guards(
    model: VariableModel, nodes, resolver
) -> list[tuple[str, str, int]]:
    """Read integer bounds off validation branches.

    `if (dimNum > 65535) { return GRAPH_FAILED; }` proves an upper bound the
    definition never states. Only branches that actually reject are used, so a
    dispatch condition like `if (s1 > 1024) useLargeTemplate()` is not mistaken
    for a legality limit.
    """
    found: list[tuple[str, str, int]] = []
    for node in nodes:
        if not VALIDATION_FN_RE.search(node.function or "") and not FAILURE_RE.search(
            node.snippet or ""
        ):
            continue
        for m in _comparisons(node.condition or ""):
            hit = _apply_bound(model, resolver, node, m)
            if hit:
                found.append(hit)
    for spec in model.variables.values():
        if spec.domain.lo is not None and spec.domain.hi is not None:
            spec.domain.completeness = "closed"
    return found


def _comparisons(condition: str):
    """Every simple `<expr> <op> <int>` conjunct of a rejection condition.

    Validation guards are usually `a > MAX || b < MIN`, so matching the whole
    condition finds nothing; each side still proves its own bound.
    """
    for part in re.split(r"\|\||&&", condition):
        m = SIMPLE_CMP_RE.match(part.strip().strip("()"))
        if m:
            yield m


def _apply_bound(model, resolver, node, m) -> tuple[str, str, int] | None:
    try:
        literal = int(m.group("rhs"))
    except ValueError:
        return None
    res = resolver.resolve(m.group("lhs"))
    atoms = [a for a in res.atoms if a.root and a.root != "CONSTANT"]
    if len(atoms) != 1:
        return None
    atom = atoms[0]
    var_id = model.var_id_for(
        atom.root,
        atom.symbol,
        getattr(atom, "index", None),
        getattr(atom, "reads", None),
    )
    if not var_id:
        return None
    spec = model.get(var_id)
    if spec is None or spec.value_type != "int":
        return None
    op = m.group("op")
    # The guard describes the *rejected* region, so the legal bound is its
    # complement: rejecting `x > N` proves `x <= N`.
    if op == ">":
        _tighten(spec.domain, hi=literal)
    elif op == ">=":
        _tighten(spec.domain, hi=literal - 1)
    elif op == "<":
        _tighten(spec.domain, lo=literal)
    elif op == "<=":
        _tighten(spec.domain, lo=literal + 1)
    else:
        return None
    if "validation_branch" not in spec.domain.source:
        spec.domain.source = f"{spec.domain.source}+validation_branch"
    spec.evidence.append(
        Evidence.at(node.file, node.line, snippet=(node.condition or "")[:120])
    )
    return (var_id, op, literal)


def _tighten(domain: Domain, *, lo: int | None = None, hi: int | None = None) -> None:
    if lo is not None:
        domain.lo = lo if domain.lo is None else max(domain.lo, lo)
    if hi is not None:
        domain.hi = hi if domain.hi is None else min(domain.hi, hi)


CONSTEXPR_INT_RE = re.compile(
    r"\b(?:static\s+)?constexpr\s+(?:static\s+)?(?:const\s+)?"
    r"(?:u?int(?:8|16|32|64)_t|size_t|unsigned(?:\s+int)?|int)\s+"
    r"([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*[uUlL]*\s*;"
)


# `ge::DataType`, transcribed from CANN metadef `graph/c_types.h`. Host tiling
# compares `GetDataType()` against these, so without them every dtype guard is
# an unreadable symbol.
#
# Transcribed rather than parsed: `graph/types.h` spells each member as
# `DT_FLOAT = ::C_DT_FLOAT`, which `parse_enums` cannot evaluate, and its
# fallback (keep counting up) would hand out wrong values at the reserved gap
# after 4. Values here are the literals from `c_types.h`; an operator that
# redefines one of these names wins, because source-scanned constants are
# applied on top.
GE_DATA_TYPE: dict[str, int] = {
    "DT_FLOAT": 0,
    "DT_FLOAT16": 1,
    "DT_INT8": 2,
    "DT_INT32": 3,
    "DT_UINT8": 4,
    # 5 is reserved and has no name
    "DT_INT16": 6,
    "DT_UINT16": 7,
    "DT_UINT32": 8,
    "DT_INT64": 9,
    "DT_UINT64": 10,
    "DT_DOUBLE": 11,
    "DT_BOOL": 12,
    "DT_STRING": 13,
    "DT_DUAL_SUB_INT8": 14,
    "DT_DUAL_SUB_UINT8": 15,
    "DT_COMPLEX64": 16,
    "DT_COMPLEX128": 17,
    "DT_QINT8": 18,
    "DT_QINT16": 19,
    "DT_QINT32": 20,
    "DT_QUINT8": 21,
    "DT_QUINT16": 22,
    "DT_RESOURCE": 23,
    "DT_STRING_REF": 24,
    "DT_DUAL": 25,
    "DT_VARIANT": 26,
    "DT_BF16": 27,
    "DT_UNDEFINED": 28,
    "DT_INT4": 29,
    "DT_UINT1": 30,
    "DT_INT2": 31,
    "DT_UINT2": 32,
    "DT_COMPLEX32": 33,
    "DT_HIFLOAT8": 34,
    "DT_FLOAT8_E5M2": 35,
    "DT_FLOAT8_E4M3FN": 36,
    "DT_FLOAT8_E8M0": 37,
    "DT_FLOAT6_E3M2": 38,
    "DT_FLOAT6_E2M3": 39,
    "DT_FLOAT4_E2M1": 40,
    "DT_FLOAT4_E1M2": 41,
    "DT_HIFLOAT4": 42,
}


def _named_constants_from(
    enums: dict[str, dict[str, int]],
    *,
    header_texts: Iterable[str] = (),
) -> dict[str, int]:
    """Flatten enum members and constexpr ints into a symbol → value map."""
    out: dict[str, int] = dict(GE_DATA_TYPE)
    for member, value in GE_DATA_TYPE.items():
        out[f"ge::{member}"] = value
    for ename, members in enums.items():
        for member, value in members.items():
            out[member] = value
            out[f"{ename}::{member}"] = value
    for text in header_texts:
        for m in CONSTEXPR_INT_RE.finditer(text or ""):
            out[m.group(1)] = int(m.group(2))
    return out


def build_variable_model(
    *,
    opdef_path: str | Path | None = None,
    tpl_schema=None,
    tpl_header: str | Path = "",
    enums: dict[str, dict[str, int]] | None = None,
    header_texts: Iterable[str] | None = None,
    text_cache: dict[str, str] | None = None,
) -> VariableModel:
    """Assemble the variable layer from every evidence source available."""
    model = VariableModel()
    enums = enums or {}
    model.named_constants = _named_constants_from(enums, header_texts=header_texts or ())

    if opdef_path:
        for decl in parse_opdef(opdef_path, text_cache=text_cache):
            model.params[f"{decl.kind}:{decl.name}"] = decl
            if decl.kind == "attr":
                model.add(_attr_spec(decl, enums))
                continue
            ev = decl.evidence()
            if decl.dtypes:
                d = _dtype_domain(decl)
                model.add(
                    VarSpec(
                        var_id=d.var_id,
                        name=decl.name,
                        value_type="enum",
                        domain=d,
                        origin=f"opdef_{decl.kind}",
                        evidence=[ev],
                    )
                )
            if decl.formats:
                d = _format_domain(decl)
                model.add(
                    VarSpec(
                        var_id=d.var_id,
                        name=decl.name,
                        value_type="enum",
                        domain=d,
                        origin=f"opdef_{decl.kind}",
                        evidence=[ev],
                    )
                )
            if decl.kind == "input" and decl.is_optional:
                var_id = f"VAR_OPT_{slug(decl.name)}"
                model.add(
                    VarSpec(
                        var_id=var_id,
                        name=decl.name,
                        value_type="bool",
                        domain=Domain(
                            var_id=var_id,
                            value_type="bool",
                            values=[False, True],
                            completeness="closed",
                            source="opdef.ParamType(OPTIONAL)",
                        ),
                        origin="opdef_optional_input",
                        evidence=[ev],
                    )
                )

    if tpl_schema is not None:
        header = str(tpl_header or "")
        for dim in tpl_schema.dims:
            model.add(_key_spec(dim, header))

    for root, var_id in PLATFORM_VARS.items():
        value_type = "enum" if root == "PLATFORM_ARCH" else "int"
        model.add(
            VarSpec(
                var_id=var_id,
                name=root.lower(),
                value_type=value_type,
                domain=Domain(
                    var_id=var_id,
                    value_type=value_type,
                    completeness="open",
                    source="platform_context",
                ),
                origin="platform",
                evidence=[Evidence.at("<platform>", 0, snippet=root)],
                description="set by the compilation target, not by the test case",
            )
        )
    # Deterministic session option: closed bool, controllable by TG.
    det_id = SESSION_VARS["SESSION_OPTION"]
    model.add(
        VarSpec(
            var_id=det_id,
            name="deterministic",
            value_type="bool",
            domain=Domain(
                var_id=det_id,
                value_type="bool",
                values=[False, True],
                completeness="closed",
                source="session_option",
            ),
            origin="session",
            evidence=[Evidence.at("<session>", 0, snippet="GetDeterministic")],
            description="context_->GetDeterministic(); session option, not op input",
        )
    )
    return model


def apply_platform_profile(model: VariableModel, profile) -> VariableModel:
    """Close PLATFORM_* domains from a locked CANN platform_config INI."""
    if profile is None:
        return model
    model.platform_profile = profile
    ev = Evidence.at(profile.ini_path, 1, snippet=profile.soc_version)

    def _close(var_id: str, value: int, *, name: str) -> None:
        spec = model.get(var_id)
        if spec is None:
            model.add(
                VarSpec(
                    var_id=var_id,
                    name=name,
                    value_type="int",
                    domain=Domain(
                        var_id=var_id,
                        value_type="int",
                        values=[value],
                        completeness="closed",
                        source=f"platform_config:{profile.soc_version}.ini",
                    ),
                    origin="platform",
                    evidence=[ev],
                )
            )
            return
        spec.domain.values = [value]
        spec.domain.completeness = "closed"
        spec.domain.source = f"platform_config:{profile.soc_version}.ini"
        if ev.id not in {e.id for e in spec.evidence}:
            spec.evidence.append(ev)

    _close(PLATFORM_VARS["PLATFORM_CORE_COUNT"], profile.aic_num, name="aic_num")
    _close(PLATFORM_VARS["PLATFORM_AIV_COUNT"], profile.vector_core_cnt, name="aiv_num")
    _close(PLATFORM_VARS["PLATFORM_L2_SIZE"], profile.l2_size, name="l2_size")
    if profile.memory_size:
        _close(PLATFORM_VARS["PLATFORM_MEMORY_SIZE"], profile.memory_size, name="memory_size")
    arch_id = PLATFORM_VARS["PLATFORM_ARCH"]
    arch = model.get(arch_id)
    if arch is not None:
        arch.domain.values = [profile.npu_arch, profile.soc_version]
        arch.domain.completeness = "closed"
        arch.domain.source = f"platform_config:{profile.soc_version}.ini"
        if ev.id not in {e.id for e in arch.evidence}:
            arch.evidence.append(ev)
    # Named aliases used in host expressions.
    model.named_constants["aicNum"] = profile.aic_num
    model.named_constants["aivNum"] = profile.vector_core_cnt
    model.named_constants["l2CacheSize"] = profile.l2_size
    model.named_constants["l2_size"] = profile.l2_size
    return model
