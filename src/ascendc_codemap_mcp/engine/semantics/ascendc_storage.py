# -*- coding: utf-8 -*-
"""AscendC / CANN storage-type catalog for Kernel Root Trace.

Terminal storage / register roots and known framework wrapper contracts.
Project-specific policy / selector class names are never catalogued here —
they must be discovered via source composition (WRAPS / ALIASES).
"""

from __future__ import annotations

import re
from typing import Any

# type spelling (as written in operator / CANN headers) → register_class
ASCENDC_REGISTER_TYPES: dict[str, str] = {
    "RegTensor": "VREG",
    "MaskReg": "MASK_REG",
    "UnalignReg": "UNALIGN_REG",
    "UnalignRegForLoad": "UNALIGN_REG",
    "UnalignRegForStore": "UNALIGN_REG",
    "AddrReg": "ADDR_REG",
}

# Direct AscendC tensor/queue types (terminal storage roots).
ASCENDC_BUFFER_TYPES: frozenset[str] = frozenset(
    {
        "LocalTensor",
        "GlobalTensor",
        "TBuf",
        "TQue",
        "TQueBind",
        "TBufPool",
    }
)

# TQue / TQueBind methods whose bodies live in CANN (kernel_tquebind_impl.h).
# EnQue internally SetFlag, DeQue internally WaitFlag — keep the TQue root;
# do not unfold to user-level flag pairing. InitBuffer is TPipe, not TQue.
TQUE_METHOD_BRIDGES: dict[str, tuple[str, str]] = {
    "EnQue": ("EnQue", "MEMORY_API"),
    "DeQue": ("DeQue", "MEMORY_API"),
    "AllocTensor": ("AllocTensor", "MEMORY_API"),
    "FreeTensor": ("FreeTensor", "MEMORY_API"),
}

# TPipe / TBufPool methods (kernel_tpipe.h). Receiver is the pipe, TQue/TBuf is an argument.
TPIPE_METHOD_BRIDGES: dict[str, tuple[str, str]] = {
    "InitBuffer": ("InitBuffer", "MEMORY_API"),
    "Destroy": ("Destroy", "MEMORY_API"),
    "FetchEventID": ("FetchEventID", "SYNC"),
    "AllocEventID": ("AllocEventID", "SYNC"),
    "ReleaseEventID": ("ReleaseEventID", "SYNC"),
    "AllocCrossSyncId": ("AllocCrossSyncId", "SYNC"),
}

# Free-function leftover-UB pop (kernel_tpipe.h / kernel_pop_stack_buffer.h).
# Does not go through TPipe::InitBuffer; dest LocalTensor/TBuf is still allocated.
STACK_BUFFER_CALLEES: frozenset[str] = frozenset({"PopStackBuffer"})

# Shared-UB setup between cores (kernel_tpipe.h friends). Not InitBuffer / TQue.
SHARE_BUFFER_CALLEES: frozenset[str] = frozenset({"InitShareBufStart", "InitShareBufEnd"})

# GlobalTensor / LocalTensor methods (kernel_tensor.h). Unique CANN spellings.
# Get is *not* here: bare Get is often a project Selector/Policy.
TENSOR_METHOD_BRIDGES: dict[str, tuple[str, str]] = {
    "SetGlobalBuffer": ("SetGlobalBuffer", "MEMORY_API"),
    "GetPhyAddr": ("GetPhyAddr", "MEMORY_API"),
    "GetValue": ("GetValue", "MEMORY_API"),
    "SetValue": ("SetValue", "MEMORY_API"),
    "Destroy": ("Destroy", "MEMORY_API"),
    "ReinterpretCast": ("ReinterpretCast", "MEMORY_API"),
}

# Token names only — is_non_storage_type matches these as whole identifiers.
# Do not put "Mutex" here: it is a substring of project MutexBuffer.
ASCENDC_NON_STORAGE_TYPES: frozenset[str] = frozenset({"TPipe", "GroupBarrier", "TQueSync"})

#: The tier a register lives in. A register is storage one level in from UB, not
#: a separate concept: a vector instruction reads and writes registers the way
#: DataCopy reads and writes UB, and `is_storage_type_text` below has always
#: accepted both spellings as storage. This was the one tier missing from the
#: vocabulary, so register declarations carried no `memory_space` at all and sat
#: outside every analysis keyed on which level of the hierarchy a value is in.
#:
#: The tier is one value, while `ASCENDC_REGISTER_TYPES` keeps the finer
#: register-file distinction (VREG / MASK_REG / ...) under `register_class`.
REGISTER_MEMORY_SPACE = "REG"

#: The physical tiers a value can live in, as CANN enumerates them in
#: ``Hardware`` (``basic_api/impl/utils/common_types.h``): GM, UB, L1, L0A,
#: L0B, L0C, BIAS, FIXBUF. ``BIAS`` (bias table) and ``FIXBUF`` (fixpipe
#: buffer) were absent, while ``C2`` was listed as if it were a tier — it is a
#: *TPosition* name, and the tier behind it is BIAS on every architecture this
#: engine supports. Anything keyed on the tier therefore mis-bucketed the
#: cube-side buffers.
HARDWARE_SPACES: frozenset[str] = frozenset(
    {"GM", "UB", "L1", "L0A", "L0B", "L0C", "BIAS", "FIXBUF"}
)

#: The hardware tiers plus the logical buckets the engine adds on top: a queue
#: is a TQue whose tier is not fixed until InitBuffer, WORKSPACE is GM the host
#: allocated, and REG is the register file one level in from UB.
BUFFER_MEMORY_SPACES: frozenset[str] = frozenset(
    HARDWARE_SPACES | {"QUEUE", "WORKSPACE", REGISTER_MEMORY_SPACE}
)

# ---------------------------------------------------------------------------
# TPosition → physical tier, transcribed from CANN ``GetPhyType``
# (cann-asc-devkit ascendc/include/basic_api/impl/kernel_event.h).
#
# Four positions — C1, C2, CO2, C2PIPE2GM — sit inside an ``#if __NPU_ARCH__``
# chain and mean different tiers on different chips, which is why they are
# split out below instead of living in one flat table. The rest are decided
# outside the chain and are the same everywhere.
# ---------------------------------------------------------------------------

# Positions CANN resolves identically on every architecture. Anything CANN
# leaves unhandled keeps the ``Hardware hard = Hardware::UB`` initialiser, so
# the VEC* family and C2PIPE2LOCAL are UB by falling through rather than by a
# branch. SPM and SHM are one enumerator (``SHM = SPM``) and GetPhyType sends
# it to L1 — this table used to say UB for both.
_TPOSITION_FIXED: dict[str, str] = {
    "GM": "GM",
    "A1": "L1",
    "B1": "L1",
    "A2": "L0A",
    "B2": "L0B",
    "CO1": "L0C",
    "TSCM": "L1",
    "SPM": "L1",
    "SHM": "L1",
    "VECIN": "UB",
    "VECOUT": "UB",
    "VECCALC": "UB",
    "LCM": "UB",  # ``LCM = VECCALC``
    "C2PIPE2LOCAL": "UB",  # no branch on any arch — falls to the UB default
}

# __NPU_ARCH__ 2201 / 3003 / 5102 / 3510 / 3113 all agree, and every
# architecture this engine builds for (arch22=2201, arch35=3510) is in that
# set, so this doubles as the answer when the architecture is unknown.
_TPOSITION_ARCH_CURRENT: dict[str, str] = {
    "C1": "L1",
    "C2": "BIAS",
    "CO2": "GM",
    "C2PIPE2GM": "FIXBUF",
}

# 1001 / 2002 predate the bias table and the fixpipe buffer.
_TPOSITION_ARCH_LEGACY: dict[str, str] = {
    "C1": "UB",
    "C2": "L0C",
    "CO2": "UB",
    "C2PIPE2GM": "UB",
}

# 3002 and 3102 hand-wave fewer positions than 3510 does; the ones they omit
# fall through to the UB initialiser rather than to the 3510 answer.
_TPOSITION_ARCH_3002: dict[str, str] = {
    "C1": "L1",
    "C2": "BIAS",
    "CO2": "UB",
    "C2PIPE2GM": "FIXBUF",
}
_TPOSITION_ARCH_3102: dict[str, str] = {
    "C1": "L1",
    "C2": "BIAS",
    "CO2": "UB",
    "C2PIPE2GM": "UB",
}

TPOSITION_ARCH_OVERRIDES: dict[int, dict[str, str]] = {
    1001: _TPOSITION_ARCH_LEGACY,
    2002: _TPOSITION_ARCH_LEGACY,
    2201: _TPOSITION_ARCH_CURRENT,
    3003: _TPOSITION_ARCH_CURRENT,
    3002: _TPOSITION_ARCH_3002,
    3102: _TPOSITION_ARCH_3102,
    5102: _TPOSITION_ARCH_CURRENT,
    3510: _TPOSITION_ARCH_CURRENT,
    3113: _TPOSITION_ARCH_CURRENT,
}

#: Flat view for the architectures this engine targets. Kept as the default so
#: callers with no architecture in hand still get the arch22/arch35 answer.
TPOSITION_TO_SPACE: dict[str, str] = {**_TPOSITION_FIXED, **_TPOSITION_ARCH_CURRENT}

# ``BufferType`` is a *project* enum, not a CANN one, so it is recorded here as
# the TPosition it denotes and the tier is then read off GetPhyType above —
# one source of truth instead of a second hand-maintained tier table that
# drifted (it mapped C2 to a "C2" tier that no hardware has).
#
# ops-transformer spells it two ways, and both are transcribed from the
# operator's own constexpr position mapping:
#   attention/common/op_kernel/buffer.h  BufferInfo<>::GetTPosition()
#   common/include/op_kernel/mem.h       GetPosition<BufferType_>()
BUFFER_TYPE_TO_TPOSITION: dict[str, str] = {
    "L1": "A1",
    "L0A": "A2",
    "L0B": "B2",
    "L0C": "CO1",
    "UB": "VECIN",
    "GM": "GM",
    "C2": "C2",
    "ASCEND_UB": "VECIN",
    "ASCEND_CB": "A1",
    "ASCEND_L0A": "A2",
    "ASCEND_L0B": "B2",
    "ASCEND_L0C": "CO1",
}

_CXX_KEYWORDS = frozenset(
    {
        "this",
        "true",
        "false",
        "nullptr",
        "return",
        "sizeof",
        "alignof",
        "if",
        "else",
        "for",
        "while",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "const",
        "constexpr",
        "static",
        "volatile",
        "typedef",
        "using",
        "namespace",
        "class",
        "struct",
        "enum",
        "template",
        "typename",
        "public",
        "private",
        "protected",
        "virtual",
        "inline",
        "new",
        "delete",
        "operator",
        "Min",
        "Max",
        "Ceil",
        "AlignUp",
        "AlignDown",
    }
)


def register_class_from_type(type_text: str) -> str | None:
    text = str(type_text or "")
    for spelling, klass in ASCENDC_REGISTER_TYPES.items():
        if spelling in text:
            return klass
    return None


def is_buffer_type(type_text: str) -> bool:
    text = str(type_text or "")
    return any(t in text for t in ASCENDC_BUFFER_TYPES)


def is_storage_wrapper_type(type_text: str) -> bool:
    """Project wrappers are proven from source WRAPS, never by class spelling."""
    del type_text
    return False


def is_non_storage_type(type_text: str) -> bool:
    text = str(type_text or "")
    return any(re.search(rf"(?:^|::)\b{re.escape(name)}\b", text) is not None for name in ASCENDC_NON_STORAGE_TYPES)


def is_storage_type_text(type_text: str) -> bool:
    return bool(register_class_from_type(type_text) or is_buffer_type(type_text))


def is_valid_storage_name(name: str) -> bool:
    text = str(name or "").strip()
    if not text or text in _CXX_KEYWORDS:
        return False
    return text.isidentifier()


def _spells_enumerator(text: str, enum_name: str, member: str) -> bool:
    """``BufferType::L1`` must not also match ``BufferType::L1Extra``."""
    return re.search(rf"\b{enum_name}::{re.escape(member)}\b", text) is not None


def tposition_from_type_text(type_text: str) -> str | None:
    """Return the TPosition/QuePosition token (VECIN/VECOUT/…) when spelled in the type."""
    text = str(type_text or "")
    for pos in TPOSITION_TO_SPACE:
        if _spells_enumerator(text, "TPosition", pos) or _spells_enumerator(
            text, "QuePosition", pos
        ):
            return pos
    return None


def tposition_memory_space(tposition: str, architecture: str = "") -> str | None:
    """The physical tier CANN ``GetPhyType`` gives *tposition* on this chip.

    Four positions are decided inside an ``#if __NPU_ARCH__`` chain, so the
    architecture is part of the question. An unknown architecture answers with
    the arch22/arch35 row, which is what every chip this engine builds for
    uses; only the pre-bias-table 1001/2002 parts differ.
    """
    pos = str(tposition or "").strip()
    if not pos:
        return None
    fixed = _TPOSITION_FIXED.get(pos)
    if fixed is not None:
        return fixed
    from ascendc_codemap_mcp.engine.semantics.ascendc_vf import architecture_npu_arch

    npu = architecture_npu_arch(architecture) if architecture else None
    row = TPOSITION_ARCH_OVERRIDES.get(npu or 0, _TPOSITION_ARCH_CURRENT)
    return row.get(pos)


def memory_space_from_type_text(type_text: str, architecture: str = "") -> str | None:
    """Resolve memory_space from CANN/AscendC position template args — not names."""
    text = str(type_text or "")
    for enum_name, pos in BUFFER_TYPE_TO_TPOSITION.items():
        if _spells_enumerator(text, "BufferType", enum_name):
            return tposition_memory_space(pos, architecture)
    pos = tposition_from_type_text(text)
    if pos is not None:
        return tposition_memory_space(pos, architecture)
    if "GlobalTensor" in text:
        return "GM"
    return None


#: Template-parameter types that name a tier or a pipe position. A parameter
#: declared with one of these is what a buffer takes its memory space from;
#: any other parameter (``typename T``, ``SyncType syncType``) is not.
TIER_PARAMETER_TYPES: frozenset[str] = frozenset({"BufferType", "TPosition", "QuePosition"})

_TEMPLATE_PARAM_RE = re.compile(r"^\s*(?:const\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*$")


def _split_template_args(text: str) -> list[str]:
    """Top-level comma split of the first ``<...>`` group, nesting-aware."""
    start = text.find("<")
    if start < 0:
        return []
    depth, buf, out = 0, [], []
    for ch in text[start:]:
        if ch == "<":
            depth += 1
            if depth == 1:
                continue
        elif ch == ">":
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and ch == ",":
            out.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def template_parameter_types(header: str) -> dict[str, str]:
    """Map parameter name → declared type for a ``template <...>`` header.

    ``template <BufferType bufferType, SyncType syncType = SyncType::X>``
    yields ``{"bufferType": "BufferType", "syncType": "SyncType"}``. Type
    parameters (``typename T``) carry no tier and are skipped.
    """
    text = str(header or "").strip()
    if not text.startswith("template"):
        return {}
    out: dict[str, str] = {}
    for arg in _split_template_args(text):
        decl = arg.split("=", 1)[0]
        hit = _TEMPLATE_PARAM_RE.match(decl)
        if hit and hit.group(1) not in {"typename", "class"}:
            out[hit.group(2)] = hit.group(1)
    return out


def tier_template_parameter(decl_text: str, header: str) -> tuple[str, str] | None:
    """The template parameter a declaration takes its tier from, or None.

    ``MutexBuffer<bufferType, syncType> ping_`` has a memory space, but only
    once the enclosing template is instantiated, so no ``BufferType::`` token
    appears and the tier resolves to nothing. Reporting nothing makes an
    absent fact look like an unrecorded one: a reader took the silence for a
    gap and inferred the tier from the wrapped ``LocalTensor`` instead.

    The parameter is read off the enclosing ``template <...>`` header rather
    than guessed from argument position, because position does not tell a tier
    apart from an element type -- the first argument of ``LocalTensor<T>`` is
    not a tier, and the first argument of ``MutexBuffer<bufferType, …>`` is.
    """
    params = template_parameter_types(header)
    if not params:
        return None
    for arg in _split_template_args(str(decl_text or "")):
        declared = params.get(arg.strip())
        if declared in TIER_PARAMETER_TYPES:
            return arg.strip(), declared
    return None


def storage_root_kind_from_space(space: str) -> str:
    if space == "GM":
        return "GlobalTensor"
    if space == "QUEUE":
        return "TQue"
    return "LocalTensor"


def resolve_buffer_decl(type_text: str, architecture: str = "") -> dict[str, Any] | None:
    """Classify a decl type_text into storage metadata (no name heuristics)."""
    text = str(type_text or "")
    if not text:
        return None
    if register_class_from_type(text):
        return None
    if is_non_storage_type(text):
        return None
    wrapper = False
    if not any(t in text for t in ASCENDC_BUFFER_TYPES):
        return None
    space = memory_space_from_type_text(text, architecture) or "UNKNOWN"
    root = "LocalTensor" if wrapper else (
        "GlobalTensor"
        if "GlobalTensor" in text
        else (
            "TQue"
            if "TQue" in text
            else ("TBuf" if "TBuf" in text else storage_root_kind_from_space(space))
        )
    )
    if "LocalTensor" in text and not wrapper:
        root = "LocalTensor"
    return {
        "is_wrapper": wrapper,
        "memory_space": space,
        "storage_root_kind": root,
    }
