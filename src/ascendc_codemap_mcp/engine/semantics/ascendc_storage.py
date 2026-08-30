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

BUFFER_MEMORY_SPACES: frozenset[str] = frozenset(
    {"GM", "UB", "L1", "L0A", "L0B", "L0C", "QUEUE", "WORKSPACE", "C2", REGISTER_MEMORY_SPACE}
)

# CANN AscendC TPosition / QuePosition → logical memory space.
TPOSITION_TO_SPACE: dict[str, str] = {
    "GM": "GM",
    "VECIN": "UB",
    "VECOUT": "UB",
    "VECCALC": "UB",
    "A1": "L1",
    "B1": "L1",
    "C1": "L1",
    "A2": "L0A",
    "B2": "L0B",
    "CO1": "L0C",
    "CO2": "L0C",
    "C2": "C2",
    "LCM": "UB",
    "TSCM": "L1",
    "SPM": "UB",
    "SHM": "UB",
    "C2PIPE2GM": "GM",
    "C2PIPE2LOCAL": "UB",
}

# Common BufferType enums used with AscendC TPosition.
BUFFER_TYPE_TO_SPACE: dict[str, str] = {
    "L1": "L1",
    "L0A": "L0A",
    "L0B": "L0B",
    "L0C": "L0C",
    "UB": "UB",
    "GM": "GM",
    "C2": "C2",
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


def tposition_from_type_text(type_text: str) -> str | None:
    """Return the TPosition/QuePosition token (VECIN/VECOUT/…) when spelled in the type."""
    text = str(type_text or "")
    for pos in TPOSITION_TO_SPACE:
        if f"TPosition::{pos}" in text or f"QuePosition::{pos}" in text:
            return pos
    return None


def memory_space_from_type_text(type_text: str) -> str | None:
    """Resolve memory_space from CANN/AscendC position template args — not names."""
    text = str(type_text or "")
    for enum_name, space in BUFFER_TYPE_TO_SPACE.items():
        token = f"BufferType::{enum_name}"
        if token in text:
            return space
    for pos, space in TPOSITION_TO_SPACE.items():
        if f"TPosition::{pos}" in text or f"QuePosition::{pos}" in text:
            return space
    if "GlobalTensor" in text:
        return "GM"
    return None


def storage_root_kind_from_space(space: str) -> str:
    if space == "GM":
        return "GlobalTensor"
    if space == "QUEUE":
        return "TQue"
    return "LocalTensor"


def resolve_buffer_decl(type_text: str) -> dict[str, Any] | None:
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
    space = memory_space_from_type_text(text) or "UNKNOWN"
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
