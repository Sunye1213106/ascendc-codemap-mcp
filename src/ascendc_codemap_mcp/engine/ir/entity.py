# -*- coding: utf-8 -*-
"""CodeMap entity ontology (unified node kinds)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityKind(str, Enum):
    FILE = "FILE"
    FUNCTION = "FUNCTION"
    METHOD = "METHOD"
    VARIABLE = "VARIABLE"
    FIELD = "FIELD"
    TYPE = "TYPE"

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    MACRO = "MACRO"
    COMPILE_VAR = "COMPILE_VAR"

    TEMPLATE = "TEMPLATE"
    TEMPLATE_ARG = "TEMPLATE_ARG"
    TEMPLATE_INSTANCE = "TEMPLATE_INSTANCE"

    BRANCH = "BRANCH"
    PREDICATE = "PREDICATE"

    TILING_KEY = "TILING_KEY"
    TILING_DATA = "TILING_DATA"
    TILING_FIELD = "TILING_FIELD"

    KERNEL = "KERNEL"
    ARCH = "ARCH"
    BUILD_VARIANT = "BUILD_VARIANT"

    # Kernel Root Trace anchors (source sites + navigation — not execution order).
    OPERATION = "OPERATION"
    BUFFER = "BUFFER"
    # AscendC::Reg / MicroAPI register-file objects (RegTensor, MaskReg, ...).
    REGISTER = "REGISTER"
    # Conservative synchronization facts; these are identities, not schedules.
    PIPE = "PIPE"
    EVENT = "EVENT"
    QUEUE = "QUEUE"

    # Escapes for legacy KB kinds during adaptation.
    OTHER = "OTHER"


@dataclass
class Entity:
    """One node in the unified CodeMap."""

    id: str
    kind: EntityKind | str
    name: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)
    file: str = ""
    line_start: int = 0
    line_end: int = 0
    status: str = "extracted"
    confidence: float = 1.0

    def kind_name(self) -> str:
        k = self.kind
        return k.value if isinstance(k, EntityKind) else str(k)

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.attrs)
        out.update(
            {
                "id": self.id,
                "kind": self.kind_name(),
                "name": self.name,
                "status": self.status,
                "confidence": round(float(self.confidence), 4),
                "file": self.file,
                "line_start": int(self.line_start),
                "line_end": int(self.line_end),
            }
        )
        return out