# -*- coding: utf-8 -*-
"""CodeMap relation ontology (unified edge kinds)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RelationKind(str, Enum):
    DECLARES = "DECLARES"
    DEFINES = "DEFINES"
    REFERENCES = "REFERENCES"

    CALLS = "CALLS"
    READS = "READS"
    WRITES = "WRITES"

    DERIVES = "DERIVES"
    FLOWS_TO = "FLOWS_TO"
    CONTROLS = "CONTROLS"

    EXPANDS_TO = "EXPANDS_TO"
    GUARDED_BY = "GUARDED_BY"

    BINDS = "BINDS"
    INSTANTIATES = "INSTANTIATES"
    SPECIALIZES = "SPECIALIZES"

    SELECTS = "SELECTS"
    LAUNCHES = "LAUNCHES"

    # Configuration-contract edges. CALLS is per-site (kind|src|dst|file:line).
    # CALLS_UNDER_GUARD carries a guard BRANCH as src so each predicate is distinct.
    CALLS_UNDER_GUARD = "CALLS_UNDER_GUARD"
    MATERIALIZES_AS = "MATERIALIZES_AS"
    ALLOCATES = "ALLOCATES"

    AVAILABLE_ON = "AVAILABLE_ON"
    ACTIVE_UNDER = "ACTIVE_UNDER"

    SAVES = "SAVES"
    RESTORES = "RESTORES"

    CONTAINS = "CONTAINS"
    RETURNS = "RETURNS"
    ALIASES = "ALIASES"

    # Kernel root-trace graph (UO canonical): wrapper / API → AscendC root.
    WRAPS = "WRAPS"
    ROOTED_AT = "ROOTED_AT"
    BACKED_BY = "BACKED_BY"

    # Source-order facts. PRECEDES is adjacency, not pairing.
    # Flag pair appearance is SIGNALS/AWAITS + UNPAIRED_FLAG_SYNC, not this kind.
    PRECEDES = "PRECEDES"
    SIGNALS = "SIGNALS"
    AWAITS = "AWAITS"

    OTHER = "OTHER"


@dataclass
class Relation:
    """One directed edge in the unified CodeMap."""

    id: str
    kind: RelationKind | str
    src: str
    dst: str
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "extracted"
    confidence: float = 1.0

    def kind_name(self) -> str:
        k = self.kind
        return k.value if isinstance(k, RelationKind) else str(k)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "kind": self.kind_name(),
            "src": self.src,
            "dst": self.dst,
            "status": self.status,
            "confidence": round(float(self.confidence), 4),
        }
        out.update(self.attrs)
        return out
