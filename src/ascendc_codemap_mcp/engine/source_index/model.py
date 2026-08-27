# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SourceFacts:
    file: str
    includes: list[str] = field(default_factory=list)
    function_spans: list[dict[str, Any]] = field(default_factory=list)
    call_sites: list[dict[str, Any]] = field(default_factory=list)
    primitive_calls: list[dict[str, Any]] = field(default_factory=list)
    type_aliases: list[dict[str, Any]] = field(default_factory=list)
    class_members: list[dict[str, Any]] = field(default_factory=list)
    buffer_decls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SourceIndex:
    by_file: dict[str, SourceFacts] = field(default_factory=dict)
    root: str = ""

    def facts_for(self, path: str | Path) -> SourceFacts | None:
        key = _norm(path)
        hit = self.by_file.get(key)
        if hit is not None:
            return hit
        name = Path(key).name.lower()
        for stored, facts in self.by_file.items():
            if stored.lower().endswith("/" + name) or Path(stored).name.lower() == name:
                return facts
        return None

    def calls_for(
        self,
        files: Iterable[str | Path],
        *,
        primitives_only: bool = False,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in files:
            facts = self.facts_for(path)
            if facts is None:
                continue
            out.extend(facts.primitive_calls if primitives_only else facts.call_sites)
        return out

    def aliases_for(self, files: Iterable[str | Path]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in files:
            facts = self.facts_for(path)
            if facts is not None:
                out.extend(facts.type_aliases)
        return out

    def members_for(self, files: Iterable[str | Path]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in files:
            facts = self.facts_for(path)
            if facts is not None:
                out.extend(facts.class_members)
        return out

    def buffers_for(self, files: Iterable[str | Path]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in files:
            facts = self.facts_for(path)
            if facts is not None:
                out.extend(facts.buffer_decls)
        return out


def _norm(path: str | Path) -> str:
    return str(path).replace("\\", "/")
