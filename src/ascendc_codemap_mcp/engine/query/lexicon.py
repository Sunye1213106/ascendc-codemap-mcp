# -*- coding: utf-8 -*-
"""Deterministic query lexicon. Not a source fact; hits go to Symbols."""
from __future__ import annotations

# (phrase, tag) — phrase is what an agent might type; tag is a code token.
ALIASES: tuple[tuple[str, str], ...] = (
    ("global memory", "GM"),
    ("unified buffer", "UB"),
    ("l1 buffer", "L1"),
    ("double buffer", "DB"),
    ("ping pong", "DB"),
    ("triple buffer", "3buff"),
    ("3 buffer", "3buff"),
    ("quad buffer", "4buff"),
    ("4 buffer", "4buff"),
    ("hard event", "hard_event"),
    ("hard sync", "hard_event"),
    ("dropout", "dropMask"),
    ("drop out", "IsDrop"),
)

_KIND_FIELDS: dict[str, tuple[str, ...]] = {
    "BUFFER": ("memory_space", "physical_space", "storage_class"),
    "REGISTER": ("memory_space", "register_class"),
    "QUEUE": ("memory_space", "physical_space"),
    "EVENT": ("mechanism", "event_type"),
    "OPERATION": ("callee", "category", "engine"),
    "TYPE": ("qualified_name",),
    "PIPE": ("memory_space",),
}


def kind_fields(kind: str) -> tuple[str, ...]:
    return _KIND_FIELDS.get(str(kind or "").upper(), ())


def entity_haystack(kind: str, name: str, data: dict[str, object] | None) -> str:
    parts = [str(name or "")]
    blob = data if isinstance(data, dict) else {}
    for key in kind_fields(kind):
        val = blob.get(key)
        if val not in (None, ""):
            parts.append(str(val))
    return "\n".join(parts)


def lexicon_tags(phrase: str) -> list[tuple[str, str]]:
    text = str(phrase or "").strip().lower()
    if not text:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for alias, tag in ALIASES:
        if alias in text or text == tag.lower():
            key = f"{alias}->{tag}"
            if key in seen:
                continue
            seen.add(key)
            out.append((f"{alias} → {tag}", tag))
    return out
