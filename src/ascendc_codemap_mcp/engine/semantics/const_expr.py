# -*- coding: utf-8 -*-
"""Integer occupancy of a compile-time value expression.

``10`` and ``{10, 11}`` share 10; ``0``/``1`` alone are too common to alias.
"""

from __future__ import annotations


def integer_occupancy(text: str) -> frozenset[int]:
    """Digits in a simple scalar or brace list. Empty if the expr is not a const."""
    raw = str(text or "").strip()
    if not raw:
        return frozenset()
    if not _simple_const_chars(raw):
        return frozenset()
    values: list[int] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch in "+-" and i + 1 < n and raw[i + 1].isdigit():
            sign = -1 if ch == "-" else 1
            i += 1
            j = i
            while j < n and raw[j].isdigit():
                j += 1
            values.append(sign * int(raw[i:j]))
            i = j
            continue
        if ch.isdigit():
            j = i
            while j < n and raw[j].isdigit():
                j += 1
            values.append(int(raw[i:j]))
            i = j
            continue
        i += 1
    return frozenset(values)


def occupancy_overlap(left: str, right: str) -> frozenset[int]:
    a = integer_occupancy(left)
    b = integer_occupancy(right)
    return a & b


def worth_sharing(overlap: frozenset[int], left: str, right: str) -> bool:
    """True when shared integers are informative, not every ``=0``/``=1``."""
    if not overlap:
        return False
    if overlap - {0, 1}:
        return True
    return "{" in str(left) and "{" in str(right)


def _simple_const_chars(text: str) -> bool:
    for ch in text:
        if ch.isalnum() or ch in "{}[]()<>,.+-_ \t":
            continue
        return False
    return True
