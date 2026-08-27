# -*- coding: utf-8 -*-
"""Minimal YAML helpers shared by uo_init update and Pilot adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

#: libyaml-backed loader when the wheel carries it, else the pure-Python one.
#: Same accepted document subset, about seven times faster -- reading the
#: 16 KB scope set costs 42ms through SafeLoader and 6ms through CSafeLoader,
#: and analyze reads scope sets often enough for that to be seconds.
#:
#: Only reads are switched. `CSafeDumper` lays out sequences and line breaks
#: slightly differently, and receipts written here are hashed and diffed
#: elsewhere, so changing how they are spelled would be a product change made
#: for a speedup that does not exist -- writing YAML is not hot.
_LOADER = None if yaml is None else getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader


def require_yaml() -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required")
    return yaml


def read_yaml(path: Path) -> dict[str, Any]:
    require_yaml()
    if not path.exists():
        return {}
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_LOADER) or {}
    return data if isinstance(data, dict) else {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
