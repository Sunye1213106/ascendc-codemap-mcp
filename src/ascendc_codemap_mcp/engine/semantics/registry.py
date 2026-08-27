# -*- coding: utf-8 -*-
"""AscendC Semantic Registry — domain roles for Kernel primitives.

Clang records ``DataCopy(dst, src, ...)``; this registry says it is a memory
transfer that writes arg0 and reads arg1, runs on MTE, etc.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_VERSION = "ascendc-semantic-registry/v1"

_DIR = Path(__file__).resolve().parent


def _load_yaml(name: str) -> dict[str, Any]:
    path = _DIR / name
    if not path.is_file():
        return {}
    from ascendc_codemap_mcp.engine.yaml_io import read_yaml

    return read_yaml(path)


@lru_cache(maxsize=1)
def load_registry() -> dict[str, dict[str, Any]]:
    """Merge category YAML files into callee → semantic record."""
    out: dict[str, dict[str, Any]] = {}
    for name in ("memory.yaml", "queue.yaml", "sync.yaml", "vector.yaml", "cube.yaml", "microapi.yaml"):
        doc = _load_yaml(name)
        ops = doc.get("ops") if isinstance(doc.get("ops"), dict) else doc
        for callee, meta in (ops or {}).items():
            if not isinstance(meta, dict):
                continue
            key = str(callee).split("::")[-1]
            row = dict(meta)
            row.setdefault("category", "UNKNOWN")
            row.setdefault("callee", key)
            row["registry_version"] = REGISTRY_VERSION
            row["source_file"] = name
            out[key] = row
    try:
        from ascendc_codemap_mcp.engine.semantics.ascendc_vf import (
            cann_vf_api_names,
            is_vf_only_api,
            vf_root_spelling,
        )

        for spell in cann_vf_api_names():
            key = vf_root_spelling(spell)
            vf_only = is_vf_only_api(key)
            row = {
                "category": "reg_compute" if vf_only else "vector_compute",
                "engine": "VECTOR",
                "confidence": "confirmed",
                "callee": key,
                "registry_version": REGISTRY_VERSION,
                "source_file": "cann_vf",
            }
            if vf_only:
                row["requires_vf"] = True
            existing = out.get(key)
            if existing is None:
                out[key] = dict(row)
            elif vf_only:
                existing.setdefault("requires_vf", True)
            if spell != key:
                alias = dict(out.get(key) or row)
                alias["callee"] = spell
                alias["alias_of"] = key
                out.setdefault(spell, alias)
    except Exception:  # noqa: BLE001
        pass
    try:
        from ascendc_codemap_mcp.engine.semantics.ascendc_util import cann_util_api_names

        for spell in cann_util_api_names():
            if spell in out:
                continue
            out[spell] = {
                "category": "util",
                "engine": "SCALAR",
                "confidence": "confirmed",
                "callee": spell,
                "registry_version": REGISTRY_VERSION,
                "source_file": "cann_util",
            }
    except Exception:  # noqa: BLE001
        pass
    return out


def lookup(callee: str) -> dict[str, Any] | None:
    name = str(callee or "").split("::")[-1].strip()
    if not name:
        return None
    return load_registry().get(name)


def is_execution_primitive(callee: str) -> bool:
    return lookup(callee) is not None


def classify(callee: str) -> tuple[str, str, str]:
    """Return (category, engine, confidence)."""
    meta = lookup(callee)
    if not meta:
        return "UNKNOWN", "UNKNOWN", "unresolved"
    return (
        str(meta.get("category") or "UNKNOWN"),
        str(meta.get("engine") or "UNKNOWN"),
        str(meta.get("confidence") or "confirmed"),
    )


def arg_effects(callee: str, args: list[str], *, receiver: str = "") -> tuple[list[str], list[str]]:
    """Map call args to buffer read/write names using registry arg roles."""
    meta = lookup(callee) or {}
    roles = meta.get("args") if isinstance(meta.get("args"), dict) else {}
    reads: list[str] = []
    writes: list[str] = []

    def _name(expr: str) -> str:
        text = str(expr or "").strip()
        if not text:
            return ""
        for prefix in ("this->", "this.", "(*this)->", "(*this)."):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        # Drop indexing / call args first so field names survive.
        for sep in ("[", "("):
            if sep in text:
                text = text.split(sep, 1)[0]
        if "->" in text:
            text = text.split("->")[-1]
        if "." in text:
            text = text.split(".")[-1]
        return text.strip()

    for idx_s, role in roles.items():
        try:
            idx = int(idx_s)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(args):
            continue
        name = _name(args[idx])
        if not name:
            continue
        role_l = str(role).lower()
        if "write" in role_l or role_l in {"dst", "dst_buffer", "write_buffer", "write_register"}:
            if name not in writes:
                writes.append(name)
        if "read" in role_l or role_l in {"src", "src_buffer", "read_buffer", "read_register"}:
            if name not in reads:
                reads.append(name)

    recv_role = str(meta.get("receiver") or "").lower()
    recv = _name(receiver)
    if recv:
        if "queue" in recv_role or recv_role in {"write", "allocate"}:
            pass  # queue identity tracked via backing, not as tensor buffer
        if "write" in recv_role and recv not in writes:
            writes.append(recv)
        if "read" in recv_role and recv not in reads:
            reads.append(recv)

    # Conventional DataCopy(dst, src) when registry missing arg map but known category.
    if not reads and not writes and str(meta.get("category") or "") == "memory_transfer":
        if len(args) >= 1 and _name(args[0]):
            writes.append(_name(args[0]))
        if len(args) >= 2 and _name(args[1]):
            reads.append(_name(args[1]))
    return reads, writes
