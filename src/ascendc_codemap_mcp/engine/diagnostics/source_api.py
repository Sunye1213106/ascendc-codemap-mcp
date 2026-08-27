# -*- coding: utf-8 -*-
"""Source vs CodeMap kernel-API needles, plus generalization blocker ranking.

Graph counts use the exact callee (``DataCopyPad`` is not ``DataCopy``).
Precision source n is unique ``(file, line, name)`` in KernelCorpus files owned
by this operator or a one-hop sibling ``.cpp`` (fusion TUs). Family ``common/``
and CANN templates stay in the corpus for locate/owner tags but are not the
gated denominator — counting cube ``Cast`` as this op's source n forces the
graph to mint internals that are not this wrapper's algorithm.
Method definitions are not calls.
A source hit with graph n < source n, or with_span < graph n, is a gap.
Source n=0 and graph n=0 is honest only when the include closure is also empty.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.passes.kernel_scan import (
    _CALL_RE,
    _is_false_lexical_callee,
    kernel_corpus,
    kernel_file_owner,
)
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text

# Precision bar vs FAG 口径. DataCopyPad before DataCopy is documentation only;
# matching is exact.
PRECISION_APIS = ("EnQue", "DeQue", "DataCopyPad", "DataCopy", "Cast")

GRAPH_API_NEEDLES = (
    "EnQue",
    "DeQue",
    "InitBuffer",
    "DataCopyPad",
    "DataCopy",
    "Copy",
    "Cast",
    "SetFlag",
    "WaitFlag",
    "SetGlobalBuffer",
    "LoadAlign",
    "LoadData",
)

# Source n>0 ⇒ graph n ≥ source n and with_span = graph n.
GATED_SOURCE_APIS = (
    "EnQue",
    "DeQue",
    "DataCopyPad",
    "DataCopy",
    "Copy",
    "Cast",
    "SetFlag",
    "WaitFlag",
    "LoadData",
)

# Precision denominator. family_common / cann remain locatable via owner.
GATED_SOURCE_OWNERS = frozenset({"this_op", "sibling_op"})
ALL_SOURCE_OWNERS = frozenset({"this_op", "sibling_op", "family_common", "cann"})

BLOCKER_ORDER = (
    "prepare_blocked",
    "extract_crash",
    "precision_gap",
    "quality_not_ready",
    "verify_packing",
    "field_owner_ambiguous",
)

_COMMENT_RE = re.compile(r"/\*.*?\*/|//.*?$", re.S | re.M)


def callee_short_name(*parts: Any) -> str:
    raw = " ".join(str(p or "") for p in parts).strip()
    if not raw:
        return ""
    short = raw.split()[0].split("::")[-1].strip()
    if "<" in short:
        short = short.split("<", 1)[0].strip()
    return short


def operation_callee(entity: Any) -> str:
    attrs = getattr(entity, "attrs", None) or {}
    if not isinstance(attrs, dict):
        attrs = {}
    return callee_short_name(
        attrs.get("callee"),
        attrs.get("api"),
        getattr(entity, "name", ""),
    )


def _has_span(entity: Any) -> bool:
    return bool(str(getattr(entity, "file", "") or "").strip()) and int(
        getattr(entity, "line_start", 0) or 0
    ) > 0


def count_graph_kernel_api(operations: Iterable[Any]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {
        needle: {"n": 0, "with_span": 0, "reached": 0} for needle in GRAPH_API_NEEDLES
    }
    for entity in operations:
        name = operation_callee(entity)
        if name not in rows:
            continue
        rows[name]["n"] += 1
        if _has_span(entity):
            rows[name]["with_span"] += 1
        attrs = getattr(entity, "attrs", None) or {}
        if isinstance(attrs, dict) and str(attrs.get("root_status") or "") == "REACHED":
            rows[name]["reached"] += 1
    return rows


def count_source_kernel_apis(
    op_dir: str | Path,
    architecture: str,
    files: Iterable[Path] | None = None,
    *,
    owners: Iterable[str] | None = None,
) -> dict[str, int]:
    """Unique ``(file, line, name)`` lexical sites in the KernelCorpus.

    Default ``owners`` is :data:`GATED_SOURCE_OWNERS` (this op + sibling
    ``.cpp``). Pass :data:`ALL_SOURCE_OWNERS` to include family ``common/``.
    ``files`` pins the snapshot used at mint time so quality does not rescan
    a different include closure.
    """
    counts = {name: 0 for name in GRAPH_API_NEEDLES}
    root = Path(op_dir)
    allow = GATED_SOURCE_OWNERS if owners is None else frozenset(str(o) for o in owners)
    try:
        scan = [Path(p) for p in files] if files is not None else kernel_corpus(root, architecture)
    except Exception:  # noqa: BLE001
        return counts
    seen: set[tuple[str, int, str]] = set()
    for path in scan:
        try:
            text = _COMMENT_RE.sub(" ", read_text(path))
        except OSError:
            continue
        if allow and kernel_file_owner(path, root) not in allow:
            continue
        file_key = str(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in _CALL_RE.finditer(line):
                name = match.group("name")
                if name not in counts:
                    continue
                if _is_false_lexical_callee(name, line, match.start()):
                    continue
                key = (file_key, line_no, name)
                if key in seen:
                    continue
                seen.add(key)
                counts[name] += 1
    return counts


def source_api_from_codemap(
    codemap: Any,
    source_root: str | Path | None = None,
    architecture: str = "",
) -> dict[str, int] | None:
    """Prefer the mint-time gated snapshot; otherwise rescan with gated owners."""
    meta: dict[str, Any] = {}
    if codemap is not None:
        raw = getattr(codemap, "meta", None) or {}
        if isinstance(raw, dict):
            meta = dict(raw.get("kernel_root_trace") or {})
    snap = meta.get("source_api_gated")
    if isinstance(snap, dict):
        return {name: int(snap.get(name) or 0) for name in GRAPH_API_NEEDLES}
    root = source_root
    arch = str(architecture or getattr(codemap, "architecture", "") or "")
    if root and arch:
        try:
            return count_source_kernel_apis(root, arch)
        except Exception:  # noqa: BLE001
            return None
    return None


def precision_gaps(
    source: dict[str, Any] | None,
    graph: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Source n>0 but graph n < source n or with_span < graph n → extraction gap."""
    src = source or {}
    api = graph or {}
    gaps: list[dict[str, Any]] = []
    for name in GATED_SOURCE_APIS:
        src_n = int(src.get(name) or 0)
        row = api.get(name) if isinstance(api.get(name), dict) else {}
        graph_n = int((row or {}).get("n") or 0)
        spanned = int((row or {}).get("with_span") or 0)
        if src_n <= 0:
            continue
        if graph_n < src_n or spanned < graph_n:
            gaps.append(
                {
                    "api": name,
                    "source": src_n,
                    "graph_n": graph_n,
                    "with_span": spanned,
                }
            )
    return gaps


def classify_blocker(case: dict[str, Any]) -> str | None:
    if not isinstance(case, dict) or case.get("audit_only"):
        return None
    failed = str(case.get("failed_step") or "")
    if failed == "prepare":
        return "prepare_blocked"
    if failed == "extract":
        return "extract_crash"
    noise = case.get("noise") if isinstance(case.get("noise"), dict) else {}
    gaps = noise.get("precision_gaps") or []
    if gaps:
        return "precision_gap"
    quality = noise.get("quality") if isinstance(noise.get("quality"), dict) else {}
    grade = str(quality.get("grade") or "")
    if failed == "quality" or (grade and grade != "ready"):
        reasons = [str(r) for r in (quality.get("not_ready_reasons") or [])]
        field_rw = ((quality.get("surfaces") or {}).get("field_rw") or {}) if isinstance(
            quality.get("surfaces"), dict
        ) else {}
        owner_unknown = int(field_rw.get("field_owner_unknown") or 0)
        if "field_owner_unknown" in reasons or owner_unknown > 0:
            return "field_owner_ambiguous"
        if int(noise.get("other_count") or 0) > 0 or grade != "ready":
            return "quality_not_ready"
    reasons = [str(r) for r in (quality.get("not_ready_reasons") or [])]
    field_rw = ((quality.get("surfaces") or {}).get("field_rw") or {}) if isinstance(
        quality.get("surfaces"), dict
    ) else {}
    owner_unknown = int(field_rw.get("field_owner_unknown") or 0)
    if "field_owner_unknown" in reasons or owner_unknown > 0:
        return "field_owner_ambiguous"
    locate = noise.get("locate") if isinstance(noise.get("locate"), dict) else {}
    locate_gaps = [str(g) for g in (locate.get("gaps") or [])]
    packing_gaps = {
        "no_tiling_key_packing_site",
        "no_tiling_field_writer",
        "no_input_span",
        "no_kernel_span",
    }
    if packing_gaps.intersection(locate_gaps):
        return "verify_packing"
    verdict = str(case.get("verdict") or "")
    if failed == "verify" or (verdict and verdict != "pass"):
        return "verify_packing"
    if case.get("ok"):
        return None
    return "verify_packing"


def rank_blockers(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = {k: [] for k in BLOCKER_ORDER}
    n = 0
    for case in cases:
        if not isinstance(case, dict) or case.get("audit_only"):
            continue
        n += 1
        kind = classify_blocker(case)
        if not kind:
            continue
        counts[kind] += 1
        if kind in samples and len(samples[kind]) < 8:
            rel = str(case.get("rel") or "")
            arch = str(case.get("architecture") or case.get("arch") or "")
            samples[kind].append(f"{rel}:{arch}".rstrip(":"))
    worst = ""
    worst_n = 0
    for kind in BLOCKER_ORDER:
        c = int(counts.get(kind) or 0)
        if c > worst_n:
            worst, worst_n = kind, c
    return {
        "n_cases": n,
        "counts": {k: int(counts.get(k) or 0) for k in BLOCKER_ORDER},
        "worst": worst or None,
        "worst_n": worst_n,
        "samples": {k: v for k, v in samples.items() if v},
    }
