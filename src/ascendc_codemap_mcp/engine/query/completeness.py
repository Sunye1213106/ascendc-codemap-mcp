# -*- coding: utf-8 -*-
"""C1–C8 completeness fence for a configuration contract card.

Contract state is COMPLETE | INCOMPLETE | AMBIGUOUS | UNKNOWN.
Envelope PARTIAL remains only for list truncation.
"""
from __future__ import annotations

from typing import Any

COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
AMBIGUOUS = "AMBIGUOUS"
UNKNOWN = "UNKNOWN"

KERNEL_OBJECT_ABSENT = "KERNEL_OBJECT_ABSENT"
KERNEL_SINK_UNRESOLVED = "KERNEL_SINK_UNRESOLVED"
KERNEL_CONSUMER_EMPTY = "KERNEL_CONSUMER_EMPTY"
PRODUCER_MISSING = "PRODUCER_MISSING"
EVIDENCE_MISSING = "EVIDENCE_MISSING"
TRANSPORT_UNKNOWN = "TRANSPORT_UNKNOWN"
DISPATCH_UNBOUND = "DISPATCH_UNBOUND"
KERNEL_REPR_MISSING = "KERNEL_REPR_MISSING"
ARCH_MISMATCH = "ARCH_MISMATCH"


def _has_loc(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    file = str(row.get("file") or "").strip()
    line = int(row.get("line") or row.get("line_start") or 0)
    return bool(file) and line > 0


def fence_contract(
    *,
    seeds: list[dict[str, Any]],
    producers: list[dict[str, Any]],
    consumers: list[dict[str, Any]],
    sinks: list[dict[str, Any]],
    transport: str,
    binds: list[dict[str, Any]] | None = None,
    kernel_repr: list[dict[str, Any]] | None = None,
    architecture: str = "",
    seed_arch: str = "",
    host_produced: bool = False,
    template_admissible: bool | None = None,
    unresolved_windows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate C1–C8. Retrieval rank never upgrades INCOMPLETE to COMPLETE."""
    n = len(seeds)
    if n == 0:
        return {
            "completeness": UNKNOWN,
            "unresolved_reason": "NO_SEED",
            "checks": {},
            "windows": list(unresolved_windows or []),
        }
    if n > 1:
        return {
            "completeness": AMBIGUOUS,
            "unresolved_reason": "MULTIPLE_SEEDS",
            "checks": {"c_seed_unique": False},
            "windows": list(unresolved_windows or []),
        }

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    windows: list[dict[str, Any]] = list(unresolved_windows or [])

    c1 = bool(producers)
    checks["c1_producer"] = c1
    if not c1:
        reasons.append(PRODUCER_MISSING)

    c2 = any(_has_loc(row) for row in producers) or _has_loc(seeds[0])
    checks["c2_producer_evidence"] = c2
    if not c2:
        reasons.append(EVIDENCE_MISSING)
        if seeds[0]:
            windows.append(
                {
                    "file": seeds[0].get("file") or "",
                    "line": seeds[0].get("line") or seeds[0].get("line_start") or 0,
                    "why": "producer_site",
                }
            )

    tport = str(transport or "").strip() or "unknown"
    c3 = tport in {"tiling_data", "dispatch", "control"}
    checks["c3_transport"] = c3
    if not c3:
        reasons.append(TRANSPORT_UNKNOWN)

    if tport == "dispatch":
        c4 = bool(binds)
        checks["c4_dispatch_bind"] = c4
        if not c4:
            reasons.append(DISPATCH_UNBOUND)
        c5 = bool(kernel_repr)
        checks["c5_kernel_repr"] = c5
        if not c5:
            reasons.append(KERNEL_REPR_MISSING)
    else:
        checks["c4_dispatch_bind"] = True
        checks["c5_kernel_repr"] = True

    if tport == "control":
        c6 = bool(sinks)
        checks["c6_kernel_consumer"] = c6
        if not c6:
            reasons.append(KERNEL_SINK_UNRESOLVED)
    else:
        c6 = bool(consumers) or bool(sinks)
        checks["c6_kernel_consumer"] = c6
        if not c6:
            reasons.append(KERNEL_CONSUMER_EMPTY)

    loc_rows = list(consumers) + list(sinks)
    c7 = (not c6) or any(_has_loc(row) for row in loc_rows)
    checks["c7_consumer_evidence"] = c7
    if c6 and not c7:
        reasons.append(EVIDENCE_MISSING)

    product_arch = str(architecture or "").strip()
    row_arch = str(seed_arch or seeds[0].get("architecture") or "").strip()
    c8 = (not product_arch) or (not row_arch) or row_arch == product_arch
    checks["c8_architecture"] = c8
    if not c8:
        reasons.append(ARCH_MISMATCH)

    if host_produced and template_admissible is False:
        reasons.append(KERNEL_OBJECT_ABSENT)
        checks["c6_kernel_consumer"] = False

    ok = all(checks.values()) and KERNEL_OBJECT_ABSENT not in reasons
    return {
        "completeness": COMPLETE if ok else INCOMPLETE,
        "unresolved_reason": "" if ok else (reasons[0] if reasons else INCOMPLETE),
        "unresolved_reasons": reasons,
        "checks": checks,
        "transport": tport,
        "windows": windows[:4],
    }


def pick_transport(
    *,
    seed_kind: str,
    has_branch_reader: bool = False,
    has_bind: bool = False,
) -> str:
    kind = str(seed_kind or "")
    if has_branch_reader:
        return "control"
    if kind in {"TILING_KEY", "TEMPLATE_ARG", "COMPILE_VAR"} or has_bind:
        return "dispatch"
    if kind in {"TILING_FIELD", "FIELD"}:
        return "tiling_data"
    return "unknown"
