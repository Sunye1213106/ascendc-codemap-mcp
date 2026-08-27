# -*- coding: utf-8 -*-
"""AscendC / CANN synchronization catalog for Kernel Root Trace.

Public roots come from CANN interface headers (not project method names):

- ``kernel_operator_block_sync_intf.h`` — HardEvent / barrier / CrossCore / IB / TQueSync
- ``kernel_common.h`` — ``AscendC::Mutex``
- ``kernel_operator_determine_compute_sync_intf.h`` — WaitPreBlock / NotifyNextBlock
- ``kernel_operator_common_intf.h`` — SuperKernel SetNextTaskStart / WaitPreTaskEnd
- ``kernel_tpipe.h`` — AllocEventID / ReleaseEventID / FetchEventID
- ``kernel_reg_compute_membar_intf.h`` — LocalMemBar (MicroAPI = Reg)
- ``core_mng/roc/kernel_operator_group_barrier_intf.h`` — GroupBarrier::Arrive
- ``simt_api/device_sync_functions.h`` — asc_syncthreads / threadfence
- CCE intrinsics used in ops: set_flag / wait_flag / pipe_barrier /
  ffts_cross_core_sync / wait_flag_dev

Two CANN contracts, kept separate:

- **Flag sync** (SetFlag/WaitFlag, CrossCore*, IB*): user-visible event identity.
  UO records SIGNALS/AWAITS and identity-level pair appearance. It does **not**
  infer happens-before, engine scheduling, or which call-site waits on which.
- **TQue** (EnQue/DeQue/AllocTensor/FreeTensor): handshake lives inside CANN
  ``TQueBind`` (EnQue → SetFlag, DeQue → WaitFlag). No user event identity;
  these APIs stay outside the flag pair check.
- **TPipe** (InitBuffer, FetchEventID, AllocEventID, GetTPipePtr): pipe/allocator,
  not TQue.

``Wait`` / ``sync`` / ``Set`` are not catalog roots — they collide with project
wrappers (Buffer::Wait) and SIMT ``cooperative_groups::sync``. GroupBarrier::Wait
is proven only via a CANN qualified name or declaration file.

Project wrapper methods are not auto-jumped to roots by name; source CALLS
(or an explicit external framework bridge) must close the path.
"""

from __future__ import annotations

import re
from typing import Any

# HardEvent enumerators from kernel_event.h (src_dst) — literal evidence only.
HARD_EVENTS: frozenset[str] = frozenset(
    {
        "MTE2_MTE1",
        "MTE1_MTE2",
        "MTE1_M",
        "M_MTE1",
        "MTE2_V",
        "V_MTE2",
        "MTE3_V",
        "V_MTE3",
        "M_V",
        "V_M",
        "V_V",
        "MTE3_MTE1",
        "MTE1_MTE3",
        "MTE1_V",
        "MTE2_M",
        "M_MTE2",
        "V_MTE1",
        "M_FIX",
        "FIX_M",
        "MTE3_MTE2",
        "MTE2_MTE3",
        "S_V",
        "V_S",
        "S_MTE2",
        "MTE2_S",
        "S_MTE3",
        "MTE3_S",
        "MTE2_FIX",
        "FIX_MTE2",
        "FIX_S",
        "M_S",
        "FIX_MTE3",
        "MTE1_FIX",
        "FIX_MTE1",
        "FIX_FIX",
        "FIX_V",
        "V_FIX",
    }
)

_HARD_EVENT_RE = re.compile(
    r"(?:HardEvent(?:Aic|Aiv)?::)?(?P<evt>"
    + "|".join(sorted(HARD_EVENTS, key=len, reverse=True))
    + r")\b"
)
_PIPE_RE = re.compile(r"\b(?P<pipe>PIPE_[A-Z0-9]+|[SMV]|MTE[123]|FIX|ALL)\b")

# CCE intrinsic spellings → public AscendC names (ops call both).
SYNC_SPELLING_ALIASES: dict[str, str] = {
    "set_flag": "SetFlag",
    "wait_flag": "WaitFlag",
    "pipe_barrier": "PipeBarrier",
}

# Types that are sync objects, not storage (token match, never substring).
ASCENDC_SYNC_TYPES: frozenset[str] = frozenset({"GroupBarrier", "TQueSync"})

# True CANN / AscendC / CCE sync roots. Short names only when unique.
SYNC_MECHANISM: dict[str, str] = {
    "SetFlag": "hard_event",
    "WaitFlag": "hard_event",
    "PipeBarrier": "barrier",
    "DataSyncBarrier": "barrier",
    "SyncAll": "barrier",
    "CrossCoreSetFlag": "cross_core",
    "CrossCoreWaitFlag": "cross_core",
    "IBSet": "inter_block",
    "IBWait": "inter_block",
    "TQueSync": "queue_sync",
    "AllocMutexID": "mutex",
    "ReleaseMutexID": "mutex",
    "Lock": "mutex",
    "Unlock": "mutex",
    # kernel_operator_determine_compute_sync_intf.h
    "InitDetermineComputeWorkspace": "determine_compute",
    "WaitPreBlock": "determine_compute",
    "NotifyNextBlock": "determine_compute",
    # kernel_operator_common_intf.h SuperKernel fusion
    "SetNextTaskStart": "superkernel",
    "WaitPreTaskEnd": "superkernel",
    # kernel_tpipe.h event-id pool (used with SetFlag/WaitFlag)
    "AllocEventID": "tpipe",
    "ReleaseEventID": "tpipe",
    "FetchEventID": "tpipe",
    "AllocCrossSyncId": "tpipe",
    # kernel_reg_compute_membar_intf.h (MicroAPI::LocalMemBar)
    "LocalMemBar": "membar",
    # GroupBarrier::Arrive — not Wait (collides with Buffer::Wait)
    "Arrive": "group_barrier",
    # simt_api/device_sync_functions.h
    "asc_syncthreads": "simt",
    "asc_threadfence": "simt",
    "asc_threadfence_block": "simt",
    # CCE hardware intrinsics (CrossCore / SyncAll lowering)
    "ffts_cross_core_sync": "cross_core",
    "wait_flag_dev": "cross_core",
}

# VF/Reg/SIMT spellings — illegal as roots on arch22 (no kernel_reg_compute / SIMT).
VF_GATED_SYNC: frozenset[str] = frozenset(
    {
        "LocalMemBar",
        "asc_syncthreads",
        "asc_threadfence",
        "asc_threadfence_block",
    }
)

# User-level flag APIs that must appear as signal + wait on the same identity.
# Mate is the other side of the family, not a happens-before edge.
FLAG_PAIR_MATE: dict[str, str] = {
    "SetFlag": "WaitFlag",
    "WaitFlag": "SetFlag",
    "CrossCoreSetFlag": "CrossCoreWaitFlag",
    "CrossCoreWaitFlag": "CrossCoreSetFlag",
    "IBSet": "IBWait",
    "IBWait": "IBSet",
}
FLAG_SYNC_CALLEES: frozenset[str] = frozenset(FLAG_PAIR_MATE)

# Barrier-family spellings (query + PRECEDES). Unique names only.
BARRIER_CALLEES: frozenset[str] = frozenset(
    {
        "PipeBarrier",
        "DataSyncBarrier",
        "SyncAll",
        "LocalMemBar",
        "asc_syncthreads",
        "asc_threadfence",
        "asc_threadfence_block",
    }
)

# TQue programming model (queue.yaml). CANN encapsulates the pipe handshake;
# these names must not enter FLAG_SYNC pairing or SIGNALS/AWAITS.
TQUE_CALLEES: frozenset[str] = frozenset(
    {
        "EnQue",
        "DeQue",
        "AllocTensor",
        "FreeTensor",
    }
)

# TPipe (kernel_tpipe.h / kernel_common.h). InitBuffer binds a TQue/TBuf; not a TQue method.
TPIPE_CALLEES: frozenset[str] = frozenset(
    {
        "InitBuffer",
        "FetchEventID",
        "AllocEventID",
        "ReleaseEventID",
        "AllocCrossSyncId",
        "GetTPipePtr",
    }
)


def _short_callee(name: str) -> str:
    return str(name or "").split("::")[-1]


def canonical_sync_name(name: str) -> str:
    """Map CCE intrinsic spellings onto the public AscendC root name."""
    return SYNC_SPELLING_ALIASES.get(_short_callee(name), _short_callee(name))


def is_flag_sync(name: str) -> bool:
    return canonical_sync_name(name) in FLAG_SYNC_CALLEES


def is_tque_callee(name: str) -> bool:
    return _short_callee(name) in TQUE_CALLEES


def is_tpipe_callee(name: str) -> bool:
    return _short_callee(name) in TPIPE_CALLEES


def is_sync_root(name: str) -> bool:
    """True for a CANN/CCE sync spelling (after intrinsic aliasing)."""
    short = _short_callee(name)
    canon = canonical_sync_name(short)
    return canon in SYNC_MECHANISM or short in SYNC_MECHANISM or canon in TPIPE_CALLEES


def flag_pair_key(identity: str, sync: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """Group key for identity-level pair appearance (mechanism, id, HardEvent)."""
    sync = sync or {}
    return (
        str(sync.get("mechanism") or ""),
        str(identity or "").strip(),
        str(sync.get("event") or ""),
    )


def parse_hard_event(text: str) -> tuple[str, str, str] | None:
    """Return (event_name, src_pipe, dst_pipe) or None — literal template evidence."""
    m = _HARD_EVENT_RE.search(str(text or ""))
    if not m:
        return None
    evt = m.group("evt")
    parts = evt.split("_", 1)
    if len(parts) != 2:
        return None
    return evt, parts[0], parts[1]


def parse_pipe_token(text: str) -> str:
    m = _PIPE_RE.search(str(text or ""))
    if not m:
        return ""
    tok = m.group("pipe")
    if tok.startswith("PIPE_"):
        return tok
    if tok in {"S", "V", "M", "FIX", "ALL"} or tok.startswith("MTE"):
        return f"PIPE_{tok}"
    return tok


def resolve_sync_site(
    callee: str,
    args: list[str] | None = None,
    targs: list[str] | None = None,
) -> dict[str, Any]:
    """Classify one sync call site from callee + template/args (catalog only).

    Returns mechanism / flag / event / pipe literals. No engine schedule fields.
    """
    raw = _short_callee(callee)
    name = canonical_sync_name(raw)
    args = [str(a) for a in (args or [])]
    targs = [str(a) for a in (targs or [])]
    joined = " ".join(targs + args)

    mechanism = SYNC_MECHANISM.get(name, "") or SYNC_MECHANISM.get(raw, "")
    cross = (
        name.startswith("CrossCore")
        or raw in {"ffts_cross_core_sync", "wait_flag_dev"}
        or mechanism == "cross_core"
    )

    if name == "SetFlag" or "SetFlag" in name:
        skind = "SetFlag"
    elif name == "WaitFlag" or "WaitFlag" in name:
        skind = "WaitFlag"
    elif name in BARRIER_CALLEES:
        skind = "BARRIER"
    elif name == "IBSet":
        skind = "IBSet"
    elif name == "IBWait":
        skind = "IBWait"
    elif name in {"Lock", "AllocMutexID"}:
        skind = "MutexLock"
    elif name in {"Unlock", "ReleaseMutexID"}:
        skind = "MutexUnlock"
    elif name in {"NotifyNextBlock", "SetNextTaskStart"}:
        skind = "Notify"
    elif name in {"WaitPreBlock", "WaitPreTaskEnd", "wait_flag_dev"}:
        skind = "Wait"
    elif name == "ffts_cross_core_sync":
        skind = "SetFlag"
    else:
        skind = name

    hard = parse_hard_event(joined)
    event = hard[0] if hard else ""
    src_pipe = hard[1] if hard else ""
    dst_pipe = hard[2] if hard else ""

    if raw in {"set_flag", "wait_flag"} and len(args) >= 2:
        src_pipe = parse_pipe_token(args[0]) or src_pipe
        dst_pipe = parse_pipe_token(args[1]) or dst_pipe
        if src_pipe.startswith("PIPE_"):
            src_pipe = src_pipe[len("PIPE_") :]
        if dst_pipe.startswith("PIPE_"):
            dst_pipe = dst_pipe[len("PIPE_") :]
        if src_pipe and dst_pipe and not event:
            event = f"{src_pipe}_{dst_pipe}"

    pipe = ""
    for t in targs:
        p = parse_pipe_token(t)
        if p:
            pipe = p
            break
    if not pipe:
        pipe = parse_pipe_token(joined)
    if not pipe and src_pipe and dst_pipe:
        pipe = f"{src_pipe}_{dst_pipe}"

    flag = ""
    if raw in {"set_flag", "wait_flag"} and len(args) >= 3:
        flag = args[2]
    elif raw in {"wait_flag_dev", "ffts_cross_core_sync"} and args:
        flag = args[-1]
    elif args:
        if skind in {"IBSet", "IBWait"} and len(args) >= 4:
            flag = args[3]
        else:
            flag = args[0]

    if not mechanism:
        if cross:
            mechanism = "cross_core"
        elif event:
            mechanism = "hard_event"
        elif skind.startswith("Mutex"):
            mechanism = "mutex"
        elif skind.startswith("IB"):
            mechanism = "inter_block"
        elif skind == "BARRIER":
            mechanism = "barrier"

    return {
        "kind": skind,
        "mechanism": mechanism or "unknown",
        "flag": str(flag),
        "pipe": str(pipe),
        "event": str(event),
        "cross_core": bool(cross),
        "src_pipe": src_pipe,
        "dst_pipe": dst_pipe,
    }
