# -*- coding: utf-8 -*-
"""Host field-sensitive SSA + function summaries (DoOpTiling derivation chain).

The clang backend is authoritative. `extract_writes_text` remains only as a
fallback for files that cannot be parsed: it is a single-line regex, so it
misses assignments spanning lines, cannot attribute a write to its enclosing
function, and has no path conditions to attach. Coverage must be computed on
the clang backend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ascendc_codemap_mcp.engine.clang_walk import (
    CallSite,
    CtrlNode,
    FieldDecl,
    LocalDecl,
    PathCond,
    WalkResult,
    WriteRecord,
    walk_file,
    TypeDecl,
    BaseDecl,
)


def _host_ir_workers(n_files: int) -> int:
    """Cap concurrent libclang TUs. Default 1 — N-way walks froze Windows."""
    import os

    n = max(1, int(n_files))
    raw = str(os.environ.get("UO_HOST_IR_WORKERS") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return max(1, min(n, int(raw)))
    return 1


def _host_ir_pool_kind() -> str:
    """Host TU parallelism. libclang releases the GIL during parse.

    Windows spawn of ProcessPool is a true-cold tax; default to threads there.
    Override with ``UO_HOST_IR_POOL=process`` or ``thread``.
    """
    import os

    raw = str(os.environ.get("UO_HOST_IR_POOL") or "").strip().lower()
    if raw in {"process", "thread"}:
        return raw
    return "thread" if os.name == "nt" else "process"


def _walk_tu_worker(payload: dict) -> WalkResult:
    """ProcessPool entry: rebuild context and walk one TU (pickle-safe)."""
    import time as _time
    from pathlib import Path

    from ascendc_codemap_mcp.engine.build_context import BuildContext
    from ascendc_codemap_mcp.engine.clang_walk import walk_file as _walk_file
    from ascendc_codemap_mcp.engine.scope_scan import ScopeSet
    from ascendc_codemap_mcp.engine.timing import log as _tlog

    p = Path(payload["path"])
    ctx = BuildContext.from_dict(payload["ctx"])
    scope = None
    raw_scope = payload.get("scope")
    if isinstance(raw_scope, dict):
        scope = ScopeSet.from_dict(raw_scope)
    t0 = _time.perf_counter()
    res = _walk_file(
        p,
        ctx,
        side=str(payload.get("side") or "host"),
        op_needle=str(payload.get("op_needle") or ""),
        scope=scope,
        logs_rejections=bool(payload.get("logs_rejections")),
    )
    dt = _time.perf_counter() - t0
    _tlog(
        f"{dt:7.3f}s{' SLOW' if dt > 180 else ''}  host_ir.walk_tu  "
        f"file={p.name} controls={len(getattr(res, 'controls', []) or [])} "
        f"writes={len(getattr(res, 'writes', []) or [])}"
    )
    return res


def _walk_tu_payload(
    path: Path,
    ctx,
    *,
    side: str,
    op_needle: str,
    scope,
    logs_rejections: bool,
) -> dict:
    from ascendc_codemap_mcp.engine.build_context import BuildContext

    scope_payload = None
    if scope is not None and hasattr(scope, "to_dict"):
        scope_payload = scope.to_dict()
    if isinstance(ctx, BuildContext):
        ctx_payload = ctx.to_dict()
    else:
        ctx_payload = {
            "raw": {},
            "cann_root": getattr(ctx, "cann_root", ""),
            "ops_root": getattr(ctx, "ops_root", ""),
            "compat_root": getattr(ctx, "compat_root", ""),
            "op_dir": getattr(ctx, "op_dir", ""),
            "arch_dir": getattr(ctx, "arch_dir", "") or "",
            "repo_root": getattr(ctx, "repo_root", ""),
        }
    return {
        "path": str(path),
        "ctx": ctx_payload,
        "side": side,
        "op_needle": op_needle,
        "scope": scope_payload,
        "logs_rejections": logs_rejections,
    }



@dataclass
class WriteEvent:
    path: str
    line: int
    rhs: str
    template_precondition: str | None = None
    file: str = ""
    function: str = ""
    version: int = 0
    path_conditions: tuple[PathCond, ...] = ()
    #: See `WriteRecord.kind`. `append` and `shrink` mean the RHS is not the
    #: destination's new value, so a consumer chasing a value must skip them.
    kind: str = "assign"
    #: See `WriteRecord.column`. Needed to order this write against a read on
    #: the same line.
    column: int = 0
    #: When this write was promoted from a callee's `__return__` (or out-param)
    #: onto the caller's assigned path. Empty for a direct write. Format:
    #: `callee_of:<FunctionName>`.
    via: str = ""

    @property
    def ssa_name(self) -> str:
        return f"{self.path}@{self.version}"

    def guards(self) -> list[str]:
        """What has to hold for this write to run.

        Bail-out negations are left out: they hold on every run that reaches key
        encoding, so as a guard they are noise, and as a *premise* they belong to
        the run as a whole rather than to one write. `HostIR.legality_premises`
        collects them.
        """
        return [
            pc.pretty()
            for pc in self.path_conditions
            if not pc.is_opaque and not pc.is_bailout
        ]

    def premises(self) -> list[str]:
        """The bail-out negations on the way here, as input legality conditions."""
        return [
            pc.pretty()
            for pc in self.path_conditions
            if not pc.is_opaque and pc.is_bailout
        ]


@dataclass
class FuncSummary:
    name: str
    file: str = ""
    line: int = 0
    line_end: int = 0
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    guards: list[str] = field(default_factory=list)
    locals: dict[str, str] = field(default_factory=dict)
    params: list[str] = field(default_factory=list)
    out_params: list[str] = field(default_factory=list)
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    returns: list[str] = field(default_factory=list)
    assigns: dict[str, str] = field(default_factory=dict)
    assign_lists: dict[str, list[str]] = field(default_factory=dict)
    #: container path → elements appended to it. Kept out of `assigns` because
    #: an element is not the container's value; see `FuncRecord.appends`.
    appends: dict[str, list[str]] = field(default_factory=dict)


_IS_LITERAL = re.compile(r"^\s*(?:-?\d[\w.]*|true|false|nullptr|NULL)\s*$")
_BARE_SELF = re.compile(r"^[A-Za-z_]\w*$")


def _rhs_mentions(var: str, rhs: str) -> bool:
    return bool(re.search(rf"\b{re.escape(var)}\b", rhs or ""))


def _pick_primary_def(var: str, candidates: list[str]) -> str | None:
    """Prefer a definition that does not re-mention the variable (breaks p=p+q cycles).

    Dropping the self-mentioning definitions leaves an accumulator with only
    its initialiser, and that is not the value anyone reads: `coreIdx = 0`
    holds before the packing loop and never after it. Taken as the definition
    it pins `blockOuter = coreIdx + 1` to 1, and every key that needs more
    than one core stops existing — narrowing the feasible set, which invents
    unreachable keys rather than merely missing distinctions.

    So a bare literal is only the answer when it is the only thing the
    variable is ever set to. Two of them and there is nothing to choose
    between — `hit = false` before a loop that sets it `true` is the same
    mistake wearing different clothes. Where an update leaves something with
    content behind — `aicNum` starts at `GetCoreNumAic()` and is clamped
    later — that survives as before.

    Returning nothing does not lose the definitions: `defs_by_function` keeps
    them all, and a reader with no single binding consults that instead.
    """
    cleaned: list[str] = []
    for c in candidates:
        n = (c or "").strip()
        if n:
            cleaned.append(n)
    if not cleaned:
        return None
    independent = [c for c in cleaned if not _rhs_mentions(var, c)]
    pool = independent or cleaned
    nonlit = [c for c in pool if not _IS_LITERAL.match(c)]
    if not nonlit and (len(set(pool)) > 1 or len(independent) < len(cleaned)):
        return None
    return (nonlit or pool)[0]


def _deref_actual(actual: str) -> str:
    """`&this->fBaseParams` and `fBaseParams` name the same thing here."""
    t = (actual or "").strip().lstrip("&*").strip()
    for prefix in ("this->", "this."):
        if t.startswith(prefix):
            t = t[len(prefix) :]
    return t


_TUPLE_CALL_RE = re.compile(
    r"^(?:std::)?(?:make_tuple|tie|forward_as_tuple)\((.*)\)$", re.DOTALL
)


def _split_top_level_args(inner: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            piece = inner[start:i].strip()
            if piece:
                args.append(piece)
            start = i + 1
    tail = inner[start:].strip()
    if tail:
        args.append(tail)
    return args


def _expand_tuple_actual(actual: str, caller_locals: dict[str, str]) -> str | None:
    """Rewrite `make_tuple(m, n)` using the caller's defining expressions for m/n."""
    t = (actual or "").strip()
    m = _TUPLE_CALL_RE.match(t)
    if not m or not caller_locals:
        return None
    prefix = t[: t.index("(")]
    parts: list[str] = []
    changed = False
    for arg in _split_top_level_args(m.group(1)):
        if arg in caller_locals and caller_locals[arg] != arg:
            parts.append(caller_locals[arg])
            changed = True
        else:
            parts.append(arg)
    if not changed:
        return None
    return f"{prefix}({', '.join(parts)})"


@dataclass
class HostIR:
    writes: list[WriteEvent] = field(default_factory=list)
    summaries: dict[str, FuncSummary] = field(default_factory=dict)
    backend: str = "text"
    class_fields: set[str] = field(default_factory=set)
    # Guarded assignments to plain locals, keyed nowhere: use local_writes_in().
    local_writes: list[WriteEvent] = field(default_factory=list)
    call_sites: list[CallSite] = field(default_factory=list)
    #: The control statements themselves. A `PathCond` on a write says which
    #: guards were on the way there; these say what the statement *was* — a
    #: loop's induction variables in particular, which a path condition cannot
    #: carry. Needed to summarise a loop rather than give up at its body.
    controls: list[CtrlNode] = field(default_factory=list)
    #: (declaring struct, member) -> declaration. Read through `field_decl()`.
    field_decls: dict[tuple[str, str], FieldDecl] = field(default_factory=dict)
    #: Local declarations, including those initialising nothing. Read through
    #: `local_decl()`.
    local_decls: list[LocalDecl] = field(default_factory=list)
    #: Class/struct declarations from the already-walked TUs (no extra Clang).
    type_decls: list[TypeDecl] = field(default_factory=list)
    #: Base-class edges from the already-walked TUs.
    base_decls: list[BaseDecl] = field(default_factory=list)

    def local_decl(self, name: str, function: str) -> LocalDecl | None:
        """Where `name` was declared in `function`, if it was declared there.

        None covers both "not a local of this function" and "declared in a
        function we did not walk"; neither lets a caller assume anything about
        the variable's starting value.
        """
        cached = getattr(self, "_local_decls_by_name", None)
        if cached is None:
            cached = {}
            for d in self.local_decls:
                # First declaration wins: a name redeclared in a nested scope
                # shadows rather than replaces, and picking the later one would
                # attribute the inner variable's initialiser to the outer.
                cached.setdefault((d.name, d.function), d)
            self._local_decls_by_name = cached
        return cached.get((name, function))

    def paths(self) -> list[str]:
        return [w.path for w in self.writes]

    def legality_premises(self) -> list[tuple[str, str, str, int]]:
        """What the operator requires of its inputs, as (text, function, file, line).

        An operator states which inputs it accepts only by rejecting the rest:
        `if (queryType == DT_HIFLOAT8) { return GRAPH_FAILED; }` is the whole
        definition, written nowhere else. Every run that produces a key got past
        all of them, so their negations hold together on any key worth asking
        about — which makes them premises of the analysis rather than guards on
        whichever statement happens to follow.

        A rejection nested inside a test states a *conditional* requirement, and
        the condition has to come with it. FAG demands `keepProb < 1` only after
        establishing that a dropout mask was passed; read unconditionally it
        rejects every run without dropout, which is most of them. So each premise
        is the implication "if control got this far, the rejection did not fire",
        and a rejection reached under conditions we cannot read is dropped rather
        than weakened into an unconditional claim.

        Deduplicated on the text: the same check reached along several paths is
        one requirement, and repeating it would just enlarge the query.
        """
        seen: dict[str, tuple[str, str, str, int]] = {}
        for w in (*self.writes, *self.local_writes):
            conds = list(w.path_conditions)
            for i, pc in enumerate(conds):
                if not pc.is_bailout or pc.is_opaque:
                    continue
                context = conds[:i]
                if any(c.is_opaque for c in context):
                    continue
                if context:
                    guarded = " && ".join(f"({c.pretty()})" for c in context)
                    text = f"!({guarded}) || ({pc.pretty()})"
                else:
                    text = pc.pretty()
                seen.setdefault(text, (text, w.function, pc.file, pc.line))
        return [seen[k] for k in sorted(seen)]

    def calls_to(self, callee: str) -> list[CallSite]:
        """Every recorded call of `callee`, with the guards reaching each one.

        A function reached from exactly one unguarded call always runs; one
        reached only under `layoutType == TND` runs exactly then. Either is a
        condition on the input, where the alternative is a free boolean.
        """
        cached = getattr(self, "_calls_by_callee", None)
        if cached is None:
            cached = {}
            for site in self.call_sites:
                cached.setdefault(site.callee, []).append(site)
            self._calls_by_callee = cached
        return cached.get(callee, [])

    def local_writes_in(self, function: str) -> dict[str, list[WriteEvent]]:
        """local name -> its guarded assignments inside `function`."""
        cached = getattr(self, "_local_writes_by_fn", None)
        if cached is None:
            cached = {}
            for w in self.local_writes:
                cached.setdefault(w.function, {}).setdefault(w.path, []).append(w)
            self._local_writes_by_fn = cached
        return cached.get(function, {})

    def writes_to(self, needle: str) -> list[WriteEvent]:
        return [w for w in self.writes if needle in w.path]

    def expand_callee_writers(self) -> list[WriteEvent]:
        """One-level callee expansion for return-valued assignments.

        A write ``path = F(...)`` records only the call site. The values and
        guards that decide ``path`` live as ``__return__`` local writes inside
        ``F``. Without promoting them, proof sites like
        ``GetDeterSparseTilingKey`` and ``SetSparseParams`` look empty even
        though the walker already recorded every return.

        Each promoted write keeps ``function=F`` (so ``writers_of`` finds the
        callee body) and prefixes the RHS with a ``via:callee_of:F`` marker
        the codemap serialises separately. Direct writes are returned first,
        unchanged.
        """
        from ascendc_codemap_mcp.engine.clang_walk import RETURN_SLOT

        by_fn: dict[str, list[WriteEvent]] = {}
        for w in self.local_writes:
            if w.path == RETURN_SLOT and w.function:
                by_fn.setdefault(w.function, []).append(w)

        # Bare identifier or TrailingCall(...). Member calls keep the method
        # name in spelling; strip a receiver prefix when present.
        call_name = re.compile(
            r"^(?:(?:this\.)?[A-Za-z_]\w*(?:\.|->))*([A-Za-z_]\w*)\s*\("
        )

        out: list[WriteEvent] = list(self.writes)
        seen: set[tuple] = set()
        for w in self.writes:
            m = call_name.match((w.rhs or "").strip())
            if not m:
                continue
            callee = m.group(1)
            returns = by_fn.get(callee) or []
            if not returns:
                # Try unqualified match when summaries use a shorter name.
                for name, rws in by_fn.items():
                    if name == callee or name.endswith("::" + callee) or name.endswith("_" + callee):
                        returns = rws
                        break
            for rw in returns:
                key = (w.path, rw.function, rw.line, rw.rhs)
                if key in seen:
                    continue
                seen.add(key)
                # Merge caller reachability with the return's own guards.
                pcs = tuple(w.path_conditions) + tuple(rw.path_conditions)
                out.append(
                    WriteEvent(
                        path=w.path,
                        line=rw.line,
                        rhs=rw.rhs,
                        template_precondition=w.template_precondition,
                        file=rw.file or w.file,
                        function=rw.function or callee,
                        path_conditions=pcs,
                        kind="assign",
                        column=rw.column,
                        via=f"callee_of:{callee}",
                    )
                )
        return out

    def field_decl(self, path: str) -> FieldDecl | None:
        """The declaration of the member `path` names, if it can be identified.

        The table is keyed on (struct, member), but a write path names a
        *variable* — `this.fBaseParams.isNzOut` — not the struct. So the member
        name has to identify the declaration on its own: when two structs
        declare it there is no way to tell which is meant, and the answer is
        None. That is the safe direction. The generated tiling-data structs
        declare many of these same names `= 0`, and guessing would turn "cannot
        prove" into "proved to be zero".
        """
        cached = getattr(self, "_decls_by_member", None)
        if cached is None:
            cached = {}
            for (_, name), decl in self.field_decls.items():
                cached.setdefault(name, []).append(decl)
            self._decls_by_member = cached
        found = cached.get((path or "").rsplit(".", 1)[-1], ())
        return found[0] if len(found) == 1 else None

    def loop_at(self, file: str, line: int) -> CtrlNode | None:
        """The loop statement a `PathCond` of loop kind came from.

        A loop guard on a write carries the file and line of its header, which
        is how a write inside a loop is matched back to the loop's induction
        variables and condition. Nested loops start on different lines, so the
        pair identifies one statement; a one-line `for (…) for (…)` would not be
        told apart, and this returns the first.
        """
        cached = getattr(self, "_loops_by_site", None)
        if cached is None:
            cached = {}
            for n in self.controls:
                if n.kind in ("for", "while", "do", "cxx_for_range"):
                    cached.setdefault((n.file, n.line), n)
            self._loops_by_site = cached
        return cached.get((file, line))

    def container_events(self, container: str, function: str) -> list[WriteEvent]:
        """Every recorded change to `container` inside `function`, in program order.

        Deliberately not built on `writes_by_tail()` or `defs_by_function()`:
        both drop events with an empty RHS, and `clear()` / `pop_back()` are
        exactly those. A rule asking "was a `push_back` the last change before
        this read" would then be blind to the one kind of event that makes the
        answer no.

        Matched on the container's own name, so a local `slicePrefix1` and a
        member `deterPrefixData.prefix1` are told apart, while `prefix1` still
        finds the member's events.
        """
        cached = getattr(self, "_container_events", None)
        if cached is None:
            cached = {}
            for w in list(self.writes) + list(self.local_writes):
                tail = re.sub(r"\[.*", "", w.path.rsplit(".", 1)[-1])
                cached.setdefault((tail, w.function), []).append(w)
            for evs in cached.values():
                evs.sort(key=lambda w: (w.file, w.line, w.column))
            self._container_events = cached
        tail = re.sub(r"\[.*", "", (container or "").rsplit(".", 1)[-1])
        return cached.get((tail, function), [])

    def sole_member_read(
        self, function: str, receiver: str, callee: str
    ) -> CallSite | None:
        """The one `receiver.callee()` call in `function`, if there is exactly one.

        The expression IR carries no source position, so a `back()` node cannot
        say which of several reads it is. When the function holds only one such
        read the position is unambiguous without it; when it holds more, there
        is no way to tell, and the answer is None rather than a guess. That is
        the safe direction: the caller falls back to an over-approximation
        instead of pinning a value to the wrong read.
        """
        cached = getattr(self, "_member_reads", None)
        if cached is None:
            cached = {}
            for s in self.call_sites:
                recv = re.sub(r"\[.*", "", (s.receiver or "").rsplit(".", 1)[-1])
                if recv:
                    cached.setdefault((s.caller, recv, s.callee), []).append(s)
            self._member_reads = cached
        tail = re.sub(r"\[.*", "", (receiver or "").rsplit(".", 1)[-1])
        found = cached.get((function, tail, callee), ())
        return found[0] if len(found) == 1 else None

    def container_writers(self, path: str) -> set[str]:
        """Functions that write `path`, counting `push_back` as a write.

        Asked in order to decide whether one variable may stand for `back(v)`
        at every read point. That holds only while the container holds still,
        and program order across functions cannot be recovered from this IR:
        writes carry a line, reads do not. So a container written in more than
        one function has to be treated as changing between reads.
        """
        cached = getattr(self, "_container_writers", None)
        if cached is None:
            cached = {}
            for w in list(self.writes) + list(self.local_writes):
                tail = re.sub(r"\[.*", "", w.path.rsplit(".", 1)[-1])
                cached.setdefault(tail, set()).add(w.function)
            self._container_writers = cached
        tail = re.sub(r"\[.*", "", (path or "").rsplit(".", 1)[-1])
        return set(cached.get(tail, ()))

    def aggregate_heads(self) -> set[str]:
        """Symbols whose *fields* the host writes — the tiling state aggregates.

        A structural stand-in for a name list. `fBaseParams` and
        `deterPrefixData` qualify because host code fills their members, and
        that is what makes a value read back out of them tiling-derived rather
        than an input. An input accessor never qualifies: nothing assigns to
        `context->GetInputShape(0)->GetStorageShape().dim`.

        Spelling the aggregates by name instead — `Params|TilingData|PrefixData`
        — silently misclassifies any operator that named its own differently.
        """
        cached = getattr(self, "_aggregate_heads", None)
        if cached is None:
            cached = set()
            for w in list(self.writes) + list(self.local_writes):
                parts = w.path.split(".")
                if len(parts) < 2:
                    continue
                cached.add(parts[0])
                # `this.fBaseParams.b` names the aggregate one level in.
                if parts[0] == "this" and len(parts) > 2:
                    cached.add(parts[1])
            cached.discard("this")
            self._aggregate_heads = cached
        return cached

    def latest_version(self, path: str) -> int:
        vs = [w.version for w in self.writes if w.path == path]
        return max(vs) if vs else -1

    def writes_by_tail(self) -> dict[str, list[WriteEvent]]:
        """Index writes by final field name for O(1) field-chase lookup."""
        cached = getattr(self, "_writes_by_tail", None)
        if cached is not None:
            return cached
        out: dict[str, list[WriteEvent]] = {}
        for w in self.writes:
            if not w.rhs.strip():
                continue
            tail = w.path.rsplit(".", 1)[-1]
            # strip residual subscripts just in case
            tail = re.sub(r"\[.*", "", tail)
            out.setdefault(tail, []).append(w)
        self._writes_by_tail = out
        return out

    def param_bound_member(self, fn: str, param: str) -> str | None:
        """The `this` member every caller passes for `param`, if they all agree.

        A free function taking `FuzzyBaseInfoParamsRegbase& fBaseParams` records
        its writes as `fBaseParams.splitAxis` — named after the parameter, not
        after the object. Those writes define `this.fBaseParams.splitAxis` only
        when no caller can pass anything else, so a single disagreeing call site,
        or an argument that is not a member of the enclosing class, gives `None`.
        """
        cached = getattr(self, "_param_binding", None)
        if cached is None:
            cached = {}
            self._param_binding = cached
        key = (fn, param)
        if key in cached:
            return cached[key]
        cached[key] = None  # break recursion through a self-call
        summary = self.summaries.get(fn)
        if not summary or param not in summary.params:
            return None
        idx = summary.params.index(param)
        seen: set[str] = set()
        for caller in self.summaries.values():
            for callee, args in caller.calls:
                if callee == fn and idx < len(args):
                    seen.add(_deref_actual(args[idx]))
        result = None
        if len(seen) == 1:
            only = next(iter(seen))
            if only in self.class_fields:
                result = only
        cached[key] = result
        return result

    def defs_by_function(self) -> dict[str, dict[str, list[str]]]:
        """Every known RHS for each local (declaration + all assignments)."""
        cached = getattr(self, "_defs_by_function", None)
        if cached is not None:
            return cached
        out: dict[str, dict[str, list[str]]] = {}
        for name, s in self.summaries.items():
            slot: dict[str, list[str]] = {}
            for var, init in s.locals.items():
                slot.setdefault(var, [])
                if init and init not in slot[var]:
                    slot[var].append(init)
            for var, hist in s.assign_lists.items():
                slot.setdefault(var, [])
                for rhs in hist:
                    if rhs and rhs not in slot[var]:
                        slot[var].append(rhs)
            for var, rhs in s.assigns.items():
                slot.setdefault(var, [])
                if rhs and rhs not in slot[var]:
                    slot[var].append(rhs)
            out[name] = slot
        self._defs_by_function = out
        return out

    def locals_by_function(self) -> dict[str, dict[str, str]]:
        """Name → primary defining expression inside each function.

        Prefers an assignment that does not re-mention the variable so
        `p = CeilDiv(...); p = p + q` still chases the CeilDiv root.
        """
        cached = getattr(self, "_locals_by_function", None)
        if cached is not None:
            return cached
        out: dict[str, dict[str, str]] = {}
        for name, defs in self.defs_by_function().items():
            picked: dict[str, str] = {}
            for var, candidates in defs.items():
                primary = _pick_primary_def(var, candidates)
                if primary:
                    picked[var] = primary
            out[name] = picked
        self._locals_by_function = out
        return out

    def params_by_function(self) -> dict[str, set[str]]:
        return {name: set(s.params) for name, s in self.summaries.items()}

    def param_bindings(self) -> dict[str, dict[str, list[str]]]:
        """callee -> parameter name -> actual argument sources seen at call sites.

        Same-name formals (`foo(inputLayout)` where the caller also has
        `inputLayout`) are expanded transitively through the caller's locals
        and, if needed, the caller's own parameter bindings.
        """
        cached = getattr(self, "_param_bindings", None)
        if cached is not None:
            return cached
        locals_map = self.locals_by_function()
        # raw edges first
        raw: dict[str, dict[str, list[str]]] = {}
        caller_of: dict[str, list[str]] = {}
        for caller in self.summaries.values():
            for callee, args in caller.calls:
                target = self.summaries.get(callee)
                if target is None or not target.params:
                    continue
                slot = raw.setdefault(callee, {})
                caller_of.setdefault(callee, [])
                if caller.name not in caller_of[callee]:
                    caller_of[callee].append(caller.name)
                for name, actual in zip(target.params, args):
                    if not actual:
                        continue
                    resolved = actual.lstrip("&").strip()
                    # Expand make_tuple/tie args through the caller's locals so
                    # `make_tuple(m, n)` becomes `make_tuple(<m's def>, ...)`
                    # and callee std::get / __tuple_elem can close without a
                    # same-name cycle back into the callee.
                    tup = _expand_tuple_actual(resolved, locals_map.get(caller.name, {}))
                    if tup:
                        resolved = tup
                    seen = slot.setdefault(name, [])
                    if resolved not in seen:
                        seen.append(resolved)

        _IN_PROGRESS = object()
        memo: dict[tuple[str, str, str], Any] = {}

        def expand(callee: str, pname: str, actual: str, stack: frozenset[str]) -> list[str]:
            actual = actual.lstrip("&").strip()
            if not actual:
                return []
            cache_key = (callee, pname, actual)
            hit = memo.get(cache_key)
            if hit is _IN_PROGRESS:
                return []
            if hit is not None:
                return hit
            key = f"{callee}::{pname}::{actual}"
            if key in stack:
                return []
            memo[cache_key] = _IN_PROGRESS
            if actual != pname:
                # expression or other name — still try one hop through a
                # caller's local of that name when it is a bare identifier
                if re.fullmatch(r"[A-Za-z_]\w*", actual):
                    out: list[str] = []
                    for cname in caller_of.get(callee, ()):
                        loc = locals_map.get(cname, {}).get(actual)
                        if loc and loc != actual:
                            out.extend(expand(cname, actual, loc, stack | {key}))
                        cparams = self.summaries.get(cname)
                        if cparams and actual in cparams.params:
                            for a2 in raw.get(cname, {}).get(actual, []):
                                out.extend(expand(cname, actual, a2, stack | {key}))
                    if out:
                        result = list(dict.fromkeys(out))
                        memo[cache_key] = result
                        return result
                memo[cache_key] = [actual]
                return [actual]
            # actual == formal name: must climb to callers
            out = []
            for cname in caller_of.get(callee, ()):
                loc = locals_map.get(cname, {}).get(pname)
                if loc and loc != pname:
                    out.extend(expand(cname, pname, loc, stack | {key}))
                for a2 in raw.get(cname, {}).get(pname, []):
                    out.extend(expand(cname, pname, a2, stack | {key}))
            result = list(dict.fromkeys(out))
            memo[cache_key] = result
            return result

        out: dict[str, dict[str, list[str]]] = {}
        for callee, slots in raw.items():
            for pname, actuals in slots.items():
                expanded: list[str] = []
                for a in actuals:
                    expanded.extend(expand(callee, pname, a, frozenset()))
                # drop pure self-refs that could not be expanded
                expanded = [e for e in expanded if e and e != pname]
                if expanded:
                    out.setdefault(callee, {})[pname] = list(dict.fromkeys(expanded))
                else:
                    out.setdefault(callee, {})[pname] = list(actuals)
        self._param_bindings = out
        return out

    def output_bindings_by_function(self) -> dict[str, dict[str, str]]:
        """caller -> local receiving an out-param write -> RHS inside the callee."""
        cached = getattr(self, "_output_bindings", None)
        if cached is not None:
            return cached
        out: dict[str, dict[str, str]] = {}
        for caller in self.summaries.values():
            slot: dict[str, str] = {}
            for callee, args in caller.calls:
                target = self.summaries.get(callee)
                if target is None or not target.out_params:
                    continue
                outs = set(target.out_params)
                for name, actual in zip(target.params, args):
                    if name not in outs or not actual:
                        continue
                    local = actual.lstrip("&").strip()
                    rhs = target.assigns.get(name)
                    if rhs and local:
                        slot[local] = rhs
            if slot:
                out[caller.name] = slot
        self._output_bindings = out
        return out


_ASSIGN = re.compile(
    r"(?P<lhs>(?:this\.)?fBaseParams\.\w+(?:\.\w+)*|(?:this\.)?\w+\.\w+(?:\.\w+)*)\s*=\s*(?P<rhs>[^;]+);"
)


def extract_writes_text(path: str | Path, template_precondition: str | None = None) -> list[WriteEvent]:
    """Deprecated fallback: single-line regex scanner. Product path uses BuildContext.

    Under-counts; never use for coverage.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    events: list[WriteEvent] = []
    for m in _ASSIGN.finditer(text):
        lhs = m.group("lhs")
        if lhs.count(".") < 1:
            continue
        line = text[: m.start()].count("\n") + 1
        events.append(
            WriteEvent(
                path=lhs,
                line=line,
                rhs=m.group("rhs").strip(),
                template_precondition=template_precondition,
                file=str(path).replace("\\", "/"),
            )
        )
    return _assign_ssa(events)


def _assign_ssa(events: list[WriteEvent]) -> list[WriteEvent]:
    """Version each field path in program order (file, line, column).

    Two writes to one path on the same line used to get an arbitrary relative
    version. `prefix0.push_back(x)` beside a `prefix0` read is exactly that
    case, and any rule asking "what was the last change before this read"
    needs the tie broken the way the source breaks it.
    """
    ordered = sorted(events, key=lambda w: (w.file, w.line, w.column))
    counter: dict[str, int] = {}
    for w in ordered:
        v = counter.get(w.path, 0)
        w.version = v
        counter[w.path] = v + 1
    return ordered


def _to_event(rec: WriteRecord, template_precondition: str | None) -> WriteEvent:
    return WriteEvent(
        path=rec.path,
        line=rec.line,
        rhs=rec.rhs,
        template_precondition=template_precondition,
        file=rec.file,
        function=rec.function,
        path_conditions=rec.path_conditions,
        kind=rec.kind,
        column=rec.column,
    )


def extract_writes_clang(
    path: str | Path,
    ctx,
    *,
    template_precondition: str | None = None,
    side: str = "host",
    op_needle: str = "",
) -> list[WriteEvent]:
    res = walk_file(path, ctx, side=side, op_needle=op_needle)
    return _assign_ssa([_to_event(r, template_precondition) for r in res.writes])


def build_host_ir(
    paths: list[str | Path],
    *,
    ctx=None,
    template_precondition: str | None = None,
    side: str = "host",
    op_needle: str = "",
    scope=None,
    logs_rejections: bool = False,
) -> HostIR:
    """Build the host IR. Uses clang when a BuildContext is supplied.

    `scope` is the scanned file set; when given it decides what the walk may
    read, in place of matching file names against the operator's own.
    `logs_rejections` is for the API layer, see `clang_walk._refuses`.
    """
    if ctx is None:
        writes: list[WriteEvent] = []
        for p in paths:
            writes.extend(extract_writes_text(p, template_precondition=template_precondition))
        return HostIR(
            writes=_assign_ssa(writes),
            summaries=_text_summaries(paths),
            backend="text",
        )

    all_writes: list[WriteEvent] = []
    all_local_writes: list[WriteEvent] = []
    all_calls: list[CallSite] = []
    all_controls: list[CtrlNode] = []
    all_field_decls: dict[tuple[str, str], FieldDecl] = {}
    all_local_decls: list[LocalDecl] = []
    all_type_decls: list[TypeDecl] = []
    all_base_decls: list[BaseDecl] = []
    seen_local_decls: set[tuple[str, str, int, int]] = set()
    seen_type_decls: set[tuple[str, str, int]] = set()
    seen_base_decls: set[tuple[str, str, str]] = set()
    seen_calls: set[tuple[str, str, str, int, int, str]] = set()
    seen_controls: set[tuple[str, int, int, str]] = set()
    summaries: dict[str, FuncSummary] = {}
    class_fields: set[str] = set()
    path_list = [Path(p) for p in paths]

    from ascendc_codemap_mcp.engine.timing import log as _tlog
    import time as _time

    def _walk_one(p: Path):
        t0 = _time.perf_counter()
        res = walk_file(
            p,
            ctx,
            side=side,
            op_needle=op_needle,
            scope=scope,
            logs_rejections=logs_rejections,
        )
        dt = _time.perf_counter() - t0
        _tlog(
            f"{dt:7.3f}s{' SLOW' if dt > 180 else ''}  host_ir.walk_tu  "
            f"file={p.name} controls={len(getattr(res, 'controls', []) or [])} "
            f"writes={len(getattr(res, 'writes', []) or [])}"
        )
        return res

    max_workers = _host_ir_workers(len(path_list)) if len(path_list) > 1 else 1
    if len(path_list) <= 1 or max_workers <= 1:
        if len(path_list) > 1:
            _tlog(f"host_ir.parallel_tus  n={len(path_list)} workers=1 pool=inline")
        results = [_walk_one(p) for p in path_list]
    else:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

        use_proc = _host_ir_pool_kind() == "process"
        results = [None] * len(path_list)
        pool_kind = "process" if use_proc else "thread"
        _tlog(
            f"host_ir.parallel_tus  n={len(path_list)} workers={max_workers} "
            f"pool={pool_kind}"
        )
        if use_proc:
            try:
                payloads = [
                    _walk_tu_payload(
                        p,
                        ctx,
                        side=side,
                        op_needle=op_needle,
                        scope=scope,
                        logs_rejections=logs_rejections,
                    )
                    for p in path_list
                ]
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    futs = {
                        pool.submit(_walk_tu_worker, pl): i
                        for i, pl in enumerate(payloads)
                    }
                    for fut in as_completed(futs):
                        results[futs[fut]] = fut.result()
            except Exception as exc:  # noqa: BLE001 — fall back to threads
                _tlog(f"host_ir.process_pool_fallback  reason={type(exc).__name__}")
                use_proc = False
                results = [None] * len(path_list)
        if not use_proc:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(_walk_one, p): i for i, p in enumerate(path_list)}
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()

    for res in results:
        all_writes.extend(_to_event(r, template_precondition) for r in res.writes)
        all_local_writes.extend(
            _to_event(r, template_precondition) for r in res.local_writes
        )
        class_fields |= res.class_fields
        # A header is walked once per TU including it, so the same declaration
        # arrives repeatedly; they agree, so the first one stands.
        for key, decl in (getattr(res, "field_decls", None) or {}).items():
            all_field_decls.setdefault(key, decl)
        for d in getattr(res, "local_decls", ()) or ():
            # Same reason as the calls below: a header walked once per TU
            # yields the same declaration repeatedly. Position identifies it.
            dkey = (d.file, d.name, d.line, d.column)
            if dkey not in seen_local_decls:
                seen_local_decls.add(dkey)
                all_local_decls.append(d)
        for td in getattr(res, "type_decls", ()) or ():
            tkey = (getattr(td, "usr", "") or td.name, td.file, td.line)
            if tkey not in seen_type_decls:
                seen_type_decls.add(tkey)
                all_type_decls.append(td)
        for bd in getattr(res, "base_decls", ()) or ():
            bkey = (bd.derived_name, bd.base_name, getattr(bd, "file", "") or "")
            if bkey not in seen_base_decls:
                seen_base_decls.add(bkey)
                all_base_decls.append(bd)
        for site in getattr(res, "call_sites", ()) or ():
            # A header included by several TUs is walked once per TU, so the
            # same physical call arrives repeatedly. Position has to include the
            # column, and identity the receiver: `syncRounds.size() +
            # syncRoundRanges.size()` agrees on caller, callee, file and line,
            # and dropping either half loses a container's only read.
            key = (
                site.caller,
                site.callee,
                site.file,
                site.line,
                getattr(site, "column", 0),
                getattr(site, "receiver", ""),
            )
            if key not in seen_calls:
                seen_calls.add(key)
                all_calls.append(site)
        for node in getattr(res, "controls", ()) or ():
            # Deduplicated on position rather than `id`: the ordinal in an id is
            # assigned in walk order, and the TUs are walked in parallel.
            ckey = (node.file, node.line, node.column, node.kind)
            if ckey not in seen_controls:
                seen_controls.add(ckey)
                all_controls.append(node)
        for name, fr in res.functions.items():
            s = summaries.setdefault(name, FuncSummary(name=name))
            fr_file = str(getattr(fr, "file", "") or "")
            fr_start = int(getattr(fr, "line", 0) or 0)
            fr_end = int(getattr(fr, "line_end", 0) or 0)
            if fr_end < fr_start:
                fr_end = fr_start
            incoming_span = max(0, fr_end - fr_start) if fr_start > 0 else 0
            existing_span = max(0, int(s.line_end or s.line or 0) - int(s.line or 0))
            if incoming_span > existing_span or (fr_file and not s.file):
                if fr_file:
                    s.file = fr_file
                if fr_start > 0:
                    s.line = fr_start
                s.line_end = fr_end
            elif fr_end > int(s.line_end or 0):
                s.line_end = fr_end
            for w in fr.writes:
                if w not in s.writes:
                    s.writes.append(w)
            for r in fr.reads:
                if r not in s.reads:
                    s.reads.append(r)
            for g in fr.guards:
                if g not in s.guards:
                    s.guards.append(g)
            for k, v in fr.locals.items():
                s.locals.setdefault(k, v)
            for prm in fr.params:
                if prm not in s.params:
                    s.params.append(prm)
            for prm in getattr(fr, "out_params", []) or []:
                if prm not in s.out_params:
                    s.out_params.append(prm)
            for c in fr.calls:
                if c not in s.calls:
                    s.calls.append(c)
            for r in fr.returns:
                if r not in s.returns:
                    s.returns.append(r)
            for k, v in fr.assigns.items():
                # last write across TUs wins (path order preserved by results[])
                s.assigns[k] = v
            for k, hist in getattr(fr, "assign_lists", {}).items():
                slot = s.assign_lists.setdefault(k, [])
                for rhs in hist:
                    if rhs and rhs not in slot:
                        slot.append(rhs)
            for k, hist in getattr(fr, "appends", {}).items():
                slot = s.appends.setdefault(k, [])
                for rhs in hist:
                    if rhs and rhs not in slot:
                        slot.append(rhs)
    return HostIR(
        writes=_assign_ssa(all_writes),
        summaries=summaries,
        backend="clang",
        class_fields=class_fields,
        local_writes=_assign_ssa(all_local_writes),
        call_sites=all_calls,
        controls=all_controls,
        field_decls=all_field_decls,
        local_decls=all_local_decls,
        type_decls=all_type_decls,
        base_decls=all_base_decls,
    )


_FN_HEADER = re.compile(
    r"^[\w:<>,\*&\s]*?\b(?P<name>\w+)\s*\([^;{]*\)\s*(?:const\s*)?\{",
    re.MULTILINE,
)
_NOT_A_FUNCTION = {"if", "for", "while", "switch", "catch", "else", "return", "do"}


def _text_summaries(paths: list[str | Path]) -> dict[str, FuncSummary]:
    """Coarse fallback: attribute writes to the nearest preceding function header."""
    out: dict[str, FuncSummary] = {}
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        for m in _FN_HEADER.finditer(text):
            name = m.group("name")
            if name in _NOT_A_FUNCTION:
                continue
            body = text[m.end() : m.end() + 4000]
            s = out.setdefault(name, FuncSummary(name=name))
            if not s.file:
                s.file = str(Path(p))
                s.line = text[: m.start()].count("\n") + 1
            for wm in _ASSIGN.finditer(body):
                lhs = wm.group("lhs")
                if lhs.count(".") >= 1 and lhs not in s.writes:
                    s.writes.append(lhs)
    return out


def assert_no_flatten(writes: Iterable[WriteEvent]) -> None:
    for w in writes:
        if "." not in w.path:
            raise AssertionError(f"flattened field path: {w.path}")


def derivation_chain(ir: HostIR, field_needle: str) -> list[dict[str, Any]]:
    """Every write to a field, with its SSA version and guarding path conditions."""
    out = []
    for w in ir.writes_to(field_needle):
        out.append(
            {
                "ssa": w.ssa_name,
                "file": w.file,
                "line": w.line,
                "rhs": w.rhs,
                "guards": w.guards(),
                "function": w.function,
                "template": w.template_precondition,
            }
        )
    return out
