# -*- coding: utf-8 -*-
"""Which kernel code each TilingKey dimension selects.

The kernel is one template parameterised by the key, and every dimension of
the key switches code in or out through `if constexpr`. Reading those branches
gives the map from a dimension to the code it decides -- which is what
"a change here affects what?" needs, and what a test aimed at one dimension
has to cover.

Deliberately read *before* instantiation. Once instantiated, `IS_ROPE` has
folded to `true` and the branch it guarded is either there or gone, with
nothing left saying which dimension decided it. Uninstantiated the condition
still names the parameter, which is the whole point. It also keeps the cost to
one parse per dtype variant rather than one per instantiation, of which there
are hundreds.

The dtype variants are parsed separately because the dtype macro is a
preprocessor value, not a template parameter: different values compile
different code, so a single parse sees only a third of the kernel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*[(<]")
QUALIFIER_RE = re.compile(r"\b([A-Za-z_]\w*)\s*::")


@dataclass
class KernelBranch:
    """One `if constexpr`, and what decides it."""

    condition: str
    file: str
    line: int
    function: str = ""
    #: `KBR_*`, assigned by :meth:`KernelIR.mint_ids`. Empty until then.
    id: str = ""
    #: TilingKey dimensions the condition names outright.
    dimensions: list[str] = field(default_factory=list)
    #: Names built from a dimension rather than being one, such as a
    #: `constexpr bool` derived from it.
    derived: list[str] = field(default_factory=list)
    #: Everything else the condition mentions.
    symbols: list[str] = field(default_factory=list)
    #: Which dtype variants compile this branch.
    variants: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.file, self.line, self.condition)


@dataclass
class KernelIR:
    """The kernel's compile-time branching, indexed by what decides it."""

    branches: list[KernelBranch] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: name → {file, line, calls, params} from the already-walked kernel TU.
    functions: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: (owner, member) → {type_text, file, line} from kernel Clang walks.
    field_decls: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    #: Clang MACRO_INSTANTIATION sites merged from kernel walks.
    macro_uses: list[dict[str, Any]] = field(default_factory=list)

    def mint_ids(self, op_root: str = "") -> None:
        """Assign `KBR_*` ids using the same material as the folded branches.

        Same scheme, but the guard differs: a folded branch carries the
        instantiated condition while these carry the one that still names the
        dimension, so the two never collide and both can sit in the graph. The
        overlap is the source location, which is what joins them.
        """
        from ascendc_codemap_mcp.engine.ids import branch_id

        ordinals: dict[tuple[str, str, str], int] = {}
        for b in self.branches:
            key = (b.file, b.function, b.condition)
            n = ordinals.get(key, 0)
            ordinals[key] = n + 1
            b.id = branch_id(
                side="kernel",
                file=b.file,
                function=b.function,
                guard=b.condition,
                ordinal=n,
                root=op_root,
            )

    def touching(self, dimension: str) -> list[KernelBranch]:
        return [
            b
            for b in self.branches
            if dimension in b.dimensions or dimension in b.derived
        ]

    def by_dimension(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in self.branches:
            for d in (*b.dimensions, *b.derived):
                counts[d] = counts.get(d, 0) + 1
        return counts

    def variant_only(self) -> list[KernelBranch]:
        """Branches that only some dtype variants compile at all."""
        return [b for b in self.branches if len(b.variants) < len(self.variants)]

    def silent_dimensions(self, dimensions: list[str]) -> list[str]:
        """Dimensions no branch was found for.

        Either the dimension decides nothing at compile time, or the inner
        template renamed it on the way down -- `DeterType` arrives as
        `DETER_SPARSE_TYPE`. Reported rather than guessed at: matching on how
        similar two names look would attach branches to the wrong dimension,
        and a wrong answer here is worse than a missing one.
        """
        seen = self.by_dimension()
        return [d for d in dimensions if not seen.get(d)]

    def unmapped_symbols(self, limit: int = 0) -> list[tuple[str, int]]:
        """Names the conditions turn on that reached no dimension, commonest
        first. Where a renamed dimension shows up."""
        counts: dict[str, int] = {}
        for b in self.branches:
            for s in b.symbols:
                counts[s] = counts.get(s, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit] if limit else ranked

    def to_dict(self) -> dict:
        return {
            "variants": list(self.variants),
            "branches": len(self.branches),
            "by_dimension": self.by_dimension(),
            "variant_only": len(self.variant_only()),
            "notes": list(self.notes),
            "detail": [
                {
                    "id": b.id,
                    "condition": b.condition,
                    "file": Path(b.file).name,
                    "line": b.line,
                    "function": b.function,
                    "dimensions": list(b.dimensions),
                    "derived": list(b.derived),
                    "variants": list(b.variants),
                }
                for b in self.branches
            ],
        }

    def to_persist_dict(self) -> dict:
        """Full branch payload for cross-process reload (export_kb)."""
        return {
            "schema": "uo-kernel-ir/v1",
            "variants": list(self.variants),
            "notes": list(self.notes),
            "branches": [
                {
                    "id": b.id,
                    "condition": b.condition,
                    "file": b.file,
                    "line": b.line,
                    "function": b.function,
                    "dimensions": list(b.dimensions),
                    "derived": list(b.derived),
                    "symbols": list(b.symbols),
                    "variants": list(b.variants),
                }
                for b in self.branches
            ],
        }


def kernel_ir_from_dict(data: dict | None) -> KernelIR | None:
    """Rebuild :class:`KernelIR` from :meth:`KernelIR.to_persist_dict`."""
    if not isinstance(data, dict):
        return None
    rows = data.get("branches")
    if not isinstance(rows, list):
        return None
    ir = KernelIR(
        variants=[str(v) for v in (data.get("variants") or [])],
        notes=[str(n) for n in (data.get("notes") or [])],
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        cond = str(row.get("condition") or "").strip()
        if not cond:
            continue
        ir.branches.append(
            KernelBranch(
                condition=cond,
                file=str(row.get("file") or ""),
                line=int(row.get("line") or 0),
                function=str(row.get("function") or ""),
                id=str(row.get("id") or ""),
                dimensions=[str(x) for x in (row.get("dimensions") or [])],
                derived=[str(x) for x in (row.get("derived") or [])],
                symbols=[str(x) for x in (row.get("symbols") or [])],
                variants=[str(x) for x in (row.get("variants") or [])],
            )
        )
    return ir


def _select_dtype_variants(
    variants: list[str | None], max_variants: int | None
) -> list[str | None]:
    """Keep preferred dtypes first; ``max_variants<=0`` or None keeps all."""
    if not variants:
        return [None]
    if max_variants is None or int(max_variants) <= 0:
        return list(variants)
    cap = int(max_variants)
    preferred = ("DT_FLOAT16", "FLOAT16", "float16", "DT_BF16", "BF16", "DT_FLOAT", "FLOAT32")
    ordered: list[str | None] = []
    for p in preferred:
        if p in variants and p not in ordered:
            ordered.append(p)
    for v in variants:
        if v not in ordered:
            ordered.append(v)
    return ordered[:cap]


def _squash(name: str) -> str:
    return name.replace("_", "").lower()


class _Dimensions:
    """Matches a kernel parameter name to the key dimension it carries.

    The two sides spell the same idea differently -- `IS_ROPE` against
    `IsRope`, `SPLIT_AXIS` against `SplitAxis` -- so the separators and case go
    before comparing. A name that merely starts with a dimension, such as a
    `constexpr bool OUTDTYPE_IS_B16` computed from `OUTDTYPE`, is reported
    separately: it is evidence about that dimension without being it.
    """

    def __init__(self, names: list[str]) -> None:
        self._exact = {_squash(n): n for n in names}
        # Longest first, so `OUTDTYPE_IS_B16` prefers `OutDType` over `Out`.
        self._prefixes = sorted(self._exact.items(), key=lambda kv: -len(kv[0]))

    def classify(self, ident: str) -> tuple[str | None, str | None]:
        squashed = _squash(ident)
        hit = self._exact.get(squashed)
        if hit:
            return hit, None
        for prefix, name in self._prefixes:
            if len(prefix) >= 4 and squashed.startswith(prefix):
                return None, name
        return None, None


def _classify(condition: str, dims: _Dimensions) -> tuple[list, list, list]:
    syntax = set(CALL_RE.findall(condition)) | set(QUALIFIER_RE.findall(condition))
    exact: list[str] = []
    derived: list[str] = []
    others: list[str] = []
    for ident in IDENT_RE.findall(condition):
        if ident in syntax:
            continue
        hit, near = dims.classify(ident)
        if hit is not None:
            if hit not in exact:
                exact.append(hit)
        elif near is not None:
            if near not in derived:
                derived.append(near)
        elif ident not in others:
            others.append(ident)
    return exact, derived, others


def build_kernel_ir(
    spec,
    ctx,
    *,
    dimensions: list[str] | None = None,
    max_variants: int | None = None,
) -> KernelIR:
    """Parse the kernel entry once per dtype variant and index its branches.

    ``max_variants`` caps how many dtype walks run (``fast`` uses 1).  Branch
    conditions are usually shared across dtypes; capping still populates the
    tilingkey→code map, and only thins ``branch.variants`` tags.
    """
    from ascendc_codemap_mcp.engine.clang_walk import walk_file

    ir = KernelIR()
    entries = [
        Path(p)
        for p in (getattr(spec, "kernel_targets", None) or [spec.kernel_entry])
        if p and Path(p).is_file()
    ]
    if not entries:
        ir.notes.append("no_kernel_entry")
        return ir

    from ascendc_codemap_mcp.engine.build_context import source_uses_dtype_variants

    if any(
        source_uses_dtype_variants(
            e,
            op_dir=getattr(ctx, "op_dir", ""),
            ops_root=getattr(ctx, "ops_root", ""),
            macros=ctx.kernel_defines() if hasattr(ctx, "kernel_defines") else None,
        )
        for e in entries
    ):
        all_variants = list((ctx.dtype_variants() or {}).get("values") or []) or [None]
    else:
        all_variants = [None]
    variants = _select_dtype_variants(all_variants, max_variants)
    ir.variants = [v or "default" for v in variants]
    if len(variants) < len(all_variants):
        ir.notes.append(
            f"dtype_variants_capped={len(variants)}/{len(all_variants)}"
        )
    dims = _Dimensions(list(dimensions or ()))

    found: dict[tuple[str, int, str], KernelBranch] = {}
    jobs: list[tuple[Path, str | None, dict[str, str] | None]] = [
        (entry, variant, None) for entry in entries for variant in variants
    ]
    mixed_assignment: dict[str, str] | None = None
    if variants and variants[0]:
        from ascendc_codemap_mcp.engine.kernel_gates import discover_kernel_gates

        gates = discover_kernel_gates(
            entries[0],
            op_dir=getattr(ctx, "op_dir", ""),
            ops_root=getattr(ctx, "ops_root", ""),
            macros=ctx.kernel_defines() if hasattr(ctx, "kernel_defines") else None,
        )
        mixed_assignment = gates.pick_mixed_orig_assignment(str(variants[0]))
        # ORIG=1 / MIXED=1 gates hide EnQue/DataCopy behind a second parse of
        # the same TU. Skipping it for speed left IFA with 13 operations.
        if mixed_assignment:
            for entry in entries:
                jobs.append((entry, variants[0], mixed_assignment))
            ir.notes.append(
                "mixed_orig_walk="
                + ",".join(f"{k}={v}" for k, v in sorted(mixed_assignment.items()))
            )

    import time as _time
    from ascendc_codemap_mcp.engine.timing import log as _tlog

    def _walk_one(job: tuple[Path, str | None, dict[str, str] | None]):
        entry, variant, orig_assignment = job
        t0 = _time.perf_counter()
        res = walk_file(
            entry,
            ctx,
            side="kernel",
            dtype_variant=variant,
            op_needle=getattr(spec, "op_needle", ""),
            scope=getattr(spec, "scope", None),
            collect_writes=False,
            orig_assignment=orig_assignment,
        )
        dt = _time.perf_counter() - t0
        label = variant or "default"
        if orig_assignment:
            label = f"{label}+mixed_orig"
        _tlog(
            f"{dt:7.3f}s{' SLOW' if dt > 180 else ''}  kernel_ir.walk  "
            f"entry={Path(entry).name} variant={label} "
            f"controls={len(getattr(res, 'controls', []) or [])}"
        )
        return label, res

    # libclang releases the GIL during parse — parallel variants cut wall time.
    if len(jobs) <= 1:
        walked = [_walk_one(j) for j in jobs]
    else:
        from concurrent.futures import ThreadPoolExecutor

        workers = min(len(jobs), 4)
        _tlog(f"kernel_ir.parallel  jobs={len(jobs)} workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            walked = list(pool.map(_walk_one, jobs))

    seen_uses: set[tuple[str, str, int, str]] = set()
    for label, res in walked:
        for node in res.controls:
            if node.kind != "if_constexpr":
                continue
            condition = (node.condition or "").strip()
            if not condition:
                continue
            key = (node.file, node.line, condition)
            branch = found.get(key)
            if branch is None:
                exact, derived, others = _classify(condition, dims)
                branch = KernelBranch(
                    condition=condition,
                    file=node.file,
                    line=node.line,
                    function=getattr(node, "function", "") or "",
                    dimensions=exact,
                    derived=derived,
                    symbols=others,
                )
                found[key] = branch
            if label not in branch.variants:
                branch.variants.append(label)
        for name, fr in (getattr(res, "functions", None) or {}).items():
            rec = ir.functions.setdefault(
                str(name),
                {
                    "file": str(getattr(fr, "file", "") or ""),
                    "line": int(getattr(fr, "line", 0) or 0),
                    "calls": [],
                    "params": list(getattr(fr, "params", None) or []),
                },
            )
            if getattr(fr, "file", "") and not rec.get("file"):
                rec["file"] = str(fr.file)
                rec["line"] = int(getattr(fr, "line", 0) or 0)
            if getattr(fr, "params", None) and not rec.get("params"):
                rec["params"] = list(fr.params)
            calls = rec.setdefault("calls", [])
            for callee, _args in getattr(fr, "calls", ()) or ():
                callee_s = str(callee)
                if callee_s and callee_s not in calls:
                    calls.append(callee_s)
        for key, decl in (getattr(res, "field_decls", None) or {}).items():
            host = str(getattr(decl, "host", "") or key[0] if isinstance(key, tuple) else "")
            member = str(getattr(decl, "name", "") or (key[1] if isinstance(key, tuple) else ""))
            if not host or not member:
                continue
            ir.field_decls.setdefault(
                (host, member),
                {
                    "type_text": str(getattr(decl, "type_text", "") or ""),
                    "file": str(getattr(decl, "file", "") or ""),
                    "line": int(getattr(decl, "line", 0) or 0),
                    "canonical_type": str(getattr(decl, "canonical_type", "") or ""),
                },
            )
        for use in getattr(res, "macro_uses", None) or []:
            if isinstance(use, dict):
                rec = {
                    "name": str(use.get("name") or ""),
                    "file": str(use.get("file") or ""),
                    "line": int(use.get("line") or 0),
                    "parent_name": str(use.get("parent_name") or ""),
                    "parent_kind": str(use.get("parent_kind") or ""),
                }
            else:
                rec = {
                    "name": str(getattr(use, "name", "") or ""),
                    "file": str(getattr(use, "file", "") or ""),
                    "line": int(getattr(use, "line", 0) or 0),
                    "parent_name": str(getattr(use, "parent_name", "") or ""),
                    "parent_kind": str(getattr(use, "parent_kind", "") or ""),
                }
            if not rec["name"]:
                continue
            key_use = (rec["name"], rec["file"], rec["line"], rec["parent_name"])
            if key_use in seen_uses:
                continue
            seen_uses.add(key_use)
            ir.macro_uses.append(rec)

    ir.branches = sorted(found.values(), key=lambda b: (b.file, b.line))
    named = sum(1 for b in ir.branches if b.dimensions or b.derived)
    ir.notes.append(
        f"entries={len(entries)} variants={len(variants)} "
        f"branches={len(ir.branches)} dimension_driven={named}"
    )
    silent = ir.silent_dimensions(list(dimensions or ()))
    if silent:
        ir.notes.append("no_branch_found_for: " + ", ".join(silent))
    return ir


def kernel_ir_isolate() -> bool:
    """Run KernelIR in a child process so host AST walks do not share the GIL.

    Windows thread-pool host walks otherwise serialize against the kernel walk.
    Override with ``UO_KERNEL_IR_ISOLATE=process`` or ``thread``.
    """
    import os

    raw = str(os.environ.get("UO_KERNEL_IR_ISOLATE") or "").strip().lower()
    if raw in {"process", "1", "true", "yes"}:
        return True
    if raw in {"thread", "0", "false", "no"}:
        return False
    return False


def kernel_ir_payload(spec, ctx, *, dimensions, max_variants) -> dict:
    """Pickle-safe snapshot for a KernelIR child process."""
    from ascendc_codemap_mcp.engine.build_context import BuildContext

    if isinstance(ctx, BuildContext):
        ctx_payload = ctx.to_dict()
    else:
        ctx_payload = {
            "raw": getattr(ctx, "raw", {}) or {},
            "cann_root": getattr(ctx, "cann_root", ""),
            "ops_root": getattr(ctx, "ops_root", ""),
            "compat_root": getattr(ctx, "compat_root", ""),
            "op_dir": getattr(ctx, "op_dir", ""),
            "arch_dir": getattr(ctx, "arch_dir", "") or "",
            "repo_root": getattr(ctx, "repo_root", ""),
        }
    scope = getattr(spec, "scope", None)
    return {
        "ctx": ctx_payload,
        "spec": {
            "kernel_targets": [
                str(p) for p in (getattr(spec, "kernel_targets", None) or []) if p
            ],
            "kernel_entry": (
                str(spec.kernel_entry) if getattr(spec, "kernel_entry", None) else ""
            ),
            "op_needle": getattr(spec, "op_needle", "") or "",
            "scope": scope.to_dict() if scope is not None else None,
            "op_dir": str(getattr(spec, "op_dir", "") or ""),
            "arch_dir": str(getattr(spec, "arch_dir", "") or ""),
        },
        "dimensions": list(dimensions or []),
        "max_variants": max_variants,
    }


def _kernel_ir_worker(payload: dict) -> KernelIR:
    """Rebuild spec/context and build KernelIR (pickle-safe child entry)."""
    from types import SimpleNamespace
    from pathlib import Path

    from ascendc_codemap_mcp.engine.build_context import BuildContext
    from ascendc_codemap_mcp.engine.scope_scan import ScopeSet

    ctx = BuildContext.from_dict(payload["ctx"])
    spec_d = dict(payload.get("spec") or {})
    scope = None
    raw_scope = spec_d.get("scope")
    if isinstance(raw_scope, dict):
        scope = ScopeSet.from_dict(raw_scope)
    entry = str(spec_d.get("kernel_entry") or "")
    spec = SimpleNamespace(
        kernel_targets=[Path(p) for p in (spec_d.get("kernel_targets") or [])],
        kernel_entry=Path(entry) if entry else None,
        op_needle=str(spec_d.get("op_needle") or ""),
        scope=scope,
        op_dir=Path(str(spec_d.get("op_dir") or "")),
        arch_dir=str(spec_d.get("arch_dir") or ""),
    )
    return build_kernel_ir(
        spec,
        ctx,
        dimensions=list(payload.get("dimensions") or []),
        max_variants=payload.get("max_variants"),
    )


def start_kernel_ir_job(payload: dict):
    """Spawn ``python -m uo_init.kernel_ir_job``; does not re-import caller ``__main__``."""
    import os
    import pickle
    import subprocess
    import sys
    from pathlib import Path

    spec_d = dict(payload.get("spec") or {})
    op_dir = Path(str(spec_d.get("op_dir") or "."))
    arch = str(spec_d.get("arch_dir") or "arch")
    job_dir = op_dir / ".ascendc-codemap" / arch / "cache"
    job_dir.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}_{id(payload)}"
    in_path = job_dir / f"kernel_ir_job_{token}.in.pkl"
    out_path = job_dir / f"kernel_ir_job_{token}.out.pkl"
    in_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    env = os.environ.copy()
    extra = [p for p in sys.path if p]
    old = str(env.get("PYTHONPATH") or "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([old] if old else []))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uo_init.kernel_ir_job", str(in_path), str(out_path)],
        env=env,
    )
    return proc, in_path, out_path


def finish_kernel_ir_job(proc, in_path, out_path) -> KernelIR:
    import pickle
    from pathlib import Path

    rc = proc.wait()
    try:
        if rc != 0:
            raise RuntimeError(f"kernel_ir worker exited {rc}")
        blob = Path(out_path).read_bytes()
        ir = pickle.loads(blob)
        if not isinstance(ir, KernelIR):
            raise TypeError(f"kernel_ir worker returned {type(ir).__name__}")
        return ir
    finally:
        for p in (in_path, out_path):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def run_kernel_ir_job(in_path: str, out_path: str) -> None:
    import pickle
    from pathlib import Path

    payload = pickle.loads(Path(in_path).read_bytes())
    ir = _kernel_ir_worker(payload)
    Path(out_path).write_bytes(pickle.dumps(ir, protocol=pickle.HIGHEST_PROTOCOL))
