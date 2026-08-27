# -*- coding: utf-8 -*-
"""In-memory AscendC CodeMap — single graph for Host/Kernel/Tiling/compile-time."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.evidence import (
    SOURCE_CLANG_AST,
    SOURCE_DSL,
    SOURCE_LEXICAL,
    SOURCE_UNSPECIFIED,
    STATE_CANDIDATE,
    STATE_RESOLVED,
    TRUST_ADVISORY,
    TRUST_AUTHORITATIVE,
    TRUST_DERIVED,
    assert_semantic_mint,
    derive_trust,
    merge_attrs,
    mint_payload,
    stamp_attrs,
)
from ascendc_codemap_mcp.engine.ir.relation import Relation, RelationKind


def append_write_site(attrs: dict[str, Any], *, file: str, line: int, rhs: str) -> None:
    """Keep every assignment site; last-write-wins RHS stays a separate attr."""
    loc = int(line or 0)
    if loc <= 0:
        return
    path = str(file or "").replace("\\", "/")
    site = {"file": path, "line": loc, "rhs": str(rhs or "")}
    sites = attrs.setdefault("write_sites", [])
    if not isinstance(sites, list):
        sites = []
        attrs["write_sites"] = sites
    key = (path, loc, site["rhs"])
    if any(
        (
            str(row.get("file") or "").replace("\\", "/"),
            int(row.get("line") or 0),
            str(row.get("rhs") or ""),
        )
        == key
        for row in sites
        if isinstance(row, dict)
    ):
        return
    sites.append(site)


def _is_truncated_kernel_branch(cond: str) -> bool:
    text = str(cond or "").strip()
    if not text:
        return False
    if "..." in text:
        return True
    if text.count("<") != text.count(">"):
        return True
    return len(text) > 80 and "<" in text


#: Host-check operands are looked up by the identifier clang recorded, not by
#: scanning source text. Ambiguous names are left unlinked.
_HOST_CHECK_OPERAND_KINDS = (
    EntityKind.INPUT,
    EntityKind.OUTPUT,
    EntityKind.TILING_FIELD,
    EntityKind.TILING_KEY,
    EntityKind.FIELD,
    EntityKind.VARIABLE,
    EntityKind.COMPILE_VAR,
)


def _unique_named(cm: "CodeMap", name: str, kinds: tuple[EntityKind, ...]) -> Entity | None:
    """The entity clang named, or None when the spelling is missing or clashes."""
    ident = str(name or "").strip()
    if not ident:
        return None
    hits: dict[str, Entity] = {}
    for kind in kinds:
        for ent in cm.by_name(ident, kind=kind):
            hits[ent.id] = ent
    if len(hits) == 1:
        return next(iter(hits.values()))
    if hits:
        return None
    if "." in ident:
        return _unique_named(cm, ident.rsplit(".", 1)[-1], kinds)
    return None


def _ingest_host_checks(cm: "CodeMap", host_ir: Any, ordinals: dict[tuple[str, str, str], int]) -> None:
    """Mint locatable Host BRANCH nodes for OP_CHECK / VALIDATION_ONLY controls.

    Edges hang off the containing function and the condition operands clang
    already resolved on the control node. No identifier is harvested from the
    condition string.
    """
    from ascendc_codemap_mcp.engine.ids import branch_id
    from ascendc_codemap_mcp.engine.source_layout import host_ir_keeps_file

    arch = str(getattr(cm, "architecture", "") or "")
    for node in getattr(host_ir, "controls", None) or []:
        universe = str(getattr(node, "universe", "") or "")
        snippet = str(getattr(node, "snippet", "") or "")
        cond = str(getattr(node, "condition", "") or "")
        haystack = f"{snippet} {cond}"
        if universe != "VALIDATION_ONLY" and "OP_CHECK" not in haystack and "OPS_CHECK" not in haystack:
            continue
        file = str(getattr(node, "file", "") or "")
        line = int(getattr(node, "line", 0) or 0)
        if not file or line <= 0:
            continue
        if not host_ir_keeps_file(file, arch):
            continue
        fn_name = str(getattr(node, "function", "") or "")
        guard = (cond or snippet).strip()[:120]
        if not guard:
            continue
        okey = (file, fn_name or "_check", guard)
        ordinal = ordinals.get(okey, 0)
        ordinals[okey] = ordinal + 1
        eid = branch_id(
            side="host",
            file=file,
            function=fn_name or "_check",
            guard=guard,
            ordinal=ordinal,
        )
        br = cm.upsert(
            EntityKind.BRANCH,
            guard,
            eid=eid,
            attrs={
                "layer": "host",
                "predicate": guard,
                "branch_kind": "host_check",
                "function": fn_name,
                "universe": universe or "VALIDATION_ONLY",
                "provenance": "clang_walk",
            },
            file=file,
            line=line,
            status="confirmed",
        )
        edge = {"provenance": "clang_walk", "file": file, "line": line}
        owner = _unique_named(cm, fn_name, (EntityKind.FUNCTION, EntityKind.METHOD))
        if owner is not None:
            cm.link(RelationKind.CONTROLS, br.id, owner.id, attrs=dict(edge))
            cm.link(RelationKind.GUARDED_BY, owner.id, br.id, attrs=dict(edge))
        for symbol in getattr(node, "reads", None) or ():
            target = _unique_named(cm, str(symbol), _HOST_CHECK_OPERAND_KINDS)
            if target is None or target.id == br.id:
                continue
            cm.link(RelationKind.READS, br.id, target.id, attrs={**edge, "symbol": str(symbol)})
            cm.link(RelationKind.GUARDED_BY, target.id, br.id, attrs={**edge, "symbol": str(symbol)})


def _ingest_host_write_events(
    cm: "CodeMap",
    events: Iterable[Any],
    ordinals: dict[tuple[str, str, str], int],
) -> None:
    """Upsert field/local writes with WRITES + GUARDED_BY. Skip evidence-poor rows."""
    from ascendc_codemap_mcp.engine.ids import branch_id

    try:
        from ascendc_codemap_mcp.engine.clang_walk import RETURN_SLOT
    except Exception:  # noqa: BLE001
        RETURN_SLOT = "__return__"

    from ascendc_codemap_mcp.engine.source_layout import host_ir_keeps_file

    for ev in events or []:
        path = str(getattr(ev, "path", "") or "").replace("->", ".")
        if not path or path == RETURN_SLOT:
            continue
        ev_file = str(getattr(ev, "file", "") or "")
        ev_line = int(getattr(ev, "line", 0) or 0)
        rhs = str(getattr(ev, "rhs", "") or "")
        if not ev_file or ev_line <= 0 or not rhs.strip():
            continue
        if not host_ir_keeps_file(ev_file, getattr(cm, "architecture", "") or ""):
            continue
        fn_name = str(getattr(ev, "function", "") or "")
        guards = list((ev.guards() if hasattr(ev, "guards") else []) or [])
        kind = EntityKind.FIELD if ("." in path) else EntityKind.VARIABLE
        field_attrs: dict[str, Any] = {
            "layer": "host",
            "rhs": rhs,
            "provenance": "clang_walk",
        }
        if guards and (not ev_file or ev_line <= 0):
            field_attrs["guards"] = [str(g)[:120] for g in guards]
        target = cm.upsert(
            kind,
            path,
            attrs=field_attrs,
            file=ev_file,
            line=ev_line,
        )
        append_write_site(target.attrs, file=ev_file, line=ev_line, rhs=rhs)
        if fn_name:
            fn = cm.upsert(EntityKind.FUNCTION, fn_name, attrs={"layer": "host", "provenance": "clang_walk"})
            rel = cm.link(
                RelationKind.WRITES,
                fn.id,
                target.id,
                attrs={"provenance": "clang_walk"},
            )
            append_write_site(rel.attrs, file=ev_file, line=ev_line, rhs=rhs)
            sites = rel.attrs.get("write_sites")
            if isinstance(sites, list) and sites:
                rel.attrs["sites"] = sites
        for guard in guards:
            gtext = str(guard or "").strip()
            if not gtext:
                continue
            okey = (ev_file, fn_name, gtext)
            ordinal = ordinals.get(okey, 0)
            ordinals[okey] = ordinal + 1
            eid = branch_id(
                side="host",
                file=ev_file,
                function=fn_name,
                guard=gtext,
                ordinal=ordinal,
            )
            br = cm.upsert(
                EntityKind.BRANCH,
                gtext[:120],
                eid=eid,
                attrs={
                    "layer": "host",
                    "predicate": gtext,
                    "branch_kind": "host_guard",
                    "function": fn_name,
                    "provenance": "clang_walk",
                },
                file=ev_file,
                line=ev_line,
                status="confirmed",
            )
            cm.link(
                RelationKind.GUARDED_BY,
                target.id,
                br.id,
                attrs={"provenance": "clang_walk"},
            )
            cm.link(
                RelationKind.CONTROLS,
                br.id,
                target.id,
                attrs={"provenance": "clang_walk"},
            )


# Legacy KB kind → CodeMap entity kind.
_KB_KIND_MAP: dict[str, EntityKind] = {
    "Variable": EntityKind.VARIABLE,
    "Field": EntityKind.FIELD,
    "Function": EntityKind.FUNCTION,
    "Method": EntityKind.METHOD,
    "File": EntityKind.FILE,
    "Type": EntityKind.TYPE,
    "Input": EntityKind.INPUT,
    "Output": EntityKind.OUTPUT,
    "Macro": EntityKind.MACRO,
    "CompileDefine": EntityKind.COMPILE_VAR,
    "CompileVar": EntityKind.COMPILE_VAR,
    "Template": EntityKind.TEMPLATE,
    "TemplateArg": EntityKind.TEMPLATE_ARG,
    "TemplateInstance": EntityKind.TEMPLATE_INSTANCE,
    "Branch": EntityKind.BRANCH,
    "Ctrl": EntityKind.BRANCH,
    "Predicate": EntityKind.PREDICATE,
    "TilingKey": EntityKind.TILING_KEY,
    "TilingKeyDim": EntityKind.TILING_KEY,
    "TilingField": EntityKind.TILING_FIELD,
    "TilingDataField": EntityKind.TILING_FIELD,
    "Kernel": EntityKind.KERNEL,
    "KernelBranch": EntityKind.BRANCH,
    "Arch": EntityKind.ARCH,
    "Architecture": EntityKind.ARCH,
    "BuildVariant": EntityKind.BUILD_VARIANT,
    "Operation": EntityKind.OPERATION,
    "Buffer": EntityKind.BUFFER,
    "Register": EntityKind.REGISTER,
    "Pipe": EntityKind.PIPE,
    "Event": EntityKind.EVENT,
    "Queue": EntityKind.QUEUE,
}

_KB_EDGE_MAP: dict[str, RelationKind] = {
    "DECLARES": RelationKind.DECLARES,
    "DEFINES": RelationKind.DEFINES,
    "REFERENCES": RelationKind.REFERENCES,
    "CALLS": RelationKind.CALLS,
    "READS": RelationKind.READS,
    "WRITES": RelationKind.WRITES,
    "DERIVES": RelationKind.DERIVES,
    "FLOWS_TO": RelationKind.FLOWS_TO,
    "CONTROLS": RelationKind.CONTROLS,
    "EXPANDS_TO": RelationKind.EXPANDS_TO,
    "GUARDED_BY": RelationKind.GUARDED_BY,
    "BINDS": RelationKind.BINDS,
    "INSTANTIATES": RelationKind.INSTANTIATES,
    "SPECIALIZES": RelationKind.SPECIALIZES,
    "SELECTS": RelationKind.SELECTS,
    "LAUNCHES": RelationKind.LAUNCHES,
    "AVAILABLE_ON": RelationKind.AVAILABLE_ON,
    "ACTIVE_UNDER": RelationKind.ACTIVE_UNDER,
    "SAVES": RelationKind.SAVES,
    "RESTORES": RelationKind.RESTORES,
    "CONTAINS": RelationKind.CONTAINS,
    "RETURNS": RelationKind.RETURNS,
    "ALIASES": RelationKind.ALIASES,
    "WRAPS": RelationKind.WRAPS,
    "ROOTED_AT": RelationKind.ROOTED_AT,
    "PRECEDES": RelationKind.PRECEDES,
    "SIGNALS": RelationKind.SIGNALS,
    "AWAITS": RelationKind.AWAITS,
    # Legacy KB edge names.
    "writes": RelationKind.WRITES,
    "reads": RelationKind.READS,
    "calls": RelationKind.CALLS,
    "controls": RelationKind.CONTROLS,
    "derives": RelationKind.DERIVES,
    "flows_to": RelationKind.FLOWS_TO,
    "selects": RelationKind.SELECTS,
    "binds": RelationKind.BINDS,
    "instantiates": RelationKind.INSTANTIATES,
    "guarded_by": RelationKind.GUARDED_BY,
}


def _eid(kind: str, name: str, *extra: str) -> str:
    raw = "|".join([kind, name, *extra])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"E_{kind}_{digest}"


def _rid(kind: str, src: str, dst: str) -> str:
    raw = f"{kind}|{src}|{dst}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"R_{kind}_{digest}"


class _NotifyMap(dict):
    """Dict that incrementally maintains CodeMap adjacency indexes.

    Passes ``pop`` / ``update`` / ``clear`` these maps directly. ``clear`` wipes
    the whole side of the index so in-place ``rel.src`` rewrites followed by
    ``clear``+``update`` stay consistent.
    """

    def __init__(self, owner: "CodeMap", role: str) -> None:
        super().__init__()
        self._owner = owner
        self._role = role

    def __setitem__(self, key, value) -> None:  # type: ignore[no-untyped-def]
        old = dict.get(self, key)
        if old is value:
            super().__setitem__(key, value)
            return
        owner = getattr(self, "_owner", None)
        if owner is not None and old is not None:
            owner._index_drop(self._role, old)
        super().__setitem__(key, value)
        if owner is not None:
            owner._index_add(self._role, value)

    def __delitem__(self, key) -> None:  # type: ignore[no-untyped-def]
        old = dict.get(self, key)
        super().__delitem__(key)
        owner = getattr(self, "_owner", None)
        if owner is not None and old is not None:
            owner._index_drop(self._role, old)

    def pop(self, key, *args):  # type: ignore[no-untyped-def]
        present = key in self
        old = dict.get(self, key) if present else None
        result = super().pop(key, *args) if args else super().pop(key)
        owner = getattr(self, "_owner", None)
        if owner is not None and present and old is not None:
            owner._index_drop(self._role, old)
        return result

    def popitem(self):  # type: ignore[no-untyped-def]
        key, old = super().popitem()
        owner = getattr(self, "_owner", None)
        if owner is not None:
            owner._index_drop(self._role, old)
        return key, old

    def clear(self) -> None:
        super().clear()
        owner = getattr(self, "_owner", None)
        if owner is not None:
            owner._index_clear(self._role)

    def update(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        incoming = dict(*args, **kwargs)
        for key, value in incoming.items():
            self[key] = value

    def setdefault(self, key, default=None):  # type: ignore[no-untyped-def]
        if key not in self:
            self[key] = default
        return self[key]


@dataclass
class CodeMap:
    """Unified operator CodeMap (Host + Kernel + compile-time overlay)."""

    op_name: str = ""
    architecture: str = ""
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._install_index_maps(dict(self.entities), dict(self.relations))

    def _install_index_maps(
        self,
        entities: dict[str, Entity] | None = None,
        relations: dict[str, Relation] | None = None,
    ) -> None:
        self._by_kind: dict[str, dict[str, Entity]] = defaultdict(dict)
        self._by_name: dict[tuple[str, str], dict[str, Entity]] = defaultdict(dict)
        self._out: dict[str, dict[str, Relation]] = defaultdict(dict)
        self._in: dict[str, dict[str, Relation]] = defaultdict(dict)
        ents = dict(self.entities if entities is None else entities)
        rels = dict(self.relations if relations is None else relations)
        self.entities = _NotifyMap(self, "entity")
        self.relations = _NotifyMap(self, "rel")
        for key, value in ents.items():
            self.entities[key] = value
        for key, value in rels.items():
            self.relations[key] = value

    def _index_add(self, role: str, value: Any) -> None:
        if role == "entity":
            kind = value.kind_name()
            self._by_kind[kind][value.id] = value
            self._by_name[(kind, value.name)][value.id] = value
            return
        self._out[value.src][value.id] = value
        self._in[value.dst][value.id] = value

    def _index_drop(self, role: str, value: Any) -> None:
        if role == "entity":
            kind = value.kind_name()
            bucket = self._by_kind.get(kind)
            if bucket is not None:
                bucket.pop(value.id, None)
            named = self._by_name.get((kind, value.name))
            if named is not None:
                named.pop(value.id, None)
            return
        outgoing = self._out.get(value.src)
        if outgoing is not None:
            outgoing.pop(value.id, None)
        incoming = self._in.get(value.dst)
        if incoming is not None:
            incoming.pop(value.id, None)

    def _index_clear(self, role: str) -> None:
        if role == "entity":
            self._by_kind.clear()
            self._by_name.clear()
            return
        self._out.clear()
        self._in.clear()

    def _index_rekey_entity(self, entity: Entity, old_name: str) -> None:
        kind = entity.kind_name()
        named = self._by_name.get((kind, old_name))
        if named is not None:
            named.pop(entity.id, None)
        self._by_name[(kind, entity.name)][entity.id] = entity

    def __getstate__(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "architecture": self.architecture,
            "entities": dict(self.entities),
            "relations": dict(self.relations),
            "meta": dict(self.meta),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.op_name = str(state.get("op_name") or "")
        self.architecture = str(state.get("architecture") or "")
        self.meta = dict(state.get("meta") or {})
        self.entities = {}
        self.relations = {}
        self._install_index_maps(state.get("entities") or {}, state.get("relations") or {})

    # -- mutation ----------------------------------------------------------
    def add_entity(self, entity: Entity, *, stamp: bool = True) -> Entity:
        existing = self.entities.get(entity.id)
        if existing is None:
            if stamp:
                entity.attrs = self._stamp_attrs(entity.attrs)
            self.entities[entity.id] = entity
            return entity
        old_name = existing.name
        old_sites = list(existing.attrs.get("definition_sites") or []) if isinstance(existing.attrs.get("definition_sites"), list) else []
        existing.attrs = merge_attrs(existing.attrs, entity.attrs)
        if old_sites:
            existing.attrs["definition_sites"] = old_sites
        if entity.name and not existing.name:
            existing.name = entity.name
        self._merge_definition_sites(existing, entity)
        self._widen_locus(existing, entity)
        self._merge_write_sites(existing, entity)
        settled = str(existing.attrs.get("root_status") or "")
        if settled in {"REACHED", "PROJECT", "BUILTIN"}:
            existing.status = "extracted"
            existing.confidence = max(float(existing.confidence or 0.0), float(entity.confidence or 0.0))
        elif str(entity.status or "") == "extracted" and str(existing.status or "").lower() in {
            "partial",
            "unresolved",
            "unknown",
            "not_extracted",
            "",
        }:
            existing.status = "extracted"
            existing.confidence = max(float(existing.confidence or 0.0), float(entity.confidence or 0.0))
        if existing.name != old_name:
            self._index_rekey_entity(existing, old_name)
        return existing

    @staticmethod
    def _merge_definition_sites(existing: Entity, incoming: Entity) -> None:
        kind = existing.kind.value if isinstance(existing.kind, EntityKind) else str(existing.kind)
        if kind not in {EntityKind.FUNCTION.value, EntityKind.METHOD.value}:
            return
        sites: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()

        def _add(file: str, line: int) -> None:
            path = str(file or "").replace("\\", "/")
            loc = int(line or 0)
            if not path or loc <= 0 or (path, loc) in seen:
                return
            seen.add((path, loc))
            sites.append({"file": path, "line": loc, "line_start": loc})

        _add(existing.file, existing.line_start)
        _add(incoming.file, incoming.line_start)
        for blob in (
            existing.attrs.get("definition_sites"),
            incoming.attrs.get("definition_sites"),
        ):
            if not isinstance(blob, list):
                continue
            for site in blob:
                if isinstance(site, dict):
                    _add(str(site.get("file") or ""), int(site.get("line") or site.get("line_start") or 0))
        if len(sites) > 1:
            existing.attrs["definition_sites"] = sites

    @staticmethod
    def _widen_locus(existing: Entity, incoming: Entity) -> None:
        """Definition span only expands. Point write/call sites do not relocate it."""
        kind = existing.kind.value if isinstance(existing.kind, EntityKind) else str(existing.kind)
        inc_start = int(incoming.line_start or 0)
        inc_end = int(incoming.line_end or 0)
        if inc_end < inc_start:
            inc_end = inc_start
        inc_span = (inc_end - inc_start) if inc_start > 0 else 0
        inc_point = inc_start > 0 and inc_end <= inc_start
        ex_start = int(existing.line_start or 0)
        ex_end = int(existing.line_end or 0)
        ex_span = (ex_end - ex_start) if ex_start > 0 else 0
        def_kinds = {
            EntityKind.FUNCTION.value,
            EntityKind.METHOD.value,
            EntityKind.KERNEL.value,
        }
        if kind in def_kinds:
            if inc_point:
                if not existing.file and incoming.file:
                    existing.file = incoming.file
                    existing.line_start = inc_start
                    existing.line_end = inc_end
                return
            if inc_span > ex_span:
                # A range is only meaningful together with the file it was read
                # from. Adopting the range while keeping the previous file used
                # to graft one file's span onto another, producing spans that
                # run past end-of-file.
                if not incoming.file:
                    return
                existing.file = incoming.file
                existing.line_start = inc_start
                existing.line_end = inc_end
                return
            if incoming.file and existing.file == incoming.file:
                if ex_start <= 0 and inc_start > 0:
                    existing.line_start = inc_start
                if inc_end > ex_end:
                    existing.line_end = inc_end
            elif not existing.file and incoming.file:
                existing.file = incoming.file
                existing.line_start = inc_start
                existing.line_end = inc_end
            return
        if incoming.file and not existing.file:
            existing.file = incoming.file
            existing.line_start = inc_start
            existing.line_end = inc_end
            return
        if incoming.file and existing.file == incoming.file:
            if kind not in {EntityKind.BRANCH.value, EntityKind.OPERATION.value}:
                # Non-definition kinds are keyed by name alone, so two locals
                # called `dim0` in different functions land on one entity. Only
                # extend the span when the incoming range touches the existing
                # one, i.e. it is the same declaration seen more completely.
                # A disjoint occurrence is a separate site and is already kept
                # in write_sites; unioning it produced spans covering most of
                # the file, which then swallowed every text-recall hit in it.
                if inc_end > ex_end and inc_start <= ex_end + 1:
                    existing.line_end = inc_end

    @staticmethod
    def _merge_write_sites(existing: Entity, incoming: Entity) -> None:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()

        def _add(site: Any) -> None:
            if not isinstance(site, dict):
                return
            file = str(site.get("file") or "").replace("\\", "/")
            line = int(site.get("line") or 0)
            rhs = str(site.get("rhs") or "")
            if line <= 0 or (file, line, rhs) in seen:
                return
            seen.add((file, line, rhs))
            merged.append({"file": file, "line": line, "rhs": rhs})

        for blob in (existing.attrs.get("write_sites"), incoming.attrs.get("write_sites")):
            if isinstance(blob, list):
                for site in blob:
                    _add(site)
        if merged:
            existing.attrs["write_sites"] = merged

    def upsert(
        self,
        kind: EntityKind | str,
        name: str,
        *,
        eid: str | None = None,
        attrs: dict[str, Any] | None = None,
        file: str = "",
        line: int = 0,
        line_end: int | None = None,
        status: str = "extracted",
        confidence: float = 1.0,
    ) -> Entity:
        kind_name = kind.value if isinstance(kind, EntityKind) else str(kind)
        attrs_doc = dict(attrs or {})
        start = int(line or 0)
        end = int(line_end) if line_end is not None and int(line_end) > 0 else start
        if end < start:
            end = start

        # A selected source Kernel is first materialised from its verified
        # __global__ signature, then the generic body scanner sees the same
        # definition as a free FUNCTION.  Forking those identities moves CALLS
        # and READS off the actual Kernel and makes entry reachability false.
        # Reuse only the exact, source-verified same-name Kernel case; ordinary
        # same-name functions remain distinct entities.
        if (
            eid is not None
            and kind_name == EntityKind.FUNCTION.value
            and attrs_doc.get("provenance") == "source_kernel_definition"
        ):
            kernels = [
                ent
                for ent in self.by_name(name, kind=EntityKind.KERNEL)
                if ent.attrs.get("source_signature") is True
                or ent.attrs.get("provenance") == "source_kernel_signature"
            ]
            if len(kernels) == 1:
                entity = kernels[0]
                entity.attrs = merge_attrs(entity.attrs, self._stamp_attrs(attrs_doc))
                if file:
                    entity.file = file
                    entity.line_start = start
                    entity.line_end = max(int(entity.line_end or 0), end)
                entity.status = status
                entity.confidence = confidence
                return entity

        entity_id = eid or _eid(kind_name, name)
        return self.add_entity(
            Entity(
                id=entity_id,
                kind=kind,
                name=name,
                attrs=self._stamp_attrs(attrs_doc),
                file=file,
                line_start=start,
                line_end=end,
                status=status,
                confidence=confidence,
            )
        )

    def _stamp_attrs(self, attrs: dict[str, Any] | None) -> dict[str, Any]:
        return stamp_attrs(
            attrs,
            build_context_id=str(self.meta.get("build_context_id") or ""),
        )

    def link(
        self,
        kind: RelationKind | str,
        src: str,
        dst: str,
        *,
        attrs: dict[str, Any] | None = None,
        status: str = "extracted",
        confidence: float = 1.0,
    ) -> Relation:
        kind_name = kind.value if isinstance(kind, RelationKind) else str(kind)
        rid = _rid(kind_name, src, dst)
        stamped = self._stamp_attrs(dict(attrs or {}))
        existing = self.relations.get(rid)
        if existing is None:
            rel = Relation(
                id=rid,
                kind=kind,
                src=src,
                dst=dst,
                attrs=stamped,
                status=status,
                confidence=confidence,
            )
            self.relations[rid] = rel
            return rel
        existing.attrs = merge_attrs(existing.attrs, stamped)
        return existing

    def mint_semantic_relation(
        self,
        kind: RelationKind | str,
        src: str,
        dst: str,
        *,
        provenance: str,
        source: str = SOURCE_CLANG_AST,
        extra: dict[str, Any] | None = None,
        evidence_ids: list[str] | None = None,
        status: str = "confirmed",
    ) -> Relation:
        trust = TRUST_AUTHORITATIVE if source == SOURCE_CLANG_AST else TRUST_DERIVED
        assert_semantic_mint(source=source, trust=trust)
        payload = mint_payload(
            provenance=provenance,
            source=source,
            trust=trust,
            semantic_state=STATE_RESOLVED,
            build_context_id=str(self.meta.get("build_context_id") or ""),
            extra=extra,
            evidence_ids=evidence_ids,
        )
        return self.link(kind, src, dst, attrs=payload, status=status)

    def mint_candidate_relation(
        self,
        kind: RelationKind | str,
        src: str,
        dst: str,
        *,
        provenance: str,
        extra: dict[str, Any] | None = None,
        status: str = "confirmed",
        source: str = SOURCE_LEXICAL,
    ) -> Relation:
        payload = mint_payload(
            provenance=provenance,
            source=source,
            trust=TRUST_ADVISORY,
            semantic_state=STATE_CANDIDATE,
            build_context_id=str(self.meta.get("build_context_id") or ""),
            extra=extra,
        )
        return self.link(kind, src, dst, attrs=payload, status=status)

    def derive_relation(
        self,
        kind: RelationKind | str,
        src: str,
        dst: str,
        *,
        provenance: str,
        rule: str,
        input_ids: list[str],
        extra: dict[str, Any] | None = None,
        status: str = "confirmed",
    ) -> Relation:
        input_trusts: list[str] = []
        for fid in input_ids:
            node = self.entities.get(fid)
            if node is not None:
                input_trusts.append(str(node.attrs.get("trust") or TRUST_ADVISORY))
                continue
            edge = self.relations.get(fid)
            if edge is not None:
                input_trusts.append(str(edge.attrs.get("trust") or TRUST_ADVISORY))
                continue
            input_trusts.append(TRUST_ADVISORY)
        trust = derive_trust(input_trusts)
        if trust == TRUST_AUTHORITATIVE:
            trust = TRUST_DERIVED
            source = SOURCE_DSL
        elif trust == TRUST_DERIVED:
            source = SOURCE_DSL
        elif trust == TRUST_ADVISORY:
            source = SOURCE_LEXICAL
        else:
            source = SOURCE_UNSPECIFIED
        payload = mint_payload(
            provenance=provenance,
            source=source,
            trust=trust,
            semantic_state=STATE_RESOLVED,
            build_context_id=str(self.meta.get("build_context_id") or ""),
            extra=extra,
            evidence_ids=input_ids,
            derivation={"rule": rule, "inputs": list(input_ids)},
        )
        return self.link(kind, src, dst, attrs=payload, status=status)

    # -- query helpers -----------------------------------------------------
    def by_kind(self, kind: EntityKind | str) -> list[Entity]:
        name = kind.value if isinstance(kind, EntityKind) else str(kind)
        return list((self._by_kind.get(name) or {}).values())

    def by_name(self, name: str, *, kind: EntityKind | str | None = None) -> list[Entity]:
        if kind is not None:
            kn = kind.value if isinstance(kind, EntityKind) else str(kind)
            return list((self._by_name.get((kn, name)) or {}).values())
        out: list[Entity] = []
        for (kind_name, ent_name), ents in self._by_name.items():
            if ent_name == name:
                out.extend(ents.values())
        return out

    def neighbors(
        self,
        entity_id: str,
        *,
        kind: RelationKind | str | None = None,
        direction: str = "out",
        include_advisory: bool = True,
    ) -> list[tuple[Relation, Entity]]:
        kn = None
        if kind is not None:
            kn = kind.value if isinstance(kind, RelationKind) else str(kind)
        hits: list[tuple[Relation, Entity]] = []
        if direction in ("out", "both"):
            for rel in (self._out.get(entity_id) or {}).values():
                if kn is not None and rel.kind_name() != kn:
                    continue
                if not include_advisory and str(rel.attrs.get("trust") or "") == TRUST_ADVISORY:
                    continue
                dst = self.entities.get(rel.dst)
                if dst is not None:
                    hits.append((rel, dst))
        if direction in ("in", "both"):
            for rel in (self._in.get(entity_id) or {}).values():
                if kn is not None and rel.kind_name() != kn:
                    continue
                if not include_advisory and str(rel.attrs.get("trust") or "") == TRUST_ADVISORY:
                    continue
                src = self.entities.get(rel.src)
                if src is not None:
                    hits.append((rel, src))
        return hits

    def has_incident(self, entity_id: str) -> bool:
        """True if any live relation mentions ``entity_id``."""
        return bool(self._out.get(entity_id) or self._in.get(entity_id))

    def find_path(
        self,
        start_id: str,
        *,
        end_kinds: Iterable[str] | None = None,
        end_id: str | None = None,
        max_depth: int = 32,
        include_advisory: bool = True,
    ) -> list[str]:
        """BFS path of entity ids from start to end_id or first end_kind."""
        ends = {str(k) for k in (end_kinds or ())}
        prev: dict[str, str | None] = {start_id: None}
        q: deque[str] = deque([start_id])
        found: str | None = None
        while q:
            cur = q.popleft()
            ent = self.entities.get(cur)
            if end_id and cur == end_id:
                found = cur
                break
            if ends and ent is not None and ent.kind_name() in ends:
                found = cur
                break
            if len(prev) > 1 and (len(prev) // 2) > max_depth * 64:
                break
            depth = 0
            walk = cur
            while prev.get(walk) is not None:
                depth += 1
                walk = prev[walk]  # type: ignore[assignment]
                if depth > max_depth:
                    break
            if depth > max_depth:
                continue
            for rel in (self._out.get(cur) or {}).values():
                if not include_advisory and str(rel.attrs.get("trust") or "") == TRUST_ADVISORY:
                    continue
                nxt = rel.dst
                if nxt in prev:
                    continue
                prev[nxt] = cur
                q.append(nxt)
        if found is None:
            return []
        path: list[str] = []
        cur2: str | None = found
        while cur2 is not None:
            path.append(cur2)
            cur2 = prev.get(cur2)
        path.reverse()
        return path

    def host_kernel_path_exists(self) -> bool:
        inputs = self.by_kind(EntityKind.INPUT)
        if not inputs:
            # Fallback: VARIABLE named like inputs also count for adapters.
            inputs = [e for e in self.entities.values() if e.kind_name() in {"INPUT", "VARIABLE"}]
        for inp in inputs[:32]:
            # Prefer a full path to KERNEL; fall back to key/instance reachability.
            to_kernel = self.find_path(inp.id, end_kinds={"KERNEL"})
            if len(to_kernel) >= 2:
                return True
            path = self.find_path(
                inp.id,
                end_kinds={"TILING_KEY", "TEMPLATE_INSTANCE"},
            )
            if len(path) >= 2 and (
                self.by_kind(EntityKind.KERNEL)
                or self.by_kind(EntityKind.TEMPLATE_INSTANCE)
            ):
                return True
        kernels = self.by_kind(EntityKind.KERNEL)
        keys = self.by_kind(EntityKind.TILING_KEY)
        return bool(kernels) and (bool(keys) or bool(self.by_kind(EntityKind.TEMPLATE_INSTANCE)))

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = defaultdict(int)
        for e in self.entities.values():
            by_kind[e.kind_name()] += 1
        by_rel: dict[str, int] = defaultdict(int)
        for r in self.relations.values():
            by_rel[r.kind_name()] += 1
        return {
            "op_name": self.op_name,
            "architecture": self.architecture,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entities_by_kind": dict(sorted(by_kind.items())),
            "relations_by_kind": dict(sorted(by_rel.items())),
            "has_host": bool(
                self.by_kind(EntityKind.FUNCTION)
                or self.by_kind(EntityKind.VARIABLE)
                or self.by_kind(EntityKind.FIELD)
            ),
            "has_kernel": bool(self.by_kind(EntityKind.KERNEL)),
            "has_host_kernel_path": self.host_kernel_path_exists(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "codemap/v1",
            "op_name": self.op_name,
            "architecture": self.architecture,
            "meta": dict(self.meta),
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
            "summary": self.summary(),
        }

    # -- adapters from legacy IR -------------------------------------------
    @classmethod
    def from_host_ir(
        cls,
        host_ir: Any,
        *,
        op_name: str = "",
        architecture: str = "",
        codemap: "CodeMap | None" = None,
    ) -> "CodeMap":
        cm = codemap or cls(op_name=op_name, architecture=architecture)
        if op_name:
            cm.op_name = op_name
        if architecture:
            cm.architecture = architecture

        from ascendc_codemap_mcp.engine.source_layout import host_ir_keeps_file

        for name, summary in (getattr(host_ir, "summaries", None) or {}).items():
            start = int(getattr(summary, "line", 0) or 0)
            end = int(getattr(summary, "line_end", 0) or 0)
            summary_file = str(getattr(summary, "file", "") or "")
            if not summary_file or not host_ir_keeps_file(summary_file, architecture):
                continue
            fn = cm.upsert(
                EntityKind.FUNCTION,
                str(name),
                attrs={"layer": "host", "provenance": "clang_walk"},
                file=str(getattr(summary, "file", "") or ""),
                line=start,
                line_end=end if end > start else None,
            )
            for callee, _args in getattr(summary, "calls", None) or []:
                other = cm.upsert(
                    EntityKind.FUNCTION,
                    str(callee),
                    attrs={"layer": "host", "provenance": "clang_walk"},
                )
                cm.link(
                    RelationKind.CALLS,
                    fn.id,
                    other.id,
                    attrs={"provenance": "clang_walk"},
                )
            for w in getattr(summary, "writes", None) or []:
                field_e = cm.upsert(
                    EntityKind.FIELD, str(w), attrs={"layer": "host", "provenance": "clang_walk"}
                )
                cm.link(
                    RelationKind.WRITES,
                    fn.id,
                    field_e.id,
                    attrs={"provenance": "clang_walk"},
                )
            for r in getattr(summary, "reads", None) or []:
                var_e = cm.upsert(
                    EntityKind.VARIABLE,
                    str(r),
                    attrs={"layer": "host", "provenance": "clang_walk"},
                )
                cm.link(
                    RelationKind.READS,
                    fn.id,
                    var_e.id,
                    attrs={"provenance": "clang_walk"},
                )

        host_branch_ordinals: dict[tuple[str, str, str], int] = {}
        _ingest_host_write_events(cm, getattr(host_ir, "writes", None) or [], host_branch_ordinals)
        _ingest_host_write_events(
            cm, getattr(host_ir, "local_writes", None) or [], host_branch_ordinals
        )

        for site in getattr(host_ir, "call_sites", None) or []:
            caller_name = str(getattr(site, "caller", "") or "")
            callee_name = str(getattr(site, "callee", "") or "")
            site_file = str(getattr(site, "file", "") or "")
            if not caller_name or not callee_name:
                continue
            if not site_file or not host_ir_keeps_file(site_file, architecture):
                continue
            caller = cm.upsert(
                EntityKind.FUNCTION,
                caller_name,
                attrs={"layer": "host", "provenance": "clang_walk"},
            )
            callee = cm.upsert(
                EntityKind.FUNCTION,
                callee_name,
                attrs={"layer": "host", "provenance": "clang_walk"},
            )
            rel = cm.link(
                RelationKind.CALLS,
                caller.id,
                callee.id,
                attrs={"provenance": "clang_walk"},
            )
            site_line = int(getattr(site, "line", 0) or 0)
            if site_line > 0:
                rel.attrs["file"] = site_file
                rel.attrs["line"] = site_line

        _ingest_host_checks(cm, host_ir, host_branch_ordinals)
        from ascendc_codemap_mcp.engine.passes.host_graph_status import enrich_host_graph_status

        enrich_host_graph_status(cm, host_ir)

        cm.meta["host_backend"] = str(getattr(host_ir, "backend", "") or "")
        return cm

    @classmethod
    def from_kernel_ir(
        cls,
        kernel_ir: Any,
        *,
        op_name: str = "",
        architecture: str = "",
        codemap: "CodeMap | None" = None,
        op_root: str = "",
    ) -> "CodeMap":
        cm = codemap or cls(op_name=op_name, architecture=architecture)
        if architecture:
            arch = cm.upsert(EntityKind.ARCH, architecture, attrs={"layer": "arch"})
        else:
            arch = None

        # Prefer already-verified KERNEL entities (source signature / tiling
        # closure). Never mint a span-less dummy KERNEL from the op name alone.
        verified_kernels = [
            e
            for e in cm.by_kind(EntityKind.KERNEL)
            if e.attrs.get("source_signature")
            or e.attrs.get("source_definition")
            or str(e.attrs.get("provenance") or "").startswith("source_kernel")
            or (e.file and int(e.line_start or 0) > 0)
        ]
        variants = list(getattr(kernel_ir, "variants", None) or [])
        for kernel in verified_kernels:
            if variants:
                existing = list(kernel.attrs.get("variants") or [])
                for v in variants:
                    if v not in existing:
                        existing.append(v)
                kernel.attrs["variants"] = existing
            if arch is not None:
                cm.link(RelationKind.AVAILABLE_ON, kernel.id, arch.id)

        mint = getattr(kernel_ir, "mint_ids", None)
        if callable(mint):
            mint(op_root or "")

        for br in getattr(kernel_ir, "branches", None) or []:
            bid = str(getattr(br, "id", "") or "").strip()
            cond = str(getattr(br, "condition", "") or "")
            br_file = str(getattr(br, "file", "") or "")
            br_line = int(getattr(br, "line", 0) or 0)
            if not bid or not br_file or br_line <= 0:
                continue
            if _is_truncated_kernel_branch(cond):
                continue
            ident = re.search(r"\b(IS_[A-Z0-9_]+|[A-Z][A-Z0-9_]{2,})\b", cond)
            name = ident.group(1) if ident else (cond[:120] or bid)
            branch = cm.upsert(
                EntityKind.BRANCH,
                name,
                eid=bid,
                attrs={
                    "layer": "kernel",
                    "condition": cond[:200],
                    "dimensions": list(getattr(br, "dimensions", None) or []),
                    "variants": list(getattr(br, "variants", None) or []),
                    "function": str(getattr(br, "function", "") or ""),
                    "provenance": "clang_kernel_branch",
                },
                file=br_file,
                line=br_line,
                status="confirmed",
            )
            for kernel in verified_kernels:
                # Every branch is linked to every verified kernel: which kernel
                # a branch sits in is not in the IR, so this is a scope guess.
                cm.link(
                    RelationKind.CONTROLS,
                    branch.id,
                    kernel.id,
                    attrs={"provenance": "source_kernel_branch_scope"},
                )
            for dim in getattr(br, "dimensions", None) or []:
                key = cm.upsert(
                    EntityKind.TILING_KEY,
                    str(dim),
                    attrs={"layer": "tiling", "provenance": "kernel_branch_dimension"},
                )
                cm.link(
                    RelationKind.SELECTS,
                    key.id,
                    branch.id,
                    attrs={"provenance": "kernel_branch_dimension"},
                )
                for kernel in verified_kernels:
                    cm.link(
                        RelationKind.SELECTS,
                        key.id,
                        kernel.id,
                        attrs={"provenance": "kernel_branch_dimension"},
                    )
        cm.meta["kernel_ir_variants"] = variants
        return cm

    @classmethod
    def from_tiling_data_ir(
        cls,
        tiling_ir: Any,
        *,
        op_name: str = "",
        architecture: str = "",
        codemap: "CodeMap | None" = None,
    ) -> "CodeMap":
        cm = codemap or cls(op_name=op_name, architecture=architecture)
        structs = getattr(tiling_ir, "structs", None) or {}
        # TilingDataIR may expose .fields or iterate structs.
        fields: list[Any] = list(getattr(tiling_ir, "fields", None) or [])
        if not fields and isinstance(structs, dict):
            for st in structs.values():
                fields.extend(getattr(st, "fields", None) or [])
        for f in fields:
            name = str(getattr(f, "name", "") or "")
            if not name:
                continue
            cm.upsert(
                EntityKind.TILING_FIELD,
                name,
                attrs={
                    "layer": "tiling",
                    "ctype": str(getattr(f, "ctype", "") or ""),
                    "struct": str(getattr(f, "struct", "") or ""),
                },
                file=str(getattr(f, "file", "") or ""),
                line=int(getattr(f, "line", 0) or 0),
            )
        return cm

    @classmethod
    def from_kb(
        cls,
        kb: Any,
        *,
        codemap: "CodeMap | None" = None,
    ) -> "CodeMap":
        cm = codemap or cls(
            op_name=str(getattr(kb, "op_name", "") or ""),
            architecture=str(getattr(kb, "architecture", "") or ""),
        )
        for node in (getattr(kb, "nodes", None) or {}).values():
            kind_raw = str(getattr(node, "kind", "") or "OTHER")
            kind = _KB_KIND_MAP.get(kind_raw, EntityKind.OTHER)
            ev0 = (getattr(node, "evidence", None) or [None])[0]
            cm.add_entity(
                Entity(
                    id=str(node.id),
                    kind=kind,
                    name=str(getattr(node, "name", "") or ""),
                    attrs={
                        "layer": str(getattr(node, "layer", "") or ""),
                        "legacy_kind": kind_raw,
                        **dict(getattr(node, "data", None) or {}),
                    },
                    file=str(getattr(ev0, "file", "") or "") if ev0 else "",
                    line_start=int(getattr(ev0, "line_start", 0) or 0) if ev0 else 0,
                    line_end=int(getattr(ev0, "line_end", 0) or 0) if ev0 else 0,
                    status=str(getattr(node, "status", "extracted") or "extracted"),
                    confidence=float(getattr(node, "confidence", 1.0) or 1.0),
                )
            )
        for edge in (getattr(kb, "edges", None) or {}).values():
            kind_raw = str(getattr(edge, "kind", "") or "OTHER")
            kind = _KB_EDGE_MAP.get(kind_raw, RelationKind.OTHER)
            cm.link(
                kind,
                str(edge.src),
                str(edge.dst),
                attrs={"legacy_kind": kind_raw, **dict(getattr(edge, "data", None) or {})},
                status=str(getattr(edge, "status", "extracted") or "extracted"),
                confidence=float(getattr(edge, "confidence", 1.0) or 1.0),
            )
        return cm
