# -*- coding: utf-8 -*-
"""CodeMap query engine — Agent-facing API over ``.uo`` / in-memory CodeMap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.store.reader import find_uo_product, read_codemap


@dataclass
class CodeMapQuery:
    codemap: CodeMap
    path: str = ""

    def find_symbol(self, name: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        ents = self.codemap.by_name(name, kind=kind)
        if not ents:
            ents = [
                e
                for e in self.codemap.entities.values()
                if name in e.name and (kind is None or e.kind_name() == kind)
            ]
        return [e.to_dict() for e in ents]

    def definition(self, name: str) -> dict[str, Any] | None:
        hits = self.find_symbol(name)
        return hits[0] if hits else None

    def references(self, name: str) -> list[dict[str, Any]]:
        targets = {e["id"] for e in self.find_symbol(name)}
        out: list[dict[str, Any]] = []
        for rel in self.codemap.relations.values():
            if rel.kind_name() in {"REFERENCES", "READS", "WRITES", "CALLS"} and (
                rel.src in targets or rel.dst in targets
            ):
                out.append(rel.to_dict())
        return out

    def callers(self, name: str) -> list[dict[str, Any]]:
        return self._adj(name, RelationKind.CALLS, direction="in")

    def callees(self, name: str) -> list[dict[str, Any]]:
        return self._adj(name, RelationKind.CALLS, direction="out")

    def writers(self, name: str) -> list[dict[str, Any]]:
        return self._adj(name, RelationKind.WRITES, direction="in")

    def readers(self, name: str) -> list[dict[str, Any]]:
        return self._adj(name, RelationKind.READS, direction="in")

    def upstream(self, name: str, *, limit: int = 32) -> list[dict[str, Any]]:
        return self._walk(name, direction="in", limit=limit)

    def downstream(self, name: str, *, limit: int = 32) -> list[dict[str, Any]]:
        return self._walk(name, direction="out", limit=limit)

    def slice_forward(
        self,
        seed_ids: list[str],
        *,
        edge_kinds: list[str] | None = None,
        depth: int = 3,
        budget: int = 500,
        include_advisory: bool = False,
    ) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.query.slice import slice_forward

        return slice_forward(
            self.codemap,
            seed_ids,
            edge_kinds=edge_kinds,
            depth=depth,
            budget=budget,
            include_advisory=include_advisory,
        )

    def slice_backward(
        self,
        seed_ids: list[str],
        *,
        edge_kinds: list[str] | None = None,
        depth: int = 3,
        budget: int = 500,
        include_advisory: bool = False,
    ) -> dict[str, Any]:
        from ascendc_codemap_mcp.engine.query.slice import slice_backward

        return slice_backward(
            self.codemap,
            seed_ids,
            edge_kinds=edge_kinds,
            depth=depth,
            budget=budget,
            include_advisory=include_advisory,
        )

    def find_path(self, start: str, end: str | None = None, *, end_kind: str = "") -> list[dict[str, Any]]:
        start_ents = self.codemap.by_name(start)
        if not start_ents:
            start_ents = [
                e for e in self.codemap.entities.values() if e.name.endswith(start) or start in e.name
            ]
        if not start_ents:
            return []
        rank = {
            "INPUT": 0,
            "OUTPUT": 1,
            "TILING_KEY": 2,
            "TILING_DATA": 3,
            "TILING_FIELD": 4,
            "FIELD": 5,
            "VARIABLE": 6,
        }
        start_ents = sorted(start_ents, key=lambda e: rank.get(e.kind_name(), 9))
        end_kinds = [end_kind] if end_kind else None
        end_id = None
        if end:
            end_ents = self.codemap.by_name(end)
            if end_ents:
                end_id = end_ents[0].id
            elif end.upper() in {k.value for k in EntityKind}:
                end_kinds = [end.upper()]
        kinds = end_kinds or (["KERNEL"] if not end else None)
        for ent in start_ents:
            path_ids = self.codemap.find_path(
                ent.id, end_id=end_id, end_kinds=kinds, include_advisory=False
            )
            if path_ids:
                return [
                    self.codemap.entities[i].to_dict()
                    for i in path_ids
                    if i in self.codemap.entities
                ]
        return []

    # ---- Operator contract -------------------------------------------------

    def operator_api(self) -> dict[str, Any]:
        """Return current-source public operator inputs, attributes and outputs."""
        inputs = self.codemap.by_kind(EntityKind.INPUT)
        tensor_inputs = sorted(
            (e for e in inputs if e.attrs.get("api_kind") == "tensor"),
            key=lambda e: int(e.attrs.get("api_index") or 0),
        )
        attributes = sorted(
            (e for e in inputs if e.attrs.get("api_kind") == "attribute"),
            key=lambda e: int(e.attrs.get("api_attr_index") or 0),
        )
        outputs = sorted(
            self.codemap.by_kind(EntityKind.OUTPUT),
            key=lambda e: int(e.attrs.get("api_index") or 0),
        )
        return {
            "tensor_inputs": [e.to_dict() for e in tensor_inputs],
            "attributes": [e.to_dict() for e in attributes],
            "outputs": [e.to_dict() for e in outputs],
        }

    def input_roots(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.codemap.by_kind(EntityKind.INPUT)]

    def output_roots(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.codemap.by_kind(EntityKind.OUTPUT)]

    # ---- Tiling ------------------------------------------------------------

    def tiling_keys(self) -> list[dict[str, Any]]:
        """Return source-declared TilingKey dimensions in packed bit order."""
        rows = self.codemap.by_kind(EntityKind.TILING_KEY)
        rows = sorted(rows, key=lambda e: int(e.attrs.get("decl_order") or 0))
        return [e.to_dict() for e in rows]

    def tiling_data(self, name: str = "") -> list[dict[str, Any]]:
        if name:
            return self.find_symbol(name, kind=EntityKind.TILING_DATA.value)
        return [e.to_dict() for e in self.codemap.by_kind(EntityKind.TILING_DATA)]

    def tiling_fields(self, owner: str = "") -> list[dict[str, Any]]:
        rows = self.codemap.by_kind(EntityKind.TILING_FIELD)
        if owner:
            rows = [e for e in rows if str(e.attrs.get("owner") or "") == owner]
        return [e.to_dict() for e in rows]

    def tiling_registrations(self) -> list[dict[str, Any]]:
        """Return REGISTER_TILING_FOR_TILINGKEY predicate→TilingData bindings."""
        out: list[dict[str, Any]] = []
        for rel in self.codemap.relations.values():
            if rel.kind_name() != RelationKind.SELECTS.value:
                continue
            src = self.codemap.entities.get(rel.src)
            dst = self.codemap.entities.get(rel.dst)
            if not src or not dst:
                continue
            if (
                src.kind_name() == EntityKind.PREDICATE.value
                and src.attrs.get("predicate_role") == "packed_tiling_key_registration"
                and dst.kind_name() == EntityKind.TILING_DATA.value
            ):
                out.append(
                    {
                        "predicate": src.to_dict(),
                        "tiling_data": dst.to_dict(),
                        "relation": rel.to_dict(),
                    }
                )
        return out

    def tpl_schema(self) -> dict[str, Any]:
        """TPL ARGS_DECL/ARGS_SEL schema from view_blob or meta."""
        blob = self._view("tiling/tpl_schema.yaml")
        if isinstance(blob, dict) and blob:
            return blob
        meta = self.codemap.meta.get("tpl_schema")
        return dict(meta) if isinstance(meta, dict) else {}

    def template_blocks(self) -> list[dict[str, Any]]:
        blob = self._view("tiling/template_blocks.yaml")
        if isinstance(blob, dict):
            return list(blob.get("blocks") or [])
        return [
            e.to_dict()
            for e in self.codemap.by_kind(EntityKind.TEMPLATE)
            if e.attrs.get("tpl_role") == "args_sel_group"
        ]

    def legal_key_count(self) -> int:
        blob = self._view("tiling/exhaustive_key_space.yaml")
        if isinstance(blob, dict) and int(blob.get("legal_key_count") or 0) > 0:
            return int(blob["legal_key_count"])
        return int(self.codemap.meta.get("legal_key_count") or 0)

    def legal_keys(self, *, limit: int = 0, offset: int = 0) -> list[dict[str, Any]]:
        """Stream legal packed keys from the index blob (not entity-per-key)."""
        blob = self._view("tiling/legal_key_index.jsonl")
        from ascendc_codemap_mcp.engine.query.legal_key_cache import expand_legal_key_rows

        rows = expand_legal_key_rows(blob)
        if offset:
            rows = rows[offset:]
        if limit and limit > 0:
            rows = rows[:limit]
        return rows

    def _view(self, name: str) -> Any:
        if not self.path:
            return None
        try:
            from ascendc_codemap_mcp.engine.store.reader import load_view_blob_checked

            checked = load_view_blob_checked(self.path, name, codemap=self.codemap)
            return checked.get("view")
        except Exception:
            return None

    # ---- Template / architecture / kernels --------------------------------

    def guards(self, name: str) -> list[dict[str, Any]]:
        return self._adj(name, RelationKind.GUARDED_BY, direction="out") + self._adj(
            name, RelationKind.CONTROLS, direction="in"
        )

    def template_instance(self, name: str = "") -> list[dict[str, Any]]:
        if name:
            return self.find_symbol(name, kind="TEMPLATE_INSTANCE")
        return [e.to_dict() for e in self.codemap.by_kind(EntityKind.TEMPLATE_INSTANCE)]

    def template_bindings(self, instance_name: str) -> list[dict[str, Any]]:
        return self._adj(instance_name, RelationKind.BINDS, direction="out")

    def active_variant(self) -> dict[str, Any] | None:
        rows = [e.to_dict() for e in self.codemap.by_kind(EntityKind.BUILD_VARIANT)]
        return rows[0] if rows else None

    def available_arch(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.codemap.by_kind(EntityKind.ARCH)]

    def selected_kernel(self, key_name: str = "") -> list[dict[str, Any]]:
        if key_name:
            return self._adj(key_name, RelationKind.SELECTS, direction="out") + self._adj(
                key_name, RelationKind.CONTROLS, direction="out"
            )
        return [e.to_dict() for e in self.codemap.by_kind(EntityKind.KERNEL)]

    # ---- Kernel Root Trace -------------------------------------------------

    def kernel_overview(self) -> dict[str, Any]:
        """Compact Kernel Root Trace summary."""
        meta = dict(self.codemap.meta.get("kernel_root_trace") or {})
        if not meta:
            meta = dict(self.codemap.meta.get("kernel_execution") or {})
        return {
            "model": "root_trace",
            "operations": len(self.codemap.by_kind(EntityKind.OPERATION)),
            "buffers": len(self.codemap.by_kind(EntityKind.BUFFER)),
            "registers": len(self.codemap.by_kind(EntityKind.REGISTER)),
            "pipes": len(self.codemap.by_kind(EntityKind.PIPE)),
            "events": len(self.codemap.by_kind(EntityKind.EVENT)),
            "queues": len(self.codemap.by_kind(EntityKind.QUEUE)),
            "reached_buffers": meta.get("reached_buffers"),
            "reached_registers": meta.get("reached_registers"),
            "reached_operations": meta.get("reached_operations"),
            "gap_count": meta.get("gap_count"),
            "meta": meta,
        }

    def kernel_root_trace(self) -> dict[str, Any]:
        """Return Kernel Root Trace meta + ROOTED_AT / WRAPS counts."""
        meta = dict(self.codemap.meta.get("kernel_root_trace") or {})
        wraps = sum(1 for r in self.codemap.relations.values() if r.kind_name() == RelationKind.WRAPS.value)
        rooted = sum(1 for r in self.codemap.relations.values() if r.kind_name() == RelationKind.ROOTED_AT.value)
        return {**meta, "wraps_relations": wraps, "rooted_at_relations": rooted}

    def operations(self, function: str = "") -> list[dict[str, Any]]:
        rows = self.codemap.by_kind(EntityKind.OPERATION)
        if function:
            rows = [
                e
                for e in rows
                if str(e.attrs.get("function") or "") == function
                or e.attrs.get("function", "").endswith(function)
            ]
        rows = sorted(rows, key=lambda e: (e.file, e.line_start, int(e.attrs.get("column") or 0)))
        return [e.to_dict() for e in rows]

    def operation(self, site_id: str) -> dict[str, Any] | None:
        ent = self.codemap.entities.get(site_id)
        if ent is None or ent.kind_name() != EntityKind.OPERATION.value:
            hits = [e for e in self.codemap.by_kind(EntityKind.OPERATION) if e.id == site_id or e.name == site_id]
            ent = hits[0] if hits else None
        return ent.to_dict() if ent else None

    def buffers(self, function: str = "") -> list[dict[str, Any]]:
        rows = self.codemap.by_kind(EntityKind.BUFFER)
        if function:
            rows = [e for e in rows if str(e.attrs.get("scope") or "") == function]
        return [e.to_dict() for e in rows]

    def registers(self, function: str = "") -> list[dict[str, Any]]:
        """AscendC MicroAPI/Reg register-file entities (RegTensor, MaskReg, ...)."""
        rows = self.codemap.by_kind(EntityKind.REGISTER)
        if function:
            rows = [e for e in rows if str(e.attrs.get("scope") or "") == function]
        rows = sorted(rows, key=lambda e: (e.file, e.line_start, e.name))
        return [e.to_dict() for e in rows]

    def locate(self, entity_id: str) -> dict[str, Any] | None:
        """Return file/line/column for quick source jump from any entity id."""
        ent = self.codemap.entities.get(entity_id)
        if ent is None:
            return None
        return {
            "id": ent.id,
            "kind": ent.kind_name(),
            "name": ent.name,
            "file": ent.file,
            "line": int(ent.line_start or ent.attrs.get("line") or 0),
            "column": int(ent.attrs.get("column") or 0),
            "function": ent.attrs.get("function") or ent.attrs.get("scope") or "",
        }

    def buffer(self, name: str) -> list[dict[str, Any]]:
        return [
            e.to_dict()
            for e in self.codemap.by_kind(EntityKind.BUFFER)
            if e.name == name or name in e.name
        ]

    def unresolved(self) -> list[dict[str, Any]]:
        return [
            e.to_dict()
            for e in self.codemap.entities.values()
            if str(e.status).lower() in {"unresolved", "partial", "unknown"}
        ]

    def source(self, name: str) -> dict[str, Any] | None:
        ent = self.definition(name)
        if not ent:
            return None
        return {
            "name": ent.get("name"),
            "file": ent.get("file"),
            "line_start": ent.get("line_start"),
            "line_end": ent.get("line_end"),
        }

    def audit(self) -> dict[str, Any]:
        """Return the evidence-backed completeness report for this CodeMap."""
        from ascendc_codemap_mcp.engine.diagnostics.audit import audit_codemap

        return audit_codemap(self.codemap)

    def summary(self) -> dict[str, Any]:
        """Agent-facing summary uses strict semantic-path semantics."""
        return dict(self.audit()["summary"])

    def _adj(
        self,
        name: str,
        kind: RelationKind,
        *,
        direction: str,
    ) -> list[dict[str, Any]]:
        ents = self.codemap.by_name(name)
        out: list[dict[str, Any]] = []
        for ent in ents:
            for _rel, other in self.codemap.neighbors(
                ent.id, kind=kind, direction=direction, include_advisory=False
            ):
                out.append(other.to_dict())
        return out

    def _walk(self, name: str, *, direction: str, limit: int) -> list[dict[str, Any]]:
        ents = self.codemap.by_name(name)
        if not ents:
            return []
        seen: set[str] = {ents[0].id}
        frontier = [ents[0].id]
        ordered: list[Entity] = []
        while frontier and len(ordered) < limit:
            cur = frontier.pop(0)
            for _rel, other in self.codemap.neighbors(
                cur, direction=direction, include_advisory=False
            ):
                if other.id in seen:
                    continue
                seen.add(other.id)
                ordered.append(other)
                frontier.append(other.id)
                if len(ordered) >= limit:
                    break
        return [e.to_dict() for e in ordered]


def open_codemap_query(
    op_root_or_uo: str | Path,
    *,
    op_name: str = "",
    architecture: str = "",
) -> CodeMapQuery:
    path = Path(op_root_or_uo).expanduser().resolve()
    if path.is_file() and path.suffix == ".uo":
        return CodeMapQuery(codemap=read_codemap(path), path=str(path))
    found = find_uo_product(path, op_name=op_name, architecture=architecture)
    if found is None or found.suffix != ".uo":
        raise FileNotFoundError(
            f"no .uo product under {path}; expected "
            ".ascendc-codemap/<arch>/<op>.<arch>.uo"
        )
    return CodeMapQuery(codemap=read_codemap(found), path=str(found))
