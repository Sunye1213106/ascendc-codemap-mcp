# -*- coding: utf-8 -*-
"""Export TG adapter pack YAML from UO host_derivation / KB.

Writes under ``.ascendc-codemap/<arch>/adapter/`` by default.
Missing runtime capabilities use Local Extension under
``.ascendc-codemap/<arch>/local/``.
TG loaders prefer the adapter dir, then local package / fixtures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.paths import require_architecture
import yaml

ADAPTER_FILES = (
    "bridge_spec.yaml",
    "feature_bindings.yaml",
    "search_hints.yaml",
    "construction_hints.yaml",
)


class AdapterPackError(ValueError):
    """Invalid adapter pack content (e.g. grid key outside knob_schema)."""


def adapter_dir(project_root: Path, *, arch: str | None = None) -> Path:
    """``.ascendc-codemap/<arch>/adapter`` under the operator source root."""
    from ascendc_codemap_mcp.engine.paths import product_dir

    root = Path(project_root).expanduser().resolve()
    arch_name = require_architecture(arch)
    return product_dir(root, arch_name) / "adapter"


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return doc if isinstance(doc, dict) else {}


def _knob_schema() -> dict[str, Any]:
    try:
        from testcase_agent.closure import workspace as W

        sem = W.replay_inputs().SEMANTICS
        if hasattr(sem, "knob_schema"):
            return dict(sem.knob_schema() or {})
    except Exception:
        pass
    return {}


def _enums() -> dict[str, Any]:
    try:
        from testcase_agent.closure import workspace as W

        sem = W.replay_inputs().SEMANTICS
        if hasattr(sem, "enums"):
            return dict(sem.enums() or {})
    except Exception:
        pass
    return {}


def gate_sampling_grid(grid: dict[str, Any], schema: dict[str, Any] | None = None) -> list[str]:
    """Return unknown grid keys. Empty list means OK."""
    sch = schema if schema is not None else _knob_schema()
    if not sch:
        # No schema available (unit fixtures) — do not hard-fail.
        return []
    return sorted(str(k) for k in grid if str(k) not in sch)


def build_bridge_spec(
    derivation: dict[str, Any],
    *,
    op_name: str = "",
    arch: str = "",
) -> dict[str, Any]:
    """Project host_derivation roots into a bridge_spec skeleton."""
    bindings: list[dict[str, Any]] = []
    unbound: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fld in derivation.get("fields") or []:
        if not isinstance(fld, dict):
            continue
        roots = list(fld.get("root_vars") or [])
        var_roots = dict(fld.get("var_roots") or {})
        for var_id, root in var_roots.items():
            var = str(var_id)
            if var in seen:
                continue
            seen.add(var)
            root_s = str(root or "")
            if root_s.startswith("INPUT_") or root_s.startswith("ATTR_") or "input" in root_s.lower():
                bindings.append({
                    "var": var,
                    "root": root_s,
                    "kind": "attr",
                    "operand": root_s,
                })
            elif roots and root_s:
                bindings.append({
                    "var": var,
                    "root": root_s,
                    "kind": "context",
                    "value": None,
                })
            else:
                unbound.append({
                    "var": var,
                    "reason": "tiling_state_or_unresolved",
                    "root": root_s,
                })
        for root in roots:
            r = str(root)
            if r in seen:
                continue
            seen.add(r)
            if r.startswith("INPUT_") or r.startswith("ATTR_"):
                bindings.append({
                    "var": r,
                    "root": r,
                    "kind": "attr",
                    "operand": r,
                })
            else:
                unbound.append({
                    "var": r,
                    "reason": "no_case_binding",
                    "root": r,
                })
    return {
        "version": 1,
        "schema": "bridge-spec/v1",
        "operator": op_name,
        "arch": arch,
        "source": "export_adapter_pack",
        "bindings": bindings,
        "unbound": unbound,
        "bridges": [],
    }


def build_feature_bindings(derivation: dict[str, Any], uo: Path | None = None) -> dict[str, Any]:
    enums = _enums()
    categorical = sorted(enums.keys()) if enums else ["layout", "dtype"]
    base_numeric: list[str] = []
    schema = _knob_schema()
    for name, meta in schema.items():
        if meta.get("kind") == "numeric" and name not in categorical:
            base_numeric.append(name)
    if not base_numeric:
        base_numeric = ["b", "s1", "s2", "n2", "g", "d"]

    derived_terms: dict[str, str] = {}
    floor_terms: list[str] = []
    static_parents: dict[str, list[str]] = {}
    dim_to_roots: dict[str, list[str]] = {}
    root_to_knobs: dict[str, list[str]] = {}
    schema_keys = list(schema.keys())

    def _knobs_for_root(root: str) -> list[str]:
        """Map one source root onto case knobs (conservative over-approx OK)."""
        r = str(root or "")
        if not r:
            return []
        named = [k for k in schema_keys if k.lower() in r.lower()]
        if named:
            return named
        if r.startswith("INPUT_"):
            return list(base_numeric)
        if r.startswith("ATTR_"):
            matched = [
                k for k in categorical
                if k.lower() in r.lower() or r.lower().endswith(k.lower())
            ]
            return matched or list(categorical)
        return []

    for fld in derivation.get("fields") or []:
        if not isinstance(fld, dict):
            continue
        dim = str(fld.get("name") or fld.get("dim") or "")
        roots = [str(r) for r in (fld.get("root_vars") or [])]
        if dim and roots:
            dim_to_roots[dim] = roots
            parents = [k for k in schema_keys if any(k.lower() in r.lower() for r in roots)]
            if parents:
                static_parents[dim] = parents
            for root in roots:
                knobs = _knobs_for_root(root)
                if not knobs:
                    continue
                bucket = root_to_knobs.setdefault(root, [])
                for k in knobs:
                    if k not in bucket:
                        bucket.append(k)

    if uo is not None:
        view = _load(Path(uo) / "ir" / "tg_host_view.yaml")
        for pred in view.get("predicates") or []:
            hint = str(pred.get("feature_hint") or "").strip()
            if hint:
                floor_terms.append(hint)
                derived_terms.setdefault(hint, "")

    return {
        "version": 1,
        "schema": "feature-bindings/v1",
        "source": "export_adapter_pack",
        "floor_terms": sorted(set(floor_terms)),
        "categorical": categorical,
        "base_numeric": base_numeric,
        "derived_terms": derived_terms,
        "static_parents": static_parents,
        "dim_to_roots": dim_to_roots,
        "root_to_knobs": root_to_knobs,
    }


def build_search_hints(*, sampling_grid: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    schema = _knob_schema()
    enums = _enums()
    grid: dict[str, list[Any]] = dict(sampling_grid or {})
    if not grid:
        for name, meta in schema.items():
            if not meta.get("mutable", True):
                continue
            if meta.get("kind") == "categorical":
                domain = list(meta.get("domain") or enums.get(name) or [])
                if domain:
                    grid[name] = domain
            elif meta.get("kind") == "bool":
                grid[name] = [False, True]
            elif meta.get("kind") == "numeric" and meta.get("domain"):
                grid[name] = list(meta["domain"])
    unknown = gate_sampling_grid(grid, schema)
    if unknown:
        raise AdapterPackError(
            f"sampling_grid keys not in knob_schema(): {unknown}"
        )
    return {
        "version": 1,
        "schema": "search-hints/v1",
        "source": "export_adapter_pack",
        "sampling_grid": grid,
        "nearest_knobs": {},
        "named_bindings": {},
    }


def build_construction_hints(derivation: dict[str, Any] | None = None) -> dict[str, Any]:
    del derivation
    schema = _knob_schema()
    defaults: dict[str, Any] = {}
    for name, meta in schema.items():
        if "default" in meta:
            defaults[name] = meta["default"]
    return {
        "version": 1,
        "schema": "construction-hints/v1",
        "source": "export_adapter_pack",
        "dtype_dim": "InputDType",
        "dtype": {},
        "defaults": defaults,
        "require": {},
        "loops": [],
        "bool_knobs": {},
        "post": [],
        "masks": [],
    }


def export_adapter_pack(
    project_root: Path,
    *,
    arch: str | None = None,
    uo: Path | None = None,
    write_package: bool = False,
    sampling_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Build and write the four adapter YAML files. Returns a receipt dict."""
    root = Path(project_root).expanduser().resolve()
    arch_name = require_architecture(arch)
    if uo is None:
        from ascendc_codemap_mcp.engine.paths import product_dir

        uo_path = product_dir(root, arch_name)
    else:
        uo_path = Path(uo)

    derivation = _load(uo_path / "ir" / "host_derivation.yaml")
    if not derivation:
        # Fall back to key_derivations projection.
        derivation = _load(uo_path / "tiling" / "key_derivations.yaml")
    manifest = _load(uo_path / "manifest.yaml")
    op_name = str(manifest.get("op_name") or "")
    arch_name = require_architecture(arch or manifest.get("architecture"))

    out_dir = adapter_dir(root, arch=arch_name)
    bridge = build_bridge_spec(derivation, op_name=op_name, arch=arch_name)
    features = build_feature_bindings(derivation, uo=uo_path)
    try:
        search = build_search_hints(sampling_grid=sampling_grid)
    except AdapterPackError as exc:
        return {
            "ok": False,
            "engine": "export_adapter_pack",
            "error": str(exc),
            "gate": "sampling_grid_knob_schema",
        }
    construct = build_construction_hints(derivation)

    written: list[str] = []
    for name, doc in (
        ("bridge_spec.yaml", bridge),
        ("feature_bindings.yaml", features),
        ("search_hints.yaml", search),
        ("construction_hints.yaml", construct),
    ):
        path = out_dir / name
        _dump(path, doc)
        written.append(str(path))

    package_written: list[str] = []
    if write_package:
        try:
            from replay.package_data import active_package_dir

            pkg = active_package_dir()
            for name, doc in (
                ("bridge_spec.yaml", bridge),
                ("feature_bindings.yaml", features),
                ("search_hints.yaml", search),
                ("construction_hints.yaml", construct),
            ):
                p = pkg / name
                _dump(p, doc)
                package_written.append(str(p))
        except Exception as exc:  # noqa: BLE001
            package_written = [f"skipped:{exc}"]

    receipt = {
        "ok": True,
        "engine": "export_adapter_pack",
        "adapter_dir": str(out_dir),
        "written": written,
        "package_written": package_written,
        "fields": len(derivation.get("fields") or []),
        "sampling_grid_keys": sorted((search.get("sampling_grid") or {}).keys()),
        "bridge_bindings": len(bridge.get("bindings") or []),
        "bridge_unbound": len(bridge.get("unbound") or []),
    }
    _dump(out_dir / "export_receipt.yaml", receipt)
    try:
        checks = uo_path / "checks"
        checks.mkdir(parents=True, exist_ok=True)
        _dump(checks / "adapter_pack_receipt.yaml", receipt)
    except Exception:
        pass
    return receipt
