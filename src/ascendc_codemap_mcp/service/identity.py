# -*- coding: utf-8 -*-
"""CodeMap identity: operator + architecture + snapshot, not a CLI path pair."""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import PRODUCT_DIR_NAME
from ascendc_codemap_mcp.engine.source_layout import is_product_architecture
from ascendc_codemap_mcp.service.envelope import fail

FORMAT = "codemap-uo"


@dataclass
class CodemapRef:
    id: str
    project: Path
    architecture: str
    op_name: str
    product: Path | None


class Registry:
    """Process-local map of ``op@arch`` → operator directory.

    Query tools take ``codemap_id``. Index/discover still take a filesystem
    path so the first snapshot can be created. After that the id is enough
    for the life of this MCP process (and for env-backed defaults).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, CodemapRef] = {}

    def put(self, ref: CodemapRef) -> CodemapRef:
        with self._lock:
            self._by_id[ref.id] = ref
            alias = make_id(ref.project.name, ref.architecture)
            if alias != ref.id:
                self._by_id[alias] = ref
        return ref

    def get(self, codemap_id: str) -> CodemapRef | None:
        with self._lock:
            return self._by_id.get(codemap_id)

    def all(self) -> list[CodemapRef]:
        with self._lock:
            seen: set[str] = set()
            out: list[CodemapRef] = []
            for ref in self._by_id.values():
                key = str(ref.product or ref.id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(ref)
            return out

    def clear(self) -> None:
        with self._lock:
            self._by_id.clear()


def make_id(op_name: str, architecture: str) -> str:
    name = str(op_name or "").strip()
    arch = str(architecture or "").strip()
    if not name or not arch:
        return ""
    return f"{name}@{arch}"


def parse_id(text: str) -> tuple[str, str] | None:
    raw = str(text or "").strip()
    if not raw or "@" not in raw:
        return None
    name, arch = raw.rsplit("@", 1)
    name = name.strip()
    arch = arch.strip()
    if not name or not is_product_architecture(arch):
        return None
    return name, arch


def snapshot_id(product: Path, meta: dict[str, Any] | None = None) -> str:
    """Cheap snapshot fingerprint. Not a full-file digest of an 80MB ``.uo``."""
    path = Path(product)
    try:
        st = path.stat()
        size = int(st.st_size)
        mtime_ns = int(st.st_mtime_ns)
    except OSError:
        size = 0
        mtime_ns = 0
    rev = str((meta or {}).get("source_revision") or "")
    blob = f"{path.resolve()}|{size}|{mtime_ns}|{rev}|{FORMAT}"
    return "cm:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def env_project() -> str:
    return str(os.environ.get("ASCENDC_CODEMAP_PROJECT") or "").strip()


def env_architecture() -> str:
    return str(os.environ.get("ASCENDC_CODEMAP_ARCHITECTURE") or "").strip()


def _op_name_from_product(product: Path, fallback: str) -> str:
    from ascendc_codemap_mcp.engine.store.reader import read_meta
    from ascendc_codemap_mcp.engine.yaml_io import read_yaml

    try:
        name = str(read_meta(product).get("op_name") or "").strip()
    except Exception:  # noqa: BLE001
        name = ""
    if name:
        return name
    try:
        yaml_name = str(
            (read_yaml(product.parent / "operator.yaml") or {}).get("op_name") or ""
        ).strip()
    except Exception:  # noqa: BLE001
        yaml_name = ""
    return yaml_name or fallback


def ref_from_product(product: Path, *, project: Path | None = None) -> CodemapRef | None:
    path = Path(product).expanduser().resolve()
    if not path.is_file() or path.suffix != ".uo":
        return None
    name = path.name
    if not name.endswith(".uo"):
        return None
    stem = name[: -len(".uo")]
    arch = ""
    op_name = stem
    if "." in stem:
        op_name, arch = stem.rsplit(".", 1)
    if not is_product_architecture(arch):
        return None
    root = project
    if root is None:
        # <op>/.ascendc-codemap/<arch>/<op>.<arch>.uo
        try:
            if path.parent.parent.name == PRODUCT_DIR_NAME:
                root = path.parent.parent.parent
        except Exception:  # noqa: BLE001
            root = None
    if root is None:
        root = path.parent
    op_name = _op_name_from_product(path, op_name or root.name)
    return CodemapRef(
        id=make_id(op_name, arch),
        project=Path(root).resolve(),
        architecture=arch,
        op_name=op_name,
        product=path,
    )


def list_products(project: Path) -> list[Path]:
    from ascendc_codemap_mcp.engine.source_layout import is_product_architecture as is_arch

    root = Path(project).expanduser().resolve()
    pilot = root / PRODUCT_DIR_NAME
    if not pilot.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(pilot.iterdir()):
        if not (child.is_dir() and is_arch(child.name)):
            continue
        for uo in sorted(child.glob("*.uo")):
            if uo.is_file():
                found.append(uo)
    return found


def bind(
    *,
    project: str | Path,
    architecture: str,
    registry: Registry,
) -> CodemapRef:
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product

    root = Path(project).expanduser().resolve()
    arch = str(architecture or "").strip()
    product = find_uo_product(root, architecture=arch) if root.is_dir() or root.is_file() else None
    if product is not None and product.is_file() and product.suffix == ".uo":
        ref = ref_from_product(product, project=root if root.is_dir() else None)
        if ref is not None:
            if not ref.architecture:
                ref.architecture = arch
            return registry.put(ref)
    op_name = root.name if root.is_dir() else Path(str(project)).name
    ref = CodemapRef(
        id=make_id(op_name, arch),
        project=root,
        architecture=arch,
        op_name=op_name,
        product=product if product is not None and product.is_file() else None,
    )
    return registry.put(ref)


def resolve(
    *,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    registry: Registry,
    require_indexed: bool = False,
) -> CodemapRef | dict[str, Any]:
    """Resolve a CodeMap. On failure return an envelope dict (``ok`` key)."""
    cid = str(codemap_id or "").strip()
    proj = str(project or "").strip() or env_project()
    arch = str(architecture or "").strip() or env_architecture()

    if cid:
        as_path = Path(cid).expanduser()
        if as_path.is_file() and as_path.suffix == ".uo":
            ref = ref_from_product(as_path)
            if ref is None:
                return fail(
                    f"not a CodeMap product: {as_path}",
                    error_code="CODEMAP_NOT_INDEXED",
                )
            registry.put(ref)
            return ref
        parsed = parse_id(cid)
        if parsed is None:
            return fail(
                "codemap_id must look like op_name@arch35",
                error_code="INVALID_CODEMAP_ID",
            )
        op_name, parsed_arch = parsed
        hit = registry.get(cid)
        if hit is not None:
            if require_indexed and (hit.product is None or not hit.product.is_file()):
                return fail(
                    f"no .uo for {cid}",
                    error_code="CODEMAP_NOT_INDEXED",
                    extra={"codemap": {"id": cid, "architecture": parsed_arch}},
                )
            return hit
        if proj:
            ref = bind(project=proj, architecture=parsed_arch, registry=registry)
            if ref.id == cid or ref.op_name == op_name or Path(proj).name == op_name:
                if require_indexed and (ref.product is None or not Path(ref.product).is_file()):
                    return fail(
                        f"no .uo for {cid}; call codemap_index first",
                        error_code="CODEMAP_NOT_INDEXED",
                        extra={"codemap": {"id": cid, "architecture": parsed_arch}},
                    )
                return ref
        return fail(
            f"unknown codemap_id {cid}; call codemap_discover with project= first",
            error_code="CODEMAP_NOT_REGISTERED",
        )

    if not proj:
        return fail("project is required", error_code="PROJECT_REQUIRED")
    if not arch:
        return fail(
            "ARCHITECTURE_MISSING_IN_RUN_STATE: architecture is required",
            error_code="ARCHITECTURE_MISSING_IN_RUN_STATE",
        )
    root = Path(proj).expanduser().resolve()
    if not root.is_dir() and not (root.is_file() and root.suffix == ".uo"):
        return fail(f"operator directory not found: {root}", error_code="OPERATOR_DIR_NOT_FOUND")
    ref = bind(project=root, architecture=arch, registry=registry)
    if require_indexed and (ref.product is None or not Path(ref.product).is_file()):
        return fail(
            f"no .uo product under {root}; expected "
            f"{PRODUCT_DIR_NAME}/{arch}/<op>.{arch}.uo. "
            "Run codemap_index first.",
            error_code="CODEMAP_NOT_INDEXED",
            extra={"codemap": {"id": ref.id, "architecture": arch}},
        )
    return ref


def is_ref(value: Any) -> bool:
    return isinstance(value, CodemapRef)


def public_handle(
    ref: CodemapRef,
    *,
    meta: dict[str, Any] | None = None,
    freshness_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(meta or {})
    info = dict(freshness_info or {})
    product = ref.product
    sid = snapshot_id(product, meta) if product is not None and Path(product).is_file() else ""
    completeness = info.get("semantic_completeness")
    if completeness is None:
        raw = meta.get("semantic_completeness")
        try:
            completeness = float(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            completeness = None
    return {
        "id": ref.id,
        "snapshot_id": sid,
        "architecture": ref.architecture,
        "op_name": ref.op_name,
        "project": str(ref.project),
        "path": str(product) if product else "",
        "source_revision": str(info.get("source_revision") or meta.get("source_revision") or ""),
        "indexed_revision": str(info.get("indexed_revision") or meta.get("source_revision") or ""),
        "freshness": str(info.get("freshness") or ""),
        "semantic_completeness": completeness,
        "format": FORMAT,
    }
