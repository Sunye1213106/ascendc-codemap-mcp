# -*- coding: utf-8 -*-
"""Discover an operator's layout from its source tree.

Every module used to hardcode FlashAttentionScoreGrad paths and regexes. This
resolves them from the Ascend C repository conventions instead:

    op_host/<snake>_def.cpp                       operator definition
    op_host/**/*_tiling*.cpp                      host tiling TUs
    op_kernel/<snake>_apt.cpp | <snake>.cpp       kernel entry
    op_kernel/<arch>/<snake>_template_tiling_key.h TilingKey DSL

Which files those roles draw from is decided by `scope_scan`, which bootstraps
from the directory layout and include graph, then (during prepare/extract) is
enriched with Clang's real dependency closure so shared headers a domain keeps
beside its operators come along. The globs here remain as a fallback for a
tree the scan cannot make sense of.

A repository that does not follow the convention can pin the answer with
`spec/operators/<op>.yaml`; discovery reports ambiguities rather than guessing.
`scope_validate` turns hard failures into blockers — it never asks a human to
confirm a file list once operator + arch are fixed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ascendc_codemap_mcp.engine import scope_scan as sscan
from ascendc_codemap_mcp.engine.source_layout import (
    ARCH_DIR_RE,
    UNIFIED_ARCH_DIR,
    arch_number,
    is_other_arch_path,
    iter_arch_source_dirs,
    match_on_disk_architecture,
)

SPEC_DIR = Path(__file__).resolve().parents[3] / "spec"
OVERRIDE_DIR = SPEC_DIR / "operators"

# `class FlashAttentionScoreGrad : public OpDef` / `OP_ADD(FlashAttentionScoreGrad)`
OP_CLASS_RE = re.compile(r"\bclass\s+([A-Z]\w*)\s*:\s*public\s+OpDef\b")
OP_ADD_RE = re.compile(r"\bOP_ADD\s*\(\s*([A-Z]\w*)\s*\)")
# `ASCENDC_TPL_ARGS_DECL(FlashAttentionScoreGrad,`
TPL_TAG_RE = re.compile(r"ASCENDC_TPL_ARGS_DECL\s*\(\s*([A-Za-z_]\w*)")


def camel_to_snake(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def snake_to_camel(name: str) -> str:
    return "".join(part.title() for part in str(name).split("_") if part)


@dataclass
class OpSpec:
    """Resolved locations for one operator, one architecture."""

    op_dir: Path
    op_name: str = ""
    op_snake: str = ""
    arch_dir: str = ""
    opdef: Path | None = None
    host_targets: list[Path] = field(default_factory=list)
    api_targets: list[Path] = field(default_factory=list)
    kernel_targets: list[Path] = field(default_factory=list)
    decl_targets: list[Path] = field(default_factory=list)
    kernel_entry: Path | None = None
    kernel_headers: list[Path] = field(default_factory=list)
    tiling_key_header: Path | None = None
    tiling_data_header: Path | None = None
    proto: Path | None = None
    docs: list[Path] = field(default_factory=list)
    available_archs: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    scope: sscan.ScopeSet | None = None
    source: str = "discovered"
    display_name_hint: str = ""

    @property
    def op_needle(self) -> str:
        """Legacy path substring for callers without a ScopeSet.

        Prefer ScopeSet / Clang closure membership. The first two words of the
        snake name exclude most CANN headers but also miss sibling ``common/``
        paths — walkers must pass ``scope=`` so shared AST nodes are kept.
        """
        parts = self.op_snake.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else self.op_snake

    @property
    def host_root(self) -> Path:
        return self.op_dir / "op_host"

    @property
    def kernel_root(self) -> Path:
        return self.op_dir / "op_kernel"

    def to_dict(self) -> dict[str, Any]:
        def rel(p: Path | None) -> str:
            if p is None:
                return ""
            try:
                return p.relative_to(self.op_dir).as_posix()
            except ValueError:
                return p.as_posix()

        return {
            "op_name": self.op_name,
            "op_snake": self.op_snake,
            "op_dir": self.op_dir.as_posix(),
            "arch_dir": self.arch_dir,
            "available_archs": list(self.available_archs),
            "opdef": rel(self.opdef),
            "host_targets": [rel(p) for p in self.host_targets],
            "api_targets": [rel(p) for p in self.api_targets],
            "kernel_targets": [rel(p) for p in self.kernel_targets],
            "decl_targets": [rel(p) for p in self.decl_targets],
            "scope_files": len(self.scope.files) if self.scope else 0,
            "kernel_entry": rel(self.kernel_entry),
            "tiling_key_header": rel(self.tiling_key_header),
            "tiling_data_header": rel(self.tiling_data_header),
            "proto": rel(self.proto),
            "docs": [rel(p) for p in self.docs],
            "ambiguities": list(self.ambiguities),
            "source": self.source,
            "display_name_hint": self.display_name_hint,
        }

    @property
    def is_unambiguous(self) -> bool:
        return not self.ambiguities


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _find_opdef(host_root: Path) -> tuple[Path | None, list[str]]:
    """Recall OpDef TUs. ``*_def.cpp`` is a candidate glob, not identity."""
    if not host_root.is_dir():
        return None, ["opdef_not_found: no op_host/*_def.cpp"]
    hinted = sorted(host_root.glob("*_def.cpp"))
    confirmed: list[Path] = []
    for path in hinted:
        text = _read(path)
        if OP_CLASS_RE.search(text) or OP_ADD_RE.search(text):
            confirmed.append(path)
    if not confirmed:
        for path in sorted(host_root.glob("*.cpp")):
            if path in hinted:
                continue
            text = _read(path)
            if OP_CLASS_RE.search(text) or OP_ADD_RE.search(text):
                confirmed.append(path)
    if confirmed:
        if len(confirmed) > 1:
            names = ", ".join(p.name for p in confirmed)
            return confirmed[0], [f"multiple_opdef: {names}"]
        return confirmed[0], []
    if hinted:
        return hinted[0], [f"display_name_hint: {hinted[0].name}"]
    return None, ["opdef_not_found: no op_host/*_def.cpp"]


def _op_name_from(opdef: Path | None, op_dir: Path) -> tuple[str, list[str]]:
    if opdef is not None:
        text = _read(opdef)
        m = OP_CLASS_RE.search(text) or OP_ADD_RE.search(text)
        if m:
            return m.group(1), []
        stem = opdef.stem[: -len("_def")] if opdef.stem.endswith("_def") else opdef.stem
        return snake_to_camel(stem), [f"display_name_hint: {opdef.name}"]
    return snake_to_camel(op_dir.name), ["display_name_hint: directory"]


def _discover_archs(op_dir: Path) -> list[str]:
    seen: set[str] = set()
    for parent in (op_dir / "op_host", op_dir / "op_kernel"):
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if child.is_dir() and ARCH_DIR_RE.match(child.name):
                seen.add(child.name)
    return sorted(seen, key=lambda name: (arch_number(name), name))


def _host_targets(host_root: Path, arch_dir: str, op_snake: str) -> list[Path]:
    """Candidate tiling TUs: glob recall only, not a semantic role.

    Headers are excluded (they are pulled in by the TUs) and other arch folders
    are excluded so a single run models exactly one hardware generation.
    """
    if not host_root.is_dir():
        return []
    out: list[Path] = []
    for path in sorted(host_root.glob("*.cpp")):
        if "_tiling" in path.stem:
            out.append(path)
    arch_root = host_root / arch_dir
    if arch_root.is_dir():
        out.extend(sorted(p for p in arch_root.glob("*.cpp") if "_tiling" in p.stem))
    for folder in iter_arch_source_dirs(host_root, arch_dir):
        if folder == arch_root:
            continue
        out.extend(sorted(p for p in folder.glob("*.cpp") if "_tiling" in p.stem))
    tiling_root = host_root / "op_tiling"
    if tiling_root.is_dir():
        for path in sorted(tiling_root.rglob("*.cpp")):
            if is_other_arch_path(path, arch_dir):
                continue
            out.append(path)
    # Prefer files that belong to this operator when the folder is shared.
    owned = [p for p in out if op_snake and p.stem.startswith(op_snake)]
    return owned or out


_INCLUDE_QUOTED_RE = re.compile(r'(?:#\s*include|__has_include)\s*\(?\s*"([^"]+)"')
# Host packing sites — not GetTilingKey() *calls* (wrappers dispatch to siblings).
_HOST_PACKING_SITE_RE = re.compile(
    r"(?:"
    r"\bGET_TPL_TILING_KEY\s*\("
    r"|(?:[A-Z][A-Z0-9_]*)?GET_TILING_?KEY\s*\("
    r"|\btilingKey_\s*(?:\+=|=)"
    r"|\btiling_key_\s*(?:\+=|=)"
    r"|\bSetTilingKey\s*\(\s*(?!tilingKey_|tiling_key_)"
    r")"
)


def _sibling_operator_dirs(kernel_entry: Path, op_dir: Path) -> list[Path]:
    """Operators whose headers this kernel includes (same family, one hop).

    Thin wrappers such as ``scatter_pa_cache`` include ``../scatter_pa_kv_cache/``
    and keep tiling TUs on the sibling. Host discovery must follow that include.
    """
    op_dir = op_dir.resolve()
    family = op_dir.parent
    try:
        text = kernel_entry.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    for match in _INCLUDE_QUOTED_RE.finditer(text):
        rel = match.group(1)
        if ".." not in rel.replace("\\", "/"):
            continue
        resolved = (kernel_entry.parent / rel).resolve()
        for parent in [resolved, *resolved.parents]:
            if parent.parent != family or parent == op_dir:
                continue
            if not (parent / "op_host").is_dir():
                continue
            if parent not in seen:
                seen.add(parent)
                found.append(parent)
            break
    return found


def _host_files_have_packing(paths: list[Path]) -> bool:
    """True when these TUs already mint tiling keys (not register/dispatch stubs)."""
    return bool(_packing_host_tus(paths))


def _packing_host_tus(paths: list[Path]) -> list[Path]:
    """Host TUs whose text contains a packing sink. Candidate recall, not a role."""
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path is None or not path.is_file():
            continue
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        try:
            text = _read(path)
        except OSError:
            continue
        if _HOST_PACKING_SITE_RE.search(text):
            out.append(path)
    return out


def _unique_paths(*groups: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for group in groups:
        for path in group:
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


def _targets_from_scope(spec: OpSpec) -> None:
    """Split the scanned scope into the sets each parsing stage consumes.

    File roles are layout bootstrap. Tiling TUs are recalled by directory/stem
    hint or packing-site text; that recall is not a semantic ``host_tiling``
    identity for the whole file. Definition / infershape TUs stay out of
    ``host_targets`` so their writes are not folded into a TilingKey run.
    """
    scope = spec.scope
    if scope is None:
        return
    spec.api_targets = scope.paths(role=sscan.ROLE_API, tu_only=True)
    spec.kernel_targets = scope.paths(role=sscan.ROLE_KERNEL_ENTRY, tu_only=True)
    hinted = scope.paths(role=sscan.ROLE_HOST_TILING, tu_only=True)
    host_owned = [
        f.path
        for f in scope.files
        if f.is_tu
        and f.side == sscan.SIDE_HOST
        and not f.shared
        and "op_host" in {p.lower() for p in f.path.parts}
    ]
    spec.host_targets = _unique_paths(hinted, _packing_host_tus(host_owned))
    spec.decl_targets = scope.paths(
        role=(sscan.ROLE_HOST_DEF, sscan.ROLE_HOST_INFERSHAPE), tu_only=True
    )


def _cpp_candidates(folder: Path, op_snake: str) -> tuple[Path | None, list[str]]:
    if not folder.is_dir():
        return None, []
    apt = sorted(folder.glob("*_apt.cpp"))
    if len(apt) == 1:
        return apt[0], []
    if len(apt) > 1:
        return apt[0], [f"multiple_kernel_entry: {', '.join(p.name for p in apt)}"]
    plain = sorted(folder.glob("*.cpp"))
    if len(plain) == 1:
        return plain[0], []
    named = [p for p in plain if p.stem == op_snake]
    if named:
        return named[0], []
    if plain:
        return plain[0], [f"multiple_kernel_entry: {', '.join(p.name for p in plain)}"]
    return None, []


def _kernel_entry(
    kernel_root: Path, op_snake: str, arch_dir: str = ""
) -> tuple[Path | None, list[str]]:
    """`*_apt.cpp` is the AscendC entry when present, else `<snake>.cpp`.

    Prefer ``op_kernel/archNN/*.cpp`` when that folder exists. Fallback for a
    tree ``scope_scan`` could not read; the scan decides which architecture an
    entry builds by path (under ``archNN/``) or includes (root-level TUs).
    """
    if not kernel_root.is_dir():
        return None, ["kernel_entry_not_found: no op_kernel/"]
    arch = str(arch_dir or "").strip()
    if arch:
        hit, notes = _cpp_candidates(kernel_root / arch, op_snake)
        if hit is not None:
            return hit, notes
    hit, notes = _cpp_candidates(kernel_root, op_snake)
    if hit is not None:
        return hit, notes
    return None, ["kernel_entry_not_found: no op_kernel/*.cpp"]


_TPL_HEADER_GLOBS = (
    "*template_tiling_key.h",
    "*tilingkey.h",
    "*_tiling_key.h",
)


def _pick_tiling_key_header(
    hits: list[Path], op_name: str, *, kernel_entry: Path | None = None
) -> tuple[Path, list[str]]:
    notes: list[str] = []
    if len(hits) > 1:
        notes.append(f"multiple_tiling_key_header: {', '.join(p.name for p in hits)}")
    if len(hits) == 1:
        return hits[0], notes
    from ascendc_codemap_mcp.engine.source_layout import quoted_include_basenames

    if kernel_entry is not None:
        names = quoted_include_basenames(kernel_entry)
        included = [h for h in hits if h.name.lower() in names]
        if included:
            return included[0], notes
    for h in hits:
        m = TPL_TAG_RE.search(_read(h))
        if m and m.group(1) == op_name:
            return h, notes
    snake = camel_to_snake(op_name)
    for h in hits:
        if snake and snake in h.name.lower():
            return h, notes
    return hits[0], notes


def _tiling_key_header(
    kernel_root: Path,
    arch_dir: str,
    op_name: str,
    *,
    kernel_entry: Path | None = None,
    op_dir: Path | None = None,
) -> tuple[Path | None, list[str]]:
    if op_dir is not None:
        from ascendc_codemap_mcp.engine.source_layout import select_tpl_decl_header

        hit = select_tpl_decl_header(Path(op_dir), arch_dir)
        if hit is not None and hit.is_file():
            return hit, []
    search_roots = [
        r for r in (*iter_arch_source_dirs(kernel_root, arch_dir), kernel_root) if r.is_dir()
    ]
    hits: list[Path] = []
    seen: set[Path] = set()
    for glob_pat in _TPL_HEADER_GLOBS:
        for root in search_roots:
            for path in sorted(root.glob(glob_pat)):
                key = path.resolve()
                if key in seen:
                    continue
                seen.add(key)
                hits.append(path)
        if hits:
            return _pick_tiling_key_header(
                hits, op_name, kernel_entry=kernel_entry
            )
    return None, ["tiling_key_header_not_found: no *template_tiling_key.h"]


def _tiling_data_header(kernel_root: Path, arch_dir: str) -> Path | None:
    roots = [*iter_arch_source_dirs(kernel_root, arch_dir), kernel_root]
    for root in roots:
        if not root.is_dir():
            continue
        hits = sorted(root.glob("*tiling_data*.h")) or sorted(root.glob("*_tiling.h"))
        if hits:
            return hits[0]
    return None


def load_override(op_name: str) -> dict[str, Any] | None:
    path = OVERRIDE_DIR / f"{op_name}.yaml"
    if not path.is_file():
        path = OVERRIDE_DIR / f"{camel_to_snake(op_name)}.yaml"
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def discover(op_dir: str | Path, *, arch_dir: str | None = None) -> OpSpec:
    """Resolve every path uo-init needs for one operator.

    `arch_dir` defaults to the newest ``arch*`` folder present. Those folders
    distinguish implementations (official ops-transformer: arch22 vs arch35).
    A tree with none is a single implementation: scan host/kernel together and
    use the ``default`` product slot. Never invent arch35.
    """
    op_dir = Path(op_dir).expanduser().resolve()
    spec = OpSpec(op_dir=op_dir)
    spec.available_archs = _discover_archs(op_dir)

    if arch_dir:
        spec.arch_dir = match_on_disk_architecture(arch_dir, spec.available_archs) or str(arch_dir)
        if spec.available_archs:
            if spec.arch_dir not in spec.available_archs:
                spec.ambiguities.append(
                    f"arch_not_present: {arch_dir} not in {spec.available_archs}"
                )
        elif spec.arch_dir != UNIFIED_ARCH_DIR:
            spec.ambiguities.append(
                f"arch_not_present: {arch_dir} not in []"
            )
    elif len(spec.available_archs) == 1:
        spec.arch_dir = spec.available_archs[0]
    elif spec.available_archs:
        spec.arch_dir = spec.available_archs[-1]
        spec.ambiguities.append(
            f"multiple_arch_dirs: {spec.available_archs}; defaulted to {spec.arch_dir}"
        )
    else:
        spec.arch_dir = UNIFIED_ARCH_DIR
        spec.ambiguities.append(
            "unified_implementation: no arch* folders; one implementation"
        )

    spec.opdef, notes = _find_opdef(spec.host_root)
    spec.ambiguities.extend(notes)

    spec.op_name, notes = _op_name_from(spec.opdef, op_dir)
    spec.ambiguities.extend(notes)
    if any(str(n).startswith("display_name_hint:") for n in notes):
        spec.display_name_hint = spec.op_name
    spec.op_snake = camel_to_snake(spec.op_name)

    # Scanned even when the spec is pinned: a pin says which files to parse,
    # while the scope says which files the walk may read once parsing pulls
    # them in, and the second question stands either way.
    spec.scope = sscan.scan(op_dir, arch_dir=spec.arch_dir)

    override = load_override(spec.op_name)
    if override:
        return _apply_override(spec, override)

    _targets_from_scope(spec)

    if not spec.host_targets:
        spec.host_targets = _host_targets(spec.host_root, spec.arch_dir, spec.op_snake)
        spec.ambiguities.append("host_targets_from_glob: scope scan found none")

    entry = spec.kernel_entry
    if entry is None and spec.kernel_targets:
        from ascendc_codemap_mcp.engine.source_layout import pick_kernel_entry

        entry = pick_kernel_entry(spec.kernel_targets, spec.arch_dir) or spec.kernel_targets[0]
    if entry is None:
        entry, _notes = _kernel_entry(spec.kernel_root, spec.op_snake, spec.arch_dir)
        if spec.kernel_entry is None:
            spec.kernel_entry = entry

    # Fusion kernels often include sibling kernel headers (IFA → PFA/incre).
    # Union sibling host TUs only when *this* op's host files are register or
    # dispatch stubs; otherwise IFA would parse PFA's tiling and balloon analyze.
    extra: list[Path] = []
    sib_names: list[str] = []
    if not _host_files_have_packing(spec.host_targets):
        sources: list[Path] = []
        if entry is not None:
            sources.append(entry)
        sources.extend(spec.host_targets)
        seen_sib: set[Path] = set()
        for src in sources:
            if src is None or not src.is_file():
                continue
            for sib in _sibling_operator_dirs(src, spec.op_dir):
                if sib in seen_sib:
                    continue
                seen_sib.add(sib)
                sib_names.append(sib.name)
                extra.extend(
                    _host_targets(
                        sib / "op_host",
                        spec.arch_dir,
                        camel_to_snake(sib.name),
                    )
                )
    if extra:
        existing = {p.resolve() for p in spec.host_targets}
        added = [p for p in extra if p.resolve() not in existing]
        if added:
            spec.host_targets = list(spec.host_targets) + added
            spec.ambiguities.append(
                "host_targets_from_sibling_kernel_include: " + ", ".join(sib_names)
            )
    if not spec.host_targets:
        spec.ambiguities.append("host_targets_not_found: no op_host tiling TU")

    if spec.kernel_targets:
        from ascendc_codemap_mcp.engine.source_layout import pick_kernel_entry

        spec.kernel_entry = pick_kernel_entry(spec.kernel_targets, spec.arch_dir)
        if spec.kernel_entry is None:
            spec.kernel_entry = spec.kernel_targets[0]
        if len(spec.kernel_targets) > 1:
            names = ", ".join(p.name for p in spec.kernel_targets)
            spec.ambiguities.append(f"multiple_kernel_entry: {names}")
    else:
        spec.kernel_entry, notes = _kernel_entry(
            spec.kernel_root, spec.op_snake, spec.arch_dir
        )
        spec.ambiguities.extend(notes)
        # Layout glob fallback can pick an arch-neutral *.cpp that builds another
        # arch. Prefer an arch-folder TU; keep the last remaining root TU.
        if spec.kernel_entry is not None and spec.arch_dir:
            from ascendc_codemap_mcp.engine.source_layout import (
                architecture_in_scope,
                architectures_match,
                path_owned_architecture,
            )

            arch = spec.arch_dir.strip()
            owned = path_owned_architecture(spec.kernel_entry)
            if owned and not architectures_match(owned, arch):
                spec.ambiguities.append(
                    f"kernel_entry_other_arch: {spec.kernel_entry.name} builds {owned}"
                )
                spec.kernel_entry = None
            elif not owned:
                includes = sscan.entry_architecture(spec.kernel_entry)
                if includes and not architecture_in_scope(includes, arch):
                    alt, _alt_notes = _cpp_candidates(
                        spec.kernel_root / spec.arch_dir, spec.op_snake
                    )
                    if alt is not None:
                        spec.kernel_entry = alt
                    else:
                        spec.ambiguities.append(
                            f"kernel_entry_kept_last_tu: {spec.kernel_entry.name} "
                            f"builds {includes} but is the only kernel TU"
                        )
                elif includes and not architectures_match(includes, arch):
                    spec.ambiguities.append(
                        f"kernel_entry_kept_last_tu: {spec.kernel_entry.name} "
                        f"builds {includes} but is the only kernel TU"
                    )

    spec.tiling_key_header, notes = _tiling_key_header(
        spec.kernel_root,
        spec.arch_dir,
        spec.op_name,
        kernel_entry=spec.kernel_entry,
        op_dir=spec.op_dir,
    )
    spec.ambiguities.extend(notes)

    spec.tiling_data_header = _tiling_data_header(spec.kernel_root, spec.arch_dir)

    spec.kernel_headers = []
    seen_h: set[Path] = set()
    for folder in iter_arch_source_dirs(spec.kernel_root, spec.arch_dir):
        for header in sorted(folder.glob("*.h")):
            key = header.resolve()
            if key in seen_h:
                continue
            seen_h.add(key)
            spec.kernel_headers.append(header)

    proto_root = op_dir / "op_graph"
    if proto_root.is_dir():
        protos = sorted(proto_root.glob("*_proto.h"))
        spec.proto = protos[0] if protos else None

    docs_root = op_dir / "docs"
    if docs_root.is_dir():
        spec.docs = sorted(docs_root.glob("aclnn*.md"))

    return spec


def _apply_override(spec: OpSpec, override: dict[str, Any]) -> OpSpec:
    """A pinned spec replaces discovery entirely: partial pinning hides drift."""
    spec.source = "override"
    spec.ambiguities = []

    def as_path(value: Any) -> Path | None:
        return spec.op_dir / str(value) if value else None

    spec.op_name = str(override.get("op_name") or spec.op_name)
    spec.op_snake = str(override.get("op_snake") or camel_to_snake(spec.op_name))
    spec.arch_dir = str(override.get("arch_dir") or spec.arch_dir)
    spec.opdef = as_path(override.get("opdef")) or spec.opdef
    spec.host_targets = [spec.op_dir / p for p in (override.get("host_targets") or [])]
    spec.kernel_entry = as_path(override.get("kernel_entry"))
    spec.tiling_key_header = as_path(override.get("tiling_key_header"))
    spec.tiling_data_header = as_path(override.get("tiling_data_header"))
    spec.proto = as_path(override.get("proto"))
    spec.docs = [spec.op_dir / p for p in (override.get("docs") or [])]

    for label, path in (
        ("opdef", spec.opdef),
        ("kernel_entry", spec.kernel_entry),
        ("tiling_key_header", spec.tiling_key_header),
    ):
        if path is not None and not path.is_file():
            spec.ambiguities.append(f"override_missing_{label}: {path}")
    for path in spec.host_targets:
        if not path.is_file():
            spec.ambiguities.append(f"override_missing_host_target: {path}")
    return spec
