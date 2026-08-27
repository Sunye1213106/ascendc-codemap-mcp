# -*- coding: utf-8 -*-
"""Where the external trees live: CANN headers and the operator sources.

Neither tree ships with this repository, and their location differs per
checkout. Previously each caller hard-coded one developer's layout, so on any
other machine the CANN-dependent tests skipped silently and the suite still
reported green. Resolution now goes explicit argument, then environment, then
the OpenCode cache file, then a short list of layouts relative to the
repository -- and when nothing matches, `explain()` says what was tried so the
failure is actionable instead of silent.

``doctor``, ``scripts/dev/check_cann.py`` and prepare all call
``require_cann_ready()``; a green check must mean prepare's CANN gate would pass.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ascendc_codemap_mcp.engine.common_paths import strip_dot_slash


@lru_cache(maxsize=1 << 16)
def _resolved(text: str) -> Path:
    try:
        return Path(text).expanduser().resolve()
    except OSError:
        return Path(text)


def resolved(path: Path | str) -> Path:
    """``Path.resolve()`` memoized on the spelling asked for.

    One analyze made ~50k of these, and each is a `_getfinalpathname` syscall
    Windows charges about 0.04ms for -- 5s of the stage spent re-asking the
    filesystem where the same few thousand files are. A stage does not move
    files while it runs, so within one the answer is a constant. Failure is
    cached too: a path that does not resolve will not resolve on the retry
    either, and the original callers all fell back to the input on OSError.
    """
    return _resolved(str(path))

#: Written by `scripts/cann_slim.py` once the trimmed tree has been proven to
#: parse the host translation units. Only a tree carrying it is auto-selected;
#: an unverified trim would fail in confusing ways deep inside clang.
#:
#: The marker records the digest of the spec that decided what to copy. What it
#: guards against is a trimmed tree that parses one operator and is missing
#: headers the next one needs: that tree looks verified, gets preferred over the
#: full one, and the shortfall surfaces much later as an unexplained parse
#: error. Comparing digests catches the stale tree at resolution time instead.
SLIM_MARKER = ".slim-verified"

#: Stored in place of the CANN install root, which is a property of the build
#: machine and not of the operator. A product that spelled toolkit headers
#: absolutely could only be read back on the machine that wrote it. Readers
#: resolve this against their own `cann_root()`.
CANN_MARKER = "<cann>/"

CANN_ENV_VARS = ("ASCENDC_CODEMAP_CANN_ROOT", "ASCEND_CANN_PACKAGE_PATH", "CANN_ROOT")
#: Official ``source set_env.sh`` install prefix. Used only when the directory
#: looks like a toolkit/install tree (``{host}/asc``).
CANN_HOME_ENV_VARS = ("ASCEND_HOME_PATH",)
CANN_DISCOVERY_ENV_VARS = CANN_ENV_VARS + CANN_HOME_ENV_VARS
OPS_ENV_VARS = ("ASCENDC_CODEMAP_OPS_ROOT", "OPS_ROOT", "OPS_TRANSFORMER_ROOT")
OP_DIR_ENV_VARS = ("ASCENDC_PROJECT_ROOT", "ASCENDC_CODEMAP_OP_DIR")

# Typical extracted-package relatives (host tuple is a placeholder). Used as
# a fixture helper and by include-heal search, NOT as a prepare fail-closed
# inventory. Official CANN .run trees are complete; ``asc/impl/include`` is a
# clang shim we create, not a file the vendor ships.
REQUIRED_CANN_RELATIVE = (
    "cann-asc-devkit/x86_64-linux/asc/include",
    "cann-asc-devkit/x86_64-linux/asc/include/basic_api",
    "cann-asc-devkit/x86_64-linux/asc/include/utils/std",
    "cann-asc-devkit/x86_64-linux/asc/include/utils/std/tuple.h",
    "cann-asc-devkit/x86_64-linux/asc/impl/basic_api",
    "cann-asc-devkit/x86_64-linux/asc/impl/include",
    "cann-metadef/x86_64-linux/include",
    "cann-npu-runtime/x86_64-linux/include/base/alog_pub.h",
    "cann-opbase/x86_64-linux/include/op_common/op_host/util/math_util.h",
)

EXTRACTED_CANN_PACKAGES = (
    "cann-asc-devkit",
    "cann-metadef",
    "cann-opbase",
    "cann-npu-runtime",
    "cann-ge-compiler",
    "cann-tbe-tik",
)


def repo_root() -> Path:
    """The AscendC CodeMap MCP checkout root."""
    return Path(__file__).resolve().parents[3]


def resolve_operator_file(op_root: Path, raw: str) -> Path | None:
    """The file a stored or extracted location names, or None.

    Locations reach this from several bases at once: operator-relative
    (`op_kernel/x.h`), ops-root-relative (`<op>/op_kernel/x.h`), a sibling tree
    (`../common/x.h`, `../../common/include/x.h`), the `<cann>/` marker, or an
    absolute path from clang. Seven passes each kept their own copy of this and
    all seven shared two defects: `lstrip('./')` is a character set, so it ate
    the parents off `../common/x.h` and left a path that resolves under the
    wrong tree, and only one level of parent was ever tried.
    """
    text = strip_dot_slash(str(raw or "").replace("\\", "/"))
    if not text:
        return None
    if text.startswith(CANN_MARKER):
        root = cann_root()
        if root is None:
            return None
        candidate = root / text[len(CANN_MARKER) :]
        return candidate if candidate.is_file() else None
    direct = Path(text)
    if direct.is_absolute():
        return direct if direct.is_file() else None

    # A path spelled from an ancestor already says how far up it goes, so
    # joining is enough. A bare relative one does not, and may be spelled from
    # any ancestor up to the checkout, so try each instead of assuming the parent.
    candidates = [op_root / text]
    if not text.startswith("../"):
        base = op_root
        for _ in range(3):
            base = base.parent
            candidates.append(base / text)
        if text.startswith(op_root.name + "/"):
            candidates.append(op_root / text[len(op_root.name) + 1 :])
    for path in candidates:
        if path.is_file():
            return path
    return None


def _env(names: tuple[str, ...]) -> Path | None:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            return Path(raw).expanduser()
    return None


def _first_existing_dir(names: tuple[str, ...]) -> Path | None:
    """First env var that names an existing directory; skip stale missing paths."""
    for name in names:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        got = Path(raw).expanduser()
        try:
            if got.is_dir():
                return got
        except OSError:
            continue
    return None


def opencode_cann_root_cache_path() -> Path:
    """Same file ``write_opencode_cann_root`` / the OpenCode plugin read.

    Optional OpenCode-written CANN root cache, if present.
    """
    xdg = (os.environ.get("XDG_CONFIG_HOME") or "").strip()
    if xdg:
        return Path(xdg).expanduser() / "opencode" / "ascendc-cann-root"
    return Path.home() / ".config" / "opencode" / "ascendc-cann-root"


def read_cached_cann_root() -> Path | None:
    cache = opencode_cann_root_cache_path()
    try:
        raw = cache.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    except OSError:
        return None
    if not raw:
        return None
    got = Path(raw).expanduser()
    try:
        return got if got.is_dir() else None
    except OSError:
        return None


def _usable_cann(path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        if path.is_dir() and _looks_like_cann(path):
            return path
    except OSError:
        return None
    return None


#: Toolkit packages use one of these host tuples under each sub-package.
CANN_HOST_DIRS = ("x86_64-linux", "aarch64-linux")


def cann_host_dir(root: Path | None) -> str | None:
    """Return ``x86_64-linux`` / ``aarch64-linux`` (or any ``*-linux``) under a tree.

    Understands both ``cann_extract.py`` package layout and an official install
    (``$ASCEND_HOME_PATH/{host}/asc``), including a root that *is* the host dir.
    """
    if root is None or not root.is_dir():
        return None
    if root.name.endswith("-linux") and (
        (root / "asc").is_dir() or (root / "include").is_dir()
    ):
        return root.name
    search = [root / name for name in EXTRACTED_CANN_PACKAGES]
    search.append(root)
    for base in search:
        if not base.is_dir():
            continue
        named = [name for name in CANN_HOST_DIRS if (base / name).is_dir()]
        if named:
            return named[0]
        try:
            found = sorted(
                child.name
                for child in base.iterdir()
                if child.is_dir() and child.name.endswith("-linux")
            )
        except OSError:
            found = []
        if found:
            return found[0]
    return None


def required_cann_relative(root: Path | None = None) -> tuple[str, ...]:
    """REQUIRED_CANN_RELATIVE with the tree's host tuple substituted."""
    host = cann_host_dir(root) or "x86_64-linux"
    if host == "x86_64-linux":
        return REQUIRED_CANN_RELATIVE
    return tuple(p.replace("/x86_64-linux/", f"/{host}/") for p in REQUIRED_CANN_RELATIVE)


def _cann_candidates() -> list[Path]:
    """Auto-discovery only. Never include a machine-specific absolute path.

    Order: checkout ``_cann/`` first (so ``cann_extract.py --dest _cann/pkg``
    works with no env), then sibling checkouts, then ``~/ascendc/cann/pkg``.
    """
    repo = repo_root()
    out: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        out.append(path)

    for base in (repo, repo.parent, repo.parent.parent):
        add(base / "_cann" / "slim")
        add(base / "_cann" / "pkg")
    add(Path.home() / "ascendc" / "cann" / "pkg")
    return out


def _ops_candidates() -> list[Path]:
    repo = repo_root()
    return [
        repo.parent / "ops-transformer",
        repo.parent / "TEST" / "ops-transformer",
        repo.parent.parent / "ops-transformer",
        repo.parent.parent / "TEST" / "ops-transformer",
    ]


def spec_path() -> Path:
    """The build context that decides which headers a trimmed tree must hold."""
    return repo_root() / "spec" / "build_context.yaml"


def spec_digest() -> str | None:
    try:
        return hashlib.sha256(spec_path().read_bytes()).hexdigest()
    except OSError:
        return None


def slim_status(path: Path) -> str | None:
    """Why `path` is not a usable trimmed tree, or None when it is.

    Read by `explain()` so a skipped trim is reported rather than silently
    stepped over.
    """
    marker = path / SLIM_MARKER
    if not marker.exists():
        return "no verification marker; run scripts/cann_slim.py"
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Markers written before they carried a digest. Those trees predate the
        # completeness check and cannot be shown to match the current spec.
        return "verification marker predates spec pinning; re-run scripts/cann_slim.py"
    want = spec_digest()
    got = record.get("spec_digest")
    if want is not None and got != want:
        return f"built from a different build_context.yaml ({got} != {want})"
    return None


def _looks_like_cann(path: Path) -> bool:
    """Extracted ``cann-*`` packages *or* an official toolkit/install prefix."""
    if not path.is_dir():
        return False
    if any((path / name).is_dir() for name in ("cann-metadef", "cann-asc-devkit")):
        return True
    if (path / "asc" / "include").is_dir() or (path / "asc" / "impl").is_dir():
        return True
    for host in CANN_HOST_DIRS:
        if (path / host / "asc").is_dir():
            return True
    return path.name.endswith("-linux") and (path / "asc").is_dir()


def _looks_like_ops(path: Path) -> bool:
    return path.is_dir() and (path / "common" / "include").is_dir()


def _discover_cann_root() -> Path | None:
    """Env → official home → OpenCode cache → checkout candidates.

    A set-but-missing env path, or a directory that is not a CANN tree, is
    skipped so a leftover ``UO_CANN_ROOT`` / cache cannot hide ``_cann/pkg``.
    """
    for got in (
        _first_existing_dir(CANN_ENV_VARS),
        _first_existing_dir(CANN_HOME_ENV_VARS),
        read_cached_cann_root(),
    ):
        usable = _usable_cann(got)
        if usable is not None:
            return usable
    for cand in _cann_candidates():
        if not _looks_like_cann(cand):
            continue
        if cand.name == "slim" and slim_status(cand) is not None:
            continue
        return cand
    return None


def cann_root(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """The extracted or installed CANN tree, or None when it cannot be located.

    Explicit / ``UO_CANN_ROOT`` paths that exist but are not a CANN tree are
    ignored so prepare can still find the tree doctor already resolved.
    """
    text = "" if explicit is None else str(explicit).strip()
    if text:
        got = _usable_cann(Path(text).expanduser())
        if got is not None:
            return got
    return _discover_cann_root()


def adapt_cann_fs_path(path: str, root: Path | None) -> str:
    """Map yaml ``cann-*/{host}/...`` onto an official install if needed.

    Extracted layout keeps the package prefix. Installed toolkit merges
    packages under ``{root}/{host}/...`` (and ``{root}/asc``). Missing
    individual files are not rewritten — clang / include_heal handle those.
    """
    posix = str(path or "").replace("\\", "/")
    if not posix or root is None:
        return posix
    try:
        if Path(posix).exists():
            return posix
    except OSError:
        pass
    root_s = str(root).replace("\\", "/").rstrip("/")
    if not root_s:
        return posix
    prefix = root_s.lower() + "/"
    if not posix.lower().startswith(prefix):
        return posix
    rest = posix[len(root_s) + 1 :]
    parts = rest.split("/")
    if parts and parts[0].startswith("cann-"):
        alt = root.joinpath(*parts[1:])
        try:
            if alt.exists():
                return alt.as_posix()
        except OSError:
            pass
    return posix


def resolve_cann_relative(root: Path, rel: str) -> Path:
    """``root / rel`` with host substitution and install-layout fallback."""
    host = cann_host_dir(root) or "x86_64-linux"
    rel_h = rel.replace("x86_64-linux", host)
    mapped = adapt_cann_fs_path(str(root / rel_h).replace("\\", "/"), root)
    return Path(mapped)


def iter_asc_dirs(root: Path) -> list[Path]:
    host = cann_host_dir(root)
    cands: list[Path] = []
    if host:
        cands.append(root / "cann-asc-devkit" / host / "asc")
        cands.append(root / host / "asc")
    cands.append(root / "asc")
    out: list[Path] = []
    seen: set[str] = set()
    for path in cands:
        try:
            if not path.is_dir():
                continue
        except OSError:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _load_cann_extract():  # pragma: no cover - import plumbing
    import importlib.util

    path = repo_root() / "scripts" / "cann_extract.py"
    spec = importlib.util.spec_from_file_location("_pilot_cann_extract", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_impl_include_shim(root: Path) -> None:
    """Best-effort ``asc/impl/include`` → ``asc/include`` for vanilla clang.

    Official CANN never ships this directory. Relative includes from
    ``asc/impl/...`` need it; missing it must not fail prepare.
    """
    pairs = [(asc / "impl" / "include", asc / "include") for asc in iter_asc_dirs(root)]
    if not pairs:
        return
    try:
        ce = _load_cann_extract()
    except Exception:  # noqa: BLE001
        ce = None
    for shim, include in pairs:
        if not include.is_dir():
            continue
        try:
            if shim.exists():
                continue
        except OSError:
            pass
        try:
            if ce is not None:
                ce.make_dir_link(shim, include, copy_fallback=True)
                continue
        except Exception:  # noqa: BLE001
            pass
        try:
            shim.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(include, shim, target_is_directory=True)
        except OSError:
            continue


def cann_layout_issues(root: Path | None = None) -> list[str]:
    """Blockers for prepare/scope_scan: cann_root missing or not a CANN tree.

    Official packages are complete. Do not fail closed on a hardcoded file
    inventory (``tuple.h``, ``alog_pub.h``, ``asc/impl/include``, …).
    """
    path = root if root is not None else cann_root()
    if path is None:
        repo = repo_root()
        looked = [str(p) for p in _cann_candidates()]
        looked.append(f"cache {opencode_cann_root_cache_path()}")
        return [
            "CANN packages not found. Set UO_CANN_ROOT / ASCEND_CANN_PACKAGE_PATH "
            "/ ASCEND_HOME_PATH "
            f"or run: python scripts/cann_extract.py <toolkit.run> --dest {repo / '_cann' / 'pkg'}"
            "\nLooked in:\n"
            + "\n".join(f"  {p}" for p in looked)
        ]
    if not _looks_like_cann(path):
        return [
            f"{path} does not look like a CANN root "
            "(need cann-asc-devkit/, cann-metadef/, or <host>/asc from an official install)."
        ]
    return []


def require_cann_ready(
    explicit: str | os.PathLike[str] | None = None,
) -> tuple[Path | None, list[str]]:
    """Single CANN gate for doctor, check_cann, and prepare (empty issues => ready)."""
    root = cann_root(explicit)
    if root is not None:
        ensure_impl_include_shim(root)
    return root, cann_layout_issues(root)


def ops_root(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """The operator source/dependency checkout, or None."""
    if explicit:
        got = Path(explicit).expanduser()
        return got if got.is_dir() else None
    got = _env(OPS_ENV_VARS)
    if got is not None:
        return got if got.is_dir() else None
    for cand in _ops_candidates():
        if _looks_like_ops(cand):
            return cand
    return None


def op_dir(
    explicit: str | os.PathLike[str] | None = None,
    *,
    relative: str,
) -> Path | None:
    """One operator's directory inside a source/dependency root.

    `relative` has no default: which operator is under analysis is an input to
    this tool, never a property of it.
    """
    if explicit:
        got = Path(explicit).expanduser()
        return got if got.is_dir() else None
    got = _env(OP_DIR_ENV_VARS)
    if got is not None:
        return got if got.is_dir() else None
    ops = ops_root()
    if ops is None:
        return None
    cand = ops.joinpath(*relative.split("/"))
    return cand if cand.is_dir() else None


@dataclass(frozen=True)
class Resolution:
    name: str
    value: Path | None
    env_vars: tuple[str, ...]
    tried: tuple[Path, ...]

    def explain(self) -> str:
        if self.value is not None:
            return f"{self.name}: {self.value}"
        env = " or ".join(self.env_vars)
        tried = "\n".join(f"    {p}" for p in self.tried)
        return f"{self.name}: NOT FOUND (set {env})\n  looked in:\n{tried}"


def resolve_all() -> list[Resolution]:
    """Everything this repository needs from outside, for diagnostics."""
    return [
        Resolution(
            "cann_root",
            cann_root(),
            CANN_DISCOVERY_ENV_VARS,
            tuple(_cann_candidates()) + (opencode_cann_root_cache_path(),),
        ),
        Resolution("ops_root", ops_root(), OPS_ENV_VARS, tuple(_ops_candidates())),
    ]


def explain() -> str:
    lines = [r.explain() for r in resolve_all()]
    for cand in _cann_candidates():
        if cand.name != "slim" or not cand.is_dir():
            continue
        why = slim_status(cand)
        if why is not None:
            lines.append(f"  skipped trimmed tree {cand}: {why}")
    return "\n".join(lines)


PRODUCT_DIR_NAME = ".ascendc-codemap"


def product_dir(op_root: Path, architecture: str) -> Path:
    """Arch-scoped CodeMap working tree ``<op>/.ascendc-codemap/<arch>/``."""
    arch = (architecture or "").strip()
    if not arch:
        raise ValueError("ARCHITECTURE_MISSING_IN_RUN_STATE: architecture required")
    return Path(op_root).expanduser().resolve() / PRODUCT_DIR_NAME / arch


def require_architecture(value: str | None) -> str:
    """Return non-empty architecture or raise a typed control-plane error."""
    arch = (value or "").strip()
    if not arch:
        raise ValueError("ARCHITECTURE_MISSING_IN_RUN_STATE")
    return arch


def architecture_from_env() -> str:
    for name in ("UO_ARCH", "ASCENDC_ARCH"):
        raw = (os.environ.get(name) or "").strip()
        if raw:
            return raw
    raise ValueError("ARCHITECTURE_MISSING_IN_RUN_STATE: architecture required")


if __name__ == "__main__":
    import sys

    print(f"repo_root: {repo_root()}")
    print(explain())
    for relative in sys.argv[1:]:
        print(f"op_dir({relative}): {op_dir(relative=relative)}")
