# -*- coding: utf-8 -*-
"""Unpack a CANN makeself `.run` installer without running its shell script.

A makeself archive is a shell script followed by a compressed tar payload. The
header declares how many lines the script occupies (`skip=`) and how many bytes
the payload is (`filesizes=`); everything else is derived, so this works across
CANN versions rather than hard-coding one release's byte offset.

CANN nests archives: the outer `.run` contains one `.run` per sub-package.

On Windows the tar members that are symlinks cannot be created the way tar
intends, so they are collected and replayed after every regular file exists:
directories become junctions, files become copies. Without this the toolkit's
include tree has holes where clang expects headers.
"""

from __future__ import annotations

import argparse
import gzip
import io
import lzma
import bz2
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

_SKIP_RE = re.compile(rb'^\s*skip\s*=\s*"?(\d+)"?\s*$', re.M)
_SIZES_RE = re.compile(rb'^\s*filesizes\s*=\s*"?([\d\s]+)"?\s*$', re.M)

_HEADER_SCAN_BYTES = 1 << 20

#: The `.run` files carry a version in their name; an installed toolkit does
#: not, and build_context.yaml refers to the installed names. Strip the version
#: so include paths stay stable across CANN releases.
_VERSION_SUFFIX_RE = re.compile(r"_\d[\w.\-]*$")

#: One package is not named after its installed directory: the installer calls
#: it cann-bisheng-compiler, the toolkit lays it down as `bisheng`.
_INSTALLED_NAME = {"cann-bisheng-compiler": "bisheng"}


@dataclass
class Payload:
    offset: int
    size: int
    compression: str


@dataclass
class LinkPlan:
    """Symlinks tar asked for, replayed after extraction."""

    links: list[tuple[Path, str]] = field(default_factory=list)
    made: int = 0
    copied: int = 0
    skipped: list[str] = field(default_factory=list)


def read_payload_spec(run: Path) -> Payload:
    with run.open("rb") as fh:
        head = fh.read(_HEADER_SCAN_BYTES)
    skip_m = _SKIP_RE.search(head)
    if skip_m is None:
        raise SystemExit(f"{run.name}: no makeself `skip=` line; not a makeself archive?")
    skip = int(skip_m.group(1))

    offset = 0
    seen = 0
    with run.open("rb") as fh:
        while seen < skip:
            line = fh.readline()
            if not line:
                raise SystemExit(f"{run.name}: file ended inside the header")
            offset += len(line)
            seen += 1
        magic = fh.read(8)

    total = run.stat().st_size
    size = total - offset
    sizes_m = _SIZES_RE.search(head)
    if sizes_m:
        declared = sum(int(x) for x in sizes_m.group(1).split())
        if declared != size:
            # Trust the arithmetic but say so: a mismatch means the header moved.
            print(f"  warn: filesizes={declared} but {size} bytes follow the header")

    if magic[:2] == b"\x1f\x8b":
        comp = "gzip"
    elif magic[:6] == bytes.fromhex("fd377a585a00"):
        comp = "xz"
    elif magic[:3] == b"BZh":
        comp = "bzip2"
    else:
        raise SystemExit(f"{run.name}: unknown payload magic {magic[:8].hex()}")
    return Payload(offset=offset, size=size, compression=comp)


class _Slice(io.RawIOBase):
    """The payload region of the .run, as a read-only stream."""

    def __init__(self, path: Path, offset: int, size: int) -> None:
        self._fh = path.open("rb")
        self._fh.seek(offset)
        self._left = size

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:  # type: ignore[override]
        if self._left <= 0:
            return 0
        want = min(len(buf), self._left)
        got = self._fh.readinto(memoryview(buf)[:want])
        self._left -= got or 0
        return got or 0

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def open_payload(run: Path, spec: Payload):
    raw = io.BufferedReader(_Slice(run, spec.offset, spec.size), buffer_size=1 << 20)
    if spec.compression == "gzip":
        return gzip.GzipFile(fileobj=raw)
    if spec.compression == "xz":
        return lzma.open(raw)
    return bz2.open(raw)


def canonical_name(run_name: str) -> str:
    """Installed directory name for an inner archive file name.

    `cann-asc-devkit_9.1.0_linux-x86_64.run` -> `cann-asc-devkit`
    """
    name = run_name
    for suffix in ("_linux-x86_64.run", "_linux-aarch64.run", "_linux-x86.run", ".run"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = _VERSION_SUFFIX_RE.sub("", name)
    return _INSTALLED_NAME.get(name, name)


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract(run: Path, dest: Path, plan: LinkPlan) -> int:
    """Extract one makeself archive. Returns the number of regular files."""
    spec = read_payload_spec(run)
    print(f"  payload: offset={spec.offset} size={spec.size} {spec.compression}")
    dest.mkdir(parents=True, exist_ok=True)
    files = 0
    with open_payload(run, spec) as stream:
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            for member in tar:
                out = dest / member.name
                if not _is_within(dest, out):
                    plan.skipped.append(f"path escapes dest: {member.name}")
                    continue
                if member.issym() or member.islnk():
                    # Replayed later: the target may not exist yet.
                    plan.links.append((out, member.linkname))
                    continue
                if member.isdir():
                    out.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isreg():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    continue
                with out.open("wb") as fh:
                    shutil.copyfileobj(src, fh, length=1 << 20)
                files += 1
                if files % 20000 == 0:
                    print(f"    {files} files...")
    return files


_REPARSE_POINT = 0x400


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
        return True
    except OSError:
        return False


def _is_reparse(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    if attrs:
        return bool(attrs & _REPARSE_POINT)
    return os.path.islink(path)


def unlink_reparse(path: Path) -> None:
    """Remove a junction/symlink without deleting the target tree."""
    if not _lexists(path):
        return
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", f'rmdir "{path}"'],
            check=False,
            capture_output=True,
        )
        if _lexists(path):
            subprocess.run(
                ["cmd", "/c", f'del /f "{path}"'],
                check=False,
                capture_output=True,
            )
        if _lexists(path):
            try:
                path.unlink()
            except OSError:
                pass
        return
    try:
        path.unlink()
    except OSError:
        pass


def mklink_junction_argv(link: Path, target: Path) -> list[str]:
    """Quoted so paths with spaces survive ``cmd /c``."""
    return ["cmd", "/c", f'mklink /J "{link}" "{target}"']


def make_dir_link(link: Path, target: Path, *, copy_fallback: bool = False) -> str:
    """Create ``link`` -> ``target`` directory.

    Returns ``junction``, ``symlink``, ``copy``, or ``exists``. Windows uses
    quoted ``mklink /J``. Dangling leftover reparse points are replaced.
    ``copy_fallback`` is for the one known ``impl/include`` shim — not for
    replaying every tar symlink (that would duplicate the toolkit).
    """
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"link target is not a directory: {target}")

    if _lexists(link):
        working = False
        try:
            working = (
                link.exists()
                and link.is_dir()
                and (_is_reparse(link) or link.is_symlink())
                and link.resolve() == target
            )
        except OSError:
            working = False
        if working:
            return "exists"
        populated_copy = (
            link.exists()
            and link.is_dir()
            and not (_is_reparse(link) or link.is_symlink())
            and any(link.iterdir())
        )
        if populated_copy:
            return "exists"
        unlink_reparse(link)
        if link.exists():
            if link.is_symlink() or link.is_file():
                link.unlink()
            elif link.is_dir():
                try:
                    link.rmdir()
                except OSError:
                    if _is_reparse(link) or link.is_symlink():
                        raise OSError(f"could not remove leftover reparse: {link}") from None
                    shutil.rmtree(link)

    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    proc = subprocess.run(
        mklink_junction_argv(link, target),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode == 0 and link.exists():
        return "junction"
    err = (proc.stderr or proc.stdout or f"rc={proc.returncode}").strip()
    if not copy_fallback:
        raise OSError(f"mklink /J failed: {err}")
    unlink_reparse(link)
    if _is_reparse(link) or link.is_symlink():
        raise OSError(f"could not remove leftover reparse before copy: {link}")
    if link.exists():
        shutil.rmtree(link)
    shutil.copytree(target, link, dirs_exist_ok=False, symlinks=False)
    return "copy"


def detect_toolkit_host(pkg: Path) -> str:
    """Host tuple under extracted packages or an official install prefix."""
    if pkg.name.endswith("-linux") and (pkg / "asc").is_dir():
        return pkg.name
    for name in ("cann-asc-devkit", "cann-metadef", "cann-opbase", "cann-npu-runtime", ""):
        base = pkg if not name else pkg / name
        if not base.is_dir():
            continue
        for host in ("x86_64-linux", "aarch64-linux"):
            if (base / host).is_dir():
                return host
        found = sorted(
            child.name
            for child in base.iterdir()
            if child.is_dir() and child.name.endswith("-linux")
        )
        if found:
            return found[0]
    return "x86_64-linux"


def replay_links(plan: LinkPlan) -> None:
    """Recreate tar symlinks as junctions (dirs) or copies (files).

    Live paths (``exists()``) are left alone — a CANN toolkit has thousands of
    these, and ``Path.resolve()`` on Windows junctions is the hot path.
    Dangling reparse points fail ``exists()`` and are rebuilt; do not also
    skip ``is_symlink()``, that was leaving header holes.
    """
    for link_path, target in plan.links:
        try:
            if link_path.exists():
                continue
        except OSError:
            pass
        resolved = (link_path.parent / target).resolve()
        link_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if resolved.is_dir():
                kind = make_dir_link(link_path, resolved, copy_fallback=False)
                if kind != "exists":
                    plan.made += 1
            elif resolved.is_file():
                if _lexists(link_path) and not link_path.exists():
                    unlink_reparse(link_path)
                if not link_path.exists():
                    shutil.copy2(resolved, link_path)
                    plan.copied += 1
            else:
                plan.skipped.append(f"dangling link {link_path} -> {target}")
        except Exception as exc:  # noqa: BLE001 - report and carry on
            plan.skipped.append(f"{link_path} -> {target}: {type(exc).__name__}: {exc}")


def iter_asc_dirs(pkg: Path) -> list[Path]:
    host = detect_toolkit_host(pkg)
    cands = [
        pkg / "cann-asc-devkit" / host / "asc",
        pkg / host / "asc",
        pkg / "asc",
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for path in cands:
        if not path.is_dir():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def apply_known_fixups(pkg: Path, plan: LinkPlan) -> None:
    """Layout repairs the toolkit assumes a POSIX filesystem provides.

    Official CANN does not ship ``asc/impl/include``; vanilla clang needs that
    junction so relative includes from ``asc/impl/...`` resolve. Missing it is
    not "the package is incomplete".
    """
    for asc in iter_asc_dirs(pkg):
        include = asc / "include"
        shim = asc / "impl" / "include"
        if not include.is_dir():
            continue
        try:
            kind = make_dir_link(shim, include, copy_fallback=True)
            if kind in {"junction", "symlink", "copy"}:
                plan.made += 1
                print(f"  fixup ({kind}): {shim} -> {include}")
            elif kind == "exists":
                print(f"  fixup already present: {shim}")
        except Exception as exc:  # noqa: BLE001
            plan.skipped.append(f"fixup {shim}: {exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ascendc-codemap-mcp cann-extract",
        description=__doc__,
    )
    ap.add_argument("run", type=Path, nargs="?", help="outer CANN .run installer")
    ap.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="where sub-packages land (recommend <checkout>/_cann/pkg)",
    )
    ap.add_argument(
        "--stage",
        type=Path,
        default=None,
        help="where the outer archive unpacks (default: <dest>/../run_package)",
    )
    ap.add_argument(
        "--inner",
        nargs="*",
        default=None,
        help="substrings of inner .run names to unpack (default: all)",
    )
    ap.add_argument("--list", action="store_true", help="only list inner archives")
    ap.add_argument(
        "--fixup",
        action="store_true",
        help="only recreate asc/impl/include (and skip unpack); repair dangling junctions",
    )
    args = ap.parse_args(argv)

    dest: Path = args.dest
    if args.fixup:
        if not dest.is_dir():
            raise SystemExit(f"--fixup dest is not a directory: {dest}")
        plan = LinkPlan()
        apply_known_fixups(dest, plan)
        print(f"fixup junctions={plan.made} skipped={len(plan.skipped)}")
        for line in plan.skipped[:20]:
            print(f"    {line}")
        return 0 if not plan.skipped else 1

    run: Path | None = args.run
    if run is None:
        raise SystemExit("provide the .run installer, or pass --fixup to repair an existing tree")
    if not run.is_file():
        raise SystemExit(f"no such file: {run}")
    stage: Path = args.stage or dest.parent / "run_package"

    plan = LinkPlan()

    marker = stage / ".extracted"
    if marker.exists():
        print(f"outer archive already staged at {stage}")
    else:
        print(f"unpacking outer archive -> {stage}")
        n = extract(run, stage, plan)
        marker.write_text(str(n), encoding="utf-8")
        print(f"  {n} files")

    inner = sorted(p for p in stage.rglob("*.run") if p.is_file())
    print(f"found {len(inner)} inner archives")
    if args.list:
        for p in inner:
            print(f"  {p.relative_to(stage)}  {p.stat().st_size / 1e6:.1f} MB")
        return 0

    wanted = inner
    if args.inner:
        needles = [s.lower() for s in args.inner]
        wanted = [p for p in inner if any(s in p.name.lower() for s in needles)]
        print(f"selected {len(wanted)} of them")

    for p in wanted:
        name = canonical_name(p.name)
        out = dest / name
        if (out / ".extracted").exists():
            print(f"skip {name} (already extracted)")
            continue
        print(f"unpacking {p.name} -> {out}")
        try:
            n = extract(p, out, plan)
        except SystemExit as exc:
            print(f"  not a makeself archive, skipping: {exc}")
            continue
        (out / ".extracted").write_text(str(n), encoding="utf-8")
        print(f"  {n} files")

    print(f"replaying {len(plan.links)} symlinks")
    replay_links(plan)
    apply_known_fixups(dest, plan)
    print(f"  junctions={plan.made} copies={plan.copied} skipped={len(plan.skipped)}")
    for line in plan.skipped[:20]:
        print(f"    {line}")
    if len(plan.skipped) > 20:
        print(f"    ... and {len(plan.skipped) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
