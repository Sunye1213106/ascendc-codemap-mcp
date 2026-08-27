# -*- coding: utf-8 -*-
"""Path canonicalization shared by TG / UO / CE.

Do not use ``str.lstrip("./")``: that is a character set, so it eats the
parents off ``../common/x.h`` and resolves a sibling file under the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CanonicalPath",
    "AMBIGUOUS",
    "posix_slash",
    "strip_dot_slash",
    "known_operator_prefix",
    "peel_known_prefix",
    "resolve_under_operator",
    "canonical_path",
]


AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class CanonicalPath:
    absolute_resolved: Path
    canonical_operator_rel: str
    canonical_repo_rel: str


def posix_slash(text: str | None) -> str:
    return str(text or "").replace("\\", "/").strip()


def strip_dot_slash(text: str | None) -> str:
    """Drop leading ``./`` segments. Never touches ``../``."""
    out = posix_slash(text)
    while out.startswith("./"):
        out = out[2:]
    return out


def known_operator_prefix(operator_root: Path, repo_root: Path | None) -> str:
    """Repo-relative posix prefix of the operator, or empty if not nested."""
    if repo_root is None:
        return ""
    try:
        rel = Path(operator_root).resolve().relative_to(Path(repo_root).resolve())
    except (ValueError, OSError):
        return ""
    return rel.as_posix().strip("/")


def peel_known_prefix(
    rel: str,
    *,
    operator_root: Path,
    repo_root: Path | None = None,
) -> str:
    """Strip only the known repo_root → operator_root prefix. Never suffix-guess."""
    text = strip_dot_slash(rel)
    prefix = known_operator_prefix(operator_root, repo_root)
    if prefix and text == prefix:
        return ""
    if prefix and text.startswith(prefix + "/"):
        return text[len(prefix) + 1 :]
    return text


def _rel_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.resolve().as_posix()


def resolve_under_operator(
    operator_root: Path,
    rel: str,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    """Map a contract path onto the operator tree. Fail closed on ambiguity.

    ``../`` is joined, not trimmed. Unknown repo prefixes are not peeled by
    trying every suffix. Bare basenames that match more than one file return
    None (callers treat that as AMBIGUOUS).
    """
    text = strip_dot_slash(rel)
    if not text:
        return None
    op = Path(operator_root)
    direct = Path(text)
    if direct.is_absolute():
        return direct if direct.is_file() else None

    peeled = peel_known_prefix(text, operator_root=op, repo_root=repo_root)
    candidates: list[Path] = []
    seen: set[Path] = set()
    for spelling in (peeled, text):
        if not spelling:
            continue
        try:
            cand = (op / spelling).resolve()
        except OSError:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        if cand.is_file():
            candidates.append(cand)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return None

    # Bare basename only: unique match under the operator, else AMBIGUOUS.
    if "/" not in peeled and "\\" not in peeled and peeled:
        hits = [p for p in op.rglob(peeled) if p.is_file()]
        if len(hits) == 1:
            return hits[0]
        return None
    return None


def canonical_path(
    operator_root: Path,
    rel: str,
    *,
    repo_root: Path | None = None,
) -> CanonicalPath | None:
    resolved = resolve_under_operator(operator_root, rel, repo_root=repo_root)
    if resolved is None:
        return None
    op = Path(operator_root).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else None
    return CanonicalPath(
        absolute_resolved=resolved,
        canonical_operator_rel=_rel_to(op, resolved),
        canonical_repo_rel=_rel_to(repo, resolved) if repo is not None else _rel_to(op, resolved),
    )
