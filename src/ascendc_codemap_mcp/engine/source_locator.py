# -*- coding: utf-8 -*-
"""Source locator over a unified ``.uo`` CodeMap.

Production resolution is ``.uo`` only.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

_KIND_ALIASES: dict[str, tuple[str, ...]] = {
    "TilingKeyDim": ("TILING_KEY", "TilingKeyDim"),
    "TILING_KEY": ("TILING_KEY", "TilingKeyDim"),
    "HostBranch": ("BRANCH", "HostBranch", "KernelBranch"),
    "KernelBranch": ("BRANCH", "HostBranch", "KernelBranch"),
    "BRANCH": ("BRANCH", "HostBranch", "KernelBranch"),
    "Predicate": ("PREDICATE", "Predicate"),
    "PREDICATE": ("PREDICATE", "Predicate"),
    "TilingDataField": ("TILING_FIELD", "TilingDataField"),
    "TILING_FIELD": ("TILING_FIELD", "TilingDataField"),
}

_SITE_KEYS = (
    "packing_value_sites",
    "host_writer_sites",
    "value_defining_sites",
    "producer_sites",
    "check_sites",
    "definition_sites",
    "fused_outer_candidates",
)


@dataclass(frozen=True)
class Location:
    entity_id: str
    kind: str
    file: str
    line_start: int
    line_end: int
    snippet: str = ""
    window_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _window_sha(snippet: str) -> str | None:
    text = str(snippet or "")
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expand_kinds(kinds: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in kinds or ():
        text = str(raw or "").strip()
        if not text:
            continue
        for item in _KIND_ALIASES.get(text, (text,)):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def snippet_from_site(site: dict[str, Any]) -> str:
    expr = str(site.get("expression") or "").strip()
    if expr:
        return expr[:400]
    guard = str(site.get("guard") or "").strip()
    if guard:
        return guard[:400]
    lhs = str(site.get("lhs") or "").strip()
    rhs = str(site.get("rhs") or "").strip()
    if lhs and rhs:
        return f"{lhs} = {rhs}"[:400]
    return str(site.get("function") or site.get("receiver") or "")[:400]


def locations_from_attr_sites(
    entity_id: str,
    kind: str,
    attrs: dict[str, Any] | None,
) -> list[Location]:
    """Turn already-extracted Host write/pack sites into jump locations."""
    if not isinstance(attrs, dict):
        return []
    out: list[Location] = []
    seen: set[tuple[str, int]] = set()
    for key in _SITE_KEYS:
        for site in attrs.get(key) or []:
            if not isinstance(site, dict):
                continue
            file = str(site.get("file") or "").replace("\\", "/")
            line = int(site.get("line") or site.get("line_start") or 0)
            if not file or line <= 0 or (file, line) in seen:
                continue
            seen.add((file, line))
            snippet = snippet_from_site(site)
            out.append(
                Location(
                    entity_id=entity_id,
                    kind=kind,
                    file=file,
                    line_start=line,
                    line_end=int(site.get("line_end") or line),
                    snippet=snippet,
                    window_sha256=_window_sha(snippet),
                )
            )
    return out


def resolve_locator_database(uo_root: str | Path) -> Path:
    """Resolve the ``.uo`` product. Direct sqlite files are migrate/test only."""
    root = Path(uo_root).expanduser().resolve()
    if root.is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(root)
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product

    found = find_uo_product(root)
    if found is not None and found.is_file() and found.suffix == ".uo":
        return found
    raise FileNotFoundError(
        f"no .uo product under {root}; expected "
        ".ascendc-codemap/<arch>/<op>.<arch>.uo"
    )


class SourceLocator:
    """Locate entities and Host write sites inside ``.uo``."""

    def __init__(self, database: str | Path):
        self.database = Path(database).expanduser().resolve()
        if not self.database.is_file():
            raise FileNotFoundError(self.database)
        self._table_names: set[str] | None = None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database.as_posix()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _tables(self) -> set[str]:
        if self._table_names is None:
            with self._connect() as connection:
                self._table_names = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
        return self._table_names

    def _is_codemap(self) -> bool:
        return "entity" in self._tables()

    def locate(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        limit: int = 20,
        include_sites: bool = False,
    ) -> list[Location]:
        needle = str(query or "").strip()
        if not needle:
            return []
        if self._is_codemap():
            return self._locate_codemap(
                needle, kinds=kinds, limit=limit, include_sites=include_sites
            )
        return []

    def locate_dim(self, name: str, *, limit: int = 20) -> list[Location]:
        return self.locate(
            name, kinds=("TilingKeyDim", "TILING_KEY"), limit=limit, include_sites=True
        )

    def locate_branch(self, branch_id: str, *, limit: int = 20) -> list[Location]:
        needle = str(branch_id or "").strip()
        if not needle:
            return []
        return self.locate(
            needle,
            kinds=("HostBranch", "KernelBranch", "Predicate", "BRANCH", "PREDICATE"),
            limit=limit,
        )

    def locate_field(self, name: str, *, limit: int = 20) -> list[Location]:
        needle = str(name or "").strip()
        if not needle:
            return []
        hits = self.locate(
            needle,
            kinds=("TilingDataField", "TILING_FIELD"),
            limit=limit,
            include_sites=True,
        )
        if hits or self._is_codemap():
            return hits
        return []

    def _locate_codemap(
        self,
        needle: str,
        *,
        kinds: Iterable[str] | None,
        limit: int,
        include_sites: bool,
    ) -> list[Location]:
        kinds_list = _expand_kinds(kinds)
        like = f"%{needle}%"
        params: list[Any] = [needle, needle, like, like, like]
        kind_filter = ""
        if kinds_list:
            placeholders = ",".join("?" for _ in kinds_list)
            kind_filter = f" AND e.kind IN ({placeholders})"
            params.extend(kinds_list)
        fetch = max(int(limit), 8)
        params.extend([needle, needle, fetch])
        sql = f"""
            SELECT
              e.id AS entity_id,
              e.kind AS kind,
              e.name AS name,
              e.data AS data,
              COALESCE(NULLIF(s.file, ''), e.file, '') AS file,
              COALESCE(NULLIF(s.line_start, 0), e.line_start, 0) AS line_start,
              COALESCE(NULLIF(s.line_end, 0), e.line_end, e.line_start, 0) AS line_end,
              IFNULL(s.snippet, '') AS snippet
            FROM entity e
            LEFT JOIN source_span s ON s.entity_id = e.id
            WHERE e.id = ?
               OR IFNULL(e.name, '') = ?
               OR e.id LIKE ?
               OR IFNULL(e.name, '') LIKE ?
               OR e.data LIKE ?
               {kind_filter}
            ORDER BY
              CASE WHEN e.id = ? OR IFNULL(e.name, '') = ? THEN 0 ELSE 1 END,
              e.kind, e.id, e.line_start
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, tuple(params)).fetchall()
        out: list[Location] = []
        seen: set[tuple[str, str, int]] = set()
        for row in rows:
            file = str(row["file"] or "").replace("\\", "/")
            line = int(row["line_start"] or 0)
            snippet = str(row["snippet"] or "")
            loc = Location(
                entity_id=str(row["entity_id"]),
                kind=str(row["kind"] or ""),
                file=file,
                line_start=line,
                line_end=int(row["line_end"] or line),
                snippet=snippet,
                window_sha256=_window_sha(snippet),
            )
            key = (loc.entity_id, loc.file, loc.line_start)
            if loc.file and loc.line_start > 0 and key not in seen:
                seen.add(key)
                out.append(loc)
            if include_sites:
                try:
                    attrs = json.loads(row["data"] or "{}")
                except json.JSONDecodeError:
                    attrs = {}
                if not isinstance(attrs, dict):
                    attrs = {}
                for extra in locations_from_attr_sites(
                    str(row["entity_id"]), str(row["kind"] or ""), attrs
                ):
                    extra_key = (extra.entity_id, extra.file, extra.line_start)
                    if extra_key in seen:
                        continue
                    seen.add(extra_key)
                    out.append(extra)
            if len(out) >= int(limit):
                break
        return out[: int(limit)]


def open_locator(uo_root: str | Path) -> SourceLocator:
    return SourceLocator(resolve_locator_database(uo_root))


def locate(
    uo_root: str | Path,
    query: str,
    *,
    kinds: Iterable[str] | None = None,
    limit: int = 20,
) -> list[Location]:
    return open_locator(uo_root).locate(query, kinds=kinds, limit=limit)
