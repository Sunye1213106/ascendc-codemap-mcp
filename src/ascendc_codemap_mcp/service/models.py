# -*- coding: utf-8 -*-
"""Pydantic output models so MCP tools publish a real outputSchema."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Coverage(BaseModel):
    model_config = ConfigDict(extra="allow")
    returned: int = 0
    total: int = 0
    truncated: bool = False
    nested_truncated: bool = False
    token_budget: int = 24_000


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = ""
    entity_id: str = ""
    file: str = ""
    line: int = 0
    source_stat_fingerprint: str = ""
    snapshot_id: str = ""


class CodemapHandle(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = ""
    alias: str = ""
    snapshot_id: str = ""
    architecture: str = ""
    op_name: str = ""
    project: str = ""
    path: str = ""
    source_revision: str = ""
    indexed_revision: str = ""
    freshness: str = ""
    semantic_completeness: float | None = None
    format: str = "codemap-uo"
    indexed: bool | None = None


class Envelope(BaseModel):
    """Shared structured result for CodeMap tools."""

    model_config = ConfigDict(extra="allow")
    ok: bool
    state: str | None = None
    updated: bool | None = None
    error: str | None = None
    error_code: str | None = None
    codemap: CodemapHandle | None = None
    verdict: str | None = None
    layer: str | None = None
    data: dict[str, Any] | None = None
    evidence: list[EvidenceItem] | None = None
    coverage: Coverage | None = None
    next_cursor: str | None = None


class DoctorResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    ok: bool
    engine: str = "codemap_doctor"
    version: str = ""
    project: str = ""
    architecture: str = ""
    cann_root: str | None = None
    clang_exe: str | None = None
    libclang_ok: bool = False
    issues: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    explain: str = ""
    product_dir: str = ""
    runtime: dict[str, Any] = Field(default_factory=dict)
