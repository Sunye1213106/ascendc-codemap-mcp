# -*- coding: utf-8 -*-
"""Per-CodeMap snapshot gate: queries never read a half-built ``.uo``."""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class _Gate:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.readers = 0
        self.writer = False


class SnapshotLocks:
    """Write-preferring gate keyed by ``codemap_id``.

    Writers drain in-flight readers and block new ones. Queries that arrive
    during an index/update fail fast with ``freshness=building`` rather than
    waiting, so they cannot observe a file mid-replace (important on Windows).
    """

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._gates: dict[str, _Gate] = {}

    def _gate(self, key: str) -> _Gate:
        with self._mu:
            gate = self._gates.get(key)
            if gate is None:
                gate = _Gate()
                self._gates[key] = gate
            return gate

    def is_writing(self, key: str) -> bool:
        gate = self._gate(key)
        with gate.cond:
            return bool(gate.writer)

    def try_read(self, key: str) -> bool:
        gate = self._gate(key)
        with gate.cond:
            if gate.writer:
                return False
            gate.readers += 1
            return True

    def release_read(self, key: str) -> None:
        gate = self._gate(key)
        with gate.cond:
            if gate.readers > 0:
                gate.readers -= 1
            gate.cond.notify_all()

    def acquire_write(self, key: str) -> None:
        gate = self._gate(key)
        with gate.cond:
            # Prefer the writer: new readers fail-fast once this is set.
            while gate.writer:
                gate.cond.wait()
            gate.writer = True
            while gate.readers:
                gate.cond.wait()

    def release_write(self, key: str) -> None:
        gate = self._gate(key)
        with gate.cond:
            gate.writer = False
            gate.cond.notify_all()

    @contextmanager
    def write(self, key: str) -> Iterator[None]:
        self.acquire_write(key)
        try:
            yield
        finally:
            self.release_write(key)

    @contextmanager
    def read(self, key: str) -> Iterator[bool]:
        ok = self.try_read(key)
        try:
            yield ok
        finally:
            if ok:
                self.release_read(key)
