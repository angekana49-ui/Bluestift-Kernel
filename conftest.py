"""Shared test fixtures: an in-memory fake Supabase client.

The fake mirrors just enough of supabase-py's fluent query builder for the
Kernel's data access: schema().table().select()/insert()/upsert()/update()
with .eq()/.ilike()/.limit()/.execute(). It is deliberately small but faithful
to the call shapes used in services/db.py and services/kc_registry.py.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest


@dataclass
class _Result:
    data: list
    count: int | None = None


class _Query:
    def __init__(self, store: "FakeSupabase", schema: str, table: str):
        self._store = store
        self._key = f"{schema}.{table}"
        self._rows = store.tables.setdefault(self._key, [])
        self._filters: list = []         # (col, value, kind)
        self._op = "select"
        self._payload = None
        self._on_conflict = None
        self._limit = None
        self._count = None

    # --- terminal-ish builders ------------------------------------------- #
    def select(self, *_args, count=None):
        self._op = "select"
        self._count = count
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict=None):
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    # --- filters --------------------------------------------------------- #
    def eq(self, col, value):
        self._filters.append((col, value, "eq"))
        return self

    def ilike(self, col, value):
        self._filters.append((col, str(value).lower(), "ilike"))
        return self

    def limit(self, n):
        self._limit = n
        return self

    # --- helpers --------------------------------------------------------- #
    def _matches(self, row) -> bool:
        for col, value, kind in self._filters:
            cell = row.get(col)
            if kind == "ilike":
                if str(cell).lower() != value:
                    return False
            elif cell != value:
                return False
        return True

    def _conflict_keys(self):
        if self._on_conflict:
            return [k.strip() for k in self._on_conflict.split(",")]
        return []

    # --- execute --------------------------------------------------------- #
    def execute(self) -> _Result:
        if self._op == "select":
            rows = [r for r in self._rows if self._matches(r)]
            if self._limit is not None:
                rows = rows[: self._limit]
            return _Result(data=[dict(r) for r in rows], count=len(rows))

        if self._op == "insert":
            payloads = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted = []
            for p in payloads:
                row = dict(p)
                row.setdefault("id", str(uuid.uuid4()))
                self._rows.append(row)
                inserted.append(dict(row))
            return _Result(data=inserted)

        if self._op == "upsert":
            payloads = self._payload if isinstance(self._payload, list) else [self._payload]
            keys = self._conflict_keys()
            out = []
            for p in payloads:
                existing = None
                if keys:
                    existing = next(
                        (r for r in self._rows if all(r.get(k) == p.get(k) for k in keys)),
                        None,
                    )
                if existing is not None:
                    existing.update(p)
                    out.append(dict(existing))
                else:
                    row = dict(p)
                    row.setdefault("id", str(uuid.uuid4()))
                    self._rows.append(row)
                    out.append(dict(row))
            return _Result(data=out)

        if self._op == "update":
            updated = []
            for r in self._rows:
                if self._matches(r):
                    r.update(self._payload)
                    updated.append(dict(r))
            return _Result(data=updated)

        return _Result(data=[])


class _Schema:
    def __init__(self, store: "FakeSupabase", schema: str):
        self._store = store
        self._schema = schema

    def table(self, name):
        return _Query(self._store, self._schema, name)


@dataclass
class FakeSupabase:
    tables: dict = field(default_factory=dict)

    def schema(self, name):
        return _Schema(self, name)

    # supabase-py also allows client.table() (public schema); not used here.
    def table(self, name):
        return _Query(self, "public", name)

    # --- test seeding helper -------------------------------------------- #
    def seed(self, key: str, rows: list[dict]):
        self.tables.setdefault(key, []).extend(dict(r) for r in rows)


@pytest.fixture
def fake_supabase():
    return FakeSupabase()
