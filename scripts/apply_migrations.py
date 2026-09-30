"""Apply the Kernel's SQL migrations via the Supabase Management API.

Runs each numbered file in migrations/ through POST /v1/projects/{ref}/database/query,
then exposes the `kernel` schema to PostgREST so the REST client can reach it.

Requires SUPABASE_ACCESS_TOKEN (a personal access token, sbp_...) and
SUPABASE_URL in the environment (.env). The project ref is derived from the URL.

Usage:
    python scripts/apply_migrations.py                      # every migration, in order
    python scripts/apply_migrations.py 011 012              # only these (prefix match)

Apply only what is new on an existing database: the early migrations create RLS
policies that are not all safe to replay.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

API = "https://api.supabase.com"
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# Schemas PostgREST must expose. The database is SHARED with the RAYA app, so
# this is the UNION both sides need (the same list as migration 009): setting
# only the Kernel's schemas would drop `learning` and `content` and break the
# app — the recurring shared-DB regression KERNEL_HANDOFF.md §6 warns about.
EXPOSED_SCHEMAS = "public, graphql_public, kernel, learning, schools, rag, content"


def _project_ref() -> str:
    url = os.getenv("SUPABASE_URL", "")
    m = re.match(r"https://([a-z0-9]+)\.supabase\.co", url)
    if not m:
        sys.exit(f"Could not parse project ref from SUPABASE_URL={url!r}")
    return m.group(1)


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def run_sql(client: httpx.Client, ref: str, token: str, sql: str) -> None:
    resp = client.post(
        f"{API}/v1/projects/{ref}/database/query",
        headers=_headers(token),
        json={"query": sql},
        timeout=120,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"SQL failed [{resp.status_code}]: {resp.text}")


def set_exposed_schemas(client: httpx.Client, ref: str, token: str) -> None:
    resp = client.patch(
        f"{API}/v1/projects/{ref}/postgrest",
        headers=_headers(token),
        json={"db_schema": EXPOSED_SCHEMAS},
        timeout=60,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Exposing schemas failed [{resp.status_code}]: {resp.text}")


def main() -> None:
    token = os.getenv("SUPABASE_ACCESS_TOKEN")
    if not token:
        sys.exit("SUPABASE_ACCESS_TOKEN is not set in .env")
    ref = _project_ref()

    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    wanted = sys.argv[1:]
    if wanted:
        files = [f for f in files if any(f.name.startswith(w) for w in wanted)]
        missing = [w for w in wanted if not any(f.name.startswith(w) for f in files)]
        if missing:
            sys.exit(f"No migration matches: {', '.join(missing)}")
    if not files:
        sys.exit(f"No .sql files found in {MIGRATIONS_DIR}")

    with httpx.Client() as client:
        for f in files:
            print(f"-> applying {f.name} ...", flush=True)
            run_sql(client, ref, token, f.read_text(encoding="utf-8"))
            print(f"   ok: {f.name}")

        print("-> exposing schemas to PostgREST ...", flush=True)
        set_exposed_schemas(client, ref, token)
        print(f"   ok: {EXPOSED_SCHEMAS}")

    print("\nAll migrations applied. PostgREST may take a few seconds to reload.")


if __name__ == "__main__":
    main()
