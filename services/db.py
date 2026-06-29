"""Supabase data access for the Kernel.

Uses the service_role key (never the anon key). Only DML is issued from here —
DDL lives in the migration files. All helpers operate on the `kernel` schema.

The client is created lazily and cached so the app can boot for /health without
credentials present.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

KERNEL_SCHEMA = "kernel"

_client = None


def _bootstrap_ssl() -> None:
    """Use the OS trust store for TLS when available.

    On managed Windows machines TLS is often intercepted by a corporate proxy
    whose root CA lives in the Windows certificate store, not in certifi. The
    optional `truststore` package bridges to the OS store so httpx/supabase can
    verify the connection. No-op if truststore isn't installed.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - best effort; fall back to default CAs
        pass


def get_client():
    """Return a cached Supabase client built from the service_role key."""
    global _client
    if _client is None:
        _bootstrap_ssl()
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _client = create_client(url, key)
    return _client


def _kernel(client, table: str):
    """Shorthand for `client.schema('kernel').table(table)`."""
    return client.schema(KERNEL_SCHEMA).table(table)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Graph reads
# --------------------------------------------------------------------------- #
def load_concept_nodes(client) -> list[dict]:
    return _kernel(client, "concept_nodes").select("*").execute().data or []


def load_concept_edges(client) -> list[dict]:
    return _kernel(client, "concept_edges").select("*").execute().data or []


def count_concept_nodes(client) -> int:
    res = _kernel(client, "concept_nodes").select("id", count="exact").execute()
    return res.count or 0


def load_labels_for_subject(client, subject: str) -> list[str]:
    """Return the existing KC labels for a subject (for extraction grounding)."""
    res = _kernel(client, "concept_nodes").select("label").eq("subject", subject).execute()
    return [r["label"] for r in (res.data or []) if r.get("label")]


# --------------------------------------------------------------------------- #
# Student concept state
# --------------------------------------------------------------------------- #
def load_student_concept_states(client, user_id: str) -> list[dict]:
    return (
        _kernel(client, "student_concept_state")
        .select("*")
        .eq("user_id", user_id)
        .execute()
        .data
        or []
    )


def load_student_concept_state(client, user_id: str, concept_id: str) -> dict | None:
    res = (
        _kernel(client, "student_concept_state")
        .select("*")
        .eq("user_id", user_id)
        .eq("concept_id", concept_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def upsert_student_concept_state(client, state: dict) -> dict:
    """Insert or update one student_concept_state row (keyed by user+concept)."""
    state = {**state, "updated_at": _now_iso()}
    res = (
        _kernel(client, "student_concept_state")
        .upsert(state, on_conflict="user_id,concept_id")
        .execute()
    )
    return res.data[0] if res.data else state


def load_states_for_concept(client, concept_id: str) -> list[dict]:
    """All students' states for one KC — used by background calibration."""
    return (
        _kernel(client, "student_concept_state")
        .select("mastery_score_raw, partial_credit_avg, struggle_index, interactions_on_kc, lambda_personal")
        .eq("concept_id", concept_id)
        .execute()
        .data
        or []
    )


def update_concept_node(client, concept_id: str, fields: dict) -> None:
    _kernel(client, "concept_nodes").update(fields).eq("id", concept_id).execute()


# --------------------------------------------------------------------------- #
# Mindset
# --------------------------------------------------------------------------- #
def load_mindset(client, user_id: str) -> dict | None:
    res = (
        _kernel(client, "student_mindset_state")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def upsert_mindset(client, user_id: str, m_score: float, detected: str) -> None:
    _kernel(client, "student_mindset_state").upsert(
        {
            "user_id": user_id,
            "m_score": m_score,
            "detected_mindset": detected,
            "updated_at": _now_iso(),
        },
        on_conflict="user_id",
    ).execute()


# --------------------------------------------------------------------------- #
# Logging — requests, outputs, insights, monitoring
# --------------------------------------------------------------------------- #
def log_kernel_request(client, request_id: str, user_id: str, payload: dict) -> None:
    """Log every /analyze call, even ones that later error out."""
    _kernel(client, "kernel_requests").insert(
        {
            "id": request_id,
            "user_id": user_id,
            "trigger": payload.get("trigger"),
            "subject": payload.get("subject"),
            "level": payload.get("level"),
            "payload": payload,
            "created_at": _now_iso(),
        }
    ).execute()


def log_kernel_output(client, request_id: str, user_id: str, output: dict) -> None:
    _kernel(client, "kernel_outputs").insert(
        {
            "request_id": request_id,
            "user_id": user_id,
            "root_gap": output.get("root_gap"),
            "root_concept_id": output.get("root_concept_id"),
            "detection_path": output.get("detection_path"),
            "confidence": output.get("confidence"),
            "llm_used": output.get("llm_used"),
            "output": output,
            "created_at": _now_iso(),
        }
    ).execute()


def log_individual_insight(client, user_id: str, request_id: str, summary: str, root_gap: str | None) -> None:
    _kernel(client, "individual_insights").insert(
        {
            "user_id": user_id,
            "request_id": request_id,
            "root_gap": root_gap,
            "insight_text": summary,
            "created_at": _now_iso(),
        }
    ).execute()


def log_monitoring(client, level: str, event: str, detail: dict | None = None) -> None:
    """Best-effort monitoring write; swallow errors so it never breaks a flow."""
    try:
        _kernel(client, "kernel_monitoring").insert(
            {
                "level": level,
                "event": event,
                "detail": detail or {},
                "created_at": _now_iso(),
            }
        ).execute()
    except Exception:  # noqa: BLE001 - monitoring must never raise
        pass
