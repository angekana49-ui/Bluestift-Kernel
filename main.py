"""Bluestift Cognitive Kernel — FastAPI application.

The brain of Bluestift, fully decoupled from any UI. RAYA (Next.js) calls it
over HTTP. It knows nothing about React or client-side auth.

Routes:
    GET  /health
    POST /analyze
    POST /load_profile
    POST /update_concept_state
    POST /seed_kcs
"""
from __future__ import annotations

import hmac
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import jwt
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from core import bkt, forgetting  # noqa: E402
from core.calibration import compute_empirical_kc_params  # noqa: E402
from core.mindset import classify_mindset  # noqa: E402
from models.schemas import (  # noqa: E402
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    LoadProfileRequest,
    LoadProfileResponse,
    SeedResponse,
    UpdateConceptStateRequest,
    UpdateConceptStateResponse,
)
from services import analyze as analyze_pipeline  # noqa: E402
from services import db  # noqa: E402
from services.analyze import _persistence, _slip_estimate, _velocity  # noqa: E402
from services.kc_registry import get_or_create_kc  # noqa: E402

KERNEL_VERSION = os.getenv("KERNEL_VERSION", "1.0.0")

DEFAULT_CORS = [
    "https://raya.thebluestift.com",
    "https://schools.thebluestift.com",
    "http://localhost:3000",
]
_cors_env = os.getenv("CORS_ORIGINS")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",")] if _cors_env else DEFAULT_CORS

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log a clear alert if the Kernel boots without full DB access."""
    try:
        client = db.get_client()
        status = db.check_db_access(client)
        if not (status["read_ok"] and status["write_ok"]):
            db.log_monitoring(client, "error", "startup_db_degraded", status)
    except Exception:  # noqa: BLE001 - never block startup (e.g. tests without creds)
        pass
    yield


app = FastAPI(title="Bluestift Cognitive Kernel", version=KERNEL_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Auth — two tiers
# --------------------------------------------------------------------------- #
# The Kernel accepts two kinds of caller:
#
#   service — presents KERNEL_API_SECRET. A trusted backend (the RAYA app) acting
#             on behalf of many students. May touch any user_id, and is the only
#             tier allowed to seed the graph. Required for background work that
#             outlives a student's request.
#
#   user    — presents a Supabase access token. Scoped to exactly one student:
#             the Kernel checks the token's `sub` against the user_id in the body
#             and refuses anything else (403).
#
# The point of the second tier is blast radius. The service secret is a skeleton
# key to every student's cognitive profile — minors included, in a product whose
# COPPA/GDPR/FERPA pages promise otherwise. Every call that already knows which
# student it is about should carry that student's token instead, so the skeleton
# key travels only where it is genuinely needed.
#
# This bounds *who may ask about whom*. It does not limit what the Kernel can
# model: any student, any subject, any level, KCs created on the fly.
#
# When KERNEL_API_SECRET is unset, auth is disabled entirely and every caller is
# treated as a service (local development) — unchanged from before.
@dataclass(frozen=True)
class Principal:
    kind: str                  # "service" | "user"
    user_id: str | None = None


SERVICE = Principal("service")


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization.strip()


def _verify_supabase_jwt(token: str) -> str | None:
    """Return the student id in a valid Supabase access token, else None.

    HS256 against SUPABASE_JWT_SECRET (the project's JWT secret). Projects using
    asymmetric signing keys would need a JWKS fetch instead; unset the variable
    and the user tier simply stays unavailable.
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if not jwt_secret or token.count(".") != 2:
        return None
    try:
        claims = jwt.decode(
            token, jwt_secret, algorithms=["HS256"], audience="authenticated"
        )
    except Exception:  # noqa: BLE001 - expired, forged, wrong audience: all "not a user"
        return None
    sub = claims.get("sub")
    return str(sub) if sub else None


async def authenticate(
    authorization: str | None = Header(default=None),
    x_kernel_secret: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> Principal:
    """Resolve the caller. 401 when it presents nothing we recognise."""
    secret = os.getenv("KERNEL_API_SECRET")
    bearer = _bearer(authorization)
    # The secret is accepted via any common convention, so it works with whatever
    # the client already sends.
    provided = x_kernel_secret or x_api_key or bearer

    if not secret:
        return SERVICE  # auth disabled

    # Service first: an exact match keeps the existing callers on their old path.
    if provided and hmac.compare_digest(provided, secret):
        return SERVICE

    if bearer:
        user_id = _verify_supabase_jwt(bearer)
        if user_id:
            return Principal("user", user_id)

    raise HTTPException(status_code=401, detail="Unauthorized")


def authorize_for(principal: Principal, user_id: str) -> None:
    """A student's token may only ever reach that student's own profile."""
    if principal.kind == "user" and principal.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")


def require_service(principal: Principal) -> None:
    """Graph-wide operations are never a student's to perform."""
    if principal.kind != "service":
        raise HTTPException(status_code=403, detail="Forbidden")


# --------------------------------------------------------------------------- #
# /health  (always open — Railway health check + app connectivity probe)
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=KERNEL_VERSION)


# --------------------------------------------------------------------------- #
# /ready  (deep health — verifies kernel-schema read + write access)
# --------------------------------------------------------------------------- #
# Surfaces the recurring shared-DB regression (exposed schema / grants reset by
# an app setup) proactively: returns 503 when the Kernel can't reach its schema.
@app.get("/ready")
async def ready() -> JSONResponse:
    try:
        status = db.check_db_access(db.get_client())
    except Exception as e:  # noqa: BLE001 - creds missing / client build failed
        status = {"read_ok": False, "write_ok": False, "detail": str(e)[:200]}
    ok = status["read_ok"] and status["write_ok"]
    status["status"] = "ok" if ok else "degraded"
    status["version"] = KERNEL_VERSION
    return JSONResponse(status_code=200 if ok else 503, content=status)


# --------------------------------------------------------------------------- #
# /analyze — main route
# --------------------------------------------------------------------------- #
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    req: AnalyzeRequest, principal: Principal = Depends(authenticate)
) -> AnalyzeResponse:
    authorize_for(principal, req.user_id)
    request_id = str(uuid.uuid4())
    payload = req.model_dump(mode="json")
    client = db.get_client()

    # Rule: every /analyze is logged, even if it errors out afterwards.
    try:
        db.log_kernel_request(client, request_id, req.user_id, payload)
    except Exception as e:  # noqa: BLE001 - logging must not block analysis
        db.log_monitoring(client, "warn", "request_log_failed", {"error": str(e)})

    try:
        output = await analyze_pipeline.run_analysis(client, request_id, payload)
    except Exception as e:  # noqa: BLE001
        # Log the full detail internally; return a generic message (no internals
        # like schema/constraint names leak to the caller). request_id correlates.
        db.log_monitoring(client, "error", "analyze_failed", {"request_id": request_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Analysis failed (request_id={request_id})") from e

    return AnalyzeResponse(kernel_version=KERNEL_VERSION, **output)


# --------------------------------------------------------------------------- #
# /load_profile
# --------------------------------------------------------------------------- #
@app.post("/load_profile", response_model=LoadProfileResponse)
async def load_profile(
    req: LoadProfileRequest, principal: Principal = Depends(authenticate)
) -> LoadProfileResponse:
    authorize_for(principal, req.user_id)
    client = db.get_client()

    nodes = {n["id"]: n for n in db.load_concept_nodes(client)}
    states = db.load_student_concept_states(client, req.user_id)

    concept_states = []
    last_update = None
    for s in states:
        node = nodes.get(s["concept_id"], {})
        k_raw = s.get("mastery_score_raw") or 0.0
        last_at = s.get("last_strong_signal_at")
        lam = forgetting.get_lambda(node, s)
        k_eff = forgetting.compute_effective_mastery(
            k_raw, node.get("type_kc", "conceptual"), last_at, lambda_override=lam
        )
        p_slip = s.get("p_slip_personal") or bkt.get_bkt_params(node)["p_slip"]
        pc_avg = s.get("partial_credit_avg") or 0.5
        status = bkt.classify_status(k_eff, p_slip, pc_avg)

        if last_at and (last_update is None or str(last_at) > str(last_update)):
            last_update = last_at

        concept_states.append(
            {
                "concept_id": s["concept_id"],
                "label": node.get("label", "unknown"),
                "k_raw": round(k_raw, 4),
                "k_effective": round(k_eff, 4),
                "v_score": s.get("v_score") or 0.5,
                "p_score": s.get("p_score") or 0.5,
                "status": status,
                "last_interaction_at": last_at,
            }
        )

    mindset_row = db.load_mindset(client, req.user_id)
    mindset = None
    if mindset_row:
        m_score = mindset_row.get("m_score") or 0.5
        mindset = {
            "m_score": m_score,
            "detected_mindset": mindset_row.get("detected_mindset") or classify_mindset(m_score),
        }

    return LoadProfileResponse(
        user_id=req.user_id,
        concept_states=concept_states,
        mindset=mindset,
        last_kernel_update=last_update,
    )


# --------------------------------------------------------------------------- #
# /update_concept_state — called by RAYA on a strong signal
# --------------------------------------------------------------------------- #
@app.post("/update_concept_state", response_model=UpdateConceptStateResponse)
async def update_concept_state(
    req: UpdateConceptStateRequest,
    background: BackgroundTasks,
    principal: Principal = Depends(authenticate),
) -> UpdateConceptStateResponse:
    authorize_for(principal, req.user_id)
    client = db.get_client()

    # Resolve the KC: by UUID when the caller holds one, otherwise by label —
    # canonicalized and created on the fly, the same path /analyze takes.
    if req.concept_id:
        node = next(
            (n for n in db.load_concept_nodes(client) if n["id"] == req.concept_id),
            None,
        )
        if node is None:
            raise HTTPException(status_code=404, detail="concept_id not found")
    else:
        node = await get_or_create_kc(
            label=req.concept_label,
            subject=req.subject,
            level=req.level,
            supabase_client=client,
        )
    concept_id = node["id"]

    state = db.load_student_concept_state(client, req.user_id, concept_id)
    params = bkt.get_bkt_params(node)
    k_raw_prev = (state.get("mastery_score_raw") if state else None) or params["p_init"]

    # Assisted attempts are capped at 0.9 — full credit can't come with help.
    pc = min(req.partial_credit_score, 0.9) if req.is_assisted else req.partial_credit_score

    k_raw = bkt.update_bkt(k_raw_prev, correct=(pc >= 0.5), partial_credit=pc, params=params)

    now = datetime.now(timezone.utc).isoformat()
    prev_count = (state.get("interactions_on_kc") if state else 0) or 0
    prev_avg = state.get("partial_credit_avg") if state else None
    new_avg = pc if prev_avg is None else (prev_avg * prev_count + pc) / (prev_count + 1)
    prev_struggle = (state.get("struggle_index") if state else 0) or 0
    interactions = prev_count + 1
    struggle_index = prev_struggle + (1 if pc < 0.5 else 0)

    # Cognitive vector: V = learning rate p(T); P = (1 - personal slip) modulated by M.
    m_row = db.load_mindset(client, req.user_id)
    m_score = (m_row.get("m_score") if m_row else None) or 0.5
    p_slip_personal = _slip_estimate(state.get("p_slip_personal") if state else None, k_raw_prev, pc < 0.5)
    v_score = _velocity(state.get("v_score") if state else None, k_raw_prev, k_raw)
    p_score = _persistence(p_slip_personal, m_score)

    db.upsert_student_concept_state(
        client,
        {
            "user_id": req.user_id,
            "concept_id": concept_id,
            "mastery_score_raw": round(k_raw, 4),
            "mastery_score_effective": round(k_raw, 4),
            "v_score": v_score,
            "p_score": p_score,
            "p_slip_personal": p_slip_personal,
            "partial_credit_avg": round(new_avg, 4),
            "struggle_index": struggle_index,
            "interactions_on_kc": interactions,
            "last_strong_signal_at": now,
        },
    )
    db.log_trajectory(client, req.user_id, concept_id, k_raw, k_raw)

    k_eff = forgetting.compute_effective_mastery(
        k_raw, node.get("type_kc", "conceptual"), now,
        lambda_override=forgetting.get_lambda(node, state),
    )
    status = bkt.classify_status(k_eff, params["p_slip"], new_avg)

    # Fire-and-forget empirical recalibration for this KC in the background.
    background.add_task(_recalibrate_kc, concept_id)

    return UpdateConceptStateResponse(
        user_id=req.user_id,
        concept_id=concept_id,
        label=node.get("label", ""),
        k_raw=round(k_raw, 4),
        k_effective=round(k_eff, 4),
        p_score=p_score,
        status=status,
        updated=True,
    )


def _recalibrate_kc(concept_id: str) -> None:
    """Background: recompute a KC's empirical params from all student states."""
    try:
        client = db.get_client()
        states = db.load_states_for_concept(client, concept_id)
        params = compute_empirical_kc_params(states)
        if params:
            params["last_calibration_at"] = datetime.now(timezone.utc).isoformat()
            db.update_concept_node(client, concept_id, params)
    except Exception as e:  # noqa: BLE001 - background work must not raise
        try:
            db.log_monitoring(db.get_client(), "warn", "recalibration_failed", {"concept_id": concept_id, "error": str(e)})
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# /seed_kcs
# --------------------------------------------------------------------------- #
@app.post("/seed_kcs", response_model=SeedResponse)
async def seed_kcs(principal: Principal = Depends(authenticate)) -> SeedResponse:
    # Seeding rewrites the shared graph for everyone — never a student's call.
    require_service(principal)

    from seed.math_kcs import seed_math_kcs

    client = db.get_client()
    if db.count_concept_nodes(client) > 0:
        return SeedResponse(
            seeded=False,
            nodes_inserted=0,
            edges_inserted=0,
            message="concept_nodes already populated; seeding skipped.",
        )

    nodes, edges = seed_math_kcs(client)
    return SeedResponse(
        seeded=True,
        nodes_inserted=nodes,
        edges_inserted=edges,
        message="Starter Math KCs seeded.",
    )
