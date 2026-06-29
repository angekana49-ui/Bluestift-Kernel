"""Tests for the Bluestift Cognitive Kernel.

Covers:
  - pure algorithm units (BKT, forgetting, mindset, detector, calibration),
  - get_or_create_kc() against the in-memory fake Supabase (LLM mocked),
  - /health and /analyze routes (LLM + DB mocked).

Run: pytest -q
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import main
from core import bkt, calibration, detector, forgetting, mindset
from core.graph import build_graph
from services import analyze as analyze_pipeline
from services import db as db_module
from services import graph_builder
from services import kc_registry


# --------------------------------------------------------------------------- #
# Pure algorithm units
# --------------------------------------------------------------------------- #
def test_bkt_success_increases_mastery():
    k0 = 0.3
    k1 = bkt.update_bkt(k0, correct=True)
    assert k1 > k0
    assert 0.0 <= k1 <= 1.0


def test_bkt_failure_decreases_mastery():
    k0 = 0.6
    k1 = bkt.update_bkt(k0, correct=False)
    assert k1 < k0


def test_bkt_partial_credit_snapped_to_bins():
    assert bkt.snap_partial_credit(0.72) == 0.7
    assert bkt.snap_partial_credit(0.29) == 0.3


def test_bkt_uses_node_params_over_priors():
    params = bkt.get_bkt_params({"p_init": 0.5, "p_slip": 0.05})
    assert params["p_init"] == 0.5
    assert params["p_slip"] == 0.05
    # Missing values fall back to priors.
    assert params["p_guess"] == bkt.BKT_PRIORS["p_guess"]


def test_selective_update_gate():
    assert bkt.should_update_bkt(3, 0.5, 0.5) is True
    assert bkt.should_update_bkt(1, 0.9, 0.4) is True   # delta > 0.3
    assert bkt.should_update_bkt(1, 0.5, 0.5) is False
    assert bkt.should_update_bkt(1, 0.5, None, anomalous_pattern=True) is True


def test_mastery_dual_condition():
    assert bkt.is_mastered(0.96, 0.1, 0.8) is True
    assert bkt.is_mastered(0.96, 0.2, 0.8) is False   # slip too high
    assert bkt.is_mastered(0.9, 0.1, 0.8) is False    # mastery too low


def test_classify_status():
    assert bkt.classify_status(0.2) == "gap"
    assert bkt.classify_status(0.5) == "partial"
    assert bkt.classify_status(0.99, p_slip=0.1, partial_credit_avg=0.8) == "mastered"


def test_forgetting_decays_over_time():
    past = datetime.now(timezone.utc) - timedelta(days=30)
    k_eff = forgetting.compute_effective_mastery(0.8, "declarative", past)
    assert k_eff < 0.8
    # No decay for a fresh signal.
    now = datetime.now(timezone.utc)
    assert forgetting.compute_effective_mastery(0.8, "declarative", now) == pytest.approx(0.8, abs=1e-3)


def test_get_lambda_priority():
    node = {"type_kc": "procedural", "lambda_decay": 0.04, "interactions_count": 50}
    # Student personal lambda wins.
    assert forgetting.get_lambda(node, {"lambda_personal": 0.07}) == 0.07
    # Then KC empirical (enough interactions).
    assert forgetting.get_lambda(node, {}) == 0.04
    # Then prior when not enough data.
    assert forgetting.get_lambda({"type_kc": "procedural"}, {}) == 0.01


def test_mindset_bounds_and_classification():
    m = mindset.compute_mindset_score(1.0, 0.0, 0.0, 0.0)
    assert m >= 0.05
    m2 = mindset.compute_mindset_score(0.0, 1.0, 1.0, 1.0)
    assert m2 <= 0.95
    assert mindset.classify_mindset(0.8) == "growth"
    assert mindset.classify_mindset(0.3) == "fixed"


def test_calibrate_personal_lambda():
    # Mastery dropped from 0.8 to 0.5 over 20 days -> positive observed lambda.
    lam = calibration.calibrate_personal_lambda(0.8, 0.5, 20, 0.02)
    assert calibration.LAMBDA_MIN <= lam <= calibration.LAMBDA_MAX
    # Not enough signal -> returns current.
    assert calibration.calibrate_personal_lambda(0.8, 0.5, 0.5, 0.02) == 0.02


def test_empirical_kc_params_needs_min_students():
    few = [{"interactions_on_kc": 6, "mastery_score_raw": 0.3} for _ in range(5)]
    assert calibration.compute_empirical_kc_params(few) is None
    many = [{"interactions_on_kc": 6, "mastery_score_raw": 0.3} for _ in range(12)]
    params = calibration.compute_empirical_kc_params(many)
    assert params is not None
    assert params["empirical_difficulty"] == 1.0  # all struggling


# --------------------------------------------------------------------------- #
# Graph + root-cause detection
# --------------------------------------------------------------------------- #
def _sample_graph():
    nodes = [
        {"id": "a", "label": "fonctions_affines"},
        {"id": "b", "label": "fonctions_polynomiales"},
        {"id": "c", "label": "derivees"},
    ]
    edges = [
        {"prerequisite_id": "a", "concept_id": "b"},
        {"prerequisite_id": "b", "concept_id": "c"},
    ]
    return build_graph(nodes, edges)


def test_detect_root_cause_walks_to_deepest_gap():
    graph = _sample_graph()
    states = {"derivees": 0.2, "fonctions_polynomiales": 0.3, "fonctions_affines": 0.31}
    result = detector.detect_root_cause(graph, ["derivees"], states)
    assert result["root_gap"] == "fonctions_affines"
    assert result["detection_path"] == ["derivees", "fonctions_polynomiales", "fonctions_affines"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_recommended_path_starts_at_root():
    graph = _sample_graph()
    path = detector.recommended_path(graph, "fonctions_affines")
    assert path[0] == "fonctions_affines"


# --------------------------------------------------------------------------- #
# Graph builder — pure validation (no LLM)
# --------------------------------------------------------------------------- #
def test_enforce_dag_drops_cycle_creating_edges():
    # a->b->c->a is a cycle; the weakest edge closing it must be dropped.
    edges = [
        {"prerequisite": "a", "concept": "b", "weight": 1.0, "agreed_by": 2},
        {"prerequisite": "b", "concept": "c", "weight": 1.0, "agreed_by": 2},
        {"prerequisite": "c", "concept": "a", "weight": 0.6, "agreed_by": 1},
    ]
    kept, dropped = graph_builder.enforce_dag(edges)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0]["prerequisite"] == "c"  # the weakest edge was sacrificed


def test_dedup_vocabulary_merges_near_duplicates():
    vocab = {
        "fonctions_affines": {"label": "fonctions_affines"},
        "fonction_affine": {"label": "fonction_affine"},  # near-dup
        "derivees": {"label": "derivees"},
    }
    kept = graph_builder._dedup_vocabulary(vocab)
    # The two affine variants collapse to one; derivees stays.
    assert "fonctions_affines" in kept
    assert "derivees" in kept
    assert len(kept) == 2


# --------------------------------------------------------------------------- #
# get_or_create_kc with mocked LLM
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_get_or_create_kc_creates_and_reuses(fake_supabase, monkeypatch):
    async def fake_llm_call(prompt, max_tokens=1000):
        return (
            json.dumps(
                {
                    "type_kc": "procedural",
                    "lambda_decay": 0.01,
                    "description": "une derivee",
                    "prerequisites": ["fonctions_affines"],
                    "tau": 0.5,
                }
            ),
            "mock-model",
        )

    monkeypatch.setattr(kc_registry, "llm_call", fake_llm_call)

    kc = await kc_registry.get_or_create_kc("Derivees", "MATH", "lycee", fake_supabase)
    assert kc["label"] == "derivees"
    # A prerequisite node + an edge were created.
    nodes = fake_supabase.tables["kernel.concept_nodes"]
    labels = {n["label"] for n in nodes}
    assert "derivees" in labels and "fonctions_affines" in labels
    assert len(fake_supabase.tables["kernel.concept_edges"]) == 1

    # Second call reuses the existing node (no duplicate).
    kc2 = await kc_registry.get_or_create_kc("derivees", "MATH", "lycee", fake_supabase)
    assert kc2["id"] == kc["id"]
    assert sum(1 for n in nodes if n["label"] == "derivees") == 1


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    return TestClient(main.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["kernel"] == "bluestift-cognitive-kernel"


@pytest.mark.asyncio
async def test_analyze_pipeline_end_to_end(fake_supabase, monkeypatch):
    # Seed a minimal graph the DFS can walk.
    fake_supabase.seed(
        "kernel.concept_nodes",
        [
            {"id": "a", "label": "fonctions_affines", "subject": "MATH", "type_kc": "procedural"},
            {"id": "b", "label": "derivees", "subject": "MATH", "type_kc": "conceptual"},
        ],
    )
    fake_supabase.seed(
        "kernel.concept_edges",
        [{"id": "e1", "prerequisite_id": "a", "concept_id": "b"}],
    )

    # Mock the LLM: first call = extraction, later = summary.
    calls = {"n": 0}

    async def fake_llm_call(prompt, max_tokens=1000):
        calls["n"] += 1
        if "kcs_mentioned" in prompt:
            return (
                json.dumps(
                    {
                        "kcs_mentioned": [
                            {"label": "derivees", "subject": "MATH", "level": "lycee"},
                            {"label": "fonctions_affines", "subject": "MATH", "level": "cycle4"},
                        ],
                        "attempts": [
                            {"kc_label": "derivees", "outcome": "failure", "partial_credit": 0.2,
                             "is_assisted": False, "response_time_estimate": "slow"},
                            {"kc_label": "fonctions_affines", "outcome": "partial", "partial_credit": 0.3,
                             "is_assisted": False, "response_time_estimate": "normal"},
                        ],
                        "blocage_type": "conceptual",
                        "langue_interaction": "fr",
                    }
                ),
                "mock-model",
            )
        return ("Tu bloques sur les derivees car les fonctions affines ne sont pas solides.", "mock-model")

    monkeypatch.setattr(analyze_pipeline, "llm_call", fake_llm_call)
    monkeypatch.setattr(kc_registry, "llm_call", fake_llm_call)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    payload = {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "conversation_history": [
            {"role": "user", "content": "Je comprends pas les derivees"},
            {"role": "assistant", "content": "Rappelle-moi ce qu'est une fonction affine"},
            {"role": "user", "content": "C'est... f(x) = ax ?"},
        ],
        "subject": "MATH",
        "level": "lycee",
        "trigger": "post_conversation",
    }

    client = TestClient(main.app)
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["user_id"] == payload["user_id"]
    assert body["kernel_version"] == main.KERNEL_VERSION
    assert "derivees" in body["mastery_map"]
    assert body["root_gap"] == "fonctions_affines"
    assert body["detection_path"][0] == "derivees"
    assert body["summary"]
    # The request was logged.
    assert len(fake_supabase.tables["kernel.kernel_requests"]) == 1
    assert len(fake_supabase.tables["kernel.kernel_outputs"]) == 1


def test_load_profile(fake_supabase, monkeypatch):
    fake_supabase.seed(
        "kernel.concept_nodes",
        [{"id": "a", "label": "fractions", "subject": "MATH", "type_kc": "procedural"}],
    )
    fake_supabase.seed(
        "kernel.student_concept_state",
        [
            {
                "id": "s1",
                "user_id": "u1",
                "concept_id": "a",
                "mastery_score_raw": 0.6,
                "v_score": 0.5,
                "p_score": 0.5,
                "partial_credit_avg": 0.6,
                "last_strong_signal_at": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
            }
        ],
    )
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    client = TestClient(main.app)
    resp = client.post("/load_profile", json={"user_id": "u1"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["concept_states"]) == 1
    cs = body["concept_states"][0]
    assert cs["label"] == "fractions"
    assert cs["k_effective"] <= cs["k_raw"]  # decay applied


def test_seed_kcs(fake_supabase, monkeypatch):
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    client = TestClient(main.app)

    resp = client.post("/seed_kcs")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["seeded"] is True
    assert body["nodes_inserted"] == 15
    assert body["edges_inserted"] == 15

    # Second call is a no-op because the table is now populated.
    resp2 = client.post("/seed_kcs")
    assert resp2.json()["seeded"] is False
