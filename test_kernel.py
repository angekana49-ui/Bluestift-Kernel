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
from core import anomaly, bkt, calibration, curriculum, detector, forgetting, mindset
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


def test_dfs_descends_through_unknown_prerequisites():
    # derivees is failing; the intermediate node was never practised (unknown)
    # and the deep root is also failing. The search must bridge the unknown.
    graph = _sample_graph()  # fonctions_affines -> fonctions_polynomiales -> derivees
    states = {"derivees": 0.2, "fonctions_affines": 0.25}  # polynomiales unknown
    known = {"derivees", "fonctions_affines"}
    result = detector.detect_root_cause(graph, ["derivees"], states, known=known)
    assert result["root_gap"] == "fonctions_affines"
    assert result["detection_path"] == ["derivees", "fonctions_polynomiales", "fonctions_affines"]


def test_dfs_stops_at_mastered_prerequisite():
    # A known-mastered prerequisite is a barrier: the search must not pass it.
    graph = _sample_graph()
    states = {"derivees": 0.2, "fonctions_polynomiales": 0.9, "fonctions_affines": 0.1}
    known = {"derivees", "fonctions_polynomiales", "fonctions_affines"}
    result = detector.detect_root_cause(graph, ["derivees"], states, known=known)
    assert result["detection_path"] == ["derivees"]  # blocked by mastered prereq


def test_root_selection_prefers_convergence_over_longest_chain():
    # variable underlies BOTH failing branches; derivee's chain is longer, but
    # variable is the convergence point and must win.
    import networkx as nx

    g = nx.DiGraph()
    g.add_edge("variable", "fonction_affine")     # prereq -> concept
    g.add_edge("fonction_affine", "derivee")
    g.add_edge("variable", "equation_lineaire")
    states = {"derivee": 0.2, "equation_lineaire": 0.2, "variable": 0.2}  # fonction_affine unknown
    known = {"derivee", "equation_lineaire", "variable"}

    result = detector.detect_root_cause(g, ["derivee", "equation_lineaire"], states, known=known)
    assert result["root_gap"] == "variable"            # convergence wins
    assert result["confidence"] > 0.6                  # 2/2 failing converge


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
def test_selective_update_gate_bootstraps_then_gates():
    attempt = {"outcome": "failure", "partial_credit": 0.2}
    # First contact (no prior state) always commits.
    assert analyze_pipeline._should_commit(attempt, None, 1) is True
    # Small change vs stored average, single attempt -> no commit.
    prev = {"partial_credit_avg": 0.25, "mastery_score_raw": 0.3}
    assert analyze_pipeline._should_commit(attempt, prev, 1) is False
    # Large partial-credit shift -> commit.
    prev_far = {"partial_credit_avg": 0.9, "mastery_score_raw": 0.9}
    assert analyze_pipeline._should_commit(attempt, prev_far, 1) is True


def test_velocity_is_learning_rate():
    # V = p(T): fraction of the remaining mastery gap closed this trial.
    fast = analyze_pipeline._velocity(None, 0.2, 0.6)   # closed half the gap
    slow = analyze_pipeline._velocity(None, 0.2, 0.25)  # barely moved
    assert fast > slow
    assert 0.05 <= analyze_pipeline._velocity(None, 0.0, 1.0) <= 0.95


def test_anomaly_false_mastery():
    # High mastery + high slip -> false mastery.
    assert anomaly.detect_false_mastery("frac", 0.96, 0.25) is not None
    assert anomaly.detect_false_mastery("frac", 0.96, 0.05) is None  # solid
    assert anomaly.detect_false_mastery("frac", 0.5, 0.25) is None   # not mastered


def test_anomaly_passive_dependency():
    assisted = [
        {"is_assisted": True, "outcome": "success", "response_time_estimate": "fast"},
        {"is_assisted": True, "outcome": "success", "response_time_estimate": "fast"},
    ]
    assert anomaly.detect_passive_dependency(assisted) is not None
    autonomous = [
        {"is_assisted": False, "outcome": "failure", "response_time_estimate": "slow"},
        {"is_assisted": False, "outcome": "partial", "response_time_estimate": "normal"},
    ]
    assert anomaly.detect_passive_dependency(autonomous) is None


def test_anomaly_re_emergence_and_orchestrator():
    # Fails the KC while every prerequisite looks mastered -> re-emergence.
    assert anomaly.detect_re_emergence("derivee", 0.2, [0.9, 0.85]) is not None
    assert anomaly.detect_re_emergence("derivee", 0.2, [0.3]) is None  # weak prereq
    alerts = anomaly.detect_anomalies(
        kc_records=[{"label": "x", "k_effective": 0.96, "p_slip": 0.3, "prereq_masteries": []}],
        attempts=[],
        blocage_type="none",
        m_score=0.2,  # fixed mindset
    )
    types = {a["alert_type"] for a in alerts}
    assert "false_mastery" in types and "fixed_mindset" in types


def test_volatility_and_inconsistency_metrics():
    climbing = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    # A monotonic climb spends all its movement on progress: no churn.
    assert anomaly.temporal_inconsistency(climbing) == 0.0
    # Oscillating around the same level: nearly all churn, no net progress.
    swinging = [0.5, 0.9, 0.2, 0.85, 0.25, 0.5]
    assert anomaly.temporal_inconsistency(swinging) > 0.9
    # Volatility is about step size, not direction — the climb is the calmer one.
    assert anomaly.volatility_score(swinging) > anomaly.volatility_score(climbing)
    # Degenerate inputs are stable, not inconsistent.
    assert anomaly.temporal_inconsistency([0.4]) == 0.0
    assert anomaly.temporal_inconsistency([0.4, 0.4, 0.4]) == 0.0


def test_anomaly_inconsistency_high():
    swinging = [0.5, 0.9, 0.2, 0.85, 0.25, 0.9, 0.3]
    alert = anomaly.detect_inconsistency_high("derivee", swinging)
    assert alert is not None
    assert alert["alert_details"]["inconsistency_rate"] > anomaly.INCONSISTENCY_HIGH
    assert alert["alert_details"]["interactions_count"] == len(swinging)

    # A steady climb never fires, however long.
    assert anomaly.detect_inconsistency_high("derivee", [0.1, 0.3, 0.4, 0.6, 0.8, 0.95]) is None
    # Too little history to judge stability at all.
    assert anomaly.detect_inconsistency_high("derivee", [0.5, 0.9, 0.2]) is None


def test_anomaly_ood_distribution():
    # Struggling on three KCs the local population finds easy -> the calibrated
    # parameters don't describe this student.
    struggling = [
        {"label": "a", "k_effective": 0.2, "p_slip": 0.1, "population_difficulty": 0.1},
        {"label": "b", "k_effective": 0.3, "p_slip": 0.1, "population_difficulty": 0.2},
        {"label": "c", "k_effective": 0.1, "p_slip": 0.1, "population_difficulty": 0.1},
    ]
    alert = anomaly.detect_ood_distribution(struggling)
    assert alert is not None
    assert alert["alert_details"]["direction"] == "below_population"
    assert alert["alert_severity"] == "high"  # the silent-failure direction

    # A student tracking the local baseline is not out of distribution.
    typical = [
        {"label": "a", "k_effective": 0.2, "p_slip": 0.1, "population_difficulty": 0.8},
        {"label": "b", "k_effective": 0.9, "p_slip": 0.1, "population_difficulty": 0.2},
        {"label": "c", "k_effective": 0.9, "p_slip": 0.1, "population_difficulty": 0.1},
    ]
    assert anomaly.detect_ood_distribution(typical) is None

    # Uncalibrated KCs carry no baseline, so there is nothing to diverge from.
    assert anomaly.detect_ood_distribution(
        [{"label": "a", "k_effective": 0.1, "p_slip": 0.1} for _ in range(5)]
    ) is None


def test_population_baseline_ignores_the_uncalibrated_default():
    # A fresh KC carries a neutral 0.5 placeholder — not a population baseline.
    assert analyze_pipeline._population_baseline({"empirical_difficulty": 0.5}) is None
    calibrated = {"empirical_difficulty": 0.15, "last_calibration_at": "2026-07-01T00:00:00Z"}
    assert analyze_pipeline._population_baseline(calibrated) == 0.15


def test_selective_update_gate_fires_on_an_unstable_history():
    state = {"partial_credit_avg": 0.5, "mastery_score_raw": 0.5}
    attempt = {"outcome": "partial", "partial_credit": 0.5}  # no shift, one attempt
    # Nothing notable in the attempt itself: the gate holds.
    assert analyze_pipeline._should_commit(attempt, state, 1, []) is False
    # Same attempt, but the KC's estimate has been oscillating -> commit, because
    # holding back would leave an unreliable value on the books.
    swinging = [0.5, 0.9, 0.2, 0.85, 0.25, 0.9, 0.3]
    assert analyze_pipeline._should_commit(attempt, state, 1, swinging) is True


def test_group_trajectories_keeps_order_per_concept():
    rows = [
        {"concept_id": "a", "k_raw": 0.1},
        {"concept_id": "b", "k_raw": 0.7},
        {"concept_id": "a", "k_raw": 0.4},
        {"concept_id": "a", "k_raw": None},  # skipped, not a data point
    ]
    assert analyze_pipeline._group_trajectories(rows) == {"a": [0.1, 0.4], "b": [0.7]}


# --------------------------------------------------------------------------- #
# School -> AI -> Student channel
# --------------------------------------------------------------------------- #
def test_parse_layers_merges_and_validates():
    rows = [
        {"layer_type": "curriculum", "payload": {"concepts": ["Fractions", "notion de variable"]}},
        {"layer_type": "curriculum", "concept_ids": ["derivation_fonction"]},  # legacy row
        {"layer_type": "kc_priorities", "payload": {"weights": {"Fractions": 3, "x": "nope", "y": 99}}},
        {"layer_type": "objectives", "payload": {"targets": [
            {"concept": "fractions", "mastery": 0.8, "due_at": "2026-12-15T00:00:00Z"},
            {"nothing": "usable"},
        ]}},
        {"layer_type": "custom_rules", "payload": {"rules": ["Toujours partir d'un exemple concret."]}},
    ]
    layers = curriculum.parse_layers(rows, school_id="s1")

    # Labels are canonicalized, so a school can write prose.
    assert layers.concepts == {"fractions", "notion_de_variable", "derivation_fonction"}
    assert layers.weights["fractions"] == 3.0
    assert "x" not in layers.weights                      # unparseable weight dropped
    assert layers.weights["y"] == curriculum.MAX_PRIORITY  # 99 clamped, not obeyed
    assert len(layers.objectives) == 1                     # the unusable target is skipped
    assert layers.rules == ["Toujours partir d'un exemple concret."]
    assert set(layers.applied()) == {"curriculum", "kc_priorities", "objectives", "custom_rules"}


def test_parse_layers_survives_garbage():
    layers = curriculum.parse_layers(
        [
            {"layer_type": "kc_priorities", "payload": "not a dict"},
            {"layer_type": "objectives", "payload": {"targets": "nope"}},
            {"layer_type": "unknown_type", "payload": {"weights": {"a": 2}}},
            {},
        ]
    )
    assert layers.is_empty


def test_school_priorities_reorder_but_never_reach_past_prerequisites():
    import networkx as nx

    graph = nx.DiGraph()
    # root -> both branches are valid next steps; alphabetically "algebre" wins.
    graph.add_edge("racine", "algebre")
    graph.add_edge("racine", "trigonometrie")

    assert detector.recommended_path(graph, "racine")[1] == "algebre"
    # The school prioritizes trigonometry: it now comes first among the
    # legitimate next steps.
    weighted = detector.recommended_path(graph, "racine", priorities={"trigonometrie": 3.0})
    assert weighted[1] == "trigonometrie"
    # The walk still starts at the root gap — priorities can't skip foundations.
    assert weighted[0] == "racine"


def test_objective_report_statuses():
    now = datetime(2026, 8, 15, tzinfo=timezone.utc)
    objectives = [
        {"concept": "fractions", "mastery": 0.8, "due_at": "2026-12-01T00:00:00Z"},
        {"concept": "derivees", "mastery": 0.8, "due_at": "2026-08-20T00:00:00Z"},
        {"concept": "limites", "mastery": 0.8, "due_at": "2026-07-01T00:00:00Z"},
        {"concept": "integrales", "mastery": 0.8, "due_at": None},
        {"concept": "jamais_vu", "mastery": 0.8, "due_at": "2026-09-01T00:00:00Z"},
    ]
    mastery = {"fractions": 0.9, "derivees": 0.4, "limites": 0.3, "integrales": 0.2}

    report = {r["concept"]: r["status"] for r in curriculum.objective_report(objectives, mastery, now=now)}
    assert report["fractions"] == "met"
    assert report["derivees"] == "at_risk"   # under target, due in 5 days
    assert report["limites"] == "overdue"    # deadline already passed
    assert report["integrales"] == "pending"  # no deadline set
    # No evidence yet is reported as unknown, not silently counted as failure.
    assert report["jamais_vu"] == "unknown"


def test_load_curriculum_layers_matches_wildcard_levels(fake_supabase):
    fake_supabase.seed(
        "schools.school_curriculum_layers",
        [
            {"id": "l1", "school_id": "s1", "subject": "MATH", "level": "lycee",
             "is_active": True, "layer_type": "curriculum", "payload": {"concepts": ["a"]}},
            {"id": "l2", "school_id": "s1", "subject": "MATH", "level": "*",
             "is_active": True, "layer_type": "kc_priorities", "payload": {"weights": {"a": 2}}},
            {"id": "l3", "school_id": "s1", "subject": "MATH", "level": "college",
             "is_active": True, "layer_type": "curriculum", "payload": {"concepts": ["b"]}},
            {"id": "l4", "school_id": "s1", "subject": "MATH", "level": "lycee",
             "is_active": False, "layer_type": "curriculum", "payload": {"concepts": ["c"]}},
        ],
    )
    rows = db_module.load_curriculum_layers(fake_supabase, "s1", "MATH", "lycee")
    # This level plus the subject-wide layer; not another level's, not an inactive one.
    assert {r["layer_type"] for r in rows} == {"curriculum", "kc_priorities"}
    assert len(rows) == 2


def test_no_school_means_no_curriculum_block(fake_supabase):
    # An independent learner: no student_identities row at all.
    assert db_module.load_school_id(fake_supabase, "u1") is None
    layers = analyze_pipeline._load_curriculum_layers(fake_supabase, "u1", "MATH", "lycee")
    assert layers.is_empty


def test_slip_and_persistence():
    # A failure while mastery looked solid raises personal slip.
    slip_hi = analyze_pipeline._slip_estimate(0.1, k_before=0.9, outcome_failure=True)
    slip_lo = analyze_pipeline._slip_estimate(0.1, k_before=0.3, outcome_failure=True)
    assert slip_hi > slip_lo
    # P = (1 - slip) modulated by mindset: growth mindset lifts persistence.
    p_growth = analyze_pipeline._persistence(0.1, m_score=0.9)
    p_fixed = analyze_pipeline._persistence(0.1, m_score=0.1)
    assert p_growth > p_fixed
    assert 0.05 <= p_growth <= 0.95


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


def test_ready_ok_when_db_reachable(client, fake_supabase, monkeypatch):
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["read_ok"] and body["write_ok"]


def test_load_recent_trajectories_is_chronological(fake_supabase):
    fake_supabase.seed(
        "kernel.learning_trajectories",
        [
            {"id": "t2", "user_id": "u1", "concept_id": "a", "k_raw": 0.6, "snapshot_at": "2026-07-02T00:00:00Z"},
            {"id": "t1", "user_id": "u1", "concept_id": "a", "k_raw": 0.3, "snapshot_at": "2026-07-01T00:00:00Z"},
            {"id": "t3", "user_id": "u2", "concept_id": "a", "k_raw": 0.9, "snapshot_at": "2026-07-03T00:00:00Z"},
        ],
    )
    rows = db_module.load_recent_trajectories(fake_supabase, "u1")
    # Only this student's rows, oldest first — the series is read in order.
    assert [r["k_raw"] for r in rows] == [0.3, 0.6]


def test_log_alert_promotes_the_stability_metrics(fake_supabase):
    alert = anomaly.detect_inconsistency_high("derivee", [0.5, 0.9, 0.2, 0.85, 0.25, 0.9, 0.3])
    db_module.log_alert(fake_supabase, "u1", alert, concept_id="c1")

    row = fake_supabase.tables["kernel.kernel_monitoring"][0]
    assert row["alert_type"] == "inconsistency_high"
    # Migration 008 gave these their own columns; a dashboard shouldn't have to
    # dig them out of the details JSON.
    assert row["inconsistency_rate"] == alert["alert_details"]["inconsistency_rate"]
    assert row["volatility_score"] == alert["alert_details"]["volatility_score"]
    assert row["interactions_count"] == 7

    # An alert without those metrics doesn't invent the columns.
    db_module.log_alert(fake_supabase, "u1", anomaly.detect_fixed_mindset(0.1))
    assert "inconsistency_rate" not in fake_supabase.tables["kernel.kernel_monitoring"][1]


def test_check_db_access_reports_read_failure(monkeypatch):
    class _Boom:
        def schema(self, *_):
            raise RuntimeError("Invalid schema: kernel")

    status = db_module.check_db_access(_Boom())
    assert status["read_ok"] is False
    assert "read:" in (status["detail"] or "")


def test_auth_enforced_when_secret_set(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)  # hermetic
    # /health stays open.
    assert client.get("/health").status_code == 200
    # Protected route without the secret -> 401.
    assert client.post("/load_profile", json={"user_id": "u1"}).status_code == 401
    # Wrong secret -> 401.
    r = client.post("/load_profile", json={"user_id": "u1"}, headers={"X-Kernel-Secret": "nope"})
    assert r.status_code == 401
    # Correct secret via each accepted convention -> passes the gate (200).
    for headers in (
        {"X-Kernel-Secret": "s3cr3t"},
        {"X-API-Key": "s3cr3t"},
        {"Authorization": "Bearer s3cr3t"},
    ):
        assert client.post("/load_profile", json={"user_id": "u1"}, headers=headers).status_code == 200


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
    # Default mode commits the attempts it extracted.
    assert len(fake_supabase.tables["kernel.student_concept_state"]) == 2

    # Diagnose-only: same diagnosis, no state written. A caller that already sent
    # its graded attempts to /update_concept_state uses this so the evidence
    # isn't counted twice.
    fake_supabase.tables["kernel.student_concept_state"].clear()
    resp2 = client.post("/analyze", json={**payload, "commit_state": False})
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["root_gap"] == "fonctions_affines"
    assert fake_supabase.tables["kernel.student_concept_state"] == []

    # An independent learner gets no curriculum block at all.
    assert resp2.json().get("curriculum") is None

    # Same student, now attached to a school that has set layers.
    fake_supabase.seed(
        "schools.student_identities",
        [{"id": "si1", "user_id": payload["user_id"], "school_id": "school-1"}],
    )
    fake_supabase.seed(
        "schools.school_curriculum_layers",
        [
            {"id": "l1", "school_id": "school-1", "subject": "MATH", "level": "*",
             "is_active": True, "layer_type": "curriculum",
             "payload": {"concepts": ["fonctions_affines", "derivees"]}},
            {"id": "l2", "school_id": "school-1", "subject": "MATH", "level": "*",
             "is_active": True, "layer_type": "objectives",
             "payload": {"targets": [{"concept": "derivees", "mastery": 0.8,
                                      "due_at": "2026-01-01T00:00:00Z"}]}},
        ],
    )
    resp3 = client.post("/analyze", json={**payload, "commit_state": False})
    assert resp3.status_code == 200, resp3.text
    school_block = resp3.json()["curriculum"]
    assert school_block["school_id"] == "school-1"
    assert set(school_block["layers_applied"]) == {"curriculum", "objectives"}
    assert school_block["root_gap_in_program"] is True
    # The deadline is long past and the student is nowhere near the target.
    assert school_block["objectives"][0]["status"] == "overdue"


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


def test_update_concept_state_by_id(fake_supabase, monkeypatch):
    fake_supabase.seed(
        "kernel.concept_nodes",
        [{"id": "a", "label": "fractions", "subject": "MATH", "type_kc": "procedural"}],
    )
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    client = TestClient(main.app)
    resp = client.post(
        "/update_concept_state",
        json={"user_id": "u1", "concept_id": "a", "partial_credit_score": 0.9},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["concept_id"] == "a"
    assert body["label"] == "fractions"   # canonical label comes back
    assert body["updated"] is True

    state = fake_supabase.tables["kernel.student_concept_state"][0]
    assert state["user_id"] == "u1" and state["concept_id"] == "a"
    assert state["interactions_on_kc"] == 1


def test_update_concept_state_resolves_a_label(fake_supabase, monkeypatch):
    """A caller that only knows the concept's name still gets a graded update."""
    fake_supabase.seed(
        "kernel.concept_nodes",
        [{"id": "a", "label": "fractions", "subject": "MATH", "type_kc": "procedural"}],
    )
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    client = TestClient(main.app)
    # "Fractions" canonicalizes onto the existing `fractions` node — no duplicate.
    resp = client.post(
        "/update_concept_state",
        json={
            "user_id": "u1",
            "concept_label": "Fractions",
            "subject": "MATH",
            "partial_credit_score": 0.2,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["concept_id"] == "a"
    assert body["label"] == "fractions"
    assert len(fake_supabase.tables["kernel.concept_nodes"]) == 1


@pytest.mark.asyncio
async def test_update_concept_state_creates_an_unknown_label(fake_supabase, monkeypatch):
    """An unseen concept is created on the fly, like /analyze does."""
    async def fake_llm_call(prompt, max_tokens=1000):
        return (
            json.dumps(
                {
                    "type_kc": "conceptual",
                    "lambda_decay": 0.02,
                    "description": "le theoreme de Pythagore",
                    "prerequisites": [],
                    "tau": 0.5,
                }
            ),
            "mock-model",
        )

    monkeypatch.setattr(kc_registry, "llm_call", fake_llm_call)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    client = TestClient(main.app)
    resp = client.post(
        "/update_concept_state",
        json={"user_id": "u1", "concept_label": "theoreme_de_pythagore", "partial_credit_score": 0.8},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["label"] == "theoreme_de_pythagore"
    assert len(fake_supabase.tables["kernel.concept_nodes"]) == 1


def test_update_concept_state_needs_a_concept(fake_supabase, monkeypatch):
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    client = TestClient(main.app)
    resp = client.post(
        "/update_concept_state", json={"user_id": "u1", "partial_credit_score": 0.5}
    )
    assert resp.status_code == 422  # neither concept_id nor concept_label


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
