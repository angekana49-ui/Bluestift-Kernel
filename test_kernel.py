"""Tests for the Bluestift Cognitive Kernel.

Covers:
  - pure algorithm units (BKT, forgetting, mindset, detector, calibration),
  - get_or_create_kc() against the in-memory fake Supabase (LLM mocked),
  - /health and /analyze routes (LLM + DB mocked).

Run: pytest -q
"""
from __future__ import annotations

import asyncio
import math
import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

import main
from core import (
    anomaly, bkt, calibration, curriculum, detector, forgetting, mindset,
    prerequisites, ratelimit,
)
from core.graph import build_graph
from services import analyze as analyze_pipeline
from services import db as db_module
from services import graph_builder
from services import kc_registry
from services import llm


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


def test_bkt_partial_credit_is_not_quantised():
    # 0.72 and 0.7 are different evidence; no bin rounds one onto the other.
    assert bkt.update_bkt(0.5, True, partial_credit=0.72) > bkt.update_bkt(0.5, True, partial_credit=0.7)


def test_bkt_uses_node_params_over_priors():
    params = bkt.get_bkt_params({"p_init": 0.5, "p_slip": 0.05})
    assert params["p_init"] == 0.5
    assert params["p_slip"] == 0.05
    # Missing values fall back to priors.
    assert params["p_guess"] == bkt.BKT_PRIORS["p_guess"]


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


def test_every_mindset_label_is_reachable_from_real_signals():
    """The score has to span the labels it feeds.

    This is the assertion the bounds test above was missing, and the reason a
    broken scale survived: clamping to [0.05, 0.95] and classifying a hand-fed
    0.8 both passed while NO combination of signals could produce a 0.8 at all.
    A sigmoid over the un-centred weighted sum only reached [0.413, 0.657], so
    every student was "mixed" for life and `detect_fixed_mindset` was dead code.
    """
    gives_up = mindset.compute_mindset_score(1.0, 0.0, 0.0, 0.0)
    average = mindset.compute_mindset_score(0.5, 0.5, 0.5, 0.5)
    digs_in = mindset.compute_mindset_score(0.0, 1.0, 1.0, 1.0)

    assert mindset.classify_mindset(gives_up) == "fixed"
    assert mindset.classify_mindset(average) == "mixed"
    assert mindset.classify_mindset(digs_in) == "growth"
    # And the alert that reads M can actually fire.
    assert anomaly.detect_fixed_mindset(gives_up) is not None
    assert anomaly.detect_fixed_mindset(digs_in) is None
    # An all-neutral student sits exactly on the neutral point, which is what
    # makes "no signal" and "average signal" comparable numbers.
    assert average == pytest.approx(mindset.NEUTRAL_M)


def test_mindset_signals_are_clamped_to_the_unit_interval():
    """An LLM asked for [0,1] can still answer 1.4. It must not move the label
    further than a maximal honest signal would."""
    assert mindset.compute_mindset_score(0.0, 5.0, 5.0, 5.0) == mindset.compute_mindset_score(
        0.0, 1.0, 1.0, 1.0
    )
    assert mindset.compute_mindset_score(-3.0, 1.0, 1.0, 1.0) == mindset.compute_mindset_score(
        0.0, 1.0, 1.0, 1.0
    )


def test_mindset_is_smoothed_not_overwritten():
    """One exchange moves the trait estimate; it does not redefine it."""
    # First reading has nothing to blend against, so it stands alone.
    assert mindset.blend_mindset(None, 0.9) == 0.9

    # A growth-mindset student has one bad afternoon. It is allowed to move the
    # estimate down to "mixed" — that is an honest "we are no longer sure" — but
    # not to "fixed", which is the label that changes how the tutor teaches.
    after_one = mindset.blend_mindset(0.9, 0.05)
    assert mindset.classify_mindset(after_one) != "fixed", (
        "one terse conversation must not brand a learner"
    )

    # Consistent evidence does get there, it just takes more than one reading.
    m, readings = 0.9, 0
    while mindset.classify_mindset(m) != "fixed":
        m = mindset.blend_mindset(m, 0.05)
        readings += 1
        assert readings < 20, "sustained evidence has to be able to move M"
    assert readings >= 3, "the estimate must not be one conversation deep"


def test_personal_lambda_learns_from_the_first_attempt_back():
    args = dict(k_stored=0.9, floor=0.3, delta_days=30, p_slip=0.1, p_guess=0.2)
    # An unexpected failure after a month: this student forgets faster.
    faster = calibration.update_personal_lambda(0.02, credit=0.0, **args)
    # A success: slower. Both move, neither jumps.
    slower = calibration.update_personal_lambda(0.02, credit=1.0, **args)
    assert slower < 0.02 < faster
    assert 0.5 < slower / 0.02 < 1 and 1 < faster / 0.02 < 2
    # No gap, or nothing above the floor to forget: nothing to learn.
    assert calibration.update_personal_lambda(0.02, credit=0.0, **{**args, "delta_days": 0.5}) == 0.02
    assert calibration.update_personal_lambda(0.02, credit=0.0, **{**args, "k_stored": 0.3}) == 0.02


def _simulate_bkt(params, students, attempts, seed):
    import random

    rng = random.Random(seed)
    sequences = []
    for _ in range(students):
        learned = rng.random() < params["p_init"]
        seq = []
        for _ in range(attempts):
            p_correct = 1 - params["p_slip"] if learned else params["p_guess"]
            seq.append(1.0 if rng.random() < p_correct else 0.0)
            learned = learned or rng.random() < params["p_transit"]
        sequences.append(seq)
    return sequences


def test_fit_bkt_em_recovers_known_parameters():
    # Where the constants come from, eventually: our own students. Simulate a
    # population with parameters far from the literature priors and check the
    # fit finds them, not the priors.
    truth = {"p_init": 0.15, "p_transit": 0.25, "p_slip": 0.05, "p_guess": 0.35}
    fitted = calibration.fit_bkt_em(_simulate_bkt(truth, 400, 12, seed=7))
    for name, value in truth.items():
        # Nearer the truth than the prior it started from, on every parameter.
        assert abs(fitted[name] - value) < abs(bkt.BKT_PRIORS[name] - value), name
        # p_init and p_guess trade off against each other when guessing is
        # common (a lucky first answer looks like prior knowledge), so they
        # get a looser bound than slip and transit.
        assert fitted[name] == pytest.approx(value, abs=0.07), name


def test_fit_bkt_em_refuses_thin_evidence_and_stays_identifiable():
    few = _simulate_bkt(bkt.BKT_PRIORS, 5, 4, seed=1)
    assert calibration.fit_bkt_em(few) is None
    # Pure noise (coin flips) must not produce an inverted model.
    import random

    rng = random.Random(3)
    noise = [[float(rng.random() < 0.5) for _ in range(10)] for _ in range(100)]
    fitted = calibration.fit_bkt_em(noise)
    assert fitted["p_slip"] < 0.5 and fitted["p_guess"] < 0.5
    assert fitted["p_slip"] + fitted["p_guess"] < 1


def test_assisted_success_is_weak_evidence():
    autonomous = bkt.update_bkt(0.3, correct=True) - 0.3
    assisted = bkt.update_bkt(0.3, correct=True, assisted=True) - 0.3
    assert 0 < assisted < 0.6 * autonomous


def test_linguistic_block_does_not_count_against_the_concept():
    assert bkt.blocage_evidence(0.0, "linguistic") is None
    assert bkt.blocage_evidence(1.0, "linguistic") == 1.0  # a success still counts
    assert bkt.blocage_evidence(0.0, "ambiguous") == 0.25
    assert bkt.blocage_evidence(0.0, "conceptual") == 0.0
    # 0.5 is the uninformative credit the ambiguous rule shrinks towards.
    assert bkt.update_bkt(0.4, correct=False, partial_credit=0.5, params={**bkt.BKT_PRIORS, "p_transit": 0.0}) == pytest.approx(0.4)


def test_mastery_requires_three_recent_autonomous_attempts():
    base = {"mastery_score_raw": 0.99, "p_slip_personal": 0.05,
            "last_strong_signal_at": datetime.now(timezone.utc).isoformat()}
    assert bkt.effective_state({}, {**base, "recent_autonomous_credits": [1.0, 1.0]})["status"] == "partial"
    assert bkt.effective_state({}, {**base, "recent_autonomous_credits": [1.0, 0.8, 1.0]})["status"] == "mastered"
    assert bkt.effective_state({}, {**base, "recent_autonomous_credits": [0.3, 0.6, 0.7]})["status"] == "partial"
    # A row from before migration 011 (no list) falls back to the average.
    assert bkt.effective_state({}, {**base, "partial_credit_avg": 0.9})["status"] == "mastered"
    assert bkt.push_autonomous_credit([0.1, 0.2, 0.3], 0.4) == [0.2, 0.3, 0.4]


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


def test_group_trajectories_keeps_order_per_concept():
    rows = [
        {"concept_id": "a", "k_raw": 0.1},
        {"concept_id": "b", "k_raw": 0.7},
        {"concept_id": "a", "k_raw": 0.4},
        {"concept_id": "a", "k_raw": None},  # skipped, not a data point
    ]
    assert analyze_pipeline._group_trajectories(rows) == {"a": [0.1, 0.4], "b": [0.7]}


# --------------------------------------------------------------------------- #
# Rate limiting — the cost guard on /analyze
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clean_ratelimit():
    ratelimit.reset()
    yield
    ratelimit.reset()


def test_per_user_limit_stops_a_runaway_client(monkeypatch):
    monkeypatch.setenv("ANALYZE_PER_USER_HOURLY", "3")
    monkeypatch.setenv("ANALYZE_GLOBAL_HOURLY", "100")

    for _ in range(3):
        assert ratelimit.check_analyze("student-a")[0] is True

    allowed, retry_after, scope = ratelimit.check_analyze("student-a")
    assert allowed is False and scope == "user"
    assert 0 < retry_after <= ratelimit.WINDOW_SECONDS + 1

    # One student's loop must not lock everyone else out.
    assert ratelimit.check_analyze("student-b")[0] is True


def test_global_ceiling_catches_a_leaked_secret(monkeypatch):
    monkeypatch.setenv("ANALYZE_PER_USER_HOURLY", "100")
    monkeypatch.setenv("ANALYZE_GLOBAL_HOURLY", "3")

    # Spread across different users, so only the global ceiling can catch it.
    for i in range(3):
        assert ratelimit.check_analyze(f"student-{i}")[0] is True

    allowed, _retry, scope = ratelimit.check_analyze("student-99")
    assert allowed is False and scope == "global"


def test_a_rejected_call_consumes_no_budget(monkeypatch):
    """A call refused per-user must not still count toward the global ceiling."""
    monkeypatch.setenv("ANALYZE_PER_USER_HOURLY", "1")
    monkeypatch.setenv("ANALYZE_GLOBAL_HOURLY", "10")

    assert ratelimit.check_analyze("noisy")[0] is True
    for _ in range(20):
        assert ratelimit.check_analyze("noisy")[0] is False  # all rejected

    # The global budget still has room for everyone else: 1 spent, not 21.
    for i in range(9):
        assert ratelimit.check_analyze(f"quiet-{i}")[0] is True


def test_window_slides(monkeypatch):
    monkeypatch.setenv("ANALYZE_PER_USER_HOURLY", "2")
    monkeypatch.setenv("ANALYZE_GLOBAL_HOURLY", "100")
    t0 = 1_000_000.0

    assert ratelimit.check_analyze("s", now=t0)[0] is True
    assert ratelimit.check_analyze("s", now=t0 + 10)[0] is True
    assert ratelimit.check_analyze("s", now=t0 + 20)[0] is False
    # Once the first events age out of the window, room returns.
    assert ratelimit.check_analyze("s", now=t0 + ratelimit.WINDOW_SECONDS + 1)[0] is True


def test_bad_limit_env_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("ANALYZE_PER_USER_HOURLY", "not-a-number")
    monkeypatch.setenv("ANALYZE_GLOBAL_HOURLY", "-5")
    # A typo in configuration must not disable the guard or block every call.
    assert ratelimit._limit("ANALYZE_PER_USER_HOURLY", 30) == 30
    assert ratelimit._limit("ANALYZE_GLOBAL_HOURLY", 300) == 300


def test_analyze_route_returns_429_with_retry_after(fake_supabase, monkeypatch):
    monkeypatch.setenv("ANALYZE_PER_USER_HOURLY", "1")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    async def fake_llm_call(prompt, max_tokens=1000):
        return (json.dumps({"kcs_mentioned": [], "attempts": [],
                            "blocage_type": "none", "langue_interaction": "fr"}), "mock")

    monkeypatch.setattr(analyze_pipeline, "llm_call", fake_llm_call)
    client = TestClient(main.app)
    payload = {"user_id": "u1", "conversation_history": [{"role": "user", "content": "salut"}]}

    assert client.post("/analyze", json=payload).status_code == 200
    blocked = client.post("/analyze", json=payload)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    # A different student is unaffected.
    assert client.post("/analyze", json={**payload, "user_id": "u2"}).status_code == 200


# --------------------------------------------------------------------------- #
# LLM chain — Gemini over plain REST (no SDK)
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in that records the request."""

    calls: list[dict] = []
    response = _FakeResponse({})

    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, headers=None, json=None):
        type(self).calls.append({"url": url, "headers": headers, "json": json})
        return type(self).response


@pytest.fixture
def fake_httpx(monkeypatch):
    import httpx

    _FakeAsyncClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    return _FakeAsyncClient


@pytest.mark.asyncio
async def test_gemini_rest_call_shape(fake_httpx):
    fake_httpx.response = _FakeResponse(
        {"candidates": [{"content": {"parts": [{"text": "une "}, {"text": "reponse"}]}}]}
    )
    text = await llm.call_gemini("explique les derivees", max_tokens=500, temperature=0.3)

    # Parts are concatenated, like the SDK's `.text` did.
    assert text == "une reponse"
    call = fake_httpx.calls[0]
    assert call["url"].endswith(f"/models/{llm.GEMINI_MODEL}:generateContent")
    # The key goes in the header, never the query string (it would land in logs).
    assert call["headers"]["x-goog-api-key"] == "gemini-key"
    assert "gemini-key" not in call["url"]
    assert call["json"]["contents"][0]["parts"][0]["text"] == "explique les derivees"
    assert call["json"]["generationConfig"] == {"maxOutputTokens": 500, "temperature": 0.3}


@pytest.mark.asyncio
async def test_gemini_blocked_response_raises(fake_httpx):
    """A safety-blocked answer has no parts: raise so the caller's fallback sees it."""
    fake_httpx.response = _FakeResponse({"candidates": [{"finishReason": "SAFETY"}]})
    with pytest.raises(RuntimeError, match="SAFETY"):
        await llm.call_gemini("...")

    fake_httpx.response = _FakeResponse({"candidates": []})
    with pytest.raises(RuntimeError, match="no candidates"):
        await llm.call_gemini("...")


@pytest.mark.asyncio
async def test_llm_chain_falls_through_groq_to_gemini(fake_httpx, monkeypatch):
    def groq_down(*_args, **_kwargs):  # _get_groq is sync: raise like a dead client
        raise RuntimeError("groq 429")

    real_sleep = asyncio.sleep  # capture before patching, or the lambda recurses
    monkeypatch.setattr(llm, "_get_groq", groq_down)
    monkeypatch.setattr(llm.asyncio, "sleep", lambda *_: real_sleep(0))  # no real backoff
    fake_httpx.response = _FakeResponse(
        {"candidates": [{"content": {"parts": [{"text": "reponse de secours"}]}}]}
    )

    text, model = await llm.llm_call("prompt")
    assert text == "reponse de secours"
    assert model == llm.GEMINI_MODEL


@pytest.mark.asyncio
async def test_llm_chain_raises_only_when_both_fail(fake_httpx, monkeypatch):
    def groq_down(*_args, **_kwargs):  # _get_groq is sync: raise like a dead client
        raise RuntimeError("groq 429")

    real_sleep = asyncio.sleep  # capture before patching, or the lambda recurses
    monkeypatch.setattr(llm, "_get_groq", groq_down)
    monkeypatch.setattr(llm.asyncio, "sleep", lambda *_: real_sleep(0))
    fake_httpx.response = _FakeResponse({}, status=500)

    with pytest.raises(RuntimeError, match="All LLMs failed"):
        await llm.llm_call("prompt")


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


# Length matters: PyJWT warns below 32 bytes, and a real Supabase JWT secret
# is far longer than that.
JWT_SECRET = "test-jwt-secret-long-enough-for-hs256-0123456789"


def _user_token(user_id: str, secret: str = JWT_SECRET, **overrides) -> str:
    """Mint a Supabase-shaped access token for a student."""
    claims = {
        "sub": user_id,
        "aud": "authenticated",
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
        **overrides,
    }
    return jwt.encode(claims, secret, algorithm="HS256")


def test_user_token_reaches_only_its_own_profile(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    token = _user_token("student-a")
    headers = {"Authorization": f"Bearer {token}"}

    # Its own profile: allowed.
    ok = client.post("/load_profile", json={"user_id": "student-a"}, headers=headers)
    assert ok.status_code == 200

    # Someone else's: refused, and 403 (authenticated, not authorised) rather
    # than 401 — this is the skeleton-key case the tier exists to close.
    denied = client.post("/load_profile", json={"user_id": "student-b"}, headers=headers)
    assert denied.status_code == 403

    # Same rule on the write routes.
    assert client.post(
        "/update_concept_state",
        json={"user_id": "student-b", "concept_label": "fractions", "partial_credit_score": 0.5},
        headers=headers,
    ).status_code == 403
    assert client.post(
        "/analyze",
        json={"user_id": "student-b", "conversation_history": [{"role": "user", "content": "hi"}]},
        headers=headers,
    ).status_code == 403


def test_service_secret_still_acts_for_anyone(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    # The app is a trusted backend acting for many students; unchanged.
    for user_id in ("student-a", "student-b"):
        resp = client.post(
            "/load_profile", json={"user_id": user_id}, headers={"X-Kernel-Secret": "s3cr3t"}
        )
        assert resp.status_code == 200


def test_bad_user_tokens_are_refused(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    def post(token):
        return client.post(
            "/load_profile",
            json={"user_id": "student-a"},
            headers={"Authorization": f"Bearer {token}"},
        ).status_code

    # Signed with the wrong key — a forged token.
    assert post(_user_token("student-a", secret="w" * 40)) == 401
    # Expired.
    expired = jwt.encode(
        {"sub": "student-a", "aud": "authenticated",
         "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())},
        JWT_SECRET, algorithm="HS256",
    )
    assert post(expired) == 401
    # Wrong audience (e.g. a service token from another system).
    assert post(_user_token("student-a", aud="something-else")) == 401
    # Not a token at all.
    assert post("garbage") == 401


def test_user_token_cannot_seed_the_graph(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    headers = {"Authorization": f"Bearer {_user_token('student-a')}"}
    # Seeding rewrites the graph for everyone.
    assert client.post("/seed_kcs", headers=headers).status_code == 403
    assert client.post("/seed_kcs", headers={"X-Kernel-Secret": "s3cr3t"}).status_code == 200


def test_user_tier_is_unavailable_without_the_jwt_secret(client, fake_supabase, monkeypatch):
    """No SUPABASE_JWT_SECRET: the Kernel simply doesn't accept user tokens."""
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    resp = client.post(
        "/load_profile",
        json={"user_id": "student-a"},
        headers={"Authorization": f"Bearer {_user_token('student-a')}"},
    )
    assert resp.status_code == 401
    # The service tier is unaffected.
    assert client.post(
        "/load_profile", json={"user_id": "student-a"}, headers={"X-Kernel-Secret": "s3cr3t"}
    ).status_code == 200


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


# --------------------------------------------------------------------------- #
# Monitoring reads — /load_alerts and /resolve_alert
# --------------------------------------------------------------------------- #
def _seed_alerts(fake):
    """Two students' alerts, one already resolved, plus an operational log row."""
    fake.seed("kernel.concept_nodes", [{"id": "c1", "label": "derivees", "subject": "MATH"}])
    fake.seed(
        "kernel.kernel_monitoring",
        [
            {"id": "a1", "level": "alert", "user_id": "student-a", "concept_id": "c1",
             "alert_type": "cognitive_overload", "alert_severity": "high",
             "alert_details": {}, "resolved": False, "created_at": "2026-08-01T00:00:00Z"},
            {"id": "a2", "level": "alert", "user_id": "student-a", "concept_id": None,
             "alert_type": "fixed_mindset", "alert_severity": "medium",
             "alert_details": {}, "resolved": False, "created_at": "2026-08-03T00:00:00Z"},
            {"id": "a3", "level": "alert", "user_id": "student-a", "concept_id": None,
             "alert_type": "false_mastery", "alert_severity": "low",
             "alert_details": {}, "resolved": True, "created_at": "2026-07-01T00:00:00Z"},
            {"id": "a4", "level": "alert", "user_id": "student-b", "concept_id": None,
             "alert_type": "passive_dependency", "alert_severity": "high",
             "alert_details": {}, "resolved": False, "created_at": "2026-08-02T00:00:00Z"},
            # Not an alert: an ordinary operational log line.
            {"id": "m1", "level": "info", "event": "readiness_probe", "detail": {}},
        ],
    )


def test_load_alerts_returns_a_students_open_alerts(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)

    resp = client.post(
        "/load_alerts", json={"user_id": "student-a"}, headers={"X-Kernel-Secret": "s3cr3t"}
    )
    assert resp.status_code == 200
    body = resp.json()

    # Only this student, only unresolved, only alert rows — never the info log.
    assert [a["id"] for a in body["alerts"]] == ["a2", "a1"]  # newest first
    assert body["scope"] == "user"
    assert body["counts_by_type"] == {"fixed_mindset": 1, "cognitive_overload": 1}
    assert body["counts_by_severity"] == {"medium": 1, "high": 1}
    assert body["truncated"] is False
    # The UUID stored on the row is useless to a human; the label is resolved.
    assert next(a for a in body["alerts"] if a["id"] == "a1")["concept_label"] == "derivees"


def test_resolved_alerts_come_back_only_when_asked(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)

    resp = client.post(
        "/load_alerts",
        json={"user_id": "student-a", "include_resolved": True},
        headers={"X-Kernel-Secret": "s3cr3t"},
    )
    assert {a["id"] for a in resp.json()["alerts"]} == {"a1", "a2", "a3"}


def test_alert_filters_narrow_by_severity_and_date(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)
    headers = {"X-Kernel-Secret": "s3cr3t"}

    high = client.post(
        "/load_alerts", json={"user_id": "student-a", "severity": "high"}, headers=headers
    )
    assert [a["id"] for a in high.json()["alerts"]] == ["a1"]

    recent = client.post(
        "/load_alerts",
        json={"user_id": "student-a", "since": "2026-08-02T00:00:00Z"},
        headers=headers,
    )
    assert [a["id"] for a in recent.json()["alerts"]] == ["a2"]


def test_truncated_says_so_rather_than_implying_completeness(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)

    resp = client.post(
        "/load_alerts",
        json={"user_id": "student-a", "limit": 1},
        headers={"X-Kernel-Secret": "s3cr3t"},
    )
    body = resp.json()
    assert len(body["alerts"]) == 1 and body["truncated"] is True


def test_a_student_reads_only_their_own_alerts(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)
    headers = {"Authorization": f"Bearer {_user_token('student-a')}"}

    assert client.post("/load_alerts", json={"user_id": "student-a"}, headers=headers).status_code == 200
    assert client.post("/load_alerts", json={"user_id": "student-b"}, headers=headers).status_code == 403


def test_school_scope_is_service_only_and_covers_the_roster(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)
    fake_supabase.seed(
        "schools.student_identities",
        [
            {"id": "i1", "user_id": "student-a", "school_id": "school-1"},
            {"id": "i2", "user_id": "student-b", "school_id": "school-1"},
            {"id": "i3", "user_id": "student-z", "school_id": "school-2"},
        ],
    )

    # The Kernel can't tell a teacher's token from a student's, so it refuses to
    # decide: a school view needs the app's own authorization, i.e. the service tier.
    student = client.post(
        "/load_alerts",
        json={"school_id": "school-1"},
        headers={"Authorization": f"Bearer {_user_token('student-a')}"},
    )
    assert student.status_code == 403

    resp = client.post(
        "/load_alerts", json={"school_id": "school-1"}, headers={"X-Kernel-Secret": "s3cr3t"}
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["scope"] == "school"
    # Both students of this school, and nobody from school-2.
    assert [a["id"] for a in body["alerts"]] == ["a2", "a4", "a1"]
    # Surfaced so a school expecting 300 students notices it is only seeing 2.
    assert body["students_in_scope"] == 2


def test_alerts_need_exactly_one_scope(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    headers = {"X-Kernel-Secret": "s3cr3t"}

    assert client.post("/load_alerts", json={}, headers=headers).status_code == 422
    assert client.post(
        "/load_alerts", json={"user_id": "student-a", "school_id": "school-1"}, headers=headers
    ).status_code == 422


def test_a_failed_read_is_a_503_never_an_empty_all_clear(client, fake_supabase, monkeypatch):
    """The one failure mode a safety dashboard must not have.

    Every other read in db.py degrades to []. Here that would render as "no
    alerts" — telling a school that nothing is wrong precisely when the Kernel
    can no longer tell.
    """
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)

    def boom(*_args, **_kwargs):
        raise RuntimeError("permission denied for table kernel_monitoring")

    monkeypatch.setattr(db_module, "load_alerts", boom)
    resp = client.post(
        "/load_alerts", json={"user_id": "student-a"}, headers={"X-Kernel-Secret": "s3cr3t"}
    )
    assert resp.status_code == 503
    assert "alerts unavailable" in resp.json()["detail"]


def test_resolve_alert_closes_reopens_and_404s(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)
    headers = {"X-Kernel-Secret": "s3cr3t"}

    done = client.post(
        "/resolve_alert", json={"alert_id": "a1", "resolved_by": "mme-durand"}, headers=headers
    )
    assert done.status_code == 200
    assert done.json()["resolved"] is True
    assert done.json()["resolved_by"] == "mme-durand"

    # It drops out of the dashboard's default view.
    open_now = client.post("/load_alerts", json={"user_id": "student-a"}, headers=headers)
    assert [a["id"] for a in open_now.json()["alerts"]] == ["a2"]

    # Closed by mistake: reopening clears the acknowledgement too.
    back = client.post(
        "/resolve_alert",
        json={"alert_id": "a1", "resolved_by": "mme-durand", "resolved": False},
        headers=headers,
    )
    assert back.json()["resolved"] is False and back.json()["resolved_by"] is None

    assert client.post(
        "/resolve_alert", json={"alert_id": "nope", "resolved_by": "x"}, headers=headers
    ).status_code == 404


def test_resolving_never_touches_an_operational_log_row(client, fake_supabase, monkeypatch):
    """`m1` is an info log, not an alert — /resolve_alert must not rewrite it."""
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)

    resp = client.post(
        "/resolve_alert", json={"alert_id": "m1", "resolved_by": "x"},
        headers={"X-Kernel-Secret": "s3cr3t"},
    )
    assert resp.status_code == 404
    row = next(r for r in fake_supabase.tables["kernel.kernel_monitoring"] if r["id"] == "m1")
    assert "resolved" not in row


def test_a_student_cannot_close_the_alert_raised_about_them(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)

    resp = client.post(
        "/resolve_alert",
        json={"alert_id": "a1", "resolved_by": "student-a"},
        headers={"Authorization": f"Bearer {_user_token('student-a')}"},
    )
    assert resp.status_code == 403


def test_roster_scope_answers_for_exactly_the_students_given(client, fake_supabase, monkeypatch):
    """A teacher sees their assigned classes — never a whole establishment."""
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_alerts(fake_supabase)

    resp = client.post(
        "/load_alerts",
        json={"user_ids": ["student-b", "student-b", "student-c"]},
        headers={"X-Kernel-Secret": "s3cr3t"},
    )
    body = resp.json()
    assert body["scope"] == "users"
    assert [a["id"] for a in body["alerts"]] == ["a4"]
    # student-a is not on the list, so their alerts stay out of this teacher's view.
    assert all(a["user_id"] != "student-a" for a in body["alerts"])
    # Duplicates in the request don't inflate the roster count.
    assert body["students_in_scope"] == 2

    # A student's token can't ask about a list, only about itself.
    assert client.post(
        "/load_alerts",
        json={"user_ids": ["student-a"]},
        headers={"Authorization": f"Bearer {_user_token('student-a')}"},
    ).status_code == 403

    # Still exactly one scope.
    assert client.post(
        "/load_alerts",
        json={"user_id": "student-a", "user_ids": ["student-b"]},
        headers={"X-Kernel-Secret": "s3cr3t"},
    ).status_code == 422


# --------------------------------------------------------------------------- #
# Calibration scheduling
# --------------------------------------------------------------------------- #
def test_a_never_calibrated_kc_is_always_due():
    assert calibration.is_calibration_due({}) is True
    assert calibration.is_calibration_due({"last_calibration_at": None}) is True


def test_cooldown_keeps_a_busy_kc_from_being_rescanned():
    now = datetime.now(timezone.utc)
    fresh = {"last_calibration_at": (now - timedelta(hours=1)).isoformat()}
    stale = {"last_calibration_at": (now - timedelta(hours=7)).isoformat()}
    # Recalibration reads every student's state for the KC: a popular KC would
    # otherwise be fully rescanned on every conversation that touches it.
    assert calibration.is_calibration_due(fresh, now) is False
    assert calibration.is_calibration_due(stale, now) is True


def test_an_unreadable_timestamp_recalibrates_rather_than_skips():
    # Losing a calibration is recoverable; silently never calibrating again
    # because one row holds junk is not.
    assert calibration.is_calibration_due({"last_calibration_at": "not-a-date"}) is True


def test_analyze_schedules_recalibration_for_the_kcs_it_committed(monkeypatch):
    """The evidence lands in /analyze, so calibration has to be triggered there."""
    scheduled: list[str] = []

    async def fake_run_analysis(_client, request_id, _payload):
        return {
            "request_id": request_id,
            "user_id": "u1",
            "root_gap": None,
            "root_concept_id": None,
            "detection_path": [],
            "mastery_map": {},
            "confidence": 0.0,
            "summary": "",
            "recommended_path": [],
            "alerts": [],
            "llm_used": "test",
            "recalibrate_concept_ids": ["kc-a", "kc-b"],
        }

    monkeypatch.setattr(analyze_pipeline, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(main, "_recalibrate_kc", lambda cid: scheduled.append(cid))
    monkeypatch.setattr(db_module, "get_client", lambda: object())
    monkeypatch.setattr(db_module, "log_kernel_request", lambda *a, **k: None)
    monkeypatch.delenv("KERNEL_API_SECRET", raising=False)

    client = TestClient(main.app)
    resp = client.post(
        "/analyze",
        json={"user_id": "u1", "conversation_history": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert scheduled == ["kc-a", "kc-b"]
    # The scheduling key is internal plumbing and must not leak into the contract.
    assert "recalibrate_concept_ids" not in resp.json()


# --------------------------------------------------------------------------- #
# Sleep discipline: the Kernel works only when asked
# --------------------------------------------------------------------------- #
def test_boot_touches_no_network(monkeypatch):
    """Starting up must not talk to anything.

    The service sleeps and is woken by a request, so a boot happens on the
    caller's latency budget — several times a day, not once a month. Any network
    call added to the lifespan is paid on every single wake, and any background
    timer would stop the container from ever sleeping again.
    """
    calls: list[str] = []
    monkeypatch.setattr(db_module, "get_client", lambda: calls.append("get_client"))
    monkeypatch.setattr(db_module, "check_db_access", lambda *_: calls.append("probe"))
    monkeypatch.setattr(db_module, "log_monitoring", lambda *a, **k: calls.append("log"))

    # Entering the TestClient context runs the lifespan startup.
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
    assert calls == [], f"boot reached the network: {calls}"


def test_no_background_thread_or_timer_is_started():
    """Nothing may keep the process busy between requests.

    Railway sleeps the service after ten minutes with no outbound traffic. A
    poller, a scheduler or a keep-alive would silently turn a $0.12/month
    service into an always-on one, and the symptom is only ever the invoice.
    """
    import threading

    before = threading.active_count()
    with TestClient(main.app) as c:
        c.get("/health")
    assert threading.active_count() <= before


# --------------------------------------------------------------------------- #
# GraphRAG — multi-hop prerequisite reasoning
# --------------------------------------------------------------------------- #
def _chain_graph():
    """calcul_litteral -> variable -> fonction -> derivation, plus limite -> derivation."""
    import networkx as nx

    g = nx.DiGraph()
    for prereq, concept in [
        ("calcul_litteral", "notion_de_variable"),
        ("notion_de_variable", "notion_de_fonction"),
        ("notion_de_fonction", "derivation_fonction"),
        ("notion_de_limite", "derivation_fonction"),
    ]:
        g.add_edge(prereq, concept)
    return g


def test_a_mastered_prerequisite_is_a_wall_not_a_door():
    """The rule that makes the answer short enough to act on.

    A student who holds notion_de_fonction demonstrably carries what it rests
    on. Listing those anyway would bury the one real gap under foundations they
    already have.
    """
    g = _chain_graph()
    report = prerequisites.gap_report(g, "derivation_fonction", {"notion_de_fonction": "mastered"})

    assert [x["label"] for x in report["gaps"]] == ["notion_de_limite"]
    # And it says WHERE it stopped, so the short list is evidently short by
    # reason rather than by the walk giving up.
    assert [x["label"] for x in report["frontier"]] == ["notion_de_fonction"]


def test_unknown_is_expanded_but_never_counted_as_mastered():
    # Never having been asked is not the same as having failed — and it is
    # certainly not the same as having understood.
    g = _chain_graph()
    report = prerequisites.gap_report(g, "derivation_fonction", {})
    labels = [x["label"] for x in report["gaps"]]
    assert "calcul_litteral" in labels  # three hops away, still reached
    assert report["frontier"] == []


def test_gaps_come_back_in_teaching_order_deepest_first():
    """Topology is the hard constraint; depth breaks the ties it leaves open."""
    g = _chain_graph()
    report = prerequisites.gap_report(g, "derivation_fonction", {})
    order = [x["label"] for x in report["gaps"]]

    # Nothing before what it rests on.
    assert order.index("calcul_litteral") < order.index("notion_de_variable")
    assert order.index("notion_de_variable") < order.index("notion_de_fonction")
    # Between independent branches, the deeper foundation leads: that is the
    # Kernel's thesis, not a cosmetic choice.
    assert order[0] == "calcul_litteral"


def test_depth_limit_reports_that_it_cut_the_list():
    g = _chain_graph()
    cut = prerequisites.gap_report(g, "derivation_fonction", {}, max_hops=1)
    assert cut["truncated"] is True
    assert {x["label"] for x in cut["gaps"]} == {"notion_de_fonction", "notion_de_limite"}

    # A walk that simply ran out of graph is NOT truncation: claiming otherwise
    # would make "nothing else is missing" unsayable.
    whole = prerequisites.gap_report(g, "derivation_fonction", {})
    assert whole["truncated"] is False


def test_an_unknown_target_yields_nothing_rather_than_raising():
    assert prerequisites.gap_report(_chain_graph(), "pas_un_concept", {})["gaps"] == []


def test_a_cycle_does_not_take_the_report_down():
    import networkx as nx

    g = nx.DiGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    g.add_edge("b", "target")
    report = prerequisites.gap_report(g, "target", {})
    assert {x["label"] for x in report["gaps"]} == {"a", "b"}


def _seed_graph_and_student(fake, *, mastered_fonction=False):
    """The maths chain, one student who is shaky on derivatives."""
    nodes = [
        {"id": "kc-calc", "label": "calcul_litteral", "subject": "MATH", "type_kc": "procedural"},
        {"id": "kc-var", "label": "notion_de_variable", "subject": "MATH", "type_kc": "conceptual"},
        {"id": "kc-fonc", "label": "notion_de_fonction", "subject": "MATH", "type_kc": "conceptual"},
        {"id": "kc-lim", "label": "notion_de_limite", "subject": "MATH", "type_kc": "conceptual"},
        {"id": "kc-deriv", "label": "derivation_fonction", "subject": "MATH", "type_kc": "conceptual"},
    ]
    edges = [
        {"id": "e1", "prerequisite_id": "kc-calc", "concept_id": "kc-var"},
        {"id": "e2", "prerequisite_id": "kc-var", "concept_id": "kc-fonc"},
        {"id": "e3", "prerequisite_id": "kc-fonc", "concept_id": "kc-deriv"},
        {"id": "e4", "prerequisite_id": "kc-lim", "concept_id": "kc-deriv"},
    ]
    fake.seed("kernel.concept_nodes", nodes)
    fake.seed("kernel.concept_edges", edges)
    if mastered_fonction:
        fake.seed(
            "kernel.student_concept_state",
            [{
                "id": "s1", "user_id": "student-a", "concept_id": "kc-fonc",
                # Clear of the 0.95 mastery line: decay starts the instant the
                # signal is written, so a fresh 0.95 reads 0.9499... a moment later.
                "mastery_score_raw": 0.97, "partial_credit_avg": 0.85,
                "p_slip_personal": 0.05,
                "last_strong_signal_at": datetime.now(timezone.utc).isoformat(),
            }],
        )


def test_prerequisite_gaps_answers_what_vector_search_cannot(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_graph_and_student(fake_supabase, mastered_fonction=True)

    resp = client.post(
        "/prerequisite_gaps",
        json={"user_id": "student-a", "concept_label": "derivation_fonction"},
        headers={"X-Kernel-Secret": "s3cr3t"},
    )
    assert resp.status_code == 200
    body = resp.json()

    # The student holds notion_de_fonction, so the walk stops there and the
    # answer is the one thing actually missing — not every ancestor in the graph.
    assert [g["label"] for g in body["gaps"]] == ["notion_de_limite"]
    assert [f["label"] for f in body["frontier"]] == ["notion_de_fonction"]
    assert body["gaps"][0]["concept_id"] == "kc-lim"
    assert body["gaps"][0]["hops"] == 1
    assert body["gaps"][0]["status"] == "unknown"
    assert body["truncated"] is False


def test_a_label_the_graph_does_not_know_is_a_404_not_a_new_node(client, fake_supabase, monkeypatch):
    """A question must not create graph nodes.

    /update_concept_state may invent a KC because it is reporting evidence about
    one. Asking what a concept rests on is a read, and a read that writes would
    let any caller grow the shared graph by typing.
    """
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_graph_and_student(fake_supabase)

    resp = client.post(
        "/prerequisite_gaps",
        json={"user_id": "student-a", "concept_label": "chimie_organique"},
        headers={"X-Kernel-Secret": "s3cr3t"},
    )
    assert resp.status_code == 404
    assert fake_supabase.tables["kernel.concept_nodes"].__len__() == 5  # nothing created


def test_a_student_reads_only_their_own_prerequisite_gaps(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_graph_and_student(fake_supabase)
    headers = {"Authorization": f"Bearer {_user_token('student-a')}"}

    ok = client.post(
        "/prerequisite_gaps",
        json={"user_id": "student-a", "concept_label": "derivation_fonction"},
        headers=headers,
    )
    assert ok.status_code == 200
    denied = client.post(
        "/prerequisite_gaps",
        json={"user_id": "student-b", "concept_label": "derivation_fonction"},
        headers=headers,
    )
    assert denied.status_code == 403


def test_teaching_material_never_crosses_between_students_or_classes(fake_supabase):
    """The leak surface of the retrieval half.

    A chunk may be global curriculum material, one student's own upload, or a
    class's. A student may see the first two and their own class's — never
    another child's document, never another class's.
    """
    fake_supabase.seed(
        "schools.student_identities",
        [{"id": "i1", "user_id": "student-a", "class_id": "class-1"}],
    )
    fake_supabase.seed(
        "rag.rag_chunks",
        [
            {"id": "c-global", "concept_id": "kc-lim", "content": "cours public",
             "user_id": None, "class_id": None, "source_type": "curriculum"},
            {"id": "c-mine", "concept_id": "kc-lim", "content": "ma fiche",
             "user_id": "student-a", "class_id": None, "source_type": "upload"},
            {"id": "c-my-class", "concept_id": "kc-lim", "content": "poly de la classe",
             "user_id": None, "class_id": "class-1", "source_type": "school"},
            {"id": "c-other-student", "concept_id": "kc-lim", "content": "fiche de Tom",
             "user_id": "student-b", "class_id": None, "source_type": "upload"},
            {"id": "c-other-class", "concept_id": "kc-lim", "content": "poly 3e B",
             "user_id": None, "class_id": "class-9", "source_type": "school"},
        ],
    )

    class_id = db_module.load_student_class_id(fake_supabase, "student-a")
    assert class_id == "class-1"
    got = {
        c["id"]
        for c in db_module.load_chunks_for_concepts(
            fake_supabase, ["kc-lim"], "student-a", class_id
        )
    }
    assert got == {"c-global", "c-mine", "c-my-class"}


def test_with_no_class_resolved_class_material_is_left_out(fake_supabase):
    """Fails closed: an unresolved class must not widen the scope."""
    fake_supabase.seed(
        "rag.rag_chunks",
        [
            {"id": "c-global", "concept_id": "kc-lim", "content": "x",
             "user_id": None, "class_id": None},
            {"id": "c-some-class", "concept_id": "kc-lim", "content": "y",
             "user_id": None, "class_id": "class-1"},
        ],
    )
    got = {
        c["id"]
        for c in db_module.load_chunks_for_concepts(fake_supabase, ["kc-lim"], "student-a", None)
    }
    assert got == {"c-global"}


def test_an_empty_corpus_says_so_instead_of_pretending(client, fake_supabase, monkeypatch):
    """resources_available is false because nothing is written yet — not because
    these concepts are undocumented. The distinction is the whole reason the
    field exists."""
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    _seed_graph_and_student(fake_supabase)

    body = client.post(
        "/prerequisite_gaps",
        json={"user_id": "student-a", "concept_label": "derivation_fonction"},
        headers={"X-Kernel-Secret": "s3cr3t"},
    ).json()
    assert body["resources_available"] is False
    assert all(g["resources"] == [] for g in body["gaps"])
    assert len(body["gaps"]) == 4  # the reasoning stands on its own without a corpus


# --------------------------------------------------------------------------- #
# Carrying more than two subjects
# --------------------------------------------------------------------------- #
def test_the_vocabulary_budget_serves_the_subject_being_analysed():
    """Label drift is how a graph fragments as subjects are added.

    The budget used to be filled in database order across every subject. Past
    it, the subject the student is actually working on could be barely present,
    the model would stop seeing its canonical labels, and it would invent
    variants — creating near-duplicate KCs exactly when the vocabulary matters
    most.
    """
    rows = (
        [{"label": f"hist_{i}", "subject": "HISTORY"} for i in range(500)]
        + [{"label": f"math_{i}", "subject": "MATH"} for i in range(200)]
    )
    rendered = analyze_pipeline._format_vocabulary(rows, "MATH")
    sent = set(rendered.split(", "))

    # Every maths label fits and is sent: the home subject is served first.
    assert {f"math_{i}" for i in range(200)} <= sent
    # And history is not squeezed out — a maths conversation must still be able
    # to name a concept from elsewhere.
    assert any(lbl.startswith("hist_") for lbl in sent)
    assert len(sent) <= analyze_pipeline.VOCABULARY_BUDGET


def test_a_large_foreign_vocabulary_cannot_crowd_out_the_home_subject():
    rows = (
        [{"label": f"phys_{i}", "subject": "PHYSICS"} for i in range(1000)]
        + [{"label": f"math_{i}", "subject": "MATH"} for i in range(50)]
    )
    sent = set(analyze_pipeline._format_vocabulary(rows, "MATH").split(", "))
    assert {f"math_{i}" for i in range(50)} <= sent
    # The home subject only needed 50 of its share; the rest goes to the others
    # rather than being wasted.
    assert len(sent) == analyze_pipeline.VOCABULARY_BUDGET


def test_a_small_graph_is_sent_whole_regardless_of_subject():
    rows = [{"label": "a", "subject": "MATH"}, {"label": "b", "subject": "PHYSICS"}]
    assert analyze_pipeline._format_vocabulary(rows, "MATH") == "a, b"


def test_the_graph_is_paged_so_it_is_never_silently_half_loaded(fake_supabase):
    """A truncated graph is not a partial read, it is a wrong answer.

    PostgREST caps rows per response. A graph cut at that cap would make the
    detector walk a subgraph and name a root that isn't one, and
    /prerequisite_gaps would report a student has nothing left to learn because
    the rest of the graph was never sent.
    """
    total = db_module.GRAPH_PAGE_SIZE * 2 + 37
    fake_supabase.seed(
        "kernel.concept_nodes",
        [{"id": f"kc-{i}", "label": f"c{i}", "subject": "MATH"} for i in range(total)],
    )
    assert len(db_module.load_concept_nodes(fake_supabase)) == total


def test_the_extraction_prompt_does_not_close_the_list_of_subjects():
    """Subject coverage was capped by a closed enum in the prompt.

    It offered MATH | ENGLISH | PHYSICS | HISTORY | CHEMISTRY | BIOLOGY | OTHER,
    so geography, philosophy, economics and the rest all collapsed onto OTHER.
    That is not cosmetic: a school's curriculum layers are matched BY SUBJECT
    (load_curriculum_layers), so a GEOGRAPHY layer could never reach KCs filed
    as OTHER, and the per-subject vocabulary budget would treat every unnamed
    discipline as one blob.
    """
    prompt = analyze_pipeline.EXTRACTION_PROMPT
    # The closed enum is gone: no `| "OTHER"` alternation on the subject field.
    assert '| "OTHER"' not in prompt
    assert '"ENGLISH" |' not in prompt
    # And the model is told, in words, to name the subject instead.
    assert "mot-cle LIBRE" in prompt
    assert 'N\'utilise PAS "OTHER"' in prompt
    # The examples stay, as examples — they must not read as the whole set.
    assert "GEOGRAPHY" in prompt and "PHILOSOPHY" in prompt


def test_update_mindset_keeps_the_stored_score_when_a_turn_says_nothing(fake_supabase):
    """A silent conversation is not evidence of an average student.

    `_update_mindset` returns the value P is modulated with, so answering 0.5 for
    a learner already measured at 0.9 would quietly flatten their persistence
    score on every exchange that happened not to carry a mindset signal.
    """
    fake_supabase.seed(
        "kernel.student_mindset_state",
        [{"user_id": "u1", "m_score": 0.9, "detected_mindset": "growth"}],
    )
    assert analyze_pipeline._update_mindset(fake_supabase, "u1", None) == 0.9
    # Nothing was written: there was nothing new to write.
    assert fake_supabase.tables["kernel.student_mindset_state"][0]["m_score"] == 0.9

    # An unmeasured student with no signal still has to yield a usable number.
    assert analyze_pipeline._update_mindset(fake_supabase, "u2", None) == mindset.NEUTRAL_M


def test_update_mindset_blends_into_the_stored_score(fake_supabase):
    fake_supabase.seed(
        "kernel.student_mindset_state",
        [{"user_id": "u1", "m_score": 0.9, "detected_mindset": "growth"}],
    )
    signals = {
        "abandon_rate": 1.0,
        "persistence_score": 0.0,
        "time_on_task": 0.0,
        "interaction_quality": 0.0,
    }
    m = analyze_pipeline._update_mindset(fake_supabase, "u1", signals)
    assert 0.05 < m < 0.9, "the reading must move the estimate without replacing it"
    row = fake_supabase.tables["kernel.student_mindset_state"][0]
    assert row["m_score"] == round(m, 4)
    assert row["detected_mindset"] == mindset.classify_mindset(m)


# --------------------------------------------------------------------------- #
# Core-correctness regressions (audit 2026-09-30)
# --------------------------------------------------------------------------- #
# Each test below pins a defect that made the Kernel's diagnosis disagree with
# its own documentation. They are phrased as the behaviour a learner is owed,
# not as the implementation, so a future refactor can't quietly reintroduce one.
def _extraction_llm(extraction: dict):
    """An LLM mock whose extraction call returns `extraction`; everything else
    (KC metadata, summary) gets a harmless answer."""

    async def fake_llm_call(prompt, max_tokens=1000):
        if "kcs_mentioned" in prompt:
            return json.dumps(extraction), "mock-model"
        if "prerequisites" in prompt:
            return json.dumps({"type_kc": "conceptual", "prerequisites": []}), "mock-model"
        return "resume", "mock-model"

    return fake_llm_call


async def _analyze(fake_supabase, monkeypatch, extraction: dict, user_id: str = "u1") -> dict:
    fake = _extraction_llm(extraction)
    monkeypatch.setattr(analyze_pipeline, "llm_call", fake)
    monkeypatch.setattr(kc_registry, "llm_call", fake)
    payload = {
        "user_id": user_id,
        "conversation_history": [{"role": "user", "content": "..."}],
        "subject": "MATH",
        "level": "lycee",
    }
    return await analyze_pipeline.run_analysis(fake_supabase, "req-1", payload)


def test_bkt_partial_credit_is_graded_not_binary():
    # Ostrow & Heffernan: a 0.3 answer is better evidence than a 0.0 one, and a
    # 0.6 answer weaker than a perfect one. A binarised update erases that.
    ks = [bkt.update_bkt(0.5, correct=False, partial_credit=b) for b in bkt.PARTIAL_CREDIT_BINS]
    assert ks == sorted(ks)
    assert len(set(ks)) == len(ks), "every credit bin must move mastery differently"
    # The ends of the scale are still the classic binary BKT update.
    assert bkt.update_bkt(0.5, correct=True, partial_credit=1.0) == pytest.approx(
        bkt.update_bkt(0.5, correct=True)
    )
    assert bkt.update_bkt(0.5, correct=False, partial_credit=0.0) == pytest.approx(
        bkt.update_bkt(0.5, correct=False)
    )


def test_bkt_params_stay_identifiable():
    # slip + guess >= 1 makes a success LOWER mastery (the model inverts). A
    # calibrated or hand-edited node must never be able to put the Kernel there.
    params = bkt.get_bkt_params({"p_slip": 0.6, "p_guess": 0.7, "p_transit": 0.0})
    assert params["p_slip"] < 0.5 and params["p_guess"] < 0.5
    assert bkt.update_bkt(0.5, correct=True, params=params) > 0.5
    assert bkt.update_bkt(0.5, correct=False, params=params) < 0.5
    # A legitimately calibrated 0.0 is data, not "missing".
    assert bkt.get_bkt_params({"p_transit": 0.0})["p_transit"] == 0.0


def test_mastered_status_requires_the_dual_condition():
    # K alone is not mastery: a KC slipped on 40% of the time is not "mastered",
    # whatever K says — that is exactly the false-mastery pattern we alert on.
    assert bkt.classify_status(0.99, p_slip=0.4, partial_credit_avg=0.9) != "mastered"
    assert bkt.classify_status(0.8, p_slip=0.1, partial_credit_avg=0.2) != "mastered"
    assert bkt.classify_status(0.8, p_slip=0.1, partial_credit_avg=0.8) == "partial"
    assert bkt.classify_status(0.99, p_slip=0.1, partial_credit_avg=0.8) == "mastered"


def test_root_gap_is_deterministic_and_prefers_the_weakest():
    # Three independent failing KCs, equal on convergence, evidence and depth.
    # The weakest one is the most urgent — and the answer must not depend on the
    # process's hash seed (it used to change between two identical requests).
    nodes = [{"id": x, "label": x} for x in ("fractions", "equations", "algebra", "geometry")]
    edges = [
        {"prerequisite_id": "fractions", "concept_id": "algebra"},
        {"prerequisite_id": "equations", "concept_id": "algebra"},
    ]
    g = build_graph(nodes, edges)
    states = {"fractions": 0.2, "equations": 0.25, "geometry": 0.1}
    result = detector.detect_root_cause(g, ["fractions", "equations", "geometry"], states)
    assert result["root_gap"] == "geometry"


def test_build_graph_breaks_prerequisite_cycles():
    # LLM-inferred prerequisites can close a loop (A needs B, B needs A). The
    # root-cause walk assumes a DAG; a cycle makes "deepest prerequisite" undefined.
    import networkx as nx

    nodes = [{"id": x, "label": x} for x in "abc"]
    edges = [
        {"prerequisite_id": "a", "concept_id": "b"},
        {"prerequisite_id": "b", "concept_id": "c"},
        {"prerequisite_id": "c", "concept_id": "a"},
    ]
    g = build_graph(nodes, edges)
    assert nx.is_directed_acyclic_graph(g)
    assert g.number_of_edges() == 2


async def test_analyze_applies_every_attempt_not_just_the_last(fake_supabase, monkeypatch):
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    attempts = [
        {"kc_label": "fractions", "outcome": "failure", "partial_credit": 0.0},
        {"kc_label": "fractions", "outcome": "failure", "partial_credit": 0.0},
        {"kc_label": "fractions", "outcome": "success", "partial_credit": 1.0},
    ]
    out = await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}],
        "attempts": attempts,
    })
    # Replay the same evidence, in order, through the model by hand.
    k = bkt.BKT_PRIORS["p_init"]
    for a in attempts:
        k = bkt.update_bkt(k, correct=a["outcome"] == "success", partial_credit=a["partial_credit"])
    assert out["mastery_map"]["fractions"]["k_raw"] == pytest.approx(k, abs=1e-4)
    row = fake_supabase.tables["kernel.student_concept_state"][0]
    assert row["interactions_on_kc"] == 3
    assert row["struggle_index"] == 2


async def test_analyze_decays_mastery_before_updating_it(fake_supabase, monkeypatch):
    # 200 days without practice: the Kernel's own view (load_profile) says the
    # concept has faded. The next observation must update THAT belief, not the
    # stale pre-holiday value.
    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    fake_supabase.seed(
        "kernel.concept_nodes",
        [{"id": "f", "label": "fractions", "subject": "MATH", "type_kc": "conceptual"}],
    )
    fake_supabase.seed(
        "kernel.student_concept_state",
        [{"user_id": "u1", "concept_id": "f", "mastery_score_raw": 0.9,
          "partial_credit_avg": 0.1, "interactions_on_kc": 4, "last_strong_signal_at": long_ago}],
    )
    out = await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}],
        "attempts": [{"kc_label": "fractions", "outcome": "success", "partial_credit": 1.0}],
    })
    decayed = forgetting.compute_effective_mastery(0.9, "conceptual", long_ago)
    assert out["mastery_map"]["fractions"]["k_raw"] == pytest.approx(
        bkt.update_bkt(decayed, correct=True), abs=1e-3
    )


async def test_mentioned_but_unpractised_kc_is_unknown_not_a_gap(fake_supabase, monkeypatch):
    # A student naming a concept is not evidence they fail it. It used to start
    # at p_init (0.3), fall under the failing threshold, and become a "root gap"
    # the tutor would then remediate — on no evidence at all.
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    out = await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}],
        "attempts": [],
    })
    assert out["mastery_map"]["fractions"]["status"] == "unknown"
    assert out["root_gap"] is None


async def test_attempt_label_follows_the_canonical_kc(fake_supabase, monkeypatch):
    # The registry canonicalises "Fractions " to "fractions"; the attempt naming
    # the raw spelling must still land on that KC instead of being dropped.
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": "Fractions ", "subject": "MATH"}],
        "attempts": [{"kc_label": "Fractions", "outcome": "failure", "partial_credit": 0.0}],
    })
    assert len(fake_supabase.tables["kernel.student_concept_state"]) == 1


async def test_success_with_a_placeholder_zero_credit_counts_as_success(fake_supabase, monkeypatch):
    # The extraction template shows `"partial_credit": 0.0`; an LLM that copies
    # it next to outcome=success must not have that success scored as a failure.
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    out = await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}],
        "attempts": [{"kc_label": "fractions", "outcome": "success", "partial_credit": 0.0}],
    })
    assert out["mastery_map"]["fractions"]["k_raw"] > bkt.BKT_PRIORS["p_init"]


async def test_kc_lookup_does_not_treat_underscore_as_a_wildcard(fake_supabase, monkeypatch):
    # In SQL LIKE, `_` matches any character. "fraction_addition" must not
    # resolve to an unrelated "fractionxaddition".
    fake_supabase.seed(
        "kernel.concept_nodes", [{"id": "x", "label": "fractionxaddition", "subject": "MATH"}]
    )
    monkeypatch.setattr(kc_registry, "llm_call", _extraction_llm({}))
    kc = await kc_registry.get_or_create_kc("fraction_addition", "MATH", "lycee", fake_supabase)
    assert kc["id"] != "x"
    assert kc["label"] == "fraction_addition"


def test_update_concept_state_decays_before_updating(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    fake_supabase.seed(
        "kernel.concept_nodes",
        [{"id": "f", "label": "fractions", "subject": "MATH", "type_kc": "conceptual"}],
    )
    fake_supabase.seed(
        "kernel.student_concept_state",
        [{"user_id": "u1", "concept_id": "f", "mastery_score_raw": 0.9,
          "interactions_on_kc": 4, "last_strong_signal_at": long_ago}],
    )
    resp = client.post(
        "/update_concept_state",
        json={"user_id": "u1", "concept_id": "f", "partial_credit_score": 1.0},
        headers={"X-Kernel-Secret": "s3cr3t"},
    )
    assert resp.status_code == 200, resp.text
    decayed = forgetting.compute_effective_mastery(0.9, "conceptual", long_ago)
    assert resp.json()["k_raw"] == pytest.approx(bkt.update_bkt(decayed, correct=True), abs=1e-3)


async def test_consistent_success_keeps_raising_mastery(fake_supabase, monkeypatch):
    # One success per session, every session. The old "strong signal" gate held
    # all of these back (no 3 attempts, no credit shift), so K froze and then
    # decayed while the student kept succeeding.
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    fake_supabase.seed(
        "kernel.student_concept_state",
        [{"user_id": "u1", "concept_id": "f", "mastery_score_raw": 0.7, "partial_credit_avg": 1.0,
          "interactions_on_kc": 3, "last_strong_signal_at": datetime.now(timezone.utc).isoformat()}],
    )
    extraction = {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}],
        "attempts": [{"kc_label": "fractions", "outcome": "success", "partial_credit": 1.0}],
    }
    ks = []
    for _ in range(3):
        out = await _analyze(fake_supabase, monkeypatch, extraction)
        ks.append(out["mastery_map"]["fractions"]["k_raw"])
    assert ks[0] > 0.7 and ks == sorted(ks) and len(set(ks)) == 3


def test_forgetting_decays_towards_the_prior_not_zero():
    # A student who once learned a concept and has not practised it in a year
    # is not below one who never saw it: with no evidence left, we are back to
    # the prior — not to certainty that they don't know.
    a_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    k = forgetting.compute_effective_mastery(0.95, "declarative", a_year_ago, floor=0.3)
    assert k == pytest.approx(0.3, abs=1e-3)
    assert k >= 0.3
    # Halfway in time, halfway in the gap above the floor (one half-life).
    half_life = math.log(2) / forgetting.LAMBDA_PRIORS["declarative"]
    halfway = datetime.now(timezone.utc) - timedelta(days=half_life)
    assert forgetting.compute_effective_mastery(0.9, "declarative", halfway, floor=0.3) == pytest.approx(0.6, abs=1e-3)
    # Time does not improve a failure: below the floor, K stays where it is.
    assert forgetting.compute_effective_mastery(0.1, "declarative", a_year_ago, floor=0.3) == pytest.approx(0.1)


async def test_analyze_logs_every_attempt_and_skips_linguistic_failures(fake_supabase, monkeypatch):
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    out = await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}],
        "attempts": [
            # Failed because the statement's vocabulary was not understood.
            {"kc_label": "fractions", "outcome": "failure", "partial_credit": 0.0,
             "blocage_type": "linguistic"},
            {"kc_label": "fractions", "outcome": "success", "partial_credit": 1.0},
        ],
    })
    # Only the success moved K.
    assert out["mastery_map"]["fractions"]["k_raw"] == pytest.approx(
        bkt.update_bkt(bkt.BKT_PRIORS["p_init"], correct=True), abs=1e-4
    )
    events = fake_supabase.tables["kernel.learning_events"]
    assert [e["counted"] for e in events] == [False, True]
    assert events[1]["k_before"] == pytest.approx(bkt.BKT_PRIORS["p_init"])
    assert all(e["source"] == "analyze" and e["request_id"] == "req-1" for e in events)


async def test_assisted_success_moves_mastery_less(fake_supabase, monkeypatch):
    fake_supabase.seed("kernel.concept_nodes", [
        {"id": "a", "label": "fractions", "subject": "MATH"},
        {"id": "b", "label": "equations", "subject": "MATH"},
    ])
    out = await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}, {"label": "equations", "subject": "MATH"}],
        "attempts": [
            {"kc_label": "fractions", "outcome": "success", "partial_credit": 1.0, "is_assisted": True},
            {"kc_label": "equations", "outcome": "success", "partial_credit": 1.0},
        ],
    })
    assert out["mastery_map"]["fractions"]["k_raw"] < out["mastery_map"]["equations"]["k_raw"]
    rows = {r["concept_id"]: r for r in fake_supabase.tables["kernel.student_concept_state"]}
    # Only the autonomous attempt enters the dual-condition window.
    assert rows["a"]["recent_autonomous_credits"] == []
    assert rows["b"]["recent_autonomous_credits"] == [1.0]


def test_update_concept_state_ignores_a_linguistic_failure(client, fake_supabase, monkeypatch):
    monkeypatch.setenv("KERNEL_API_SECRET", "s3cr3t")
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    resp = client.post(
        "/update_concept_state",
        json={"user_id": "u1", "concept_id": "f", "partial_credit_score": 0.0,
              "blocage_type": "linguistic"},
        headers={"X-Kernel-Secret": "s3cr3t"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["updated"] is False
    assert resp.json()["status"] == "unknown"
    assert fake_supabase.tables["kernel.student_concept_state"] == []
    event = fake_supabase.tables["kernel.learning_events"][0]
    assert event["counted"] is False and event["source"] == "update_concept_state"


def test_recalibration_fits_bkt_params_from_logged_evidence(fake_supabase, monkeypatch):
    monkeypatch.setattr(db_module, "get_client", lambda: fake_supabase)
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    truth = {"p_init": 0.15, "p_transit": 0.25, "p_slip": 0.05, "p_guess": 0.35}
    events, t0 = [], datetime(2026, 1, 1, tzinfo=timezone.utc)
    for u, seq in enumerate(_simulate_bkt(truth, 200, 10, seed=11)):
        for i, c in enumerate(seq):
            events.append({"user_id": f"u{u}", "concept_id": "f", "credit": c, "is_assisted": False,
                           "counted": True, "source": "analyze",
                           "created_at": (t0 + timedelta(minutes=10 * u + i)).isoformat()})
    fake_supabase.seed("kernel.learning_events", events)
    main._recalibrate_kc("f")
    node = fake_supabase.tables["kernel.concept_nodes"][0]
    assert node["last_calibration_at"]
    for name, value in truth.items():
        assert abs(node[name] - value) < abs(bkt.BKT_PRIORS[name] - value), name


def test_recommended_path_climbs_back_to_where_the_student_is_stuck():
    # racine -> algebre -> equations (the student is stuck on equations), and
    # racine -> aaa_unrelated. The old walk took the alphabetically first
    # dependent and led away from the problem.
    nodes = [{"id": x, "label": x} for x in ("racine", "algebre", "equations", "aaa_unrelated")]
    edges = [
        {"prerequisite_id": "racine", "concept_id": "algebre"},
        {"prerequisite_id": "algebre", "concept_id": "equations"},
        {"prerequisite_id": "racine", "concept_id": "aaa_unrelated"},
    ]
    g = build_graph(nodes, edges)
    path = detector.recommended_path(g, "racine", detection_path=["equations", "algebre", "racine"])
    assert path == ["racine", "algebre", "equations"]


async def test_one_label_is_one_kc_across_subjects(fake_supabase, monkeypatch):
    # A physics conversation about vecteurs practises the maths KC, instead of
    # creating a second "vecteurs" node the graph would silently merge.
    fake_supabase.seed("kernel.concept_nodes", [{"id": "v", "label": "vecteurs", "subject": "MATH"}])
    monkeypatch.setattr(kc_registry, "llm_call", _extraction_llm({}))
    kc = await kc_registry.get_or_create_kc("vecteurs", "PHYSICS", "lycee", fake_supabase)
    assert kc["id"] == "v"
    assert len(fake_supabase.tables["kernel.concept_nodes"]) == 1


def test_build_graph_resolves_legacy_duplicate_labels_deterministically():
    nodes = [
        {"id": "new", "label": "vecteurs", "subject": "PHYSICS", "created_at": "2026-08-01T00:00:00Z"},
        {"id": "old", "label": "vecteurs", "subject": "MATH", "created_at": "2026-07-01T00:00:00Z"},
        {"id": "f", "label": "forces", "subject": "PHYSICS", "created_at": "2026-08-01T00:00:00Z"},
    ]
    edges = [{"prerequisite_id": "new", "concept_id": "f"}]
    for order in (nodes, list(reversed(nodes))):
        g = build_graph(order, edges)
        assert g.nodes["vecteurs"]["id"] == "old"
        assert g.graph["duplicate_labels"] == {"vecteurs": ["old", "new"]}
        assert g.has_edge("vecteurs", "forces")  # the duplicate's edges survive


async def test_kc_creation_cascade_is_bounded(fake_supabase, monkeypatch):
    # Every inferred KC claims three brand-new prerequisites: unbounded, one
    # unknown concept costs 40 sequential LLM calls on the request path.
    calls = {"n": 0}

    async def fanout(prompt, max_tokens=1000):
        calls["n"] += 1
        return json.dumps({"type_kc": "conceptual",
                           "prerequisites": [f"p{calls['n']}_{i}" for i in range(3)]}), "mock"

    monkeypatch.setattr(kc_registry, "llm_call", fanout)
    kc = await kc_registry.get_or_create_kc("tout_nouveau", "MATH", "lycee", fake_supabase)
    assert kc["label"] == "tout_nouveau"
    assert calls["n"] == kc_registry.MAX_LLM_CALLS_PER_REQUEST
    # The KCs past the budget still exist, with neutral metadata.
    assert len(fake_supabase.tables["kernel.concept_nodes"]) > calls["n"]


async def test_weak_positive_evidence_is_not_a_failure(fake_supabase, monkeypatch):
    # One assisted success on a new KC leaves K under 0.5 but ABOVE the prior:
    # the evidence points towards knowing. It used to make the KC "failing",
    # and so a root gap.
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    out = await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}],
        "attempts": [{"kc_label": "fractions", "outcome": "success", "partial_credit": 1.0,
                      "is_assisted": True}],
    })
    assert bkt.BKT_PRIORS["p_init"] < out["mastery_map"]["fractions"]["k_raw"] < 0.5
    assert out["root_gap"] is None


def test_failing_threshold_is_the_prior_capped_at_one_half():
    assert detector.failing_threshold(0.3) == 0.3
    assert detector.failing_threshold(0.8) == detector.FAILING_THRESHOLD


# --------------------------------------------------------------------------- #
# The state chain: first come, first served; no lost update
# --------------------------------------------------------------------------- #
async def test_concurrent_analyses_of_one_student_lose_no_evidence(fake_supabase, monkeypatch):
    # Two conversations analysed at the same time, each with one failure on the
    # same KC. Both read the state, both write: without the chain, the second
    # write erased the first and the student had 1 interaction instead of 2.
    fake_supabase.seed("kernel.concept_nodes", [{"id": "f", "label": "fractions", "subject": "MATH"}])
    extraction = {
        "kcs_mentioned": [{"label": "fractions", "subject": "MATH"}],
        "attempts": [{"kc_label": "fractions", "outcome": "failure", "partial_credit": 0.0}],
    }

    async def slow_llm(prompt, max_tokens=1000):
        await asyncio.sleep(0.01)  # lets the two requests interleave
        if "kcs_mentioned" in prompt:
            return json.dumps(extraction), "mock"
        return "resume", "mock"

    monkeypatch.setattr(analyze_pipeline, "llm_call", slow_llm)
    monkeypatch.setattr(kc_registry, "llm_call", slow_llm)
    payload = {"user_id": "u1", "conversation_history": [{"role": "user", "content": "..."}],
               "subject": "MATH", "level": "lycee"}
    await asyncio.gather(
        analyze_pipeline.run_analysis(fake_supabase, "r1", payload),
        analyze_pipeline.run_analysis(fake_supabase, "r2", payload),
    )
    rows = fake_supabase.tables["kernel.student_concept_state"]
    assert len(rows) == 1
    assert rows[0]["interactions_on_kc"] == 2
    assert rows[0]["struggle_index"] == 2
    assert rows[0]["version"] == 2
    k = bkt.update_bkt(bkt.update_bkt(bkt.BKT_PRIORS["p_init"], False), False)
    assert rows[0]["mastery_score_raw"] == pytest.approx(k, abs=1e-3)


def test_a_write_from_a_stale_state_is_rejected(fake_supabase):
    fake_supabase.seed("kernel.student_concept_state",
                       [{"user_id": "u1", "concept_id": "f", "mastery_score_raw": 0.4, "version": 3}])
    row = {"user_id": "u1", "concept_id": "f", "mastery_score_raw": 0.9}
    with pytest.raises(db_module.StaleState):
        db_module.commit_student_concept_state(fake_supabase, row, read_version=2)
    assert fake_supabase.tables["kernel.student_concept_state"][0]["mastery_score_raw"] == 0.4
    db_module.commit_student_concept_state(fake_supabase, row, read_version=3)
    stored = fake_supabase.tables["kernel.student_concept_state"][0]
    assert stored["mastery_score_raw"] == 0.9 and stored["version"] == 4
    # A first block where another request already wrote one is stale too.
    with pytest.raises(db_module.StaleState):
        db_module.commit_student_concept_state(fake_supabase, row, read_version=None)


async def test_student_queue_serves_in_arrival_order():
    from services import serial

    order = []

    async def request(name, hold):
        async with serial.student("u1"):
            order.append(name)
            await asyncio.sleep(hold)

    await asyncio.gather(request("first", 0.02), request("second", 0.0), request("third", 0.0))
    assert order == ["first", "second", "third"]
    assert "u1" not in serial._queues  # nothing left behind once idle


# --------------------------------------------------------------------------- #
# tau (rigour) and the spacing effect
# --------------------------------------------------------------------------- #
def test_tau_is_neutral_at_one_half_and_demands_more_when_higher():
    assert bkt.rigor_credit(0.7, 0.5) == pytest.approx(0.7)
    assert bkt.rigor_credit(0.7, 0.8) < 0.7 < bkt.rigor_credit(0.7, 0.3)
    for tau in (0.2, 0.5, 0.9):  # full success and full failure never move
        assert bkt.rigor_credit(1.0, tau) == 1.0 and bkt.rigor_credit(0.0, tau) == 0.0
    assert bkt.mastery_threshold(0.5) == pytest.approx(bkt.MASTERY_THRESHOLD)
    assert bkt.mastery_threshold(0.8) > bkt.mastery_threshold(0.5) > bkt.mastery_threshold(0.3)
    # K 0.96 is mastery on a neutral KC, not on a rigorous one.
    assert bkt.classify_status(0.96, 0.05, 0.9, tau=0.5) == "mastered"
    assert bkt.classify_status(0.96, 0.05, 0.9, tau=0.8) == "partial"
    # A rigorous KC fades faster by default; a missing tau is neutral.
    assert forgetting.base_lambda({"type_kc": "conceptual", "tau": 0.8}) > forgetting.base_lambda({"type_kc": "conceptual"})
    assert bkt.kc_tau({}) == bkt.TAU_NEUTRAL and bkt.kc_tau({"tau": "junk"}) == bkt.TAU_NEUTRAL


def test_spaced_reviews_slow_forgetting_and_lapses_undo_it():
    node = {"type_kc": "declarative"}
    month_ago = datetime.now(timezone.utc) - timedelta(days=30)

    def k_after_a_month(state):
        return forgetting.compute_effective_mastery(
            0.9, "declarative", month_ago, lambda_override=forgetting.get_lambda(node, state), floor=0.3
        )

    fresh = k_after_a_month({})
    reviewed = k_after_a_month({"review_count": 2})
    assert reviewed > fresh
    # Two reviews = one half-life doubling.
    assert forgetting.get_lambda(node, {"review_count": 2}) == pytest.approx(forgetting.get_lambda(node, {}) / 2)
    assert k_after_a_month({"review_count": 2, "lapse_count": 2}) == pytest.approx(fresh)
    # Lapses never make a student forget faster than their base rate.
    assert k_after_a_month({"lapse_count": 5}) == pytest.approx(fresh)


async def test_a_retrieval_after_a_gap_counts_as_review_or_lapse(fake_supabase, monkeypatch):
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    fake_supabase.seed("kernel.concept_nodes", [
        {"id": x, "label": x, "subject": "MATH"} for x in ("rev", "lap", "help", "same_day")
    ])
    fake_supabase.seed("kernel.student_concept_state", [
        {"user_id": "u1", "concept_id": x, "mastery_score_raw": 0.8, "interactions_on_kc": 3,
         "last_strong_signal_at": when}
        for x, when in (("rev", three_days_ago), ("lap", three_days_ago),
                        ("help", three_days_ago), ("same_day", now))
    ])
    await _analyze(fake_supabase, monkeypatch, {
        "kcs_mentioned": [{"label": x, "subject": "MATH"} for x in ("rev", "lap", "help", "same_day")],
        "attempts": [
            {"kc_label": "rev", "outcome": "success", "partial_credit": 1.0},
            {"kc_label": "lap", "outcome": "failure", "partial_credit": 0.0},
            {"kc_label": "help", "outcome": "success", "partial_credit": 1.0, "is_assisted": True},
            {"kc_label": "same_day", "outcome": "success", "partial_credit": 1.0},
        ],
    })
    rows = {r["concept_id"]: r for r in fake_supabase.tables["kernel.student_concept_state"]}
    assert (rows["rev"]["review_count"], rows["rev"]["lapse_count"]) == (1, 0)
    assert (rows["lap"]["review_count"], rows["lap"]["lapse_count"]) == (0, 1)
    assert (rows["help"]["review_count"], rows["help"]["lapse_count"]) == (0, 0)       # help masks retention
    assert (rows["same_day"]["review_count"], rows["same_day"]["lapse_count"]) == (0, 0)  # no gap, no test
