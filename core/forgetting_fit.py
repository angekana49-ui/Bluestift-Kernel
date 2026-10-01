"""Fit the forgetting constants on what students actually remembered.

core/forgetting.py decays mastery between sessions at

    lambda = base_lambda(KC type, tau) * 2 ** -(REVIEW_GAIN * reviews - LAPSE_PENALTY * lapses)

and every number in it is a design value: the per-type priors and the two
half-life regression weights (Settles & Meeder 2016). This module fits them on
kernel.learning_events.

The fit replays each student's history on each KC through the Kernel's own
model — BKT updates within a session, decay between sessions, reviews and
lapses counted exactly as services/analyze.py counts them — and scores the one
prediction that tests forgetting: the first unassisted answer after a gap of a
day or more (a *retrieval*). Its credit c is scored against the predicted
P(correct) = K_eff (1 - slip) + (1 - K_eff) guess by cross-entropy. The constants
that make those predictions most accurate win.

Two guards against fooling ourselves:
- Students are split once, by a stable hash, into a fitting set and a held-out
  set. The report compares the constants on students the fit never saw.
- The current constants and a no-spacing model (reviews ignored) are scored on
  the same held-out students. Adopt the fitted values only if they beat both.

The personal lambda (calibration.update_personal_lambda) is left out on
purpose: it is learned online from these same retrievals, and including it
would fit the constants to the residue of a model already tuned on the answer.
"""
from __future__ import annotations

import hashlib
import math

from core import bkt, forgetting

# [design] Retrievals needed before a fit is reported at all. Below this the
# held-out set is a few dozen predictions and the comparison is noise.
MIN_RETRIEVALS_FOR_FIT = 200
# [design] Share of students held out, by hash.
HOLDOUT_SHARE = 0.2

# Search grids. Weights in half-lives per review / lapse; per-type multipliers
# of the forgetting prior, on a log2 scale (x1/8 .. x8).
WEIGHT_GRID = [round(0.1 * i, 2) for i in range(0, 21)]          # 0 .. 2
SCALE_GRID = [2.0 ** (0.5 * i) for i in range(-6, 7)]            # 1/8 .. 8
MAX_ROUNDS = 6

_EPS = 1e-6


def sequences(events: list[dict], nodes_by_id: dict[str, dict]) -> list[dict]:
    """Group counted learning_events into per-(student, KC) histories, oldest first.

    Returns [{"user_id", "node", "events": [(days, credit, assisted), ...]}],
    `days` measured from the first event of that history.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for e in events:
        if e.get("counted") is False or e.get("credit") is None:
            continue
        node = nodes_by_id.get(e.get("concept_id"))
        if node is None:
            continue
        grouped.setdefault((e["user_id"], e["concept_id"]), []).append(e)
    out = []
    for (user_id, concept_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: str(r["created_at"]))
        t0 = forgetting._parse_ts(rows[0]["created_at"])
        evs = [
            (
                (forgetting._parse_ts(r["created_at"]) - t0).total_seconds() / 86400.0,
                float(r["credit"]),
                bool(r.get("is_assisted")),
            )
            for r in rows
        ]
        out.append({"user_id": user_id, "node": nodes_by_id[concept_id], "events": evs})
    return out


def _decay(k: float, floor: float, lam: float, days: float) -> float:
    if k <= floor or days <= 0:
        return k
    return floor + (k - floor) * math.exp(-lam * days)


def replay(seq: dict, gain: float, penalty: float, scale: float):
    """Yield (predicted P(correct), credit) for each retrieval in one history."""
    node = seq["node"]
    params = bkt.get_bkt_params(node)
    floor = params["p_init"]
    # The type prior (level 3 of base_lambda), not a KC's empirical lambda:
    # the empirical one is an average of personal lambdas, themselves learned
    # from these retrievals.
    prior_node = {k: v for k, v in node.items() if k not in ("lambda_decay", "interactions_count")}
    base = forgetting.base_lambda(prior_node, None) * scale
    k, reviews, lapses, last_t = floor, 0, 0, None
    for t, credit, assisted in seq["events"]:
        gap = 0.0 if last_t is None else t - last_t
        strength = max(0.0, gain * reviews - penalty * lapses)
        k = _decay(k, floor, base * 2.0 ** -strength, gap)
        if last_t is not None and gap >= forgetting.REVIEW_MIN_GAP_DAYS and not assisted:
            yield k * (1 - params["p_slip"]) + (1 - k) * params["p_guess"], credit
            if credit >= 0.5:
                reviews += 1
            else:
                lapses += 1
        k = bkt.update_bkt(k, correct=credit >= 0.5, partial_credit=credit, params=params, assisted=assisted)
        last_t = t


def item_losses(seqs: list[dict], gain: float, penalty: float, scales: dict[str, float]) -> list[float]:
    """Cross-entropy, in bits, of each retrieval prediction."""
    out = []
    for seq in seqs:
        scale = scales.get(seq["node"].get("type_kc", "conceptual"), 1.0)
        for p, c in replay(seq, gain, penalty, scale):
            p = min(1 - _EPS, max(_EPS, p))
            out.append(-(c * math.log2(p) + (1 - c) * math.log2(1 - p)))
    return out


def log_loss(seqs: list[dict], gain: float, penalty: float, scales: dict[str, float]) -> tuple[float, int]:
    """Mean cross-entropy (bits) of the retrieval predictions, and their count."""
    losses = item_losses(seqs, gain, penalty, scales)
    return (sum(losses) / len(losses) if losses else float("nan")), len(losses)


def _held_out(user_id: str) -> bool:
    digest = hashlib.sha256(str(user_id).encode()).digest()
    return digest[0] / 256.0 < HOLDOUT_SHARE


# 95% likelihood-ratio threshold for one parameter: chi2(1) = 3.84, in nats of
# the total log-likelihood, i.e. 2 * n * delta_mean_bits * ln 2 < 3.84.
_LR_95 = 3.84


def fit(seqs: list[dict]) -> dict:
    """Fit REVIEW_GAIN, LAPSE_PENALTY and a multiplier per KC type.

    Coordinate descent over the grids, from the current constants. Returns a
    report; `fitted` is None when there is not enough data.
    """
    train = [s for s in seqs if not _held_out(s["user_id"])]
    test = [s for s in seqs if _held_out(s["user_id"])]
    types = sorted({s["node"].get("type_kc", "conceptual") for s in train})

    current = {"gain": forgetting.REVIEW_GAIN, "penalty": forgetting.LAPSE_PENALTY,
               "scales": {t: 1.0 for t in types}}
    _, n_train = log_loss(train, current["gain"], current["penalty"], current["scales"])
    _, n_test = log_loss(test, current["gain"], current["penalty"], current["scales"])
    report = {"retrievals_train": n_train, "retrievals_test": n_test,
              "min_retrievals": MIN_RETRIEVALS_FOR_FIT, "fitted": None, "adopt": False}
    if n_train < MIN_RETRIEVALS_FOR_FIT or n_test == 0:
        return report

    def loss(cand: dict) -> float:
        return log_loss(train, cand["gain"], cand["penalty"], cand["scales"])[0]

    def variants(base: dict):
        """(parameter name, grid value, candidate) for one coordinate sweep."""
        for key in ("gain", "penalty"):
            for v in WEIGHT_GRID:
                yield key, v, {**base, key: v}
        for t in types:
            for v in SCALE_GRID:
                yield t, v, {**base, "scales": {**base["scales"], t: v}}

    best = {"gain": current["gain"], "penalty": current["penalty"], "scales": dict(current["scales"])}
    best_loss = loss(best)
    for _ in range(MAX_ROUNDS):
        improved = False
        for _name, _v, cand in variants(best):
            lv = loss(cand)
            if lv < best_loss - 1e-9:
                best, best_loss, improved = cand, lv, True
        if not improved:
            break

    # How well determined each value is: the grid values whose likelihood is
    # not significantly worse (95%, likelihood ratio, the others held at the
    # fit). A wide interval means the data cannot tell — keep the design value.
    within: dict[str, list[float]] = {}
    for name, v, cand in variants(best):
        if 2 * n_train * (loss(cand) - best_loss) * math.log(2) < _LR_95:
            within.setdefault(name, []).append(v)

    def interval(name: str, to_value) -> list[float]:
        vals = within.get(name) or []
        return [round(to_value(min(vals)), 5), round(to_value(max(vals)), 5)] if vals else []

    def prior(t: str):
        return lambda s_: forgetting.LAMBDA_PRIORS.get(t, 0.02) * s_

    # Held out: is the fitted model better on students the fit never saw, and
    # by more than chance? Paired, per retrieval: mean loss difference and its
    # standard error.
    fitted_items = item_losses(test, best["gain"], best["penalty"], best["scales"])

    def compare(other: dict) -> dict:
        other_items = item_losses(test, other["gain"], other["penalty"], other["scales"])
        diffs = [o - f for f, o in zip(fitted_items, other_items)]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        return {"loss": round(sum(other_items) / len(other_items), 5),
                "fitted_better_by": round(mean, 5),
                "standard_error": round(math.sqrt(var / len(diffs)), 5)}

    no_spacing = {"gain": 0.0, "penalty": 0.0, "scales": current["scales"]}
    vs_current, vs_no_spacing = compare(current), compare(no_spacing)
    report.update({
        "fitted": {
            "REVIEW_GAIN": best["gain"],
            "LAPSE_PENALTY": best["penalty"],
            "LAMBDA_PRIORS": {t: round(prior(t)(s_), 5) for t, s_ in best["scales"].items()},
        },
        "interval_95": {
            "REVIEW_GAIN": interval("gain", float),
            "LAPSE_PENALTY": interval("penalty", float),
            "LAMBDA_PRIORS": {t: interval(t, prior(t)) for t in types},
        },
        # Held-out cross-entropy in bits per retrieval: lower is better.
        "held_out": {
            "fitted_loss": round(sum(fitted_items) / len(fitted_items), 5),
            "vs_current": vs_current,
            "vs_no_spacing": vs_no_spacing,
        },
    })
    # Adopt only on a held-out gain of at least two standard errors over BOTH
    # the current constants and the model that ignores reviews.
    report["adopt"] = all(
        c["fitted_better_by"] > 2 * c["standard_error"] for c in (vs_current, vs_no_spacing)
    )
    return report


def simulate(rng, students: int, gain: float, penalty: float, scales: dict[str, float],
             kc_types=("procedural", "conceptual", "declarative")) -> list[dict]:
    """Histories whose answers are drawn from this very model with known constants.

    Checks that the fit recovers what generated the data — identifiability,
    not the model's truth. Each student meets every KC type over 2-6 sessions
    of 1-3 answers, sessions 0.5 to 60 days apart.
    """
    seqs = []
    for i in range(students):
        for j, kc_type in enumerate(kc_types):
            node = {"id": f"kc{j}", "type_kc": kc_type}
            params = bkt.get_bkt_params(node)
            floor = params["p_init"]
            base = forgetting.base_lambda(node, None) * scales.get(kc_type, 1.0)
            k, reviews, lapses, t, last_t, events = floor, 0, 0, 0.0, None, []
            for s in range(rng.randint(2, 6)):
                if s:
                    t += rng.choice((0.5, 1, 2, 4, 7, 14, 30, 60))
                for a in range(rng.randint(1, 3)):
                    gap = 0.0 if last_t is None else t - last_t
                    strength = max(0.0, gain * reviews - penalty * lapses)
                    k = _decay(k, floor, base * 2.0 ** -strength, gap)
                    p = k * (1 - params["p_slip"]) + (1 - k) * params["p_guess"]
                    credit = 1.0 if rng.random() < p else 0.0
                    if last_t is not None and gap >= forgetting.REVIEW_MIN_GAP_DAYS:
                        reviews, lapses = reviews + (credit >= 0.5), lapses + (credit < 0.5)
                    k = bkt.update_bkt(k, correct=credit >= 0.5, partial_credit=credit, params=params)
                    events.append((t, credit, False))
                    last_t = t
            seqs.append({"user_id": f"u{i}", "node": node, "events": events})
    return seqs
