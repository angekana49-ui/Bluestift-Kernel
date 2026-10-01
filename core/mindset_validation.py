"""Does M predict anything? Validation of the mindset score on what came next.

M claims to read a student's belief about their own ability (Dweck). A belief
cannot be observed, so the claim is tested by its consequences: a student read
as growth-minded should, in the weeks that follow,

- PERSIST — retry after a failure rather than drop the concept;
- LEARN BEYOND THEIR LEVEL — answer better than the Kernel's own mastery
  estimate predicted (credit minus the BKT prediction from k_before). The
  outcome is a residual on purpose: "high-M students score higher" would only
  show that M tracks level, which it must not (see core/mindset.py);
- COME BACK — practise again at all.

Each analysis logged with a mindset_trace (services/analyze.py) is one
prediction; its outcomes are measured on the student's learning_events in the
window after it. Associations are Spearman correlations, with a 95% interval
from a bootstrap that resamples STUDENTS, not analyses — one student's twenty
analyses are not twenty independent observations.

The same machinery tests the smoothing hypothesis. M is an exponential moving
average of per-conversation readings with weight mindset.EMA_WEIGHT (0.3,
a design value). Recomputing the average at other weights over the same
readings shows which weight predicts best. If 1.0 (no smoothing) wins, M is a
state that moves with each conversation, not a trait; if a small weight wins,
it is closer to a trait.

Correlation, not causation, and only as good as the outcomes: the "persist"
outcome is measured within sessions the tutor shapes, and RAYA adapts to M.
"""
from __future__ import annotations

import hashlib
import random

from core import bkt, forgetting

# [design] Window after an analysis in which its outcomes are measured.
OUTCOME_WINDOW_DAYS = 30
# [design] "Came back": any practice in this many days after the analysis.
RETURN_WINDOW_DAYS = 14
# Without a request_id (graded attempts from /update_concept_state), attempts
# closer than this belong to the same session.
SESSION_GAP_HOURS = 1.0
# [design] Below these, the report says "not enough data" instead of numbers.
MIN_STUDENTS = 30
MIN_ANALYSES = 100
BOOTSTRAP_SAMPLES = 500
EMA_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

PREDICTORS = ("m_score", "observed", "conversation_linear", "measured_linear")
OUTCOMES = ("persistence", "learning_residual", "returned")


def _days(a, b) -> float:
    return (forgetting._parse_ts(b) - forgetting._parse_ts(a)).total_seconds() / 86400.0


def _sessions(events: list[dict]) -> list[list[dict]]:
    """Group one student's events (oldest first) into sessions."""
    sessions: list[list[dict]] = []
    for e in events:
        if sessions:
            last = sessions[-1][-1]
            same_request = e.get("request_id") and e.get("request_id") == last.get("request_id")
            close = not e.get("request_id") and _days(last["created_at"], e["created_at"]) * 24 < SESSION_GAP_HOURS
            if same_request or close:
                sessions[-1].append(e)
                continue
        sessions.append([e])
    return sessions


def _ema(readings: list[float | None], weight: float) -> list[float | None]:
    """The running M at each analysis under an EMA of this weight."""
    out, m = [], None
    for r in readings:
        if r is not None:
            m = r if m is None else (1 - weight) * m + weight * r
        out.append(m)
    return out


def records(traces: list[dict], events: list[dict], nodes_by_id: dict[str, dict]) -> list[dict]:
    """One record per analysis: its predictors and the outcomes that followed."""
    by_user_events: dict[str, list[dict]] = {}
    for e in events:
        if e.get("counted") is False or e.get("credit") is None:
            continue
        by_user_events.setdefault(e["user_id"], []).append(e)
    for evs in by_user_events.values():
        evs.sort(key=lambda e: str(e["created_at"]))

    by_user_traces: dict[str, list[dict]] = {}
    for t in traces:
        by_user_traces.setdefault(t["user_id"], []).append(t)

    out = []
    for user_id, user_traces in sorted(by_user_traces.items()):
        user_traces.sort(key=lambda t: str(t["created_at"]))
        observed = [t.get("observed") for t in user_traces]
        emas = {w: _ema(observed, w) for w in EMA_GRID}
        evs = by_user_events.get(user_id, [])
        for i, t in enumerate(user_traces):
            after = [
                e for e in evs
                if e.get("request_id") != t.get("request_id")
                and 0 < _days(t["created_at"], e["created_at"]) <= OUTCOME_WINDOW_DAYS
            ]
            rec = {"user_id": user_id, **{p: t.get(p) for p in PREDICTORS}}
            for w in EMA_GRID:
                rec[f"ema_{w}"] = emas[w][i]
            rec["returned"] = float(any(
                _days(t["created_at"], e["created_at"]) <= RETURN_WINDOW_DAYS for e in after
            ))
            rec["persistence"] = _persistence(after)
            rec["learning_residual"] = _residual(after, nodes_by_id)
            out.append(rec)
    return out


def _persistence(events: list[dict]) -> float | None:
    """Share of autonomous failures followed by another try on that KC in the same session."""
    failures = retried = 0
    for session in _sessions(events):
        for i, e in enumerate(session):
            if e.get("is_assisted") or float(e["credit"]) >= 0.5:
                continue
            failures += 1
            retried += any(x["concept_id"] == e["concept_id"] for x in session[i + 1:])
    return retried / failures if failures else None


def _residual(events: list[dict], nodes_by_id: dict[str, dict]) -> float | None:
    """Mean of credit minus the Kernel's predicted P(correct), autonomous attempts."""
    diffs = []
    for e in events:
        if e.get("is_assisted") or e.get("k_before") is None:
            continue
        params = bkt.get_bkt_params(nodes_by_id.get(e["concept_id"]))
        k = float(e["k_before"])
        predicted = k * (1 - params["p_slip"]) + (1 - k) * params["p_guess"]
        diffs.append(float(e["credit"]) - predicted)
    return sum(diffs) / len(diffs) if diffs else None


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0  # ties share their mean rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else None


def _pairs(recs: list[dict], predictor: str, outcome: str):
    return [(r[predictor], r[outcome], r["user_id"]) for r in recs
            if r.get(predictor) is not None and r.get(outcome) is not None]


def association(recs: list[dict], predictor: str, outcome: str, seed: int = 0) -> dict:
    """Spearman rho with a student-level bootstrap 95% interval."""
    pairs = _pairs(recs, predictor, outcome)
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    users = sorted({p[2] for p in pairs})
    result = {"rho": None if rho is None else round(rho, 3), "n": len(pairs),
              "students": len(users), "interval_95": None}
    if rho is None or len(users) < 2:
        return result
    by_user: dict[str, list] = {}
    for p in pairs:
        by_user.setdefault(p[2], []).append(p)
    # Seeded per (predictor, outcome) so the report is reproducible.
    tag = hashlib.sha256(f"{predictor}|{outcome}|{seed}".encode()).digest()
    rng = random.Random(int.from_bytes(tag[:8], "big"))
    stats = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [p for u in rng.choices(users, k=len(users)) for p in by_user[u]]
        r = spearman([p[0] for p in sample], [p[1] for p in sample])
        if r is not None:
            stats.append(r)
    if stats:
        stats.sort()
        result["interval_95"] = [round(stats[int(0.025 * len(stats))], 3),
                                 round(stats[int(0.975 * len(stats)) - 1], 3)]
    return result


def report(recs: list[dict]) -> dict:
    students = len({r["user_id"] for r in recs})
    out = {"analyses": len(recs), "students": students,
           "min_analyses": MIN_ANALYSES, "min_students": MIN_STUDENTS, "enough_data": False}
    if len(recs) < MIN_ANALYSES or students < MIN_STUDENTS:
        return out
    out["enough_data"] = True
    out["associations"] = {
        p: {o: association(recs, p, o) for o in OUTCOMES} for p in PREDICTORS
    }
    # Smoothing: the mean rho over the three outcomes, per EMA weight.
    ema = {}
    for w in EMA_GRID:
        rhos = [association(recs, f"ema_{w}", o)["rho"] for o in OUTCOMES]
        rhos = [r for r in rhos if r is not None]
        ema[str(w)] = round(sum(rhos) / len(rhos), 3) if rhos else None
    out["ema_weight_mean_rho"] = ema
    scored = {w: v for w, v in ema.items() if v is not None}
    out["best_ema_weight"] = float(max(scored, key=scored.get)) if scored else None
    out["ema_vs_current_95"] = _ema_differences(recs)
    return out


def _mean_rho(recs: list[dict], key: str) -> float | None:
    rhos = []
    for o in OUTCOMES:
        pairs = _pairs(recs, key, o)
        r = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if r is not None:
            rhos.append(r)
    return sum(rhos) / len(rhos) if rhos else None


def _ema_differences(recs: list[dict], seed: int = 0) -> dict[str, list[float] | None]:
    """95% interval of mean-rho(weight w) - mean-rho(current weight), students resampled.

    Paired: every weight is scored on the same resample, so the interval is
    about the DIFFERENCE, which is far tighter than the two rhos' own
    intervals. Only a weight whose interval excludes 0 is a real improvement.
    """
    from core import mindset

    current = f"ema_{mindset.EMA_WEIGHT}"
    if current not in recs[0]:
        return {}
    by_user: dict[str, list[dict]] = {}
    for r in recs:
        by_user.setdefault(r["user_id"], []).append(r)
    users = sorted(by_user)
    rng = random.Random(seed)
    diffs: dict[float, list[float]] = {w: [] for w in EMA_GRID}
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [r for u in rng.choices(users, k=len(users)) for r in by_user[u]]
        base = _mean_rho(sample, current)
        if base is None:
            continue
        for w in EMA_GRID:
            v = _mean_rho(sample, f"ema_{w}")
            if v is not None:
                diffs[w].append(v - base)
    out = {}
    for w, ds in diffs.items():
        if not ds:
            out[str(w)] = None
            continue
        ds.sort()
        out[str(w)] = [round(ds[int(0.025 * len(ds))], 4), round(ds[int(0.975 * len(ds)) - 1], 4)]
    return out


def simulate(rng: random.Random, students: int, sessions: int = 8, noise: float = 0.25):
    """Traces and events from students with a latent, stable mindset g in [0, 1].

    Per-conversation readings are g plus noise, so a smoothed M should predict
    better than one reading. Outcomes depend on g: retry after failure with
    probability 0.3 + 0.5 g, answer 0.15 (g - 0.5) better than the model
    predicts, come back with probability 0.5 + 0.4 g.
    """
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    traces, events = [], []
    for i in range(students):
        g = rng.random()
        uid = f"u{i}"
        t = t0
        for s in range(sessions):
            t += timedelta(days=rng.choice((1, 3, 7)))
            reading = min(1.0, max(0.0, g + rng.gauss(0, noise)))
            req = f"r{i}-{s}"
            if s and rng.random() > 0.5 + 0.4 * g:
                continue  # did not come back for this one
            for a in range(rng.randint(2, 5)):
                k = rng.random()
                predicted = k * 0.9 + (1 - k) * 0.2
                p = min(0.99, max(0.01, predicted + 0.15 * (g - 0.5)))
                credit = 1.0 if rng.random() < p else 0.0
                kc = f"kc{rng.randint(0, 3)}"
                events.append({"user_id": uid, "concept_id": kc, "credit": credit,
                               "is_assisted": False, "counted": True, "k_before": k,
                               "request_id": req, "created_at": (t + timedelta(minutes=a)).isoformat()})
                if credit < 0.5 and rng.random() < 0.3 + 0.5 * g:
                    events.append({"user_id": uid, "concept_id": kc, "credit": 1.0,
                                   "is_assisted": False, "counted": True, "k_before": k,
                                   "request_id": req,
                                   "created_at": (t + timedelta(minutes=a, seconds=30)).isoformat()})
            traces.append({"user_id": uid, "request_id": req,
                           "created_at": (t + timedelta(minutes=10)).isoformat(),
                           "observed": reading, "conversation_linear": reading,
                           "measured_linear": None, "m_score": None})
    # m_score as the Kernel would have stored it, at the current weight.
    by_user: dict[str, list[dict]] = {}
    for tr in traces:
        by_user.setdefault(tr["user_id"], []).append(tr)
    from core import mindset

    for trs in by_user.values():
        for tr, m in zip(trs, _ema([tr["observed"] for tr in trs], mindset.EMA_WEIGHT)):
            tr["m_score"] = m
    return traces, events
