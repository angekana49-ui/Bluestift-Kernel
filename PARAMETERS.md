# Kernel parameters — where every constant comes from

Every number the Kernel reasons with is listed here with its **provenance**:

- **prior** — taken from the literature, with its source. A starting point, not a
  truth about our students: these were measured on (mostly North-American)
  intelligent-tutor datasets.
- **design** — our choice. A hypothesis, stated as one. Each says what would
  replace it.
- **fitted** — estimated on our own students' data. Only the BKT parameters are
  fitted today (see [Calibration](#calibration--how-a-prior-becomes-a-fitted-value)).

The rule: **a prior or a design constant is a placeholder until data replaces it.**
None of these values should be defended as "the right one". They should be
defended as "a sensible starting point, and here is how it gets corrected".

When a constant changes, check its effect with the synthetic benchmark
(`python scripts/eval_kernel.py --sessions 3`) and record the before/after in the
commit message.

---

## Knowledge tracing — `core/bkt.py`

| Constant | Value | Provenance | Replaced by |
|---|---|---|---|
| `BKT_PRIORS.p_init` | 0.3 | prior — typical fitted value, Cognitive Tutor / ASSISTments (Corbett & Anderson 1995; Baker, Corbett & Aleven 2008) | **fitted** per KC by `fit_bkt_em` |
| `BKT_PRIORS.p_transit` | 0.1 | prior — same | **fitted** per KC |
| `BKT_PRIORS.p_slip` | 0.1 | prior — same | **fitted** per KC |
| `BKT_PRIORS.p_guess` | 0.2 | prior — same | **fitted** per KC |
| `MAX_SLIP_OR_GUESS` | 0.45 | design — keeps the model identifiable. At slip + guess ≥ 1 a correct answer becomes evidence of *not* knowing (Baker, Corbett & Aleven 2008 discuss degenerate fits). | stays: a safety bound, not an estimate |
| `ASSISTED_GUESS` | 0.5 | design — with help, an unlearned student succeeds about half the time. Motivated by Bastani et al. 2024 (assisted practice barely predicts unassisted exam results); the value itself is ours. | a separate fit of the guess rate on assisted attempts once `learning_events` hold enough of them |
| `blocage_evidence` ambiguous weight | ½ (credit pulled halfway to 0.5) | design — 0.5 is exactly the uninformative credit (both likelihoods equal) | compare the outcomes of ambiguous vs conceptual failures in `learning_events` |
| `MASTERY_THRESHOLD` | 0.95 | prior — Corbett & Anderson 1995 mastery criterion | stays unless validated otherwise on K-12 |
| `MAX_SLIP_FOR_MASTERY` | 0.15 | design — no published value for K-12 | link to held-out performance once available |
| `MIN_PARTIAL_CREDIT_AVG` | 0.7 | design | same |
| `RECENT_AUTONOMOUS_WINDOW` | 3 | design — "the last 3 autonomous attempts" (from the Bluestift corpus condensate) | same |
| `GAP_THRESHOLD` | 0.4 | design — display label only ("gap" vs "partial") | — |
| `PARTIAL_CREDIT_BINS` | 0, 0.3, 0.6, 0.7, 0.8, 1.0 | prior, **to verify** — the grading scale reported by graders (Ostrow & Heffernan 2015). The update takes any credit in [0, 1] and does not snap to it. | — |

**Model choices, not constants:**

- **Partial credit as soft evidence.** A credit *c* is read as "correct with weight *c*, incorrect with weight 1 − *c*". *c* = 0 and *c* = 1 are exactly classic BKT.
- **Every attempt updates K, in order.** There is no "strong signal" gate on the *state*. Drift comes from re-estimating *parameters* on thin data, and that is guarded by the calibration gates below.
- **No extra asymmetry.** BKT is already asymmetric through slip < guess: from K = 0.5, a failure moves K by −0.39 and a success by +0.32. Adding a rule "errors weigh more" on top would count the same thing twice.

## Forgetting — `core/forgetting.py`

K_effective = floor + (K_raw − floor) · e^(−λ·Δt), where floor = the KC's p_init. A K below the floor is left alone.

| Constant | Value | Provenance | Replaced by |
|---|---|---|---|
| `LAMBDA_PRIORS.procedural` | 0.01 /day (half-life ≈ 69 days) | design — the ordering "facts fade faster than procedures" follows the retention literature; the values are ours | KC `lambda_decay` from the average of personal λ (`compute_empirical_kc_params`) |
| `LAMBDA_PRIORS.conceptual` | 0.02 /day (≈ 35 days) | design | same |
| `LAMBDA_PRIORS.declarative` | 0.05 /day (≈ 14 days) | design | same |
| `MIN_INTERACTIONS_FOR_EMPIRICAL` | 10 | design | — |

Known limit: a constant λ ignores the spacing effect (each review slows forgetting). The reference model for that is half-life regression (Settles & Meeder 2016). It is the target once the data exists.

## Calibration — `core/calibration.py`

| Constant | Value | Provenance |
|---|---|---|
| `LAMBDA_MIN`, `LAMBDA_MAX` | 0.001, 0.2 /day | design — bounds on a personal λ |
| `LAMBDA_STEP` | 0.5 | design — step of the online maximum-likelihood update of a personal λ. One unexpected failure a month after a solid session raises λ by ~20%, one success lowers it by ~10%. |
| `MIN_STUDENTS_FOR_KC_CALIBRATION` | 10 | design |
| `MIN_INTERACTIONS_PER_STUDENT` | 5 | design |
| `CALIBRATION_COOLDOWN_HOURS` | 6 | design — cost bound |
| `MIN_SEQUENCES_FOR_BKT_FIT` | 30 students | design |
| `MIN_EVENTS_FOR_BKT_FIT` | 150 attempts | design |
| `BKT_PRIOR_STRENGTH` | 10 pseudo-observations | design — how hard the fit is pulled towards the literature priors |

### Calibration — how a prior becomes a fitted value

1. Every graded attempt is logged in `kernel.learning_events` (migration 011): the credit, whether it was assisted, the blocage type, and whether it counted as evidence.
2. After an interaction, once the KC's cooldown has passed, the background recalibration runs `fit_bkt_em`. This is a MAP Baum-Welch fit of p_init, p_transit, p_slip and p_guess on that KC's autonomous, counted attempts, per student and in time order.
3. The fit always shrinks towards the **literature** priors, never towards the previous fit. With little data it stays close to the prior; the prior's pull fades as evidence accumulates.
4. The fitted values are written to `concept_nodes`, and `get_bkt_params` uses them from then on, bounded by `MAX_SLIP_OR_GUESS`.

`test_fit_bkt_em_recovers_known_parameters` checks that the fit recovers the parameters of a simulated population that differs from the priors.

## Root-gap detection — `core/detector.py`

| Constant | Value | Provenance |
|---|---|---|
| `FAILING_THRESHOLD` | 0.5 | design — the cap. A KC is **failing** below `min(0.5, its p_init)`, i.e. when the evidence has moved the posterior below the prior. |
| `UNKNOWN_BUDGET` | 2 | design — how many never-practised prerequisites in a row the search may cross |
| confidence weights | 0.4 convergence, 0.3 depth, 0.3 severity | design — **not a calibrated probability**. Do not present `confidence` as one. |

## Mindset — `core/mindset.py`

| Constant | Value | Provenance |
|---|---|---|
| `W_ABANDON` (on 1 − abandon), `W_PERSISTENCE`, `W_TIME`, `W_QUALITY` | 0.35, 0.30, 0.20, 0.15 | design — from the Bluestift corpus condensate. The signals are read by the extraction LLM from the text; it cannot measure time on task. |
| `GAIN` | 6.0 | design — spreads the score over [~0.05, ~0.95] |
| `EMA_WEIGHT` | 0.3 | design — symmetric. The condensate's "degrades fast, rebuilds slowly" is a hypothesis, not implemented. |
| `M_FLOOR`, `M_CEIL` | 0.05, 0.95 | design |
| labels | growth ≥ 0.66, fixed ≤ 0.40 | design |

Caveat: mindset interventions have small average effects (Sisk et al. 2018, d ≈ 0.08). M is a behavioural proxy, not a validated measure of Dweck's construct.

## Cognitive vector heuristics — `services/analyze.py`

| Quantity | Formula | Provenance |
|---|---|---|
| personal slip | EMA 0.8 / 0.2 of "failed while K ≥ 0.7" | design — a heuristic, not the BKT slip |
| V (velocity) | EMA 0.7 / 0.3 of the share of the mastery gap closed per attempt | design — inspired by individualised p(T) (Yudelson, Koedinger & Gordon 2013). **Stored, not used in inference.** |
| P (persistence) | (1 − personal slip) × (0.6 + 0.4·M) | design — our construction, not Corbett's. **Stored, not used in inference.** |

## Anomaly detectors — `core/anomaly.py`

All design, and every one is to be calibrated on real alert outcomes:

- false mastery: K ≥ 0.9 and slip > 0.15
- fixed mindset: M ≤ 0.4
- cognitive overload: failure rate ≥ 0.5 and conceptual block
- passive dependency: assisted ratio ≥ 0.5
- re-emergence: prerequisites ≥ 0.7 and KC < 0.4
- inconsistency > 0.40 over up to 20 snapshots, from 6 onwards (the 0.40 and the 20-interaction window come from Hooshyar, to verify)
- OOD deviation 0.4, from 3 KCs onwards

## Cost bounds

| Constant | Value | Where |
|---|---|---|
| `MAX_DEPTH` | 3 | `services/kc_registry.py` — prerequisite recursion |
| `MAX_LLM_CALLS_PER_REQUEST` | 6 | `services/kc_registry.py` — KC inference per request |
| rate limits | 30 per student per hour, 300 per hour overall | `core/ratelimit.py` |

## Declared but not used

- **τ (`tau`)** is stored per KC. The condensate says it should modulate the mastery threshold, the update speed, λ and the EMT entry point. None of that is implemented. Define it before using it.
