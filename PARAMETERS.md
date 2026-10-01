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
| `MASTERY_TAU_SLOPE` | 0.10 | design — rigour moves the mastery threshold: 0.93 at τ 0.3, 0.95 at 0.5, 0.98 at 0.8 | outcomes on held-out checks |
| `GAP_THRESHOLD` | 0.4 | design — display label only ("gap" vs "partial") | — |
| `PARTIAL_CREDIT_BINS` | 0, 0.3, 0.6, 0.7, 0.8, 1.0 | prior, **to verify** — the grading scale reported by graders (Ostrow & Heffernan 2015). The update takes any credit in [0, 1] and does not snap to it. | — |

### τ — the KC's rigour

τ ∈ [0, 1] says how **exact** a KC must be to count as known. It is inferred with the KC from its type and target level, and 0.5 is neutral. **At τ = 0.5 every rule reduces exactly to the rule without τ.** τ only acts on rules we own, never on a parameter fitted to data:

| Rule | Formula | At τ 0.3 / 0.5 / 0.8 |
|---|---|---|
| What a partial answer is worth (`rigor_credit`) | credit^(2τ) | a 0.7 answer counts 0.81 / 0.70 / 0.56 |
| Mastery threshold (`mastery_threshold`) | 0.95 + 0.10·(τ − 0.5) | 0.93 / 0.95 / 0.98 |
| Forgetting prior (`forgetting.base_lambda`, level 3 only) | λ_type · (1 + (τ − 0.5)) | ×0.8 / ×1 / ×1.3 |
| EMT entry point | exposed in `/load_profile` for RAYA | — |

All [design]. Full success and full failure are never changed by τ.

**Model choices, not constants:**

- **Partial credit as soft evidence.** A credit *c* is read as "correct with weight *c*, incorrect with weight 1 − *c*". *c* = 0 and *c* = 1 are exactly classic BKT.
- **Every attempt updates K, in order.** There is no "strong signal" gate on the *state*. Drift comes from re-estimating *parameters* on thin data, and that is guarded by the calibration gates below.
- **No extra asymmetry.** BKT is already asymmetric through slip < guess: from K = 0.5, a failure moves K by −0.39 and a success by +0.32. Adding a rule "errors weigh more" on top would count the same thing twice.

## Forgetting — `core/forgetting.py`

K_effective = floor + (K_raw − floor) · e^(−λ·Δt), where floor = the KC's p_init. A K below the floor is left alone.

Forgetting depends on **days and reviews**: λ = base_λ · 2^(−max(0, 0.5·reviews − 0.5·lapses)). A *review* is a successful autonomous first attempt after ≥ 1 day without practice; a *lapse* is a failed one. Each review multiplies the half-life by √2, and a lapse cancels a review. Lapses never make a student forget faster than their base rate; the personal λ covers that case. This is the counts-based form of half-life regression (Settles & Meeder 2016); the spacing effect itself goes back to Ebbinghaus (see Cepeda et al. 2006).

| Constant | Value | Provenance | Replaced by |
|---|---|---|---|
| `LAMBDA_PRIORS.procedural` | 0.01 /day (half-life ≈ 69 days) | design — the ordering "facts fade faster than procedures" follows the retention literature; the values are ours | KC `lambda_decay` from the average of personal λ (`compute_empirical_kc_params`) |
| `LAMBDA_PRIORS.conceptual` | 0.02 /day (≈ 35 days) | design | same |
| `LAMBDA_PRIORS.declarative` | 0.05 /day (≈ 14 days) | design | same |
| `MIN_INTERACTIONS_FOR_EMPIRICAL` | 10 | design | — |
| `REVIEW_GAIN`, `LAPSE_PENALTY` | 0.5, 0.5 half-lives | design | half-life regression weights fitted on `learning_events` |
| `REVIEW_MIN_GAP_DAYS` | 1 | design — a retrieval the same day tests short-term memory, not retention | — |
| `TAU_LAMBDA_SLOPE` | 1.0 | design — rigour scales the forgetting prior | — |

The personal λ is stored as a **base** rate (reviews are applied on top of it, never baked in).

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

## Active probing — `core/probe.py`

| Constant | Value | Provenance | Replaced by |
|---|---|---|---|
| `PROBE_MAX_HOPS` | 4 | design — same depth as `/prerequisite_gaps` | — |
| `PROBE_MAX_HYPOTHESES` | 60 | design — cost bound (hypotheses × questions × evidence) | — |
| `PROPAGATION` | 0.85 | design — under "the gap is h", a concept built on h is unlearned with this probability, any other learned with it. 1 would assume a single gap and perfect propagation; the benchmark's generator does, real students don't. | fit on `learning_events` once probes have answers: the rate at which a failed prerequisite's dependents are also failed |
| `PROBE_MIN_GAIN` | 0.05 bit | design — below it a question is not worth the student's time | — |
| prior over the gap's location | uniform, "no gap" weighted as one concept | design — no concept is assumed a likelier gap than another | the observed distribution of confirmed roots |

The evidence on a practised concept enters as the likelihood ratio
`odds(K) / odds(p_init)`: how far the student's record moved that concept's
belief from its prior. A question is chosen by maximum expected information
about the gap's location, and asked only if its two answers lead the detector to
different roots.

## Mindset — `core/mindset.py`

| Constant | Value | Provenance |
|---|---|---|
| `W_ABANDON` (on 1 − abandon), `W_PERSISTENCE`, `W_TIME`, `W_QUALITY` | 0.35, 0.30, 0.20, 0.15 | design — the corpus formula. **Abandon after an error is measured** in the conversation (was a failure followed by another try on the same KC?) whenever there was an error; the other three are read by the extraction LLM, which cannot measure time on task. |
| `W_MEASURED` | learning speed 0.30, progression 0.30, recovery after error 0.20, stability (1 − slip) 0.20 | design — the Kernel's own traces, renormalised over the ones available |
| `MEASURED_HALF_WEIGHT_AT`, `MAX_MEASURED_SHARE` | 5 data points, 0.5 | design — the measured share grows with its evidence; the conversation always keeps half |
| `GAIN` | 6.0 | design — spreads the score over [~0.05, ~0.95] |
| `EMA_WEIGHT` | 0.3 | design — symmetric. The condensate's "degrades fast, rebuilds slowly" is a hypothesis, not implemented. |
| `M_FLOOR`, `M_CEIL` | 0.05, 0.95 | design |
| labels | growth ≥ 0.66, fixed ≤ 0.40 | design |

**Not inputs of M, on purpose: the level K and the forgetting rate.** They measure ability and memory; feeding them into M would label a weak or forgetful student "fixed mindset" for their level, the stereotype-threat mechanism the corpus warns about. M reads *dynamics* (how the student reacts to difficulty, whether they improve), never *level*.

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

## State chain — `services/serial.py`, migration 012

| Constant | Value | Provenance |
|---|---|---|
| `MAX_CHAIN_RETRIES` | 3 | design — recomputations of a write rejected as stale before giving up (the evidence stays logged) |

Requests for one student are served first come, first served (a FIFO queue per student in the process), and every state write is the next block of its chain: accepted only from the `version` it was read at.

## Cost bounds

| Constant | Value | Where |
|---|---|---|
| `MAX_DEPTH` | 3 | `services/kc_registry.py` — prerequisite recursion |
| `MAX_LLM_CALLS_PER_REQUEST` | 6 | `services/kc_registry.py` — KC inference per request |
| rate limits | 30 per student per hour, 300 per hour overall | `core/ratelimit.py` |

