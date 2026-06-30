# Bluestift Cognitive Kernel — Post-MVP Roadmap

> What the v1 Kernel covers, and everything that comes after it.
> v1 is complete and deployed; this document is the forward plan.

---

## Where v1 stands

The pure Kernel is **~98% of the v1 spec** (master prompt + DB Patch v2 + corpus).
Shipped and verified live:

- 5 API routes; LLM chain (Groq `openai/gpt-oss-120b` → Gemini `gemini-3.1-flash-lite`).
- Cognitive vector **K, V, P, M** with canonical semantics (V = individualised
  p(T); P = 1 − p(S) modulated by M).
- BKT (asymmetric, partial-credit, dual mastery criterion), exponential
  forgetting with 3-level lambda, personal-lambda calibration, empirical KC
  recalibration.
- Selective-update gate, learning-trajectory snapshots.
- Root-cause detection by **convergence** over a dense, canonical, auto-generated
  prerequisite graph (LLM graph builder with cross-model corroboration + DAG
  validation).
- Dynamic KCs for any subject; runtime label canonicalization.
- **Pedagogical-safety anomaly detection** → `kernel_monitoring` + `/analyze` alerts.
- 8 migrations, 30 tests, deployed on Railway.

What follows is deliberately **not** in v1.

---

## 1. RAYA integration (highest priority)

The Kernel is the brain; RAYA is the voice. The flywheel only turns once they
talk. The contract already exists — this is wiring, not new kernel work.

- **`POST /analyze`** after each conversation: RAYA sends `conversation_history`,
  receives `root_gap`, `detection_path`, `recommended_path`, `summary`, and
  `alerts`. RAYA acts on alerts (e.g. `passive_dependency` → switch to a
  goal-free / vicarious mode; `fixed_mindset` → mindset intervention before any
  retry).
- **`POST /update_concept_state`** on strong, structured signals (a graded
  attempt) so the Kernel updates outside the post-conversation batch.
- **`POST /load_profile`** to read the student's cognitive state for the dynamic
  prompt layer (inject K, V, P, M + curriculum anchor).
- Enrich `learning.messages` (RAYA-side writes): `blocage_type`,
  `langue_interaction`, `partial_credit_score`, `is_assisted`, `concept_id`,
  `emt_level`. The Kernel reads these as the raw signal; RAYA produces them.
- Decide sync vs async: `/analyze` is the heavy call; keep it post-conversation,
  not per-message.

**Done when:** a real student conversation in RAYA produces a Kernel analysis,
updates state, and changes RAYA's next move.

---

## 2. School → AI → Student channel

The corpus calls this the strongest **differentiator** (§4.1): the school does
not just read reports — it actively calibrates the Kernel and RAYA. The
`schools.school_curriculum_layers` table exists but no logic consumes it.

- Ingest active layers into the analysis context:
  - `curriculum` → constrain/seed the KC graph to the national program (MINESEC).
  - `kc_priorities` → weight multipliers feeding the sequencing decision.
  - `objectives` → mastery targets + deadlines surfaced on the dashboard.
  - `custom_rules` → instructions injected into RAYA's prompt (layers 3/4).
- Teacher override of Kernel inferences (scalable oversight, Amodei) — validate a
  sample, extrapolate the rest.
- An institutional dashboard exposing K/V/P/M per student per KC, the
  `kernel_monitoring` alerts, behavioural/dropout risk, and equity (K
  distribution per KC). Pitch line: *"audit what RAYA tells your students."*

---

## 3. Finish the anomaly / monitoring layer

Two detectors were deferred because they need population baselines or history:

- **`ood_distribution`** — distributional shift detection. All KT priors come
  from North-American/Estonian data; a sub-Saharan deployment risks silent
  failure (Goodhart, Amodei). Flag when a student's patterns diverge from the
  calibrated distribution; trigger local recalibration once N > threshold.
- **`inconsistency_high`** — temporal inconsistency > 0.40 over 20 interactions
  (Hooshyar), plus `volatility_score`. Compute from `learning_trajectories`
  once enough snapshots exist; this is the stability metric for the dashboard.
- Wire richer anomaly signals back into the selective-update gate's `anomalous`
  flag (currently a simple high-mastery-failure heuristic).

---

## 4. Detection-quality tuning (needs real data)

- **Confidence calibration** — the current confidence blend (convergence + depth
  + severity) is heuristic; calibrate against observed remediation outcomes.
- **Root-selection refinement** — when convergence ties, the longest-chain
  tiebreak can pick a branch over the most foundational gap; refine with data.
- **Graph quality** — reduce residual over-granularity and near-duplicate KCs the
  fuzzy/LLM dedup misses; make cross-model corroboration reliable (Gemini
  free-tier 429s currently make it intermittent — add quota-aware backoff or a
  third provider).
- **Per-population parameters** (flywheel level 3) — BKT priors and lambda that
  vary by school level × mindset × interaction language.

---

## 5. Responsible-DKT (the architectural evolution)

v1 is Bayesian BKT + LLM heuristics. The corpus targets a **hybrid
neural-symbolic** model (Hooshyar 2026, Responsible-DKT): DKT backbone +
injected symbolic rules (`mastered` / `not_mastered` / `avg_embed`), AUC ~0.90,
temporally stable, interpretable, works with ~10% of the training data.

- Cold-start vs warm modes (Baker): transfer/prior for trials 1–2, individual
  tracking from trial 3+.
- Latent inter-skill relations bootstrapped from interaction data (DKVMN J_ij).
- Requires accumulated local data first — this is a later, data-gated upgrade.

---

## 6. GraphRAG + content pipeline

- **GraphRAG** (multi-hop) over the content graph so RAYA can answer
  "which prerequisites of this KC has the student not yet mastered?" — impossible
  with vector RAG alone. The `rag.conversation_embeddings` table + pgvector are
  already provisioned.
- Content-graph construction pipeline: ontology (national curriculum schema) →
  LLM/RAG extraction from textbooks → equivalence fusion → teacher validation
  (human-in-the-loop) → prerequisite bootstrap.

---

## 7. Operational hardening

- **GitHub auto-deploy** — connect the repo in the Railway dashboard so pushes
  redeploy (current deploys are manual `railway up`).
- **Auth** — JWT on the Kernel (deferred in v1; today it trusts the caller).
- **Monitoring/alerting dashboard** on top of `kernel_monitoring`.
- **Offline-first** — sub-Saharan connectivity: deferred sync, state-conflict
  resolution (not addressed in the corpus; design needed).
- **Multilingual KT** — FR/EN (and local languages); impact on cognitive
  modelling is undocumented and needs design.
- **Learner simulator** (GenMentor-style) — synthetic bootstrap before the first
  real cohort, if cold-start data proves too slow.

---

## Suggested order

1. **RAYA integration** — turn on the flywheel (nothing else matters without it).
2. **School channel + dashboard** — the differentiator, and what institutions buy.
3. **Finish anomaly layer (OOD, inconsistency)** — pedagogical-safety story.
4. **Data-gated work** — confidence calibration, per-population params,
   Responsible-DKT — once real interactions accumulate.
5. **GraphRAG, offline, multilingual, auth** — as scale and context demand.

---

## Open tensions (from the corpus, unresolved by design)

- Interpretability vs raw performance (Responsible-DKT vs SAKT/AKT/SAINT+).
- Algorithmic guardrail generation at scale (no two half-time teachers per class).
- Selective-update threshold not empirically set for K-12.
- M as a quantifiable vector — a design decision, not a literature result.
- MDP sequencing vs a hard national-curriculum constraint.
