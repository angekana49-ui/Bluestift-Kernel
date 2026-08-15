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
- 9 migrations, 45 tests, deployed on Railway.

### Shipped beyond the original v1 spec

- **Multi-subject** — the same builder produces a coherent graph for any subject
  (MATH + PHYSICS graphs live); extraction is grounded on the all-subject vocabulary.
- **Cross-subject bridges (basic)** — `scripts/build_bridges.py` generates
  prerequisite edges *between* subjects (foundational → applied); the detector
  traverses them with no change. Verified: a physics conversation traces its root
  gap into maths. Refinement still needed — see §4.

What follows is deliberately **not** in v1 (or only basic).

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

## 3. Finish the anomaly / monitoring layer — **shipped**

The two deferred detectors are in, both reading beyond a single conversation:

- **`inconsistency_high`** — temporal inconsistency (Hooshyar) over the last 20
  `learning_trajectories` snapshots, alongside `volatility_score`. Inconsistency
  is total variation vs net displacement: a monotonic climb scores 0, an estimate
  that swings and ends where it began scores 1. Fires above 0.40. The corpus
  specifies a 20-interaction window; we compute from 6 snapshots up over whatever
  part of the window exists, and carry `interactions_count` in the alert — waiting
  for a full 20 would leave the first cohort unprotected.
- **`ood_distribution`** — compares the student against the *local* population
  baseline each KC accumulates (`empirical_difficulty`), not against the imported
  priors. A large signed mean deviation across 3+ calibrated KCs means the
  parameters don't describe this student. Direction is reported:
  `below_population` is the silent-failure case (Goodhart, Amodei) and is raised
  at high severity. KCs that haven't been calibrated carry no baseline and are
  skipped — the neutral 0.5 placeholder a new KC is created with would otherwise
  manufacture divergence out of nothing.
- The selective-update gate's `anomalous` flag now also fires on an unstable
  history, not only on failing a KC that looked mastered: when the estimate is
  already oscillating, holding an update back keeps an unreliable value on the
  books, so that is precisely where fresh evidence should count.
- Alerts write `inconsistency_rate`, `volatility_score` and `interactions_count`
  into their own `kernel_monitoring` columns (migration 008 already defined them),
  so the dashboard can filter and chart without parsing the details JSON.

Still open here:

- **Trigger local recalibration** once a population's N passes threshold, instead
  of only flagging the divergence.
- Calibrate the thresholds (0.40 inconsistency, 0.4 OOD deviation, the 6-snapshot
  floor) against real outcomes — they are reasoned defaults, not measured ones.

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
- **Cross-subject refinement** — a basic version ships (see "Shipped beyond v1").
  Next: fold bridge generation into the main build, validate bridges with teachers,
  tune root-stop depth so a cross-subject chain stops at the meaningful root
  (e.g. `derivation_fonction`) instead of the most elementary leaf, and avoid weak
  generic links (e.g. "multiplication" pulled in everywhere).

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

1. ~~**RAYA integration**~~ — done: RAYA calls `/analyze` (chat, challenges,
   assignments), reacts to alerts, and sends graded attempts to
   `/update_concept_state` with their real partial credit.
2. ~~**Finish anomaly layer (OOD, inconsistency)**~~ — done (§3); thresholds still
   want calibrating against real outcomes.
3. **School channel + dashboard** — the differentiator, and what institutions buy.
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
