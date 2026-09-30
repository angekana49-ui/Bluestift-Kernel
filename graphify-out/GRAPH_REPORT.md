# Graph Report - .  (2026-10-01)

## Corpus Check
- 33 files · ~59,181 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 898 nodes · 1720 edges · 97 communities (69 shown, 28 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 46 edges (avg confidence: 0.69)
- Token cost: 111,487 input · 0 output

## Community Hubs (Navigation)
- Test Fake Supabase
- Mindset M (measured)
- Calibration & EM Fit
- Test Suite
- Root-Gap Detector
- Anomaly Detectors
- LLM Chain
- School Curriculum Layers
- BKT Core & tau
- FastAPI App & Auth
- API Schemas
- Shared DB & Migrations
- Effective State & Routes
- Alert Access Tests
- Prerequisite Walk (GraphRAG)
- DB Reads
- Evidence Update Rules
- README & Dependencies
- Offline Graph Builder
- Corpus Errata
- RAYA Handoff Contract
- Forgetting & Spacing
- Analyze Orchestration
- School Channel Roadmap
- Safety Alerts & Cognitive Vector
- Moat & Detection Concepts
- DB Access Layer
- Rate Limiting
- Cognitive Vector Heuristics
- DB Logging
- Pitch & Literature
- Extraction & Observations
- School Data Reads
- Request Validators
- Graph Build Script
- RAG Scoping
- Inconsistency & Monitoring
- Prod Check Script
- Learning Events Logging
- Vocabulary Generation
- GraphRAG Tests
- Sliding Window Limiter
- State Chain (CAS)
- Core Tables Migration
- Cross-Subject Bridges
- Fake HTTP Client
- BKT Parameters
- Test Fixtures
- Patch v2 Tables
- Chain Commit & Session
- Vocabulary Budget
- Boot & Paging Tests
- Math Seed
- Learning Events Table
- Trajectory Grouping
- Population Baseline
- _group_trajectories()
- _population_baseline()
- kernel.student_concept_state
- kernel.concept_edges
- kernel.individual_insights
- kernel.learning_trajectories
- kernel.student_concept_state
- kernel.student_mindset_state
- kernel.kernel_monitoring
- kernel.kernel_outputs
- kernel.kernel_requests
- schools.school_curriculum_layers
- kernel.concept_edges
- kernel.concept_nodes
- kernel.individual_insights
- kernel.kernel_monitoring
- kernel.kernel_outputs
- kernel.kernel_requests
- kernel.learning_trajectories
- kernel.student_concept_state
- kernel.student_mindset_state
- schools.school_curriculum_layers
- kernel.kernel_monitoring
- kernel.student_concept_state
- kernel.student_concept_state
- on_event
- rag.conversation_embeddings
- Kernel Design Rules
- schools.school_curriculum_layers

## God Nodes (most connected - your core abstractions)
1. `run_analysis()` - 45 edges
2. `_kernel()` - 38 edges
3. `Principal` - 25 edges
4. `update_concept_state()` - 24 edges
5. `Kernel to RAYA Handoff` - 24 edges
6. `Condensat Errata` - 23 edges
7. `get_or_create_kc()` - 21 edges
8. `_Query` - 19 edges
9. `Bluestift Kernel README` - 17 edges
10. `Post-MVP Roadmap` - 17 edges

## Surprising Connections (you probably didn't know these)
- `GraphRAG Content Pipeline` --semantically_similar_to--> `Cold-start LLM graph distillation`  [INFERRED] [semantically similar]
  POST_MVP_ROADMAP.md → README.md
- `Dynamic KCs (open graph)` --references--> `get_or_create_kc()`  [EXTRACTED]
  README.md → services/kc_registry.py
- `Responsible-DKT (neural-symbolic)` --semantically_similar_to--> `Bayesian Knowledge Tracing`  [INFERRED] [semantically similar]
  POST_MVP_ROADMAP.md → README.md
- `Data Flywheel (priors -> self-calibration)` --semantically_similar_to--> `Living parameters`  [INFERRED] [semantically similar]
  ARCHITECTURE_DIAGRAMS.md → README.md
- `State chain (FIFO per student + versioned writes)` --references--> `commit_student_concept_state()`  [INFERRED]
  PARAMETERS.md → services/db.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **/analyze pipeline: extract, KCs, forget, BKT, anomalies, root-cause** — architecture_diagrams_analyze_pipeline, services_kc_registry_get_or_create_kc, core_forgetting, core_bkt, core_anomaly, core_detector, services_llm [EXTRACTED 1.00]
- **Living parameters flywheel: priors -> learning_events -> EM fit** — parameters_bkt_priors, readme_learning_events, parameters_bkt_em_calibration, core_calibration_fit_bkt_em, architecture_diagrams_flywheel [INFERRED 0.85]
- **tau modulates credit, mastery threshold and forgetting prior** — parameters_tau, core_bkt_rigor_credit, core_bkt_mastery_threshold, core_forgetting_base_lambda [EXTRACTED 1.00]

## Communities (97 total, 28 thin omitted)

### Community 0 - "Test Fake Supabase"
Cohesion: 0.06
Nodes (34): _clause(), fake_supabase(), FakeSupabase, _like_to_regex(), _or_clause_matches(), _parse_or(), fixture, _Query (+26 more)

### Community 1 - "Mindset M (measured)"
Cohesion: 0.07
Nodes (40): M formula correction, blend_mindset(), classify_mindset(), combine(), compute_mindset_score(), conversation_linear(), measured_linear(), Mindset score M. M = sigmoid(GAIN * (linear - 0.5)), where `linear` blends two… (+32 more)

### Community 2 - "Calibration & EM Fit"
Cohesion: 0.07
Nodes (36): Removed selective-update gate, compute_empirical_kc_params(), _emissions(), fit_bkt_em(), is_calibration_due(), datetime, Self-calibration of living parameters. The Kernel starts on literature priors…, P(observation | learned), P(observation | not learned) for credit c. The same… (+28 more)

### Community 3 - "Test Suite"
Cohesion: 0.07
Nodes (22): _analyze(), _extraction_llm(), Tests for the Bluestift Cognitive Kernel. Covers: - pure algorithm units (BKT,…, The one failure mode a safety dashboard must not have. Every other read in…, Nothing may keep the process busy between requests. Railway sleeps the service…, Subject coverage was capped by a closed enum in the prompt. It offered MATH |…, An LLM mock whose extraction call returns `extraction`; everything else (KC…, test_a_failed_read_is_a_503_never_an_empty_all_clear() (+14 more)

### Community 4 - "Root-Gap Detector"
Cohesion: 0.08
Nodes (34): _compute_confidence(), detect_root_cause(), _dfs_find_root(), failing_threshold(), _path_to_root(), DiGraph, Root-cause detection via DFS over the Kernel Graph. Given a set of failing KCs,…, Build a surface -> root chain for the chosen root. Prefers a DFS chain that… (+26 more)

### Community 5 - "Anomaly Detectors"
Cohesion: 0.11
Nodes (29): _alert(), detect_anomalies(), detect_cognitive_overload(), detect_false_mastery(), detect_fixed_mindset(), detect_inconsistency_high(), detect_ood_distribution(), detect_passive_dependency() (+21 more)

### Community 6 - "LLM Chain"
Cohesion: 0.11
Nodes (23): asyncio, google-generativeai deliberately excluded, call_gemini(), call_groq(), _gemini_generate(), _get_groq(), llm_call(), LLM chain with automatic fallback: Groq (primary) -> Gemini (fallback). The… (+15 more)

### Community 7 - "School Curriculum Layers"
Cohesion: 0.12
Nodes (22): _as_list(), _clamp_priority(), CurriculumLayers, normalize(), objective_report(), _parse_due(), parse_layers(), _parse_target() (+14 more)

### Community 8 - "BKT Core & tau"
Cohesion: 0.12
Nodes (23): classify_status(), is_mastered(), kc_tau(), mastery_threshold(), observation(), push_autonomous_credit(), Bayesian Knowledge Tracing (BKT). Posterior update of a learner's mastery…, The KC's rigour, bounded to [0, 1]; neutral when missing or unreadable. (+15 more)

### Community 9 - "FastAPI App & Auth"
Cohesion: 0.15
Nodes (23): datetime, FastAPI, get, JSONResponse, authenticate(), _bearer(), health(), lifespan() (+15 more)

### Community 10 - "API Schemas"
Cohesion: 0.15
Nodes (22): BaseModel, Enum, AlertOut, AnalyzeRequest, AnalyzeResponse, BlocageType, ConceptStateOut, FrontierItemOut (+14 more)

### Community 11 - "Shared DB & Migrations"
Cohesion: 0.13
Nodes (19): Client, Graceful Degradation, /ready Deep Health Probe, Shared-DB Rules (exposed schemas union), State chain (FIFO per student + versioned writes), GraphRAG Content Pipeline, Migration 011 learning_events, Migration 012 state chain (+11 more)

### Community 12 - "Effective State & Routes"
Cohesion: 0.19
Nodes (20): BackgroundTasks, effective_state(), mastery_credit(), The credit figure the dual condition reads for a student's KC state. The mean…, One KC's current standing for one student: decayed mastery and status. The same…, credit_sequences(), Group learning_events rows (already ordered by time) into per-student credit…, analyze() (+12 more)

### Community 13 - "Alert Access Tests"
Cohesion: 0.12
Nodes (20): Two students' alerts, one already resolved, plus an operational log row., `m1` is an info log, not an alert — /resolve_alert must not rewrite it., A teacher sees their assigned classes — never a whole establishment., Mint a Supabase-shaped access token for a student., No SUPABASE_JWT_SECRET: the Kernel simply doesn't accept user tokens., _seed_alerts(), test_a_student_cannot_close_the_alert_raised_about_them(), test_a_student_reads_only_their_own_alerts() (+12 more)

### Community 14 - "Prerequisite Walk (GraphRAG)"
Cohesion: 0.15
Nodes (18): gap_report(), prerequisite_frontier(), DiGraph, Multi-hop prerequisite reasoning — the graph half of GraphRAG. The question…, The full answer: what this concept rests on, and what is missing. `gaps` are…, Walk prerequisites breadth-first, stopping at mastered concepts. Returns every…, Order concepts so nothing is taught before what it rests on. Topological over…, teaching_order() (+10 more)

### Community 15 - "DB Reads"
Cohesion: 0.11
Nodes (18): rag.conversation_embeddings, kernel.concept_nodes, count_concept_nodes(), _kernel(), load_alerts(), load_labels_by_subject(), load_learning_events_for_concept(), load_mindset() (+10 more)

### Community 16 - "Evidence Update Rules"
Cohesion: 0.12
Nodes (18): blocage_evidence(), The credit an attempt counts for, given why the student was blocked. K measures…, Bayesian update of K after one observation. An assisted attempt is a different…, update_bkt(), compute_effective_mastery(), Apply exponential decay to a raw mastery score. Args: k_raw: stored mastery…, test_analyze_applies_every_attempt_not_just_the_last(), test_analyze_decays_mastery_before_updating_it() (+10 more)

### Community 17 - "README & Dependencies"
Cohesion: 0.12
Nodes (16): Bluestift Kernel README, Bluestift Cognitive Kernel, Dynamic KCs (open graph), LLM Chain Groq -> Gemini, Truststore OS TLS Trust, pytest, pytest-asyncio, FastAPI (+8 more)

### Community 18 - "Offline Graph Builder"
Cohesion: 0.18
Nodes (17): _bridges_from_provider(), canonicalize_vocabulary(), generate_cross_subject_edges(), generate_edges(), _norm(), _parse_vocab_items(), _prereqs_from_provider(), Offline knowledge-graph builder — distill a curriculum graph from public LLMs.… (+9 more)

### Community 19 - "Corpus Errata"
Cohesion: 0.14
Nodes (16): Condensat Errata, AutoTutor EMT sequence, Bluestift corpus strategic condensate, Kornell, Hays & Bjork 2009, Linguistic block not counted, Mueller & Dweck 1998, Literature vs design vs transposition, EMT entry level from K, P, M, tau (+8 more)

### Community 20 - "RAYA Handoff Contract"
Cohesion: 0.16
Nodes (16): Kernel to RAYA Handoff, commit_state: false, Operational Hardening, /analyze route, KERNEL_API_SECRET Auth, /load_profile route, App profile cache (profile-cache.ts), Railway Deployment (primary) (+8 more)

### Community 21 - "Forgetting & Spacing"
Cohesion: 0.16
Nodes (14): /analyze 7-Step Pipeline, days_since(), get_lambda(), _parse_ts(), datetime, Exponential forgetting / decay of mastery. K_effective = floor + (K_raw -…, Whole-and-fractional days elapsed since a timestamp (datetime or ISO str)., How much slower than its base rate this student forgets this KC, in (0, 1]. (+6 more)

### Community 22 - "Analyze Orchestration"
Cohesion: 0.19
Nodes (13): School -> AI -> Student Channel, Edge et al. 2024 GraphRAG, schools.school_curriculum_layers, Post-MVP Roadmap, Institutional Dashboard, kc_priorities layer, Offline-first sync, school_curriculum_layers (+5 more)

### Community 23 - "School Channel Roadmap"
Cohesion: 0.17
Nodes (13): Pedagogical-Safety Alert Types, POST /analyze API Contract, EMT Prompt Injection of Cognitive Vector, Corbett & Anderson 1995, Open Design Tensions, RAYA Integration (priority 1), Selective-Update Gate, Pedagogical-Safety Anomaly Detection (+5 more)

### Community 24 - "Safety Alerts & Cognitive Vector"
Cohesion: 0.19
Nodes (12): Kernel Moat, Sub-Saharan population calibration, Sub-Saharan Calibration Context, Confidence Calibration, DFS Root-Cause by Convergence, Cross-Subject Bridges, Paged graph reads, Root-Gap Detection (+4 more)

### Community 25 - "Moat & Detection Concepts"
Cohesion: 0.17
Nodes (12): _bootstrap_ssl(), _load_all(), load_all_labels(), load_concept_edges(), load_labels_for_subject(), load_states_for_concept(), Supabase data access for the Kernel. Uses the service_role key (never the anon…, Return KC labels across all subjects — grounds cross-subject extraction so a… (+4 more)

### Community 26 - "DB Access Layer"
Cohesion: 0.20
Nodes (10): node_id(), Return the DB id for a node label, or None if absent., Execute the full analyze pipeline and return the response dict., run_analysis(), get_or_create_kc(), LLMBudget, _normalize_label(), A per-request allowance of KC-inference LLM calls, shared by recursion. (+2 more)

### Community 27 - "Rate Limiting"
Cohesion: 0.18
Nodes (11): check_analyze(), _limit(), Rate limiting for the expensive routes. `/analyze` is the costly path: every…, Gate one /analyze call. Returns (allowed, retry_after_seconds, scope). The…, Cost bounds (MAX_DEPTH, MAX_LLM_CALLS_PER_REQUEST), A call refused per-user must not still count toward the global ceiling., test_a_rejected_call_consumes_no_budget(), test_bad_limit_env_falls_back_to_the_default() (+3 more)

### Community 28 - "Cognitive Vector Heuristics"
Cohesion: 0.18
Nodes (12): _estimate_personal_lambda(), _next_state(), _persistence(), V dimension — individualized learning rate, i.e. p(T) (Yudelson 2013).…, Personal p(S) — a slip is a failure made while mastery already looked solid., P dimension — resistance to slip = (1 - p(S)), modulated by mindset M.…, The student_concept_state row after this session's evidence (pure). `ks` is the…, Update the student's personal forgetting rate from the first attempt back. Only… (+4 more)

### Community 29 - "DB Logging"
Cohesion: 0.17
Nodes (12): check_db_access(), log_alert(), log_kernel_request(), log_trajectory(), _now_iso(), Append a temporal mastery snapshot (best-effort; never breaks the flow)., Log every /analyze call, even ones that later error out., Probe kernel-schema read and write access, for readiness/health. A recurring… (+4 more)

### Community 30 - "Pitch & Literature"
Cohesion: 0.25
Nodes (11): Architecture Diagrams, Data Flywheel (priors -> self-calibration), Bloom 2-sigma (1984), Kernel is extended BKT + symbolic rules, not DKT, Khajah, Lindsey & Mozer 2016, Piech et al. 2015 (DKT), Xiong et al. 2016, Pitch One-Pager (+3 more)

### Community 31 - "Extraction & Observations"
Cohesion: 0.22
Nodes (11): _load_curriculum_layers(), Load and parse the student's school layers; empty when they have no school.…, load_curriculum_layers(), load_school_id(), load_school_student_ids(), The school a student belongs to, or None for an independent learner.…, Active curriculum layers for a school, for this subject. Level is matched…, Every student the app has registered under this school. (+3 more)

### Community 32 - "School Data Reads"
Cohesion: 0.20
Nodes (10): _bounded(), _create_edge_if_absent(), _escape_like(), _fallback_kc_data(), _find_kc(), Dynamic KC registry. The Kernel Graph is open: KCs are created on the fly the…, Insert a prerequisite edge unless it already exists (idempotent)., Make a label match itself only under ILIKE. Labels are snake_case, and `_` is… (+2 more)

### Community 33 - "Request Validators"
Cohesion: 0.20
Nodes (7): model_validator, LoadAlertsRequest, PrerequisiteGapsRequest, A graded attempt on one KC. The caller identifies the KC either by `concept_id`…, What does this student still need before they can hold this concept? Identify…, Read pedagogical-safety alerts, for one student, a class list, or a school.…, UpdateConceptStateRequest

### Community 34 - "Graph Build Script"
Cohesion: 0.31
Nodes (8): Cold-start LLM graph distillation, _amain(), main(), Build a curriculum KC graph from the public LLMs and (optionally) persist it.…, build_graph(), persist_graph(), Run the full offline build and return a structured result (no DB writes).…, Insert the built graph into Supabase (idempotent). Returns (nodes, edges).

### Community 35 - "RAG Scoping"
Cohesion: 0.25
Nodes (8): extract_kcs(), _format_conversation(), generate_summary(), _learning_events(), The /analyze pipeline orchestration. Pulls together the LLM extraction, dynamic…, Run the extraction LLM call. Returns (parsed_data, llm_used). `known_labels` is…, Generate the learner-facing summary. Returns (summary, llm_used)., learning_events rows for a session; k_before is set on counted attempts.

### Community 36 - "Inconsistency & Monitoring"
Cohesion: 0.22
Nodes (9): load_chunks_for_concepts(), load_student_class_id(), _rag(), The class a student belongs to, or None. App-owned table, read-only., Teaching material attached to these concepts, scoped to one student. The graph…, The leak surface of the retrieval half. A chunk may be global curriculum…, Fails closed: an unresolved class must not widen the scope., test_teaching_material_never_crosses_between_students_or_classes() (+1 more)

### Community 37 - "Prod Check Script"
Cohesion: 0.25
Nodes (8): Fail loudly with 503 on safety screen, kernel.kernel_monitoring table, Anomaly detector thresholds, Hooshyar (inconsistency / Responsible-DKT), Hooshyar 2026, inconsistency_high Detector, Learning-Trajectory Snapshots, /load_alerts route

### Community 38 - "Learning Events Logging"
Cohesion: 0.36
Nodes (4): bad(), ok(), say(), check_prod.sh script

### Community 39 - "Vocabulary Generation"
Cohesion: 0.25
Nodes (8): log_individual_insight(), log_kernel_output(), log_learning_events(), log_monitoring(), Append graded attempts to kernel.learning_events (best-effort). The raw…, Best-effort: a logging/permission failure must not break the analysis., Best-effort: never break the analysis if the insight write fails., Best-effort monitoring write; swallow errors so it never breaks a flow.

### Community 40 - "GraphRAG Tests"
Cohesion: 0.29
Nodes (8): _dedup_vocabulary(), generate_strands(), generate_vocabulary(), generate_vocabulary_stranded(), Generate the canonical KC vocabulary in one shot. Returns label -> node., Ask the LLM for the curriculum strands (sub-domains) of a subject. Returns []…, Generate vocabulary strand-by-strand for depth, then merge + dedup. A single…, Collapse near-duplicate labels by string similarity (keeps the first seen).…

### Community 41 - "Sliding Window Limiter"
Cohesion: 0.25
Nodes (8): The maths chain, one student who is shaky on derivatives., A question must not create graph nodes. /update_concept_state may invent a KC…, resources_available is false because nothing is written yet — not because these…, _seed_graph_and_student(), test_a_label_the_graph_does_not_know_is_a_404_not_a_new_node(), test_a_student_reads_only_their_own_prerequisite_gaps(), test_an_empty_corpus_says_so_instead_of_pretending(), test_prerequisite_gaps_answers_what_vector_search_cannot()

### Community 42 - "State Chain (CAS)"
Cohesion: 0.33
Nodes (3): Counts events per key over a rolling window. Thread-safe., Check every limit first, then record the event against all of them. `checks` is…, SlidingWindow

### Community 43 - "Core Tables Migration"
Cohesion: 0.29
Nodes (7): Exception, commit_student_concept_state(), The state row moved on since it was read: recompute on the new tip., Write a state row as the next block of its chain (compare-and-set).…, Insert or update one student_concept_state row (keyed by user+concept)., StaleState, upsert_student_concept_state()

### Community 44 - "Cross-Subject Bridges"
Cohesion: 0.48
Nodes (6): kernel.concept_edges, kernel.concept_nodes, kernel.individual_insights, kernel.kernel_outputs, kernel.kernel_requests, kernel.student_concept_state

### Community 45 - "Fake HTTP Client"
Cohesion: 0.38
Nodes (6): _amain(), main(), Build cross-subject prerequisite bridges (e.g. PHYSICS depends on MATH). The…, enforce_dag(), DiGraph, Keep the largest acyclic subset of edges. Returns (kept, dropped). `base` seeds…

### Community 46 - "BKT Parameters"
Cohesion: 0.29
Nodes (3): _FakeAsyncClient, Minimal httpx.AsyncClient stand-in that records the request., test_bad_user_tokens_are_refused()

### Community 47 - "Test Fixtures"
Cohesion: 0.33
Nodes (6): get_bkt_params(), _param(), One param from the node, or the prior when missing/unreadable; bounded. `None`…, Read BKT params from a concept node, falling back to literature priors. A node…, test_bkt_params_stay_identifiable(), test_bkt_uses_node_params_over_priors()

### Community 48 - "Patch v2 Tables"
Cohesion: 0.33
Nodes (6): Clear all counters (tests)., reset(), _clean_ratelimit(), client(), fake_httpx(), fixture

### Community 49 - "Chain Commit & Session"
Cohesion: 0.33
Nodes (5): kernel.kernel_monitoring, kernel.learning_trajectories, kernel.student_mindset_state, schools.school_curriculum_layers, kernel.concept_nodes

### Community 50 - "Vocabulary Budget"
Cohesion: 0.33
Nodes (6): _commit_session(), _persist_state(), Apply one session's evidence to one KC state, in order. Pure. Returns (lam, ks,…, Compute the session update and, if `commit`, write it as the next block. A…, Write the state row as the next block of its chain, and one trajectory snapshot…, _session_update()

### Community 51 - "Boot & Paging Tests"
Cohesion: 0.33
Nodes (6): _format_vocabulary(), Render the known-KC vocabulary for the extraction prompt. The budget used to be…, Label drift is how a graph fragments as subjects are added. The budget used to…, test_a_large_foreign_vocabulary_cannot_crowd_out_the_home_subject(), test_a_small_graph_is_sent_whole_regardless_of_subject(), test_the_vocabulary_budget_serves_the_subject_being_analysed()

### Community 52 - "Math Seed"
Cohesion: 0.20
Nodes (6): A caller that only knows the concept's name still gets a graded update., Starting up must not talk to anything. The service sleeps and is woken by a…, A truncated graph is not a partial read, it is a wrong answer. PostgREST caps…, test_boot_touches_no_network(), test_the_graph_is_paged_so_it_is_never_silently_half_loaded(), test_update_concept_state_resolves_a_label()

### Community 53 - "Learning Events Table"
Cohesion: 0.50
Nodes (3): Starter Math KCs and prerequisite edges. These are a *seed*, not an exhaustive…, Insert the starter Math KCs and edges. Returns (nodes, edges) inserted.…, seed_math_kcs()

### Community 54 - "Trajectory Grouping"
Cohesion: 0.50
Nodes (4): _attempt_credit(), _observations(), Turn extracted attempts into observations the update can use. An attempt's own…, The credit in [0, 1] an extracted attempt stands for, or None if unusable. The…

### Community 56 - "_group_trajectories()"
Cohesion: 0.67
Nodes (3): _group_trajectories(), Group trajectory snapshots into a k_raw series per concept, oldest first. The…, test_group_trajectories_keeps_order_per_concept()

### Community 57 - "_population_baseline()"
Cohesion: 0.67
Nodes (3): _population_baseline(), The KC's calibrated empirical difficulty, or None if it has none yet. New KCs…, test_population_baseline_ignores_the_uncalibrated_default()

## Ambiguous Edges - Review These
- `Operational Hardening` → `KERNEL_API_SECRET Auth`  [AMBIGUOUS]
  POST_MVP_ROADMAP.md · relation: conceptually_related_to

## Knowledge Gaps
- **28 isolated node(s):** `kernel.kernel_monitoring`, `kernel.student_mindset_state`, `schools.school_curriculum_layers`, `Truststore OS TLS Trust`, `google-generativeai` (+23 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **28 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Operational Hardening` and `KERNEL_API_SECRET Auth`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Bluestift Kernel README` connect `README & Dependencies` to `Test Fake Supabase`, `RAG Scoping`, `Root-Gap Detector`, `Anomaly Detectors`, `Test Suite`, `School Curriculum Layers`, `FastAPI App & Auth`, `API Schemas`, `Corpus Errata`, `RAYA Handoff Contract`, `Forgetting & Spacing`, `Analyze Orchestration`, `Learning Events Table`, `Moat & Detection Concepts`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `Kernel to RAYA Handoff` connect `RAYA Handoff Contract` to `Mindset M (measured)`, `Calibration & EM Fit`, `Prod Check Script`, `Shared DB & Migrations`, `README & Dependencies`, `Corpus Errata`, `Analyze Orchestration`, `School Channel Roadmap`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Why does `run_analysis()` connect `DB Access Layer` to `Test Fake Supabase`, `Mindset M (measured)`, `Calibration & EM Fit`, `Test Suite`, `Root-Gap Detector`, `Anomaly Detectors`, `School Curriculum Layers`, `BKT Core & tau`, `Shared DB & Migrations`, `Effective State & Routes`, `DB Reads`, `Evidence Update Rules`, `Forgetting & Spacing`, `Moat & Detection Concepts`, `DB Logging`, `Extraction & Observations`, `RAG Scoping`, `Vocabulary Generation`, `Test Fixtures`, `Vocabulary Budget`, `Trajectory Grouping`, `_group_trajectories()`, `_population_baseline()`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `Principal` (e.g. with `AnalyzeRequest` and `AnalyzeResponse`) actually correct?**
  _`Principal` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `kernel.kernel_monitoring`, `kernel.student_mindset_state`, `schools.school_curriculum_layers` to the rest of the system?**
  _28 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Test Fake Supabase` be split into smaller, more focused modules?**
  _Cohesion score 0.05513784461152882 - nodes in this community are weakly interconnected._