"""The /analyze pipeline orchestration.

Pulls together the LLM extraction, dynamic KC creation, forgetting decay, BKT
updates, graph build, root-cause DFS, and the natural-language summary. Kept out
of main.py so the route handler stays thin and this stays unit-testable.
"""
from __future__ import annotations

from datetime import datetime, timezone

from core import anomaly, bkt, calibration, curriculum, detector, forgetting, mindset
from core import probe as active_probe
from core.graph import build_graph, node_id
from services import db, serial
from services.kc_registry import LLMBudget, _normalize_label, get_or_create_kc
from services.llm import extract_json, llm_call

EXTRACTION_PROMPT = """\
Tu es un systeme d'analyse pedagogique. Analyse cette conversation eleve-tuteur.

Conversation:
{conversation}

Matiere declaree : {subject}
Niveau declare : {level}

Concepts deja connus dans le graphe pour cette matiere :
{known_concepts}
Si un concept mentionne correspond a l'un d'eux, REUTILISE EXACTEMENT son label.
Ne cree un nouveau label snake_case que si aucun concept connu ne correspond.

Reponds UNIQUEMENT en JSON valide, sans markdown, sans explication :
{{
  "kcs_mentioned": [
    {{
      "label": "nom_du_concept_en_snake_case",
      "subject": "MATH",
      "level": "cycle3" | "cycle4" | "lycee" | "college" | "unknown"
    }}
  ],
  "attempts": [
    {{
      "kc_label": "nom_du_concept",
      "outcome": "success" | "failure" | "partial",
      "partial_credit": 0.0 | 0.3 | 0.6 | 0.7 | 0.8 | 1.0,
      "is_assisted": false,
      "blocage_type": "conceptual" | "linguistic" | "ambiguous" | "none",
      "response_time_estimate": "fast" | "normal" | "slow"
    }}
  ],
  "blocage_type": "conceptual" | "linguistic" | "ambiguous" | "none",
  "langue_interaction": "fr" | "en" | "ar" | "other",
  "mindset_signals": {{
    "abandon_rate": 0.0,
    "persistence_score": 0.0,
    "time_on_task": 0.0,
    "interaction_quality": 0.0
  }}
}}

Pour chaque tentative :
- partial_credit : la note de la reponse sur cette echelle (1.0 = entierement
  juste, 0.0 = entierement fausse), coherente avec outcome.
- is_assisted : true si l'eleve a ete aide (indice, reformulation, debut de
  solution) avant de repondre.
- blocage_type de la tentative : "linguistic" si l'echec vient de la langue
  (vocabulaire, comprehension de l'enonce) et non du concept ; "conceptual" si
  c'est le concept qui manque ; "ambiguous" si on ne peut pas trancher ;
  "none" si pas de blocage.

Pour mindset_signals (chaque valeur entre 0.0 et 1.0, juge depuis la conversation) :
- abandon_rate : a quel point l'eleve abandonne / se decourage (1 = abandonne vite).
- persistence_score : a quel point il persevere malgre la difficulte (1 = tres tenace).
- time_on_task : engagement et effort percu dans l'echange (1 = tres investi).
- interaction_quality : richesse et reflexion de ses reponses (1 = tres elaborees).

Note : les KCs peuvent etre dans n'importe quelle matiere scolaire.
Si le sujet traite n'est pas {subject}, ajuste le champ subject en consequence.

Le champ "subject" est un mot-cle LIBRE, en MAJUSCULES et sans accent
(MATH, PHYSICS, HISTORY, CHEMISTRY, BIOLOGY, GEOGRAPHY, PHILOSOPHY,
ECONOMICS, ENGLISH, FRENCH, ...). N'utilise PAS "OTHER" : nomme la matiere.
Reutilise exactement le mot-cle deja employe pour cette matiere s'il existe.
"""

SUMMARY_PROMPT = """\
Tu es RAYA, un tuteur bienveillant. En une seule phrase courte, en {langue},
explique a l'eleve pourquoi il bloque, sans jargon et sans le decourager.

Concept ou il bloque : {surface}
Lacune racine detectee : {root_gap}
Chemin de detection : {path}

Reponds UNIQUEMENT par la phrase, sans guillemets.
"""


def _format_conversation(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


# How many KC labels the extraction prompt can carry. The budget is a context
# cost, not a preference — past this the prompt bloats and extraction degrades.
VOCABULARY_BUDGET = 400
# Share of the budget reserved for subjects OTHER than the one being analysed.
# Cross-subject detection is a headline capability: a physics conversation must
# still be able to name a maths concept, so foreign vocabulary cannot be
# squeezed to zero.
CROSS_SUBJECT_SHARE = 0.25


def _format_vocabulary(rows: list[dict], subject: str | None = None) -> str:
    """Render the known-KC vocabulary for the extraction prompt.

    The budget used to be filled in whatever order the database returned rows,
    across every subject. At 154 concepts that was harmless. Past the budget it
    is not: the subject the student is actually working on can end up barely
    represented, the model stops seeing its canonical labels, and it invents
    variants instead of reusing them. That is label drift, and it fragments the
    graph into near-duplicates precisely as new subjects are added — the moment
    the vocabulary matters most.

    So the current subject is served first, and a quarter of the budget is held
    back for everything else, because a physics conversation still has to be
    able to say `derivation_fonction`.
    """
    labels = [r["label"] for r in rows if r.get("label")]
    if not labels:
        return "(aucun concept connu pour l'instant)"
    if len(set(labels)) <= VOCABULARY_BUDGET or not subject:
        return ", ".join(sorted(set(labels))[:VOCABULARY_BUDGET])

    own = sorted({r["label"] for r in rows if r.get("subject") == subject})
    foreign = sorted({r["label"] for r in rows if r.get("subject") != subject})

    cross_budget = int(VOCABULARY_BUDGET * CROSS_SUBJECT_SHARE)
    own_budget = VOCABULARY_BUDGET - cross_budget
    # A small home subject hands its unused room back rather than wasting it.
    if len(own) < own_budget:
        cross_budget += own_budget - len(own)

    chosen = own[:own_budget] + foreign[:cross_budget]
    return ", ".join(sorted(set(chosen)))


async def extract_kcs(
    conversation: list[dict],
    subject: str,
    level: str,
    known_labels: list[dict] | None = None,
) -> tuple[dict, str]:
    """Run the extraction LLM call. Returns (parsed_data, llm_used).

    `known_labels` is the existing KC vocabulary as `{label, subject}` rows,
    injected into the prompt so the LLM reuses canonical labels instead of
    inventing variants (label drift). It carries the subject because the
    vocabulary has a budget, and past it the choice of WHICH labels to send is
    what keeps the graph from fragmenting — see `_format_vocabulary`.
    """
    prompt = EXTRACTION_PROMPT.format(
        conversation=_format_conversation(conversation),
        subject=subject,
        level=level,
        known_concepts=_format_vocabulary(known_labels or [], subject),
    )
    response_text, llm_used = await llm_call(prompt, max_tokens=1200)
    try:
        data = extract_json(response_text)
        if not isinstance(data, dict):
            raise ValueError("expected object")
    except Exception:  # noqa: BLE001 - degrade to an empty extraction
        data = {
            "kcs_mentioned": [],
            "attempts": [],
            "blocage_type": "ambiguous",
            "langue_interaction": "fr",
        }
    return data, llm_used


async def generate_summary(
    surface: str, root_gap: str | None, path: list[str], langue: str
) -> tuple[str, str]:
    """Generate the learner-facing summary. Returns (summary, llm_used)."""
    if not root_gap:
        return (
            "On n'a pas encore assez de signal pour cibler une lacune precise — "
            "continue, j'observe.",
            "none",
        )
    prompt = SUMMARY_PROMPT.format(
        langue=langue or "fr",
        surface=surface or root_gap,
        root_gap=root_gap,
        path=" -> ".join(path),
    )
    try:
        summary, llm_used = await llm_call(prompt, max_tokens=200)
        return summary.strip().strip('"'), llm_used
    except Exception:  # noqa: BLE001
        return (
            f"Tu bloques sur « {surface or root_gap} » parce que « {root_gap} » "
            "n'est pas encore solide.",
            "none",
        )


async def run_analysis(client, request_id: str, payload: dict) -> dict:
    """Execute the full analyze pipeline and return the response dict."""
    user_id = payload["user_id"]
    subject = payload.get("subject", "MATH")
    level = payload.get("level", "unknown")
    conversation = payload["conversation_history"]
    commit_state = payload.get("commit_state", True)
    # KCs whose empirical parameters should be recomputed after this response
    # goes out. A set: the same KC can be committed once per analysis, but this
    # keeps that guarantee local rather than assumed.
    recalibrate_ids: set[str] = set()

    # 1. LLM extraction of mentioned KCs + attempt evaluations. The existing KC
    #    vocabulary (across ALL subjects) is fed to the prompt to curb label drift,
    #    including cross-subject references (e.g. a physics chat mentioning maths).
    known_labels = db.load_labels_by_subject(client)
    extraction, llm_used = await extract_kcs(conversation, subject, level, known_labels)
    langue = extraction.get("langue_interaction", "fr")

    # 2. Resolve each mentioned KC, creating unknown ones on the fly. The
    #    registry canonicalises labels, so remember which raw spelling became
    #    which KC — the attempts below name KCs in the LLM's spelling, not ours.
    kc_budget = LLMBudget()  # one allowance of KC-inference calls for the whole request
    resolved: dict[str, dict] = {}  # canonical label -> concept_nodes row
    canonical: dict[str, str] = {}  # normalized raw label -> canonical label
    for mention in extraction.get("kcs_mentioned", []):
        label = (mention.get("label") or "").strip()
        if not label:
            continue
        kc = await get_or_create_kc(
            label=label,
            subject=mention.get("subject", subject),
            level=mention.get("level", level),
            supabase_client=client,
            budget=kc_budget,
        )
        resolved[kc["label"]] = kc
        canonical[_normalize_label(label)] = kc["label"]

    # Attempts grouped per canonical KC, in conversation order. An attempt on a
    # KC the extraction forgot to list is still evidence: resolve it too.
    attempts_by_label: dict[str, list[dict]] = {}
    for attempt in extraction.get("attempts", []):
        raw = (attempt.get("kc_label") or "").strip()
        if not raw:
            continue
        label = canonical.get(_normalize_label(raw))
        if label is None:
            kc = await get_or_create_kc(
                label=raw, subject=subject, level=level, supabase_client=client,
                budget=kc_budget,
            )
            label = kc["label"]
            resolved[label] = kc
            canonical[_normalize_label(raw)] = label
        attempts_by_label.setdefault(label, []).append(attempt)

    # 3-5 read the student's state and write it back: one request at a time
    #     per student, first come first served (services/serial.py). The LLM
    #     calls before and after run outside the queue.
    async with serial.student(user_id):
        # 3. Load existing student states for this user.
        states_rows = db.load_student_concept_states(client, user_id)
        states_by_concept = {s["concept_id"]: s for s in states_rows}

        # 3a. Mastery history, one query for the whole student, grouped per KC. Feeds
        #     the temporal-inconsistency detector.
        trajectories = _group_trajectories(db.load_recent_trajectories(client, user_id))

        # 3b. Mindset M (needed to modulate P during the loop): the conversation
        #     reading, plus what the student measurably did — their reaction to
        #     errors here, and their learning dynamics in the states above.
        session_attempts = [
            (lbl, c)
            for lbl, atts in attempts_by_label.items()
            for c in (_attempt_credit(a) for a in atts)
            if c is not None
        ]
        mindset_trace: dict = {}
        m_score = _update_mindset(
            client, user_id, extraction.get("mindset_signals"),
            states_rows, trajectories, session_attempts, trace=mindset_trace,
        )

        # 4-5. Decay -> effective mastery, then a BKT update on every attempt.
        mastery_map: dict[str, dict] = {}
        effective_states: dict[str, float] = {}  # label -> k_effective (for DFS)
        kc_records: list[dict] = []              # per-KC snapshot for anomaly detection
        label_to_concept: dict[str, str] = {}

        for label, kc in resolved.items():
            concept_id = kc["id"]
            label_to_concept[label] = concept_id
            state = states_by_concept.get(concept_id)
            params = bkt.get_bkt_params(kc)
            k_raw = state["mastery_score_raw"] if state else params["p_init"]
            last_at = state.get("last_strong_signal_at") if state else None

            lam = forgetting.get_lambda(kc, state)
            k_eff = forgetting.compute_effective_mastery(
                k_raw, kc.get("type_kc", "conceptual"), last_at, lambda_override=lam,
                floor=params["p_init"],
            )

            # Each attempt becomes an observation: its credit, whether it was made
            # with help, and whether it counts at all (a failure caused by a
            # linguistic block is no evidence about the concept). Every graded
            # attempt is logged; only counted ones move K.
            observations = _observations(
                attempts_by_label.get(label, []), extraction.get("blocage_type"), bkt.kc_tau(kc)
            )
            evidence = [o for o in observations if o["counted"]]
            committed_slip = None
            new_state = None
            history = trajectories.get(concept_id, [])

            # Named in the conversation, never practised, no history: there is no
            # evidence either way. Report it as unknown and keep it out of the
            # diagnosis — p_init sits under the failing threshold, so it used to turn
            # every merely-mentioned concept into a "gap" the tutor would remediate.
            if state is None and not evidence:
                mastery_map[label] = {"k_raw": round(k_raw, 4), "k_effective": round(k_raw, 4), "status": "unknown"}
                continue

            # Every attempt is evidence. There used to be a "strong signal" gate
            # here (3+ attempts, a 0.3 credit shift, or an anomaly), justified as
            # protection against drift. But drift comes from re-estimating
            # PARAMETERS on too little data — which the calibration cooldown
            # guards — not from updating the STATE: BKT absorbs noisy answers
            # through slip and guess by design. The gate's real effect was that a
            # student succeeding once per session was never updated, so their
            # mastery decayed while they kept succeeding.
            #
            # `commit_state` only decides whether the evidence is WRITTEN:
            # diagnose-only mode still diagnoses from this conversation.
            if evidence:
                # Update the belief the Kernel actually holds now — the decayed one,
                # the same value /load_profile shows — applying EVERY attempt, in
                # order (see _session_update). Committed as the next block of this
                # KC's chain: if another request moved the state on meanwhile, the
                # update is recomputed on the new tip instead of overwriting it.
                committed = False
                try:
                    state, lam, ks, new_state = _commit_session(
                        client, user_id, concept_id, kc, params, state, evidence,
                        m_score, commit=commit_state,
                    )
                    committed = commit_state
                except Exception as e:  # noqa: BLE001 - degrade, still return the analysis
                    # A write failure must not sink the diagnosis: this
                    # conversation's evidence still informs it, in memory.
                    db.log_monitoring(client, "warn", "persist_state_failed", {"error": str(e)[:300]})
                    lam, ks, new_state = _session_update(
                        user_id, concept_id, kc, params, state, evidence, m_score
                    )
                k_raw = k_eff = ks[-1]  # fresh signal -> no decay yet
                committed_slip = new_state["p_slip_personal"]
                if committed:
                    db.log_learning_events(client, _learning_events(
                        user_id, concept_id, request_id, observations, ks
                    ))
                    # This KC's evidence just changed, so its empirical
                    # parameters may have too. The freshness check is free;
                    # the recalibration runs in the background after the
                    # response, and only for KCs actually past cooldown.
                    if calibration.is_calibration_due(kc):
                        recalibrate_ids.add(concept_id)

            p_slip = committed_slip or (state.get("p_slip_personal") if state else None) or bkt.get_bkt_params(kc)["p_slip"]
            status = bkt.classify_status(
                k_eff, p_slip, bkt.mastery_credit(new_state or state), bkt.kc_tau(kc)
            )
            kc_records.append(
                {
                    "label": label,
                    "k_effective": k_eff,
                    "p_slip": p_slip,
                    "trajectory": history,
                    "population_difficulty": _population_baseline(kc),
                }
            )

            mastery_map[label] = {
                "k_raw": round(k_raw, 4),
                "k_effective": round(k_eff, 4),
                "status": status,
            }
            effective_states[label] = k_eff

    # 6. Build the NetworkX graph from the full DB graph.
    nodes = db.load_concept_nodes(client)
    edges = db.load_concept_edges(client)
    graph = build_graph(nodes, edges)

    # Fold in mastery for *all* graph nodes (default 0.5 when unknown) so the DFS
    # can reason about prerequisites the student hasn't explicitly touched.
    all_states: dict[str, float] = {n["label"]: 0.5 for n in nodes}
    # What the student showed in EARLIER conversations counts too. The search
    # used to know only the KCs this conversation mentioned, so a prerequisite
    # failed last week — or mastered — read as never practised the moment the
    # talk moved on, and a probe's answer would have been forgotten by the next
    # analysis. Decayed like any other state. These inform the prerequisites
    # only: what the student is failing NOW is still this conversation's call.
    history_states = _history_states(graph, states_by_concept, effective_states, mastery_map)
    all_states.update(history_states)
    all_states.update(effective_states)

    # 6b. Pedagogical-safety anomaly detection -> persist alerts to monitoring.
    for rec in kc_records:
        rec["prereq_masteries"] = (
            [all_states.get(p, 0.5) for p in graph.predecessors(rec["label"])]
            if rec["label"] in graph else []
        )
    alerts = anomaly.detect_anomalies(
        kc_records, extraction.get("attempts", []), extraction.get("blocage_type"), m_score
    )
    for alert in alerts:
        db.log_alert(client, user_id, alert, label_to_concept.get(alert.get("concept")))

    # 7. DFS root-cause from failing KCs. `known` = labels with real evidence, so
    #    the DFS can descend into untouched (suspected) prerequisites.
    #    "Failing" is relative to each KC's own prior (detector.failing_threshold).
    known = set(effective_states) | set(history_states)
    weak_below = {
        lbl: detector.failing_threshold(
            bkt.get_bkt_params(resolved.get(lbl) or dict(graph.nodes[lbl]))["p_init"]
        )
        for lbl in known
    }
    failing = [lbl for lbl, eff in effective_states.items() if eff < weak_below[lbl]]
    detection = detector.detect_root_cause(
        graph, failing, all_states, known=known, weak_below=weak_below
    )
    root_gap = detection["root_gap"]
    detection_path = detection["detection_path"]
    confidence = detection["confidence"]

    root_concept_id = node_id(graph, root_gap) if root_gap else None

    # 7a. The one question that would best settle the diagnosis (core/probe.py).
    #     RAYA asks it and sends the answer to /update_concept_state; the next
    #     analysis reads it back through the student's history above.
    probe = active_probe.choose_probe(graph, failing, all_states, known, weak_below, root_gap)

    # 7b. School layer: if the student belongs to a school, its curriculum layers
    #     shape the sequencing and its objectives are reported against real state.
    #     Best-effort throughout — no school, or a school with bad JSON, must land
    #     the student on exactly the analysis they'd get without one.
    layers = _load_curriculum_layers(client, user_id, subject, level)
    rec_path = detector.recommended_path(
        graph, root_gap, priorities=layers.weights, detection_path=detection_path
    )

    # 8. Natural-language summary.
    surface = detection_path[0] if detection_path else (failing[0] if failing else "")
    summary, summary_llm = await generate_summary(surface, root_gap, detection_path, langue)

    output = {
        "request_id": request_id,
        "user_id": user_id,
        "root_gap": root_gap,
        "root_concept_id": root_concept_id,
        "detection_path": detection_path,
        "mastery_map": mastery_map,
        "confidence": confidence,
        "summary": summary,
        "recommended_path": rec_path,
        "alerts": [{"type": a["alert_type"], "severity": a["alert_severity"]} for a in alerts],
        "probe": probe,
        "llm_used": llm_used if llm_used != "none" else summary_llm,
        # Not part of the API contract: the route pops this and schedules the
        # work after responding. It rides along because only this function knows
        # which KCs it actually committed to.
        "recalibrate_concept_ids": sorted(recalibrate_ids),
    }

    if not layers.is_empty:
        output["curriculum"] = {
            "school_id": layers.school_id,
            "layers_applied": layers.applied(),
            "objectives": curriculum.objective_report(layers.objectives, effective_states),
            # Whether the detected root gap is part of the school's program. A
            # gap outside it is still the truth and still reported — the school
            # just knows it is looking at something off-program.
            "root_gap_in_program": (
                root_gap in layers.concepts if (root_gap and layers.concepts) else None
            ),
            "rules": layers.rules,
        }

    # 9. Persist output + insight (request was already logged by the caller).
    # The M reading rides along in the log, not the response: it is what
    # scripts/validate_mindset.py tests against what the student did next.
    db.log_kernel_output(client, request_id, user_id, {**output, "mindset_trace": mindset_trace})
    db.log_individual_insight(client, user_id, request_id, summary, root_gap)

    return output


def _load_curriculum_layers(client, user_id: str, subject: str, level: str):
    """Load and parse the student's school layers; empty when they have no school.

    Wrapped whole: a student's analysis must not depend on their school's data
    being present or well-formed.
    """
    try:
        school_id = db.load_school_id(client, user_id)
        if not school_id:
            return curriculum.CurriculumLayers()
        rows = db.load_curriculum_layers(client, school_id, subject, level)
        return curriculum.parse_layers(rows, school_id=school_id)
    except Exception as e:  # noqa: BLE001 - degrade to no school context
        db.log_monitoring(client, "warn", "curriculum_layers_failed", {"error": str(e)[:300]})
        return curriculum.CurriculumLayers()


def _history_states(graph, states_by_concept: dict, current: dict, mastery_map: dict) -> dict[str, float]:
    """label -> effective mastery for the student's stored KCs this conversation left out.

    Same decay as the session KCs (forgetting.get_lambda, floor p_init), read
    off the graph's node attributes. A stored row is evidence by construction:
    one is only written when an attempt was counted.
    """
    by_id = {data.get("id"): label for label, data in graph.nodes(data=True)}
    out: dict[str, float] = {}
    for concept_id, state in states_by_concept.items():
        label = by_id.get(concept_id)
        if label is None or label in current or label in mastery_map:
            continue
        kc = dict(graph.nodes[label])
        params = bkt.get_bkt_params(kc)
        out[label] = forgetting.compute_effective_mastery(
            state["mastery_score_raw"], kc.get("type_kc", "conceptual"),
            state.get("last_strong_signal_at"),
            lambda_override=forgetting.get_lambda(kc, state), floor=params["p_init"],
        )
    return out


def _group_trajectories(rows: list[dict]) -> dict[str, list[float]]:
    """Group trajectory snapshots into a k_raw series per concept, oldest first.

    The query already orders by snapshot_at, so append order is chronological.
    """
    series: dict[str, list[float]] = {}
    for row in rows:
        concept_id = row.get("concept_id")
        k_raw = row.get("k_raw")
        if concept_id is None or k_raw is None:
            continue
        series.setdefault(concept_id, []).append(float(k_raw))
    return series


def _population_baseline(kc: dict) -> float | None:
    """The KC's calibrated empirical difficulty, or None if it has none yet.

    New KCs are created with a neutral 0.5 placeholder, so the value alone can't
    say whether a baseline exists. `last_calibration_at` is what distinguishes a
    real, population-derived difficulty from that default — comparing a student
    against the placeholder would manufacture out-of-distribution signal from
    nothing.
    """
    if not kc.get("last_calibration_at"):
        return None
    difficulty = kc.get("empirical_difficulty")
    return float(difficulty) if difficulty is not None else None


def _velocity(prev_v, k_before: float, k_after: float) -> float:
    """V dimension — individualized learning rate, i.e. p(T) (Yudelson 2013).

    Estimates the fraction of the remaining mastery gap closed on this trial (an
    empirical p(T)) and smooths it across trials, so V reflects the student's
    typical learning speed rather than a single jump. Bounded [0.05, 0.95].
    """
    room = max(1e-3, 1.0 - k_before)
    realized = max(0.0, min(1.0, (k_after - k_before) / room))
    if prev_v is None:
        return round(max(0.05, min(0.95, realized)), 4)
    return round(max(0.05, min(0.95, 0.7 * prev_v + 0.3 * realized)), 4)


def _slip_estimate(prev_slip, k_before: float, outcome_failure: bool) -> float:
    """Personal p(S) — a slip is a failure made while mastery already looked solid."""
    prior = prev_slip if prev_slip is not None else 0.1  # literature p(S)
    is_slip = 1.0 if (outcome_failure and k_before >= 0.7) else 0.0
    return round(max(0.01, min(0.5, 0.8 * prior + 0.2 * is_slip)), 4)


def _persistence(p_slip: float, m_score: float) -> float:
    """P dimension — resistance to slip = (1 - p(S)), modulated by mindset M.

    Corbett's inverse-slip persistence, raised or lowered by the student's
    mindset (a growth mindset sustains effort under difficulty). Bounded.
    """
    base = 1.0 - p_slip
    return round(max(0.05, min(0.95, base * (0.6 + 0.4 * m_score))), 4)


def _session_update(user_id, concept_id, kc, params, state, evidence, m_score):
    """Apply one session's evidence to one KC state, in order. Pure.

    Returns (lam, ks, new_state): the decay rate used, the mastery trajectory
    through the counted attempts (ks[0] = the decayed belief before the first),
    and the state row to write.
    """
    lam = forgetting.get_lambda(kc, state)
    k_raw = state["mastery_score_raw"] if state else params["p_init"]
    last_at = state.get("last_strong_signal_at") if state else None
    ks = [forgetting.compute_effective_mastery(
        k_raw, kc.get("type_kc", "conceptual"), last_at, lambda_override=lam,
        floor=params["p_init"],
    )]
    for o in evidence:
        ks.append(bkt.update_bkt(
            ks[-1], correct=o["credit"] >= 0.5, partial_credit=o["credit"],
            params=params, assisted=o["assisted"],
        ))
    new_state = _next_state(user_id, concept_id, ks, evidence, state, lam, m_score, params)
    return lam, ks, new_state


def _commit_session(client, user_id, concept_id, kc, params, state, evidence, m_score, commit=True):
    """Compute the session update and, if `commit`, write it as the next block.

    A write computed from a stale state is rejected by the database
    (db.StaleState); the state is then re-read and the SAME evidence reapplied
    on top of it, so both requests' evidence ends up in the chain. Returns
    (state it was built on, lam, ks, new_state).
    """
    for _ in range(serial.MAX_CHAIN_RETRIES):
        lam, ks, new_state = _session_update(user_id, concept_id, kc, params, state, evidence, m_score)
        if not commit:
            return state, lam, ks, new_state
        try:
            _persist_state(client, new_state, ks[-1], state.get("version", 0) if state else None)
            return state, lam, ks, new_state
        except db.StaleState:
            state = db.load_student_concept_state(client, user_id, concept_id)
    raise RuntimeError(f"state chain for {concept_id} kept moving; evidence logged, not applied")


def _next_state(user_id, concept_id, ks, evidence, prev_state, lam, m_score, params) -> dict:
    """The student_concept_state row after this session's evidence (pure).

    `ks` is the mastery trajectory through the counted attempts — ks[0] the
    belief before the first, ks[i + 1] the belief after evidence[i] — so the
    per-attempt estimates (slip, velocity) see the mastery each attempt was
    actually made at.
    """
    credits = [o["credit"] for o in evidence]
    prev = prev_state or {}
    prev_count = prev.get("interactions_on_kc", 0) or 0
    prev_struggle = prev.get("struggle_index", 0) or 0
    interactions = prev_count + len(credits)
    struggle_index = prev_struggle + sum(1 for c in credits if c < 0.5)

    # Running average of partial credit (incremental mean over all attempts).
    prev_avg = prev.get("partial_credit_avg")
    prior_n = prev_count if prev_avg is not None else 0  # no average yet: nothing to weigh
    new_avg = ((prev_avg or 0.0) * prior_n + sum(credits)) / (prior_n + len(credits))

    # The dual mastery condition reads the last three AUTONOMOUS credits.
    recent = prev.get("recent_autonomous_credits") or []
    for o in evidence:
        if not o["assisted"]:
            recent = bkt.push_autonomous_credit(recent, o["credit"])

    # Cognitive vector: V = learning rate p(T); P = (1 - personal slip) modulated by M.
    p_slip_personal = prev.get("p_slip_personal")
    v_score = prev.get("v_score")
    for k_before, k_after, c in zip(ks, ks[1:], credits):
        p_slip_personal = _slip_estimate(p_slip_personal, k_before, c < 0.5)
        v_score = _velocity(v_score, k_before, k_after)

    row = {
        "user_id": user_id,
        "concept_id": concept_id,
        "mastery_score_raw": round(ks[-1], 4),
        "mastery_score_effective": round(ks[-1], 4),  # fresh signal -> no decay
        "v_score": v_score,
        "p_score": _persistence(p_slip_personal, m_score),
        "p_slip_personal": p_slip_personal,
        "partial_credit_avg": round(new_avg, 4),
        "recent_autonomous_credits": recent,
        "struggle_index": struggle_index,
        "interactions_on_kc": interactions,
        "last_strong_signal_at": datetime.now(timezone.utc).isoformat(),
    }

    # Spacing: the first autonomous attempt after a real gap is a retrieval
    # test. Success is a review (the memory gets more durable), failure a lapse.
    reviews, lapses = prev.get("review_count") or 0, prev.get("lapse_count") or 0
    first = evidence[0]
    is_retrieval = (
        prev_state is not None
        and prev.get("last_strong_signal_at")
        and not first["assisted"]
        and forgetting.days_since(prev["last_strong_signal_at"]) >= forgetting.REVIEW_MIN_GAP_DAYS
    )
    if is_retrieval:
        if first["credit"] >= 0.5:
            reviews += 1
        else:
            lapses += 1
    row["review_count"] = reviews
    row["lapse_count"] = lapses

    # Personal forgetting rate, learned from how that retrieval compares with
    # what the decay model predicted for it. `lam` is the rate that prediction
    # used (base x spacing); what is stored is the BASE rate, so the reviews
    # keep being applied on top of it rather than baked in twice.
    personal_lambda = _estimate_personal_lambda(prev_state, first, lam, params)
    if personal_lambda is not None:
        row["lambda_personal"] = round(personal_lambda / forgetting.spacing_factor(prev_state), 5)
    return row


def _persist_state(client, row: dict, k_final: float, read_version: int | None = None) -> None:
    """Write the state row as the next block of its chain, and one trajectory
    snapshot for the session. Raises db.StaleState if the chain moved on."""
    db.commit_student_concept_state(client, row, read_version)
    # One snapshot per session, not per attempt: the inconsistency detector reads
    # this series, and within-session steps would read as oscillation.
    db.log_trajectory(client, row["user_id"], row["concept_id"], k_final, k_final)


def _observations(
    attempts: list[dict], conversation_blocage: str | None, tau: float = bkt.TAU_NEUTRAL
) -> list[dict]:
    """Turn extracted attempts into observations the update can use.

    An attempt's own blocage_type wins over the conversation's; an attempt
    with no usable credit is dropped entirely. The rules themselves live in
    bkt.observation, shared with /update_concept_state.
    """
    out = []
    for a in attempts:
        credit = _attempt_credit(a)
        if credit is None:
            continue
        blocage = a.get("blocage_type") or conversation_blocage
        out.append(bkt.observation(credit, bool(a.get("is_assisted")), blocage, tau))
    return out


def _learning_events(user_id, concept_id, request_id, observations, ks, source="analyze") -> list[dict]:
    """learning_events rows for a session; k_before is set on counted attempts."""
    events, step = [], 0
    for o in observations:
        k_before = None
        if o["counted"]:
            k_before = round(ks[step], 4)
            step += 1
        events.append({
            "user_id": user_id,
            "concept_id": concept_id,
            "credit": o["credit"],
            "is_assisted": o["assisted"],
            "blocage_type": o["blocage_type"],
            "counted": o["counted"],
            "source": source,
            "request_id": request_id,
            "k_before": k_before,
        })
    return events


def _attempt_credit(attempt: dict) -> float | None:
    """The credit in [0, 1] an extracted attempt stands for, or None if unusable.

    The outcome is the LLM's judgement and the credit its grading; when they
    contradict, the outcome wins. The extraction template shows
    `"partial_credit": 0.0`, and a model that copies it next to
    outcome="success" must not have that success scored as a failure.
    Help is not discounted here: an assisted attempt is a different
    observation, handled by the update itself (see bkt.update_bkt).
    """
    raw = attempt.get("partial_credit")
    try:
        pc = max(0.0, min(1.0, float(raw))) if raw is not None else None
    except (TypeError, ValueError):
        pc = None

    outcome = attempt.get("outcome")
    if outcome == "success":
        credit = pc if pc is not None and pc >= 0.5 else 1.0
    elif outcome == "failure":
        credit = pc if pc is not None and pc < 0.5 else 0.0
    elif outcome == "partial":
        credit = pc if pc is not None else 0.5
    else:
        credit = pc  # unknown outcome: the grade alone, if there is one

    return credit


def _estimate_personal_lambda(prev_state: dict | None, first: dict, lam: float, params: dict):
    """Update the student's personal forgetting rate from the first attempt back.

    Only an autonomous attempt after a real gap says anything about forgetting:
    help masks it.
    """
    if not prev_state or not prev_state.get("last_strong_signal_at") or first["assisted"]:
        return None
    k_stored = prev_state.get("mastery_score_raw")
    delta_days = forgetting.days_since(prev_state["last_strong_signal_at"])
    if k_stored is None or delta_days < 1:
        return None
    return calibration.update_personal_lambda(
        lam, k_stored, params["p_init"], delta_days, first["credit"],
        prev_state.get("p_slip_personal") or params["p_slip"], params["p_guess"],
    )


def _signal(signals: dict, key: str) -> float:
    """One behavioural signal, defaulting to neutral when absent or unparseable.

    Neutral and not 0.0: the signals do not share a direction of "good", so a
    zero default would read a missing `abandon_rate` as a student who never
    gives up and a missing `persistence_score` as one who never tries. Per-field
    rather than all-or-nothing, so one junk value from the model does not
    discard the three usable ones alongside it.
    """
    try:
        return float(signals[key])
    except (KeyError, TypeError, ValueError):
        return mindset.NEUTRAL_M


def _update_mindset(
    client,
    user_id: str,
    signals: dict | None,
    states: list[dict] | None = None,
    trajectories: dict[str, list[float]] | None = None,
    session_attempts: list[tuple[str, float]] | None = None,
    trace: dict | None = None,
) -> float:
    """Fold this conversation's mindset reading into the stored score M.

    Smoothed rather than overwritten (see mindset.EMA_WEIGHT): M is a trait
    estimate taken through one exchange, and the label it feeds decides whether
    the tutor stops teaching the concept to address the student's self-belief.

    When the exchange said nothing about mindset, the STORED score is returned —
    "this conversation carried no signal" is not the same claim as "this student
    is average", and P is modulated by whichever one we hand back.

    `trace`, when given, is filled with every intermediate reading, so the
    score can later be validated against what the student actually did.
    """
    previous = None
    row = db.load_mindset(client, user_id)
    if row is not None and row.get("m_score") is not None:
        try:
            previous = float(row["m_score"])
        except (TypeError, ValueError):
            previous = None
    fallback = previous if previous is not None else mindset.NEUTRAL_M

    # Measured on what the student did (see core/mindset.py): how they react to
    # an error in this conversation, and their learning dynamics so far.
    behaviour = mindset.session_behaviour(session_attempts or [])
    measured, evidence = mindset.measured_linear(
        states or [], trajectories or {}, behaviour["recovery"]
    )

    conversation = None
    if signals:
        # "Abandon after an error" is measured whenever there was an error to
        # abandon after; the LLM's reading of it only fills in when there wasn't.
        abandon = behaviour["abandon_after_error"]
        conversation = mindset.conversation_linear(
            abandon_rate=abandon if abandon is not None else _signal(signals, "abandon_rate"),
            persistence_score=_signal(signals, "persistence_score"),
            time_on_task=_signal(signals, "time_on_task"),
            interaction_quality=_signal(signals, "interaction_quality"),
        )

    observed = mindset.combine(conversation, measured, evidence)
    if trace is not None:
        trace.update({
            "previous": previous,
            "conversation_linear": conversation,
            "measured_linear": measured,
            "measured_evidence": evidence,
            "abandon_after_error": behaviour["abandon_after_error"],
            "recovery": behaviour["recovery"],
            "failures": behaviour["failures"],
            "observed": observed,
            "m_score": fallback if observed is None else None,
            "ema_weight": mindset.EMA_WEIGHT,
        })
    if observed is None:
        return fallback
    m = mindset.blend_mindset(previous, observed)
    if trace is not None:
        trace["m_score"] = m
    db.upsert_mindset(client, user_id, round(m, 4), mindset.classify_mindset(m))
    return m
