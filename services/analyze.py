"""The /analyze pipeline orchestration.

Pulls together the LLM extraction, dynamic KC creation, forgetting decay, BKT
updates, graph build, root-cause DFS, and the natural-language summary. Kept out
of main.py so the route handler stays thin and this stays unit-testable.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from core import anomaly, bkt, calibration, detector, forgetting, mindset
from core.graph import build_graph, node_id
from services import db
from services.kc_registry import get_or_create_kc
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
      "subject": "MATH" | "ENGLISH" | "PHYSICS" | "HISTORY" | "CHEMISTRY" | "BIOLOGY" | "OTHER",
      "level": "cycle3" | "cycle4" | "lycee" | "college" | "unknown"
    }}
  ],
  "attempts": [
    {{
      "kc_label": "nom_du_concept",
      "outcome": "success" | "failure" | "partial",
      "partial_credit": 0.0,
      "is_assisted": false,
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

Pour mindset_signals (chaque valeur entre 0.0 et 1.0, juge depuis la conversation) :
- abandon_rate : a quel point l'eleve abandonne / se decourage (1 = abandonne vite).
- persistence_score : a quel point il persevere malgre la difficulte (1 = tres tenace).
- time_on_task : engagement et effort percu dans l'echange (1 = tres investi).
- interaction_quality : richesse et reflexion de ses reponses (1 = tres elaborees).

Note : les KCs peuvent etre dans n'importe quelle matiere scolaire.
Si le sujet traite n'est pas {subject}, ajuste le champ subject en consequence.
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


def _format_vocabulary(labels: list[str]) -> str:
    """Render the known-KC vocabulary for the extraction prompt.

    Capped so a large graph doesn't blow the context; the cap keeps the most
    relevant (here: simply the provided order) labels.
    """
    if not labels:
        return "(aucun concept connu pour l'instant)"
    capped = labels[:400]
    return ", ".join(sorted(set(capped)))


async def extract_kcs(
    conversation: list[dict],
    subject: str,
    level: str,
    known_labels: list[str] | None = None,
) -> tuple[dict, str]:
    """Run the extraction LLM call. Returns (parsed_data, llm_used).

    `known_labels` is the existing KC vocabulary for the subject, injected into
    the prompt so the LLM reuses canonical labels instead of inventing variants
    (label drift). This is the lightweight half of KC canonicalization; the
    Kernel still owns the ontology and creates genuinely new KCs on the fly.
    """
    prompt = EXTRACTION_PROMPT.format(
        conversation=_format_conversation(conversation),
        subject=subject,
        level=level,
        known_concepts=_format_vocabulary(known_labels or []),
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

    # 1. LLM extraction of mentioned KCs + attempt evaluations. The existing KC
    #    vocabulary (across ALL subjects) is fed to the prompt to curb label drift,
    #    including cross-subject references (e.g. a physics chat mentioning maths).
    known_labels = db.load_all_labels(client)
    extraction, llm_used = await extract_kcs(conversation, subject, level, known_labels)
    langue = extraction.get("langue_interaction", "fr")

    # 2. Resolve each mentioned KC, creating unknown ones on the fly.
    resolved: dict[str, dict] = {}  # label -> concept_nodes row
    for mention in extraction.get("kcs_mentioned", []):
        label = (mention.get("label") or "").strip()
        if not label:
            continue
        kc = await get_or_create_kc(
            label=label,
            subject=mention.get("subject", subject),
            level=mention.get("level", level),
            supabase_client=client,
        )
        resolved[kc["label"]] = kc

    # 3. Load existing student states for this user.
    states_rows = db.load_student_concept_states(client, user_id)
    states_by_concept = {s["concept_id"]: s for s in states_rows}

    # 3b. Mindset M from the conversation (needed to modulate P during the loop).
    m_score = _update_mindset(client, user_id, extraction.get("mindset_signals"))

    # 4-5. Decay -> effective mastery, then a selective BKT update on attempts.
    attempts_by_label: dict[str, dict] = {a.get("kc_label"): a for a in extraction.get("attempts", [])}
    # How many times each KC is attempted in this conversation (= one "session").
    attempt_counts = Counter(a.get("kc_label") for a in extraction.get("attempts", []))

    mastery_map: dict[str, dict] = {}
    effective_states: dict[str, float] = {}  # label -> k_effective (for DFS)
    kc_records: list[dict] = []              # per-KC snapshot for anomaly detection
    label_to_concept: dict[str, str] = {}

    for label, kc in resolved.items():
        concept_id = kc["id"]
        label_to_concept[label] = concept_id
        state = states_by_concept.get(concept_id)
        k_raw = state["mastery_score_raw"] if state else bkt.get_bkt_params(kc)["p_init"]
        last_at = state.get("last_strong_signal_at") if state else None

        lam = forgetting.get_lambda(kc, state)
        k_eff = forgetting.compute_effective_mastery(
            k_raw, kc.get("type_kc", "conceptual"), last_at, lambda_override=lam
        )

        attempt = attempts_by_label.get(label)
        committed_slip = None
        # Selective-update gate: commit a BKT update only on a strong signal —
        # the first contact (bootstrap), 3+ attempts this session, a large
        # partial-credit shift, or an anomaly. Avoids overreacting to noise.
        if attempt and _should_commit(attempt, state, attempt_counts.get(label, 1)):
            pc = attempt.get("partial_credit")
            if attempt.get("is_assisted"):
                pc = min(pc if pc is not None else 0.9, 0.9)
            params = bkt.get_bkt_params(kc)
            k_before = k_raw
            k_raw = bkt.update_bkt(
                k_before,
                correct=(attempt.get("outcome") == "success"),
                partial_credit=pc,
                params=params,
            )
            # Decay is relative to "now" after a fresh signal -> k_eff == k_raw.
            k_eff = k_raw
            committed_slip = _persist_state(
                client, user_id, concept_id, kc, k_before, k_raw, pc, attempt, state, lam, m_score
            )

        p_slip = committed_slip or (state.get("p_slip_personal") if state else None) or bkt.get_bkt_params(kc)["p_slip"]
        pc_avg = (state.get("partial_credit_avg") if state else None) or 0.5
        status = bkt.classify_status(k_eff, p_slip, pc_avg)
        kc_records.append({"label": label, "k_effective": k_eff, "p_slip": p_slip})

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
    failing = [lbl for lbl, eff in effective_states.items() if eff < detector.FAILING_THRESHOLD]
    detection = detector.detect_root_cause(graph, failing, all_states, known=set(effective_states))
    root_gap = detection["root_gap"]
    detection_path = detection["detection_path"]
    confidence = detection["confidence"]

    root_concept_id = node_id(graph, root_gap) if root_gap else None
    rec_path = detector.recommended_path(graph, root_gap)

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
        "llm_used": llm_used if llm_used != "none" else summary_llm,
    }

    # 9. Persist output + insight (request was already logged by the caller).
    db.log_kernel_output(client, request_id, user_id, output)
    db.log_individual_insight(client, user_id, request_id, summary, root_gap)

    return output


def _should_commit(attempt: dict, prev_state: dict | None, attempts_this_session: int) -> bool:
    """Selective-update gate (Corbett-style strong-signal rule).

    First contact bootstraps the state; afterwards a BKT update commits only on a
    strong signal: 3+ attempts on the KC this session, a partial-credit shift
    > 0.3 vs the stored average, or a flagged anomaly.
    """
    if prev_state is None:
        return True  # bootstrap the very first observation
    pc = attempt.get("partial_credit")
    prev_avg = prev_state.get("partial_credit_avg")
    anomalous = attempt.get("outcome") == "failure" and (prev_state.get("mastery_score_raw") or 0) >= 0.8
    return bkt.should_update_bkt(attempts_this_session, pc if pc is not None else 0.0, prev_avg, anomalous)


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


def _persist_state(client, user_id, concept_id, kc, k_before, k_after, pc, attempt, prev_state, lam, m_score) -> None:
    """Write the updated student_concept_state: K, V, P, personal slip/lambda, trajectory."""
    now = datetime.now(timezone.utc).isoformat()
    prev = prev_state or {}
    prev_count = prev.get("interactions_on_kc", 0) or 0
    prev_struggle = prev.get("struggle_index", 0) or 0
    is_struggle = attempt.get("outcome") == "failure"
    interactions = prev_count + 1
    struggle_index = prev_struggle + (1 if is_struggle else 0)

    # Running average of partial credit (incremental mean).
    prev_avg = prev.get("partial_credit_avg")
    pc_val = pc if pc is not None else (1.0 if attempt.get("outcome") == "success" else 0.0)
    new_avg = pc_val if prev_avg is None else (prev_avg * prev_count + pc_val) / (prev_count + 1)

    # Cognitive vector: V = learning rate p(T); P = (1 - personal slip) modulated by M.
    p_slip_personal = _slip_estimate(prev.get("p_slip_personal"), k_before, is_struggle)

    row = {
        "user_id": user_id,
        "concept_id": concept_id,
        "mastery_score_raw": round(k_after, 4),
        "mastery_score_effective": round(k_after, 4),  # fresh signal -> no decay
        "v_score": _velocity(prev.get("v_score"), k_before, k_after),
        "p_score": _persistence(p_slip_personal, m_score),
        "p_slip_personal": p_slip_personal,
        "partial_credit_avg": round(new_avg, 4),
        "struggle_index": struggle_index,
        "interactions_on_kc": interactions,
        "last_strong_signal_at": now,
    }

    # Personal forgetting rate: if the student returns after a real gap, estimate
    # lambda from the observed decay (stored mastery -> demonstrated level now).
    personal_lambda = _estimate_personal_lambda(prev_state, pc_val, lam)
    if personal_lambda is not None:
        row["lambda_personal"] = round(personal_lambda, 5)

    db.upsert_student_concept_state(client, row)
    db.log_trajectory(client, user_id, concept_id, k_after, k_after)
    return p_slip_personal


def _estimate_personal_lambda(prev_state: dict | None, observed_now: float, current_lambda: float):
    """Estimate the student's personal decay rate from observed forgetting."""
    if not prev_state or not prev_state.get("last_strong_signal_at"):
        return None
    prev_k = prev_state.get("mastery_score_raw") or 0.0
    delta_days = forgetting.days_since(prev_state["last_strong_signal_at"])
    if delta_days < 1 or prev_k <= 0 or observed_now <= 0:
        return None
    return calibration.calibrate_personal_lambda(prev_k, observed_now, delta_days, current_lambda)


def _update_mindset(client, user_id: str, signals: dict | None) -> float:
    """Compute and persist the mindset score M; return it (0.5 if no signal)."""
    if not signals:
        return 0.5
    try:
        m = mindset.compute_mindset_score(
            abandon_rate=float(signals.get("abandon_rate", 0.0)),
            persistence_score=float(signals.get("persistence_score", 0.0)),
            time_on_task=float(signals.get("time_on_task", 0.0)),
            interaction_quality=float(signals.get("interaction_quality", 0.0)),
        )
    except (TypeError, ValueError):
        return 0.5
    db.upsert_mindset(client, user_id, round(m, 4), mindset.classify_mindset(m))
    return m
