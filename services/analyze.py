"""The /analyze pipeline orchestration.

Pulls together the LLM extraction, dynamic KC creation, forgetting decay, BKT
updates, graph build, root-cause DFS, and the natural-language summary. Kept out
of main.py so the route handler stays thin and this stays unit-testable.
"""
from __future__ import annotations

from datetime import datetime, timezone

from core import bkt, detector, forgetting
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
  "langue_interaction": "fr" | "en" | "ar" | "other"
}}

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
    capped = labels[:120]
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

    # 1. LLM extraction of mentioned KCs + attempt evaluations. The existing
    #    KC vocabulary for the subject is fed to the prompt to curb label drift.
    known_labels = db.load_labels_for_subject(client, subject)
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

    # 4-5. Decay -> effective mastery, then a selective BKT update on attempts.
    attempts_by_label: dict[str, dict] = {a.get("kc_label"): a for a in extraction.get("attempts", [])}

    mastery_map: dict[str, dict] = {}
    effective_states: dict[str, float] = {}  # label -> k_effective (for DFS)

    for label, kc in resolved.items():
        concept_id = kc["id"]
        state = states_by_concept.get(concept_id)
        k_raw = state["mastery_score_raw"] if state else bkt.get_bkt_params(kc)["p_init"]
        last_at = state.get("last_strong_signal_at") if state else None

        lam = forgetting.get_lambda(kc, state)
        k_eff = forgetting.compute_effective_mastery(
            k_raw, kc.get("type_kc", "conceptual"), last_at, lambda_override=lam
        )

        # Apply a BKT update when this KC has an attempt in the conversation.
        attempt = attempts_by_label.get(label)
        if attempt:
            pc = attempt.get("partial_credit")
            if attempt.get("is_assisted"):
                pc = min(pc if pc is not None else 0.9, 0.9)
            params = bkt.get_bkt_params(kc)
            k_raw = bkt.update_bkt(
                k_raw,
                correct=(attempt.get("outcome") == "success"),
                partial_credit=pc,
                params=params,
            )
            # Decay is relative to "now" after a fresh signal -> k_eff == k_raw.
            k_eff = k_raw
            _persist_state(client, user_id, concept_id, k_raw, pc, attempt, state)

        p_slip = (state.get("p_slip_personal") if state else None) or bkt.get_bkt_params(kc)["p_slip"]
        pc_avg = (state.get("partial_credit_avg") if state else None) or 0.5
        status = bkt.classify_status(k_eff, p_slip, pc_avg)

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

    # 7. DFS root-cause from failing KCs.
    failing = [lbl for lbl, eff in effective_states.items() if eff < detector.FAILING_THRESHOLD]
    detection = detector.detect_root_cause(graph, failing, all_states)
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
        "llm_used": llm_used if llm_used != "none" else summary_llm,
    }

    # 9. Persist output + insight (request was already logged by the caller).
    db.log_kernel_output(client, request_id, user_id, output)
    db.log_individual_insight(client, user_id, request_id, summary, root_gap)

    return output


def _persist_state(client, user_id, concept_id, k_raw, pc, attempt, prev_state) -> None:
    """Write the updated student_concept_state after a BKT update."""
    now = datetime.now(timezone.utc).isoformat()
    prev_count = (prev_state or {}).get("interactions_on_kc", 0) or 0
    prev_struggle = (prev_state or {}).get("struggle_index", 0) or 0
    is_struggle = (attempt.get("outcome") == "failure")

    # Running average of partial credit (simple incremental mean).
    prev_avg = (prev_state or {}).get("partial_credit_avg")
    pc_val = pc if pc is not None else (1.0 if attempt.get("outcome") == "success" else 0.0)
    if prev_avg is None:
        new_avg = pc_val
    else:
        new_avg = (prev_avg * prev_count + pc_val) / (prev_count + 1)

    db.upsert_student_concept_state(
        client,
        {
            "user_id": user_id,
            "concept_id": concept_id,
            "mastery_score_raw": round(k_raw, 4),
            "partial_credit_avg": round(new_avg, 4),
            "struggle_index": prev_struggle + (1 if is_struggle else 0),
            "interactions_on_kc": prev_count + 1,
            "last_strong_signal_at": now,
        },
    )
