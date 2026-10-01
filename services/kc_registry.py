"""Dynamic KC registry.

The Kernel Graph is open: KCs are created on the fly the first time a student
touches an unknown concept, in any subject. When a KC is created, its
prerequisites are inferred by the LLM and created recursively (bounded depth).
"""
from __future__ import annotations

from . import display_names
from .db import _kernel
from .llm import extract_json, llm_call

MAX_DEPTH = 3

INFER_PREREQUISITES_PROMPT = """\
Tu es un expert en sciences de l'education et en ontologies curriculaires.

Un eleve vient de mentionner ce concept dans une conversation d'apprentissage :
- Concept : {label}
- Matiere : {subject}
- Niveau : {level}

Reponds UNIQUEMENT en JSON valide, sans markdown :
{{
  "type_kc": "procedural" | "declarative" | "conceptual",
  "lambda_decay": 0.01 a 0.05,
  "description": "description courte du concept en une phrase",
  "display_names": {{"en": "...", "fr": "...", "es": "...", "de": "..."}},
  "prerequisites": ["label_prereq_1", "label_prereq_2"],
  "tau": 0.3 a 0.8
}}

Regles pour lambda_decay :
- procedural (regles, calculs) : 0.01
- conceptual (idees abstraites) : 0.02
- declarative (faits, definitions) : 0.05

display_names : le nom du concept tel qu'un enseignant l'ecrirait, 2 a 7 mots,
majuscule au premier mot seulement. "en" en anglais americain (vocabulaire des
ecoles des Etats-Unis), "fr" avec les accents, "es" espagnol, "de" allemand.

prerequisites : concepts qu'un eleve DOIT maitriser avant ce concept.
Maximum 3 prerequis. Liste vide si c'est un concept fondamental.

tau (rigueur) : a quel point ce concept doit etre EXACT pour etre considere
comme su, a ce niveau. Depend du type ET du niveau vise :
- 0.3 : une idee generale, une intuition (ex. "une fonction associe une valeur
  a une autre" en cycle 3) ; une reponse approchee montre deja la comprehension.
- 0.5 : cas general.
- 0.8 : une definition a restituer mot pour mot, une procedure qui doit etre
  executee sans erreur (ex. une formule de derivation en lycee) ; une reponse
  presque juste ne suffit pas.
"""


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _escape_like(value: str) -> str:
    """Make a label match itself only under ILIKE.

    Labels are snake_case, and `_` is LIKE's any-one-character wildcard:
    unescaped, "fraction_addition" also matches "fractionXaddition".
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


KC_TYPES = ("procedural", "declarative", "conceptual")


def _bounded(value, default: float, lo: float, hi: float) -> float:
    """An LLM-provided number, coerced and bounded; the default if unreadable.

    The prompt shows ranges like `0.01 a 0.05`, and a model will sometimes echo
    that string back. Inserted as-is it fails the float column and takes the
    whole /analyze down with it.
    """
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def _find_kc(client, norm: str, subject: str) -> dict | None:
    """The KC with this label, in ANY subject — preferring the requested one.

    The label is the KC's identity everywhere downstream: the graph is keyed by
    it, and so are mastery_map, detection_path and recommended_path. The DB
    only enforces UNIQUE(label, subject), so a second "vecteurs" under PHYSICS
    used to become a second node that the graph then silently merged with the
    MATH one (attributes overwritten, ids crossed). One label, one KC: a
    physics conversation about vecteurs practises the same concept.
    """
    rows = (
        _kernel(client, "concept_nodes")
        .select("*")
        .ilike("label", _escape_like(norm))
        .execute()
    ).data or []
    if not rows:
        return None
    return min(rows, key=lambda r: (r.get("subject") != subject, str(r.get("created_at") or ""), str(r["id"])))


# [design] LLM calls one request may spend inferring new KCs. Depth 3 with 3
# prerequisites each is up to 1 + 3 + 9 + 27 = 40 sequential calls (each with
# its own retries) for a single unknown concept — minutes on the request path.
# Past the budget a KC is still created, with neutral metadata and no inferred
# prerequisites; scripts/build_graph.py fills the structure in offline.
MAX_LLM_CALLS_PER_REQUEST = 6


class LLMBudget:
    """A per-request allowance of KC-inference LLM calls, shared by recursion."""

    def __init__(self, calls: int = MAX_LLM_CALLS_PER_REQUEST):
        self.remaining = calls

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def _fallback_kc_data(label: str) -> dict:
    return {
        "type_kc": "conceptual",
        "lambda_decay": 0.02,
        "description": f"Concept: {label}",
        "prerequisites": [],
        "tau": 0.5,
    }


async def get_or_create_kc(
    label: str,
    subject: str,
    level: str,
    supabase_client,
    depth: int = 0,
    budget: LLMBudget | None = None,
) -> dict:
    """Find a KC in the DB; create it (and its prerequisites) via LLM if absent.

    Recursion is bounded at MAX_DEPTH to avoid infinite prerequisite chains,
    and the LLM calls by `budget` — pass one budget for a whole request.

    Returns the concept_nodes row (existing or newly created).
    """
    norm = _normalize_label(label)
    budget = budget or LLMBudget()

    # 1. Look up by label (case-insensitive), in any subject.
    found = _find_kc(supabase_client, norm, subject)
    if found:
        return found

    # 2. Infer the KC's metadata via the LLM (with a safe fallback).
    kc_data = _fallback_kc_data(label)
    if budget.take():
        prompt = INFER_PREREQUISITES_PROMPT.format(label=label, subject=subject, level=level)
        try:
            response_text, _llm_used = await llm_call(prompt)
            parsed = extract_json(response_text)
            if not isinstance(parsed, dict):
                raise ValueError("expected object")
            kc_data = parsed
        except Exception:  # noqa: BLE001 - degrade gracefully on bad LLM output
            pass

    # 3. Insert the new KC.
    type_kc = kc_data.get("type_kc")
    row = {
        "label": norm,
        "subject": subject,
        "level": level,
        "description": str(kc_data.get("description") or "")[:500],
        "type_kc": type_kc if type_kc in KC_TYPES else "conceptual",
        "lambda_decay": _bounded(kc_data.get("lambda_decay"), 0.02, 0.001, 0.2),
        "tau": _bounded(kc_data.get("tau"), 0.5, 0.0, 1.0),
        "empirical_difficulty": 0.5,  # neutral default
        "display_names": display_names.clean(kc_data.get("display_names")),
    }
    try:
        try:
            inserted = _kernel(supabase_client, "concept_nodes").insert(row).execute()
        except Exception as e:  # noqa: BLE001
            # Migration 013 not applied yet: a name is not worth losing the KC.
            if "display_names" not in str(e):
                raise
            row.pop("display_names")
            inserted = _kernel(supabase_client, "concept_nodes").insert(row).execute()
        created_kc = inserted.data[0]
    except Exception:
        # UNIQUE(label, subject): another request created this KC while we were
        # waiting on the LLM. Theirs is as good as ours — use it.
        found = _find_kc(supabase_client, norm, subject)
        if not found:
            raise
        return found

    # 4. Recursively create prerequisites and wire edges (prereq -> concept).
    if depth < MAX_DEPTH:
        for prereq_label in (kc_data.get("prerequisites") or [])[:3]:
            if not prereq_label:
                continue
            prereq = await get_or_create_kc(
                label=prereq_label,
                subject=subject,
                level=level,
                supabase_client=supabase_client,
                depth=depth + 1,
                budget=budget,
            )
            if prereq["id"] == created_kc["id"]:
                continue  # guard against self-loops
            _create_edge_if_absent(supabase_client, created_kc["id"], prereq["id"])

    return created_kc


def _create_edge_if_absent(client, concept_id: str, prerequisite_id: str) -> None:
    """Insert a prerequisite edge unless it already exists (idempotent)."""
    existing = (
        _kernel(client, "concept_edges")
        .select("id")
        .eq("concept_id", concept_id)
        .eq("prerequisite_id", prerequisite_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return
    _kernel(client, "concept_edges").insert(
        {
            "concept_id": concept_id,
            "prerequisite_id": prerequisite_id,
            "weight": 1.0,
        }
    ).execute()
