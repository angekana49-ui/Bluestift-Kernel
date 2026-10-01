"""Grow the graph finer where learners actually get stuck.

The graph is open: a concept is created the first time a learner touches it,
with its prerequisites inferred then. But only then. A coarse node from the
seed or an early conversation ("fractions_egales_et_operations") keeps the
coarse prerequisites it was born with, and the finer ideas a learner really
trips on ("sens_du_denominateur") are never created — extraction is told to
reuse known labels, so nothing finer ever gets a name.

The detector shows when this bites: the root gap it returns is itself a
failing concept, because there is nothing below it worth descending into. Run
end to end on simulated learners, the diagnosis then stayed on the symptom
("you're stuck on solving linear equations") or climbed to the wrong
prerequisite.

So when the diagnosis bottoms out on a failing concept, the Kernel asks once
for what lies just below it and wires that in. Once per concept, ever, for every
learner (migration 014's `deepened_at`): the graph grows where people struggle,
in any subject, without a curriculum written in advance. It runs after the
response is sent; the next analysis is the one that can descend further.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import db, kc_registry
from . import llm as llm_module
from .llm import extract_json, llm_call

# [design] Finer prerequisites asked for per concept. Three is the same cap as
# at creation (kc_registry): enough to split a coarse idea, few enough that the
# detector's search stays narrow.
MAX_FINER = 3

# [design] LLM calls one deepening may spend: the question itself, then the new
# KCs' own metadata and prerequisites (kc_registry's recursion) on what is left.
DEEPEN_LLM_CALLS = 5

DEEPEN_PROMPT = """\
Tu es un expert en sciences de l'education et en ontologies curriculaires.

Des eleves echouent sur ce concept, et le diagnostic ne trouve aucune cause
plus profonde dans le graphe : il y manque probablement des idees plus fines.
- Concept : {label}
- Matiere : {subject}
- Niveau : {level}
- Description : {description}
- Prerequis deja dans le graphe : {prerequisites}

Donne au plus {max_finer} prerequis PLUS FINS : les idees precises, a l'interieur
de ce concept ou juste en dessous, sans lesquelles on ne peut pas le reussir.
Exemple, pour l'addition de fractions : "sens_du_denominateur",
"fractions_unitaires". Pas les prerequis deja listes, pas le concept lui-meme,
pas de concepts plus avances.

Concepts deja connus dans le graphe (si l'un correspond, REUTILISE EXACTEMENT son
label) :
{known_concepts}

Labels en snake_case, sans accents, en francais, concis et canoniques.
Reponds UNIQUEMENT en JSON valide, sans markdown :
{{"prerequisites": ["label_1", "label_2"]}}
"""


VERIFY_PROMPT = """\
Deux concepts scolaires ({subject}) :
A. {a} (niveau {level_a}) : {description_a}
B. {b} (niveau {level_b}) : {description_b}

Question : B est-il un PREREQUIS de A, c'est-a-dire qu'on apprend B AVANT A et
qu'on ne peut pas reussir A sans B ? Si B s'apprend apres A, en meme temps, ou
s'il est seulement voisin, la reponse est non.
Reponds UNIQUEMENT par oui ou par non.
"""


async def _confirmed_prerequisite(concept: dict, candidate: dict, subject: str) -> bool:
    """Both models' yes on wiring an EXISTING concept below this one.

    A new, finer concept only touches the node it was made for. An existing
    one is already wired into everyone's graph, and a wrong edge there misleads
    every diagnosis that passes through it: in a local run the proposer put
    "pente_ligne" (slope) below solving linear equations, and the diagnostic
    question then went to slope. The same corroboration scripts/build_graph.py
    relies on, grounded on each concept's description and level: on labels
    alone one model said yes to slope; grounded, both models answered seven
    test pairs right, direction included. A provider that does not answer
    counts as a no.
    """
    prompt = VERIFY_PROMPT.format(
        subject=subject,
        a=concept["label"], level_a=concept.get("level") or "?",
        description_a=str(concept.get("description") or "")[:300],
        b=candidate["label"], level_b=candidate.get("level") or "?",
        description_b=str(candidate.get("description") or "")[:300],
    )
    for ask in (llm_module.call_gemini, llm_module.call_groq):
        try:
            answer = await ask(prompt, max_tokens=20, temperature=0.0)
        except Exception:  # noqa: BLE001 - a silent verifier is a no
            return False
        if not (answer or "").strip().lower().lstrip(" \"'«").startswith("oui"):
            return False
    return True


def should_deepen(graph, root_gap: str | None, failing: list[str]) -> bool:
    """The diagnosis bottomed out: the root is itself failing, and not deepened."""
    if not root_gap or root_gap not in failing or root_gap not in graph:
        return False
    return not graph.nodes[root_gap].get("deepened_at")


def _depends_on(edges: list[dict], concept_id: str, target_id: str) -> bool:
    """Whether `concept_id` already (transitively) has `target_id` as a prerequisite."""
    below: dict[str, list[str]] = {}
    for e in edges:
        below.setdefault(e["concept_id"], []).append(e["prerequisite_id"])
    seen, stack = set(), [concept_id]
    while stack:
        current = stack.pop()
        if current == target_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(below.get(current, []))
    return False


async def deepen(client, label: str, subject: str, level: str, known_concepts: str = "") -> dict:
    """Ask once for `label`'s finer prerequisites and wire them in.

    Returns a small report. Never raises for an expected reason: no such KC,
    already deepened, migration 014 missing, or an unusable LLM reply.
    """
    node = kc_registry._find_kc(client, label, subject)
    if not node:
        return {"deepened": False, "reason": "unknown concept"}
    if node.get("deepened_at"):
        return {"deepened": False, "reason": "already deepened"}

    # Claim it first: a second request deepening the same concept meanwhile
    # finds it claimed, rather than adding a second set of prerequisites.
    try:
        db.update_concept_node(client, node["id"], {"deepened_at": datetime.now(timezone.utc).isoformat()})
    except Exception:  # noqa: BLE001 - migration 014 not applied: do not deepen at all
        return {"deepened": False, "reason": "migration 014 not applied"}

    edges = db.load_concept_edges(client)
    nodes = {n["id"]: n for n in db.load_concept_nodes(client)}
    current = [nodes[e["prerequisite_id"]]["label"] for e in edges
               if e["concept_id"] == node["id"] and e["prerequisite_id"] in nodes]

    budget = kc_registry.LLMBudget(DEEPEN_LLM_CALLS)
    budget.take()
    try:
        text, _model = await llm_call(DEEPEN_PROMPT.format(
            label=label, subject=subject, level=level,
            description=str(node.get("description") or "")[:300],
            prerequisites=", ".join(current) or "aucun",
            max_finer=MAX_FINER,
            known_concepts=known_concepts or "(aucun)",
        ), max_tokens=400)
        proposed = extract_json(text)
    except Exception:  # noqa: BLE001 - every provider down, or no JSON
        return {"deepened": False, "reason": "no usable reply"}
    proposed = proposed.get("prerequisites") if isinstance(proposed, dict) else None
    if not isinstance(proposed, list):
        return {"deepened": False, "reason": "no usable reply"}

    added = []
    for raw in proposed[:MAX_FINER]:
        if not isinstance(raw, str) or not raw.strip():
            continue
        finer = kc_registry._normalize_label(raw)
        if finer == node["label"] or finer in current:
            continue
        existing = kc_registry._find_kc(client, finer, subject)
        if existing and not await _confirmed_prerequisite(node, existing, subject):
            continue
        prereq = existing or await kc_registry.get_or_create_kc(
            label=finer, subject=subject, level=level,
            supabase_client=client, depth=1, budget=budget,
        )
        # An existing concept that already depends on this one would make a
        # cycle, and the detector's search assumes there is none.
        if prereq["id"] == node["id"] or _depends_on(db.load_concept_edges(client), prereq["id"], node["id"]):
            continue
        kc_registry._create_edge_if_absent(client, node["id"], prereq["id"])
        added.append(prereq["label"])
    return {"deepened": True, "concept": node["label"], "added": added}
