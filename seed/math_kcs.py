"""Starter Math KCs and prerequisite edges.

These are a *seed*, not an exhaustive list. The graph grows automatically via
`get_or_create_kc()` for any subject. `seed_kcs` inserts these only when the
concept_nodes table is empty.
"""
from __future__ import annotations

from services.db import _kernel

MATH_KCS = [
    {"label": "nombres_entiers", "subject": "MATH", "level": "cycle3", "type_kc": "declarative", "lambda_decay": 0.05},
    {"label": "operations_de_base", "subject": "MATH", "level": "cycle3", "type_kc": "procedural", "lambda_decay": 0.01},
    {"label": "fractions", "subject": "MATH", "level": "cycle3", "type_kc": "procedural", "lambda_decay": 0.02},
    {"label": "variables_et_expressions", "subject": "MATH", "level": "cycle4", "type_kc": "conceptual", "lambda_decay": 0.02},
    {"label": "equations_1er_degre", "subject": "MATH", "level": "cycle4", "type_kc": "procedural", "lambda_decay": 0.01},
    {"label": "inequations", "subject": "MATH", "level": "cycle4", "type_kc": "procedural", "lambda_decay": 0.01},
    {"label": "systemes_equations", "subject": "MATH", "level": "cycle4", "type_kc": "procedural", "lambda_decay": 0.015},
    {"label": "notion_de_fonction", "subject": "MATH", "level": "cycle4", "type_kc": "conceptual", "lambda_decay": 0.02},
    {"label": "fonctions_lineaires", "subject": "MATH", "level": "cycle4", "type_kc": "conceptual", "lambda_decay": 0.02},
    {"label": "fonctions_affines", "subject": "MATH", "level": "cycle4", "type_kc": "procedural", "lambda_decay": 0.015},
    {"label": "fonctions_polynomiales", "subject": "MATH", "level": "lycee", "type_kc": "conceptual", "lambda_decay": 0.02},
    {"label": "derivees_intro", "subject": "MATH", "level": "lycee", "type_kc": "conceptual", "lambda_decay": 0.025},
    {"label": "derivees_calcul", "subject": "MATH", "level": "lycee", "type_kc": "procedural", "lambda_decay": 0.01},
    {"label": "derivees_applications", "subject": "MATH", "level": "lycee", "type_kc": "procedural", "lambda_decay": 0.01},
    {"label": "limites_fonctions", "subject": "MATH", "level": "lycee", "type_kc": "conceptual", "lambda_decay": 0.03},
]

# Edges as (prerequisite_label, concept_label).
MATH_EDGES = [
    ("nombres_entiers", "fractions"),
    ("nombres_entiers", "operations_de_base"),
    ("operations_de_base", "variables_et_expressions"),
    ("fractions", "variables_et_expressions"),
    ("variables_et_expressions", "equations_1er_degre"),
    ("variables_et_expressions", "inequations"),
    ("equations_1er_degre", "systemes_equations"),
    ("variables_et_expressions", "notion_de_fonction"),
    ("notion_de_fonction", "fonctions_lineaires"),
    ("fonctions_lineaires", "fonctions_affines"),
    ("fonctions_affines", "fonctions_polynomiales"),
    ("fonctions_polynomiales", "derivees_intro"),
    ("derivees_intro", "derivees_calcul"),
    ("derivees_calcul", "derivees_applications"),
    ("fonctions_polynomiales", "limites_fonctions"),
]


def seed_math_kcs(client) -> tuple[int, int]:
    """Insert the starter Math KCs and edges. Returns (nodes, edges) inserted.

    Idempotent at the row level: existing labels are skipped, so re-running is
    safe even though the route only triggers it on an empty table.
    """
    label_to_id: dict[str, str] = {}

    for kc in MATH_KCS:
        existing = (
            _kernel(client, "concept_nodes")
            .select("id")
            .eq("label", kc["label"])
            .eq("subject", kc["subject"])
            .limit(1)
            .execute()
        )
        if existing.data:
            label_to_id[kc["label"]] = existing.data[0]["id"]
            continue
        inserted = (
            _kernel(client, "concept_nodes")
            .insert({**kc, "empirical_difficulty": 0.5, "tau": 0.5})
            .execute()
        )
        label_to_id[kc["label"]] = inserted.data[0]["id"]

    nodes_inserted = len(label_to_id)

    edges_inserted = 0
    for prereq_label, concept_label in MATH_EDGES:
        prereq_id = label_to_id.get(prereq_label)
        concept_id = label_to_id.get(concept_label)
        if not prereq_id or not concept_id:
            continue
        existing = (
            _kernel(client, "concept_edges")
            .select("id")
            .eq("concept_id", concept_id)
            .eq("prerequisite_id", prereq_id)
            .limit(1)
            .execute()
        )
        if existing.data:
            continue
        _kernel(client, "concept_edges").insert(
            {"concept_id": concept_id, "prerequisite_id": prereq_id, "weight": 1.0}
        ).execute()
        edges_inserted += 1

    return nodes_inserted, edges_inserted
