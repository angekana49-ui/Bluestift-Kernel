"""Offline knowledge-graph builder — distill a curriculum graph from public LLMs.

This solves cold-start: instead of waiting for real student data (or scraping
curricula), it mines the structure already trained into Groq/Gemini to produce a
dense, canonical KC graph on day one.

Method (the safe one — naive looping reintroduces label drift and cycles):
  1. NODES FIRST  — one call enumerates the canonical KC vocabulary for a subject.
  2. EDGES SECOND — prerequisites are generated in a CLOSED WORLD: the LLM may
                    only reference labels from the fixed vocabulary, so it cannot
                    invent variants.
  3. CROSS-MODEL  — prerequisites are generated with BOTH providers on the same
                    vocabulary; edges both agree on get full weight, single-model
                    edges get a lower weight. A cheap ensemble that cuts
                    hallucinations.
  4. VALIDATION   — dedup near-duplicate labels, drop self-loops and any edge that
                    would create a cycle (a prerequisite graph must be a DAG).

The LLM supplies STRUCTURE (nodes + edges). Real student data later calibrates
the PARAMETERS (difficulty, decay, prerequisite weights) on top — the flywheel.

Run via scripts/build_graph.py.
"""
from __future__ import annotations

import asyncio
from difflib import SequenceMatcher

import networkx as nx

from core.forgetting import LAMBDA_PRIORS
from services.db import _kernel
from services.llm import PROVIDERS, extract_json

# Two labels closer than this (fuzzy ratio) are treated as the same KC.
DEDUP_THRESHOLD = 0.92
# Edge weights by corroboration level.
WEIGHT_BOTH = 1.0
WEIGHT_SINGLE = 0.6

VALID_TYPES = {"procedural", "declarative", "conceptual"}


VOCAB_PROMPT = """\
Tu es un expert en ingenierie curriculaire. Enumere les Knowledge Components (KC)
fondamentaux de la matiere {subject} pour les niveaux : {levels}.

Regles de granularite :
- Un KC = un concept enseignable en 1 a 3 seances (ni trop large, ni trop fin).
- Couvre la progression complete, du plus elementaire au plus avance.
- Entre 20 et 40 KCs, sans doublon.
- Labels en snake_case, sans accents, concis et canoniques.
- LANGUE OBLIGATOIRE : labels ET descriptions en FRANCAIS uniquement, jamais en anglais.

Reponds UNIQUEMENT en JSON valide, sans markdown :
{{
  "kcs": [
    {{
      "label": "nom_en_snake_case",
      "type_kc": "procedural" | "declarative" | "conceptual",
      "level": "cycle3" | "cycle4" | "college" | "lycee",
      "description": "une phrase courte"
    }}
  ]
}}
"""

STRANDS_PROMPT = """\
Tu es un expert en ingenierie curriculaire. Pour la matiere {subject} aux niveaux
{levels}, liste les grands STRANDS (sous-domaines) qui structurent le programme.
Exemple pour les maths : nombres et calcul, algebre, fonctions et analyse,
geometrie, statistiques et probabilites.

LANGUE OBLIGATOIRE : noms de strands en FRANCAIS uniquement.

Reponds UNIQUEMENT en JSON valide, sans markdown :
{{ "strands": ["nom_strand_1", "nom_strand_2", "..."] }}
"""

STRAND_VOCAB_PROMPT = """\
Tu es un expert en ingenierie curriculaire. Pour la matiere {subject} (niveaux
{levels}), enumere la progression COMPLETE et PROFONDE des Knowledge Components du
strand : "{strand}".

Regles :
- Du plus elementaire au plus avance, SANS sauter d'etape intermediaire (chaque
  concept doit pouvoir s'appuyer sur un concept immediatement plus simple du strand).
- Un KC = un concept enseignable en 1 a 3 seances.
- 6 a 10 KCs pour ce strand, sans doublon. Reste a une granularite moyenne :
  un concept solide, pas une micro-competence (ex: "fonction_affine" et non
  "ecrire_equation_fonction_affine" + "tracer_fonction_affine" + "utiliser_fonction_affine").
- Labels en snake_case, sans accents, concis et canoniques.
- LANGUE OBLIGATOIRE : labels ET descriptions en FRANCAIS uniquement, JAMAIS en
  anglais (ex: "calcul_moyenne" et non "calculate_mean").

Reponds UNIQUEMENT en JSON valide, sans markdown :
{{
  "kcs": [
    {{
      "label": "nom_en_snake_case",
      "type_kc": "procedural" | "declarative" | "conceptual",
      "level": "cycle3" | "cycle4" | "college" | "lycee",
      "description": "une phrase courte"
    }}
  ]
}}
"""

PREREQ_PROMPT = """\
Voici la liste EXHAUSTIVE et FERMEE des KCs de la matiere {subject} :
{labels}

Pour chaque KC, donne ses prerequis DIRECTS, choisis EXCLUSIVEMENT dans cette
liste (n'invente jamais de label hors liste). Regles :
- Un prerequis doit etre strictement plus elementaire que le concept.
- Maximum 3 prerequis directs par concept.
- Pas de cycle. Liste vide pour les concepts fondamentaux.

Reponds UNIQUEMENT en JSON valide, sans markdown :
{{
  "edges": [
    {{ "concept": "label_du_concept", "prerequisites": ["label_prereq", "..."] }}
  ]
}}
"""


def _norm(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# Step 1 — vocabulary
# --------------------------------------------------------------------------- #
def _parse_vocab_items(raw: list, subject: str) -> dict[str, dict]:
    """Turn raw LLM KC items into normalized node dicts keyed by label."""
    vocab: dict[str, dict] = {}
    for kc in raw or []:
        label = _norm(kc.get("label", ""))
        if not label:
            continue
        type_kc = kc.get("type_kc", "conceptual")
        if type_kc not in VALID_TYPES:
            type_kc = "conceptual"
        vocab[label] = {
            "label": label,
            "subject": subject,
            "level": kc.get("level", "unknown"),
            "type_kc": type_kc,
            "description": kc.get("description", ""),
            "lambda_decay": LAMBDA_PRIORS.get(type_kc, 0.02),
            "tau": 0.5,
            "empirical_difficulty": 0.5,
        }
    return vocab


async def generate_vocabulary(subject: str, levels: list[str]) -> dict[str, dict]:
    """Generate the canonical KC vocabulary in one shot. Returns label -> node."""
    from services.llm import llm_call

    prompt = VOCAB_PROMPT.format(subject=subject, levels=", ".join(levels))
    text, _model = await llm_call(prompt, max_tokens=4000)
    data = extract_json(text)
    raw = data.get("kcs", []) if isinstance(data, dict) else []
    return _dedup_vocabulary(_parse_vocab_items(raw, subject))


async def generate_strands(subject: str, levels: list[str]) -> list[str]:
    """Ask the LLM for the curriculum strands (sub-domains) of a subject.

    Returns [] on any failure so the caller can fall back to a single-shot
    vocabulary. Token budget is generous: reasoning models (e.g. gpt-oss) spend
    tokens thinking before emitting the JSON.
    """
    from services.llm import llm_call

    prompt = STRANDS_PROMPT.format(subject=subject, levels=", ".join(levels))
    try:
        text, _model = await llm_call(prompt, max_tokens=3000)
        data = extract_json(text)
        strands = data.get("strands", []) if isinstance(data, dict) else []
        return [s for s in strands if isinstance(s, str) and s.strip()]
    except Exception:  # noqa: BLE001 - fall back to single-shot vocab
        return []


async def _vocab_for_strand(subject: str, levels: list[str], strand: str) -> dict[str, dict]:
    from services.llm import llm_call

    prompt = STRAND_VOCAB_PROMPT.format(subject=subject, levels=", ".join(levels), strand=strand)
    try:
        text, _model = await llm_call(prompt, max_tokens=4000)
        data = extract_json(text)
    except Exception:  # noqa: BLE001 - a failed strand just contributes nothing
        return {}
    raw = data.get("kcs", []) if isinstance(data, dict) else []
    return _parse_vocab_items(raw, subject)


async def generate_vocabulary_stranded(subject: str, levels: list[str]) -> dict[str, dict]:
    """Generate vocabulary strand-by-strand for depth, then merge + dedup.

    A single broad prompt trades depth for breadth (it skips intermediate steps
    in each sub-domain). Generating per strand forces a complete progression in
    every strand; the union is then deduplicated into one canonical vocabulary.
    """
    strands = await generate_strands(subject, levels)
    if not strands:
        return await generate_vocabulary(subject, levels)

    per_strand = await asyncio.gather(
        *[_vocab_for_strand(subject, levels, s) for s in strands]
    )
    merged: dict[str, dict] = {}
    for vocab in per_strand:
        merged.update(vocab)
    return _dedup_vocabulary(merged)


def _dedup_vocabulary(vocab: dict[str, dict]) -> dict[str, dict]:
    """Collapse near-duplicate labels by string similarity (keeps the first seen).

    This only catches lexically close labels. Semantic synonyms at different
    wordings (e.g. fonction_affine vs fonctions_affines_et_pentes) are handled by
    the LLM pass in `canonicalize_vocabulary`.
    """
    kept: dict[str, dict] = {}
    for label, node in vocab.items():
        dupe_of = next(
            (k for k in kept if SequenceMatcher(None, k, label).ratio() >= DEDUP_THRESHOLD),
            None,
        )
        if dupe_of is None:
            kept[label] = node
    return kept


CANONICALIZE_PROMPT = """\
Voici une liste de Knowledge Components (KC) de la matiere {subject} :
{labels}

Identifie les DOUBLONS et QUASI-SYNONYMES : concepts identiques formules
differemment, ou plusieurs micro-competences du meme concept (ex:
"ecrire_fonction_affine", "tracer_fonction_affine", "utiliser_fonction_affine"
sont le meme concept que "fonction_affine").

Pour chaque groupe redondant, choisis UN label canonique (le plus simple et
general, deja present dans la liste) et liste les autres a fusionner dedans.
Ne fusionne JAMAIS deux concepts reellement distincts.

Reponds UNIQUEMENT en JSON valide, sans markdown :
{{ "merges": [ {{ "canonical": "label_garde", "aliases": ["label_a_fusionner", "..."] }} ] }}
"""


async def canonicalize_vocabulary(subject: str, vocab: dict[str, dict]) -> dict[str, dict]:
    """LLM pass that merges semantic duplicates/synonyms into canonical labels.

    Returns the vocabulary with alias labels dropped (their canonical kept). On
    any failure the vocabulary is returned unchanged.
    """
    from services.llm import llm_call

    if len(vocab) < 2:
        return vocab
    prompt = CANONICALIZE_PROMPT.format(subject=subject, labels=", ".join(vocab.keys()))
    try:
        text, _model = await llm_call(prompt, max_tokens=4000)
        data = extract_json(text)
    except Exception:  # noqa: BLE001 - keep the vocab as-is on failure
        return vocab

    drop: set[str] = set()
    for merge in (data.get("merges", []) if isinstance(data, dict) else []):
        canonical = _norm(merge.get("canonical", ""))
        if canonical not in vocab:
            continue
        for alias in merge.get("aliases", []) or []:
            alias = _norm(alias)
            if alias != canonical and alias in vocab:
                drop.add(alias)
    return {label: node for label, node in vocab.items() if label not in drop}


# --------------------------------------------------------------------------- #
# Step 2 + 3 — closed-world prerequisites, corroborated across providers
# --------------------------------------------------------------------------- #
async def _prereqs_from_provider(call, subject: str, labels: list[str]) -> set[tuple[str, str]]:
    """Get (prerequisite, concept) edges from one provider, filtered to the vocab.

    Retries with backoff so a transient rate limit (e.g. Gemini free-tier 429,
    which resets per minute) doesn't silently drop a provider's corroboration.
    """
    prompt = PREREQ_PROMPT.format(subject=subject, labels=", ".join(labels))
    label_set = set(labels)

    text = None
    for attempt in range(3):
        try:
            text = await call(prompt, max_tokens=6000)
            break
        except Exception:  # noqa: BLE001 - retry, then give up gracefully
            if attempt < 2:
                await asyncio.sleep(2 ** attempt + 1)  # ~2s, 3s
            else:
                return set()
    try:
        data = extract_json(text)
    except Exception:  # noqa: BLE001
        return set()

    edges: set[tuple[str, str]] = set()
    for item in (data.get("edges", []) if isinstance(data, dict) else []):
        concept = _norm(item.get("concept", ""))
        if concept not in label_set:
            continue
        for prereq in item.get("prerequisites", []) or []:
            prereq = _norm(prereq)
            if prereq in label_set and prereq != concept:
                edges.add((prereq, concept))
    return edges


async def generate_edges(subject: str, labels: list[str]) -> list[dict]:
    """Generate prerequisite edges, corroborated across every available provider.

    Returns a list of {prerequisite, concept, weight, agreed_by}. Edges proposed
    by multiple providers get full weight; single-provider edges get less.
    """
    results = await asyncio.gather(
        *[_prereqs_from_provider(call, subject, labels) for call in PROVIDERS.values()],
        return_exceptions=True,
    )
    provider_edges = [r for r in results if isinstance(r, set)]
    n_providers = max(1, len(provider_edges))

    tally: dict[tuple[str, str], int] = {}
    for edges in provider_edges:
        for e in edges:
            tally[e] = tally.get(e, 0) + 1

    out = []
    for (prereq, concept), votes in tally.items():
        agreed = votes >= 2
        out.append(
            {
                "prerequisite": prereq,
                "concept": concept,
                "weight": WEIGHT_BOTH if agreed else WEIGHT_SINGLE,
                "agreed_by": votes,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Step 4 — DAG validation (drop cycle-creating edges, weakest first)
# --------------------------------------------------------------------------- #
def enforce_dag(edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the largest acyclic subset of edges. Returns (kept, dropped)."""
    graph = nx.DiGraph()
    kept, dropped = [], []
    # Add stronger, multi-provider edges first so they win ties over cycles.
    for edge in sorted(edges, key=lambda e: (-e["weight"], -e["agreed_by"])):
        graph.add_edge(edge["prerequisite"], edge["concept"])
        if nx.is_directed_acyclic_graph(graph):
            kept.append(edge)
        else:
            graph.remove_edge(edge["prerequisite"], edge["concept"])
            dropped.append(edge)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def build_graph(subject: str, levels: list[str], stranded: bool = True) -> dict:
    """Run the full offline build and return a structured result (no DB writes).

    `stranded` generates vocabulary per sub-domain for depth (recommended);
    set False for a single broad vocabulary call.
    """
    if stranded:
        vocab = await generate_vocabulary_stranded(subject, levels)
    else:
        vocab = await generate_vocabulary(subject, levels)
    # Merge semantic synonyms/duplicates before wiring edges.
    vocab = await canonicalize_vocabulary(subject, vocab)
    labels = list(vocab.keys())
    raw_edges = await generate_edges(subject, labels)
    kept_edges, dropped_edges = enforce_dag(raw_edges)
    return {
        "subject": subject,
        "levels": levels,
        "nodes": vocab,
        "edges": kept_edges,
        "dropped_edges": dropped_edges,
        "stats": {
            "node_count": len(vocab),
            "edge_count": len(kept_edges),
            "dropped_cycles": len(dropped_edges),
            "agreed_edges": sum(1 for e in kept_edges if e["agreed_by"] >= 2),
        },
    }


def persist_graph(client, result: dict) -> tuple[int, int]:
    """Insert the built graph into Supabase (idempotent). Returns (nodes, edges)."""
    label_to_id: dict[str, str] = {}

    for label, node in result["nodes"].items():
        existing = (
            _kernel(client, "concept_nodes")
            .select("id")
            .eq("label", label)
            .eq("subject", node["subject"])
            .limit(1)
            .execute()
        )
        if existing.data:
            label_to_id[label] = existing.data[0]["id"]
            continue
        inserted = _kernel(client, "concept_nodes").insert(node).execute()
        label_to_id[label] = inserted.data[0]["id"]

    edges_inserted = 0
    for edge in result["edges"]:
        prereq_id = label_to_id.get(edge["prerequisite"])
        concept_id = label_to_id.get(edge["concept"])
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
            {"concept_id": concept_id, "prerequisite_id": prereq_id, "weight": edge["weight"]}
        ).execute()
        edges_inserted += 1

    return len(label_to_id), edges_inserted
