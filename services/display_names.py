"""Display names for concepts, in the languages the app speaks.

A KC's label is its identity, and it is French snake_case by design: extraction
maps an English or Spanish conversation onto the same labels, so the graph does
not split by language. The label was never meant to be read by a person, yet
the app showed it as-is. `concept_nodes.display_names` (migration 013) holds
what a person reads, one entry per locale the app ships.

Names come from the LLM, so they are cleaned like any other model output that
reaches a screen or a prompt: known locales only, plain text, bounded length.
"""
from __future__ import annotations

import json
import re

from .llm import extract_json, llm_call

# The app's message catalogues (lib/i18n): en, fr, es, de.
LOCALES = ("en", "fr", "es", "de")

# [design] Long enough for "Solving systems of two linear equations", short
# enough to sit in a table cell.
MAX_NAME_LENGTH = 80

# Markup, template and control characters have no business in a concept name.
_UNSAFE = re.compile(r"[<>{}\[\]`\\\x00-\x1f\x7f]")


def clean(raw) -> dict[str, str]:
    """Keep only well-formed names, for the known locales."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for locale in LOCALES:
        value = raw.get(locale)
        if not isinstance(value, str):
            continue
        value = " ".join(_UNSAFE.sub(" ", value).split())[:MAX_NAME_LENGTH].strip()
        if value:
            out[locale] = value
    return out


def missing_locales(names) -> list[str]:
    names = names if isinstance(names, dict) else {}
    return [locale for locale in LOCALES if not names.get(locale)]


NAMES_PROMPT = """\
Tu nommes des concepts scolaires pour des enseignants et des eleves.

Pour chaque concept ci-dessous (label technique, matiere, niveau, description),
donne le nom qu'un enseignant ecrirait dans un programme ou un carnet de notes,
dans chaque langue :
- 2 a 7 mots, majuscule au premier mot seulement, sans point final ;
- "en" : anglais americain, vocabulaire des ecoles des Etats-Unis
  (ex. "Solving linear equations", "Slope of a line") ;
- "fr" : francais avec les accents (ex. "Résoudre des équations linéaires") ;
- "es" : espagnol ; "de" : allemand ;
- le nom decrit le concept, pas l'eleve ni la conversation.

Concepts :
{concepts}

Reponds UNIQUEMENT en JSON valide, sans markdown, un objet dont les cles sont
EXACTEMENT les labels donnes :
{{"label_1": {{"en": "...", "fr": "...", "es": "...", "de": "..."}}}}
"""


def _describe(nodes: list[dict]) -> str:
    return json.dumps(
        [
            {
                "label": n["label"],
                "subject": n.get("subject"),
                "level": n.get("level"),
                "description": str(n.get("description") or "")[:200],
            }
            for n in nodes
        ],
        ensure_ascii=False,
    )


async def name_batch(nodes: list[dict]) -> tuple[dict[str, dict[str, str]], str]:
    """Display names for a batch of concept rows, keyed by label.

    Only labels that were asked about come back: a model that invents or
    renames a key gets nothing written for it. Returns (names, model_used).
    """
    if not nodes:
        return {}, "none"
    asked = {n["label"] for n in nodes}
    text, model = await llm_call(
        NAMES_PROMPT.format(concepts=_describe(nodes)),
        max_tokens=_TOKENS_PER_CONCEPT * len(nodes) + _REASONING_ALLOWANCE,
    )
    try:
        parsed = extract_json(text)
    except ValueError:  # json.JSONDecodeError: nothing parseable came back
        return {}, model
    if not isinstance(parsed, dict):
        return {}, model
    out = {}
    for label, names in parsed.items():
        if label in asked:
            cleaned = clean(names)
            if cleaned:
                out[label] = cleaned
    return out, model


# [design] Concepts per LLM call. The primary model reasons before it answers,
# and the reasoning comes out of the same token budget: batches of 20 came back
# cut off mid-JSON about half the time. 8 per call, with room to think.
BACKFILL_BATCH = 8
_TOKENS_PER_CONCEPT = 250
_REASONING_ALLOWANCE = 2000


async def backfill(client, batch_size: int = BACKFILL_BATCH, dry_run: bool = False) -> dict:
    """Name every concept still missing a locale. Existing names are kept.

    A batch the LLM fails on is skipped and reported, not retried: running the
    backfill again picks up exactly what is still missing.
    """
    from . import db

    todo = [n for n in db.load_concept_nodes(client) if missing_locales(n.get("display_names"))]
    report = {"missing": len(todo), "named": 0, "unnamed": 0, "failed_batches": 0, "dry_run": dry_run, "examples": {}}
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        try:
            names, _model = await name_batch(batch)
        except Exception:  # noqa: BLE001 - every provider down: report, move on
            report["failed_batches"] += 1
            report["unnamed"] += len(batch)
            continue
        for node in batch:
            new = names.get(node["label"])
            if not new:
                # A reply cut off mid-JSON names nobody: say so, run again.
                report["unnamed"] += 1
                continue
            merged = {**new, **clean(node.get("display_names"))}
            if len(report["examples"]) < 8:
                report["examples"][node["label"]] = merged
            if not dry_run:
                db.update_concept_node(client, node["id"], {"display_names": merged})
            report["named"] += 1
    return report
