"""Build cross-subject prerequisite bridges (e.g. PHYSICS depends on MATH).

The data model and the detector are already subject-agnostic: /analyze loads the
whole graph and the convergence search traverses any edge. The only missing piece
is generating the bridge edges — which this script does, closed-world over both
subjects' vocabularies, corroborated across models and validated as a DAG against
the full existing graph.

Direction: foundational -> applied (the foundational subject is the prerequisite).

Usage:
    python scripts/build_bridges.py PHYSICS MATH --dry-run
    python scripts/build_bridges.py PHYSICS MATH        # persist
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def _amain(applied: str, foundational: str, dry_run: bool) -> None:
    from core.graph import build_graph as build_nx
    from services import db
    from services.graph_builder import enforce_dag, generate_cross_subject_edges

    client = db.get_client()
    nodes = db.load_concept_nodes(client)
    edges = db.load_concept_edges(client)

    applied_labels = [n["label"] for n in nodes if n["subject"] == applied]
    foundational_labels = [n["label"] for n in nodes if n["subject"] == foundational]
    if not applied_labels or not foundational_labels:
        sys.exit(f"Need nodes for both subjects (got {len(applied_labels)} {applied}, "
                 f"{len(foundational_labels)} {foundational}).")

    print(f"Bridging {applied} -> {foundational} "
          f"({len(applied_labels)} applied, {len(foundational_labels)} foundational) ...\n")

    candidates = await generate_cross_subject_edges(
        applied, foundational, applied_labels, foundational_labels
    )
    base = build_nx(nodes, edges)  # full existing graph, label-keyed
    kept, dropped = enforce_dag(candidates, base=base)

    print(f"=== {len(kept)} bridges kept ({sum(1 for e in kept if e['agreed_by'] >= 2)} agreed), "
          f"{len(dropped)} dropped (cycles) ===\n")
    for e in sorted(kept, key=lambda e: e["concept"]):
        mark = "==" if e["agreed_by"] >= 2 else "--"
        print(f"  MATH:{e['prerequisite']:<28} {mark}> {applied}:{e['concept']}")

    if dry_run:
        print("\n[dry-run] Nothing written.")
        return

    label_to_id = {n["label"]: n["id"] for n in nodes}
    inserted = 0
    for e in kept:
        pid, cid = label_to_id.get(e["prerequisite"]), label_to_id.get(e["concept"])
        if not pid or not cid:
            continue
        existing = (
            db._kernel(client, "concept_edges")
            .select("id").eq("concept_id", cid).eq("prerequisite_id", pid).limit(1).execute()
        )
        if existing.data:
            continue
        db._kernel(client, "concept_edges").insert(
            {"concept_id": cid, "prerequisite_id": pid, "weight": e["weight"]}
        ).execute()
        inserted += 1
    print(f"\nInserted {inserted} cross-subject bridges.")


def main() -> None:
    p = argparse.ArgumentParser(description="Build cross-subject prerequisite bridges.")
    p.add_argument("applied", help="Applied subject (e.g. PHYSICS)")
    p.add_argument("foundational", help="Foundational subject (e.g. MATH)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(_amain(args.applied, args.foundational, args.dry_run))


if __name__ == "__main__":
    main()
