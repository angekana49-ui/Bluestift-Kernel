"""Build a curriculum KC graph from the public LLMs and (optionally) persist it.

Cold-start tool: distills a dense, canonical prerequisite graph out of Groq/Gemini
for a subject, with closed-world prerequisites, cross-model corroboration, and DAG
validation. See services/graph_builder.py for the method.

Usage:
    # Inspect only (no DB writes) — recommended first:
    python scripts/build_graph.py MATH cycle3 cycle4 lycee --dry-run

    # Generate and persist into Supabase:
    python scripts/build_graph.py MATH cycle3 cycle4 lycee
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/build_graph.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def _amain(subject: str, levels: list[str], dry_run: bool) -> None:
    from services import db
    from services.graph_builder import build_graph, persist_graph

    print(f"Building graph for {subject} (levels: {', '.join(levels)}) ...\n")
    result = await build_graph(subject, levels)
    stats = result["stats"]

    print("=== STATS ===")
    print(f"  nodes:          {stats['node_count']}")
    print(f"  edges (kept):   {stats['edge_count']}")
    print(f"  edges agreed:   {stats['agreed_edges']} (both models)")
    print(f"  cycles dropped: {stats['dropped_cycles']}")

    print("\n=== NODES ===")
    for label, node in sorted(result["nodes"].items(), key=lambda kv: kv[1]["level"]):
        print(f"  [{node['level']:<7}] {label:<32} ({node['type_kc']})")

    print("\n=== EDGES (prerequisite -> concept) ===")
    for e in sorted(result["edges"], key=lambda e: e["concept"]):
        mark = "==" if e["agreed_by"] >= 2 else "--"  # == both models, -- single
        print(f"  {e['prerequisite']:<30} {mark}> {e['concept']:<30} w={e['weight']}")

    if result["dropped_edges"]:
        print("\n=== DROPPED (would create cycles) ===")
        for e in result["dropped_edges"]:
            print(f"  {e['prerequisite']} -> {e['concept']}")

    if dry_run:
        print("\n[dry-run] Nothing written to the database.")
        return

    print("\nPersisting to Supabase ...")
    nodes, edges = persist_graph(db.get_client(), result)
    print(f"Inserted {nodes} nodes and {edges} edges.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a KC graph from public LLMs.")
    parser.add_argument("subject", help="Subject tag, e.g. MATH, PHYSICS, ENGLISH")
    parser.add_argument("levels", nargs="+", help="One or more levels, e.g. cycle4 lycee")
    parser.add_argument("--dry-run", action="store_true", help="Generate & inspect; no DB writes")
    args = parser.parse_args()

    if not args.dry_run:
        # Persisting needs DB creds; fail early with a clear message.
        import os

        if not os.getenv("SUPABASE_URL"):
            sys.exit("SUPABASE_URL not set; use --dry-run to inspect without a DB.")

    asyncio.run(_amain(args.subject, args.levels, args.dry_run))


if __name__ == "__main__":
    main()
