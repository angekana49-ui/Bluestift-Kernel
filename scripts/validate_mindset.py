"""Validate the mindset score M on what students did next (read-only).

For every analysis logged with a mindset_trace, measures in the following
weeks whether the student persisted after failures, learned beyond what their
mastery predicted, and came back — and how strongly M (and each of its parts)
predicted that. Also tests the EMA smoothing weight. See
core/mindset_validation.py for the method and its limits.

Nothing is written. Read the report as evidence for or against the design
values in core/mindset.py (EMA_WEIGHT, W_MEASURED, the conversation weights).

    python scripts/validate_mindset.py                  # on production data
    python scripts/validate_mindset.py --synthetic 200  # on simulated students
    python scripts/validate_mindset.py --json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from core import mindset, mindset_validation as mv  # noqa: E402


def _from_db():
    from services import db

    client = db.get_client()
    nodes = {n["id"]: n for n in db.load_concept_nodes(client)}
    return db.load_mindset_traces(client), db.load_all_learning_events(client), nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", type=int, metavar="STUDENTS",
                        help="validate on simulated students with a stable latent mindset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        traces, events = mv.simulate(random.Random(args.seed), args.synthetic)
        nodes = {}
    else:
        traces, events, nodes = _from_db()
    result = mv.report(mv.records(traces, events, nodes))
    if args.json:
        print(json.dumps(result))
        return

    print(f"{result['analyses']} analyses with a mindset trace, {result['students']} students")
    if not result["enough_data"]:
        print(f"Not enough data: at least {result['min_analyses']} analyses from "
              f"{result['min_students']} students are needed. Keep the design values.")
        return
    print("\nSpearman rho [95%, students resampled] - positive means 'predicts'")
    print(f"  {'':<22}" + "".join(f"{o:<26}" for o in mv.OUTCOMES))
    for p, row in result["associations"].items():
        cells = []
        for o in mv.OUTCOMES:
            a = row[o]
            cells.append("-" if a["rho"] is None else f"{a['rho']:+.3f} {a['interval_95']}")
        print(f"  {p:<22}" + "".join(f"{c:<26}" for c in cells))
    print(f"\nSmoothing (mean rho over the outcomes), current weight {mindset.EMA_WEIGHT}:")
    diffs = result.get("ema_vs_current_95") or {}
    for w, v in result["ema_weight_mean_rho"].items():
        print(f"  weight {w:<5} {v}   vs current [95%]: {diffs.get(w)}")
    print(f"  best: {result['best_ema_weight']}  (1.0 = no smoothing: M behaves as a state)")
    print("  Change EMA_WEIGHT only for a weight whose interval vs current excludes 0.")
    print("\nAn interval that contains 0 is no evidence that M predicts that outcome.")


if __name__ == "__main__":
    main()
