"""Fit the forgetting constants on kernel.learning_events (read-only).

Replays every student's history through the Kernel's decay-and-review model and
reports the constants that best predict the first answer after each gap — on
students held out of the fit. See core/forgetting_fit.py for the method.

Nothing is written. If the report says `adopt: true`, copy the fitted values
into core/forgetting.py (REVIEW_GAIN, LAPSE_PENALTY, LAMBDA_PRIORS), record the
run in PARAMETERS.md, and rerun scripts/eval_kernel.py.

    python scripts/fit_forgetting.py                  # on production data
    python scripts/fit_forgetting.py --synthetic 600  # on simulated students
    python scripts/fit_forgetting.py --json
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

from core import forgetting, forgetting_fit  # noqa: E402


def _from_db() -> list[dict]:
    from services import db

    client = db.get_client()
    nodes = {n["id"]: n for n in db.load_concept_nodes(client)}
    return forgetting_fit.sequences(db.load_all_learning_events(client), nodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", type=int, metavar="STUDENTS",
                        help="fit simulated students (true constants: gain 1.0, "
                             "penalty 0.2, priors x2 / x1 / x0.5) instead of the DB")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        seqs = forgetting_fit.simulate(
            random.Random(args.seed), args.synthetic, gain=1.0, penalty=0.2,
            scales={"procedural": 2.0, "conceptual": 1.0, "declarative": 0.5},
        )
    else:
        seqs = _from_db()
    report = forgetting_fit.fit(seqs)
    report["histories"] = len(seqs)
    report["current"] = {
        "REVIEW_GAIN": forgetting.REVIEW_GAIN,
        "LAPSE_PENALTY": forgetting.LAPSE_PENALTY,
        "LAMBDA_PRIORS": forgetting.LAMBDA_PRIORS,
    }
    if args.json:
        print(json.dumps(report))
        return

    print(f"{report['histories']} (student, KC) histories, "
          f"{report['retrievals_train']} retrievals to fit, {report['retrievals_test']} held out")
    if report["fitted"] is None:
        print(f"Not enough data: at least {report['min_retrievals']} retrievals are needed "
              "(first unassisted answer after a gap of a day or more). Keep the design values.")
        return
    fitted, ci, held = report["fitted"], report["interval_95"], report["held_out"]
    print(f"\n  {'':<20} {'current':<10} {'fitted':<10} 95% (conditional)")
    for key in ("REVIEW_GAIN", "LAPSE_PENALTY"):
        print(f"  {key:<20} {report['current'][key]:<10} {fitted[key]:<10} {ci[key]}")
    for t, v in fitted["LAMBDA_PRIORS"].items():
        print(f"  {'lambda ' + t:<20} {forgetting.LAMBDA_PRIORS.get(t, 0.02):<10} {v:<10} "
              f"{ci['LAMBDA_PRIORS'][t]}")
    print(f"\nHeld-out loss (bits per retrieval, lower is better): fitted {held['fitted_loss']}")
    for name in ("vs_current", "vs_no_spacing"):
        c = held[name]
        print(f"  {name:<20} {c['loss']}  (fitted better by {c['fitted_better_by']} "
              f"+/- {c['standard_error']})")
    print("\nAdopt: " + ("YES - better than both by more than 2 standard errors."
                         if report["adopt"] else "no - keep the current values."))
    print("Intervals hold the other constants at their fitted values, so they understate "
          "the uncertainty when two constants trade off.")


if __name__ == "__main__":
    main()
