"""Synthetic-student benchmark: does the Kernel find the gap we planted?

Every simulated student has a KNOWN root gap: one concept they never learned,
and so every concept built on it is unlearned too; everything else is learned.
They then hold one or more conversations — on the concept they are stuck on,
some foundations, something unrelated — answering with realistic noise (slips,
guesses, help that inflates success, failures caused by the wording of the
statement rather than the concept). The Kernel sees only those conversations,
through the real /analyze pipeline (in-memory DB that persists between a
student's sessions, the LLM replaced by the ground-truth extraction), and we
score its diagnosis after the last one.

This measures the Kernel's reasoning, not the extraction LLM: extraction is
assumed perfect, so the numbers are the ceiling the rest of the stack can reach.

The same script runs against any version of the code, so a before/after is a
like-for-like comparison:

    python scripts/eval_kernel.py                           # 1 session each
    python scripts/eval_kernel.py --sessions 3              # evidence accumulates
    python scripts/eval_kernel.py --sessions 3 --warmup 1000   # + calibrated params
    python scripts/eval_kernel.py --json
    python scripts/eval_kernel.py --sessions 3 --probe kernel  # tutor asks the Kernel's probe
    python scripts/eval_kernel.py --sessions 3 --probe random  # same effort, a random prerequisite

With --probe, each conversation is first analysed diagnose-only (commit_state
false), the tutor asks ONE unassisted question on the chosen concept, and that
answer joins the conversation before the real analysis. `random` is the control:
one extra question in every conversation (so at least the Kernel's effort), on
a prerequisite of the surface concept drawn at random among those the student
has not practised yet.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import networkx as nx  # noqa: E402

from conftest import FakeSupabase  # noqa: E402
from services import analyze as analyze_pipeline  # noqa: E402
from services import kc_registry  # noqa: E402

# The simulated population. Deliberately NOT the Kernel's own priors: a model
# evaluated on data generated from its own assumptions proves nothing.
TRUE_SLIP = 0.08
TRUE_GUESS = 0.25
ASSISTED_SUCCESS_WHEN_UNLEARNED = 0.65  # help makes an unlearned student look competent
ASSISTED_RATE = 0.2
LINGUISTIC_FAILURE_RATE = 0.10          # fails on the wording, whatever they know
GAP_PROBE_RATE = 0.5                    # sessions where the tutor probes the gap itself
MENTIONED_ONLY = 2                      # concepts named in passing, never practised


def make_curriculum(rng: random.Random, layers: int = 5, width: int = 5) -> nx.DiGraph:
    """A layered prerequisite DAG: each concept rests on 1-2 of the layer below."""
    g = nx.DiGraph()
    grid = [[f"kc_{layer}_{i}" for i in range(width)] for layer in range(layers)]
    for row in grid:
        g.add_nodes_from(row)
    for layer in range(1, layers):
        for kc in grid[layer]:
            for prereq in rng.sample(grid[layer - 1], rng.choice((1, 2))):
                g.add_edge(prereq, kc)
    return g


def make_profile(rng: random.Random, g: nx.DiGraph, with_gap: bool) -> dict:
    """Plant a gap (or none) and the concept the student is stuck on."""
    gap = rng.choice(sorted(n for n in g if g.out_degree(n) > 0)) if with_gap else None
    unlearned = ({gap} | nx.descendants(g, gap)) if gap else set()
    if gap:
        # Stuck on something built on the gap, 1-3 hops above it.
        above = sorted(n for n in nx.descendants(g, gap) if 1 <= nx.shortest_path_length(g, gap, n) <= 3)
        surface = rng.choice(above)
    else:
        surface = rng.choice(sorted(n for n in g if g.in_degree(n) > 0))
    return {"gap": gap, "unlearned": unlearned, "surface": surface}


def make_attempt(rng: random.Random, kc: str, learned: bool, assisted: bool) -> dict:
    """One graded answer, with the population's noise."""
    if rng.random() < LINGUISTIC_FAILURE_RATE:
        correct, blocage = False, "linguistic"
    else:
        if learned:
            p = 1 - TRUE_SLIP
        else:
            p = ASSISTED_SUCCESS_WHEN_UNLEARNED if assisted else TRUE_GUESS
        correct = rng.random() < p
        blocage = "none" if correct else "conceptual"
    return {
        "kc_label": kc,
        "outcome": "success" if correct else "failure",
        "partial_credit": rng.choice((1.0, 0.8)) if correct else rng.choice((0.0, 0.3)),
        "is_assisted": assisted,
        "blocage_type": blocage,
        "response_time_estimate": "normal",
    }


def make_session(rng: random.Random, g: nx.DiGraph, profile: dict) -> tuple[dict, bool]:
    """One conversation's ground-truth extraction, and whether it probed the gap."""
    gap, unlearned, surface = profile["gap"], profile["unlearned"], profile["surface"]
    practised = {surface}
    probed = bool(gap) and rng.random() < GAP_PROBE_RATE
    if probed:
        practised.add(gap)
    foundations = sorted(a for a in nx.ancestors(g, surface) if a not in unlearned)
    practised.update(rng.sample(foundations, min(2, len(foundations))))
    practised.add(rng.choice(sorted(n for n in g if n not in practised)))

    attempts = []
    for kc in sorted(practised):
        for _ in range(rng.choice((1, 2, 3))):
            attempts.append(make_attempt(rng, kc, kc not in unlearned, rng.random() < ASSISTED_RATE))
    rng.shuffle(attempts)  # interleaved, as in a real conversation
    mentioned = sorted(practised) + rng.sample(sorted(n for n in g if n not in practised), MENTIONED_ONLY)
    extraction = {
        "kcs_mentioned": [{"label": kc, "subject": "MATH", "level": "lycee"} for kc in mentioned],
        "attempts": attempts,
        "blocage_type": "none",
        "langue_interaction": "fr",
    }
    return extraction, probed


def calibrate(g: nx.DiGraph, rng: random.Random, warmup: int, sessions: int) -> dict[str, dict]:
    """The data flywheel, simulated: fit BKT parameters on a warm-up cohort.

    Only what the Kernel would actually log is used — the counted, autonomous
    attempts, in conversation order — through the production fit
    (calibration.fit_bkt_em). KCs with too little evidence keep the priors.
    """
    from core import calibration

    per_kc: dict[str, dict[int, list[float]]] = {}
    for i in range(warmup):
        profile = make_profile(rng, g, with_gap=i % 4 != 0)
        for _ in range(sessions):
            extraction, _ = make_session(rng, g, profile)
            for a in extraction["attempts"]:
                if a["is_assisted"] or a["blocage_type"] == "linguistic":
                    continue
                per_kc.setdefault(a["kc_label"], {}).setdefault(i, []).append(a["partial_credit"])
    fitted = {}
    for kc, by_student in per_kc.items():
        params = calibration.fit_bkt_em(list(by_student.values()))
        if params:
            fitted[kc] = params
    return fitted


def new_db(g: nx.DiGraph, params: dict) -> FakeSupabase:
    fake = FakeSupabase()
    fake.seed("kernel.concept_nodes", [
        {"id": n, "label": n, "subject": "MATH", "type_kc": "conceptual",
         "created_at": "2026-01-01T00:00:00Z", **params.get(n, {})} for n in g
    ])
    fake.seed("kernel.concept_edges", [
        {"id": f"{u}->{v}", "prerequisite_id": u, "concept_id": v} for u, v in g.edges
    ])
    return fake


async def diagnose(fake: FakeSupabase, extraction: dict, user: int, session, commit: bool = True) -> dict:
    async def ground_truth_llm(prompt, max_tokens=1000):
        if "kcs_mentioned" in prompt:
            return json.dumps(extraction), "ground-truth"
        return "ok", "ground-truth"

    analyze_pipeline.llm_call = ground_truth_llm
    kc_registry.llm_call = ground_truth_llm
    payload = {
        "user_id": f"00000000-0000-0000-0000-{user:012d}",
        "conversation_history": [{"role": "user", "content": "..."}],
        "subject": "MATH",
        "level": "lycee",
        "commit_state": commit,
    }
    return await analyze_pipeline.run_analysis(fake, f"req-{user}-{session}", payload)


def random_probe(rng: random.Random, g: nx.DiGraph, profile: dict, practised: set) -> str | None:
    """The control: a prerequisite of the surface, <= 4 hops, not practised yet."""
    surface = profile["surface"]
    pool = sorted(
        a for a in nx.ancestors(g, surface)
        if a not in practised and nx.shortest_path_length(g, a, surface) <= 4
    )
    return rng.choice(pool) if pool else None


async def run(students: int, seed: int, warmup: int = 0, sessions: int = 1, probe: str = "none") -> dict:
    rng = random.Random(seed)
    g = make_curriculum(rng)
    # A separate stream for the warm-up cohort, so the evaluated students are
    # the same with and without calibration.
    fitted = calibrate(g, random.Random(seed + 10_000), warmup, sessions) if warmup else {}

    exact = on_path = gap_students = 0
    named_roots = false_alarm = clean_students = mislabelled_learned = 0
    misses = {"too_deep": 0, "too_shallow": 0, "unrelated": 0, "none": 0}
    # Exact-root rate split by whether the gap itself was ever practised: with
    # no evidence on the gap, only the graph's structure can point at it.
    by_evidence = {"practised": [0, 0], "not_practised": [0, 0]}
    # The probe's answers get their own stream: the sessions themselves are the
    # same draws in every mode, so modes differ only by the extra question.
    probe_rng = random.Random(seed + 20_000)
    questions = probes_on_gap = 0

    for i in range(students):
        with_gap = i % 4 != 0  # a quarter of the students have no gap at all
        profile = make_profile(rng, g, with_gap)
        fake = new_db(g, fitted)
        probed_ever = False
        practised_ever: set[str] = set()
        for s in range(sessions):
            extraction, probed = make_session(rng, g, profile)
            probed_ever = probed_ever or probed
            practised_ever |= {a["kc_label"] for a in extraction["attempts"]}
            target = None
            if probe == "kernel":
                preview = await diagnose(fake, extraction, i, f"{s}-preview", commit=False)
                target = (preview.get("probe") or {}).get("label")
            elif probe == "random":
                # The control asks in EVERY conversation — at least as many
                # questions as the Kernel, which stays silent when no answer
                # would teach it anything.
                target = random_probe(probe_rng, g, profile, practised_ever)
            if target:
                questions += 1
                probes_on_gap += target == profile["gap"]
                probed_ever = probed_ever or target == profile["gap"]
                practised_ever.add(target)
                extraction["attempts"].append(
                    make_attempt(probe_rng, target, target not in profile["unlearned"], assisted=False)
                )
                if target not in {m["label"] for m in extraction["kcs_mentioned"]}:
                    extraction["kcs_mentioned"].append({"label": target, "subject": "MATH", "level": "lycee"})
            out = await diagnose(fake, extraction, i, s)
        root, gap = out.get("root_gap"), profile["gap"]
        named_roots += root is not None
        if with_gap:
            gap_students += 1
            exact += root == gap
            on_path += gap in (out.get("detection_path") or [])
            key = "practised" if probed_ever else "not_practised"
            by_evidence[key][0] += root == gap
            by_evidence[key][1] += 1
            if root != gap:
                if root is None:
                    misses["none"] += 1
                elif root in nx.ancestors(g, gap):
                    misses["too_deep"] += 1
                elif root in nx.descendants(g, gap):
                    misses["too_shallow"] += 1
                else:
                    misses["unrelated"] += 1
        else:
            clean_students += 1
            false_alarm += root is not None
        # A concept the student HAS learned, reported as a gap.
        mislabelled_learned += sum(
            1 for kc, entry in out.get("mastery_map", {}).items()
            if entry.get("status") == "gap" and kc not in profile["unlearned"]
        )
    return {
        "students": students,
        "seed": seed,
        "sessions": sessions,
        "warmup": warmup,
        "calibrated_kcs": len(fitted),
        "probe_mode": probe,
        # Extra diagnostic questions asked, and how many landed on the gap itself.
        "probe_questions_per_student": round(questions / students, 3),
        "probe_on_gap_share": round(probes_on_gap / questions, 3) if questions else None,
        # Recall: share of gap students whose planted gap is named exactly.
        "root_gap_exact": round(exact / gap_students, 3),
        # Precision: share of NAMED roots that are the planted gap.
        "precision": round(exact / named_roots, 3) if named_roots else None,
        "gap_on_detection_path": round(on_path / gap_students, 3),
        "false_root_on_gapless_students": round(false_alarm / clean_students, 3),
        "learned_kcs_labelled_gap_per_student": round(mislabelled_learned / students, 3),
        "miss_breakdown": {k: round(v / gap_students, 3) for k, v in misses.items()},
        "root_gap_exact_when_gap": {
            k: round(hit / n, 3) if n else None for k, (hit, n) in by_evidence.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--students", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sessions", type=int, default=1, help="conversations per student")
    parser.add_argument("--warmup", type=int, default=0,
                        help="calibrate BKT params on this many warm-up students first")
    parser.add_argument("--probe", choices=("none", "kernel", "random"), default="none",
                        help="one extra diagnostic question per conversation, and who picks it")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run(args.students, args.seed, args.warmup, args.sessions, args.probe))
    if args.json:
        print(json.dumps(result))
        return
    print(f"Synthetic benchmark - {result['students']} students x {result['sessions']} session(s), "
          f"seed {result['seed']}, {result['calibrated_kcs']} KCs calibrated on "
          f"{result['warmup']} warm-up students, probe: {result['probe_mode']}")
    print(f"  root gap found exactly (recall)   {result['root_gap_exact']:.1%}")
    print(f"  named roots that are right        {result['precision']:.1%}")
    print(f"  planted gap on the detection path {result['gap_on_detection_path']:.1%}")
    print(f"  false root on gap-free students   {result['false_root_on_gapless_students']:.1%}")
    print(f"  learned KCs labelled 'gap'/student {result['learned_kcs_labelled_gap_per_student']:.2f}")
    print("  misses: " + ", ".join(f"{k} {v:.1%}" for k, v in result["miss_breakdown"].items()))
    if result["probe_mode"] != "none":
        print(f"  probe questions/student {result['probe_questions_per_student']:.2f}, "
              f"on the gap {result['probe_on_gap_share'] or 0:.1%}")
    split = result["root_gap_exact_when_gap"]
    print(f"  exact when the gap was practised {split['practised']:.1%}, "
          f"when it never was {split['not_practised']:.1%}")


if __name__ == "__main__":
    main()
