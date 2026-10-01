"""Give every concept a readable name per locale (migration 013).

Asks the LLM, in batches, for the en/fr/es/de name of each concept still
missing one, and writes it to concept_nodes.display_names. Names already there
are kept, so it is safe to run again. New concepts are named at creation; this
is for the ones created before.

    python scripts/backfill_display_names.py --dry-run   # show, write nothing
    python scripts/backfill_display_names.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from services import db, display_names  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="name the concepts but write nothing")
    parser.add_argument("--batch", type=int, default=display_names.BACKFILL_BATCH)
    args = parser.parse_args()

    report = asyncio.run(display_names.backfill(db.get_client(), args.batch, args.dry_run))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
