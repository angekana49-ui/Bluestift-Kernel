"""First come, first served: one student's state changes one request at a time.

Two requests for the same student (a graded attempt landing while a
conversation is being analysed, a client retry) used to read the same state,
each apply their own evidence, and write — the second write erasing the first.

Two layers, like a chain of blocks:

1. `student(user_id)` — an in-process FIFO queue per student. A request waits
   for the one before it and then reads the state that one left. asyncio.Lock
   wakes its waiters in arrival order.
2. The `version` column (migration 012) — for what one process cannot see
   (several workers or instances). A write is accepted only if the row is
   still at the height it was read at; otherwise it raises `db.StaleState` and
   the caller recomputes on the new tip.

Only the read-modify-write section is serialised. The LLM calls around it run
in parallel, so a waiting request waits for the milliseconds of a state
update, not for someone else's extraction.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

# user_id -> [lock, number of requests holding or waiting on it]. An entry is
# dropped when its last request leaves, so the table only holds active students.
_queues: dict[str, list] = {}


@asynccontextmanager
async def student(user_id: str):
    """Hold this student's slot for the duration of the block, in arrival order."""
    entry = _queues.setdefault(user_id, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if entry[1] == 0 and _queues.get(user_id) is entry:
            del _queues[user_id]


# How many times a write rejected as stale is recomputed on the new tip before
# giving up. Past one process a conflict is rare; three in a row means something
# is hammering this student's state and the evidence is better logged than lost
# in a loop.
MAX_CHAIN_RETRIES = 3
