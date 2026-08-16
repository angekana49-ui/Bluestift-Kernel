"""Rate limiting for the expensive routes.

`/analyze` is the costly path: every call runs at least two LLM requests and
holds the container awake and busy. Nothing bounded it, so a retry loop in a
client — or anyone holding the service secret — could burn the host's credits
and the LLM quota at the same time, silently, until both ran out.

Two limits, because they fail differently:

  per-user  — one student cannot monopolise the Kernel, however the caller
              behaves. This is the one that catches a client-side retry loop.
  global    — a ceiling on total spend per window, whatever the distribution
              of callers. This is the one that catches a compromised secret.

Sliding-window counters held in process. That is deliberate: the Kernel runs a
single replica, and a shared store (Redis) would be another always-on
dependency to pay for and keep alive — the exact cost this module exists to
avoid. The trade-off is that the window resets on restart and would not span
replicas; revisit if the service ever scales horizontally.
"""
from __future__ import annotations

import os
import threading
import time

WINDOW_SECONDS = 3600

# Generous next to real use: a student's conversations, challenges and
# assignments together land well under this. Low enough that a runaway loop is
# stopped within seconds rather than after a month of billing.
DEFAULT_PER_USER_HOURLY = 30
DEFAULT_GLOBAL_HOURLY = 300


def _limit(name: str, fallback: int) -> int:
    try:
        value = int(os.getenv(name, ""))
    except ValueError:
        return fallback
    return value if value > 0 else fallback


class SlidingWindow:
    """Counts events per key over a rolling window. Thread-safe."""

    def __init__(self, window_seconds: int = WINDOW_SECONDS):
        self._window = window_seconds
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> list[float]:
        cutoff = now - self._window
        kept = [t for t in self._events.get(key, []) if t > cutoff]
        if kept:
            self._events[key] = kept
        else:
            self._events.pop(key, None)
        return kept

    def hit_all(
        self, checks: list[tuple[str, str, int]], now: float | None = None
    ) -> tuple[bool, int, str]:
        """Check every limit first, then record the event against all of them.

        `checks` is a list of (scope, key, limit). Checking and recording happen
        under one lock so a rejected call consumes none of the counters — one
        that failed the per-user limit must not still count toward the global
        ceiling, or a single looping client would eventually lock everyone out.

        Returns (allowed, retry_after_seconds, failing_scope).
        """
        now = now if now is not None else time.time()
        with self._lock:
            pruned = {key: self._prune(key, now) for _scope, key, _limit in checks}
            for scope, key, limit in checks:
                events = pruned[key]
                if len(events) >= limit:
                    # Room frees up when the oldest event leaves the window.
                    retry_after = max(1, int(events[0] + self._window - now) + 1)
                    return False, retry_after, scope
            for _scope, key, _limit in checks:
                events = pruned[key]
                events.append(now)
                self._events[key] = events
            return True, 0, ""

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


_analyze_window = SlidingWindow()


def check_analyze(user_id: str, now: float | None = None) -> tuple[bool, int, str]:
    """Gate one /analyze call. Returns (allowed, retry_after_seconds, scope).

    The global ceiling is reported first when both are exceeded: it is the more
    serious condition, since it means total spend is already past its budget
    whatever the caller.
    """
    return _analyze_window.hit_all(
        [
            ("global", "__global__", _limit("ANALYZE_GLOBAL_HOURLY", DEFAULT_GLOBAL_HOURLY)),
            ("user", f"user:{user_id}", _limit("ANALYZE_PER_USER_HOURLY", DEFAULT_PER_USER_HOURLY)),
        ],
        now,
    )


def reset() -> None:
    """Clear all counters (tests)."""
    _analyze_window.reset()
