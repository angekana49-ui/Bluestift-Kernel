# Kernel → RAYA app handoff

> Drop this into the Next.js repo (e.g. `docs/kernel-handoff.md`). It captures the
> Cognitive Kernel's current state and everything the app must align on after the
> Kernel's recent changes. Kernel repo: `github.com/angekana49-ui/Bluestift-Kernel`.

---

## 1. Status & connection

- **Live:** `https://bluestift-kernel-production.up.railway.app`
- **Health:** `GET /health` → `{ "status": "ok", "version": "1.0.0" }` (open, no auth)
- **Deep health:** `GET /ready` → `{ "read_ok", "write_ok", "status": "ok"|"degraded" }`
  returns **503** when the Kernel can't reach its DB schema (see §6). Open, no auth.

### ⚠️ Auth is now REQUIRED
The protected routes (`/analyze`, `/load_profile`, `/update_concept_state`,
`/seed_kcs`) now require a shared secret. Set the **same** value on both sides:

- App `.env.local`: `KERNEL_API_SECRET=<secret>`
- Kernel (Railway): `KERNEL_API_SECRET=<same secret>` (already set)

The Kernel accepts the secret via **any** of these headers (use whichever the
client already sends):
```
Authorization: Bearer <secret>
X-Kernel-Secret: <secret>
X-API-Key: <secret>
```
Missing/wrong secret → **401**. `/health` and `/ready` stay open.

> **Action:** confirm `lib/kernel/client.ts` sends the secret in one of the above
> headers, and that an "Analyze" call returns 200 (not 401).

---

## 2. API contract

### `POST /analyze` (main route)
Request:
```json
{
  "user_id": "uuid",                 // = public.users.id = auth.users.id
  "conversation_history": [          // max 200 messages
    { "role": "user", "content": "..." },       // content max 8000 chars
    { "role": "assistant", "content": "..." }
  ],
  "subject": "MATH",                 // any subject tag; extraction can override
  "level": "lycee",
  "trigger": "post_conversation"
}
```
Response (note the **new `alerts`** field):
```json
{
  "request_id": "uuid",
  "user_id": "uuid",
  "root_gap": "notion_de_variable",
  "root_concept_id": "uuid",
  "detection_path": ["derivation_fonction", "...", "notion_de_variable"],
  "mastery_map": { "derivation_fonction": { "k_raw": 0.2, "k_effective": 0.18, "status": "gap" } },
  "confidence": 0.95,
  "summary": "Tu bloques parce que ...",
  "recommended_path": ["notion_de_variable", "..."],
  "alerts": [{ "type": "cognitive_overload", "severity": "medium" }],
  "kernel_version": "1.0.0",
  "llm_used": "openai/gpt-oss-120b"
}
```

### `POST /load_profile`
`{ "user_id": "uuid" }` → cognitive profile (per-KC `k_raw`, `k_effective`,
`v_score`, `p_score`, `status`, `last_interaction_at`) + `mindset { m_score,
detected_mindset }`. Use for the dynamic prompt layer.

### `POST /update_concept_state`
`{ user_id, concept_id, partial_credit_score, is_assisted, response_time_ms,
blocage_type }` → updates one KC on a strong signal. Optional today (the Kernel
also derives updates from `/analyze`), but preferred for graded attempts.

---

## 3. What changed since the app was built — align on these

1. **Auth** — now enforced (see §1). Was open.
2. **`alerts` in `/analyze`** — pedagogical-safety flags. Types:
   `passive_dependency`, `false_mastery`, `re_emergence_error`,
   `cognitive_overload`, `fixed_mindset`. RAYA should react (see §4).
3. **`/ready`** — new deep-health probe. Point a deeper connectivity check at it
   (the current `/api/kernel/health` only tests liveness).
4. **Input limits** — `conversation_history` ≤ 200 messages, `content` ≤ 8000
   chars. Trim long histories before calling `/analyze` (send the last N turns).
5. **Cognitive vector semantics** (for the prompt injection, §5): V = learning
   rate p(T); P = resistance to slip modulated by mindset M.
6. **Multi-subject + cross-subject** — the graph spans subjects; a physics
   conversation can trace its root gap into maths. Just send the real `subject`.
7. **Graceful degradation** — if the shared DB regresses, `/analyze` still returns
   the diagnosis (state writes are best-effort). Watch `/ready` for `degraded`.

---

## 4. Reacting to `alerts` (pedagogical safety)

| Alert | Meaning | Suggested RAYA response |
|---|---|---|
| `passive_dependency` | Answers too fast, no errors, no questions | Switch to goal-free / demand an attempt |
| `false_mastery` | High mastery but high slip | Retest on a harder/held-out context |
| `cognitive_overload` | Frequent errors mid-solving | Reduce task complexity; worked examples |
| `fixed_mindset` | Low M, quick give-ups | Mindset intervention (process feedback) BEFORE any retry |
| `re_emergence_error` | Simple KC ok → complex KC fails | Decompose the KC |

---

## 5. Injecting the cognitive vector into RAYA's prompt

From `/load_profile`, inject per active KC: **K** (mastery), **V** (learning
rate), **P** (persistence), and the global **M** (mindset). Drives the EMT entry
level: low K+P → vicarious/assertion; solid K+P → pump; low M → deflect to content
before any retry.

---

## 6. Shared-DB rules (IMPORTANT — don't lock the Kernel out)

The Kernel and the app share one Supabase project. The app's setup MUST NOT:
- drop `kernel` from the PostgREST **exposed schemas**, or
- reset the `service_role` grants on the `kernel` schema.

The exposed schemas must be the **union** both sides need:
```
public, graphql_public, kernel, learning, schools, rag, content
```
If the Kernel ever returns `degraded` on `/ready` (or 500s with "permission
denied for table kernel_*"), re-run the Kernel's `migrations/009_shared_db_hardening.sql`
(re-asserts the union + grants + reloads PostgREST).

---

## 7. Still on the app side (from your raya-status)

- Store `emt_level` on RAYA messages (light EMT classification).
- Call `/update_concept_state` directly with real `partial_credit_score` /
  `concept_id` on graded attempts (not only the `/analyze`-derived updates).
- Add a `/ready`-based deep health check alongside the liveness probe.
- Keep the chat hot path non-blocking on the Kernel (already the case).
