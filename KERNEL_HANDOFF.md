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

### ⚠️ Auth: two tiers

The protected routes (`/analyze`, `/load_profile`, `/update_concept_state`,
`/load_alerts`, `/resolve_alert`, `/seed_kcs`) require a credential. `/health`
and `/ready` stay open.

**Service tier — the shared secret.** Unchanged, and still what most calls use.
Set the **same** value on both sides:

- App `.env.local`: `KERNEL_API_SECRET=<secret>`
- Kernel (Railway): `KERNEL_API_SECRET=<same secret>`

Accepted via any of these headers, whichever the client already sends:
```
Authorization: Bearer <secret>
X-Kernel-Secret: <secret>
X-API-Key: <secret>
```

This secret can read and write **any** student's cognitive profile. It is the
right credential for background work — a cache refresh, a fire-and-forget
analysis after a chat turn — which has no live session to borrow from.

**User tier — the student's own Supabase token.** Send the student's
`access_token` as `Authorization: Bearer <token>` and the Kernel scopes the call
to them: it checks the token's `sub` against the `user_id` in the body and
returns **403** for anyone else's profile. `/seed_kcs`, `/resolve_alert` and the
school scope of `/load_alerts` are service-only (403).

Use it on any call that is about one student who is right there — the profile
and analyze proxies. It means a leak of the service secret is not a skeleton key
to every child's cognitive profile.

> **Deployment order matters.** The Kernel only accepts user tokens once
> `SUPABASE_JWT_SECRET` (the Supabase project's JWT secret) is set on Railway.
> The app therefore gates this behind `KERNEL_USER_SCOPED_AUTH=1`, off by
> default. Set the Kernel's variable **first**, then flip the app's — the other
> order 401s every scoped call.

Neither tier limits what the Kernel can model. Any student, any subject, any
level, KCs created on the fly. This is only about who may ask about whom.

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
  "curriculum": {
    "school_id": "uuid",
    "layers_applied": ["curriculum", "kc_priorities", "objectives"],
    "objectives": [
      { "concept": "derivees", "target_mastery": 0.8, "observed_mastery": 0.4,
        "due_at": "2026-12-15T00:00:00Z", "status": "at_risk" }
    ],
    "root_gap_in_program": true,
    "rules": ["Toujours partir d'un exemple concret."]
  },
  "kernel_version": "1.0.0",
  "llm_used": "openai/gpt-oss-120b"
}
```

`curriculum` is **absent (null)** unless the student belongs to a school that has
set layers — most students won't have it, so treat it as optional. When present:
`recommended_path` has already been reordered by the school's priorities;
`objectives[].status` is one of `met` / `at_risk` / `overdue` / `pending` /
`unknown` (`unknown` = no evidence on that concept yet, not a failure); and
`rules` are the school's instructions for RAYA's prompt — the same intent as the
app's existing `class_instructions` / `school_directives` nudges.

### `POST /load_profile`
`{ "user_id": "uuid" }` → cognitive profile (per-KC `k_raw`, `k_effective`,
`v_score`, `p_score`, `status`, `last_interaction_at`) + `mindset { m_score,
detected_mindset }`. Use for the dynamic prompt layer.

### `POST /update_concept_state`
`{ user_id, concept_id | concept_label, subject, level, partial_credit_score,
is_assisted, response_time_ms, blocage_type }` → updates one KC on a strong
signal. **Use this for every graded attempt** — it carries the real score, where
`/analyze` can only re-infer one from prose.

Identify the KC either way:
- `concept_label` — a plain concept name (`"derivation_fonction"`). The Kernel
  canonicalizes it onto an existing KC, or creates it. This is the normal path:
  grading knows the concept's name, never its UUID.
- `concept_id` — a `kernel.concept_nodes` UUID. The response returns the resolved
  `concept_id` and canonical `label`, so a caller can cache the id and skip
  resolution next time.

> **`/analyze` is rate-limited.** Two LLM calls per request, on a host billed
> by the minute, with no ceiling was a way to burn the credit balance and the
> LLM quota at once. Limits: **30 calls per student per hour**, **300 per hour
> overall** (both tunable). Over the limit → **429** with a `Retry-After`
> header. Treat it as "come back later", never as a failure to retry
> immediately — an immediate retry is what trips it. The fire-and-forget call
> sites already swallow it harmlessly.

### `POST /load_alerts` (new — the school dashboard's data source)

The Kernel has been *writing* pedagogical-safety alerts all along; this is the
first way to read them back. Exactly one scope per call:

```json
{ "user_id": "uuid" }              // one student — their own token works
{ "school_id": "uuid" }            // every student of a school — service only
```
Optional: `include_resolved` (default false), `severity` (`low`/`medium`/`high`),
`since` (ISO timestamp), `limit` (1–500, default 100).

```json
{
  "scope": "school",
  "school_id": "uuid",
  "students_in_scope": 42,
  "alerts": [{
    "id": "uuid", "user_id": "uuid",
    "concept_id": "uuid", "concept_label": "derivation_fonction",
    "alert_type": "cognitive_overload", "alert_severity": "high",
    "alert_details": {}, "inconsistency_rate": null, "volatility_score": null,
    "interactions_count": null,
    "resolved": false, "resolved_by": null, "resolved_at": null,
    "created_at": "2026-08-14T09:12:00Z"
  }],
  "counts_by_type": { "cognitive_overload": 3 },
  "counts_by_severity": { "high": 3 },
  "truncated": false
}
```

**The school scope is service-only, and that is your job to gate.** The Kernel
cannot tell a teacher's token from a parent's or a student's — it has no staff
directory. So it refuses to decide: the app checks that the caller really teaches
at that school, then calls with the service secret. If you skip that check, one
forged session opens a whole school's alerts.

Three fields exist to keep a dashboard honest, please surface them:
`students_in_scope` (a school expecting 300 and seeing 12 has a roster problem,
not a quiet week), `truncated` (the limit cut the list — don't render it as
"everything"), and the **503**: this route deliberately fails loudly instead of
returning `[]`, because an empty list on a safety screen reads as "all clear".
Show an error state, never an empty one.

### `POST /resolve_alert`
`{ alert_id, resolved_by, resolved }` → acknowledge an alert, or reopen it with
`resolved: false`. Service-only: a student must not be able to close the alert
raised about them. `resolved_by` is free text (the Kernel has no staff
directory) — pass the teacher's id or name; you vouch for it. Unknown id → 404.

Resolved alerts leave the default `/load_alerts` view, so the dashboard shrinks
as staff work through it. A list that never shrinks is a list people stop reading.

> **Pairing it with `/analyze`:** send `commit_state: false` on the `/analyze`
> call that follows graded updates. Otherwise the Kernel re-derives the same
> attempts from the conversation and commits them *on top of* yours — the same
> evidence counted twice, inflating mastery. The diagnosis (root gap, path,
> alerts) is returned either way.

---

## 3. What changed since the app was built — align on these

1. **Auth** — now enforced (see §1). Was open.
2. **`alerts` in `/analyze`** — pedagogical-safety flags. Types:
   `passive_dependency`, `false_mastery`, `re_emergence_error`,
   `cognitive_overload`, `fixed_mindset`, `inconsistency_high`,
   `ood_distribution`. RAYA should react (see §4).
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
| `inconsistency_high` | Mastery estimate oscillates instead of settling | Treat the KC's K as unreliable: re-establish with a clean, unassisted check before sequencing on it |
| `ood_distribution` | The student doesn't match the population the parameters were calibrated on | Don't harden decisions on K here; `direction: below_population` in the details is the silent-failure case and warrants a human look |

The last two read beyond a single conversation (trajectory history, population
baselines), so they surface on students with some history rather than on turn one.

Alerts arrive live in the `/analyze` response *and* are persisted. Read the
history back with `/load_alerts` — that's what the school dashboard is built on —
and close them with `/resolve_alert`.

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

The Kernel also **reads** two app-owned tables in the `schools` schema:
`student_identities` (to map a student to their school) and
`school_curriculum_layers` (the layers themselves, extended by the Kernel's
migration 010). It never writes to them. If the app changes their shape, the
Kernel degrades to "no school context" rather than failing — but the school
channel goes quiet, so tell the Kernel side.

---

## 7. Still on the app side (from your raya-status)

- Store `emt_level` on RAYA messages (light EMT classification).
- Call `/update_concept_state` directly with real `partial_credit_score` /
  `concept_id` on graded attempts (not only the `/analyze`-derived updates).
- Add a `/ready`-based deep health check alongside the liveness probe.
- Keep the chat hot path non-blocking on the Kernel (already the case).
