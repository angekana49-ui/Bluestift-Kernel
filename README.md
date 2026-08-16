# Bluestift Cognitive Kernel

The algorithm underneath Bluestift — a new way to envision AI tutoring.

The Kernel is the **brain** of Bluestift: an independent Python/FastAPI service
that turns learning interactions into **root-gap detection**. It is fully
decoupled from every interface. RAYA (Next.js) calls it over HTTP — the Kernel
knows nothing about React or client-side auth. It is autonomous.

> **Status:** v1 complete. Hosted on Railway (Hobby) with Serverless enabled —
> see [Deploy](#deploy) for the cost rules that keep it inside the plan's
> included usage.
> **Live:** https://bluestift-kernel-production.up.railway.app
> See [POST_MVP_ROADMAP.md](POST_MVP_ROADMAP.md) for what comes next.

---

## What it does

Given a student↔RAYA conversation, the Kernel:

1. **Extracts** the knowledge components (KCs) mentioned, the attempts, and
   behavioural signals (LLM).
2. **Creates KCs on the fly** for any subject — the graph is open, not a fixed list.
3. **Decays** stored mastery over time (exponential forgetting).
4. **Updates** the cognitive vector with Bayesian Knowledge Tracing on strong signals.
5. **Walks** the prerequisite graph (DFS) to find the *deepest* underlying gap,
   chosen by **convergence** (the concept many failing KCs trace back to) —
   **across subjects**: a physics struggle can trace to a maths root.
6. **Detects anomalies** (false mastery, passive dependency, cognitive overload,
   fixed mindset, re-emergence errors) for pedagogical safety.
7. **Explains** the gap in one learner-friendly sentence.
8. **Logs** everything to Supabase and returns a structured analysis.

Its parameters are **living**: BKT and decay values start from literature priors,
then self-calibrate from real data (per student, per KC).

---

## The cognitive vector (K, V, P, M)

Each KC, per student, carries a four-dimensional state (Luckin / corpus §1.2):

| Dim | Meaning | Operationalisation |
|-----|---------|--------------------|
| **K** | Mastery probability | BKT p(L); dual mastery criterion (K ≥ 0.95 AND low slip AND partial-credit ≥ 0.7) |
| **V** | Learning rate (individualised p(T), Yudelson) | smoothed fraction of the remaining mastery gap closed per trial |
| **P** | Persistence / resistance to slip (Corbett) | (1 − personal p(S)), modulated by mindset M |
| **M** | Mindset (Dweck) | sigmoid of behavioural signals, bounded to [0.05, 0.95]; global per student |

---

## Architecture

```
.
├── main.py                  # FastAPI app + routes
├── core/
│   ├── graph.py             # NetworkX Kernel Graph (KCs + edges)
│   ├── bkt.py               # Bayesian Knowledge Tracing + mastery criterion
│   ├── forgetting.py        # Exponential decay -> K_effective, 3-level lambda
│   ├── mindset.py           # Mindset score M (sigmoid blend)
│   ├── detector.py          # DFS root-cause by convergence
│   ├── calibration.py       # Self-calibration of living parameters
│   ├── anomaly.py           # Pedagogical-safety anomaly detection
│   └── curriculum.py        # School curriculum layers -> sequencing + objectives
├── services/
│   ├── llm.py               # LLM chain: Groq primary -> Gemini fallback (REST)
│   ├── kc_registry.py       # get_or_create_kc() — dynamic KCs
│   ├── db.py                # Supabase read/write (service_role)
│   ├── analyze.py           # /analyze pipeline orchestration
│   └── graph_builder.py     # Offline curriculum-graph builder (cold-start)
├── models/schemas.py        # Pydantic request/response models
├── seed/math_kcs.py         # Starter Math KCs (examples, not exhaustive)
├── scripts/
│   ├── build_graph.py       # CLI: distill a KC graph from the LLMs
│   ├── build_bridges.py     # CLI: generate cross-subject prerequisite bridges
│   └── apply_migrations.py  # CLI: apply migrations via the Management API
├── migrations/              # 10 numbered Supabase SQL migrations
├── conftest.py              # In-memory fake Supabase for tests
├── test_kernel.py           # 60 tests
├── requirements.txt / requirements-dev.txt
└── Procfile / railway.toml / render.yaml
```

---

## API

| Method | Route                    | Auth | Purpose                                            |
|--------|--------------------------|------|----------------------------------------------------|
| GET    | `/health`                | open | Liveness + version.                                |
| GET    | `/ready`                 | open | Deep health — kernel-schema read+write (503 if degraded). |
| POST   | `/analyze`               | 🔒   | **Main route.** Conversation → root-gap + alerts.  |
| POST   | `/load_profile`          | 🔒   | Full cognitive profile with K_effective recomputed. |
| POST   | `/update_concept_state`  | 🔒   | Manual KC update on a strong signal (called by RAYA). |
| POST   | `/seed_kcs`              | 🔒   | Seed starter Math KCs if the table is empty.       |

🔒 = authenticated. Two tiers of caller:

- **service** — presents `KERNEL_API_SECRET` (via `Authorization: Bearer`,
  `X-Kernel-Secret`, or `X-API-Key`). A trusted backend acting for many
  students: may touch any `user_id`, and is the only tier allowed to `/seed_kcs`.
- **user** — presents a Supabase access token. Scoped to one student: the Kernel
  checks the token's `sub` against the `user_id` in the body and returns **403**
  for anyone else's. Requires `SUPABASE_JWT_SECRET`; without it this tier is
  simply unavailable.

Unrecognised or missing credentials → **401**. `KERNEL_API_SECRET` unset = open
(dev only). This bounds *who may ask about whom*; it places no limit on what the
Kernel can model — any student, any subject, KCs created on the fly.

See [KERNEL_HANDOFF.md](KERNEL_HANDOFF.md) for the app-integration contract.

Interactive docs at `/docs`.

### Example: `POST /analyze`

```json
{
  "user_id": "11111111-1111-1111-1111-111111111111",
  "conversation_history": [
    { "role": "user", "content": "Je comprends pas les dérivées" },
    { "role": "assistant", "content": "Rappelle-moi ce qu'est une fonction affine" },
    { "role": "user", "content": "C'est... f(x) = ax ?" }
  ],
  "subject": "MATH",
  "level": "lycee",
  "trigger": "post_conversation"
}
```

Returns `root_gap`, `detection_path` (surface → root chain), `mastery_map`,
`confidence`, `summary`, `recommended_path`, `alerts` (pedagogical-safety flags),
plus `kernel_version` and `llm_used`.

---

## Local setup

Requires **Python 3.11+** (3.12 pinned for deploy via `.python-version`).

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash)
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt -r requirements-dev.txt

uvicorn main:app --reload --port 8000
```

Create a `.env` (never committed — every env file is gitignored, including
examples) with:

| Variable | | |
|---|---|---|
| `SUPABASE_URL` | required | `https://<project-ref>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | required | the **service_role** key, never the anon key |
| `GROQ_API_KEY` | required | primary LLM |
| `GEMINI_API_KEY` | required | fallback LLM — set both in production, or one rate limit takes `/analyze` down |
| `KERNEL_API_SECRET` | prod | service-tier secret for the 🔒 routes; must match the RAYA app's. Empty = auth disabled (local only) |
| `SUPABASE_JWT_SECRET` | recommended | the Supabase project's JWT secret, enabling the user tier (per-student scoped tokens). Unset = only the service secret is accepted |
| `KERNEL_VERSION` | optional | reported by `/health`; defaults to `1.0.0` |
| `CORS_ORIGINS` | optional | comma-separated; defaults to the RAYA + schools domains and `localhost:3000` |
| `ANALYZE_PER_USER_HOURLY` | optional | `/analyze` calls allowed per student per hour (default 30). Stops a client retry loop from burning credits and LLM quota |
| `ANALYZE_GLOBAL_HOURLY` | optional | total `/analyze` calls per hour across everyone (default 300). The ceiling that catches a leaked service secret |
| `SUPABASE_ACCESS_TOKEN` | migrations only | personal token (`sbp_...`) for `scripts/apply_migrations.py`; not needed by the running service |

Open http://localhost:8000/health → `{"status":"ok", ...}`.

> **Corporate-proxy TLS note:** on managed Windows machines, outbound TLS may be
> intercepted. The kernel uses [`truststore`](https://pypi.org/project/truststore/)
> (in requirements) to trust the OS certificate store automatically. For Node/CLI
> tools, export the OS roots to a PEM and set `NODE_EXTRA_CA_CERTS`.

### LLM models

- Primary (Groq): `openai/gpt-oss-120b`
- Fallback (Gemini, direct REST via httpx — no SDK): `gemini-3.1-flash-lite`

The chain never crashes `/analyze`: Groq → Gemini, and only raises if both fail.

### Database migrations

Apply the ten SQL files **in order** in the Supabase SQL editor (or via the
Management API with `scripts/apply_migrations.py`):

```
001_schemas.sql              # schemas: kernel, learning, schools, rag, content
002_core_tables.sql          # concept_nodes, concept_edges, student_concept_state, logs
003_kernel_patch_v2.sql      # living-parameter columns + mindset/curriculum/monitoring/trajectories
004_extensions_rls.sql       # pgvector, indexes, RLS policies
005_rls_internal_tables.sql  # lock internal log tables (service-only)
006_grant_service_role.sql   # grant service_role on custom schemas
007_rls_policies.sql         # full RLS policy set
008_kernel_monitoring_alerts.sql  # pedagogical-safety alert schema
009_shared_db_hardening.sql       # re-assert exposed schemas + grants (shared DB)
010_school_curriculum_layers.sql  # school layer types + payloads (School -> AI -> Student)
```

> **Shared DB:** the Kernel shares its Supabase project with the RAYA app. If the
> app's setup ever drops `kernel` from the exposed schemas or resets grants,
> re-run `009_shared_db_hardening.sql` (it re-asserts the union + grants + reload).

> PostgREST must **expose** the `kernel`, `schools`, `rag` schemas
> (Dashboard → Project Settings → API → Exposed schemas).

Then seed the starter graph (or build a dense one — see below):

```bash
curl -X POST http://localhost:8000/seed_kcs
```

---

## Cold-start: build a curriculum graph from the LLMs

Instead of waiting for real data, distill a dense, canonical prerequisite graph
out of the public models (closed-world prerequisites, cross-model corroboration,
DAG validation):

```bash
# Inspect only (no DB writes):
python scripts/build_graph.py MATH cycle3 cycle4 lycee --dry-run

# Generate and persist:
python scripts/build_graph.py MATH cycle3 cycle4 lycee
```

The LLM supplies the **structure** (nodes + edges); real student data later
calibrates the **parameters** (difficulty, decay) — the flywheel.

### Cross-subject bridges

The graph spans subjects in one table, and the detector traverses any edge — so
once **bridge** edges exist, a physics struggle can trace to a maths root. Generate
them closed-world over two subjects' vocabularies (foundational → applied),
cross-model corroborated and DAG-validated:

```bash
python scripts/build_bridges.py PHYSICS MATH --dry-run   # inspect
python scripts/build_bridges.py PHYSICS MATH             # persist
```

Example: `acceleration_moyenne` (physics) → `derivation_fonction` (maths) — no
detector change needed; the convergence search crosses the bridge automatically.

---

## Tests

```bash
pytest -q
```

60 tests. The suite mocks the LLM and uses an in-memory fake Supabase
(`conftest.py`), so **no network or real keys are required**. Coverage: BKT,
forgetting, mindset, detector (convergence), calibration, the cognitive vector
(V/P/slip), anomaly detectors (including temporal inconsistency and OOD),
school curriculum layers, graph-builder validation, `get_or_create_kc`, and the
`/health`, `/ready`, `/analyze`, `/load_profile`, `/update_concept_state`,
`/seed_kcs` routes.

---

## Deploy

### Railway (primary)

`railway.toml` (health check `/health`). Set the env vars from the table above
in the Railway dashboard — not from a file. Deployed via:

```bash
railway up --detach --service bluestift-kernel
```

**Keeping it inside the plan.** Railway bills per minute while the container
runs, so the service is built to spend as little time running as possible. The
Kernel is called fire-and-forget after a conversation, never in the chat's
critical path, so sleeping costs no student anything.

- **Enable Serverless** on the service (dashboard — there is no config key for
  it). Railway sleeps a service after 10 minutes with **no outbound traffic**.
- **Nothing may hold a connection open at rest.** No database pooler, no
  realtime websocket, no telemetry — any of them and the service never sleeps.
  The Supabase client is HTTP-only and opens nothing when idle, which is what
  makes this work. Verify before adding a dependency that keeps a socket.
- **Never point an uptime monitor at `/health`.** A ping every few minutes keeps
  the container awake around the clock and turns a ~$0.10/month service into a
  ~$1/month one. `scripts/check_prod.sh` is for running by hand, not on a timer.
- `/analyze` is rate-limited (see the env table): two LLM calls per request and
  no ceiling was a way to burn both credits and LLM quota on a retry loop.

At ~57 MB resident, the service costs roughly $0.10/month asleep most of the day
and under $1/month even if it never slept — inside the Hobby plan's included $5
either way. It was the pre-slimming footprint, at ~100 MB, that could not fit the
Free plan's $1 credit and got the deployment stopped.

### Render (fallback)

`render.yaml` is complete and current — blueprint, plan, region, and all six
secrets declared. Usable as-is if Railway is ever the wrong answer.

Both hosts also work via the `Procfile`:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Design rules

- **service_role only** — never the anon key.
- **No DDL from the Kernel** — only INSERT/UPDATE/SELECT (DDL lives in migrations).
- **Every `/analyze` is logged** — a `kernel_request` row is written even on error.
- **Automatic LLM fallback** — Groq → Gemini; never crashes if one provider is down.
- **Dynamic KCs** — `get_or_create_kc()` for any subject, no closed list.
- **Strict Pydantic validation** on all inputs/outputs.
- **Living parameters** — priors from the literature, refined from real data.
- **Pedagogical safety** — anomalies flagged to `kernel_monitoring` and returned to RAYA.
- **Free tier only** — Supabase, Groq, Gemini, Railway/Render.

See [POST_MVP_ROADMAP.md](POST_MVP_ROADMAP.md) for everything beyond v1.
