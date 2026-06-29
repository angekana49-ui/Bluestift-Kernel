# Bluestift Cognitive Kernel

The algorithm underneath Bluestift — a new way to envision AI tutoring.

The Kernel is the **brain** of Bluestift: an independent Python/FastAPI service
that takes learning interactions in and produces **root-gap detection** out. It
is fully decoupled from every interface. RAYA (Next.js) calls it over HTTP — the
Kernel knows nothing about React or client-side auth. It is autonomous.

---

## What it does

Given a student↔tutor conversation, the Kernel:

1. **Extracts** the knowledge components (KCs) mentioned and evaluates attempts (LLM).
2. **Creates KCs on the fly** for any subject — the graph is open, not a fixed list.
3. **Decays** stored mastery over time (exponential forgetting).
4. **Updates** mastery with Bayesian Knowledge Tracing (BKT) on strong signals.
5. **Walks** the prerequisite graph (DFS) to find the *deepest* underlying gap.
6. **Explains** the gap in one learner-friendly sentence.
7. **Logs** everything to Supabase and returns a structured analysis.

Its parameters are **living**: BKT and decay values start from literature priors,
then self-calibrate from real data (per student, per KC).

---

## Architecture

```
.
├── main.py                  # FastAPI app + routes
├── core/
│   ├── graph.py             # NetworkX Kernel Graph (KCs + edges)
│   ├── bkt.py               # Bayesian Knowledge Tracing + mastery criterion
│   ├── mindset.py           # Mindset score M (sigmoid blend)
│   ├── forgetting.py        # Exponential decay -> K_effective
│   ├── detector.py          # DFS root-cause detection
│   └── calibration.py       # Self-calibration of living parameters
├── services/
│   ├── llm.py               # LLM chain: Groq primary -> Gemini fallback
│   ├── kc_registry.py       # get_or_create_kc() — dynamic KCs
│   ├── db.py                # Supabase read/write (service_role)
│   └── analyze.py           # /analyze pipeline orchestration
├── models/
│   └── schemas.py           # Pydantic request/response models
├── seed/
│   └── math_kcs.py          # Starter Math KCs (examples, not exhaustive)
├── migrations/              # 4 numbered Supabase SQL migrations
├── conftest.py              # In-memory fake Supabase for tests
├── test_kernel.py           # Test suite
├── requirements.txt
├── Procfile / railway.toml / render.yaml
└── .env.example
```

---

## API

| Method | Route                    | Purpose                                            |
|--------|--------------------------|----------------------------------------------------|
| GET    | `/health`                | Liveness + version.                                |
| POST   | `/analyze`               | **Main route.** Conversation → root-gap detection. |
| POST   | `/load_profile`          | Full cognitive profile with K_effective recomputed. |
| POST   | `/update_concept_state`  | Manual KC update on a strong signal (called by RAYA). |
| POST   | `/seed_kcs`              | Seed starter Math KCs if the table is empty.       |

Interactive docs at `/docs` once running.

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

Returns `root_gap`, `detection_path`, `mastery_map`, `confidence`, `summary`,
`recommended_path`, plus `kernel_version` and `llm_used`.

---

## Local setup

Requires **Python 3.11+**.

```bash
# 1. Clone and enter the repo
cd Bluestift-Kernel

# 2. Create a virtual environment
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash)
# source .venv/bin/activate        # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
#   then fill in SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY, GEMINI_API_KEY

# 5. Run
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/health → `{"status":"ok", ...}`.

### Database migrations

Apply the four SQL files **in order** in the Supabase SQL editor (or via the CLI):

```
migrations/001_schemas.sql          # schemas: kernel, learning, schools, rag, content
migrations/002_core_tables.sql      # concept_nodes, concept_edges, student_concept_state, logs
migrations/003_kernel_patch_v2.sql  # living-parameter columns + mindset/curriculum/monitoring/trajectories
migrations/004_extensions_rls.sql   # pgvector, indexes, RLS policies
```

Then seed the starter Math graph:

```bash
curl -X POST http://localhost:8000/seed_kcs
```

> **Keys:** the Kernel uses the Supabase **`service_role`** key, never the anon
> key. It issues **DML only** — all DDL lives in the migration files.

---

## Tests

```bash
pytest -q
```

The suite mocks the LLM and uses an in-memory fake Supabase (`conftest.py`), so
**no network or real keys are required**. It covers the algorithm units (BKT,
forgetting, mindset, detector, calibration), `get_or_create_kc()`, and the
`/health`, `/analyze`, `/load_profile`, `/seed_kcs` routes.

---

## Deploy

### Railway (primary)

Push the repo; Railway picks up `railway.toml` (NIXPACKS builder, health check on
`/health`). Set the env vars from `.env.example` in the project settings.

### Render (fallback)

`render.yaml` defines the web service. Cold starts (~30s on free tier) are
acceptable for the MVP. Set the secret env vars (`sync: false`) in the dashboard.

Both platforms also work via the `Procfile`:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Design rules

- **service_role only** — never the anon key.
- **No DDL from the Kernel** — only INSERT/UPDATE/SELECT.
- **Every `/analyze` is logged** — a `kernel_request` row is written even on error.
- **Automatic LLM fallback** — Groq → Gemini; the service never crashes if one provider is down.
- **Dynamic KCs** — `get_or_create_kc()` for any subject, no closed list.
- **Strict Pydantic validation** on all inputs/outputs.
- **Living parameters** — priors from the literature, refined from real data.
- **Free tier only** — Supabase, Groq, Gemini, Railway/Render.

## Not in this build (post-MVP)

JWT auth on the Kernel · GraphRAG multi-hop · neural Responsible-DKT (MVP is
Bayesian BKT + LLM heuristics) · learner simulator · realtime WebSocket · offline cache.
