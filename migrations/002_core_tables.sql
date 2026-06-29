-- 002_core_tables.sql
-- Core Kernel tables: the graph, student state, and request/output logging.

-- Needed for gen_random_uuid().
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- --------------------------------------------------------------------------- --
-- concept_nodes — the KC graph nodes.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.concept_nodes (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    label                text NOT NULL,
    subject              text NOT NULL,
    level                text DEFAULT 'unknown',
    description          text DEFAULT '',
    type_kc              text DEFAULT 'conceptual',   -- procedural | declarative | conceptual
    lambda_decay         float DEFAULT 0.02,
    tau                  float DEFAULT 0.5,
    empirical_difficulty float DEFAULT 0.5,
    created_at           timestamptz DEFAULT now(),
    UNIQUE (label, subject)
);

-- --------------------------------------------------------------------------- --
-- concept_edges — prerequisite -> concept dependencies.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.concept_edges (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    concept_id      uuid NOT NULL REFERENCES kernel.concept_nodes(id) ON DELETE CASCADE,
    prerequisite_id uuid NOT NULL REFERENCES kernel.concept_nodes(id) ON DELETE CASCADE,
    weight          float DEFAULT 1.0,
    created_at      timestamptz DEFAULT now(),
    UNIQUE (concept_id, prerequisite_id)
);

-- --------------------------------------------------------------------------- --
-- student_concept_state — per-student mastery on each KC.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.student_concept_state (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             uuid NOT NULL,
    concept_id          uuid NOT NULL REFERENCES kernel.concept_nodes(id) ON DELETE CASCADE,
    mastery_score_raw   float DEFAULT 0.3,
    v_score             float DEFAULT 0.5,
    p_score             float DEFAULT 0.5,
    created_at          timestamptz DEFAULT now(),
    updated_at          timestamptz DEFAULT now(),
    UNIQUE (user_id, concept_id)
);

-- --------------------------------------------------------------------------- --
-- kernel_requests — log of every /analyze call.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.kernel_requests (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid NOT NULL,
    trigger     text,
    subject     text,
    level       text,
    payload     jsonb,
    created_at  timestamptz DEFAULT now()
);

-- --------------------------------------------------------------------------- --
-- kernel_outputs — result of each analysis.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.kernel_outputs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id      uuid REFERENCES kernel.kernel_requests(id) ON DELETE CASCADE,
    user_id         uuid NOT NULL,
    root_gap        text,
    root_concept_id uuid,
    detection_path  jsonb,
    confidence      float,
    llm_used        text,
    output          jsonb,
    created_at      timestamptz DEFAULT now()
);

-- --------------------------------------------------------------------------- --
-- individual_insights — learner-readable insight per analysis.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.individual_insights (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      uuid NOT NULL,
    request_id   uuid REFERENCES kernel.kernel_requests(id) ON DELETE SET NULL,
    root_gap     text,
    insight_text text,
    created_at   timestamptz DEFAULT now()
);
