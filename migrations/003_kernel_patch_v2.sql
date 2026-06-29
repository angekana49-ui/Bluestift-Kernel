-- 003_kernel_patch_v2.sql
-- Patch v2: extra living-parameter columns + mindset, curriculum, monitoring
-- and trajectory tables.

-- --------------------------------------------------------------------------- --
-- student_concept_state — extra mastery + personal living-parameter columns.
-- --------------------------------------------------------------------------- --
ALTER TABLE kernel.student_concept_state ADD COLUMN IF NOT EXISTS mastery_score_effective float;
ALTER TABLE kernel.student_concept_state ADD COLUMN IF NOT EXISTS struggle_index int DEFAULT 0;
ALTER TABLE kernel.student_concept_state ADD COLUMN IF NOT EXISTS partial_credit_avg float;
ALTER TABLE kernel.student_concept_state ADD COLUMN IF NOT EXISTS last_strong_signal_at timestamptz;
ALTER TABLE kernel.student_concept_state ADD COLUMN IF NOT EXISTS lambda_personal float;
ALTER TABLE kernel.student_concept_state ADD COLUMN IF NOT EXISTS p_slip_personal float;
ALTER TABLE kernel.student_concept_state ADD COLUMN IF NOT EXISTS interactions_on_kc int DEFAULT 0;

-- --------------------------------------------------------------------------- --
-- concept_nodes — empirical BKT params + calibration bookkeeping.
-- --------------------------------------------------------------------------- --
ALTER TABLE kernel.concept_nodes ADD COLUMN IF NOT EXISTS p_init float DEFAULT 0.3;
ALTER TABLE kernel.concept_nodes ADD COLUMN IF NOT EXISTS p_transit float DEFAULT 0.1;
ALTER TABLE kernel.concept_nodes ADD COLUMN IF NOT EXISTS p_slip float DEFAULT 0.1;
ALTER TABLE kernel.concept_nodes ADD COLUMN IF NOT EXISTS p_guess float DEFAULT 0.2;
ALTER TABLE kernel.concept_nodes ADD COLUMN IF NOT EXISTS interactions_count int DEFAULT 0;
ALTER TABLE kernel.concept_nodes ADD COLUMN IF NOT EXISTS last_calibration_at timestamptz;

-- --------------------------------------------------------------------------- --
-- student_mindset_state — mindset score M per student.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.student_mindset_state (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          uuid NOT NULL UNIQUE,
    m_score          float DEFAULT 0.5,
    detected_mindset text DEFAULT 'mixed',   -- fixed | mixed | growth
    abandon_rate         float,
    persistence_score    float,
    time_on_task         float,
    interaction_quality  float,
    updated_at       timestamptz DEFAULT now()
);

-- --------------------------------------------------------------------------- --
-- school_curriculum_layers — curriculum layers per school.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS schools.school_curriculum_layers (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    school_id   uuid NOT NULL,
    subject     text NOT NULL,
    level       text NOT NULL,
    layer_name  text NOT NULL,
    concept_ids jsonb DEFAULT '[]'::jsonb,
    created_at  timestamptz DEFAULT now()
);

-- --------------------------------------------------------------------------- --
-- kernel_monitoring — production alerts and metrics.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.kernel_monitoring (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    level      text NOT NULL,        -- info | warn | error
    event      text NOT NULL,
    detail     jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now()
);

-- --------------------------------------------------------------------------- --
-- learning_trajectories — temporal snapshots of a student's mastery.
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS kernel.learning_trajectories (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid NOT NULL,
    concept_id  uuid REFERENCES kernel.concept_nodes(id) ON DELETE CASCADE,
    k_raw       float,
    k_effective float,
    snapshot_at timestamptz DEFAULT now()
);
