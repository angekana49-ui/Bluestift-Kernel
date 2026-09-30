-- 011_learning_events.sql
-- Keep the evidence, not just the estimate.
--
-- Until now the Kernel stored only its current belief per (student, KC) and a
-- trajectory of that belief. The observations themselves were thrown away, so
-- the BKT parameters (p_init, p_transit, p_slip, p_guess) could never be fitted
-- to our own students: every KC ran forever on literature priors measured on
-- North-American datasets. learning_events is the raw material for that fit
-- (see core/calibration.py::fit_bkt_em). One row per graded attempt.
--
-- Additive and idempotent.

CREATE TABLE IF NOT EXISTS kernel.learning_events (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      uuid NOT NULL,
    concept_id   uuid NOT NULL REFERENCES kernel.concept_nodes(id) ON DELETE CASCADE,
    credit       float NOT NULL CHECK (credit >= 0 AND credit <= 1),
    is_assisted  boolean NOT NULL DEFAULT false,
    blocage_type text,                -- conceptual | linguistic | ambiguous | none
    counted      boolean NOT NULL DEFAULT true,  -- false: logged but not used as evidence
    source       text NOT NULL,       -- analyze | update_concept_state
    request_id   uuid,                -- the /analyze request, when there is one
    k_before     float,               -- the Kernel's belief when the attempt was made
    created_at   timestamptz DEFAULT now()
);

-- Calibration reads one KC at a time, in order per student.
CREATE INDEX IF NOT EXISTS learning_events_concept_idx
  ON kernel.learning_events (concept_id, user_id, created_at);
-- Erasure and export (GDPR) read one student at a time.
CREATE INDEX IF NOT EXISTS learning_events_user_idx
  ON kernel.learning_events (user_id);

-- Service-only, like the other internal tables (see 005).
ALTER TABLE kernel.learning_events ENABLE ROW LEVEL SECURITY;
GRANT ALL ON kernel.learning_events TO service_role;

-- The dual mastery condition asks for solid credit on the LAST THREE AUTONOMOUS
-- attempts — not an all-time average that assisted work inflates.
-- Deliberately NO default: NULL marks a row written before this migration, for
-- which the Kernel falls back to the all-time average until new attempts fill
-- the list. A '[]' default would silently strip "mastered" from every existing
-- student on deploy.
ALTER TABLE kernel.student_concept_state
  ADD COLUMN IF NOT EXISTS recent_autonomous_credits jsonb;

COMMENT ON TABLE kernel.learning_events IS
  'One row per graded attempt; the evidence BKT parameters are fitted on. See migration 011.';
COMMENT ON COLUMN kernel.student_concept_state.recent_autonomous_credits IS
  'Credits of the last 3 unassisted attempts, oldest first. Feeds the dual mastery condition.';

NOTIFY pgrst, 'reload schema';
