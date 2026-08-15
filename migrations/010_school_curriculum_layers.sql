-- 010_school_curriculum_layers.sql
-- The School → AI → Student channel: give the layers table a shape the Kernel
-- can actually consume.
--
-- schools.school_curriculum_layers has existed since 003 but only ever held a
-- name and a list of concept_ids, and nothing read it. A school needs to say
-- four different kinds of thing, each with its own payload:
--
--   curriculum    — the concepts that belong to the official program. Analysis
--                   is anchored to these instead of the open graph.
--   kc_priorities — weight multipliers per concept. What the school wants taught
--                   first, feeding the sequencing decision.
--   objectives    — mastery targets with deadlines, per concept.
--   custom_rules  — instructions for RAYA's prompt (consumed app-side).
--
-- Additive and idempotent: existing rows keep working and are read as the
-- `curriculum` type, which is what a bare concept_ids list always meant.

ALTER TABLE schools.school_curriculum_layers
  ADD COLUMN IF NOT EXISTS layer_type text NOT NULL DEFAULT 'curriculum';
ALTER TABLE schools.school_curriculum_layers
  ADD COLUMN IF NOT EXISTS payload jsonb DEFAULT '{}'::jsonb;
ALTER TABLE schools.school_curriculum_layers
  ADD COLUMN IF NOT EXISTS is_active boolean DEFAULT true;
ALTER TABLE schools.school_curriculum_layers
  ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

ALTER TABLE schools.school_curriculum_layers
  DROP CONSTRAINT IF EXISTS school_curriculum_layers_type_chk;
ALTER TABLE schools.school_curriculum_layers
  ADD CONSTRAINT school_curriculum_layers_type_chk
  CHECK (layer_type IN ('curriculum', 'kc_priorities', 'objectives', 'custom_rules'));

-- The Kernel's read is always "the active layers for this school, subject and
-- level", on the hot path of /analyze.
CREATE INDEX IF NOT EXISTS school_curriculum_layers_lookup_idx
  ON schools.school_curriculum_layers (school_id, is_active, subject, level);

-- Payload shapes, by layer_type. The Kernel tolerates a missing or malformed
-- payload (it degrades to "no layer"), so these are conventions, not constraints:
--
--   kc_priorities : {"weights": {"derivation_fonction": 2.0, "fractions": 1.5}}
--                   A multiplier per concept label; > 1 pulls the concept
--                   earlier in the recommended path, < 1 pushes it later.
--   objectives    : {"targets": [{"concept": "fractions", "mastery": 0.8,
--                                 "due_at": "2026-12-15T00:00:00Z"}]}
--   curriculum    : {"concepts": ["fractions", "..."]}  (or the legacy
--                   concept_ids column, which is read as the same thing)
--   custom_rules  : {"rules": ["..."]}  — RAYA's prompt layer, not the Kernel's.

COMMENT ON COLUMN schools.school_curriculum_layers.layer_type IS
  'curriculum | kc_priorities | objectives | custom_rules — see migration 010.';
COMMENT ON COLUMN schools.school_curriculum_layers.payload IS
  'Layer-specific JSON; shape depends on layer_type. See migration 010.';
