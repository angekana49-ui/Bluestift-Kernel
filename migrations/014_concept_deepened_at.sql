-- 014_concept_deepened_at.sql
-- When the Kernel last asked for a concept's finer prerequisites.
--
-- A concept only got prerequisites when it was created, so a coarse node
-- ("fractions_egales_et_operations") never grew the finer ideas a learner
-- actually trips on ("sens_du_denominateur"). When the diagnosis bottoms out on
-- a failing concept, services/deepen.py asks once for what lies just below it
-- and wires it in. This column is the "once": set when a concept is deepened,
-- so each concept costs one LLM call, ever, for every learner.
--
-- Without it the Kernel does not deepen at all (it cannot tell "done" from
-- "never"), which is the behaviour before this migration.
--
-- Additive and idempotent.

ALTER TABLE kernel.concept_nodes
  ADD COLUMN IF NOT EXISTS deepened_at timestamptz;

COMMENT ON COLUMN kernel.concept_nodes.deepened_at IS
  'Set once the Kernel has asked for this concept''s finer prerequisites. See services/deepen.py.';

NOTIFY pgrst, 'reload schema';
