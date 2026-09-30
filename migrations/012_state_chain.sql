-- 012_state_chain.sql
-- Each student_concept_state row becomes the tip of a chain.
--
-- `version` is the block height: every write sets it to the previous value + 1
-- and is accepted only if the row is still at the height it was read at. A write
-- computed from a stale state is rejected and recomputed on the new tip, so two
-- requests for the same student can no longer silently overwrite each other's
-- evidence (read-modify-write race). kernel.learning_events (migration 011) is
-- the ledger the state is derived from.
--
-- review_count / lapse_count feed the spacing effect in forgetting: a successful
-- retrieval after a real gap slows forgetting, a failed one (a lapse) undoes it.
--
-- Additive and idempotent. Existing rows start at version 0 with no reviews.

ALTER TABLE kernel.student_concept_state
  ADD COLUMN IF NOT EXISTS version int NOT NULL DEFAULT 0;
ALTER TABLE kernel.student_concept_state
  ADD COLUMN IF NOT EXISTS review_count int NOT NULL DEFAULT 0;
ALTER TABLE kernel.student_concept_state
  ADD COLUMN IF NOT EXISTS lapse_count int NOT NULL DEFAULT 0;

COMMENT ON COLUMN kernel.student_concept_state.version IS
  'Chain height: +1 per write; a write is accepted only from the current height. See migration 012.';
COMMENT ON COLUMN kernel.student_concept_state.review_count IS
  'Successful retrievals after a gap of >= 1 day (spacing effect). See core/forgetting.py.';
COMMENT ON COLUMN kernel.student_concept_state.lapse_count IS
  'Failed retrievals after a gap of >= 1 day. See core/forgetting.py.';

NOTIFY pgrst, 'reload schema';
