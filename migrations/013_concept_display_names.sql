-- 013_concept_display_names.sql
-- What a person reads for a concept, per language.
--
-- `label` is the KC's identity and stays French snake_case: extraction maps a
-- conversation in any language onto the same labels, so the graph does not
-- split by language. But the label was being shown as-is — a teacher in Ohio
-- read "resoudre_equations_lineaires". `display_names` holds the readable
-- name per locale the app ships: {"en": ..., "fr": ..., "es": ..., "de": ...}.
--
-- Filled for new KCs at creation (services/kc_registry.py) and for existing
-- ones by scripts/backfill_display_names.py. A missing locale falls back, in
-- the app, to the label made readable.
--
-- Additive and idempotent.

ALTER TABLE kernel.concept_nodes
  ADD COLUMN IF NOT EXISTS display_names jsonb NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN kernel.concept_nodes.display_names IS
  'Readable name per locale ({"en","fr","es","de"}). The label stays the identity. See migration 013.';

NOTIFY pgrst, 'reload schema';
