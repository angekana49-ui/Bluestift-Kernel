-- 009_shared_db_hardening.sql
-- The Kernel shares its Supabase project with the RAYA Next.js app, and their
-- setups can clobber each other's PostgREST config (observed: a RAYA setup reset
-- the exposed schemas and dropped the service_role grants on kernel tables).
--
-- This migration re-asserts, idempotently:
--   1. the UNION of exposed schemas both sides need,
--   2. the service_role grants the Kernel requires,
--   3. a PostgREST reload so the change takes effect immediately.
--
-- Run it AFTER any RAYA/app setup that touches exposed schemas. If RAYA later
-- adds a schema, extend the union list below to include it.

-- 1. Expose all schemas both sides need (UNION — do not drop kernel/learning/content).
ALTER ROLE authenticator SET pgrst.db_schemas =
  'public, graphql_public, kernel, learning, schools, rag, content';

-- 2. service_role access on the Kernel's custom schemas.
GRANT USAGE ON SCHEMA kernel, schools, rag TO service_role;
GRANT ALL ON ALL TABLES    IN SCHEMA kernel  TO service_role;
GRANT ALL ON ALL TABLES    IN SCHEMA schools TO service_role;
GRANT ALL ON ALL TABLES    IN SCHEMA rag     TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA kernel  TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA schools TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA rag     TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA kernel  GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA schools GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag     GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA kernel  GRANT ALL ON SEQUENCES TO service_role;

-- 3. Reload PostgREST so the exposed-schema change is picked up immediately.
NOTIFY pgrst, 'reload config';
NOTIFY pgrst, 'reload schema';
