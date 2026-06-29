-- 006_grant_service_role.sql
-- Grant the service_role (used by the Kernel) access to the custom schemas.
-- Custom schemas don't inherit the privileges Supabase sets up on `public`.
-- service_role bypasses RLS but still needs table-level GRANTs.

GRANT USAGE ON SCHEMA kernel  TO service_role;
GRANT USAGE ON SCHEMA schools TO service_role;
GRANT USAGE ON SCHEMA rag     TO service_role;

GRANT ALL ON ALL TABLES    IN SCHEMA kernel  TO service_role;
GRANT ALL ON ALL TABLES    IN SCHEMA schools TO service_role;
GRANT ALL ON ALL TABLES    IN SCHEMA rag     TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA kernel  TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA schools TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA rag     TO service_role;

-- Future tables created by later migrations inherit the same grants.
ALTER DEFAULT PRIVILEGES IN SCHEMA kernel  GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA schools GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag     GRANT ALL ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA kernel  GRANT ALL ON SEQUENCES TO service_role;
