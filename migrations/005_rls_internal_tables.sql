-- 005_rls_internal_tables.sql
-- Internal, service-only tables: enable RLS with no policy so anon/authenticated
-- get no access while the service_role (which bypasses RLS) keeps writing freely.
-- Protects student conversation payloads once the kernel schema is exposed to
-- PostgREST.
ALTER TABLE kernel.kernel_requests           ENABLE ROW LEVEL SECURITY;
ALTER TABLE kernel.kernel_outputs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE kernel.kernel_monitoring         ENABLE ROW LEVEL SECURITY;
ALTER TABLE schools.school_curriculum_layers ENABLE ROW LEVEL SECURITY;
