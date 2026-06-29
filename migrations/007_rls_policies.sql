-- 007_rls_policies.sql
-- Bluestift Kernel — RLS policies (idempotent, safe to re-run).
-- service_role bypasses RLS (used by the Kernel). These rules govern the
-- `authenticated` client role only (e.g. RAYA reading the signed-in student).
-- `anon` gets no access anywhere.

-- --- Grants required for the `authenticated` role to reach the data --------
-- (Policies do nothing without USAGE on the schema + table privileges.)
GRANT USAGE ON SCHEMA kernel TO authenticated;
GRANT USAGE ON SCHEMA rag    TO authenticated;

GRANT SELECT ON kernel.concept_nodes           TO authenticated;
GRANT SELECT ON kernel.concept_edges           TO authenticated;
GRANT SELECT ON kernel.student_concept_state   TO authenticated;
GRANT SELECT ON kernel.student_mindset_state   TO authenticated;
GRANT SELECT ON kernel.individual_insights     TO authenticated;
GRANT SELECT ON kernel.learning_trajectories   TO authenticated;
GRANT SELECT ON rag.conversation_embeddings    TO authenticated;

-- ============================================================================
-- 1. STUDENT-OWNED DATA — a user may read only their own rows.
-- ============================================================================
ALTER TABLE kernel.student_concept_state ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS scs_owner_select ON kernel.student_concept_state;
CREATE POLICY scs_owner_select ON kernel.student_concept_state
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

ALTER TABLE kernel.student_mindset_state ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS mindset_owner_select ON kernel.student_mindset_state;
CREATE POLICY mindset_owner_select ON kernel.student_mindset_state
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

ALTER TABLE kernel.individual_insights ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS insights_owner_select ON kernel.individual_insights;
CREATE POLICY insights_owner_select ON kernel.individual_insights
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

ALTER TABLE kernel.learning_trajectories ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS trajectories_owner_select ON kernel.learning_trajectories;
CREATE POLICY trajectories_owner_select ON kernel.learning_trajectories
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

ALTER TABLE rag.conversation_embeddings ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rag_owner_select ON rag.conversation_embeddings;
CREATE POLICY rag_owner_select ON rag.conversation_embeddings
    FOR SELECT TO authenticated
    USING (auth.uid() = user_id);

-- ============================================================================
-- 2. SHARED CURRICULUM GRAPH — readable by any authenticated user.
-- ============================================================================
ALTER TABLE kernel.concept_nodes ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS nodes_read_all ON kernel.concept_nodes;
CREATE POLICY nodes_read_all ON kernel.concept_nodes
    FOR SELECT TO authenticated
    USING (true);

ALTER TABLE kernel.concept_edges ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS edges_read_all ON kernel.concept_edges;
CREATE POLICY edges_read_all ON kernel.concept_edges
    FOR SELECT TO authenticated
    USING (true);

-- ============================================================================
-- 3. INTERNAL / SERVICE-ONLY TABLES — RLS on, NO policy on purpose.
--    service_role bypasses RLS (the Kernel keeps writing); anon/authenticated
--    get zero access. These hold conversation payloads & ops logs.
-- ============================================================================
ALTER TABLE kernel.kernel_requests           ENABLE ROW LEVEL SECURITY;
ALTER TABLE kernel.kernel_outputs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE kernel.kernel_monitoring         ENABLE ROW LEVEL SECURITY;
ALTER TABLE schools.school_curriculum_layers ENABLE ROW LEVEL SECURITY;
-- (No policies = locked to everyone except service_role. Intentional.)
