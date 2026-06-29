-- 004_extensions_rls.sql
-- pgvector for RAG embeddings, performance indexes, and RLS policies.
-- The Kernel uses the service_role key, which bypasses RLS; these policies
-- protect the data when accessed through the anon/authenticated roles (RAYA).

-- --------------------------------------------------------------------------- --
-- Extensions.
-- --------------------------------------------------------------------------- --
CREATE EXTENSION IF NOT EXISTS vector;

-- RAG store for enriched conversation embeddings (gemini/groq-agnostic, 768-dim
-- matches common free-tier embedding models; adjust to your embedder).
CREATE TABLE IF NOT EXISTS rag.conversation_embeddings (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid NOT NULL,
    concept_id uuid REFERENCES kernel.concept_nodes(id) ON DELETE SET NULL,
    content    text NOT NULL,
    embedding  vector(768),
    created_at timestamptz DEFAULT now()
);

-- --------------------------------------------------------------------------- --
-- Indexes.
-- --------------------------------------------------------------------------- --
CREATE INDEX IF NOT EXISTS idx_concept_nodes_subject       ON kernel.concept_nodes (subject);
CREATE INDEX IF NOT EXISTS idx_concept_nodes_label         ON kernel.concept_nodes (lower(label));
CREATE INDEX IF NOT EXISTS idx_concept_edges_concept       ON kernel.concept_edges (concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_edges_prerequisite  ON kernel.concept_edges (prerequisite_id);
CREATE INDEX IF NOT EXISTS idx_scs_user                    ON kernel.student_concept_state (user_id);
CREATE INDEX IF NOT EXISTS idx_scs_concept                 ON kernel.student_concept_state (concept_id);
CREATE INDEX IF NOT EXISTS idx_requests_user               ON kernel.kernel_requests (user_id);
CREATE INDEX IF NOT EXISTS idx_outputs_user                ON kernel.kernel_outputs (user_id);
CREATE INDEX IF NOT EXISTS idx_insights_user               ON kernel.individual_insights (user_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_user           ON kernel.learning_trajectories (user_id);

-- IVFFlat index for vector similarity search (build after some data exists).
CREATE INDEX IF NOT EXISTS idx_conv_embeddings_vec
    ON rag.conversation_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- --------------------------------------------------------------------------- --
-- Row Level Security.
-- --------------------------------------------------------------------------- --
ALTER TABLE kernel.student_concept_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE kernel.student_mindset_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE kernel.individual_insights   ENABLE ROW LEVEL SECURITY;
ALTER TABLE kernel.learning_trajectories ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag.conversation_embeddings  ENABLE ROW LEVEL SECURITY;

-- A student may read only their own rows. service_role bypasses RLS entirely.
DROP POLICY IF EXISTS scs_owner_select ON kernel.student_concept_state;
CREATE POLICY scs_owner_select ON kernel.student_concept_state
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS mindset_owner_select ON kernel.student_mindset_state;
CREATE POLICY mindset_owner_select ON kernel.student_mindset_state
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS insights_owner_select ON kernel.individual_insights;
CREATE POLICY insights_owner_select ON kernel.individual_insights
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS trajectories_owner_select ON kernel.learning_trajectories;
CREATE POLICY trajectories_owner_select ON kernel.learning_trajectories
    FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS rag_owner_select ON rag.conversation_embeddings;
CREATE POLICY rag_owner_select ON rag.conversation_embeddings
    FOR SELECT USING (auth.uid() = user_id);

-- concept_nodes / concept_edges are shared curriculum: readable by any
-- authenticated user, writable only via service_role (no anon write policy).
ALTER TABLE kernel.concept_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE kernel.concept_edges ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS nodes_read_all ON kernel.concept_nodes;
CREATE POLICY nodes_read_all ON kernel.concept_nodes
    FOR SELECT USING (auth.role() = 'authenticated');

DROP POLICY IF EXISTS edges_read_all ON kernel.concept_edges;
CREATE POLICY edges_read_all ON kernel.concept_edges
    FOR SELECT USING (auth.role() = 'authenticated');
