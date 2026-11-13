-- ============================================================================
-- ENTERPRISE MULTI-TENANT ADK COMPOSITE MEMORY PLATFORM DDL
-- Engine: PostgreSQL 16+ with pgvector & pg_trgm
-- Features:
--   1. Platform App Registry & Stratified TBox Control Plane
--   2. LIST Partitioning by tenant_id (avoids HNSW post-filtering starvation)
--   3. Kernel-enforced Row-Level Security (RLS) via transaction-local session vars
--   4. ACID Transactional Outbox for eventual consistency with Neptune / Neo4j
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Platform Control Plane: App Entitlements, Quotas & Compiled Ontology TBox
CREATE TABLE IF NOT EXISTS platform_app_registry (
    tenant_id TEXT NOT NULL,
    app_name TEXT NOT NULL,
    domain_namespace TEXT NOT NULL,
    read_namespaces TEXT[] NOT NULL,   -- e.g., ARRAY['app:t1:sre_bot', 'domain:t1:infra', 'global:t1:core']
    write_namespace TEXT NOT NULL,     -- e.g., 'domain:t1:infra' or 'app:t1:sre_bot'
    ontology_schema JSONB NOT NULL,    -- CompiledTBox JSON descriptor
    max_rpm INTEGER NOT NULL DEFAULT 600,
    max_active_vertices INTEGER NOT NULL DEFAULT 500000,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, app_name)
);

-- 2. Shared Domain Pack Catalog (Layer 1 Federated Domain Ontologies)
CREATE TABLE IF NOT EXISTS platform_domain_packs (
    pack_id TEXT NOT NULL,             -- e.g., 'payments@v2.1'
    domain_name TEXT NOT NULL,         -- e.g., 'payments'
    version TEXT NOT NULL,
    tbox_spec JSONB NOT NULL,          -- Classes, RelationRules, SHACL constraints
    turtle_source TEXT,                -- Optional raw OWL/SHACL .ttl definition
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (pack_id)
);

-- 3. Tier 1: Immutable Episodic Memory & Session Chunks (Partitioned by tenant_id)
CREATE TABLE IF NOT EXISTS episodic_memories (
    episode_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    author TEXT NOT NULL,
    content_text TEXT NOT NULL,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content_text)) STORED,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    custom_metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, episode_id)
) PARTITION BY LIST (tenant_id);

-- 4. Tier 2 Bridge: Graph Entity & Fact Seeding Index
CREATE TABLE IF NOT EXISTS graph_seed_index (
    neptune_vertex_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    namespace TEXT NOT NULL,           -- Controls cross-app sharing ('app:*', 'domain:*', 'global:*')
    entity_class TEXT NOT NULL,        -- Enforced by Ontology TBox
    upper_class TEXT NOT NULL DEFAULT 'p:Concept',
    canonical_name TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    summary_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', canonical_name || ' ' || summary_text)) STORED,
    embedding VECTOR(1536) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, neptune_vertex_id)
) PARTITION BY LIST (tenant_id);

-- 5. Transactional Outbox for Idempotent Graph Sync (Neptune / Neo4j)
CREATE TABLE IF NOT EXISTS graph_transactional_outbox (
    outbox_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,      -- 'UPSERT_GRAPH_BATCH', 'RECORD_TRACE', 'PURGE_LINEAGE'
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING', -- 'PENDING', 'RETRY', 'PROCESSED', 'DEAD_LETTER'
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    PRIMARY KEY (outbox_id)
);

CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON graph_transactional_outbox (status, created_at)
    WHERE status IN ('PENDING', 'RETRY');

-- 6. Helper Procedure to Provision a New Tenant Partition with Local HNSW + GIN Indexes
CREATE OR REPLACE FUNCTION provision_tenant_memory_partitions(p_tenant_id TEXT)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    safe_suffix TEXT := regexp_replace(lower(p_tenant_id), '[^a-z0-9_]', '_', 'g');
BEGIN
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS episodic_memories_%s PARTITION OF episodic_memories FOR VALUES IN (%L)',
        safe_suffix, p_tenant_id
    );
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_episodic_%s_hnsw ON episodic_memories_%s USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)',
        safe_suffix, safe_suffix
    );
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_episodic_%s_tsv ON episodic_memories_%s USING gin (content_tsv)',
        safe_suffix, safe_suffix
    );

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS graph_seed_index_%s PARTITION OF graph_seed_index FOR VALUES IN (%L)',
        safe_suffix, p_tenant_id
    );
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_seed_%s_hnsw ON graph_seed_index_%s USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)',
        safe_suffix, safe_suffix
    );
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS idx_seed_%s_tsv ON graph_seed_index_%s USING gin (summary_tsv)',
        safe_suffix, safe_suffix
    );
END;
$$;

-- Provision default tenant 't1'
SELECT provision_tenant_memory_partitions('t1');

-- 7. Dedicated Non-Superuser Application Role (Required so RLS is never bypassed by superuser)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'platform_app_user') THEN
        CREATE ROLE platform_app_user NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO platform_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO platform_app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO platform_app_user;

-- 8. Kernel-Enforced Row-Level Security (RLS)
ALTER TABLE episodic_memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodic_memories FORCE ROW LEVEL SECURITY;

ALTER TABLE graph_seed_index ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_seed_index FORCE ROW LEVEL SECURITY;

-- Episodic Policy: Strictly isolated to (tenant_id, app_name, user_id)
DROP POLICY IF EXISTS episodic_isolation_policy ON episodic_memories;
CREATE POLICY episodic_isolation_policy ON episodic_memories
    USING (
        tenant_id = current_setting('app.tenant_id', true)
        AND app_name = current_setting('app.app_name', true)
        AND user_id = current_setting('app.user_id', true)
    )
    WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        AND app_name = current_setting('app.app_name', true)
    );

-- Graph Seed Read Policy: Read across authorized namespaces ('app:*', 'domain:*', 'global:*')
DROP POLICY IF EXISTS graph_seed_read_policy ON graph_seed_index;
CREATE POLICY graph_seed_read_policy ON graph_seed_index
    FOR SELECT USING (
        tenant_id = current_setting('app.tenant_id', true)
        AND namespace = ANY (string_to_array(current_setting('app.read_namespaces', true), ','))
    );

-- Graph Seed Write Policy: Write only to permitted write_namespace (or app private overflow)
DROP POLICY IF EXISTS graph_seed_write_policy ON graph_seed_index;
CREATE POLICY graph_seed_write_policy ON graph_seed_index
    FOR INSERT WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        AND namespace = current_setting('app.write_namespace', true)
    );

DROP POLICY IF EXISTS graph_seed_update_policy ON graph_seed_index;
CREATE POLICY graph_seed_update_policy ON graph_seed_index
    FOR UPDATE USING (
        tenant_id = current_setting('app.tenant_id', true)
        AND namespace = current_setting('app.write_namespace', true)
    );
