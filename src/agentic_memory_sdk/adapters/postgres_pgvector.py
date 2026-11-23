from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Sequence

from ..auth import CloudAuthProvider, EnterpriseTlsConfig
from ..governance import EntityResolutionGuard
from ..models import (
    CompiledTBox,
    EpisodicRecord,
    GraphVertexUpsert,
    OutboxOperation,
    PlatformTenantContext,
)
from ..ports import EpisodicAndSeedStoreAdapter, RegistryStoreAdapter


class PostgresPgVectorAdapter(RegistryStoreAdapter, EpisodicAndSeedStoreAdapter):
    """
    Production PostgreSQL + pgvector adapter enforcing Row-Level Security (RLS)
    via transaction-local session variables, partitioned HNSW indexes per tenant,
    Reciprocal Rank Fusion (RRF) hybrid search, and an ACID Transactional Outbox.
    Supports Enterprise Cloud Auth (AWS RDS IAM, GCP Cloud SQL IAM, Azure Entra ID,
    Vault dynamic secrets) and mTLS (`EnterpriseTlsConfig`).
    """

    def __init__(
        self,
        dsn: str,
        min_pool_size: int = 5,
        max_pool_size: int = 30,
        auth_provider: CloudAuthProvider | None = None,
        tls_config: EnterpriseTlsConfig | None = None,
    ):
        self.dsn = dsn
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.auth_provider = auth_provider
        self.tls_config = tls_config
        self._pool = None
        self._provisioned_tenants: set[str] = set()

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            pool_kwargs: dict = {
                "min_size": self.min_pool_size,
                "max_size": self.max_pool_size,
            }
            if self.auth_provider is not None:
                pool_kwargs["user"] = await self.auth_provider.get_username()
                # asyncpg natively invokes an async callable `password` on every new connection,
                # enabling zero-downtime AWS IAM / GCP IAM / Azure Entra token rotation!
                pool_kwargs["password"] = self.auth_provider.get_password
            if self.tls_config is not None:
                pool_kwargs["ssl"] = self.tls_config.build_ssl_context()

            self._pool = await asyncpg.create_pool(self.dsn, **pool_kwargs)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def _rls_tx(self, ctx: PlatformTenantContext):
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.tenant_id', $1, true)", ctx.tenant_id
                )
                await conn.execute(
                    "SELECT set_config('app.app_name', $1, true)", ctx.app_name
                )
                await conn.execute(
                    "SELECT set_config('app.user_id', $1, true)", ctx.user_id
                )
                await conn.execute(
                    "SELECT set_config('app.read_namespaces', $1, true)",
                    ",".join(ctx.read_namespaces),
                )
                await conn.execute(
                    "SELECT set_config('app.write_namespace', $1, true)",
                    ctx.write_namespace,
                )
                yield conn

    async def register_app(
        self,
        tenant_id: str,
        app_name: str,
        read_namespaces: list[str],
        write_namespace: str,
        tbox: CompiledTBox,
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Ensure tenant partitions exist before registering app
            await conn.execute("SELECT provision_tenant_memory_partitions($1)", tenant_id)
            self._provisioned_tenants.add(tenant_id)
            await conn.execute(
                """
                INSERT INTO platform_app_registry
                    (tenant_id, app_name, domain_namespace, read_namespaces, write_namespace, ontology_schema, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
                ON CONFLICT (tenant_id, app_name) DO UPDATE
                SET domain_namespace = EXCLUDED.domain_namespace,
                    read_namespaces = EXCLUDED.read_namespaces,
                    write_namespace = EXCLUDED.write_namespace,
                    ontology_schema = EXCLUDED.ontology_schema,
                    updated_at = NOW()
                """,
                tenant_id,
                app_name,
                write_namespace,
                read_namespaces,
                write_namespace,
                tbox.model_dump_json(),
            )

    async def resolve_context(
        self, tenant_id: str, app_name: str, user_id: str
    ) -> PlatformTenantContext:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT read_namespaces, write_namespace, ontology_schema, max_rpm, max_active_vertices
                FROM platform_app_registry
                WHERE tenant_id = $1 AND app_name = $2
                """,
                tenant_id,
                app_name,
            )
            app_private_ns = f"app:{tenant_id}:{app_name}"
            user_private_ns = f"user:{tenant_id}:{user_id}"
            if not row:
                # Zero-config fallback: auto-provision Tier 0 isolated context + cross-app user namespace
                return PlatformTenantContext(
                    tenant_id=tenant_id,
                    app_name=app_name,
                    user_id=user_id,
                    app_private_namespace=app_private_ns,
                    write_namespace=app_private_ns,
                    read_namespaces=[
                        app_private_ns,
                        user_private_ns,
                        f"global:{tenant_id}:core",
                    ],
                    tbox=CompiledTBox(),
                )
            merged_read_ns = list(
                dict.fromkeys([*list(row["read_namespaces"]), user_private_ns])
            )
            return PlatformTenantContext(
                tenant_id=tenant_id,
                app_name=app_name,
                user_id=user_id,
                app_private_namespace=app_private_ns,
                write_namespace=row["write_namespace"],
                read_namespaces=merged_read_ns,
                tbox=CompiledTBox.model_validate_json(row["ontology_schema"]),
                max_rpm=row["max_rpm"],
                max_active_vertices=row["max_active_vertices"],
            )

    async def init_schema(
        self, embedding_dim: int = 1536, default_tenant: str = "t1"
    ) -> None:
        """
        Programmatically initializes the PostgreSQL + pgvector RLS schema, partitions,
        HNSW vector indexes, and non-superuser role with configurable embedding dimensions.
        """
        from importlib import resources as _pkg_resources
        from pathlib import Path

        ddl: str | None = None
        try:
            bundled = _pkg_resources.files("agentic_memory_sdk.sql").joinpath(
                "001_postgres_rls_and_partitions.sql"
            )
            if bundled.is_file():
                ddl = bundled.read_text(encoding="utf-8")
        except Exception:
            ddl = None

        if ddl is None:
            sql_path = (
                Path(__file__).resolve().parents[3]
                / "docs"
                / "sql"
                / "001_postgres_rls_and_partitions.sql"
            )
            if sql_path.exists():
                ddl = sql_path.read_text(encoding="utf-8")

        if ddl is None:
            raise FileNotFoundError(
                "Could not locate bundled PostgreSQL schema '001_postgres_rls_and_partitions.sql' "
                "in package resources ('agentic_memory_sdk.sql') or repository 'docs/sql/'."
            )

        pool = await self._get_pool()
        ddl = ddl.replace("vector(1536)", f"vector({embedding_dim})")
        async with pool.acquire() as conn:
            await conn.execute(ddl)
            if default_tenant != "t1":
                await conn.execute(
                    "SELECT provision_tenant_memory_partitions($1)", default_tenant
                )

    async def resolve_existing_vertex_id(
        self,
        ctx: PlatformTenantContext,
        entity_class: str,
        canonical_name: str,
        embedding: list[float],
        similarity_threshold: float = 0.91,
    ) -> str | None:
        async with self._rls_tx(ctx) as conn:
            rows = await conn.fetch(
                """
                SELECT neptune_vertex_id,
                       canonical_name,
                       1 - (embedding <=> $1::vector) AS sim
                FROM graph_seed_index
                WHERE tenant_id = $5
                  AND namespace = ANY($6::text[])
                  AND entity_class = $2
                  AND (
                      LOWER(canonical_name) = LOWER($3)
                      OR (1 - (embedding <=> $1::vector)) >= $4
                  )
                ORDER BY sim DESC
                LIMIT 5
                """,
                str(embedding),
                entity_class,
                canonical_name,
                similarity_threshold,
                ctx.tenant_id,
                ctx.read_namespaces,
            )
            for row in rows:
                if EntityResolutionGuard.can_merge(
                    row["canonical_name"],
                    canonical_name,
                    float(row["sim"]),
                    similarity_threshold,
                ):
                    return row["neptune_vertex_id"]
            return None

    async def commit_episodes_and_outbox(
        self,
        ctx: PlatformTenantContext,
        episodes: Sequence[EpisodicRecord],
        vertex_upserts: Sequence[GraphVertexUpsert],
        outbox_op: OutboxOperation | None,
    ) -> None:
        # Auto-provision tenant partitions on first write for Tier-0 zero-config apps
        if ctx.tenant_id not in self._provisioned_tenants:
            pool = await self._get_pool()
            async with pool.acquire() as pconn:
                await pconn.execute(
                    "SELECT provision_tenant_memory_partitions($1)", ctx.tenant_id
                )
            self._provisioned_tenants.add(ctx.tenant_id)

        async with self._rls_tx(ctx) as conn:
            for ep in episodes:
                await conn.execute(
                    """
                    INSERT INTO episodic_memories
                        (episode_id, tenant_id, app_name, user_id, session_id, namespace, author, content_text, embedding, custom_metadata)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10::jsonb)
                    ON CONFLICT (tenant_id, episode_id) DO NOTHING
                    """,
                    ep.episode_id,
                    ep.tenant_id,
                    ep.app_name,
                    ep.user_id,
                    ep.session_id,
                    ep.namespace,
                    ep.author,
                    ep.content_text,
                    str(ep.embedding),
                    json.dumps(ep.metadata),
                )

            for v in vertex_upserts:
                await conn.execute(
                    "SELECT set_config('app.write_namespace', $1, true)", v.namespace
                )
                await conn.execute(
                    """
                    INSERT INTO graph_seed_index
                        (neptune_vertex_id, tenant_id, namespace, entity_class, upper_class, canonical_name, summary_text, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
                    ON CONFLICT (tenant_id, neptune_vertex_id) DO UPDATE
                    SET summary_text = EXCLUDED.summary_text,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                    """,
                    v.vertex_id,
                    v.tenant_id,
                    v.namespace,
                    v.entity_class,
                    v.upper_class,
                    v.canonical_name,
                    v.summary,
                    str(v.embedding),
                )

            if outbox_op is not None:
                await conn.execute(
                    """
                    INSERT INTO graph_transactional_outbox
                        (outbox_id, tenant_id, operation_type, payload, status, attempts)
                    VALUES ($1::uuid, $2, $3, $4::jsonb, 'PENDING', 0)
                    """,
                    outbox_op.outbox_id,
                    outbox_op.tenant_id,
                    outbox_op.operation_type,
                    json.dumps(outbox_op.payload),
                )

    async def hybrid_seed_search(
        self,
        ctx: PlatformTenantContext,
        query_text: str,
        query_embedding: list[float],
        limit: int = 8,
    ) -> list[GraphVertexUpsert]:
        async with self._rls_tx(ctx) as conn:
            rows = await conn.fetch(
                """
                WITH vector_search AS (
                    SELECT neptune_vertex_id, namespace, entity_class, upper_class, canonical_name, summary_text,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS v_rank
                    FROM graph_seed_index
                    WHERE tenant_id = $4 AND namespace = ANY($5::text[])
                    LIMIT 25
                ),
                fts_search AS (
                    SELECT neptune_vertex_id, namespace, entity_class, upper_class, canonical_name, summary_text,
                           ROW_NUMBER() OVER (
                               ORDER BY ts_rank_cd(summary_tsv, websearch_to_tsquery('english', $2)) DESC
                           ) AS f_rank
                    FROM graph_seed_index
                    WHERE tenant_id = $4
                      AND namespace = ANY($5::text[])
                      AND summary_tsv @@ websearch_to_tsquery('english', $2)
                    LIMIT 25
                )
                SELECT COALESCE(v.neptune_vertex_id, f.neptune_vertex_id) AS vertex_id,
                       COALESCE(v.namespace, f.namespace) AS namespace,
                       COALESCE(v.entity_class, f.entity_class) AS entity_class,
                       COALESCE(v.upper_class, f.upper_class) AS upper_class,
                       COALESCE(v.canonical_name, f.canonical_name) AS canonical_name,
                       COALESCE(v.summary_text, f.summary_text) AS summary_text,
                       COALESCE(1.0 / (60 + v.v_rank), 0.0) + COALESCE(1.0 / (60 + f.f_rank), 0.0) AS rrf_score
                FROM vector_search v
                FULL OUTER JOIN fts_search f ON v.neptune_vertex_id = f.neptune_vertex_id
                ORDER BY rrf_score DESC
                LIMIT $3
                """,
                str(query_embedding),
                query_text,
                limit,
                ctx.tenant_id,
                ctx.read_namespaces,
            )
            seeds = [
                GraphVertexUpsert(
                    vertex_id=r["vertex_id"],
                    tenant_id=ctx.tenant_id,
                    namespace=r["namespace"],
                    entity_class=r["entity_class"],
                    upper_class=r["upper_class"],
                    canonical_name=r["canonical_name"],
                    summary=r["summary_text"],
                    embedding=[],
                )
                for r in rows
            ]
            seen_ids = {s.vertex_id for s in seeds}
            user_ns = f"user:{ctx.tenant_id}:{ctx.user_id}"
            if user_ns in ctx.read_namespaces:
                user_rows = await conn.fetch(
                    """
                    SELECT neptune_vertex_id AS vertex_id, namespace, entity_class,
                           upper_class, canonical_name, summary_text
                    FROM graph_seed_index
                    WHERE tenant_id = $1 AND namespace = $2
                    LIMIT 5
                    """,
                    ctx.tenant_id,
                    user_ns,
                )
                for ur in user_rows:
                    if ur["vertex_id"] not in seen_ids:
                        seeds.append(
                            GraphVertexUpsert(
                                vertex_id=ur["vertex_id"],
                                tenant_id=ctx.tenant_id,
                                namespace=ur["namespace"],
                                entity_class=ur["entity_class"],
                                upper_class=ur["upper_class"],
                                canonical_name=ur["canonical_name"],
                                summary=ur["summary_text"],
                                embedding=[],
                            )
                        )
                        seen_ids.add(ur["vertex_id"])
            return seeds

    async def search_episodes(
        self,
        ctx: PlatformTenantContext,
        query_embedding: list[float],
        limit: int = 4,
    ) -> list[EpisodicRecord]:
        async with self._rls_tx(ctx) as conn:
            rows = await conn.fetch(
                """
                SELECT episode_id::text, session_id, namespace, author, content_text, created_at
                FROM episodic_memories
                WHERE tenant_id = $3 AND app_name = $4 AND user_id = $5
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                str(query_embedding),
                limit,
                ctx.tenant_id,
                ctx.app_name,
                ctx.user_id,
            )
            return [
                EpisodicRecord(
                    episode_id=r["episode_id"],
                    tenant_id=ctx.tenant_id,
                    app_name=ctx.app_name,
                    user_id=ctx.user_id,
                    session_id=r["session_id"],
                    namespace=r["namespace"],
                    author=r["author"],
                    content_text=r["content_text"],
                    embedding=[],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    async def fetch_pending_outbox(self, batch_size: int = 25) -> list[OutboxOperation]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    SELECT outbox_id::text, tenant_id, operation_type, payload, attempts
                    FROM graph_transactional_outbox
                    WHERE status IN ('PENDING', 'RETRY') AND attempts < 5
                    ORDER BY created_at ASC
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    batch_size,
                )
                return [
                    OutboxOperation(
                        outbox_id=r["outbox_id"],
                        tenant_id=r["tenant_id"],
                        operation_type=r["operation_type"],
                        payload=json.loads(r["payload"]),
                        attempts=r["attempts"],
                    )
                    for r in rows
                ]

    async def mark_outbox_processed(self, outbox_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE graph_transactional_outbox SET status = 'PROCESSED', processed_at = NOW() WHERE outbox_id = $1::uuid",
                outbox_id,
            )

    async def mark_outbox_failed(self, outbox_id: str, error_msg: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE graph_transactional_outbox
                SET status = CASE WHEN attempts + 1 >= 5 THEN 'DEAD_LETTER' ELSE 'RETRY' END,
                    attempts = attempts + 1,
                    last_error = $2
                WHERE outbox_id = $1::uuid
                """,
                outbox_id,
                error_msg[:500],
            )

    async def purge_user_data(self, ctx: PlatformTenantContext) -> list[str]:
        async with self._rls_tx(ctx) as conn:
            rows = await conn.fetch(
                "DELETE FROM episodic_memories WHERE user_id = $1 RETURNING episode_id::text",
                ctx.user_id,
            )
            return [r["episode_id"] for r in rows]
