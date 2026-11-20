from __future__ import annotations

import json
from typing import Sequence

from ..auth import CloudAuthProvider, EnterpriseTlsConfig
from ..models import (
    GraphEdgeMutation,
    GraphFactReadResult,
    GraphVertexUpsert,
    PlatformTenantContext,
    PlaybookRecord,
    ReasoningTraceStep,
)
from ..ports import GraphStoreAdapter, PlaybookStoreAdapter
from .postgres_pgvector import PostgresPgVectorAdapter


class _PostgresPoolBase:
    """Shared asyncpg connection pool & Enterprise Cloud Auth helper for PostgreSQL adapters."""

    def __init__(
        self,
        dsn: str,
        min_pool_size: int = 2,
        max_pool_size: int = 20,
        auth_provider: CloudAuthProvider | None = None,
        tls_config: EnterpriseTlsConfig | None = None,
    ):
        self.dsn = dsn
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.auth_provider = auth_provider
        self.tls_config = tls_config
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg

            pool_kwargs: dict = {
                "min_size": self.min_pool_size,
                "max_size": self.max_pool_size,
            }
            if self.auth_provider is not None:
                pool_kwargs["user"] = await self.auth_provider.get_username()
                pool_kwargs["password"] = self.auth_provider.get_password
            if self.tls_config is not None:
                pool_kwargs["ssl"] = self.tls_config.build_ssl_context()
            self._pool = await asyncpg.create_pool(self.dsn, **pool_kwargs)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


class PostgresPlaybookAdapter(_PostgresPoolBase, PlaybookStoreAdapter):
    """
    Standalone PostgreSQL + `pgvector` implementation of `PlaybookStoreAdapter`.
    Can be deployed on its own dedicated PostgreSQL database or alongside Neo4j / Neptune
    when an organization prefers PostgreSQL over Redis for Tier 3 Procedural Playbooks.
    Supports Enterprise Cloud Auth (`auth_provider`) and mTLS (`tls_config`).
    """

    async def init_schema(self, embedding_dim: int = 1536) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS platform_playbooks (
                    tenant_id           TEXT NOT NULL,
                    playbook_id         TEXT NOT NULL,
                    namespace           TEXT NOT NULL,
                    app_name            TEXT NOT NULL,
                    trigger_intent      TEXT NOT NULL,
                    workflow_steps      JSONB NOT NULL,
                    resolution_summary  TEXT NOT NULL,
                    success_count       INTEGER NOT NULL DEFAULT 1,
                    embedding           vector({embedding_dim}),
                    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (tenant_id, playbook_id)
                );
                """
            )

    async def upsert_playbook(self, playbook: PlaybookRecord) -> PlaybookRecord:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT playbook_id, success_count
                FROM platform_playbooks
                WHERE tenant_id = $1 AND (playbook_id = $2 OR LOWER(trigger_intent) = LOWER($3))
                LIMIT 1
                """,
                playbook.tenant_id,
                playbook.playbook_id,
                playbook.trigger_intent,
            )
            if existing:
                playbook.playbook_id = existing["playbook_id"]
                playbook.success_count = int(existing["success_count"]) + 1
            await conn.execute(
                """
                INSERT INTO platform_playbooks
                    (tenant_id, playbook_id, namespace, app_name, trigger_intent,
                     workflow_steps, resolution_summary, success_count, embedding, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9::vector, NOW())
                ON CONFLICT (tenant_id, playbook_id) DO UPDATE
                SET workflow_steps = EXCLUDED.workflow_steps,
                    resolution_summary = EXCLUDED.resolution_summary,
                    success_count = EXCLUDED.success_count,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                """,
                playbook.tenant_id,
                playbook.playbook_id,
                playbook.namespace,
                playbook.app_name,
                playbook.trigger_intent,
                json.dumps(playbook.workflow_steps),
                playbook.resolution_summary,
                playbook.success_count,
                str(playbook.embedding),
            )
            return playbook

    async def match_playbooks(
        self,
        tenant_id: str,
        read_namespaces: list[str],
        query_embedding: list[float],
        similarity_threshold: float = 0.65,
        limit: int = 2,
    ) -> list[PlaybookRecord]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT playbook_id, namespace, app_name, trigger_intent,
                       workflow_steps, resolution_summary, success_count,
                       1 - (embedding <=> $1::vector) AS sim
                FROM platform_playbooks
                WHERE tenant_id = $2
                  AND namespace = ANY($3::text[])
                  AND (1 - (embedding <=> $1::vector)) >= $4
                ORDER BY sim DESC, success_count DESC
                LIMIT $5
                """,
                str(query_embedding),
                tenant_id,
                read_namespaces,
                similarity_threshold,
                limit,
            )
            return [
                PlaybookRecord(
                    playbook_id=r["playbook_id"],
                    tenant_id=tenant_id,
                    namespace=r["namespace"],
                    app_name=r["app_name"],
                    trigger_intent=r["trigger_intent"],
                    workflow_steps=json.loads(r["workflow_steps"]),
                    resolution_summary=r["resolution_summary"],
                    success_count=int(r["success_count"]),
                )
                for r in rows
            ]


class PostgresGraphAdapter(_PostgresPoolBase, GraphStoreAdapter):
    """
    Standalone PostgreSQL implementation of `GraphStoreAdapter` (`platform_graph_vertices`,
    bi-temporal `platform_graph_edges` with `t_valid`/`t_invalid` functional invalidation,
    and `platform_reasoning_traces`).
    Can be paired with `OpenSearchSeedAdapter` or `RedisEpisodicAndSeedAdapter`, or run
    on its own dedicated PostgreSQL instance.
    """

    async def init_schema(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS platform_graph_vertices (
                    tenant_id       TEXT NOT NULL,
                    vertex_id       TEXT NOT NULL,
                    namespace       TEXT NOT NULL,
                    entity_class    TEXT NOT NULL,
                    upper_class     TEXT NOT NULL,
                    canonical_name  TEXT NOT NULL,
                    summary         TEXT NOT NULL,
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (tenant_id, vertex_id)
                );

                CREATE TABLE IF NOT EXISTS platform_graph_edges (
                    tenant_id           TEXT NOT NULL,
                    edge_id             TEXT NOT NULL,
                    namespace           TEXT NOT NULL,
                    source_vertex_id    TEXT NOT NULL,
                    predicate           TEXT NOT NULL,
                    target_vertex_id    TEXT NOT NULL,
                    fact_summary        TEXT NOT NULL,
                    t_valid             TEXT NOT NULL,
                    t_invalid           TEXT NOT NULL DEFAULT 'ACTIVE',
                    t_created           TEXT NOT NULL,
                    confidence          DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    reinforcement_count INTEGER NOT NULL DEFAULT 1,
                    source_episode_id   TEXT NOT NULL DEFAULT '',
                    source_app          TEXT NOT NULL DEFAULT '',
                    user_id             TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (tenant_id, edge_id)
                );

                CREATE INDEX IF NOT EXISTS idx_pg_graph_edges_lookup
                    ON platform_graph_edges (tenant_id, namespace, source_vertex_id, target_vertex_id, t_invalid);

                CREATE TABLE IF NOT EXISTS platform_reasoning_traces (
                    tenant_id   TEXT NOT NULL,
                    run_id      TEXT NOT NULL,
                    session_id  TEXT NOT NULL,
                    app_name    TEXT NOT NULL,
                    namespace   TEXT NOT NULL,
                    steps_json  JSONB NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (tenant_id, run_id)
                );
                """
            )

    async def upsert_vertices_and_bitemporal_edges(
        self,
        tenant_id: str,
        vertices: Sequence[GraphVertexUpsert],
        edges: Sequence[GraphEdgeMutation],
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for v in vertices:
                    await conn.execute(
                        """
                        INSERT INTO platform_graph_vertices
                            (tenant_id, vertex_id, namespace, entity_class, upper_class, canonical_name, summary, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                        ON CONFLICT (tenant_id, vertex_id) DO UPDATE
                        SET summary = EXCLUDED.summary,
                            entity_class = EXCLUDED.entity_class,
                            upper_class = EXCLUDED.upper_class,
                            updated_at = NOW()
                        """,
                        tenant_id,
                        v.vertex_id,
                        v.namespace,
                        v.entity_class,
                        v.upper_class,
                        v.canonical_name,
                        v.summary,
                    )

                for e in edges:
                    if e.is_functional:
                        await conn.execute(
                            """
                            UPDATE platform_graph_edges
                            SET t_invalid = $6
                            WHERE tenant_id = $1
                              AND source_vertex_id = $2
                              AND predicate = $3
                              AND namespace = $4
                              AND t_invalid = 'ACTIVE'
                              AND target_vertex_id <> $5
                            """,
                            tenant_id,
                            e.source_vertex_id,
                            e.predicate,
                            e.namespace,
                            e.target_vertex_id,
                            e.t_valid,
                        )

                    existing = await conn.fetchrow(
                        """
                        SELECT edge_id, reinforcement_count, confidence
                        FROM platform_graph_edges
                        WHERE tenant_id = $1
                          AND source_vertex_id = $2
                          AND predicate = $3
                          AND target_vertex_id = $4
                          AND namespace = $5
                          AND t_invalid = 'ACTIVE'
                        """,
                        tenant_id,
                        e.source_vertex_id,
                        e.predicate,
                        e.target_vertex_id,
                        e.namespace,
                    )
                    if existing:
                        await conn.execute(
                            """
                            UPDATE platform_graph_edges
                            SET fact_summary = $3,
                                reinforcement_count = $4,
                                confidence = LEAST(1.0, $5 + 0.03)
                            WHERE tenant_id = $1 AND edge_id = $2
                            """,
                            tenant_id,
                            existing["edge_id"],
                            e.fact_summary,
                            int(existing["reinforcement_count"]) + 1,
                            float(existing["confidence"]),
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO platform_graph_edges
                                (tenant_id, edge_id, namespace, source_vertex_id, predicate, target_vertex_id,
                                 fact_summary, t_valid, t_invalid, t_created, confidence, reinforcement_count,
                                 source_episode_id, source_app, user_id)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 1, $12, $13, $14)
                            ON CONFLICT (tenant_id, edge_id) DO NOTHING
                            """,
                            tenant_id,
                            e.edge_id,
                            e.namespace,
                            e.source_vertex_id,
                            e.predicate,
                            e.target_vertex_id,
                            e.fact_summary,
                            e.t_valid,
                            e.t_invalid,
                            e.t_created,
                            e.confidence,
                            e.source_episode_id,
                            e.source_app,
                            e.user_id,
                        )

    async def expand_active_subgraph(
        self,
        ctx: PlatformTenantContext,
        seed_vertex_ids: Sequence[str],
        as_of_iso: str | None = None,
        max_hops: int = 2,
        limit: int = 25,
    ) -> list[GraphFactReadResult]:
        if not seed_vertex_ids:
            return []

        pool = await self._get_pool()
        frontier = list(dict.fromkeys(seed_vertex_ids))
        visited_vertices = set(frontier)
        seen_edges: set[str] = set()
        results: list[GraphFactReadResult] = []

        async with pool.acquire() as conn:
            for _hop in range(max(1, max_hops)):
                if not frontier or len(results) >= limit:
                    break

                if as_of_iso is None:
                    rows = await conn.fetch(
                        """
                        SELECT e.edge_id, e.source_vertex_id, e.predicate, e.target_vertex_id,
                               e.fact_summary, e.t_valid, e.t_invalid, e.namespace,
                               e.source_episode_id, e.confidence, e.reinforcement_count,
                               s.canonical_name AS src_name, s.entity_class AS src_cls,
                               t.canonical_name AS tgt_name, t.entity_class AS tgt_cls
                        FROM platform_graph_edges e
                        JOIN platform_graph_vertices s
                          ON s.tenant_id = e.tenant_id AND s.vertex_id = e.source_vertex_id
                        JOIN platform_graph_vertices t
                          ON t.tenant_id = e.tenant_id AND t.vertex_id = e.target_vertex_id
                        WHERE e.tenant_id = $1
                          AND e.namespace = ANY($2::text[])
                          AND s.namespace = ANY($2::text[])
                          AND t.namespace = ANY($2::text[])
                          AND (e.source_vertex_id = ANY($3::text[]) OR e.target_vertex_id = ANY($3::text[]))
                          AND e.t_invalid = 'ACTIVE'
                        LIMIT $4
                        """,
                        ctx.tenant_id,
                        ctx.read_namespaces,
                        frontier,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT e.edge_id, e.source_vertex_id, e.predicate, e.target_vertex_id,
                               e.fact_summary, e.t_valid, e.t_invalid, e.namespace,
                               e.source_episode_id, e.confidence, e.reinforcement_count,
                               s.canonical_name AS src_name, s.entity_class AS src_cls,
                               t.canonical_name AS tgt_name, t.entity_class AS tgt_cls
                        FROM platform_graph_edges e
                        JOIN platform_graph_vertices s
                          ON s.tenant_id = e.tenant_id AND s.vertex_id = e.source_vertex_id
                        JOIN platform_graph_vertices t
                          ON t.tenant_id = e.tenant_id AND t.vertex_id = e.target_vertex_id
                        WHERE e.tenant_id = $1
                          AND e.namespace = ANY($2::text[])
                          AND s.namespace = ANY($2::text[])
                          AND t.namespace = ANY($2::text[])
                          AND (e.source_vertex_id = ANY($3::text[]) OR e.target_vertex_id = ANY($3::text[]))
                          AND e.t_valid <= $4
                          AND (e.t_invalid = 'ACTIVE' OR e.t_invalid > $4)
                        LIMIT $5
                        """,
                        ctx.tenant_id,
                        ctx.read_namespaces,
                        frontier,
                        as_of_iso,
                        limit,
                    )

                next_frontier: list[str] = []
                for r in rows:
                    if r["edge_id"] in seen_edges:
                        continue
                    seen_edges.add(r["edge_id"])
                    results.append(
                        GraphFactReadResult(
                            source_name=r["src_name"],
                            source_class=r["src_cls"],
                            predicate=r["predicate"],
                            target_name=r["tgt_name"],
                            target_class=r["tgt_cls"],
                            fact_summary=r["fact_summary"],
                            t_valid=r["t_valid"],
                            t_invalid=r["t_invalid"],
                            namespace=r["namespace"],
                            source_episode_id=r["source_episode_id"],
                            confidence=float(r["confidence"]),
                            reinforcement_count=int(r["reinforcement_count"]),
                        )
                    )
                    for vid in (r["source_vertex_id"], r["target_vertex_id"]):
                        if vid not in visited_vertices:
                            visited_vertices.add(vid)
                            next_frontier.append(vid)
                    if len(results) >= limit:
                        break
                frontier = next_frontier

        return results[:limit]

    async def record_reasoning_trace(
        self,
        ctx: PlatformTenantContext,
        run_id: str,
        session_id: str,
        steps: Sequence[ReasoningTraceStep],
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO platform_reasoning_traces
                    (tenant_id, run_id, session_id, app_name, namespace, steps_json)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (tenant_id, run_id) DO UPDATE
                SET steps_json = EXCLUDED.steps_json
                """,
                ctx.tenant_id,
                run_id,
                session_id,
                ctx.app_name,
                ctx.app_private_namespace,
                json.dumps([s.model_dump() for s in steps]),
            )

    async def purge_episode_lineage(
        self, tenant_id: str, episode_ids: Sequence[str], user_id: str
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM platform_graph_edges
                WHERE tenant_id = $1 AND (source_episode_id = ANY($2::text[]) OR user_id = $3)
                """,
                tenant_id,
                list(episode_ids),
                user_id,
            )


class PostgresUnifiedAdapter(
    PostgresPgVectorAdapter, PostgresGraphAdapter, PostgresPlaybookAdapter
):
    """
    Single-Stack Pure PostgreSQL Unified Adapter (`RegistryStoreAdapter` +
    `EpisodicAndSeedStoreAdapter` + `GraphStoreAdapter` + `PlaybookStoreAdapter`).

    Combines `PostgresPgVectorAdapter`, `PostgresGraphAdapter`, and `PostgresPlaybookAdapter`
    over a single shared connection pool.
    """

    async def init_schema(
        self, embedding_dim: int = 1536, default_tenant: str = "t1"
    ) -> None:
        await PostgresPgVectorAdapter.init_schema(
            self, embedding_dim=embedding_dim, default_tenant=default_tenant
        )
        await PostgresGraphAdapter.init_schema(self)
        await PostgresPlaybookAdapter.init_schema(self, embedding_dim=embedding_dim)
