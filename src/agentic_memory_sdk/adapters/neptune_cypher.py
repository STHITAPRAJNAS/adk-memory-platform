from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Sequence

from ..auth import AwsIamDatabaseAuthProvider
from ..models import (
    GraphEdgeMutation,
    GraphFactReadResult,
    GraphVertexUpsert,
    PlatformTenantContext,
    ReasoningTraceStep,
)
from ..ports import GraphStoreAdapter


class NeptuneOpenCypherAdapter(GraphStoreAdapter):
    """
    Production Amazon Neptune adapter using parameterized openCypher
    with mandatory tenant/namespace guards, idempotent batch MERGE,
    bi-temporal functional edge invalidation, and reasoning trace recording.
    Supports AWS IAM SigV4 and cross-account STS AssumeRole via `aws_auth_provider`.
    """

    def __init__(
        self,
        endpoint_by_tenant: dict[str, str],
        region: str = "us-east-1",
        aws_auth_provider: AwsIamDatabaseAuthProvider | None = None,
    ):
        self.endpoints = endpoint_by_tenant
        self.region = region
        self.aws_auth_provider = aws_auth_provider
        self._clients: dict[str, Any] = {}

    def _client(self, tenant_id: str):
        import boto3

        ep = self.endpoints.get(tenant_id) or self.endpoints["default"]
        if ep not in self._clients:
            endpoint_url = ep if ep.startswith("http") else f"https://{ep}:8182"
            session = (
                self.aws_auth_provider.get_boto3_session()
                if self.aws_auth_provider is not None
                else boto3.Session(region_name=self.region)
            )
            self._clients[ep] = session.client(
                "neptunedata",
                region_name=self.region,
                endpoint_url=endpoint_url,
            )
        return self._clients[ep]

    async def _run_cypher(
        self, tenant_id: str, query: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        client = self._client(tenant_id)
        resp = await asyncio.to_thread(
            client.execute_open_cypher_query,
            openCypherQuery=query,
            parameters=json.dumps(params),
        )
        return resp.get("results", [])

    async def upsert_vertices_and_bitemporal_edges(
        self,
        tenant_id: str,
        vertices: Sequence[GraphVertexUpsert],
        edges: Sequence[GraphEdgeMutation],
    ) -> None:
        if vertices:
            vertex_dicts = [v.model_dump(exclude={"embedding"}) for v in vertices]
            await self._run_cypher(
                tenant_id,
                """
                UNWIND $vertices AS v
                MERGE (n:Entity {`~id`: v.vertex_id, tenant_id: v.tenant_id})
                ON CREATE SET n.canonical_name = v.canonical_name,
                              n.entity_class = v.entity_class,
                              n.upper_class = v.upper_class,
                              n.summary = v.summary,
                              n.namespace = v.namespace,
                              n.updated_at = v.updated_at
                ON MATCH SET  n.summary = v.summary,
                              n.updated_at = v.updated_at
                """,
                {"vertices": vertex_dicts},
            )

        for e in edges:
            if e.is_functional:
                # Close t_invalid on any prior active functional edge with a different target
                await self._run_cypher(
                    tenant_id,
                    """
                    MATCH (s:Entity {`~id`: $src_id, tenant_id: $tenant_id})
                          -[r:RELATES {predicate: $pred, namespace: $ns, t_invalid: 'ACTIVE'}]->
                          (old_t:Entity)
                    WHERE old_t.`~id` <> $tgt_id
                    SET r.t_invalid = $t_valid
                    """,
                    {
                        "src_id": e.source_vertex_id,
                        "tenant_id": tenant_id,
                        "pred": e.predicate,
                        "ns": e.namespace,
                        "tgt_id": e.target_vertex_id,
                        "t_valid": e.t_valid,
                    },
                )

            await self._run_cypher(
                tenant_id,
                """
                MATCH (s:Entity {`~id`: $src_id, tenant_id: $tenant_id})
                MATCH (t:Entity {`~id`: $tgt_id, tenant_id: $tenant_id})
                MERGE (s)-[r:RELATES {
                    predicate: $pred,
                    namespace: $ns,
                    tenant_id: $tenant_id,
                    t_invalid: 'ACTIVE'
                }]->(t)
                ON CREATE SET r.edge_id = $edge_id,
                              r.t_valid = $t_valid,
                              r.t_created = $t_created,
                              r.fact_summary = $summary,
                              r.confidence = $conf,
                              r.reinforcement_count = 1,
                              r.source_episode_id = $ep_id,
                              r.source_app = $app,
                              r.user_id = $uid
                ON MATCH SET  r.fact_summary = $summary,
                              r.reinforcement_count = coalesce(r.reinforcement_count, 1) + 1,
                              r.confidence = CASE
                                  WHEN coalesce(r.confidence, 0.9) + 0.03 > 1.0 THEN 1.0
                                  ELSE coalesce(r.confidence, 0.9) + 0.03
                              END,
                              r.updated_at = $t_valid
                """,
                {
                    "src_id": e.source_vertex_id,
                    "tgt_id": e.target_vertex_id,
                    "tenant_id": tenant_id,
                    "pred": e.predicate,
                    "ns": e.namespace,
                    "edge_id": e.edge_id,
                    "t_valid": e.t_valid,
                    "t_created": e.t_created,
                    "summary": e.fact_summary,
                    "conf": e.confidence,
                    "ep_id": e.source_episode_id,
                    "app": e.source_app,
                    "uid": e.user_id,
                },
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

        if as_of_iso is None:
            temporal_filter = "r.t_invalid = 'ACTIVE'"
        else:
            temporal_filter = (
                "r.t_valid <= $as_of AND (r.t_invalid = 'ACTIVE' OR r.t_invalid > $as_of)"
            )

        frontier = list(dict.fromkeys(seed_vertex_ids))
        visited_vertices = set(frontier)
        seen_edges: set[tuple[str, str, str, str, str]] = set()
        out: list[GraphFactReadResult] = []

        for _hop in range(max(1, max_hops)):
            if not frontier or len(out) >= limit:
                break

            results = await self._run_cypher(
                ctx.tenant_id,
                f"""
                MATCH (s:Entity)-[r:RELATES]->(t:Entity)
                WHERE (s.`~id` IN $seed_ids OR t.`~id` IN $seed_ids)
                  AND s.tenant_id = $tenant_id
                  AND t.tenant_id = $tenant_id
                  AND s.namespace IN $read_ns
                  AND t.namespace IN $read_ns
                  AND r.namespace IN $read_ns
                  AND {temporal_filter}
                RETURN s.`~id` AS src_id,
                       s.canonical_name AS src,
                       coalesce(s.entity_class, 'p:Concept') AS src_cls,
                       r.predicate AS pred,
                       t.`~id` AS tgt_id,
                       t.canonical_name AS tgt,
                       coalesce(t.entity_class, 'p:Concept') AS tgt_cls,
                       r.fact_summary AS summary,
                       r.t_valid AS t_valid,
                       r.t_invalid AS t_invalid,
                       r.namespace AS ns,
                       coalesce(r.source_episode_id, '') AS ep_id,
                       coalesce(r.confidence, 1.0) AS conf
                LIMIT $limit
                """,
                {
                    "seed_ids": frontier,
                    "tenant_id": ctx.tenant_id,
                    "read_ns": ctx.read_namespaces,
                    "as_of": as_of_iso or "",
                    "limit": limit,
                },
            )
            next_frontier: list[str] = []
            for r in results:
                edge_key = (
                    r["src_id"],
                    r["pred"],
                    r["tgt_id"],
                    r["ns"],
                    r["t_valid"],
                )
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                out.append(
                    GraphFactReadResult(
                        source_name=r["src"],
                        source_class=r.get("src_cls", "p:Concept"),
                        predicate=r["pred"],
                        target_name=r["tgt"],
                        target_class=r.get("tgt_cls", "p:Concept"),
                        fact_summary=r["summary"],
                        t_valid=r["t_valid"],
                        t_invalid=r["t_invalid"],
                        namespace=r["ns"],
                        source_episode_id=r.get("ep_id", ""),
                        confidence=float(r["conf"]),
                    )
                )
                for vid in (r["src_id"], r["tgt_id"]):
                    if vid not in visited_vertices:
                        visited_vertices.add(vid)
                        next_frontier.append(vid)
                if len(out) >= limit:
                    break
            frontier = next_frontier

        return out[:limit]

    async def record_reasoning_trace(
        self,
        ctx: PlatformTenantContext,
        run_id: str,
        session_id: str,
        steps: Sequence[ReasoningTraceStep],
    ) -> None:
        step_dicts = [s.model_dump() for s in steps]
        await self._run_cypher(
            ctx.tenant_id,
            """
            MERGE (run:AgentRun {`~id`: $run_id, tenant_id: $tenant_id})
            ON CREATE SET run.app_name = $app,
                          run.session_id = $session_id,
                          run.namespace = $ns,
                          run.created_at = $now
            WITH run
            UNWIND $steps AS st
            MERGE (tc:ToolCall {`~id`: $run_id + '_' + toString(st.step_index), tenant_id: $tenant_id})
            SET tc.tool_name = st.tool_name,
                tc.input_summary = st.tool_input_summary,
                tc.output_summary = st.tool_output_summary,
                tc.status = st.status,
                tc.namespace = $ns
            MERGE (run)-[:EXECUTED_STEP {step_index: st.step_index}]->(tc)
            WITH tc, st
            UNWIND st.touched_vertex_ids AS vid
            MATCH (e:Entity {`~id`: vid, tenant_id: $tenant_id})
            MERGE (tc)-[:TOUCHED_ENTITY]->(e)
            """,
            {
                "run_id": run_id,
                "tenant_id": ctx.tenant_id,
                "app": ctx.app_name,
                "session_id": session_id,
                "ns": ctx.app_private_namespace,
                "now": datetime.now(timezone.utc).isoformat(),
                "steps": step_dicts,
            },
        )

    async def purge_episode_lineage(
        self, tenant_id: str, episode_ids: Sequence[str], user_id: str
    ) -> None:
        await self._run_cypher(
            tenant_id,
            """
            MATCH ()-[r:RELATES {tenant_id: $tenant_id}]->()
            WHERE r.source_episode_id IN $ep_ids OR r.user_id = $uid
            DELETE r
            """,
            {"tenant_id": tenant_id, "ep_ids": list(episode_ids), "uid": user_id},
        )
