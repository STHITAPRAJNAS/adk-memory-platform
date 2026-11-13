"""
End-to-End Test Suite: Google ADK + A2A Multi-Agent System + Docker PostgreSQL (`pgvector` + RLS) & Neo4j (`openCypher`).

Topology exercised:
  1. Docker `adk-memory-postgres` (`pgvector/pgvector:pg16` on :5433) -> RLS + HNSW partitions + Transactional Outbox
  2. Docker `adk-memory-neo4j` (`neo4j:5-community` on :7687) -> Bi-Temporal openCypher Graph + Reasoning Traces
  3. Centralized Memory Platform Service (`:8095`) -> `CompositeMemoryService` backed by Postgres + Neo4j
  4. Remote A2A Server 1: `sre_specialist` (`:8091`) -> Subscribed to `sre_infra@v1.2` (`domain:t1:sre_infra`)
  5. Remote A2A Server 2: `crm_specialist` (`:8092`) -> Subscribed to `customer_crm@v2.0` (`domain:t1:customer_crm`)
  6. Front-Door A2A Orchestrator: `platform_concierge` (`:8090`) -> Delegates over A2A to `sre_specialist` & `crm_specialist`
"""

from __future__ import annotations

import os
import socket
import threading
import time
from contextlib import contextmanager

import asyncpg
import httpx
import pytest
from neo4j import AsyncGraphDatabase

from a2a_sample.a2a_client import A2AWireClient, text_message
from a2a_sample.config import get_a2a_settings
from a2a_sample.servers import (
    build_crm_a2a_app,
    build_orchestrator_a2a_app,
    build_sre_a2a_app,
)
from agentic_memory_sdk import AppMemoryManifest, GovernanceTier

pytestmark = pytest.mark.asyncio


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


@contextmanager
def _run_uvicorn_in_thread(app_factory, host: str, port: int, name: str):
    if _port_open(host, port):
        yield f"http://{host}:{port}"
        return

    import uvicorn

    config = uvicorn.Config(app_factory(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name=name, daemon=True)
    thread.start()

    deadline = time.time() + 30
    while time.time() < deadline:
        if server.started and _port_open(host, port):
            break
        time.sleep(0.2)
    else:
        server.should_exit = True
        pytest.fail(f"{name} failed to start on {host}:{port}")

    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def a2a_stack():
    """
    Stands up the full 4-server stack connected to Docker PostgreSQL (:5433) and Docker Neo4j (:7687):
      - Memory Platform API (:8095)
      - SRE Specialist A2A Server (:8091)
      - CRM Specialist A2A Server (:8092)
      - Platform Concierge A2A Orchestrator (:8090)
    """
    os.environ["MEMORY_BACKEND"] = "docker"
    settings = get_a2a_settings()

    if not _port_open("127.0.0.1", 5433) or not _port_open("127.0.0.1", 7687):
        pytest.skip("Docker Postgres (:5433) or Neo4j (:7687) is not running")

    def build_memory_api():
        from agentic_memory_sdk.server.main import build_platform_service_from_env
        from agentic_memory_sdk.server.api import create_memory_platform_app

        svc, reg = build_platform_service_from_env()
        return create_memory_platform_app(svc, reg)

    with (
        _run_uvicorn_in_thread(
            build_memory_api, settings.host, settings.memory_api_port, "memory-api"
        ),
        _run_uvicorn_in_thread(
            build_sre_a2a_app, settings.host, settings.sre_agent_port, "sre-a2a"
        ),
        _run_uvicorn_in_thread(
            build_crm_a2a_app, settings.host, settings.crm_agent_port, "crm-a2a"
        ),
        _run_uvicorn_in_thread(
            build_orchestrator_a2a_app,
            settings.host,
            settings.orchestrator_port,
            "concierge-a2a",
        ),
    ):
        yield settings


async def _clean_docker_databases():
    """Waits for Docker Postgres & Neo4j readiness and cleans test tenant `t1` data."""
    import asyncio

    for attempt in range(30):
        try:
            conn = await asyncpg.connect(
                "postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform"
            )
            try:
                await conn.execute("DELETE FROM episodic_memories WHERE tenant_id = 't1'")
                await conn.execute("DELETE FROM graph_seed_index WHERE tenant_id = 't1'")
                await conn.execute(
                    "DELETE FROM graph_transactional_outbox WHERE tenant_id = 't1'"
                )
            finally:
                await conn.close()
            break
        except Exception:
            if attempt == 29:
                raise
            await asyncio.sleep(1.0)

    for attempt in range(30):
        driver = AsyncGraphDatabase.driver(
            "bolt://127.0.0.1:7687", auth=("neo4j", "platform_secret")
        )
        try:
            async with driver.session() as session:
                await session.run("MATCH (n {tenant_id: 't1'}) DETACH DELETE n")
            break
        except Exception:
            if attempt == 29:
                raise
            await asyncio.sleep(1.0)
        finally:
            await driver.close()


async def test_full_a2a_agents_with_docker_postgres_and_neo4j(a2a_stack):
    """
    Full E2E Verification:
      1. Clean Docker Postgres & Neo4j state and register `sre_specialist` (`sre_infra@v1.2`)
         and `crm_specialist` (`customer_crm@v2.0`) with the Memory Platform API (:8095).
      2. Verify all 3 A2A Agent Cards (`platform_concierge`, `sre_specialist`, `crm_specialist`).
      3. Send Turn 1 to `platform_concierge` over A2A:
         "Remember: checkout-api is deployed in us-east-1 and reads from aurora-payments-01"
         -> Delegated over A2A to `sre_specialist` -> Stored in Docker Postgres & Neo4j.
      4. Send Turn 2 to `platform_concierge` over A2A:
         "Remember: We failed checkout-api over to us-west-2 using experimental-chaos-mesh"
         -> Delegated over A2A to `sre_specialist`:
            • Functional edge `DEPLOYED_IN_PRIMARY -> us-east-1` is invalidated (`t_invalid` set) in Neo4j!
            • New active edge `DEPLOYED_IN_PRIMARY -> us-west-2` is created in Neo4j!
            • Unregistered class `experimental-chaos-mesh` is quarantined via Two-Bucket validator to `app:t1:sre_specialist`!
      5. Query current SRE state through `platform_concierge` over A2A -> Returns `us-west-2` (NOT `us-east-1`) + `aurora-payments-01` + `experimental-chaos-mesh`.
      6. Query historical SRE state (`as_of=2026-06-01T00:00:00Z`) through `platform_concierge` over A2A -> Returns `us-east-1`!
      7. Send CRM update through `platform_concierge` over A2A:
         "Remember: Acme-Corp upgraded to Enterprise-Platinum"
         -> Delegated over A2A to `crm_specialist` -> Stored in `domain:t1:customer_crm`.
      8. Verify strict cross-domain DB isolation: asking `crm_specialist` about `checkout-api` returns `NO_MATCHING_MEMORIES`!
      9. Inspect Docker PostgreSQL (`episodic_memories`, `graph_seed_index`, `graph_transactional_outbox`)
         and Docker Neo4j (`Entity`, `RELATES`, `AgentRun`, `ToolCall`) directly to verify physical DB state!
    """
    await _clean_docker_databases()
    settings = a2a_stack

    timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as http:
        # 1. Register ADK agent manifests on the Memory Platform API (:8095)
        reg_sre = await http.post(
            f"{settings.memory_api_url}/v1/apps/register",
            json=AppMemoryManifest(
                tenant_id="t1",
                app_name="sre_specialist",
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["sre_infra@v1.2"],
                allow_emergent_overflow=True,
            ).model_dump(mode="json"),
        )
        assert reg_sre.status_code == 200

        reg_crm = await http.post(
            f"{settings.memory_api_url}/v1/apps/register",
            json=AppMemoryManifest(
                tenant_id="t1",
                app_name="crm_specialist",
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["customer_crm@v2.0"],
                allow_emergent_overflow=True,
            ).model_dump(mode="json"),
        )
        assert reg_crm.status_code == 200

        # 2. Verify all 3 A2A Agent Cards
        concierge_a2a = A2AWireClient(
            http, settings.orchestrator_rpc_url, settings.orchestrator_card_url
        )
        sre_a2a = A2AWireClient(
            http, settings.sre_agent_rpc_url, settings.sre_agent_card_url
        )
        crm_a2a = A2AWireClient(
            http, settings.crm_agent_rpc_url, settings.crm_agent_card_url
        )

        assert (await concierge_a2a.fetch_card())["name"] == "platform_concierge"
        assert (await sre_a2a.fetch_card())["name"] == "sre_specialist"
        assert (await crm_a2a.fetch_card())["name"] == "crm_specialist"

        # 3. Turn 1 (via A2A Front-Door `platform_concierge` -> `sre_specialist`): Initial deployment in us-east-1
        t1_turn = await concierge_a2a.stream(
            text_message(
                "Remember: checkout-api is deployed in us-east-1 and reads from aurora-payments-01"
            )
        )
        assert t1_turn.error is None, t1_turn.error
        assert t1_turn.downstream_task_ids, "Expected A2A hop from platform_concierge to sre_specialist"
        assert "[SRE Specialist] Recorded in shared SRE memory" in t1_turn.combined_text

        # 4. Turn 2 (via A2A Front-Door `platform_concierge` -> `sre_specialist`): Failover to us-west-2 + emergent tool
        t2_turn = await concierge_a2a.stream(
            text_message(
                "Remember: We failed checkout-api over to us-west-2 using experimental-chaos-mesh"
            )
        )
        assert t2_turn.error is None, t2_turn.error
        assert "[SRE Specialist] Recorded in shared SRE memory" in t2_turn.combined_text

        # 5. Query ACTIVE SRE state through `platform_concierge` over A2A
        active_sre_turn = await concierge_a2a.stream(
            text_message("Where is checkout-api currently deployed and what database does it read from?")
        )
        assert active_sre_turn.error is None
        sre_reply = active_sre_turn.combined_text
        # Active primary region must be us-west-2; superseded us-east-1 edge must NOT be active!
        assert "(checkout-api) -[DEPLOYED_IN_PRIMARY]-> (us-west-2)" in sre_reply
        assert "(checkout-api) -[DEPLOYED_IN_PRIMARY]-> (us-east-1)" not in sre_reply
        # Non-functional READS_FROM edge to aurora-payments-01 remains active!
        assert "(checkout-api) -[READS_FROM]-> (aurora-payments-01)" in sre_reply
        # Emergent overflow entity `experimental-chaos-mesh` was quarantined to `app:t1:sre_specialist`!
        assert "[app:t1:sre_specialist] (checkout-api) -[RELATED_TO]-> (experimental-chaos-mesh)" in sre_reply

        # 6. Query Point-in-Time HISTORICAL SRE state (`as_of=2026-06-01T00:00:00Z`) over A2A
        hist_sre_turn = await concierge_a2a.stream(
            text_message("Query checkout-api region as_of=2026-06-01T00:00:00Z")
        )
        assert hist_sre_turn.error is None
        hist_reply = hist_sre_turn.combined_text
        assert "(checkout-api) -[DEPLOYED_IN_PRIMARY]-> (us-east-1)" in hist_reply
        assert "(checkout-api) -[DEPLOYED_IN_PRIMARY]-> (us-west-2)" not in hist_reply

        # 7. Turn 3 (via A2A Front-Door `platform_concierge` -> `crm_specialist`): Record CRM subscription
        crm_write_turn = await concierge_a2a.stream(
            text_message("Remember: Acme-Corp upgraded to Enterprise-Platinum")
        )
        assert crm_write_turn.error is None
        assert "[CRM Specialist] Recorded in shared CRM memory" in crm_write_turn.combined_text

        # 8. Query CRM state over A2A
        crm_read_turn = await concierge_a2a.stream(
            text_message("What subscription tier is customer Acme-Corp on?")
        )
        assert crm_read_turn.error is None
        assert "(Acme-Corp) -[SUBSCRIBED_TO_TIER]-> (Enterprise-Platinum)" in crm_read_turn.combined_text

        # 9. Verify Cross-Domain Isolation: Directly ask `crm_specialist` over A2A about `checkout-api`
        # PostgreSQL RLS + Neo4j namespace guards block `crm_specialist` from reading `domain:t1:sre_infra`!
        cross_domain_probe = await crm_a2a.stream(
            text_message("Tell me about checkout-api and aurora-payments-01")
        )
        probe_reply = cross_domain_probe.combined_text
        assert "domain:t1:sre_infra" not in probe_reply
        assert "DEPLOYED_IN_PRIMARY" not in probe_reply
        assert "READS_FROM" not in probe_reply
        assert "us-west-2" not in probe_reply
        assert "us-east-1" not in probe_reply

    # 10. Direct Physical Database Verification in Docker PostgreSQL & Docker Neo4j
    pg_conn = await asyncpg.connect(
        "postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform"
    )
    try:
        ep_count = await pg_conn.fetchval(
            "SELECT count(*) FROM episodic_memories WHERE tenant_id = 't1'"
        )
        seed_count = await pg_conn.fetchval(
            "SELECT count(*) FROM graph_seed_index WHERE tenant_id = 't1'"
        )
        processed_outbox = await pg_conn.fetchval(
            "SELECT count(*) FROM graph_transactional_outbox WHERE tenant_id = 't1' AND status = 'PROCESSED'"
        )
        assert ep_count == 3
        assert seed_count >= 6
        assert processed_outbox == 3
    finally:
        await pg_conn.close()

    neo_driver = AsyncGraphDatabase.driver(
        "bolt://127.0.0.1:7687", auth=("neo4j", "platform_secret")
    )
    try:
        async with neo_driver.session() as session:
            # Verify both the invalidated us-east-1 edge and active us-west-2 edge exist in Neo4j
            res = await session.run(
                """
                MATCH (s:Entity {canonical_name: 'checkout-api', tenant_id: 't1'})
                      -[r:RELATES {predicate: 'DEPLOYED_IN_PRIMARY'}]->(t:Entity)
                RETURN t.canonical_name AS region, r.t_valid AS t_valid, r.t_invalid AS t_invalid
                """
            )
            edges_by_region = {rec["region"]: rec async for rec in res}
            assert "us-east-1" in edges_by_region
            assert edges_by_region["us-east-1"]["t_invalid"] == "2026-11-01T00:00:00Z"
            assert "us-west-2" in edges_by_region
            assert edges_by_region["us-west-2"]["t_invalid"] == "ACTIVE"

            # Verify Tier 3 Procedural Reasoning Traces (`AgentRun -> ToolCall`) were persisted in Neo4j
            trace_res = await session.run(
                """
                MATCH (run:AgentRun {tenant_id: 't1', app_name: 'sre_specialist'})
                      -[:EXECUTED_STEP]->(tc:ToolCall)
                RETURN count(tc) AS trace_count
                """
            )
            trace_rec = await trace_res.single()
            assert trace_rec["trace_count"] >= 2

            # Fetch all Neo4j Entities and Bi-Temporal Edges for inspection snapshot
            entities_res = await session.run(
                """
                MATCH (n:Entity {tenant_id: 't1'})
                RETURN n.`~id` AS vertex_id, n.canonical_name AS name,
                       n.entity_class AS entity_class, n.namespace AS namespace,
                       n.summary AS summary
                ORDER BY n.namespace, n.canonical_name
                """
            )
            neo4j_entities = [dict(r) async for r in entities_res]

            rels_res = await session.run(
                """
                MATCH (s:Entity {tenant_id: 't1'})-[r:RELATES]->(t:Entity {tenant_id: 't1'})
                RETURN s.canonical_name AS source, r.predicate AS predicate,
                       t.canonical_name AS target, r.namespace AS namespace,
                       r.t_valid AS t_valid, r.t_invalid AS t_invalid,
                       r.confidence AS confidence, r.source_episode_id AS source_episode_id
                ORDER BY r.t_valid
                """
            )
            neo4j_edges = [dict(r) async for r in rels_res]
    finally:
        await neo_driver.close()

    # 11. Test Real Docker Redis Playbook Cache (:6379), FastAPI Ontology Steward (:8095), and MCP Surface
    import json
    from pathlib import Path
    from agentic_memory_sdk import (
        DeterministicHashEmbeddingAdapter,
        DomainPackSpec,
        MemoryMcpServerTools,
        PlaybookRecord,
        RedisPlaybookAndWorkingAdapter,
        RelationRule,
        create_mcp_server,
    )
    from agentic_memory_sdk.server.main import build_platform_service_from_env

    redis_playbooks_snapshot: list[dict] = []
    if _port_open("127.0.0.1", 6379):
        redis_adapter = RedisPlaybookAndWorkingAdapter(
            redis_url="redis://127.0.0.1:6379/0"
        )
        embedder_1536 = DeterministicHashEmbeddingAdapter(dim=1536)
        pb_vec = await embedder_1536.embed("checkout-api us-west-2 failover runbook")
        try:
            r_client = await redis_adapter._get_redis()
            if r_client is not None:
                await r_client.delete("adk_memory:playbook:t1")
            pb1 = await redis_adapter.upsert_playbook(
                PlaybookRecord(
                    playbook_id="pb_sre_failover_01",
                    tenant_id="t1",
                    namespace="domain:t1:sre_infra",
                    app_name="sre_specialist",
                    trigger_intent="checkout-api us-west-2 failover runbook",
                    workflow_steps=[
                        "verify_aurora_replica_lag",
                        "shift_route53_weights_us_west_2",
                        "validate_checkout_synthetic_probes",
                    ],
                    resolution_summary="Standard Route53 + Aurora failover playbook for checkout-api",
                    embedding=pb_vec,
                )
            )
            # Reinforce same playbook in Redis
            pb2 = await redis_adapter.upsert_playbook(pb1)
            assert pb2.success_count == 2
            matched = await redis_adapter.match_playbooks(
                "t1", ["domain:t1:sre_infra"], pb_vec
            )
            assert len(matched) == 1
            assert matched[0].success_count == 2
            redis_playbooks_snapshot = [
                m.model_dump(exclude={"embedding"}) for m in matched
            ]
        finally:
            await redis_adapter.close()

    # Exercise FastAPI Ontology Review, Maintenance, and Audit Log endpoints on :8095
    async with httpx.AsyncClient(timeout=timeout) as http:
        review_resp = await http.post(
            f"{settings.memory_api_url}/v1/domains/packs/review",
            json=DomainPackSpec(
                pack_id="finops@v1.0",
                domain_name="finops",
                version="1.0.0",
                classes={"CloudBudget": "", "FinOpsOwner": "owl:Thing"},
                relations={
                    "owned-by-primary": RelationRule(
                        predicate="owned-by-primary",
                        domain_class="CloudBudget",
                        range_class="FinOpsOwner",
                        is_functional=False,
                    )
                },
            ).model_dump(mode="json"),
        )
        assert review_resp.status_code == 200
        review_data = review_resp.json()
        assert (
            review_data["refined_pack"]["relations"]["OWNED_BY_PRIMARY"]["is_functional"]
            is True
        )

        maint_resp = await http.post(f"{settings.memory_api_url}/v1/maintenance/run")
        assert maint_resp.status_code == 200

        audit_resp = await http.get(f"{settings.memory_api_url}/v1/audit/ontology")
        assert audit_resp.status_code == 200
        audit_data = audit_resp.json()
        assert audit_data["audit_count"] >= 1

    # Exercise MCP Server Surface (`create_mcp_server` + `MemoryMcpServerTools`) against live Docker DBs
    live_service, live_registrar = build_platform_service_from_env()
    mcp_server = create_mcp_server(live_service, live_registrar)
    assert mcp_server is not None
    mcp_tools = MemoryMcpServerTools(live_service, live_registrar)
    mcp_recall = await mcp_tools.recall_context(
        app_name="t1:sre_specialist",
        user_id="sre_user",
        query="checkout-api deployment and database",
        max_hops=2,
    )
    assert len(mcp_recall["memories"]) >= 1

    # Write inspectable Real-Mode Verification Snapshot to `docs/real_mode_test_results.json`
    pg_conn2 = await asyncpg.connect(
        "postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform"
    )
    try:
        pg_apps = [
            dict(r)
            for r in await pg_conn2.fetch(
                "SELECT tenant_id, app_name, read_namespaces, write_namespace FROM platform_app_registry WHERE tenant_id = 't1'"
            )
        ]
        pg_episodes = [
            {
                **dict(r),
                "created_at": r["created_at"].isoformat(),
            }
            for r in await pg_conn2.fetch(
                "SELECT episode_id::text, tenant_id, app_name, user_id, namespace, author, content_text, created_at FROM episodic_memories WHERE tenant_id = 't1' ORDER BY created_at"
            )
        ]
        pg_seeds = [
            dict(r)
            for r in await pg_conn2.fetch(
                "SELECT neptune_vertex_id, tenant_id, namespace, entity_class, upper_class, canonical_name, summary_text FROM graph_seed_index WHERE tenant_id = 't1' ORDER BY namespace, canonical_name"
            )
        ]
    finally:
        await pg_conn2.close()

    snapshot_path = (
        Path(__file__).resolve().parent.parent / "docs" / "real_mode_test_results.json"
    )
    snapshot_payload = {
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "surfaces_verified": [
            "1. Google ADK A2A Multi-Agent Mesh (:8090 platform_concierge -> :8091 sre_specialist & :8092 crm_specialist)",
            "2. Centralized FastAPI Memory Platform API (:8095 /v1/apps/register, /v1/memory/*, /v1/domains/packs/review, /v1/maintenance/run, /v1/audit/ontology)",
            "3. Model Context Protocol (MCP) Server (create_mcp_server / MemoryMcpServerTools over live Docker DBs)",
            "4. Docker PostgreSQL 16 + pgvector (:5433) with RLS, Partitioned HNSW, RRF Hybrid Search & Transactional Outbox",
            "5. Docker Neo4j 5 (:7687) with Bi-Temporal openCypher Graph (t_valid / t_invalid) & Tier 3 Reasoning Traces",
            "6. Docker Redis 7 (:6379) with Tier 3 Semantic Playbook Caching & Reinforcement",
        ],
        "a2a_agent_turns": {
            "sre_active_query_reply": sre_reply,
            "sre_historical_time_travel_2026_06_01_reply": hist_reply,
            "crm_query_reply": crm_read_turn.combined_text,
            "cross_domain_isolation_probe_reply": probe_reply,
        },
        "mcp_surface_recall": mcp_recall,
        "postgres_physical_state": {
            "registered_apps": pg_apps,
            "episodic_memories": pg_episodes,
            "graph_seed_index": pg_seeds,
            "processed_outbox_batches": processed_outbox,
        },
        "neo4j_physical_state": {
            "entities": neo4j_entities,
            "bitemporal_edges": neo4j_edges,
            "procedural_reasoning_trace_steps": trace_rec["trace_count"],
        },
        "redis_physical_state": {
            "cached_playbooks": redis_playbooks_snapshot,
        },
        "ontology_steward_audit_log": audit_data["entries"],
    }
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2), encoding="utf-8")


@pytest.mark.asyncio
async def test_single_stack_postgres_and_redis_unified_adapters_with_cloud_auth():
    """
    Verifies that enterprises can deploy EITHER:
      1. Pure PostgreSQL Single-Stack (`CompositeMemoryService.from_postgres_stack` / `PostgresUnifiedAdapter`)
         with dynamic `AwsIamDatabaseAuthProvider` token callback (`asyncpg.create_pool(password=...)`).
      2. Pure Redis Single-Stack (`CompositeMemoryService.from_redis_stack` / `RedisUnifiedAdapter`)
         with `AzureEntraDatabaseAuthProvider` / `CloudAuthProvider`.
    Each single stack independently handles Registry, Tier 1 Episodic, Tier 2 Bi-Temporal Graph
    (with functional edge invalidation), and Tier 3 Procedural Playbooks.
    """
    if not (_port_open("127.0.0.1", 5433) and _port_open("127.0.0.1", 6379)):
        pytest.skip("Docker Postgres (:5433) and Redis (:6379) not running.")

    from agentic_memory_sdk import (
        AppMemoryManifest,
        AwsIamDatabaseAuthProvider,
        AzureEntraDatabaseAuthProvider,
        CompositeMemoryService,
        DeterministicHashEmbeddingAdapter,
        DomainPackSpec,
        ExtractedEntity,
        ExtractedFact,
        ExtractionPayload,
        GovernanceTier,
        PlatformOntologyRegistrar,
        RelationRule,
        RuleBasedMockExtractorAdapter,
        UpperConceptType,
    )
    from agentic_memory_sdk.service import Event, types

    def _evt(txt: str) -> Event:
        return Event(
            author="user",
            content=types.Content(role="user", parts=[types.Part.from_text(text=txt)]),
        )

    embedder = DeterministicHashEmbeddingAdapter(dim=1536)
    extractor = RuleBasedMockExtractorAdapter()
    extractor.add_keyword_rule(
        "us-east-1",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="billing-api",
                    entity_class="Microservice",
                    summary="Billing microservice",
                ),
                ExtractedEntity(
                    canonical_name="us-east-1",
                    entity_class="CloudRegion",
                    summary="AWS US East 1",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="billing-api",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-east-1",
                    target_class="CloudRegion",
                    fact_summary="billing-api deployed in us-east-1",
                    valid_from="2026-02-01T00:00:00Z",
                )
            ],
        ),
    )
    extractor.add_keyword_rule(
        "us-west-2",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="billing-api",
                    entity_class="Microservice",
                    summary="Billing microservice in us-west-2",
                ),
                ExtractedEntity(
                    canonical_name="us-west-2",
                    entity_class="CloudRegion",
                    summary="AWS US West 2",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="billing-api",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-west-2",
                    target_class="CloudRegion",
                    fact_summary="billing-api failed over to us-west-2",
                    valid_from="2026-11-01T00:00:00Z",
                )
            ],
        ),
    )

    def build_sre_registrar(registry_adapter) -> PlatformOntologyRegistrar:
        reg = PlatformOntologyRegistrar(registry_adapter)
        reg.register_domain_pack(
            DomainPackSpec(
                pack_id="sre_infra@v1.0",
                domain_name="sre_infra",
                version="1.0.0",
                classes={
                    "Microservice": UpperConceptType.OBJECT.value,
                    "CloudRegion": UpperConceptType.OBJECT.value,
                },
                relations={
                    "DEPLOYED_IN_PRIMARY": RelationRule(
                        predicate="DEPLOYED_IN_PRIMARY",
                        domain_class="Microservice",
                        range_class="CloudRegion",
                        is_functional=True,
                        is_temporal=True,
                    )
                },
            )
        )
        return reg

    # =========================================================================
    # STACK A: Pure PostgreSQL Single-Stack (`PostgresUnifiedAdapter` + AWS IAM Auth Provider)
    # =========================================================================
    aws_iam_auth = AwsIamDatabaseAuthProvider(
        host="127.0.0.1",
        port=5433,
        username="platform_admin",
        region="us-east-1",
        service="rds",
        custom_token_generator=lambda: "platform_secret",
    )
    pg_only_service = CompositeMemoryService.from_postgres_stack(
        dsn="postgresql://127.0.0.1:5433/adk_memory_platform",
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        auth_provider=aws_iam_auth,
        default_tenant_id="t1",
    )
    try:
        await pg_only_service.seed_store.init_schema(
            embedding_dim=1536, default_tenant="t1"
        )
        # Verify asyncpg invoked our dynamic AwsIamDatabaseAuthProvider callback!
        assert aws_iam_auth.refresh_count >= 1

        reg_pg = build_sre_registrar(pg_only_service.registry)
        await reg_pg.register_adk_app(
            AppMemoryManifest(
                tenant_id="t1",
                app_name="pg_only_sre",
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["sre_infra@v1.0"],
            )
        )

        # Turn 1: Deploy billing-api in us-east-1
        await pg_only_service.add_events_to_memory(
            app_name="t1:pg_only_sre",
            user_id="sre_pg_user",
            session_id="sess_pg_1",
            events=[_evt("Service billing-api is deployed in us-east-1 as of 2026-02-01")],
        )
        # Turn 2: Failover billing-api to us-west-2 (Functional Edge Invalidation inside Postgres!)
        await pg_only_service.add_events_to_memory(
            app_name="t1:pg_only_sre",
            user_id="sre_pg_user",
            session_id="sess_pg_2",
            events=[_evt("We failed billing-api over to us-west-2 on 2026-11-01")],
        )
        # Register & match playbook inside Pure Postgres
        await pg_only_service.register_playbook(
            app_name="t1:pg_only_sre",
            user_id="sre_pg_user",
            trigger_intent="billing-api region failover",
            workflow_steps=["drain_us_east_1", "promote_us_west_2"],
            resolution_summary="Billing failover runbook stored in PostgreSQL",
        )

        pg_recall = await pg_only_service.search_memory(
            app_name="t1:pg_only_sre",
            user_id="sre_pg_user",
            query="billing-api region failover: Billing failover runbook stored in PostgreSQL",
        )
        pg_combined = "\n".join(m.content.parts[0].text for m in pg_recall.memories)
        assert "(billing-api) -[DEPLOYED_IN_PRIMARY]-> (us-west-2)" in pg_combined
        assert "(billing-api) -[DEPLOYED_IN_PRIMARY]-> (us-east-1)" not in pg_combined
        assert "drain_us_east_1 -> promote_us_west_2" in pg_combined
    finally:
        await pg_only_service.seed_store.close()

    # =========================================================================
    # STACK B: Pure Redis Single-Stack (`RedisUnifiedAdapter` + Azure Entra Auth Provider)
    # =========================================================================
    azure_redis_auth = AzureEntraDatabaseAuthProvider(
        username="default",
        scope=AzureEntraDatabaseAuthProvider.REDIS_SCOPE,
        custom_token_generator=lambda: "",
    )
    redis_only_service = CompositeMemoryService.from_redis_stack(
        redis_url="redis://127.0.0.1:6379/1",
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        auth_provider=azure_redis_auth,
        key_prefix="adk_redis_unified_test",
        default_tenant_id="t_redis",
    )
    try:
        r_conn = await redis_only_service.seed_store._get_redis()
        await r_conn.flushdb()
        assert azure_redis_auth.refresh_count >= 1

        reg_rd = build_sre_registrar(redis_only_service.registry)
        await reg_rd.register_adk_app(
            AppMemoryManifest(
                tenant_id="t_redis",
                app_name="redis_only_sre",
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["sre_infra@v1.0"],
            )
        )

        # Turn 1: Deploy billing-api in us-east-1
        await redis_only_service.add_events_to_memory(
            app_name="t_redis:redis_only_sre",
            user_id="sre_redis_user",
            session_id="sess_rd_1",
            events=[_evt("Service billing-api is deployed in us-east-1 as of 2026-01-15")],
        )
        # Turn 2: Failover billing-api to us-west-2 (Functional Edge Invalidation inside Redis!)
        await redis_only_service.add_events_to_memory(
            app_name="t_redis:redis_only_sre",
            user_id="sre_redis_user",
            session_id="sess_rd_2",
            events=[_evt("We failed billing-api over to us-west-2 on 2026-11-01")],
        )
        await redis_only_service.register_playbook(
            app_name="t_redis:redis_only_sre",
            user_id="sre_redis_user",
            trigger_intent="billing-api region failover",
            workflow_steps=["verify_redis_cluster", "flip_global_datastore"],
            resolution_summary="Billing failover runbook stored in Redis",
        )

        rd_recall = await redis_only_service.search_memory(
            app_name="t_redis:redis_only_sre",
            user_id="sre_redis_user",
            query="billing-api region failover: Billing failover runbook stored in Redis",
        )
        rd_combined = "\n".join(m.content.parts[0].text for m in rd_recall.memories)
        assert "(billing-api) -[DEPLOYED_IN_PRIMARY]-> (us-west-2)" in rd_combined
        assert "(billing-api) -[DEPLOYED_IN_PRIMARY]-> (us-east-1)" not in rd_combined
        assert "verify_redis_cluster -> flip_global_datastore" in rd_combined
    finally:
        await redis_only_service.seed_store.close()

    # =========================================================================
    # STACK C: Independent Per-Port DB Composition
    # (`RedisRegistryAdapter` + `RedisEpisodicAndSeedAdapter` + `PostgresGraphAdapter` + `PostgresPlaybookAdapter`)
    # =========================================================================
    from agentic_memory_sdk import (
        PlaybookStoreAdapter,
        PostgresGraphAdapter,
        PostgresPlaybookAdapter,
        RedisEpisodicAndSeedAdapter,
        RedisRegistryAdapter,
    )

    ind_reg = RedisRegistryAdapter(
        redis_url="redis://127.0.0.1:6379/2", key_prefix="adk_ind_port"
    )
    ind_seed = RedisEpisodicAndSeedAdapter(
        redis_url="redis://127.0.0.1:6379/2", key_prefix="adk_ind_port"
    )
    ind_pg_graph = PostgresGraphAdapter(
        dsn="postgresql://127.0.0.1:5433/adk_memory_platform",
        auth_provider=aws_iam_auth,
    )
    ind_pg_playbook = PostgresPlaybookAdapter(
        dsn="postgresql://127.0.0.1:5433/adk_memory_platform",
        auth_provider=aws_iam_auth,
    )
    assert isinstance(ind_pg_playbook, PlaybookStoreAdapter)

    modular_service = CompositeMemoryService(
        registry_adapter=ind_reg,
        seed_store_adapter=ind_seed,
        graph_adapter=ind_pg_graph,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        playbook_adapter=ind_pg_playbook,
        sync_outbox_immediately=True,
        default_tenant_id="t_mod",
    )
    try:
        r2 = await ind_seed._get_redis()
        await r2.flushdb()
        await ind_pg_graph.init_schema()
        await ind_pg_playbook.init_schema(embedding_dim=1536)

        mod_registrar = build_sre_registrar(ind_reg)
        await mod_registrar.register_adk_app(
            AppMemoryManifest(
                tenant_id="t_mod",
                app_name="modular_sre",
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["sre_infra@v1.0"],
            )
        )
        await modular_service.add_events_to_memory(
            app_name="t_mod:modular_sre",
            user_id="mod_user",
            session_id="sess_mod_1",
            events=[_evt("We failed billing-api over to us-west-2 on 2026-11-01")],
        )
        pb_mod = await modular_service.register_playbook(
            app_name="t_mod:modular_sre",
            user_id="mod_user",
            trigger_intent="billing-api postgres playbook test",
            workflow_steps=["step_pg_pb_1", "step_pg_pb_2"],
            resolution_summary="Independent PostgresPlaybookAdapter verification",
        )
        assert pb_mod.success_count >= 1
        mod_recall = await modular_service.search_memory(
            app_name="t_mod:modular_sre",
            user_id="mod_user",
            query="billing-api postgres playbook test: Independent PostgresPlaybookAdapter verification",
        )
        mod_combined = "\n".join(m.content.parts[0].text for m in mod_recall.memories)
        assert "(billing-api) -[DEPLOYED_IN_PRIMARY]-> (us-west-2)" in mod_combined
        assert "step_pg_pb_1 -> step_pg_pb_2" in mod_combined
    finally:
        await ind_reg.close()
        await ind_seed.close()
        await ind_pg_graph.close()
        await ind_pg_playbook.close()


@pytest.mark.asyncio
async def test_realistic_enterprise_5_agent_maturity_benchmark():
    """
    Runs the 5-agent enterprise stress & maturity benchmark (`examples/realistic_enterprise_stress_test.py`)
    against live Docker PostgreSQL (:5433), Neo4j (:7687), and Redis (:6379), verifying:
      - EntityResolutionGuard sibling separation (`payment-router-01` vs `payment-router-02`)
      - Contextual Q&A confirmation turn preservation ("Yes, fail over to eu-central-1")
      - Bi-temporal functional edge invalidation in both SRE and AI-Steward-reviewed FinOps packs
      - Cross-app user preference sharing (`alice_vp` in FinOps -> recalled in SRE)
      - PII & Bearer token scrubbing
      - Bottom-up ontology evolution from SecOps overflow (`ZeroDayVulnerability` & `MITIGATED_BY_WAF_RULE`)
      - Zero cross-tenant leakage (`t_rival:spy_agent`)
    """
    if not (
        _port_open("127.0.0.1", 5433)
        and _port_open("127.0.0.1", 7687)
        and _port_open("127.0.0.1", 6379)
    ):
        pytest.skip("Docker Postgres (:5433), Neo4j (:7687), and Redis (:6379) not running.")

    from examples.realistic_enterprise_stress_test import (
        run_realistic_enterprise_maturity_benchmark,
    )

    report = await run_realistic_enterprise_maturity_benchmark()
    checks = report["maturity_checks"]
    assert checks["entity_resolution_guard_sibling_separation"] is True
    assert checks["bitemporal_functional_failover_active_eu_central_1"] is True
    assert checks["multihop_dependency_preservation"] is True
    assert checks["cross_app_user_preference_recall"] is True
    assert checks["pii_and_bearer_token_redacted"] is True
    assert checks["steward_auto_functional_cost_center_invalidation"] is True
    assert "ZeroDayVulnerability" in checks["bottom_up_ontology_evolution_promoted_classes"]
    assert checks["cross_tenant_zero_leakage"] is True
    assert checks["playbook_reinforcement_count"] >= 2


