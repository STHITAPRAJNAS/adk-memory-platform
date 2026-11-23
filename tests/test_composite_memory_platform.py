from __future__ import annotations

from pathlib import Path
from typing import Sequence

import httpx
import pytest
from pydantic import BaseModel

from agentic_memory_sdk import (
    AppMemoryManifest,
    CompiledTBox,
    CompositeMemoryService,
    DeterministicHashEmbeddingAdapter,
    DomainPackLoader,
    DomainPackSpec,
    EntityResolutionGuard,
    ExtractedEntity,
    ExtractedFact,
    ExtractionMode,
    ExtractionPayload,
    GovernanceTier,
    GraphFactReadResult,
    GraphVertexUpsert,
    InMemoryUnifiedAdapter,
    MemoryMcpServerTools,
    MemoryScope,
    PIIScrubber,
    PlatformOntologyRegistrar,
    PlatformTenantContext,
    ReasoningTraceStep,
    RedisPlaybookAndWorkingAdapter,
    RelationRule,
    RemoteCompositeMemoryService,
    RuleBasedMockExtractorAdapter,
    SmartMemoryClassifier,
    UpperConceptType,
    create_mcp_server,
    create_memory_platform_app,
    get_adk_memory_tools,
)
from agentic_memory_sdk.service import Event, types


def _make_event(text: str, author: str = "user") -> Event:
    return Event(
        author=author,
        content=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
    )


@pytest.mark.asyncio
async def test_domain_agnostic_loader_json_pydantic_and_turtle():
    """Verifies that arbitrary domains can be loaded via JSON, Pydantic models, or OWL Turtle (.ttl)."""
    repo_root = Path(__file__).parent.parent

    # 1. Load SRE Domain Pack from JSON
    sre_pack = DomainPackLoader.from_json_file(
        repo_root / "examples" / "domain_packs" / "sre_incident_ops.json"
    )
    assert sre_pack.domain_name == "sre_infra"
    assert sre_pack.relations["DEPLOYED_IN_PRIMARY"].is_functional is True
    assert sre_pack.relations["DEPENDS_ON"].is_transitive is True

    # 2. Load Clinical Trials Domain Pack from JSON
    clinical_pack = DomainPackLoader.from_json_file(
        repo_root / "examples" / "domain_packs" / "healthcare_clinical.json"
    )
    assert clinical_pack.domain_name == "clinical_trials"
    assert clinical_pack.relations["CURRENT_PHASE"].is_functional is True

    # 3. Load Custom Domain Pack from Python Pydantic classes
    class Ticket(BaseModel):
        id: str

    class SupportAgent(BaseModel):
        name: str

    pyd_pack = DomainPackLoader.from_pydantic_models(
        pack_id="helpdesk@v1.0",
        entity_models=[
            (Ticket, UpperConceptType.EVENT),
            (SupportAgent, UpperConceptType.ACTOR),
        ],
        relations=[
            RelationRule(
                predicate="HANDLED_BY",
                domain_class="Ticket",
                range_class="SupportAgent",
                is_functional=True,
            )
        ],
    )
    assert pyd_pack.classes["Ticket"] == "p:Event"
    assert pyd_pack.relations["HANDLED_BY"].is_functional is True

    # 4. Load OWL/RDFS Turtle (.ttl) file
    ttl_text = (
        repo_root / "examples" / "domain_packs" / "payments_tbox.ttl"
    ).read_text(encoding="utf-8")
    ttl_pack = DomainPackLoader.from_turtle("payments@v2.1", ttl_text)
    assert "Microservice" in ttl_pack.classes
    assert ttl_pack.relations["DEPLOYED_IN_PRIMARY"].is_functional is True


@pytest.mark.asyncio
async def test_bitemporal_conflict_invalidation_and_point_in_time_travel():
    """
    Verifies:
      - Subclass hierarchy validation (Microservice -> SoftwareAsset -> Asset)
      - Functional property (`is_functional=True`) automatically closes `t_invalid` on old edge
      - Non-functional property (`is_functional=False`) keeps multiple active edges
      - Point-in-time historical queries (`as_of_iso`) recover the superseded state
    """
    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)

    repo_root = Path(__file__).parent.parent
    sre_pack = DomainPackLoader.from_json_file(
        repo_root / "examples" / "domain_packs" / "sre_incident_ops.json"
    )
    registrar.register_domain_pack(sre_pack)

    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t1",
            app_name="sre_agent",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["sre_infra@v1.2"],
        )
    )

    # Turn 1 (Jan 2026): Auth-Service deployed in us-east-1 and reads from DB-1
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="Auth-Service",
                    entity_class="Microservice",
                    summary="Authentication microservice",
                ),
                ExtractedEntity(
                    canonical_name="us-east-1",
                    entity_class="CloudRegion",
                    summary="US East region",
                ),
                ExtractedEntity(
                    canonical_name="Aurora-01",
                    entity_class="DatabaseCluster",
                    summary="Primary relational DB",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Auth-Service",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-east-1",
                    target_class="CloudRegion",
                    fact_summary="Auth-Service runs in us-east-1",
                    valid_from="2026-01-15T00:00:00Z",
                ),
                ExtractedFact(
                    source_entity="Auth-Service",
                    source_class="Microservice",
                    predicate="READS_FROM",
                    target_entity="Aurora-01",
                    target_class="DatabaseCluster",
                    fact_summary="Auth-Service reads from Aurora-01",
                    valid_from="2026-01-15T00:00:00Z",
                ),
            ],
        )
    )
    await service.add_events_to_memory(
        app_name="t1:sre_agent",
        user_id="u1",
        events=[_make_event("Auth-Service is in us-east-1 and reads from Aurora-01")],
    )

    # Turn 2 (Nov 2026): Auth-Service fails over to us-west-2 and adds Replica-02
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="Auth-Service",
                    entity_class="Microservice",
                    summary="Authentication microservice",
                ),
                ExtractedEntity(
                    canonical_name="us-west-2",
                    entity_class="CloudRegion",
                    summary="US West region",
                ),
                ExtractedEntity(
                    canonical_name="Replica-02",
                    entity_class="DatabaseCluster",
                    summary="Read replica DB",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Auth-Service",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-west-2",
                    target_class="CloudRegion",
                    fact_summary="Auth-Service failed over to us-west-2",
                    valid_from="2026-11-01T00:00:00Z",
                ),
                ExtractedFact(
                    source_entity="Auth-Service",
                    source_class="Microservice",
                    predicate="READS_FROM",
                    target_entity="Replica-02",
                    target_class="DatabaseCluster",
                    fact_summary="Auth-Service also reads from Replica-02",
                    valid_from="2026-11-01T00:00:00Z",
                ),
            ],
        )
    )
    await service.add_events_to_memory(
        app_name="t1:sre_agent",
        user_id="u1",
        events=[_make_event("Auth-Service failed over to us-west-2 and added Replica-02")],
    )

    # Active query should show us-west-2 (NOT us-east-1) and BOTH Aurora-01 and Replica-02
    active_resp = await service.search_memory(
        app_name="t1:sre_agent", user_id="u1", query="Auth-Service deployment and databases"
    )
    graph_text = active_resp.memories[0].content.parts[0].text
    assert "us-west-2" in graph_text
    assert "(us-east-1)" not in graph_text
    assert "Aurora-01" in graph_text
    assert "Replica-02" in graph_text

    # Point-in-time historical query for June 2026 should show us-east-1 and NOT us-west-2
    historical_resp = await service.search_memory(
        app_name="t1:sre_agent",
        user_id="u1",
        query="Auth-Service deployment",
        as_of_iso="2026-06-01T00:00:00Z",
    )
    hist_text = historical_resp.memories[0].content.parts[0].text
    assert "(us-east-1)" in hist_text
    assert "(us-west-2)" not in hist_text


@pytest.mark.asyncio
async def test_two_bucket_shacl_overflow_and_multi_tenant_isolation():
    """
    Verifies:
      - Valid domain facts land in `domain:t1:customer_crm` (shared across CRM apps)
      - Invalid/emergent classes or low-confidence facts are quarantined to `app:t1:crm_bot_a`
      - `crm_bot_b` (in the same domain) sees the governed `domain:t1:customer_crm` facts,
        but CANNOT see `crm_bot_a`'s quarantined overflow or private episodes!
      - Tenant `t2` CANNOT see anything from Tenant `t1`.
    """
    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)

    repo_root = Path(__file__).parent.parent
    crm_pack = DomainPackLoader.from_json_file(
        repo_root / "examples" / "domain_packs" / "customer_crm.json"
    )
    registrar.register_domain_pack(crm_pack)

    for bot in ("crm_bot_a", "crm_bot_b"):
        await registrar.register_adk_app(
            AppMemoryManifest(
                tenant_id="t1",
                app_name=bot,
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["customer_crm@v2.0"],
                allow_emergent_overflow=True,
                min_domain_confidence=0.85,
            )
        )

    # `crm_bot_a` extracts 1 valid domain fact + 1 novel/unregistered entity class ("QuantumWidget")
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="GlobalCorp",
                    entity_class="CustomerAccount",
                    summary="Enterprise customer GlobalCorp",
                    confidence=0.95,
                ),
                ExtractedEntity(
                    canonical_name="Enterprise-Platinum",
                    entity_class="SubscriptionPlan",
                    summary="Platinum subscription plan",
                    confidence=0.95,
                ),
                ExtractedEntity(
                    canonical_name="SecretPrototypeX",
                    entity_class="UnregisteredExperimentalClass",
                    summary="Experimental feature mentioned in chat",
                    confidence=0.92,
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="GlobalCorp",
                    source_class="CustomerAccount",
                    predicate="SUBSCRIBED_TO_TIER",
                    target_entity="Enterprise-Platinum",
                    target_class="SubscriptionPlan",
                    fact_summary="GlobalCorp is on the Enterprise-Platinum plan",
                    confidence=0.95,
                ),
                ExtractedFact(
                    source_entity="GlobalCorp",
                    source_class="CustomerAccount",
                    predicate="INVENTED_PREDICATE",
                    target_entity="SecretPrototypeX",
                    target_class="UnregisteredExperimentalClass",
                    fact_summary="GlobalCorp asked about SecretPrototypeX",
                    confidence=0.90,
                ),
            ],
        )
    )

    await service.add_events_to_memory(
        app_name="t1:crm_bot_a",
        user_id="alice",
        events=[_make_event("GlobalCorp upgraded to Enterprise-Platinum and asked about SecretPrototypeX")],
    )

    # 1. `crm_bot_a` can see BOTH the shared domain fact AND its own `app:t1:crm_bot_a` overflow fact
    res_a = await service.search_memory(
        app_name="t1:crm_bot_a", user_id="alice", query="GlobalCorp SecretPrototypeX"
    )
    text_a = res_a.memories[0].content.parts[0].text
    assert "[domain:t1:customer_crm] (GlobalCorp) -[SUBSCRIBED_TO_TIER]-> (Enterprise-Platinum)" in text_a
    assert "[app:t1:crm_bot_a] (GlobalCorp) -[RELATED_TO]-> (SecretPrototypeX)" in text_a

    # 2. `crm_bot_b` (different app in the same `customer_crm` domain) sees the governed domain fact,
    #    but is strictly blocked from seeing `crm_bot_a`'s private overflow or episodic turns!
    res_b = await service.search_memory(
        app_name="t1:crm_bot_b", user_id="bob", query="GlobalCorp SecretPrototypeX"
    )
    text_b = res_b.memories[0].content.parts[0].text
    assert "[domain:t1:customer_crm] (GlobalCorp) -[SUBSCRIBED_TO_TIER]-> (Enterprise-Platinum)" in text_b
    assert "SecretPrototypeX" not in text_b
    assert len(res_b.memories) == 1  # No private episodes from `crm_bot_a` leaked

    # 3. Tenant `t2` sees zero memories
    res_t2 = await service.search_memory(
        app_name="t2:crm_bot_a", user_id="alice", query="GlobalCorp"
    )
    assert len(res_t2.memories) == 0


@pytest.mark.asyncio
async def test_edge_cases_pii_circuit_breaker_and_gdpr_purge():
    """
    Verifies:
      - Edge 7: PII / API keys are scrubbed before embedding and extraction
      - Edge 5: Graph DB outage triggers circuit breaker fallback to verified seed summaries
      - Edge 8: GDPR `purge_user_lineage` cascades deletion across episodes and derived graph edges
    """
    raw = "My SSN is 123-45-6789 and api_key='sk-1234567890abcdef12345678'."
    scrubbed = PIIScrubber.scrub(raw)
    assert "123-45-6789" not in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "sk-1234567890abcdef12345678" not in scrubbed

    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()

    # Broken Graph Adapter to simulate Neptune / Neo4j outage
    class BrokenGraphAdapter(InMemoryUnifiedAdapter):
        async def expand_active_subgraph(
            self,
            ctx: PlatformTenantContext,
            seed_vertex_ids: Sequence[str],
            as_of_iso: str | None = None,
            max_hops: int = 2,
            limit: int = 25,
        ) -> list[GraphFactReadResult]:
            raise ConnectionError("Simulated Graph DB cluster timeout")

    broken_graph = BrokenGraphAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=broken_graph,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=False,
    )

    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="Project-Apollo",
                    entity_class="p:Concept",
                    summary="Next-gen platform migration initiative",
                )
            ],
            facts=[],
        )
    )
    await service.add_events_to_memory(
        app_name="t1:notes_app",
        user_id="user_99",
        events=[_make_event("Working on Project-Apollo with SSN 123-45-6789")],
    )

    # Search when Graph is down -> Degrades gracefully to Seed Cache without failing the ADK turn
    resp = await service.search_memory(
        app_name="t1:notes_app", user_id="user_99", query="Project-Apollo"
    )
    assert len(resp.memories) >= 1
    assert "GRAPH_DEGRADED_FALLBACK_MODE" in resp.memories[0].content.parts[0].text
    assert "Project-Apollo" in resp.memories[0].content.parts[0].text

    # GDPR Purge removes all user episodes
    purged = await service.purge_user_lineage(app_name="t1:notes_app", user_id="user_99")
    assert len(purged) == 1
    assert len(store.episodes) == 0


@pytest.mark.asyncio
async def test_fastapi_memory_service_and_remote_adk_client():
    """
    Verifies the centralized FastAPI Memory-as-a-Service server and the thin
    `RemoteCompositeMemoryService` client used by 100s of ADK agent apps.
    """
    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)
    app = create_memory_platform_app(service, registrar)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://memory-platform") as http_client:
        # 1. Register a Domain Pack over REST using OWL Turtle (.ttl)
        repo_root = Path(__file__).parent.parent
        ttl_content = (
            repo_root / "examples" / "domain_packs" / "payments_tbox.ttl"
        ).read_text(encoding="utf-8")
        r_pack = await http_client.post(
            "/v1/domains/packs/turtle",
            json={"pack_id": "payments@v2.1", "turtle_content": ttl_content},
        )
        assert r_pack.status_code == 200
        assert r_pack.json()["domain_name"] == "payments"

        # 2. Connect an ADK agent using `RemoteCompositeMemoryService`
        remote_memory = RemoteCompositeMemoryService(
            base_url="http://memory-platform",
            client=http_client,
        )
        await remote_memory.register_manifest(
            AppMemoryManifest(
                tenant_id="jpmc",
                app_name="dispute_resolution_agent",
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["payments@v2.1"],
            )
        )

        # 3. Ingest events via RemoteCompositeMemoryService
        extractor.enqueue(
            ExtractionPayload(
                entities=[
                    ExtractedEntity(
                        canonical_name="Ledger-API",
                        entity_class="Microservice",
                        summary="Core ledger settlement microservice",
                    ),
                    ExtractedEntity(
                        canonical_name="eu-central-1",
                        entity_class="Region",
                        summary="Frankfurt region",
                    ),
                ],
                facts=[
                    ExtractedFact(
                        source_entity="Ledger-API",
                        source_class="Microservice",
                        predicate="DEPLOYED_IN_PRIMARY",
                        target_entity="eu-central-1",
                        target_class="Region",
                        fact_summary="Ledger-API primary region is eu-central-1",
                    )
                ],
            )
        )
        await remote_memory.add_events_to_memory(
            app_name="jpmc:dispute_resolution_agent",
            user_id="analyst_1",
            events=[_make_event("Ledger-API is deployed in eu-central-1.")],
        )

        # 4. Search memory via RemoteCompositeMemoryService
        search_res = await remote_memory.search_memory(
            app_name="jpmc:dispute_resolution_agent",
            user_id="analyst_1",
            query="Where is Ledger-API deployed?",
        )
        assert len(search_res.memories) == 2
        assert "Ledger-API" in search_res.memories[0].content.parts[0].text
        assert "eu-central-1" in search_res.memories[0].content.parts[0].text

        # 5. Record Procedural Decision Trace (Tier 3)
        trace_res = await remote_memory.record_decision_trace(
            app_name="jpmc:dispute_resolution_agent",
            user_id="analyst_1",
            run_id="run_101",
            session_id="sess_1",
            steps=[
                ReasoningTraceStep(
                    step_index=1,
                    tool_name="check_region_health",
                    tool_input_summary="region=eu-central-1",
                    tool_output_summary="status=HEALTHY",
                    touched_vertex_ids=[
                        GraphVertexUpsert.deterministic_id(
                            "jpmc", "domain:jpmc:payments", "Region", "eu-central-1"
                        )
                    ],
                )
            ],
        )
        assert trace_res["status"] == "recorded"

        # 6. Test MCP Server Tools wrapper & Co-Hosted FastAPI + MCP HTTP endpoints
        mcp_tools = MemoryMcpServerTools(service)
        mcp_recall = await mcp_tools.recall_context(
            app_name="jpmc:dispute_resolution_agent",
            user_id="analyst_1",
            query="Ledger-API",
        )
        assert len(mcp_recall["memories"]) >= 1

        # Verify co-hosted MCP endpoints on the same FastAPI app
        mcp_list_resp = await http_client.get("/v1/mcp/tools")
        assert mcp_list_resp.status_code == 200
        assert "recall_context" in mcp_list_resp.json()["tools"]

        mcp_call_resp = await http_client.post(
            "/v1/mcp/call",
            json={
                "tool_name": "recall_context",
                "arguments": {
                    "app_name": "jpmc:dispute_resolution_agent",
                    "user_id": "analyst_1",
                    "query": "Ledger-API",
                },
            },
        )
        assert mcp_call_resp.status_code == 200
        assert len(mcp_call_resp.json()["result"]["memories"]) >= 1


@pytest.mark.asyncio
async def test_hardened_entity_resolution_guard_multihop_outbox_and_mcp():
    """
    Verifies all 5 pre-PyPI hardening upgrades:
      1. `EntityResolutionGuard` prevents false-merging of distinct identifiers (`us-east-1` vs `us-east-2`,
         `aurora-01` vs `aurora-02`) even under 0.99 vector similarity, while merging `Checkout API` and `checkout-api`.
      2. Multi-hop (`max_hops=2`) subgraph traversal discovers 2-hop chains (`Gateway-API -> Checkout-API -> Aurora-01`).
      3. Background Outbox Worker (`start_outbox_worker` / `stop_outbox_worker`) drains queued graph writes asynchronously.
      4. `create_mcp_server` (`FastMCP`) and `get_adk_memory_tools` expose full MCP & ADK tool surfaces.
    """
    # 1. Unit check on EntityResolutionGuard
    assert EntityResolutionGuard.can_merge("us-east-1", "us-east-2", 0.99, 0.91) is False
    assert EntityResolutionGuard.can_merge("aurora-01", "aurora-02", 0.98, 0.91) is False
    assert EntityResolutionGuard.can_merge("phase-2", "phase-3", 0.97, 0.91) is False
    assert EntityResolutionGuard.can_merge("Checkout API", "checkout-api", 0.80, 0.91) is True
    assert EntityResolutionGuard.can_merge("Payment Gateway", "Payment Gateway Service", 0.94, 0.91) is True

    # 2. End-to-end check with Constant-Vector Embedder (simulating 1.0 cosine similarity between all entities)
    class ConstantHighSimEmbedder:
        async def embed(self, text: str) -> list[float]:
            return [0.5] * 16

    store = InMemoryUnifiedAdapter()
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=ConstantHighSimEmbedder(),
        extractor_adapter=extractor,
        sync_outbox_immediately=False,  # Test async background worker!
    )
    registrar = PlatformOntologyRegistrar(store)
    repo_root = Path(__file__).parent.parent
    sre_pack = DomainPackLoader.from_json_file(
        repo_root / "examples" / "domain_packs" / "sre_incident_ops.json"
    )
    registrar.register_domain_pack(sre_pack)
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t1",
            app_name="sre_topology_agent",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["sre_infra@v1.2"],
        )
    )

    # Start background outbox worker with fast 50ms interval
    await service.start_outbox_worker(interval_sec=0.05)
    try:
        extractor.enqueue(
            ExtractionPayload(
                entities=[
                    ExtractedEntity(
                        canonical_name="api-gateway-01",
                        entity_class="Microservice",
                        summary="Edge ingress gateway 01",
                    ),
                    ExtractedEntity(
                        canonical_name="api-gateway-02",
                        entity_class="Microservice",
                        summary="Edge ingress gateway 02 (must NOT merge with 01 despite 1.0 cosine sim)",
                    ),
                    ExtractedEntity(
                        canonical_name="checkout-service",
                        entity_class="Microservice",
                        summary="Middle tier checkout microservice",
                    ),
                    ExtractedEntity(
                        canonical_name="aurora-orders-01",
                        entity_class="DatabaseCluster",
                        summary="Backend Aurora database cluster",
                    ),
                ],
                facts=[
                    ExtractedFact(
                        source_entity="api-gateway-01",
                        source_class="Microservice",
                        predicate="DEPENDS_ON",
                        target_entity="checkout-service",
                        target_class="Microservice",
                        fact_summary="api-gateway-01 depends on checkout-service",
                    ),
                    ExtractedFact(
                        source_entity="checkout-service",
                        source_class="Microservice",
                        predicate="READS_FROM",
                        target_entity="aurora-orders-01",
                        target_class="DatabaseCluster",
                        fact_summary="checkout-service reads from aurora-orders-01",
                    ),
                ],
            )
        )

        await service.add_events_to_memory(
            app_name="t1:sre_topology_agent",
            user_id="sre_1",
            events=[_make_event("api-gateway-01 depends on checkout-service which reads from aurora-orders-01")],
        )

        # Wait briefly for background outbox worker to drain pending graph batch
        import asyncio

        for _ in range(20):
            if not store.outbox:
                break
            await asyncio.sleep(0.03)
        assert len(store.outbox) == 0, "Background outbox worker should have drained the outbox"

        # Verify `api-gateway-01` and `api-gateway-02` were NOT merged into one vertex despite 1.0 cosine similarity
        canonical_names = {s.canonical_name for s in store.seeds.values()}
        assert "api-gateway-01" in canonical_names
        assert "api-gateway-02" in canonical_names

        # Verify 1-hop vs 2-hop expansion starting strictly from `api-gateway-01`
        ctx = await store.resolve_context("t1", "sre_topology_agent", "sre_1")
        gw1_vid = GraphVertexUpsert.deterministic_id(
            "t1", "domain:t1:sre_infra", "Microservice", "api-gateway-01"
        )
        hop1_facts = await store.expand_active_subgraph(ctx, [gw1_vid], max_hops=1)
        assert len(hop1_facts) == 1
        assert hop1_facts[0].target_name == "checkout-service"

        hop2_facts = await store.expand_active_subgraph(ctx, [gw1_vid], max_hops=2)
        assert len(hop2_facts) == 2
        hop2_targets = {f.target_name for f in hop2_facts}
        assert "checkout-service" in hop2_targets
        assert "aurora-orders-01" in hop2_targets

        # Verify FastMCP Server creation and ADK tools helper
        mcp_server = create_mcp_server(service, registrar)
        assert mcp_server is not None

        adk_tools = get_adk_memory_tools(
            service, default_app_name="t1:sre_topology_agent", default_user_id="sre_1"
        )
        assert len(adk_tools) == 2
        recall_res = await adk_tools[0](query="api-gateway-01")
        assert len(recall_res["memories"]) >= 1
    finally:
        await service.stop_outbox_worker()


@pytest.mark.asyncio
async def test_smart_memory_classification_cross_app_user_sharing_and_auto_maintenance():
    """
    Verifies:
      1. Smart Memory Classification (`SmartMemoryClassifier` + `MemoryScope`):
         - Ephemeral chitchat ("ok", "thanks!") is flagged as ephemeral and skips LLM extraction.
         - User preferences (`p:UserPreference` / `PREFERS`) route to `user:{tenant}:{user_id}` and
           follow the SAME user across different domain apps (`crm_agent` -> `sre_agent`), while remaining
           invisible to other users (`bob`).
      2. Behind-the-scenes Auto-Summarization & Auto-Cleanup (`run_memory_maintenance`):
         - Prunes ephemeral chitchat episodes.
         - Compacts multi-turn session episodes into a single `[AUTO_SESSION_SUMMARY]` record.
         - Cleans up stale low-confidence (`< 0.55`) emergent overflow edges.
    """
    assert SmartMemoryClassifier.is_ephemeral_turn("Thanks!") is True
    assert SmartMemoryClassifier.is_ephemeral_turn("ok sounds good") is True
    assert SmartMemoryClassifier.is_ephemeral_turn("checkout-api is down in us-east-1") is False
    # Contextual confirmation check: "Yes" or "ok" after a question is NOT dropped!
    assert (
        SmartMemoryClassifier.is_ephemeral_turn(
            "Yes", preceding_text="Should I fail checkout-api over to us-west-2?"
        )
        is False
    )

    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)
    repo_root = Path(__file__).parent.parent
    registrar.register_domain_pack(
        DomainPackLoader.from_json_file(repo_root / "examples" / "domain_packs" / "customer_crm.json")
    )
    registrar.register_domain_pack(
        DomainPackLoader.from_json_file(repo_root / "examples" / "domain_packs" / "sre_incident_ops.json")
    )

    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t1",
            app_name="crm_agent",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["customer_crm@v2.0"],
        )
    )
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t1",
            app_name="sre_agent",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["sre_infra@v1.2"],
        )
    )

    # 1. User `alice` tells `crm_agent` her personal communication preference + a noisy low-confidence rumour
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="Alice",
                    entity_class="p:Actor",
                    summary="VP of Engineering Alice",
                    scope=MemoryScope.USER_GLOBAL,
                ),
                ExtractedEntity(
                    canonical_name="Concise-Markdown-Tables",
                    entity_class="p:UserPreference",
                    summary="Prefers executive updates formatted as concise markdown tables",
                    scope=MemoryScope.USER_GLOBAL,
                ),
                ExtractedEntity(
                    canonical_name="Unverified-Rumour-Node",
                    entity_class="UnknownClass",
                    summary="Low-confidence rumor",
                    confidence=0.40,
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Alice",
                    source_class="p:Actor",
                    predicate="PREFERS",
                    target_entity="Concise-Markdown-Tables",
                    target_class="p:UserPreference",
                    fact_summary="Alice prefers concise markdown tables in all responses",
                    confidence=0.95,
                    scope=MemoryScope.USER_GLOBAL,
                ),
                ExtractedFact(
                    source_entity="Alice",
                    source_class="p:Actor",
                    predicate="HEARD_RUMOUR",
                    target_entity="Unverified-Rumour-Node",
                    target_class="UnknownClass",
                    fact_summary="Low confidence rumor",
                    confidence=0.40,
                ),
            ],
        )
    )
    await service.add_events_to_memory(
        app_name="t1:crm_agent",
        user_id="alice",
        session_id="sess_alice_1",
        events=[
            _make_event("I always prefer concise markdown tables in responses."),
            _make_event("Understood, I will format summaries as markdown tables.", author="crm_agent"),
            _make_event("Also check ACME renewal pipeline for Q4."),
            _make_event("ACME Q4 renewal pipeline is on track.", author="crm_agent"),
            _make_event("Thanks!"),  # Ephemeral chitchat!
        ],
    )

    # 2. Verify cross-app user preference sharing:
    #    When `alice` talks to `sre_agent` (a completely different domain!), `sre_agent` sees `user:t1:alice`
    sre_alice = await service.search_memory(
        app_name="t1:sre_agent",
        user_id="alice",
        query="Alice Concise-Markdown-Tables preference",
    )
    sre_alice_text = sre_alice.memories[0].content.parts[0].text
    assert "[user:t1:alice] (Alice) -[PREFERS]-> (Concise-Markdown-Tables)" in sre_alice_text

    # 3. Verify User Isolation: when `bob` talks to `sre_agent` or `crm_agent`, Bob CANNOT see Alice's preference!
    sre_bob = await service.search_memory(
        app_name="t1:sre_agent",
        user_id="bob",
        query="Alice Concise-Markdown-Tables preference",
    )
    assert len(sre_bob.memories) == 0

    # 3.5. Verify LLM-based ephemeral classification on a long 14-word pleasantry that bypasses word-count heuristics!
    extractor.enqueue(
        ExtractionPayload(
            is_ephemeral_chitchat=True,
            classification_reason="Multi-sentence weekend pleasantry with no durable facts",
            entities=[],
            facts=[],
        )
    )
    await service.add_events_to_memory(
        app_name="t1:crm_agent",
        user_id="alice",
        session_id="sess_alice_1",
        events=[
            _make_event(
                "Hey everyone, hope you all have a wonderful weekend ahead and talk Monday!"
            )
        ],
    )
    assert store.episodes[-1].metadata.get("ephemeral") is True
    assert (
        store.episodes[-1].metadata.get("ephemeral_reason")
        == "Multi-sentence weekend pleasantry with no durable facts"
    )

    # 4. Run behind-the-scenes Auto-Summarization & Auto-Cleanup
    stats = await service.run_memory_maintenance(
        min_episodes_to_compact=4,
        min_overflow_confidence=0.55,
    )
    assert stats["pruned_ephemeral_episodes"] == 2
    assert stats["compacted_raw_episodes"] == 4
    assert stats["summaries_created"] == 1
    assert stats["pruned_low_confidence_overflow_edges"] == 1
    assert len(store.episodes) == 1
    assert store.episodes[0].metadata["is_summary"] is True
    assert "[AUTO_SESSION_SUMMARY (4 turns)]" in store.episodes[0].content_text


@pytest.mark.asyncio
async def test_ontology_review_auto_evolution_from_overflow_and_audit_provenance():
    """
    Verifies:
      1. Rich Layer 0 Defaults (`CompiledTBox()`): Zero-config agents automatically have
         functional `HAS_STATUS`, `ASSIGNED_TO`, `OWNED_BY`, `LOCATED_IN` and transitive `DEPENDS_ON`, `PART_OF`.
      2. Top-Down Ontology Review (`review_and_refine_domain_pack`): Automatically anchors unrooted classes
         under Layer 0 `UpperConceptType`, normalizes predicates, infers `is_functional=True` / `is_transitive=True`,
         repairs missing domain/range classes, and logs an `OntologyAuditLogEntry`.
      3. Bottom-Up Schema Evolution (`evolve_app_ontology_from_overflow`): Promotes high-confidence
         quarantined overflow classes & predicates into the live TBox and promotes the quarantined ABox
         vertices/edges while keeping `source_episode_id`, `t_valid`, `t_created`, and `user_id` provenance intact!
    """
    # 1. Verify rich Layer 0 defaults out-of-the-box
    default_tbox = CompiledTBox()
    assert default_tbox.relations["HAS_STATUS"].is_functional is True
    assert default_tbox.relations["ASSIGNED_TO"].is_functional is True
    assert default_tbox.relations["DEPENDS_ON"].is_transitive is True

    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)

    # 2. Top-Down Review of an incomplete/imperfect DomainPackSpec submitted by a service team
    draft_pack = DomainPackSpec(
        pack_id="logistics@v1.0",
        domain_name="logistics",
        version="1.0.0",
        classes={
            "ShipmentOrder": "",  # Unanchored!
            "CarrierAgent": "owl:Thing",  # Unanchored!
        },
        relations={
            "assigned-to-primary": RelationRule(
                predicate="assigned-to-primary",
                domain_class="ShipmentOrder",
                range_class="WarehouseFacility",  # Missing from classes!
                is_functional=False,  # Author forgot is_functional=True on a primary relation!
            ),
            "depends-on": RelationRule(
                predicate="depends-on",
                domain_class="ShipmentOrder",
                range_class="ShipmentOrder",
                is_transitive=False,  # Author forgot is_transitive=True!
            ),
        },
    )
    refined_pack, review_audit = registrar.review_and_refine_domain_pack(draft_pack, tenant_id="t1")
    assert refined_pack.version == "1.0.1"
    assert refined_pack.classes["CarrierAgent"] == "p:Actor"
    assert refined_pack.classes["WarehouseFacility"] == "p:Object"
    assert refined_pack.relations["ASSIGNED_TO_PRIMARY"].is_functional is True
    assert refined_pack.relations["DEPENDS_ON"].is_transitive is True
    assert review_audit.action == "LLM_REVIEW_AND_REFINE"
    assert len(review_audit.modifications_summary) >= 4

    # 3. Bottom-Up Evolution from Quarantined Overflow with Full Provenance Preservation
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t1",
            app_name="logistics_bot",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["logistics@v1.0"],
            allow_emergent_overflow=True,
        )
    )

    # During runtime, agent encounters a novel class `AutonomousDroneService` and predicate `DEPLOYED_IN_PRIMARY`
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="Drone-Alpha-01",
                    entity_class="AutonomousDroneService",  # Not in logistics@v1.0 yet!
                    summary="Heavy-lift delivery drone",
                    confidence=0.92,
                ),
                ExtractedEntity(
                    canonical_name="Hub-East-01",
                    entity_class="WarehouseFacility",
                    summary="Eastern fulfillment hub",
                    confidence=0.95,
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Drone-Alpha-01",
                    source_class="AutonomousDroneService",
                    predicate="deployed-in-primary",  # Not in logistics@v1.0 yet!
                    target_entity="Hub-East-01",
                    target_class="WarehouseFacility",
                    fact_summary="Drone-Alpha-01 is deployed in Hub-East-01",
                    confidence=0.92,
                    valid_from="2026-11-10T00:00:00Z",
                )
            ],
        )
    )
    await service.add_events_to_memory(
        app_name="t1:logistics_bot",
        user_id="ops_lead",
        events=[_make_event("Drone-Alpha-01 is deployed in Hub-East-01")],
    )

    # Before evolution: edge is quarantined in `app:t1:logistics_bot` as `RELATED_TO`
    assert store.edges[0]["namespace"] == "app:t1:logistics_bot"
    assert store.edges[0]["predicate"] == "RELATED_TO"
    original_ep_id = store.edges[0]["source_episode_id"]
    original_t_valid = store.edges[0]["t_valid"]
    original_t_created = store.edges[0]["t_created"]

    # Trigger behind-the-scenes Bottom-Up Ontology Evolution!
    evolve_audit = await service.evolve_app_ontology_from_overflow(
        "t1:logistics_bot",
        min_confidence_to_promote=0.80,
        registrar=registrar,
    )
    assert evolve_audit.action == "AUTO_EVOLVE_FROM_OVERFLOW"
    assert evolve_audit.new_version == "1.1.0"
    assert evolve_audit.added_or_modified_classes["AutonomousDroneService"] == "p:Object"
    assert "DEPLOYED_IN_PRIMARY" in evolve_audit.added_or_modified_relations
    assert evolve_audit.added_or_modified_relations["DEPLOYED_IN_PRIMARY"]["is_functional"] is True
    assert evolve_audit.source_episode_ids == [original_ep_id]

    # Verify quarantined edge was promoted to `domain:t1:logistics` with 100% intact provenance!
    promoted_edge = store.edges[0]
    assert promoted_edge["namespace"] == "domain:t1:logistics"
    assert promoted_edge["predicate"] == "DEPLOYED_IN_PRIMARY"
    assert promoted_edge["source_episode_id"] == original_ep_id
    assert promoted_edge["t_valid"] == original_t_valid
    assert promoted_edge["t_created"] == original_t_created
    assert promoted_edge["source_app"] == "logistics_bot"
    assert promoted_edge["user_id"] == "ops_lead"


@pytest.mark.asyncio
async def test_hot_path_vs_sleep_time_batch_extraction_and_redis_playbook_cache():
    """
    Verifies:
      1. Both `ExtractionMode.HOT` and `ExtractionMode.SLEEP_TIME` are supported per-app or per-call:
         - In `SLEEP_TIME` mode, 3 separate `add_events_to_memory` turns commit in <10ms with ZERO
           LLM extraction calls (`len(store.seeds) == 0`).
         - When `run_memory_maintenance()` runs, all 3 pending turns are batch-extracted in 1 single
           LLM call (`sleep_time_extracted_batches == 1`, `sleep_time_extracted_episodes == 3`)!
      2. `RedisPlaybookAndWorkingAdapter` (`register_playbook`):
         - Caches and reinforces proven Tier 3 procedural playbooks (`trigger_intent -> workflow_steps`)
           and automatically injects `[MATCHED_PROCEDURAL_PLAYBOOKS]` on `search_memory()`.
    """
    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()
    playbook_cache = RedisPlaybookAndWorkingAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        playbook_adapter=playbook_cache,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)
    repo_root = Path(__file__).parent.parent
    registrar.register_domain_pack(
        DomainPackLoader.from_json_file(
            repo_root / "examples" / "domain_packs" / "sre_incident_ops.json"
        )
    )

    # Register `sre_batch_bot` with `ExtractionMode.SLEEP_TIME`
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t1",
            app_name="sre_batch_bot",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["sre_infra@v1.2"],
            extraction_mode=ExtractionMode.SLEEP_TIME,
        )
    )

    # Queue only ONE extraction payload to prove 3 hot turns only consume 1 batched LLM call during sleep-time!
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="billing-worker",
                    entity_class="Microservice",
                    summary="Async billing worker service",
                ),
                ExtractedEntity(
                    canonical_name="us-east-1",
                    entity_class="CloudRegion",
                    summary="AWS US East 1",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="billing-worker",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-east-1",
                    target_class="CloudRegion",
                    fact_summary="billing-worker runs in us-east-1",
                )
            ],
        )
    )

    # Ingest 3 separate turns on the hot path -> Should defer extraction!
    for msg in (
        "Investigating billing-worker latency.",
        "Confirmed billing-worker is deployed in us-east-1.",
        "Restarted billing-worker pods.",
    ):
        await service.add_events_to_memory(
            app_name="t1:sre_batch_bot",
            user_id="sre_oncall",
            session_id="sess_batch_1",
            events=[_make_event(msg)],
        )

    # Verify zero graph vertices were extracted on the hot path and queued_payloads still has the 1 item!
    assert len(store.seeds) == 0
    assert len(extractor.queued_payloads) == 1

    # Now run Sleep-Time Maintenance -> Batches all 3 pending turns into 1 LLM call!
    maint_stats = await service.run_memory_maintenance(min_episodes_to_compact=10)
    assert maint_stats["sleep_time_extracted_batches"] == 1
    assert maint_stats["sleep_time_extracted_episodes"] == 3
    assert len(extractor.queued_payloads) == 0
    assert len(store.seeds) == 2
    assert len(store.edges) == 1

    # Register a reusable Tier 3 Agent Playbook (reinforced twice)
    await service.register_playbook(
        app_name="t1:sre_batch_bot",
        user_id="sre_oncall",
        trigger_intent="billing-worker us-east-1 pod restart runbook",
        workflow_steps=["drain_envoy_ingress", "kubectl_rollout_restart", "verify_prometheus_p99"],
        resolution_summary="Drain Envoy before rolling restart of billing-worker in us-east-1",
    )
    pb = await service.register_playbook(
        app_name="t1:sre_batch_bot",
        user_id="sre_oncall",
        trigger_intent="billing-worker us-east-1 pod restart runbook",
        workflow_steps=["drain_envoy_ingress", "kubectl_rollout_restart", "verify_prometheus_p99"],
        resolution_summary="Drain Envoy before rolling restart of billing-worker in us-east-1",
    )
    assert pb.success_count == 2

    # Search memory -> Should recall BOTH `[MATCHED_PROCEDURAL_PLAYBOOKS]` AND `[ACTIVE_ONTOLOGY_GRAPH_FACTS]`!
    recall = await service.search_memory(
        app_name="t1:sre_batch_bot",
        user_id="sre_oncall",
        query="billing-worker us-east-1 pod restart runbook",
    )
    authors = [m.author for m in recall.memories]
    assert "platform_playbook_cache" in authors
    assert "platform_knowledge_graph" in authors
    pb_text = recall.memories[0].content.parts[0].text
    assert "drain_envoy_ingress -> kubectl_rollout_restart -> verify_prometheus_p99" in pb_text
    assert "verified=2x" in pb_text


@pytest.mark.asyncio
async def test_enterprise_cloud_auth_providers_and_token_rotation():
    """
    Verifies Enterprise Cloud Database Auth Providers (`AwsIamDatabaseAuthProvider`,
    `GcpCloudSqlIamAuthProvider`, `AzureEntraDatabaseAuthProvider`,
    `VaultOrSecretManagerAuthProvider`, `EnterpriseTlsConfig`):
      - Token caching within TTL window (avoids hitting IAM/STS on every query).
      - Forced rotation via `invalidate()`.
      - Dynamic username & password rotation via `VaultOrSecretManagerAuthProvider`.
      - mTLS / SSLContext generation via `EnterpriseTlsConfig`.
    """
    import ssl
    from agentic_memory_sdk import (
        AwsIamDatabaseAuthProvider,
        AzureEntraDatabaseAuthProvider,
        EnterpriseTlsConfig,
        GcpCloudSqlIamAuthProvider,
        VaultOrSecretManagerAuthProvider,
    )

    counter = {"aws": 0, "gcp": 0, "azure": 0, "vault": 0}

    def next_aws_token() -> str:
        counter["aws"] += 1
        return f"aws-rds-iam-token-v{counter['aws']}"

    aws_auth = AwsIamDatabaseAuthProvider(
        host="aurora-prod.cluster-xyz.us-east-1.rds.amazonaws.com",
        port=5432,
        username="adk_iam_role",
        region="us-east-1",
        service="rds",
        token_ttl_sec=900,
        refresh_margin_sec=120,
        custom_token_generator=next_aws_token,
    )
    # First call mints token v1; second call hits cache without minting
    assert await aws_auth.get_password() == "aws-rds-iam-token-v1"
    assert await aws_auth.get_password() == "aws-rds-iam-token-v1"
    assert aws_auth.refresh_count == 1
    # Invalidate forces rotation on next connection acquisition
    aws_auth.invalidate()
    assert await aws_auth.get_password() == "aws-rds-iam-token-v2"
    assert aws_auth.refresh_count == 2

    gcp_auth = GcpCloudSqlIamAuthProvider(
        username="sa-memory@my-gcp-project.iam",
        custom_token_generator=lambda: "ya29.gcp-alloydb-oauth-token",
    )
    assert await gcp_auth.get_credentials() == (
        "sa-memory@my-gcp-project.iam",
        "ya29.gcp-alloydb-oauth-token",
    )

    azure_auth = AzureEntraDatabaseAuthProvider(
        username="mi-adk-memory-prod",
        scope=AzureEntraDatabaseAuthProvider.POSTGRES_SCOPE,
        custom_token_generator=lambda: "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.azure-entra",
    )
    assert await azure_auth.get_password() == "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.azure-entra"

    async def fetch_vault_secret() -> tuple[str, str]:
        counter["vault"] += 1
        return f"v_db_user_{counter['vault']}", f"v_db_secret_{counter['vault']}"

    vault_auth = VaultOrSecretManagerAuthProvider(secret_fetcher=fetch_vault_secret)
    u1, p1 = await vault_auth.get_credentials()
    assert (u1, p1) == ("v_db_user_1", "v_db_secret_1")
    vault_auth.invalidate()
    u2, p2 = await vault_auth.get_credentials()
    assert (u2, p2) == ("v_db_user_2", "v_db_secret_2")

    tls_cfg = EnterpriseTlsConfig(enabled=True, verify_hostname=True)
    ssl_ctx = tls_cfg.build_ssl_context()
    assert isinstance(ssl_ctx, ssl.SSLContext)
    assert ssl_ctx.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.asyncio
async def test_bedrock_openai_and_ollama_model_and_embedding_adapters():
    """
    Verifies the multi-provider Model & Embedding adapters:
      1. `BedrockEmbeddingAdapter` (Amazon Titan V2 + Cohere Embed v3 + target_dim=1536 alignment)
      2. `BedrockOntologyExtractorAdapter` (Amazon Bedrock Converse `toolUse` structured extraction)
      3. `OpenAICompatibleEmbeddingAdapter` & `OpenAICompatibleOntologyExtractorAdapter`
      4. `OllamaEmbeddingAdapter` & `OllamaOntologyExtractorAdapter`
    """
    import io
    import json
    import httpx
    from agentic_memory_sdk import (
        AzureEntraDatabaseAuthProvider,
        BedrockEmbeddingAdapter,
        BedrockOntologyExtractorAdapter,
        CompiledTBox,
        OllamaEmbeddingAdapter,
        OllamaOntologyExtractorAdapter,
        OpenAICompatibleEmbeddingAdapter,
        OpenAICompatibleOntologyExtractorAdapter,
    )

    # 1. Amazon Bedrock Mock Client (Titan V2, Cohere v3, and Claude 3.5 Converse)
    class _FakeBedrockRuntime:
        def __init__(self):
            self.last_model_id = ""
            self.last_body = {}

        def invoke_model(self, *, modelId: str, body: str, **kwargs):
            self.last_model_id = modelId
            self.last_body = json.loads(body)
            if "cohere" in modelId:
                payload = {"embeddings": [[0.5, 0.5, 0.5, 0.5]]}
            else:
                payload = {"embedding": [0.25] * self.last_body.get("dimensions", 8)}
            return {"body": io.BytesIO(json.dumps(payload).encode("utf-8"))}

        def converse(self, *, modelId: str, messages: list, toolConfig: dict, **kwargs):
            self.last_model_id = modelId
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tu_1",
                                    "name": "emit_extraction_payload",
                                    "input": {
                                        "is_ephemeral_chitchat": False,
                                        "classification_reason": "SRE deployment fact",
                                        "entities": [
                                            {
                                                "canonical_name": "orders-api",
                                                "entity_class": "Microservice",
                                                "summary": "Orders API on Bedrock",
                                                "confidence": 0.96,
                                                "scope": "auto",
                                            }
                                        ],
                                        "facts": [],
                                    },
                                }
                            }
                        ],
                    }
                }
            }

    fake_bedrock = _FakeBedrockRuntime()
    titan_emb = BedrockEmbeddingAdapter(
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=8,
        target_dim=16,
        bedrock_client=fake_bedrock,
    )
    vec_titan = await titan_emb.embed("orders-api failover")
    assert len(vec_titan) == 16
    assert vec_titan[:8] == [0.25] * 8
    assert vec_titan[8:] == [0.0] * 8

    cohere_emb = BedrockEmbeddingAdapter(
        model_id="cohere.embed-english-v3",
        target_dim=4,
        bedrock_client=fake_bedrock,
    )
    vec_cohere = await cohere_emb.embed("orders-api failover")
    assert vec_cohere == [0.5, 0.5, 0.5, 0.5]

    bedrock_extractor = BedrockOntologyExtractorAdapter(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        bedrock_client=fake_bedrock,
    )
    extracted_bd = await bedrock_extractor.extract(
        "orders-api is running in us-east-1", CompiledTBox()
    )
    assert len(extracted_bd.entities) == 1
    assert extracted_bd.entities[0].canonical_name == "orders-api"

    # 2. OpenAI / Azure OpenAI Compatible Adapters + CloudAuthProvider Bearer token
    seen_headers: dict[str, str] = {}

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        path = request.url.path
        if path.endswith("/embeddings"):
            return httpx.Response(
                200, json={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}
            )
        if path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "is_ephemeral_chitchat": False,
                                        "classification_reason": "",
                                        "entities": [
                                            {
                                                "canonical_name": "openai-svc",
                                                "entity_class": "p:Object",
                                                "summary": "Extracted via OpenAI adapter",
                                            }
                                        ],
                                        "facts": [],
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        if path.endswith("/api/embed"):
            return httpx.Response(200, json={"embeddings": [[0.6, 0.8]]})
        if path.endswith("/api/chat"):
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": json.dumps(
                            {
                                "is_ephemeral_chitchat": False,
                                "classification_reason": "",
                                "entities": [
                                    {
                                        "canonical_name": "ollama-local-svc",
                                        "entity_class": "p:Object",
                                        "summary": "Extracted via local Ollama",
                                    }
                                ],
                                "facts": [],
                            }
                        )
                    }
                },
            )
        return httpx.Response(404)

    mock_http = httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler))
    try:
        entra_auth = AzureEntraDatabaseAuthProvider(
            username="azure-openai",
            custom_token_generator=lambda: "entra-jwt-token-xyz",
        )
        oai_emb = OpenAICompatibleEmbeddingAdapter(
            model="text-embedding-3-small",
            dimensions=4,
            auth_provider=entra_auth,
            http_client=mock_http,
        )
        v_oai = await oai_emb.embed("hello world")
        assert v_oai == [0.1, 0.2, 0.3, 0.4]
        assert seen_headers.get("authorization") == "Bearer entra-jwt-token-xyz"

        oai_ext = OpenAICompatibleOntologyExtractorAdapter(
            model="gpt-4o-mini",
            auth_provider=entra_auth,
            http_client=mock_http,
        )
        p_oai = await oai_ext.extract("openai-svc test", CompiledTBox())
        assert p_oai.entities[0].canonical_name == "openai-svc"

        # 3. Ollama Local Neural Embedding & Extractor Adapters
        ollama_emb = OllamaEmbeddingAdapter(
            model="nomic-embed-text",
            target_dim=4,
            http_client=mock_http,
        )
        v_ol = await ollama_emb.embed("local embed test")
        assert v_ol == [0.6, 0.8, 0.0, 0.0]

        ollama_ext = OllamaOntologyExtractorAdapter(
            model="qwen2.5:7b",
            http_client=mock_http,
        )
        p_ol = await ollama_ext.extract("ollama-local-svc test", CompiledTBox())
        assert p_ol.entities[0].canonical_name == "ollama-local-svc"
    finally:
        await mock_http.aclose()


@pytest.mark.asyncio
async def test_retail_omnichannel_suite_and_three_consumption_patterns():
    """
    Verifies:
      1. The 4-Agent Omnichannel Retail & Supply-Chain Example Suite (`examples/retail_omnichannel_suite.py`)
         and its W3C OWL Turtle specification (`examples/domain_packs/retail_omnichannel_tbox.ttl`).
      2. All 3 Agent Consumption Patterns (`examples/three_consumption_patterns_demo.py`):
         - Pattern 1: Direct In-Process SDK (`CompositeMemoryService` + `get_adk_memory_tools`)
         - Pattern 2: FastAPI REST Service + `RemoteCompositeMemoryService`
         - Pattern 3: MCP Server (`create_mcp_server`, `MemoryMcpServerTools`, `/v1/mcp/call`)
    """
    from pathlib import Path
    from examples.retail_omnichannel_suite import run_retail_omnichannel_suite
    from examples.three_consumption_patterns_demo import run_three_consumption_patterns_demo

    ttl_path = (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "domain_packs"
        / "retail_omnichannel_tbox.ttl"
    )
    retail_ttl_spec = DomainPackLoader.from_turtle_file(
        "retail_omnichannel@1.0.0", ttl_path
    )
    assert "RetailCustomer" in retail_ttl_spec.classes
    assert retail_ttl_spec.relations["ENROLLED_IN_TIER"].is_functional is True
    assert retail_ttl_spec.relations["SHIPPED_FROM_PRIMARY"].is_functional is True

    retail_res = await run_retail_omnichannel_suite()
    assert all(retail_res["checks"].values()), f"Retail checks failed: {retail_res['checks']}"

    patterns_res = await run_three_consumption_patterns_demo()
    assert all(patterns_res.values()), f"Consumption pattern checks failed: {patterns_res}"


@pytest.mark.asyncio
async def test_latest_mcp_protocol_2026_07_28_and_2025_11_25_streamable_http():
    """
    Verifies compliance with the latest Model Context Protocol specifications
    (`2026-07-28` Stateless Request/Response Core with Header Routing `Mcp-Method` / `Mcp-Name`,
    and `2025-11-25` Streamable HTTP) over `/mcp/http/`:
      1. `tools/list` returns all 11 tools with explicit `ToolAnnotations` (`readOnlyHint`,
         `destructiveHint`, `idempotentHint`, `openWorldHint`, `title`) and `outputSchema`.
      2. `tools/call` under `2026-07-28` (`Mcp-Method`, `Mcp-Name`, `_meta` envelope) returns
         typed `structuredContent` alongside `TextContent`.
      3. `resources/list`, `resources/templates/list`, and `resources/read` expose live
         `memory://governance/health`, `memory://domains/packs`, `memory://audit/ontology`,
         and `memory://apps/{tenant_id}/{app_name}/tbox`.
      4. `prompts/list` and `prompts/get` expose `memory_grounded_assistant` and
         `ontology_steward_domain_designer`.
    """
    store = InMemoryUnifiedAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=DeterministicHashEmbeddingAdapter(),
        extractor_adapter=RuleBasedMockExtractorAdapter(),
    )
    app = create_memory_platform_app(service=service, enable_outbox_worker=False)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
    ) as client:
        # 1. Verify /v1/mcp/tools metadata endpoint
        meta_resp = await client.get("/v1/mcp/tools")
        assert meta_resp.status_code == 200
        meta_json = meta_resp.json()
        assert meta_json["protocol_version"] == "2026-07-28"
        assert len(meta_json["tools"]) == 17
        assert "induce_domain_pack" in meta_json["tools"]
        assert "check_answerable" in meta_json["tools"]
        assert "certify_domain_pack" in meta_json["tools"]
        assert "edit_working_memory_block" in meta_json["tools"]
        assert len(meta_json["resources"]) == 4
        assert len(meta_json["prompts"]) == 2

        # 2. Verify 2025-11-25 Streamable HTTP tools/list + ToolAnnotations
        r_tools = await client.post(
            "/mcp/http/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-11-25",
            },
        )
        assert r_tools.status_code == 200
        tools_list = r_tools.json()["result"]["tools"]
        assert len(tools_list) == 17
        by_name = {t["name"]: t for t in tools_list}
        assert by_name["recall_context"]["annotations"]["readOnlyHint"] is True
        assert by_name["recall_context"]["annotations"]["destructiveHint"] is False
        assert by_name["purge_user_lineage"]["annotations"]["destructiveHint"] is True
        assert "outputSchema" in by_name["recall_context"]

        # 3. Verify 2026-07-28 Stateless Core tools/call with Mcp-Method & Mcp-Name routing headers
        r_call_2026 = await client.post(
            "/mcp/http/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "inspect_platform_governance",
                    "arguments": {},
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                        "io.modelcontextprotocol/clientCapabilities": {},
                    },
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2026-07-28",
                "Mcp-Method": "tools/call",
                "Mcp-Name": "inspect_platform_governance",
            },
        )
        assert r_call_2026.status_code == 200
        call_res = r_call_2026.json()["result"]
        assert call_res["structuredContent"]["circuit_breaker_state"] == "CLOSED"

        # 4. Verify MCP Resources (resources/list, resources/templates/list, resources/read)
        r_res_list = await client.post(
            "/mcp/http/",
            json={"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-11-25",
            },
        )
        assert r_res_list.status_code == 200
        res_uris = {r["uri"] for r in r_res_list.json()["result"]["resources"]}
        assert "memory://governance/health" in res_uris
        assert "memory://domains/packs" in res_uris
        assert "memory://audit/ontology" in res_uris

        r_res_read = await client.post(
            "/mcp/http/",
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": {"uri": "memory://governance/health"},
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-11-25",
            },
        )
        assert r_res_read.status_code == 200
        assert "CLOSED" in r_res_read.json()["result"]["contents"][0]["text"]

        # 5. Verify MCP Prompts (prompts/list, prompts/get)
        r_prompts = await client.post(
            "/mcp/http/",
            json={"jsonrpc": "2.0", "id": 5, "method": "prompts/list", "params": {}},
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-11-25",
            },
        )
        assert r_prompts.status_code == 200
        prompt_names = {p["name"] for p in r_prompts.json()["result"]["prompts"]}
        assert "memory_grounded_assistant" in prompt_names
        assert "ontology_steward_domain_designer" in prompt_names


@pytest.mark.asyncio
async def test_autonomous_ontology_induction_canonicalization_and_empirical_cardinality():
    """
    Verifies all 4 Autonomous Ontology Discovery & Evolution upgrades:
      1. LLM Extractor Two-Bucket Prompt instructs models to emit descriptive PascalCase classes
         and UPPER_SNAKE_CASE predicates when `allow_emergent_overflow=True`.
      2. Class & Predicate Canonicalization (`ColdChainShipments` / `cold_chain_shipment` -> `ColdChainShipment`,
         `shippedViaCarrier` -> `SHIPPED_VIA_CARRIER`) + Multi-Domain Layer-0 Upper Ontology parent inference
         (Retail, Healthcare, Supply Chain, Finance, SRE).
      3. Multi-occurrence `min_support_count` gating + Empirical ABox Graph Cardinality inference
         (`1:1` functional vs. `1:N` multi-valued) in `evolve_app_ontology_from_overflow`.
      4. Cold-Start Autonomous Domain Pack Induction (`induce_domain_pack`) across SDK, FastAPI REST
         (`POST /v1/domains/packs/induce`), and MCP (`induce_domain_pack`).
    """
    from agentic_memory_sdk.llm_extractors import GeminiOntologyExtractorAdapter

    # 1. Verify Two-Bucket Discovery Prompt vs Strict Closed-World Prompt
    open_tbox = CompiledTBox(allow_emergent_overflow=True)
    closed_tbox = CompiledTBox(allow_emergent_overflow=False)
    open_prompt = GeminiOntologyExtractorAdapter.build_tbox_system_instruction(open_tbox)
    closed_prompt = GeminiOntologyExtractorAdapter.build_tbox_system_instruction(closed_tbox)
    assert "EMERGENT ONTOLOGY DISCOVERY" in open_prompt
    assert "PascalCase" in open_prompt
    assert "STRICT CLOSED-WORLD TBOX MODE" in closed_prompt

    # 2. Verify Canonicalization & Expanded Multi-Domain Layer-0 Alignment
    assert PlatformOntologyRegistrar.canonicalize_class_name("cold_chain_shipments") == "ColdChainShipment"
    assert PlatformOntologyRegistrar.canonicalize_class_name("ColdChainShipments") == "ColdChainShipment"
    assert PlatformOntologyRegistrar.canonicalize_class_name("ReturnPolicies") == "ReturnPolicy"
    assert PlatformOntologyRegistrar.canonicalize_class_name("OrderStatus") == "OrderStatus"
    assert PlatformOntologyRegistrar.canonicalize_predicate_name("shippedViaCarrier") == "SHIPPED_VIA_CARRIER"

    assert PlatformOntologyRegistrar.infer_upper_concept_parent("FulfillmentWarehouse") == "p:Object"
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("AttendingPhysician") == "p:Actor"
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("ColdChainShipment") == "p:Event"
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("LoyaltyTier") == "p:Concept"

    # Empirical 1:N observation overrides non-explicit functional assumption, while >=3 1:1 pairs infer functional
    func_1_to_1, _ = PlatformOntologyRegistrar.infer_relation_flags(
        "HANDLED_BY_COURIER",
        observed_pairs=[("SHP-1", "COUR-A"), ("SHP-2", "COUR-B"), ("SHP-3", "COUR-A")],
    )
    assert func_1_to_1 is True
    func_1_to_n, _ = PlatformOntologyRegistrar.infer_relation_flags(
        "INCLUDES_SKU",
        observed_pairs=[("ORD-1", "SKU-1"), ("ORD-1", "SKU-2")],
    )
    assert func_1_to_n is False

    # 3. Verify `min_support_count=2` gating + synonym pooling in `evolve_app_ontology_from_overflow`
    store = InMemoryUnifiedAdapter()
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=DeterministicHashEmbeddingAdapter(),
        extractor_adapter=extractor,
    )
    registrar = PlatformOntologyRegistrar(store)
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t_induce",
            app_name="cold_chain_agent",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            allow_emergent_overflow=True,
        )
    )

    # Turn 1: `ColdChainShipments` appears once, `NoiseArtifact` appears once
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="SHP-901",
                    entity_class="ColdChainShipments",
                    summary="Vaccine cold chain shipment 901",
                    confidence=0.92,
                ),
                ExtractedEntity(
                    canonical_name="NOISE-1",
                    entity_class="OneOffHallucination",
                    summary="Single-occurrence noise entity",
                    confidence=0.85,
                ),
            ],
            facts=[],
        )
    )
    await service.add_events_to_memory(
        app_name="t_induce:cold_chain_agent",
        user_id="ops_1",
        events=[_make_event("ColdChainShipments SHP-901 departed")],
    )

    # Turn 2: `cold_chain_shipment` appears a second time (pools with `ColdChainShipments` -> support=2!)
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="SHP-902",
                    entity_class="cold_chain_shipment",
                    summary="Biologics cold chain shipment 902",
                    confidence=0.94,
                ),
            ],
            facts=[],
        )
    )
    await service.add_events_to_memory(
        app_name="t_induce:cold_chain_agent",
        user_id="ops_1",
        events=[_make_event("cold_chain_shipment SHP-902 departed")],
    )

    # Evolve with min_support_count=2: `ColdChainShipment` (support=2) is promoted, `OneOffHallucination` (support=1) stays quarantined!
    audit = await service.evolve_app_ontology_from_overflow(
        "t_induce:cold_chain_agent",
        min_confidence_to_promote=0.80,
        min_support_count=2,
        registrar=registrar,
    )
    assert "ColdChainShipment" in audit.added_or_modified_classes
    assert audit.added_or_modified_classes["ColdChainShipment"] == "p:Event"
    assert "OneOffHallucination" not in audit.added_or_modified_classes

    # 4. Verify Cold-Start Autonomous Domain Pack Induction (`induce_domain_pack`) via FastAPI & MCP
    app = create_memory_platform_app(
        service=service, registrar=registrar, enable_outbox_worker=False
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
    ) as client:
        remote_client = RemoteCompositeMemoryService(
            base_url="http://127.0.0.1", http_client=client
        )
        induced_res = await remote_client.induce_domain_pack(
            domain_name="supply_chain_auto",
            transcripts=[
                "Order ORD-501 shipped from FC-Seattle-01 and includes SKU-100 and SKU-200.",
                "Order ORD-502 shipped from FC-Reno-02 and includes SKU-300.",
                "Microservice order-router-01 depends on inventory-service-01 and deployed in us-west-2.",
            ],
            min_support_count=1,
            auto_register=True,
        )
        assert induced_res["status"] == "induced_and_registered"
        pack_data = induced_res["pack"]
        assert pack_data["classes"]["CustomerOrder"] == "p:Event"
        assert pack_data["classes"]["FulfillmentCenter"] == "p:Object"
        assert pack_data["classes"]["ProductSku"] == "p:Object"
        assert pack_data["relations"]["SHIPPED_FROM_PRIMARY"]["is_functional"] is True
        assert pack_data["relations"]["INCLUDES_SKU"]["is_functional"] is False
        assert pack_data["relations"]["DEPENDS_ON"]["is_transitive"] is True

        # Also test MCP `induce_domain_pack` via `/v1/mcp/call`
        mcp_induce = await client.post(
            "/v1/mcp/call",
            json={
                "tool_name": "induce_domain_pack",
                "arguments": {
                    "domain_name": "secops_auto",
                    "transcripts": [
                        "Incident INC-9001 affects payment-gateway-01 and mitigated by Patch(KB-5501)."
                    ],
                },
            },
        )
        assert mcp_induce.status_code == 200
        mcp_pack = mcp_induce.json()["result"]["pack"]
        assert "IncidentEvent" in mcp_pack["classes"]
        assert "MITIGATED_BY" in mcp_pack["relations"]


@pytest.mark.asyncio
async def test_sql_ddl_ossie_yaml_check_answerable_and_steward_certification():
    """
    Verifies the 3 lightweight capabilities inspired by NVIDIA Auto-Ontology:
      1. `DomainPackLoader.from_sql_ddl()` and `POST /v1/domains/packs/sql-ddl`:
         Parses relational `CREATE TABLE` DDL, foreign keys, and bridge tables into
         a Layer-1 `DomainPackSpec` anchored under Layer 0 Upper Ontology.
      2. `DomainPackLoader.from_ossie_yaml()` and `POST /v1/domains/packs/ossie`:
         Parses Apache Ossie / BI Semantic Model YAML (`terms`, `synonyms`, `relationships`,
         `sql_attributes`) into a Layer-1 `DomainPackSpec`.
      3. `check_answerable()` across SDK, FastAPI (`POST /v1/memory/check-answerable`),
         `RemoteCompositeMemoryService`, and MCP (`check_answerable`):
         Accurately distinguishes `ANSWERABLE` (coverage=1.0), `PARTIAL` (e.g., order covered
         but "ad spend" missing from ontology and memory), and `UNANSWERABLE` (coverage=0.0).
      4. Steward Certification (`certify_domain_pack`) across SDK, FastAPI, and MCP.
    """
    # 1. Test `DomainPackLoader.from_sql_ddl` directly
    sample_ddl = """
    CREATE TABLE sales.customers (
        customer_id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL
    );
    CREATE TABLE warehouse.fulfillment_centers (
        fc_id TEXT PRIMARY KEY,
        region_code TEXT NOT NULL
    );
    CREATE TABLE warehouse.stock_items (
        sku_id TEXT PRIMARY KEY,
        item_name TEXT NOT NULL
    );
    CREATE TABLE sales.customer_orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT REFERENCES sales.customers(customer_id),
        primary_fc_id TEXT REFERENCES warehouse.fulfillment_centers(fc_id)
    );
    CREATE TABLE sales.order_lines (
        line_id TEXT PRIMARY KEY,
        FOREIGN KEY (order_id) REFERENCES sales.customer_orders(order_id),
        FOREIGN KEY (sku_id) REFERENCES warehouse.stock_items(sku_id)
    );
    """
    sql_pack = DomainPackLoader.from_sql_ddl("wwi_sales@1.0.0", sample_ddl)
    assert sql_pack.classes["Customer"] == "p:Actor"
    assert sql_pack.classes["CustomerOrder"] == "p:Event"
    assert sql_pack.classes["FulfillmentCenter"] == "p:Object"
    assert sql_pack.classes["StockItem"] == "p:Object"
    assert sql_pack.relations["HAS_CUSTOMER"].domain_class == "CustomerOrder"
    assert sql_pack.relations["HAS_CUSTOMER"].range_class == "Customer"
    assert sql_pack.relations["HAS_CUSTOMER"].is_functional is True
    assert sql_pack.relations["HAS_PRIMARY_FC"].is_functional is True
    # Bridge table `order_lines` synthesizes `INCLUDES_STOCK_ITEM` (many-to-many -> is_functional=False)
    assert "INCLUDES_STOCK_ITEM" in sql_pack.relations
    assert sql_pack.relations["INCLUDES_STOCK_ITEM"].is_functional is False

    # 2. Test `DomainPackLoader.from_ossie_yaml` directly
    sample_ossie_yaml = """
terms:
  - name: CustomerOrder
    synonyms: [purchase_order, invoice, sales_order]
  - name: FulfillmentCenter
    synonyms: [warehouse, depot]
  - name: LoyaltyTier
    synonyms: [membership_level, vip_tier]
relationships:
  - name: SHIPPED_FROM_PRIMARY
    source: CustomerOrder
    target: FulfillmentCenter
    cardinality: many_to_one
  - name: ENROLLED_IN_TIER
    source: RetailCustomer
    target: LoyaltyTier
    cardinality: many_to_one
"""
    ossie_pack = DomainPackLoader.from_ossie_yaml(
        "retail_ossie@1.0.0", sample_ossie_yaml
    )
    assert "CustomerOrder" in ossie_pack.classes
    assert "RetailCustomer" in ossie_pack.classes
    assert "purchase_order" in ossie_pack.synonyms["CustomerOrder"]
    assert ossie_pack.relations["SHIPPED_FROM_PRIMARY"].is_functional is True
    assert ossie_pack.relations["ENROLLED_IN_TIER"].is_functional is True

    # 3. End-to-End FastAPI + RemoteClient + MCP test for SQL DDL, Ossie, Certification & `check_answerable`
    store = InMemoryUnifiedAdapter()
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=DeterministicHashEmbeddingAdapter(),
        extractor_adapter=extractor,
    )
    registrar = PlatformOntologyRegistrar(store)
    app = create_memory_platform_app(
        service=service, registrar=registrar, enable_outbox_worker=False
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
    ) as client:
        remote_client = RemoteCompositeMemoryService(
            base_url="http://127.0.0.1", http_client=client
        )

        # Register via REST `/v1/domains/packs/sql-ddl` & `/v1/domains/packs/ossie`
        ddl_resp = await remote_client.register_domain_pack_from_sql_ddl(
            "wwi_sales@1.0.0", sample_ddl
        )
        assert ddl_resp["status"] == "reviewed_and_registered"

        ossie_resp = await remote_client.register_domain_pack_from_ossie(
            "retail_ossie@1.0.0", sample_ossie_yaml
        )
        assert ossie_resp["status"] == "reviewed_and_registered"
        assert ossie_resp["certified"] is False

        # Certify `retail_ossie@1.0.0` via REST `/v1/domains/packs/certify`
        cert_resp = await remote_client.certify_domain_pack(
            "retail_ossie@1.0.0", certified_by="chief_data_steward"
        )
        assert cert_resp["status"] == "certified"
        assert cert_resp["certified"] is True
        assert cert_resp["certified_by"] == "chief_data_steward"

        # Onboard app subscribed to `retail_ossie@1.0.0` and ingest order ORD-88410
        await remote_client.register_manifest(
            AppMemoryManifest(
                tenant_id="nordstrom",
                app_name="analytics_concierge",
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["retail_ossie@1.0.0"],
            )
        )
        extractor.enqueue(
            ExtractionPayload(
                entities=[
                    ExtractedEntity(
                        canonical_name="ORD-88410",
                        entity_class="CustomerOrder",
                        summary="Order ORD-88410 shipped from FC-Seattle-01",
                    ),
                    ExtractedEntity(
                        canonical_name="FC-Seattle-01",
                        entity_class="FulfillmentCenter",
                        summary="Primary Seattle fulfillment center",
                    ),
                ],
                facts=[
                    ExtractedFact(
                        source_entity="ORD-88410",
                        source_class="CustomerOrder",
                        predicate="SHIPPED_FROM_PRIMARY",
                        target_entity="FC-Seattle-01",
                        target_class="FulfillmentCenter",
                        fact_summary="Order ORD-88410 shipped from FC-Seattle-01",
                    )
                ],
            )
        )
        await remote_client.add_events_to_memory(
            app_name="nordstrom:analytics_concierge",
            user_id="analyst_maya",
            events=[_make_event("Order ORD-88410 shipped from FC-Seattle-01.")],
        )

        # Case A: Fully Answerable Question (100% coverage)
        ans_full = await remote_client.check_answerable(
            app_name="nordstrom:analytics_concierge",
            user_id="analyst_maya",
            query="Where was order ORD-88410 shipped from?",
        )
        assert ans_full["answerable"] is True
        assert ans_full["status"] == "ANSWERABLE"
        assert ans_full["coverage_score"] == 1.0
        assert "ORD-88410" in ans_full["matched_entities"]

        # Case B: Partial Coverage Question (asks about 'ad spend' which is not in the ontology or memory!)
        ans_partial = await remote_client.check_answerable(
            app_name="nordstrom:analytics_concierge",
            user_id="analyst_maya",
            query="What was the ad spend for order ORD-88410?",
        )
        assert ans_partial["answerable"] is False
        assert ans_partial["status"] == "PARTIAL"
        assert 0.0 < ans_partial["coverage_score"] < 1.0
        assert "ad spend" in ans_partial["unmatched_concepts"]

        # Case C: Completely Unanswerable Question (0.0 coverage) via MCP `check_answerable`
        mcp_unans = await client.post(
            "/v1/mcp/call",
            json={
                "tool_name": "check_answerable",
                "arguments": {
                    "app_name": "nordstrom:analytics_concierge",
                    "user_id": "analyst_maya",
                    "query": "What is the payroll tax and carbon footprint in Antarctica?",
                },
            },
        )
        assert mcp_unans.status_code == 200
        unans_res = mcp_unans.json()["result"]
        assert unans_res["answerable"] is False
        assert unans_res["status"] == "UNANSWERABLE"
        assert unans_res["coverage_score"] == 0.0


@pytest.mark.asyncio
async def test_langgraph_base_store_stategraph_nodes_tools_and_chat_history():
    """
    Verifies all 4 LangGraph & LangChain integration adapters in `langgraph_integration.py`
    (reachable via both `agentic_composite_memory` and `agentic_memory_sdk`):
      1. `LangGraphCompositeMemoryStore` (`BaseStore` interface: `aput`, `aget`, `asearch`,
         `adelete`, `alist_namespaces`, `abatch`).
      2. `create_langgraph_memory_nodes` (`recall_node`, `check_answerable_node`, `commit_node`
         with automatic tool-call reasoning trace extraction).
      3. `get_langgraph_memory_tools` (`recall_memory_context`, `check_memory_answerable`,
         `commit_memory_episode`, `register_memory_playbook`, `record_memory_decision_trace`).
      4. `CompositeMemoryChatMessageHistory` (`aadd_messages`, `aget_messages`, `aclear`).
    """
    from agentic_composite_memory import (
        CompositeMemoryChatMessageHistory,
        LangGraphCompositeMemoryStore,
        create_langgraph_memory_nodes,
        get_langgraph_memory_tools,
    )
    from agentic_composite_memory.langgraph_integration import AIMessage, HumanMessage

    store = InMemoryUnifiedAdapter()
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=DeterministicHashEmbeddingAdapter(),
        extractor_adapter=extractor,
    )
    registrar = PlatformOntologyRegistrar(store)
    await registrar. induce_domain_pack_from_payloads if False else None
    _, _, _ = await service.induce_domain_pack(
        domain_name="sre_lg",
        transcripts=[
            "Microservice payment-router-01 deployed in us-east-1 and depends on ledger-db-01."
        ],
        auto_register=True,
        registrar=registrar,
    )
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="acme",
            app_name="langgraph_sre",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["sre_lg@1.0.0"],
        )
    )

    # 1. Test `LangGraphCompositeMemoryStore` (`aput`, `aget`, `asearch`, `alist_namespaces`)
    lg_store = LangGraphCompositeMemoryStore(
        service,
        default_app_name="acme:langgraph_sre",
        default_user_id="alice_sre",
    )
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="payment-router-01",
                    entity_class="Microservice",
                    summary="Core payment router service",
                ),
                ExtractedEntity(
                    canonical_name="us-east-1",
                    entity_class="CloudRegion",
                    summary="AWS US East region",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="payment-router-01",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN",
                    target_entity="us-east-1",
                    target_class="CloudRegion",
                    fact_summary="payment-router-01 deployed in us-east-1",
                )
            ],
        )
    )
    ns = ("acme:langgraph_sre", "alice_sre", "memories")
    await lg_store.aput(
        ns,
        "mem_1",
        {"text": "payment-router-01 is deployed in us-east-1."},
    )
    fetched_item = await lg_store.aget(ns, "mem_1")
    assert fetched_item is not None
    assert fetched_item.value["text"] == "payment-router-01 is deployed in us-east-1."

    namespaces = await lg_store.alist_namespaces(prefix=("acme:langgraph_sre",))
    assert ns in namespaces

    search_hits = await lg_store.asearch(
        ("acme:langgraph_sre", "alice_sre"),
        query="Where is payment-router-01 deployed?",
    )
    assert len(search_hits) >= 1
    assert any("payment-router-01" in h.value["text"] for h in search_hits)

    # 2. Test `create_langgraph_memory_nodes` (`recall_node`, `check_answerable_node`, `commit_node`)
    nodes = create_langgraph_memory_nodes(
        service,
        default_app_name="acme:langgraph_sre",
        default_user_id="alice_sre",
        inject_system_message=True,
    )
    initial_state = {
        "messages": [
            HumanMessage(content="Where is payment-router-01 deployed?"),
        ]
    }
    cfg = {"configurable": {"thread_id": "lg_thread_42", "user_id": "alice_sre"}}

    ans_update = await nodes.check_answerable_node(initial_state, cfg)
    assert ans_update["answerability"]["answerable"] is True

    recall_update = await nodes.recall_node(initial_state, cfg)
    assert "payment-router-01" in recall_update["memory_context"]
    assert len(recall_update["messages"]) == 2  # Injected SystemMessage + HumanMessage

    # Simulate an AI response with tool_calls and run `commit_node`
    post_turn_state = {
        "messages": [
            HumanMessage(content="Fail over payment-router-01 to us-west-2."),
            {
                "role": "ai",
                "content": "Failing over payment-router-01 to us-west-2 now.",
                "tool_calls": [
                    {
                        "name": "trigger_region_failover",
                        "args": {"service": "payment-router-01", "target": "us-west-2"},
                    }
                ],
            },
        ]
    }
    commit_update = await nodes.commit_node(post_turn_state, cfg)
    assert commit_update["memory_committed_events"] == 2
    assert commit_update["memory_recorded_traces"] == 1

    # 3. Test `get_langgraph_memory_tools` (`recall_memory_context`, `register_memory_playbook`, etc.)
    lg_tools = {
        t.name: t
        for t in get_langgraph_memory_tools(
            service,
            default_app_name="acme:langgraph_sre",
            default_user_id="alice_sre",
        )
    }
    assert set(lg_tools.keys()) == {
        "recall_memory_context",
        "check_memory_answerable",
        "commit_memory_episode",
        "register_memory_playbook",
        "record_memory_decision_trace",
    }
    pb_res = await lg_tools["register_memory_playbook"].ainvoke(
        {
            "trigger_intent": "payment-router-01 latency failover",
            "workflow_steps": ["drain_us_east_1", "shift_traffic_us_west_2"],
            "resolution_summary": "Automated LangGraph SRE region failover SOP",
        }
    )
    assert pb_res["status"] == "registered"

    recall_tool_res = await lg_tools["recall_memory_context"].ainvoke(
        {"query": "payment-router-01 latency failover"}
    )
    assert any(
        "shift_traffic_us_west_2" in m["text"] for m in recall_tool_res["memories"]
    )

    # 4. Test `CompositeMemoryChatMessageHistory`
    history = CompositeMemoryChatMessageHistory(
        service,
        app_name="acme:langgraph_sre",
        user_id="alice_sre",
        session_id="chat_hist_1",
    )
    await history.aadd_messages(
        [
            HumanMessage(content="Check payment-router-01 status."),
            AIMessage(content="payment-router-01 is healthy."),
        ]
    )
    msgs = await history.aget_messages()
    assert len(msgs) == 2
    await history.aclear()
    assert len(history.messages) == 0


@pytest.mark.asyncio
async def test_layer0_cognitive_memory_primitives_guardrails_hypotheses_lessons_and_goals():
    """
    Verifies the 6 expanded Layer-0 Cognitive Memory Primitives & Runtime Contracts:
      1. `p:PolicyGuardrail` (`MUST_NOT_EXECUTE`, `REQUIRES_APPROVAL_FOR`, `CONSTRAINED_BY_POLICY`):
         Auto-detected by `SmartMemoryClassifier`, pinned against session compaction, and boosted
         in `TokenBudgetSerializer.compute_salience`.
      2. `p:WorkingHypothesis` (`HYPOTHESIZES_CAUSE`, `ASSUMES_STATE`):
         Prevents Epistemic Collapse—an unconfirmed hypothesis on a functional predicate (`HAS_STATUS`)
         NEVER invalidates the confirmed functional edge, and gets tagged `[UNCONFIRMED_HYPOTHESIS]`.
      3. `p:EpisodicLesson` (`LEARNED_FROM_FAILURE`, `AVOIDS_PITFALL`, `CORRECTED_BY_USER`):
         Pinned against session compaction and given top salience priority.
      4. `p:ProspectiveGoal` (`HAS_PENDING_GOAL`, `HAS_GOAL_STATUS`, `BLOCKED_BY`):
         Functional lifecycle transitions (`PENDING` -> `COMPLETED`) supersede prior goal states.
      5. `p:QuantitativeMetric` (`MEASURED_AT_VALUE`, `HAS_THRESHOLD`) & `p:SpatialLocation`:
         Functional metric measurement superseding and head-noun Layer-0 alignment.
    """
    from agentic_memory_sdk.governance import TokenBudgetSerializer

    # 1. Verify Head-Noun Alignment for all new Layer-0 Cognitive Classes
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("ComplianceGuardrail") == "p:PolicyGuardrail"
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("RootCauseHypothesis") == "p:WorkingHypothesis"
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("PostmortemLesson") == "p:EpisodicLesson"
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("MigrationGoal") == "p:ProspectiveGoal"
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("LatencyMetric") == "p:QuantitativeMetric"
    assert PlatformOntologyRegistrar.infer_upper_concept_parent("DeliveryVenue") == "p:SpatialLocation"

    store = InMemoryUnifiedAdapter()
    extractor = RuleBasedMockExtractorAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=DeterministicHashEmbeddingAdapter(),
        extractor_adapter=extractor,
    )
    registrar = PlatformOntologyRegistrar(store)
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t_cog",
            app_name="cognitive_agent",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
        )
    )

    # 2. Verify Epistemic Collapse Protection (`p:WorkingHypothesis` never invalidates a confirmed functional edge!)
    # Step 2a: Ingest confirmed functional fact: (ORD-500) -[HAS_STATUS]-> (DELIVERED)
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(canonical_name="ORD-500", entity_class="p:Event", summary="Order 500"),
                ExtractedEntity(canonical_name="DELIVERED", entity_class="p:Concept", summary="Confirmed delivered status"),
            ],
            facts=[
                ExtractedFact(
                    source_entity="ORD-500",
                    source_class="p:Event",
                    predicate="HAS_STATUS",
                    target_entity="DELIVERED",
                    target_class="p:Concept",
                    fact_summary="ORD-500 confirmed delivered at locker",
                    confidence=0.95,
                )
            ],
        )
    )
    await service.add_events_to_memory(
        app_name="t_cog:cognitive_agent",
        user_id="u1",
        session_id="cog_sess",
        events=[_make_event("Order ORD-500 is confirmed DELIVERED.")],
    )

    # Step 2b: Ingest an unconfirmed WorkingHypothesis on the SAME functional predicate HAS_STATUS!
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(canonical_name="ORD-500", entity_class="p:Event", summary="Order 500"),
                ExtractedEntity(
                    canonical_name="LOST_IN_TRANSIT_THEORY",
                    entity_class="p:WorkingHypothesis",
                    summary="Unconfirmed theory that parcel was lost",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="ORD-500",
                    source_class="p:Event",
                    predicate="HAS_STATUS",
                    target_entity="LOST_IN_TRANSIT_THEORY",
                    target_class="p:Concept",
                    fact_summary="Hypothesis: ORD-500 might be lost in transit (unconfirmed)",
                    confidence=0.90,
                )
            ],
        )
    )
    await service.add_events_to_memory(
        app_name="t_cog:cognitive_agent",
        user_id="u1",
        session_id="cog_sess",
        events=[_make_event("Hypothesis: ORD-500 might be lost in transit (unconfirmed).")],
    )

    # Verify the confirmed `(ORD-500) -[HAS_STATUS]-> (DELIVERED)` edge is STILL ACTIVE!
    active_edges = [
        e for e in store.edges if e["tenant_id"] == "t_cog" and e["t_invalid"] == "ACTIVE"
    ]
    delivered_edges = [
        e for e in active_edges if e["predicate"] == "HAS_STATUS" and "DELIVERED" in e["fact_summary"] or "confirmed delivered" in e["fact_summary"]
    ]
    assert len(delivered_edges) == 1, "Confirmed functional edge must NOT be invalidated by a WorkingHypothesis!"
    hyp_edges = [
        e for e in active_edges if "[UNCONFIRMED_HYPOTHESIS]" in e["fact_summary"]
    ]
    assert len(hyp_edges) == 1

    # 3. Verify Compaction Immunity for `p:PolicyGuardrail`, `p:EpisodicLesson`, and `p:ProspectiveGoal`
    await service.add_events_to_memory(
        app_name="t_cog:cognitive_agent",
        user_id="u1",
        session_id="compact_sess",
        events=[
            _make_event("Policy Guardrail: Must never restart aurora-ledger-01 during market hours."),
            _make_event("Lesson learned: Do not use DHL for Alaska locker deliveries; use FedEx."),
            _make_event("Prospective goal: Follow up when WH-Dallas-02 scans parcel."),
            _make_event("Routine status check 1: conveyor nominal."),
            _make_event("Routine status check 2: temperature nominal."),
            _make_event("Routine status check 3: scanner nominal."),
        ],
    )
    maint = await service.run_memory_maintenance(min_episodes_to_compact=3)
    assert maint["compacted_raw_episodes"] == 3  # Only the 3 routine status checks were compacted!

    remaining_texts = [ep.content_text for ep in store.episodes if ep.session_id == "compact_sess"]
    assert any("Policy Guardrail: Must never restart aurora-ledger-01" in t for t in remaining_texts)
    assert any("Lesson learned: Do not use DHL" in t for t in remaining_texts)
    assert any("Prospective goal: Follow up when WH-Dallas-02" in t for t in remaining_texts)
    assert any("[AUTO_SESSION_SUMMARY (3 turns)]" in t for t in remaining_texts)

    # 4. Verify Salience Ranking: Guardrail / Lesson > Standard Fact > Unconfirmed Hypothesis
    guardrail_fact = GraphFactReadResult(
        source_name="aurora-ledger-01",
        source_class="p:Object",
        predicate="CONSTRAINED_BY_POLICY",
        target_name="No-Market-Hours-Restart",
        target_class="p:PolicyGuardrail",
        fact_summary="Never restart during market hours",
        t_valid="2026-11-21T00:00:00Z",
        t_invalid="ACTIVE",
        namespace="app:t_cog:cognitive_agent",
        confidence=0.90,
    )
    standard_fact = GraphFactReadResult(
        source_name="aurora-ledger-01",
        source_class="p:Object",
        predicate="LOCATED_IN",
        target_name="us-east-1",
        target_class="p:Object",
        fact_summary="Located in us-east-1",
        t_valid="2026-11-21T00:00:00Z",
        t_invalid="ACTIVE",
        namespace="app:t_cog:cognitive_agent",
        confidence=0.90,
    )
    hypothesis_fact = GraphFactReadResult(
        source_name="aurora-ledger-01",
        source_class="p:Object",
        predicate="HYPOTHESIZES_CAUSE",
        target_name="Noisy-Neighbor",
        target_class="p:WorkingHypothesis",
        fact_summary="[UNCONFIRMED_HYPOTHESIS] Suspected noisy neighbor",
        t_valid="2026-11-21T00:00:00Z",
        t_invalid="ACTIVE",
        namespace="app:t_cog:cognitive_agent",
        confidence=0.90,
    )
    assert TokenBudgetSerializer.compute_salience(guardrail_fact) > TokenBudgetSerializer.compute_salience(standard_fact)
    assert TokenBudgetSerializer.compute_salience(standard_fact) > TokenBudgetSerializer.compute_salience(hypothesis_fact)


@pytest.mark.asyncio
async def test_v020_reconciliation_ttl_working_blocks_rbac_jwt_console_ui_and_metrics():
    """
    End-to-end validation of the v0.2.0 enterprise parity layers:
      1. Free-text cognitive memory reconciliation (`ADD -> UPDATE -> RESOLVE -> NOOP`) & Memory Revisions.
      2. Configurable `MemoryRetentionPolicy` TTL sweep & few-shot / custom-topic prompt injection.
      3. Letta-style self-editable Core Working Memory Blocks (`replace` / `append` + recall injection).
      4. Manual memory inspection, inline edit (re-embed + `MANUAL_EDIT` revision), and delete.
      5. Enterprise API AuthN/AuthZ (HS256 JWT + API-Key + RBAC + cross-tenant isolation).
      6. Bundled Visual Console UI (`GET /ui`), OpenTelemetry/Prometheus metrics (`/v1/metrics`, `/metrics`),
         and 17-tool MCP Server parity.
    """
    from datetime import datetime, timedelta, timezone
    import httpx
    from agentic_memory_sdk import (
        AuthPolicyConfig,
        AuthPrincipal,
        FewShotExtractionExample,
        GeminiOntologyExtractorAdapter,
        MemoryRetentionPolicy,
        MemoryRole,
        mint_hs256_jwt,
    )

    store = InMemoryUnifiedAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=DeterministicHashEmbeddingAdapter(),
        extractor_adapter=RuleBasedMockExtractorAdapter(),
    )
    registrar = PlatformOntologyRegistrar(store)
    tbox = await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="acme",
            app_name="concierge",
            governance_tier=GovernanceTier.TIER1_APP_TYPED,
            retention_policy=MemoryRetentionPolicy(
                hypothesis_ttl_days=7,
                completed_goal_ttl_days=14,
            ),
            custom_extraction_topics=["Travel dietary preferences", "SLA commitments"],
            few_shot_examples=[
                FewShotExtractionExample(
                    input_text="I prefer window seats on transatlantic flights.",
                    expected_entities=[{"canonical_name": "Window-Seat", "entity_class": "p:UserPreference"}],
                    expected_facts=[],
                    notes="Personal travel preference",
                )
            ],
        )
    )

    # 1. Verify custom topics & few-shot examples are injected into the TBox LLM prompt
    sys_prompt = GeminiOntologyExtractorAdapter.build_tbox_system_instruction(tbox)
    assert "CUSTOM PRIORITY EXTRACTION TOPICS" in sys_prompt
    assert "Travel dietary preferences" in sys_prompt
    assert "FEW-SHOT EXTRACTION EXAMPLES" in sys_prompt
    assert "Window-Seat" in sys_prompt

    # 2. Free-text Reconciliation (`ADD` -> `NOOP` -> `UPDATE` -> `RESOLVE`)
    app_id = "acme:concierge"
    await service.add_events_to_memory(
        app_name=app_id,
        user_id="maya",
        events=[_make_event("My preference is vegetarian meals on business flights.")],
    )
    # Duplicate turn -> NOOP (should not create duplicate episode)
    before_dup = len(store.episodes)
    await service.add_events_to_memory(
        app_name=app_id,
        user_id="maya",
        events=[_make_event("My preference is vegetarian meals on business flights.")],
    )
    assert len(store.episodes) == before_dup

    # Conflicting preference update -> UPDATE supersedes vegetarian episode
    await service.add_events_to_memory(
        app_name=app_id,
        user_id="maya",
        events=[_make_event("I switched from vegetarian meals to pescatarian meals on business flights.")],
    )
    # Goal + Resolution -> RESOLVE
    await service.add_events_to_memory(
        app_name=app_id,
        user_id="maya",
        events=[_make_event("Pending goal: renew passport before Tokyo summit.")],
    )
    await service.add_events_to_memory(
        app_name=app_id,
        user_id="maya",
        events=[_make_event("Completed renew passport before Tokyo summit.")],
    )

    # Expired TTL episode -> swept during run_memory_maintenance()
    expired_iso = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await service.add_events_to_memory(
        app_name=app_id,
        user_id="maya",
        events=[_make_event("Ephemeral one-day gate code GATE-404.")],
        custom_metadata={"expires_at": expired_iso},
    )
    maint = await service.run_memory_maintenance(min_episodes_to_compact=20)
    assert maint["expired_memories"] == 1

    # 3. Working Memory Blocks (replace + append) injected into search_memory
    service.upsert_working_memory_block(
        app_name=app_id,
        user_id="maya",
        label="human_profile",
        value="Maya Lin — VP Engineering",
        mode="replace",
    )
    service.upsert_working_memory_block(
        app_name=app_id,
        user_id="maya",
        label="human_profile",
        value="Prefers concise bullet summaries.",
        mode="append",
    )
    recalled = await service.search_memory(app_name=app_id, user_id="maya", query="business flights")
    recalled_blob = "\n".join(m.content.parts[0].text for m in recalled.memories)
    assert "[WORKING_MEMORY_BLOCKS]" in recalled_blob
    assert "Maya Lin — VP Engineering" in recalled_blob
    assert "Prefers concise bullet summaries." in recalled_blob
    assert "pescatarian" in recalled_blob.lower()
    assert "GATE-404" not in recalled_blob

    revs = service.get_memory_revisions(app_name=app_id, user_id="maya")
    rev_actions = {r.action for r in revs}
    assert {"ADD", "UPDATE", "RESOLVE", "EXPIRE"}.issubset(rev_actions)

    # 4. Enterprise FastAPI AuthN/AuthZ (JWT HS256 + API Key + RBAC + Tenant Isolation) + Console UI + Metrics
    jwt_secret = "unit-test-super-secret-key-at-least-32-bytes"
    auth_policy = AuthPolicyConfig(
        enabled=True,
        jwt_secret=jwt_secret,
        jwt_issuer="acme-idp",
        api_keys={
            "admin-key-acme": AuthPrincipal(
                subject="ops-admin",
                tenant_id="acme",
                roles=[MemoryRole.PLATFORM_ADMIN],
            ),
        },
    )
    agent_jwt = mint_hs256_jwt(
        {"sub": "concierge-bot", "tenant_id": "acme", "roles": ["agent"], "iss": "acme-idp"},
        secret=jwt_secret,
    )
    other_tenant_jwt = mint_hs256_jwt(
        {"sub": "other-bot", "tenant_id": "globex", "roles": ["agent"], "iss": "acme-idp"},
        secret=jwt_secret,
    )

    app = create_memory_platform_app(
        service=service,
        registrar=registrar,
        enable_outbox_worker=False,
        auth_policy=auth_policy,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test.local") as client:
        # Public Console UI & Health work without token
        ui_resp = await client.get("/ui")
        assert ui_resp.status_code == 200
        assert "Agentic Composite Memory Console" in ui_resp.text

        # Unauthenticated API request -> 401
        unauth = await client.get(f"/v1/memory/items?app_name={app_id}&user_id=maya")
        assert unauth.status_code == 401

        # Cross-tenant JWT spoofing -> 403 Tenant Isolation Violation
        cross_tenant = await client.get(
            f"/v1/memory/items?app_name={app_id}&user_id=maya",
            headers={"Authorization": f"Bearer {other_tenant_jwt}"},
        )
        assert cross_tenant.status_code == 403
        assert "Tenant isolation violation" in cross_tenant.json()["detail"]

        # Agent JWT can read items & edit working blocks, but CANNOT run GDPR purge (403 RBAC)
        items_resp = await client.get(
            f"/v1/memory/items?app_name={app_id}&user_id=maya",
            headers={"Authorization": f"Bearer {agent_jwt}"},
        )
        assert items_resp.status_code == 200
        items = items_resp.json()["memories"]
        assert len(items) >= 2

        purge_denied = await client.post(
            "/v1/memory/purge",
            json={"app_name": app_id, "user_id": "maya"},
            headers={"Authorization": f"Bearer {agent_jwt}"},
        )
        assert purge_denied.status_code == 403

        # Admin API Key can inline-edit & delete memory, inspect revisions & Prometheus metrics
        target_ep_id = items[0]["episode_id"]
        edit_resp = await client.put(
            f"/v1/memory/items/{target_ep_id}",
            json={
                "app_name": app_id,
                "user_id": "maya",
                "new_content": "Manually verified memory content by steward.",
                "actor": "ops-admin",
            },
            headers={"X-API-Key": "admin-key-acme"},
        )
        assert edit_resp.status_code == 200
        assert edit_resp.json()["revision"]["action"] == "MANUAL_EDIT"

        del_resp = await client.delete(
            f"/v1/memory/items/{target_ep_id}?app_name={app_id}&user_id=maya",
            headers={"X-API-Key": "admin-key-acme"},
        )
        assert del_resp.status_code == 200
        assert del_resp.json()["revision"]["action"] == "DELETE"

        metrics_json = await client.get("/v1/metrics", headers={"X-API-Key": "admin-key-acme"})
        assert metrics_json.status_code == 200
        assert metrics_json.json()["counters"]["manual_edits"] >= 1
        assert metrics_json.json()["counters"]["auth_denied"] >= 1

        metrics_prom = await client.get("/metrics", headers={"X-API-Key": "admin-key-acme"})
        assert metrics_prom.status_code == 200
        assert "adk_memory_episodes_ingested_total" in metrics_prom.text

        # Verify 17 MCP tools registered
        mcp_list = await client.get("/v1/mcp/tools", headers={"X-API-Key": "admin-key-acme"})
        assert mcp_list.status_code == 200
        assert len(mcp_list.json()["tools"]) == 17
        assert "edit_working_memory_block" in mcp_list.json()["tools"]


@pytest.mark.asyncio
async def test_builtin_configurable_sleep_time_maintenance_worker_and_fastapi_lifespan():
    """
    Verifies the opt-in built-in Sleep-Time Maintenance Worker (`start_maintenance_worker` /
    `stop_maintenance_worker` and FastAPI `enable_maintenance_worker=True`):
      1. Ingests a `SLEEP_TIME` deferred episode (`extraction_pending=True`) and an expired TTL episode.
      2. Starts the background maintenance worker via FastAPI lifespan (`maintenance_interval_sec=0.05`).
      3. Confirms the worker automatically extracts the deferred episode into the Knowledge Graph,
         sweeps the expired TTL episode, and reports `maintenance_worker_running=True` in `/health`.
      4. Confirms clean shutdown on lifespan exit (`maintenance_worker_running=False`).
    """
    import asyncio
    from datetime import datetime, timedelta, timezone
    import httpx

    store = InMemoryUnifiedAdapter()
    ext = RuleBasedMockExtractorAdapter()
    ext.add_keyword_rule(
        "checkout-api runs on eks-prod-01",
        ExtractionPayload(
            entities=[
                ExtractedEntity(canonical_name="checkout-api", entity_class="p:Object", summary="Checkout API"),
                ExtractedEntity(canonical_name="eks-prod-01", entity_class="p:Object", summary="EKS Prod Cluster 01"),
            ],
            facts=[
                ExtractedFact(
                    source_entity="checkout-api",
                    source_class="p:Object",
                    predicate="RELATED_TO",
                    target_entity="eks-prod-01",
                    target_class="p:Object",
                    fact_summary="checkout-api runs on eks-prod-01",
                )
            ],
        ),
    )
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=DeterministicHashEmbeddingAdapter(),
        extractor_adapter=ext,
        default_extraction_mode=ExtractionMode.SLEEP_TIME,
    )
    registrar = PlatformOntologyRegistrar(store)
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="t_sched",
            app_name="sre_bot",
            extraction_mode=ExtractionMode.SLEEP_TIME,
        )
    )

    # 1. Hot-path ingest in SLEEP_TIME mode (zero graph edges yet!) + expired TTL memory
    await service.add_events_to_memory(
        app_name="t_sched:sre_bot",
        user_id="u1",
        events=[_make_event("checkout-api runs on eks-prod-01")],
    )
    past_exp = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await service.add_events_to_memory(
        app_name="t_sched:sre_bot",
        user_id="u1",
        events=[_make_event("One-time token EXPIRED-999")],
        custom_metadata={"expires_at": past_exp},
    )
    assert len(store.edges) == 0  # Deferred!

    # 2. Start FastAPI app with built-in maintenance worker enabled
    app = create_memory_platform_app(
        service=service,
        registrar=registrar,
        enable_outbox_worker=False,
        enable_maintenance_worker=True,
        maintenance_interval_sec=0.05,
        maintenance_run_immediately=True,
    )
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.15)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test.local") as client:
            health = (await client.get("/health")).json()
            assert health["maintenance_worker_running"] is True
            assert health["maintenance_last_stats"] is not None
        assert len(store.edges) >= 1  # Automatically batch-extracted by background worker!
        assert not any("EXPIRED-999" in ep.content_text for ep in store.episodes)
        assert service.observability.counters["maintenance_cycles_completed"] >= 1

    # 3. Clean shutdown verified
    assert service.maintenance_worker_running is False


@pytest.mark.asyncio
async def test_v022_security_hardening_tenant_smuggling_mcp_rbac_jwt_claims_and_complete_gdpr_purge():
    """
    Security regression test validating every critical/high hardening fix in v0.2.2:
      1. Query-vs-body tenant smuggling blocked (`?app_name=tenant_a:app` + body `tenant_b:app` -> 403).
      2. `/v1/mcp/call` enforces BOTH per-tool RBAC AND tenant/user isolation inside `arguments`.
      3. Actual routes (`/v1/lineage/purge`, `/v1/traces`, `/v1/apps/{tenant}/{app}`) enforce exact RBAC & tenant isolation.
      4. JWTs without `exp` or without `tenant_id` are strictly rejected (`401`).
      5. Complete GDPR `purge_user_lineage` erases user text from Episodic, Graph, Working Memory Blocks, AND Revision History.
      6. `adk-memory serve-api --host 0.0.0.0` refuses to start without auth enabled.
    """
    import base64
    import hashlib
    import hmac
    import json
    import time
    import httpx
    from agentic_memory_sdk import (
        AuthPolicyConfig,
        MemoryRole,
        mint_hs256_jwt,
    )
    from agentic_memory_sdk.cli import main as cli_main

    store = InMemoryUnifiedAdapter()
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=DeterministicHashEmbeddingAdapter(),
        extractor_adapter=RuleBasedMockExtractorAdapter(),
    )
    registrar = PlatformOntologyRegistrar(store)
    await registrar.register_adk_app(AppMemoryManifest(tenant_id="tenant_a", app_name="app"))
    await registrar.register_adk_app(AppMemoryManifest(tenant_id="tenant_b", app_name="app"))

    # Seed secret memory in t1 (default tenant), tenant_b, and user data + working block in tenant_a
    await registrar.register_adk_app(AppMemoryManifest(tenant_id="t1", app_name="app"))
    await service.add_events_to_memory(
        app_name="t1:app",
        user_id="bob",
        events=[_make_event("DEFAULT TENANT t1 SECRET CODE ZETA-000.")],
    )
    await service.add_events_to_memory(
        app_name="tenant_b:app",
        user_id="bob",
        events=[_make_event("TOP SECRET tenant_b launch code ALPHA-999.")],
    )
    await service.add_events_to_memory(
        app_name="tenant_a:app",
        user_id="alice",
        events=[_make_event("Alice private medical note for GDPR purge test.")],
    )
    await service.async_upsert_working_memory_block(
        app_name="tenant_a:app",
        user_id="alice",
        label="private_notes",
        value="Alice sensitive working scratchpad text.",
    )

    # Verify short JWT secrets (<32 bytes) are rejected
    with pytest.raises(ValueError, match="at least 32 bytes"):
        AuthPolicyConfig(enabled=True, jwt_secret="too-short")

    secret = "hardened-test-secret-with-more-than-32-bytes-entropy"
    auth_policy = AuthPolicyConfig(enabled=True, jwt_secret=secret)
    reader_a_jwt = mint_hs256_jwt(
        {"sub": "reader-a", "tenant_id": "tenant_a", "roles": [MemoryRole.READER.value]},
        secret=secret,
    )
    compliance_a_jwt = mint_hs256_jwt(
        {"sub": "comp-a", "tenant_id": "tenant_a", "roles": [MemoryRole.COMPLIANCE_ADMIN.value]},
        secret=secret,
    )
    global_admin_jwt = mint_hs256_jwt(
        {"sub": "global-admin", "tenant_id": "*", "roles": [MemoryRole.PLATFORM_ADMIN.value]},
        secret=secret,
    )

    def _raw_jwt(payload: dict) -> str:
        def _b64(obj: dict) -> str:
            return base64.urlsafe_b64encode(json.dumps(obj, separators=(",", ":")).encode()).rstrip(b"=").decode()
        h = _b64({"alg": "HS256", "typ": "JWT"})
        p = _b64(payload)
        sig = base64.urlsafe_b64encode(
            hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        return f"{h}.{p}.{sig}"

    jwt_no_exp = _raw_jwt({"sub": "bad1", "tenant_id": "tenant_a", "roles": ["reader"]})
    jwt_no_tenant = _raw_jwt({"sub": "bad2", "roles": ["reader"], "exp": int(time.time()) + 3600})

    app = create_memory_platform_app(
        service=service,
        registrar=registrar,
        enable_outbox_worker=False,
        auth_policy=auth_policy,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test.local"
    ) as client:
        # 1. Query-vs-body tenant smuggling MUST be blocked (403)
        smuggle = await client.post(
            "/v1/memory/search?app_name=tenant_a:app",
            json={"app_name": "tenant_b:app", "user_id": "bob", "query": "launch code"},
            headers={"Authorization": f"Bearer {reader_a_jwt}"},
        )
        assert smuggle.status_code == 403

        # 2. /v1/mcp/call cross-tenant read MUST be blocked (403)
        mcp_cross = await client.post(
            "/v1/mcp/call",
            json={
                "tool_name": "recall_context",
                "arguments": {"app_name": "tenant_b:app", "user_id": "bob", "query": "launch code"},
            },
            headers={"Authorization": f"Bearer {reader_a_jwt}"},
        )
        assert mcp_cross.status_code == 403

        # 3. /v1/mcp/call RBAC for destructive tool MUST be blocked for READER (403)
        mcp_purge = await client.post(
            "/v1/mcp/call",
            json={
                "tool_name": "purge_user_lineage",
                "arguments": {"app_name": "tenant_a:app", "user_id": "alice"},
            },
            headers={"Authorization": f"Bearer {reader_a_jwt}"},
        )
        assert mcp_purge.status_code == 403

        # 4. Real route /v1/lineage/purge, /v1/traces, and /v1/apps/{tenant}/{app} RBAC & isolation
        purge_real = await client.post(
            "/v1/lineage/purge",
            json={"app_name": "tenant_b:app", "user_id": "bob"},
            headers={"Authorization": f"Bearer {reader_a_jwt}"},
        )
        assert purge_real.status_code == 403

        trace_denied = await client.post(
            "/v1/traces",
            json={"app_name": "tenant_a:app", "user_id": "alice", "run_id": "r1", "session_id": "s1", "steps": []},
            headers={"Authorization": f"Bearer {reader_a_jwt}"},
        )
        assert trace_denied.status_code == 403

        path_cross = await client.get(
            "/v1/apps/tenant_b/app",
            headers={"Authorization": f"Bearer {reader_a_jwt}"},
        )
        assert path_cross.status_code == 403

        # 5. JWT without `exp` or without `tenant_id` MUST be rejected (401)
        r_no_exp = await client.get(
            "/v1/memory/items?app_name=tenant_a:app&user_id=alice",
            headers={"Authorization": f"Bearer {jwt_no_exp}"},
        )
        assert r_no_exp.status_code == 401

        r_no_tenant = await client.get(
            "/v1/memory/items?app_name=tenant_a:app&user_id=alice",
            headers={"Authorization": f"Bearer {jwt_no_tenant}"},
        )
        assert r_no_tenant.status_code == 401

        # 6. Bare app_name (no tenant prefix) resolves to caller's OWN tenant, NEVER `t1`!
        bare_search = await client.post(
            "/v1/memory/search",
            json={"app_name": "app", "user_id": "bob", "query": "ZETA-000"},
            headers={"Authorization": f"Bearer {reader_a_jwt}"},
        )
        assert bare_search.status_code == 200
        assert "ZETA-000" not in json.dumps(bare_search.json())

        bare_purge = await client.post(
            "/v1/lineage/purge",
            json={"app_name": "app", "user_id": "bob"},
            headers={"Authorization": f"Bearer {compliance_a_jwt}"},
        )
        assert bare_purge.status_code == 200
        # Confirm t1's bob memory is STILL untouched!
        assert any("ZETA-000" in ep.content_text for ep in store.episodes)

        # 7. Unfiltered GET /v1/memory/revisions returns ONLY caller's tenant (`tenant_a`)
        unfiltered_revs = await client.get(
            "/v1/memory/revisions",
            headers={"Authorization": f"Bearer {reader_a_jwt}"},
        )
        assert unfiltered_revs.status_code == 200
        rev_tenants = {r["tenant_id"] for r in unfiltered_revs.json()["revisions"]}
        assert rev_tenants == {"tenant_a"}

        # 8. MCP for READER: tools/list succeeds (200), cross-tenant resource read blocked (403)
        mcp_list_reader = await client.post(
            "/mcp/http/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-11-25",
                "Authorization": f"Bearer {reader_a_jwt}",
            },
        )
        assert mcp_list_reader.status_code == 200

        mcp_res_cross = await client.post(
            "/mcp/http/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {"uri": "memory://apps/t1/app/tbox"},
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-11-25",
                "Authorization": f"Bearer {reader_a_jwt}",
            },
        )
        assert mcp_res_cross.status_code == 403

        # 9. Global admin routes (/v1/maintenance/run, /v1/outbox/flush) reject single-tenant roles (403)
        maint_single = await client.post(
            "/v1/maintenance/run",
            headers={"Authorization": f"Bearer {compliance_a_jwt}"},
        )
        assert maint_single.status_code == 403
        maint_global = await client.post(
            "/v1/maintenance/run",
            headers={"Authorization": f"Bearer {global_admin_jwt}"},
        )
        assert maint_global.status_code == 200

    # 10. API Rate Limiting returns 429 when exceeded & bounds key table size
    from agentic_memory_sdk.security import (
        ApiRateLimiter,
        AuthPrincipal,
        reset_current_principal,
        set_current_principal,
    )

    bounded_rl = ApiRateLimiter(default_rpm=5, max_keys=3)
    for idx in range(10):
        assert bounded_rl.check_and_record(f"client_{idx}") is True
    assert len(bounded_rl._hits) <= 3

    # 11. Service-layer `allowed_users` enforcement (protects /mcp/sse & direct service calls)
    restricted_principal = AuthPrincipal(
        subject="user-scoped-agent",
        tenant_id="tenant_a",
        roles=[MemoryRole.AGENT],
        allowed_users=["alice"],
    )
    tok = set_current_principal(restricted_principal)
    try:
        with pytest.raises(PermissionError, match="User scope violation"):
            await service.search_memory(app_name="tenant_a:app", user_id="bob", query="test")
        with pytest.raises(PermissionError, match="User scope violation"):
            service.upsert_working_memory_block(
                app_name="tenant_a:app", user_id="bob", label="x", value="y"
            )
    finally:
        reset_current_principal(tok)

    rl_app = create_memory_platform_app(
        service=service,
        registrar=registrar,
        enable_outbox_worker=False,
        auth_policy=AuthPolicyConfig(enabled=True, jwt_secret=secret, api_rate_limit_rpm=2),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rl_app), base_url="http://test.local"
    ) as rl_client:
        hdrs = {"Authorization": f"Bearer {reader_a_jwt}"}
        assert (await rl_client.get("/v1/memory/revisions", headers=hdrs)).status_code == 200
        assert (await rl_client.get("/v1/memory/revisions", headers=hdrs)).status_code == 200
        assert (await rl_client.get("/v1/memory/revisions", headers=hdrs)).status_code == 429

    # 12. Bring-Your-Own-Auth (BYOA) `custom_request_authenticator` (e.g., Istio/Kong headers, zero Bearer token)
    async def gateway_header_auth(headers):
        sub = headers.get("x-auth-subject")
        tid = headers.get("x-auth-tenant")
        if sub and tid:
            return AuthPrincipal(subject=sub, tenant_id=tid, roles=[MemoryRole.READER])
        return None

    byoa_app = create_memory_platform_app(
        service=service,
        registrar=registrar,
        enable_outbox_worker=False,
        auth_policy=AuthPolicyConfig(enabled=True, custom_request_authenticator=gateway_header_auth),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=byoa_app), base_url="http://test.local"
    ) as byoa_client:
        # Missing gateway headers -> 401
        assert (await byoa_client.get("/v1/memory/revisions")).status_code == 401
        # Valid gateway headers for tenant_a -> 200 (scoped to tenant_a)
        gw_hdrs = {"X-Auth-Subject": "spiffe://mesh/ns/a", "X-Auth-Tenant": "tenant_a"}
        ok_resp = await byoa_client.get("/v1/memory/revisions", headers=gw_hdrs)
        assert ok_resp.status_code == 200
        assert all(r["tenant_id"] == "tenant_a" for r in ok_resp.json()["revisions"])
        # Cross-tenant request with gateway headers -> 403
        cross_gw = await byoa_client.get(
            "/v1/memory/items?app_name=tenant_b:app&user_id=bob", headers=gw_hdrs
        )
        assert cross_gw.status_code == 403

    # 6. Complete GDPR purge erases working memory blocks AND memory revisions
    assert len(await service.async_get_working_memory_blocks("tenant_a:app", "alice")) == 1
    assert len(await service.async_get_memory_revisions("tenant_a:app", "alice")) >= 2
    await service.purge_user_lineage("tenant_a:app", "alice")
    assert len(await service.async_get_working_memory_blocks("tenant_a:app", "alice")) == 0
    assert len(await service.async_get_memory_revisions("tenant_a:app", "alice")) == 0

    # 7. CLI refuses insecure 0.0.0.0 bind without auth
    rc = cli_main(["serve-api", "--host", "0.0.0.0", "--port", "8095"])
    assert rc == 2











