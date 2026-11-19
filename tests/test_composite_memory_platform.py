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
        assert len(meta_json["tools"]) == 12
        assert "induce_domain_pack" in meta_json["tools"]
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
        assert len(tools_list) == 12
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





