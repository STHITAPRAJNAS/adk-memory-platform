from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

from agentic_composite_memory import (
    AppMemoryManifest,
    CompositeMemoryService,
    DeterministicHashEmbeddingAdapter,
    DomainPackLoader,
    ExtractedEntity,
    ExtractedFact,
    ExtractionPayload,
    GovernanceTier,
    InMemoryUnifiedAdapter,
    MemoryMcpServerTools,
    PlatformOntologyRegistrar,
    RemoteCompositeMemoryService,
    RuleBasedMockExtractorAdapter,
    create_mcp_server,
    create_memory_platform_app,
    get_adk_memory_tools,
)
from agentic_composite_memory.service import Event, types


def _make_event(text: str, author: str = "user") -> Event:
    return Event(
        author=author,
        content=types.Content(role=author, parts=[types.Part.from_text(text=text)]),
    )


async def run_three_consumption_patterns_demo() -> dict[str, Any]:
    """
    End-to-end demonstration of all 3 Agent Consumption Patterns on the same
    Retail Omnichannel (`retail_omnichannel@1.0.0`) platform instance:
      1. Pattern 1 — Direct In-Process SDK (`CompositeMemoryService` + `get_adk_memory_tools`)
      2. Pattern 2 — FastAPI Service + Lightweight Remote Client (`create_memory_platform_app` + `RemoteCompositeMemoryService`)
      3. Pattern 3 — Model Context Protocol (`create_mcp_server`, `MemoryMcpServerTools`, and Co-Hosted `/v1/mcp/call`)
    """
    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()

    extractor.add_keyword_rule(
        "ord-7700 shipped from wh-seattle-01",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="ORD-7700",
                    entity_class="RetailOrder",
                    summary="Omnichannel order ORD-7700",
                ),
                ExtractedEntity(
                    canonical_name="WH-Seattle-01",
                    entity_class="FulfillmentWarehouse",
                    summary="Pacific Northwest warehouse #01 in Seattle",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="ORD-7700",
                    source_class="RetailOrder",
                    predicate="SHIPPED_FROM_PRIMARY",
                    target_entity="WH-Seattle-01",
                    target_class="FulfillmentWarehouse",
                    fact_summary="Order ORD-7700 shipped from WH-Seattle-01",
                )
            ],
        ),
    )

    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        playbook_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)

    # =========================================================================
    # PATTERN 1: Direct In-Process Python SDK Usage
    # =========================================================================
    pack_path = Path(__file__).resolve().parent / "domain_packs" / "retail_omnichannel.json"
    retail_spec = DomainPackLoader.from_json_file(pack_path)
    registrar.register_domain_pack(retail_spec)
    await registrar.onboard_app(
        AppMemoryManifest(
            tenant_id="rei_coop",
            app_name="gear_specialist_agent",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["retail_omnichannel@1.0.0"],
        )
    )

    # 1A. Use ADK FunctionTools bound directly to In-Process `service`
    adk_tools = get_adk_memory_tools(service, default_app_name="rei_coop:gear_specialist_agent")
    assert len(adk_tools) >= 2

    # 1B. Ingest via Direct SDK
    await service.add_events_to_memory(
        app_name="rei_coop:gear_specialist_agent",
        user_id="climber_alex",
        session_id="sdk_sess_1",
        events=[_make_event("Order ORD-7700 shipped from WH-Seattle-01.")],
    )
    sdk_recall = await service.search_memory(
        app_name="rei_coop:gear_specialist_agent",
        user_id="climber_alex",
        query="Where did ORD-7700 ship from?",
    )
    sdk_text = "\n".join(m.content.parts[0].text for m in sdk_recall.memories)

    # =========================================================================
    # PATTERN 2: FastAPI REST Service + `RemoteCompositeMemoryService` Client
    # =========================================================================
    fastapi_app = create_memory_platform_app(service, registrar, mount_mcp=True)
    transport = ASGITransport(app=fastapi_app)

    async with AsyncClient(transport=transport, base_url="http://memory-platform.local") as http_client:
        remote_client = RemoteCompositeMemoryService(
            base_url="http://memory-platform.local",
            http_client=http_client,
        )

        # 2A. Register a Tier-3 Playbook over FastAPI REST (`/v1/playbooks/register`)
        pb_resp = await http_client.post(
            "/v1/playbooks/register",
            json={
                "app_name": "rei_coop:gear_specialist_agent",
                "user_id": "climber_alex",
                "trigger_intent": "ORD-7700 fulfillment warehouse and tracking",
                "workflow_steps": [
                    "lookup_order_ord_7700",
                    "verify_wh_seattle_01_carrier_scan",
                ],
                "resolution_summary": "Standard Seattle warehouse tracking workflow",
            },
        )
        assert pb_resp.status_code == 200

        # 2B. Recall via `RemoteCompositeMemoryService` (used inside lightweight ADK Agent Pods)
        remote_recall = await remote_client.search_memory(
            app_name="rei_coop:gear_specialist_agent",
            user_id="climber_alex",
            query="ORD-7700 fulfillment warehouse and tracking",
        )
        remote_text = "\n".join(m.content.parts[0].text for m in remote_recall.memories)

        # =====================================================================
        # PATTERN 3: Model Context Protocol (MCP Server + Co-Hosted HTTP Bridge)
        # =====================================================================
        # 3A. Standalone FastMCP server object (`/mcp/http`, `/mcp/sse`, or `stdio`)
        mcp_server = create_mcp_server(service, registrar)
        assert mcp_server is not None

        # 3B. Direct `MemoryMcpServerTools` invocation
        mcp_tools = MemoryMcpServerTools(service, registrar)
        mcp_direct = await mcp_tools.recall_context(
            app_name="rei_coop:gear_specialist_agent",
            user_id="climber_alex",
            query="ORD-7700 fulfillment warehouse and tracking",
        )

        # 3C. Co-Hosted Stateless MCP REST Bridge (`GET /v1/mcp/tools` & `POST /v1/mcp/call`)
        tools_list_resp = await http_client.get("/v1/mcp/tools")
        assert tools_list_resp.status_code == 200
        mcp_tool_names = tools_list_resp.json()["tools"]
        for expected_tool in (
            "recall_context",
            "commit_episode",
            "register_playbook",
            "register_domain_pack",
            "evolve_app_ontology",
            "run_memory_maintenance",
            "inspect_platform_governance",
        ):
            assert expected_tool in mcp_tool_names

        gov_inspect = await http_client.post(
            "/v1/mcp/call",
            json={
                "tool_name": "inspect_platform_governance",
                "arguments": {"app_name": "rei_coop:gear_specialist_agent"},
            },
        )
        assert gov_inspect.status_code == 200
        assert "RetailOrder" in gov_inspect.json()["result"]["app_governance"]["classes"]

        mcp_http_call = await http_client.post(
            "/v1/mcp/call",
            json={
                "tool_name": "recall_context",
                "arguments": {
                    "app_name": "rei_coop:gear_specialist_agent",
                    "user_id": "climber_alex",
                    "query": "ORD-7700 fulfillment warehouse and tracking",
                },
            },
        )
        assert mcp_http_call.status_code == 200
        mcp_http_result = mcp_http_call.json()["result"]

    return {
        "pattern_1_sdk_verified": "(ORD-7700) -[SHIPPED_FROM_PRIMARY]-> (WH-Seattle-01)" in sdk_text,
        "pattern_2_fastapi_remote_client_verified": (
            "[MATCHED_PROCEDURAL_PLAYBOOKS]" in remote_text
            and "(ORD-7700) -[SHIPPED_FROM_PRIMARY]-> (WH-Seattle-01)" in remote_text
        ),
        "pattern_3_mcp_verified": (
            len(mcp_direct["memories"]) >= 2
            and len(mcp_http_result["memories"]) >= 2
            and "recall_context" in tools_list_resp.json()["tools"]
        ),
    }


if __name__ == "__main__":
    out = asyncio.run(run_three_consumption_patterns_demo())
    print(json.dumps(out, indent=2))
    assert all(out.values()), f"Failed checks: {out}"
