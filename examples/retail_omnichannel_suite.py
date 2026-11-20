from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

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
    MemoryScope,
    PlatformOntologyRegistrar,
    ReasoningTraceStep,
    RuleBasedMockExtractorAdapter,
)
from agentic_composite_memory.service import Event, types


def _make_event(text: str, author: str = "user") -> Event:
    return Event(
        author=author,
        content=types.Content(role=author, parts=[types.Part.from_text(text=text)]),
    )


def _configure_retail_extractor() -> RuleBasedMockExtractorAdapter:
    """
    Configures deterministic multi-agent Retail Omnichannel extraction rules covering:
      1. Personal Stylist: Customer sizing/fabric preferences (`USER_GLOBAL`) + Loyalty Tier upgrade (`Silver` -> `Ambassador-VIP`).
      2. Order Fulfillment: Multi-hop order -> SKU -> Warehouse topology + sibling SKU separation (`SKU-LINEN-01` vs `SKU-LINEN-02`) + functional warehouse re-routing (`WH-Newark-01` -> `WH-Dallas-02`).
      3. Merchandising Trend Agent: Emergent social trend concept (`ViralTikTokTrend` / `FEATURED_IN_TREND`) quarantined in app overflow and promoted bottom-up.
    """
    ext = RuleBasedMockExtractorAdapter()

    # 1. Shopper preference + initial Silver loyalty tier
    ext.add_keyword_rule(
        "sustainable organic linen in size medium",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="maya_lin",
                    entity_class="RetailCustomer",
                    summary="Omnichannel retail shopper Maya Lin",
                ),
                ExtractedEntity(
                    canonical_name="Organic-Linen-Size-M",
                    entity_class="p:UserPreference",
                    summary="Prefers sustainable organic linen garments in size Medium",
                    scope=MemoryScope.USER_GLOBAL,
                ),
                ExtractedEntity(
                    canonical_name="Silver-Tier",
                    entity_class="LoyaltyTier",
                    summary="Standard Silver loyalty program tier",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="maya_lin",
                    source_class="RetailCustomer",
                    predicate="PREFERS",
                    target_entity="Organic-Linen-Size-M",
                    target_class="p:UserPreference",
                    fact_summary="maya_lin prefers sustainable organic linen garments in size Medium",
                    scope=MemoryScope.USER_GLOBAL,
                    valid_from="2026-02-01T00:00:00Z",
                ),
                ExtractedFact(
                    source_entity="maya_lin",
                    source_class="RetailCustomer",
                    predicate="ENROLLED_IN_TIER",
                    target_entity="Silver-Tier",
                    target_class="LoyaltyTier",
                    fact_summary="maya_lin is enrolled in Silver-Tier loyalty program",
                    valid_from="2026-02-01T00:00:00Z",
                ),
            ],
        ),
    )

    # 2. Loyalty Tier Upgrade (`Silver-Tier` -> `Ambassador-VIP`)
    ext.add_keyword_rule(
        "upgraded to ambassador-vip",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="maya_lin",
                    entity_class="RetailCustomer",
                    summary="Omnichannel retail shopper Maya Lin",
                ),
                ExtractedEntity(
                    canonical_name="Ambassador-VIP",
                    entity_class="LoyaltyTier",
                    summary="Top-tier Ambassador VIP loyalty status with free overnight shipping",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="maya_lin",
                    source_class="RetailCustomer",
                    predicate="ENROLLED_IN_TIER",
                    target_entity="Ambassador-VIP",
                    target_class="LoyaltyTier",
                    fact_summary="maya_lin upgraded to Ambassador-VIP loyalty status",
                    valid_from="2026-10-15T00:00:00Z",
                )
            ],
        ),
    )

    # 3. Order Fulfillment Topology + Sibling SKUs (`SKU-LINEN-01` vs `SKU-LINEN-02`)
    ext.add_keyword_rule(
        "ord-9001 contains sku-linen-01",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="ORD-9001",
                    entity_class="RetailOrder",
                    summary="Maya Lin online order ORD-9001",
                ),
                ExtractedEntity(
                    canonical_name="SKU-LINEN-01",
                    entity_class="ProductSKU",
                    summary=" Riviera Organic Linen Blazer (Ivory, Size M)",
                ),
                ExtractedEntity(
                    canonical_name="SKU-LINEN-02",
                    entity_class="ProductSKU",
                    summary="Riviera Organic Linen Blazer (Navy, Size L)",
                ),
                ExtractedEntity(
                    canonical_name="WH-Newark-01",
                    entity_class="FulfillmentWarehouse",
                    summary="Northeast regional fulfillment center #01 in Newark",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="ORD-9001",
                    source_class="RetailOrder",
                    predicate="CONTAINS_SKU",
                    target_entity="SKU-LINEN-01",
                    target_class="ProductSKU",
                    fact_summary="Order ORD-9001 contains item SKU-LINEN-01",
                    valid_from="2026-11-10T09:00:00Z",
                ),
                ExtractedFact(
                    source_entity="SKU-LINEN-01",
                    source_class="ProductSKU",
                    predicate="STOCKED_AT",
                    target_entity="WH-Newark-01",
                    target_class="FulfillmentWarehouse",
                    fact_summary="SKU-LINEN-01 is stocked at WH-Newark-01",
                    valid_from="2026-11-10T09:00:00Z",
                ),
                ExtractedFact(
                    source_entity="ORD-9001",
                    source_class="RetailOrder",
                    predicate="SHIPPED_FROM_PRIMARY",
                    target_entity="WH-Newark-01",
                    target_class="FulfillmentWarehouse",
                    fact_summary="Order ORD-9001 assigned to ship from WH-Newark-01",
                    valid_from="2026-11-10T09:00:00Z",
                ),
            ],
        ),
    )

    # 4. Functional Warehouse Re-Routing (`WH-Newark-01` -> `WH-Dallas-02`)
    ext.add_keyword_rule(
        "re-routed ord-9001 to ship from wh-dallas-02",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="ORD-9001",
                    entity_class="RetailOrder",
                    summary="Maya Lin online order ORD-9001",
                ),
                ExtractedEntity(
                    canonical_name="WH-Dallas-02",
                    entity_class="FulfillmentWarehouse",
                    summary="South-Central express fulfillment hub #02 in Dallas",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="ORD-9001",
                    source_class="RetailOrder",
                    predicate="SHIPPED_FROM_PRIMARY",
                    target_entity="WH-Dallas-02",
                    target_class="FulfillmentWarehouse",
                    fact_summary="Order ORD-9001 re-routed to ship from WH-Dallas-02 for overnight VIP delivery",
                    valid_from="2026-11-12T15:00:00Z",
                )
            ],
        ),
    )

    # 5. Emergent Social Trend in Merchandising Agent (`ViralTikTokTrend` not in base TBox)
    ext.add_keyword_rule(
        "coastal-grandmother-aesthetic",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="SKU-LINEN-01",
                    entity_class="ProductSKU",
                    summary="Riviera Organic Linen Blazer (Ivory, Size M)",
                ),
                ExtractedEntity(
                    canonical_name="Coastal-Grandmother-Aesthetic",
                    entity_class="ViralTikTokTrend",
                    summary="Viral social commerce style trend driving linen demand",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="SKU-LINEN-01",
                    source_class="ProductSKU",
                    predicate="FEATURED_IN_TREND",
                    target_entity="Coastal-Grandmother-Aesthetic",
                    target_class="ViralTikTokTrend",
                    fact_summary="SKU-LINEN-01 is featured in viral trend Coastal-Grandmother-Aesthetic",
                    confidence=0.93,
                )
            ],
        ),
    )

    return ext


async def run_retail_omnichannel_suite() -> dict[str, Any]:
    """
    Runs the 4-Agent Omnichannel Retail & Supply-Chain Simulation demonstrating
    that `agentic-composite-memory` is 100% domain-agnostic.
    """
    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = _configure_retail_extractor()

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

    # 1. Load & Register the Retail Omnichannel Domain Pack from JSON
    pack_path = Path(__file__).resolve().parent / "domain_packs" / "retail_omnichannel.json"
    retail_spec = DomainPackLoader.from_json_file(pack_path)
    registrar.register_domain_pack(retail_spec)

    # 2. Onboard 4 Retail Agents under Tenant `aurora_retail`
    for app_name in (
        "personal_stylist_agent",
        "order_fulfillment_agent",
        "returns_concierge_agent",
        "merchandising_trend_agent",
    ):
        await registrar.onboard_app(
            AppMemoryManifest(
                tenant_id="aurora_retail",
                app_name=app_name,
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["retail_omnichannel@1.0.0"],
            )
        )

    # 3. Agent 1 (`personal_stylist_agent`): Record shopper preference + initial Silver tier, then upgrade to Ambassador-VIP
    await service.add_events_to_memory(
        app_name="aurora_retail:personal_stylist_agent",
        user_id="maya_lin",
        session_id="stylist_sess_1",
        events=[
            _make_event(
                "Shopper maya_lin loves sustainable organic linen in size medium and is enrolled in Silver-Tier."
            )
        ],
    )
    await service.add_events_to_memory(
        app_name="aurora_retail:personal_stylist_agent",
        user_id="maya_lin",
        session_id="stylist_sess_2",
        events=[
            _make_event("Congratulations! maya_lin has been upgraded to Ambassador-VIP tier.")
        ],
    )

    # 4. Agent 2 (`order_fulfillment_agent`): Ingest order with PAN PII, sibling SKUs, and warehouse re-routing
    await service.add_events_to_memory(
        app_name="aurora_retail:order_fulfillment_agent",
        user_id="maya_lin",
        session_id="order_sess_1",
        events=[
            _make_event(
                "Checkout (Card 4532-0151-1283-0099): ORD-9001 contains SKU-LINEN-01 ( stocked at WH-Newark-01 ), "
                "while sibling SKU-LINEN-02 is navy."
            )
        ],
    )
    await service.add_events_to_memory(
        app_name="aurora_retail:order_fulfillment_agent",
        user_id="maya_lin",
        session_id="order_sess_2",
        events=[
            _make_event(
                "Weather delay at Newark: re-routed ORD-9001 to ship from WH-Dallas-02."
            )
        ],
    )

    # 5. Agent 3 (`returns_concierge_agent`): Register a VIP Instant Exchange Playbook & Record Reasoning Trace
    pb = await service.register_playbook(
        app_name="aurora_retail:returns_concierge_agent",
        user_id="maya_lin",
        trigger_intent="ORD-9001 VIP exchange and fulfillment warehouse status",
        workflow_steps=[
            "verify_ambassador_vip_tier",
            "dispatch_instant_replacement_from_wh_dallas_02",
            "issue_prepaid_return_label_no_restock_fee",
        ],
        resolution_summary="Instant no-fee VIP exchange workflow for Ambassador-VIP shoppers",
    )
    await service.record_decision_trace(
        app_name="aurora_retail:returns_concierge_agent",
        user_id="maya_lin",
        run_id="retail_run_501",
        session_id="returns_sess_1",
        steps=[
            ReasoningTraceStep(
                step_index=1,
                tool_name="check_warehouse_dispatch",
                tool_input_summary="order_id=ORD-9001",
                tool_output_summary="shipped_from=WH-Dallas-02",
                touched_vertex_ids=[],
            )
        ],
    )

    # 6. Agent 4 (`merchandising_trend_agent`): Ingest emergent TikTok trend -> Promote via Bottom-Up Ontology Evolution
    await service.add_events_to_memory(
        app_name="aurora_retail:merchandising_trend_agent",
        user_id="merch_planner",
        session_id="trend_sess_1",
        events=[
            _make_event(
                "Merchandising alert: SKU-LINEN-01 is spiking due to Coastal-Grandmother-Aesthetic."
            )
        ],
    )
    evolution_audit = await service.evolve_app_ontology_from_overflow(
        "aurora_retail:merchandising_trend_agent",
        min_confidence_to_promote=0.85,
        registrar=registrar,
    )

    # 7. Cross-Agent Recall in `returns_concierge_agent` (combining Playbook + 2-Hop Graph + Shopper Preferences)
    concierge_recall = await service.search_memory(
        app_name="aurora_retail:returns_concierge_agent",
        user_id="maya_lin",
        query="ORD-9001 VIP exchange and fulfillment warehouse status",
        max_hops=2,
    )
    concierge_context = "\n".join(
        m.content.parts[0].text for m in concierge_recall.memories
    )

    # Point-in-time historical query: Where was ORD-9001 scheduled to ship from on 2026-11-11 (before Dallas re-route)?
    historical_recall = await service.search_memory(
        app_name="aurora_retail:order_fulfillment_agent",
        user_id="maya_lin",
        query="ORD-9001 fulfillment warehouse",
        as_of_iso="2026-11-11T00:00:00Z",
    )
    historical_context = "\n".join(
        m.content.parts[0].text for m in historical_recall.memories
    )

    # 8. Steward Certification + Pre-flight Answerability / Coverage Check (`check_answerable`)
    certified_pack, _ = registrar.certify_domain_pack(
        "retail_omnichannel@1.0.0", certified_by="retail_chief_architect"
    )
    ans_ok = await service.check_answerable(
        app_name="aurora_retail:returns_concierge_agent",
        user_id="maya_lin",
        query="Where is order ORD-9001 shipped from?",
        registrar=registrar,
    )
    ans_gap = await service.check_answerable(
        app_name="aurora_retail:returns_concierge_agent",
        user_id="maya_lin",
        query="What was the return on ad spend for order ORD-9001?",
        registrar=registrar,
    )

    summary = {
        "pack_id": retail_spec.pack_id,
        "domain_name": retail_spec.domain_name,
        "checks": {
            "loyalty_tier_upgraded_to_ambassador_vip": (
                "(maya_lin) -[ENROLLED_IN_TIER]-> (Ambassador-VIP)" in concierge_context
                and "(maya_lin) -[ENROLLED_IN_TIER]-> (Silver-Tier)" not in concierge_context
            ),
            "warehouse_rerouted_to_dallas_02": (
                "(ORD-9001) -[SHIPPED_FROM_PRIMARY]-> (WH-Dallas-02)" in concierge_context
                and "(ORD-9001) -[SHIPPED_FROM_PRIMARY]-> (WH-Newark-01)" not in concierge_context
            ),
            "point_in_time_historical_warehouse_newark_01": (
                "(ORD-9001) -[SHIPPED_FROM_PRIMARY]-> (WH-Newark-01)" in historical_context
            ),
            "multihop_order_to_sku_to_warehouse": (
                "(ORD-9001) -[CONTAINS_SKU]-> (SKU-LINEN-01)" in concierge_context
                and "(SKU-LINEN-01) -[STOCKED_AT]-> (WH-Newark-01)" in concierge_context
            ),
            "cross_agent_shopper_preference_recalled": (
                "Organic-Linen-Size-M" in concierge_context
            ),
            "credit_card_pan_redacted": "4532-0151-1283-0099" not in concierge_context,
            "vip_exchange_playbook_auto_injected": (
                "[MATCHED_PROCEDURAL_PLAYBOOKS]" in concierge_context
                and "dispatch_instant_replacement_from_wh_dallas_02" in concierge_context
            ),
            "emergent_tiktok_trend_promoted": (
                "ViralTikTokTrend" in evolution_audit.added_or_modified_classes
                and "FEATURED_IN_TREND" in evolution_audit.added_or_modified_relations
            ),
            "domain_pack_steward_certified": certified_pack.certified is True,
            "check_answerable_detects_covered_vs_ad_spend_gap": (
                ans_ok["answerable"] is True
                and ans_gap["answerable"] is False
                and ans_gap["status"] == "PARTIAL"
                and "return on ad spend" in ans_gap["unmatched_concepts"]
            ),
        },
        "playbook_id": pb.playbook_id,
        "concierge_context_window": concierge_context,
    }
    return summary


if __name__ == "__main__":
    result = asyncio.run(run_retail_omnichannel_suite())
    print(json.dumps(result, indent=2))
    assert all(result["checks"].values()), f"Failed checks: {result['checks']}"
