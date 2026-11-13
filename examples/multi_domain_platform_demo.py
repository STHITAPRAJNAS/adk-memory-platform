"""
End-to-End Multi-Domain Platform Demonstration.
Shows how a single domain-agnostic `CompositeMemoryService` hosts arbitrary ADK agent apps:
  1. Tier 0 Zero-Config Agent (`exec_notes_bot`) -> No ontology config required
  2. Tier 1 App-Scoped Typed Agent (`release_tracker_bot`) -> Uses inline Pydantic models
  3. Tier 2 Shared Domain Pack Agents (`sre_triage_bot` & `sre_postmortem_bot`) -> Share `sre_infra@v1.2`
  4. Tier 2 Shared CRM Domain Agent (`billing_support_bot`) -> Uses `customer_crm@v2.0`
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from agentic_memory_sdk import (
    AppMemoryManifest,
    CompositeMemoryService,
    DeterministicHashEmbeddingAdapter,
    DomainPackLoader,
    ExtractedEntity,
    ExtractedFact,
    ExtractionPayload,
    GovernanceTier,
    InMemoryUnifiedAdapter,
    PlatformOntologyRegistrar,
    RelationRule,
    RuleBasedMockExtractorAdapter,
    UpperConceptType,
)
from agentic_memory_sdk.service import Event, types


class PullRequest(BaseModel):
    pr_number: int


class Developer(BaseModel):
    handle: str


async def run_demo() -> None:
    # 1. Bootstrap Unified Platform Memory Service (Swap InMemoryUnifiedAdapter for Postgres + Neptune in prod)
    store = InMemoryUnifiedAdapter()
    embedder = DeterministicHashEmbeddingAdapter(dim=64)
    extractor = RuleBasedMockExtractorAdapter()

    memory_service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(store)

    # 2. Load Plug-and-Play Domain Packs from JSON & OWL Turtle files
    packs_dir = Path(__file__).parent / "domain_packs"
    sre_pack = DomainPackLoader.from_json_file(packs_dir / "sre_incident_ops.json")
    crm_pack = DomainPackLoader.from_json_file(packs_dir / "customer_crm.json")
    registrar.register_domain_pack(sre_pack)
    registrar.register_domain_pack(crm_pack)

    # 3. Register Diverse ADK Agent Apps across Governance Tiers
    # App 1: Zero-Config General Assistant (Tier 0)
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="acme",
            app_name="exec_notes_bot",
            governance_tier=GovernanceTier.TIER0_AUTO,
        )
    )

    # App 2: App-Scoped Typed Bot (Tier 1 - compiled from Python classes)
    pr_pack = DomainPackLoader.from_pydantic_models(
        pack_id="dev_workflow@v1.0",
        entity_models=[
            (PullRequest, UpperConceptType.OBJECT),
            (Developer, UpperConceptType.ACTOR),
        ],
        relations=[
            RelationRule(
                predicate="ASSIGNED_REVIEWER",
                domain_class="PullRequest",
                range_class="Developer",
                is_functional=True,
            )
        ],
    )
    registrar.register_domain_pack(pr_pack)
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="acme",
            app_name="release_tracker_bot",
            governance_tier=GovernanceTier.TIER1_APP_TYPED,
            subscribed_domain_packs=["dev_workflow@v1.0"],
        )
    )

    # Apps 3 & 4: Two SRE Agents sharing `sre_infra@v1.2` (Tier 2 Domain-Shared)
    for app_name in ("sre_triage_bot", "sre_postmortem_bot"):
        await registrar.register_adk_app(
            AppMemoryManifest(
                tenant_id="acme",
                app_name=app_name,
                governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                subscribed_domain_packs=["sre_infra@v1.2"],
                allow_emergent_overflow=True,
            )
        )

    # App 5: Customer CRM Agent in a separate business domain (`customer_crm@v2.0`)
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="acme",
            app_name="billing_support_bot",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["customer_crm@v2.0"],
        )
    )

    # 4. Simulate Turn 1 in `sre_triage_bot`: Checkout-API deployed in us-east-1
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="Checkout-API",
                    entity_class="Microservice",
                    summary="Core checkout microservice",
                ),
                ExtractedEntity(
                    canonical_name="us-east-1",
                    entity_class="CloudRegion",
                    summary="AWS US East N. Virginia region",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Checkout-API",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-east-1",
                    target_class="CloudRegion",
                    fact_summary="Checkout-API primary region is us-east-1",
                    valid_from="2026-01-10T00:00:00Z",
                )
            ],
        )
    )
    await memory_service.add_events_to_memory(
        app_name="acme:sre_triage_bot",
        user_id="sre_oncall_1",
        events=[
            Event(
                author="user",
                content=types.Content(
                    parts=[types.Part.from_text(text="Checkout-API is running in us-east-1.")]
                ),
            )
        ],
    )

    # 5. Simulate Turn 2 in `sre_postmortem_bot`: Failover Checkout-API to us-west-2
    extractor.enqueue(
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="Checkout-API",
                    entity_class="Microservice",
                    summary="Core checkout microservice failed over to west",
                ),
                ExtractedEntity(
                    canonical_name="us-west-2",
                    entity_class="CloudRegion",
                    summary="AWS US West Oregon region",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Checkout-API",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-west-2",
                    target_class="CloudRegion",
                    fact_summary="Checkout-API failed over to primary region us-west-2",
                    valid_from="2026-11-01T00:00:00Z",
                )
            ],
        )
    )
    await memory_service.add_events_to_memory(
        app_name="acme:sre_postmortem_bot",
        user_id="sre_oncall_2",
        events=[
            Event(
                author="user",
                content=types.Content(
                    parts=[types.Part.from_text(text="We failed Checkout-API over to us-west-2.")]
                ),
            )
        ],
    )

    # 6. Query current active state from `sre_triage_bot` (Bi-temporal invalidation hides us-east-1)
    active_res = await memory_service.search_memory(
        app_name="acme:sre_triage_bot",
        user_id="sre_oncall_1",
        query="Checkout-API region",
    )
    print("=== ACTIVE STATE IN SRE DOMAIN ===")
    print(active_res.memories[0].content.parts[0].text)

    # 7. Verify `billing_support_bot` (CRM domain) cannot read `domain:acme:sre_infra`
    crm_res = await memory_service.search_memory(
        app_name="acme:billing_support_bot",
        user_id="support_rep_1",
        query="Checkout-API region",
    )
    print("=== ISOLATED CRM AGENT SEARCH FOR SRE ENTITY ===")
    print("Memories returned:", len(crm_res.memories))


if __name__ == "__main__":
    asyncio.run(run_demo())
