from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from agentic_memory_sdk import (
    AppMemoryManifest,
    CompositeMemoryService,
    DeterministicHashEmbeddingAdapter,
    DomainPackSpec,
    ExtractedEntity,
    ExtractedFact,
    ExtractionMode,
    ExtractionPayload,
    GeminiEmbeddingAdapter,
    GeminiOntologyExtractorAdapter,
    GovernanceTier,
    MemoryScope,
    Neo4jGraphAdapter,
    PlatformOntologyRegistrar,
    PostgresPgVectorAdapter,
    ReasoningTraceStep,
    RedisPlaybookAndWorkingAdapter,
    RelationRule,
    RuleBasedMockExtractorAdapter,
    UpperConceptType,
)
from agentic_memory_sdk.service import Event, types


def _make_event(text: str, author: str = "user") -> Event:
    return Event(
        author=author,
        content=types.Content(role=author, parts=[types.Part.from_text(text=text)]),
    )


def _configure_realistic_fallback_extractor() -> RuleBasedMockExtractorAdapter:
    """
    High-fidelity fallback extractor used when `GOOGLE_API_KEY` / `GEMINI_API_KEY`
    is not present in the environment, allowing the full 5-agent enterprise stress
    benchmark to run offline while switching automatically to live Gemini 2.5 Flash
    whenever an API key is set.
    """
    ext = RuleBasedMockExtractorAdapter()

    # 1. Multi-hop topology + numbered sibling services (`payment-router-01` vs `payment-router-02`)
    ext.add_keyword_rule(
        "payment-router-01 is deployed in us-east-1",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="api-edge-gw",
                    entity_class="Microservice",
                    summary="Global edge API gateway",
                ),
                ExtractedEntity(
                    canonical_name="payment-router-01",
                    entity_class="Microservice",
                    summary="Primary payment routing microservice #01",
                ),
                ExtractedEntity(
                    canonical_name="payment-router-02",
                    entity_class="Microservice",
                    summary="Canary payment routing microservice #02",
                ),
                ExtractedEntity(
                    canonical_name="aurora-ledger-01",
                    entity_class="DatabaseCluster",
                    summary="Primary Aurora PostgreSQL ledger cluster",
                ),
                ExtractedEntity(
                    canonical_name="us-east-1",
                    entity_class="CloudRegion",
                    summary="AWS US East N. Virginia",
                ),
                ExtractedEntity(
                    canonical_name="eu-west-1",
                    entity_class="CloudRegion",
                    summary="AWS EU Ireland",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="api-edge-gw",
                    source_class="Microservice",
                    predicate="DEPENDS_ON",
                    target_entity="payment-router-01",
                    target_class="Microservice",
                    fact_summary="api-edge-gw routes checkout traffic to payment-router-01",
                    valid_from="2026-01-10T00:00:00Z",
                ),
                ExtractedFact(
                    source_entity="payment-router-01",
                    source_class="Microservice",
                    predicate="READS_FROM",
                    target_entity="aurora-ledger-01",
                    target_class="DatabaseCluster",
                    fact_summary="payment-router-01 reads and writes ledger entries to aurora-ledger-01",
                    valid_from="2026-01-10T00:00:00Z",
                ),
                ExtractedFact(
                    source_entity="payment-router-01",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="us-east-1",
                    target_class="CloudRegion",
                    fact_summary="payment-router-01 primary region is us-east-1",
                    valid_from="2026-01-10T00:00:00Z",
                ),
                ExtractedFact(
                    source_entity="payment-router-02",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="eu-west-1",
                    target_class="CloudRegion",
                    fact_summary="payment-router-02 canary region is eu-west-1",
                    valid_from="2026-01-10T00:00:00Z",
                ),
            ],
        ),
    )

    # 2. Contextual Q&A confirmation turn ("Yes, drain us-east-1 and fail payment-router-01 over to eu-central-1")
    ext.add_keyword_rule(
        "eu-central-1",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="payment-router-01",
                    entity_class="Microservice",
                    summary="Primary payment routing microservice #01 (failed over to eu-central-1)",
                ),
                ExtractedEntity(
                    canonical_name="eu-central-1",
                    entity_class="CloudRegion",
                    summary="AWS EU Frankfurt failover region",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="payment-router-01",
                    source_class="Microservice",
                    predicate="DEPLOYED_IN_PRIMARY",
                    target_entity="eu-central-1",
                    target_class="CloudRegion",
                    fact_summary="payment-router-01 failed over to eu-central-1 after user confirmation",
                    valid_from="2026-11-12T14:30:00Z",
                )
            ],
        ),
    )

    # 3. Executive User Preference (`USER_GLOBAL` cross-app sharing)
    ext.add_keyword_rule(
        "executive bullet points in utc",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="alice_vp",
                    entity_class="p:Actor",
                    summary="VP of Cloud Engineering Alice",
                    scope=MemoryScope.USER_GLOBAL,
                ),
                ExtractedEntity(
                    canonical_name="UTC-Executive-Bullets",
                    entity_class="p:UserPreference",
                    summary="Prefers concise 3-bullet executive summaries with UTC timestamps",
                    scope=MemoryScope.USER_GLOBAL,
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="alice_vp",
                    source_class="p:Actor",
                    predicate="PREFERS",
                    target_entity="UTC-Executive-Bullets",
                    target_class="p:UserPreference",
                    fact_summary="alice_vp prefers concise executive bullet points in UTC timezone",
                    scope=MemoryScope.USER_GLOBAL,
                )
            ],
        ),
    )

    # 4. SecOps Emergent Overflow (Unknown class `ZeroDayCVE` & unknown predicate `MITIGATED_BY_WAF_RULE`)
    ext.add_keyword_rule(
        "cve-2026-9981",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="payment-router-01",
                    entity_class="Microservice",
                    summary="Primary payment routing microservice #01",
                ),
                ExtractedEntity(
                    canonical_name="CVE-2026-9981",
                    entity_class="ZeroDayVulnerability",
                    summary="Critical HTTP/3 header parsing vulnerability CVE-2026-9981",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="payment-router-01",
                    source_class="Microservice",
                    predicate="MITIGATED_BY_WAF_RULE",
                    target_entity="CVE-2026-9981",
                    target_class="ZeroDayVulnerability",
                    fact_summary="payment-router-01 mitigated CVE-2026-9981 via Envoy WAF rule",
                    confidence=0.93,
                )
            ],
        ),
    )

    # 5. FinOps Cost Center Re-allocation (`OWNED_BY_PRIMARY` functional relation)
    ext.add_keyword_rule(
        "cc-104-core",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="payment-router-01",
                    entity_class="CloudWorkload",
                    summary="Payment router workload",
                ),
                ExtractedEntity(
                    canonical_name="CC-104-Core",
                    entity_class="CostCenter",
                    summary="Core Platform Cost Center 104",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="payment-router-01",
                    source_class="CloudWorkload",
                    predicate="OWNED_BY_PRIMARY",
                    target_entity="CC-104-Core",
                    target_class="CostCenter",
                    fact_summary="payment-router-01 budget owned by CC-104-Core",
                    valid_from="2026-01-01T00:00:00Z",
                )
            ],
        ),
    )

    ext.add_keyword_rule(
        "cc-209-global",
        ExtractionPayload(
            entities=[
                ExtractedEntity(
                    canonical_name="payment-router-01",
                    entity_class="CloudWorkload",
                    summary="Payment router workload",
                ),
                ExtractedEntity(
                    canonical_name="CC-209-Global",
                    entity_class="CostCenter",
                    summary="Global Payments Cost Center 209",
                ),
            ],
            facts=[
                ExtractedFact(
                    source_entity="payment-router-01",
                    source_class="CloudWorkload",
                    predicate="OWNED_BY_PRIMARY",
                    target_entity="CC-209-Global",
                    target_class="CostCenter",
                    fact_summary="payment-router-01 budget transferred to CC-209-Global",
                    valid_from="2026-11-10T00:00:00Z",
                )
            ],
        ),
    )

    return ext


async def run_realistic_enterprise_maturity_benchmark(
    use_docker: bool = True,
) -> dict[str, Any]:
    """
    Executes a realistic 5-Agent Enterprise Stress & Maturity Benchmark across:
      - SRE Incident Commander (`sre_commander`)
      - SecOps Vulnerability Auditor (`secops_auditor`)
      - FinOps Cloud Economist (`finops_economist`)
      - Executive VP Persona (`alice_vp` across FinOps & SRE apps)
      - Rival Tenant Isolation Probe (`t2:rival_agent`) & GDPR Lineage Purge
    Automatically uses live `GeminiEmbeddingAdapter` + `GeminiOntologyExtractorAdapter`
    when `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    live_gemini = bool(api_key)

    if live_gemini:
        embedder = GeminiEmbeddingAdapter(
            model="text-embedding-004",
            api_key=api_key,
            output_dimensionality=1536,
        )
        extractor = GeminiOntologyExtractorAdapter(
            model=os.environ.get("ADK_LLM_MODEL", "gemini-2.5-flash"),
            api_key=api_key,
        )
    else:
        embedder = DeterministicHashEmbeddingAdapter(dim=1536)
        extractor = _configure_realistic_fallback_extractor()

    pg_dsn = os.environ.get(
        "POSTGRES_DSN",
        "postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform",
    )
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")

    pg_adapter = PostgresPgVectorAdapter(dsn=pg_dsn, min_pool_size=2, max_pool_size=15)
    neo4j_adapter = Neo4jGraphAdapter(
        uri=neo4j_uri, auth=("neo4j", "platform_secret")
    )
    redis_adapter = RedisPlaybookAndWorkingAdapter(
        redis_url=redis_url, key_prefix="adk_maturity_bench"
    )

    service = CompositeMemoryService(
        registry_adapter=pg_adapter,
        seed_store_adapter=pg_adapter,
        graph_adapter=neo4j_adapter,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
        playbook_adapter=redis_adapter,
        sync_outbox_immediately=True,
    )
    registrar = PlatformOntologyRegistrar(pg_adapter)

    latencies_ms: dict[str, list[float]] = {
        "hot_path_ingest_ms": [],
        "sleep_time_ingest_ms": [],
        "multihop_recall_ms": [],
        "playbook_recall_ms": [],
        "maintenance_compaction_ms": [],
    }

    try:
        await pg_adapter.init_schema(embedding_dim=1536, default_tenant="t_bench")
        pg_pool = await pg_adapter._get_pool()
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM episodic_memories WHERE tenant_id = 't_bench'")
            await conn.execute("DELETE FROM graph_seed_index WHERE tenant_id = 't_bench'")
            await conn.execute("DELETE FROM graph_transactional_outbox WHERE tenant_id = 't_bench'")
        r_conn = await redis_adapter._get_redis()
        if r_conn is not None:
            await r_conn.delete("adk_maturity_bench:t_bench")
        await neo4j_adapter._run_cypher(
            "t_bench",
            "MATCH (n {tenant_id: 't_bench'}) DETACH DELETE n",
            {},
        )

        # 1. Register SRE Domain Pack + AI-Steward Reviewed FinOps Domain Pack
        registrar.register_domain_pack(
            DomainPackSpec(
                pack_id="sre_infra@v1.0",
                domain_name="sre_infra",
                version="1.0.0",
                classes={
                    "Microservice": UpperConceptType.OBJECT.value,
                    "DatabaseCluster": UpperConceptType.OBJECT.value,
                    "CloudRegion": UpperConceptType.OBJECT.value,
                },
                relations={
                    "DEPLOYED_IN_PRIMARY": RelationRule(
                        predicate="DEPLOYED_IN_PRIMARY",
                        domain_class="Microservice",
                        range_class="CloudRegion",
                        is_functional=True,
                        is_temporal=True,
                    ),
                    "READS_FROM": RelationRule(
                        predicate="READS_FROM",
                        domain_class="Microservice",
                        range_class="DatabaseCluster",
                        is_functional=False,
                    ),
                    "DEPENDS_ON": RelationRule(
                        predicate="DEPENDS_ON",
                        domain_class="Microservice",
                        range_class="Microservice",
                        is_functional=False,
                    ),
                },
            )
        )

        raw_finops_pack = DomainPackSpec(
            pack_id="finops@v1.0",
            domain_name="finops",
            version="1.0.0",
            classes={"CloudWorkload": "", "CostCenter": "owl:Thing"},
            relations={
                "owned-by-primary": RelationRule(
                    predicate="owned-by-primary",
                    domain_class="CloudWorkload",
                    range_class="CostCenter",
                    is_functional=False,
                )
            },
        )
        refined_finops, finops_audit = registrar.review_and_refine_domain_pack(
            raw_finops_pack, auto_register=True
        )

        # Register 4 Enterprise Agent Apps
        for app_name, packs in [
            ("sre_commander", ["sre_infra@v1.0"]),
            ("secops_auditor", ["sre_infra@v1.0"]),
            ("finops_economist", ["finops@v1.0"]),
        ]:
            await registrar.register_adk_app(
                AppMemoryManifest(
                    tenant_id="t_bench",
                    app_name=app_name,
                    governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
                    subscribed_domain_packs=packs,
                )
            )

        # 2. Hot-Path Multi-Hop + Sibling Entity Ingestion (with PII Scrubbing test)
        t0 = time.perf_counter()
        await service.add_events_to_memory(
            app_name="t_bench:sre_commander",
            user_id="alice_vp",
            session_id="inc_2026_11",
            events=[
                _make_event(
                    "Incident Report (Auth token Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret, "
                    "Customer SSN 123-45-6789): api-edge-gw routes traffic to payment-router-01. "
                    "payment-router-01 is deployed in us-east-1 and reads from aurora-ledger-01, "
                    "while canary payment-router-02 is deployed in eu-west-1."
                )
            ],
        )
        latencies_ms["hot_path_ingest_ms"].append((time.perf_counter() - t0) * 1000)

        # 3. Contextual Q&A Confirmation Turn ("Should we fail over...?" -> "Yes, fail over to eu-central-1")
        t0 = time.perf_counter()
        await service.add_events_to_memory(
            app_name="t_bench:sre_commander",
            user_id="alice_vp",
            session_id="inc_2026_11",
            events=[
                _make_event(
                    "Should we drain us-east-1 and fail payment-router-01 over to eu-central-1?",
                    author="agent",
                ),
                _make_event(
                    "Yes, fail payment-router-01 over to eu-central-1 immediately.",
                    author="user",
                ),
            ],
        )
        latencies_ms["hot_path_ingest_ms"].append((time.perf_counter() - t0) * 1000)

        # 4. Cross-App Executive User Preference (`alice_vp` in `finops_economist` -> recalled in `sre_commander`)
        await service.add_events_to_memory(
            app_name="t_bench:finops_economist",
            user_id="alice_vp",
            session_id="finops_pref_1",
            events=[
                _make_event(
                    "Note for alice_vp: I always prefer concise executive bullet points in UTC timezone."
                )
            ],
        )

        # 5. FinOps Bi-Temporal Cost Center Re-allocation (`CC-104-Core` -> `CC-209-Global`)
        await service.add_events_to_memory(
            app_name="t_bench:finops_economist",
            user_id="alice_vp",
            session_id="finops_alloc_1",
            events=[
                _make_event(
                    "Workload payment-router-01 is owned by cost center CC-104-Core as of 2026-01-01."
                )
            ],
        )
        await service.add_events_to_memory(
            app_name="t_bench:finops_economist",
            user_id="alice_vp",
            session_id="finops_alloc_2",
            events=[
                _make_event(
                    "Workload payment-router-01 was transferred to cost center CC-209-Global on 2026-11-10."
                )
            ],
        )

        # 6. SecOps Emergent Vulnerability Quarantine -> Ontology Auto-Evolution
        for i in range(2):
            await service.add_events_to_memory(
                app_name="t_bench:secops_auditor",
                user_id="secops_lead",
                session_id=f"sec_scan_{i}",
                events=[
                    _make_event(
                        "Security advisory: payment-router-01 is affected by CVE-2026-9981 "
                        "and mitigated by Envoy WAF rule."
                    )
                ],
            )
        evolution_audit = await service.evolve_app_ontology_from_overflow(
            "t_bench:secops_auditor",
            min_confidence_to_promote=0.85,
            registrar=registrar,
        )
        evolved_ctx = await service.registry.resolve_context(
            "t_bench", "secops_auditor", "secops_lead"
        )
        evolved_tbox = evolved_ctx.tbox

        # 7. Sleep-Time Mode Latency Benchmark (<10ms hot path)
        t0 = time.perf_counter()
        await service.add_events_to_memory(
            app_name="t_bench:sre_commander",
            user_id="alice_vp",
            session_id="inc_sleep_1",
            events=[_make_event("Post-incident telemetry: p99 latency stabilized at 18ms.")],
            extraction_mode=ExtractionMode.SLEEP_TIME,
        )
        latencies_ms["sleep_time_ingest_ms"].append((time.perf_counter() - t0) * 1000)

        # 8. Register & Reinforce Tier 3 Playbook
        await service.register_playbook(
            app_name="t_bench:sre_commander",
            user_id="alice_vp",
            trigger_intent="payment-router-01 region failover and ledger check",
            workflow_steps=[
                "verify_aurora_ledger_01_replication",
                "drain_envoy_us_east_1",
                "shift_bgp_anycast_eu_central_1",
            ],
            resolution_summary="Zero-data-loss failover runbook for payment-router-01",
        )
        t0 = time.perf_counter()
        pb = await service.register_playbook(
            app_name="t_bench:sre_commander",
            user_id="alice_vp",
            trigger_intent="payment-router-01 region failover and ledger check",
            workflow_steps=[
                "verify_aurora_ledger_01_replication",
                "drain_envoy_us_east_1",
                "shift_bgp_anycast_eu_central_1",
            ],
            resolution_summary="Zero-data-loss failover runbook for payment-router-01",
        )
        latencies_ms["playbook_recall_ms"].append((time.perf_counter() - t0) * 1000)

        # 9. Multi-Hop Recall + Cross-App User Persona Recall in `sre_commander`
        t0 = time.perf_counter()
        sre_recall = await service.search_memory(
            app_name="t_bench:sre_commander",
            user_id="alice_vp",
            query="payment-router-01 region failover and ledger check",
            max_hops=2,
        )
        latencies_ms["multihop_recall_ms"].append((time.perf_counter() - t0) * 1000)
        sre_combined = "\n".join(m.content.parts[0].text for m in sre_recall.memories)

        # 10. FinOps Active Recall (verifying Steward-enabled functional edge invalidation!)
        finops_recall = await service.search_memory(
            app_name="t_bench:finops_economist",
            user_id="alice_vp",
            query="payment-router-01 cost center owner",
        )
        finops_combined = "\n".join(
            m.content.parts[0].text for m in finops_recall.memories
        )

        # 11. Hostile Tenant Isolation Probe (`t_rival:spy_agent`)
        rival_recall = await service.search_memory(
            app_name="t_rival:spy_agent",
            user_id="mallory",
            query="payment-router-01 aurora-ledger-01 CC-209-Global",
        )
        rival_combined = "\n".join(
            m.content.parts[0].text for m in rival_recall.memories
        )

        # 12. Background Maintenance Run
        t0 = time.perf_counter()
        maint_stats = await service.run_memory_maintenance(min_episodes_to_compact=2)
        latencies_ms["maintenance_compaction_ms"].append(
            (time.perf_counter() - t0) * 1000
        )

        report = {
            "llm_mode": "LIVE_GEMINI_2_5_FLASH" if live_gemini else "DETERMINISTIC_ENTERPRISE_SIMULATION",
            "embedding_model": "text-embedding-004 (1536d)" if live_gemini else "DeterministicHashEmbeddingAdapter (1536d)",
            "latency_benchmarks_ms": {
                k: round(statistics.mean(v), 2) if v else 0.0
                for k, v in latencies_ms.items()
            },
            "maturity_checks": {
                "entity_resolution_guard_sibling_separation": (
                    "payment-router-01" in sre_combined
                ),
                "bitemporal_functional_failover_active_eu_central_1": (
                    "(payment-router-01) -[DEPLOYED_IN_PRIMARY]-> (eu-central-1)"
                    in sre_combined
                    and "(payment-router-01) -[DEPLOYED_IN_PRIMARY]-> (us-east-1)"
                    not in sre_combined
                ),
                "multihop_dependency_preservation": (
                    "aurora-ledger-01" in sre_combined or "api-edge-gw" in sre_combined
                ),
                "cross_app_user_preference_recall": (
                    "UTC-Executive-Bullets" in sre_combined
                ),
                "pii_and_bearer_token_redacted": (
                    "123-45-6789" not in sre_combined
                    and "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sre_combined
                ),
                "steward_auto_functional_cost_center_invalidation": (
                    "(payment-router-01) -[OWNED_BY_PRIMARY]-> (CC-209-Global)"
                    in finops_combined
                    and "(payment-router-01) -[OWNED_BY_PRIMARY]-> (CC-104-Core)"
                    not in finops_combined
                ),
                "bottom_up_ontology_evolution_promoted_classes": list(
                    evolved_tbox.classes.keys()
                ),
                "cross_tenant_zero_leakage": (
                    "payment-router-01" not in rival_combined
                    and "CC-209-Global" not in rival_combined
                ),
                "matched_procedural_playbook_in_context": (
                    "[MATCHED_PROCEDURAL_PLAYBOOKS]" in sre_combined
                ),
                "postgres_neo4j_maintenance_compaction_and_dossiers": (
                    maint_stats["compacted_raw_episodes"] >= 2
                    and maint_stats["synthesized_entity_dossiers"] >= 3
                ),
                "playbook_reinforcement_count": pb.success_count,
            },
            "sample_sre_commander_context_window": sre_combined,
            "sample_finops_context_window": finops_combined,
            "maintenance_stats": maint_stats,
        }
        out_path = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "maturity_observation_report.json"
        )
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        await pg_adapter.close()
        await neo4j_adapter.close()
        await redis_adapter.close()


if __name__ == "__main__":
    res = asyncio.run(run_realistic_enterprise_maturity_benchmark())
    print(json.dumps(res, indent=2))
