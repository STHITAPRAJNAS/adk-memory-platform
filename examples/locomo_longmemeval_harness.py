"""
Synthetic 6-Category Memory Evaluation Harness (LoCoMo / LongMemEval-Inspired Categories).

Note: This is a lightweight 6-scenario synthetic check covering LoCoMo/LongMemEval-style
reasoning categories (single-hop, multi-hop, temporal updates, contradiction reconciliation,
unanswerable abstention, and TTL retention). It does NOT load the official external LoCoMo or
LongMemEval benchmark datasets.

Evaluates `agentic-composite-memory` across 6 synthetic evaluation categories:
  1. Single-Hop Factual Recall
  2. Multi-Hop Relational Reasoning (2-hop graph traversal)
  3. Temporal State Update & Point-in-Time Audit (`as_of_iso`)
  4. Free-Text Preference & Goal Reconciliation (`ADD -> UPDATE -> RESOLVE`)
  5. Out-of-Domain Unanswerable Abstention (`check_answerable`)
  6. TTL Retention Sweep & Letta-Style Core Working Memory Blocks

Supports BOTH:
  - `--mode deterministic` (default, zero API keys, runs in <1s for CI)
  - `--mode gemini` / `--mode openai` / `--mode ollama` (real frontier LLM extraction & embeddings)

Usage:
  python examples/locomo_longmemeval_harness.py --mode deterministic
  GEMINI_API_KEY=... python examples/locomo_longmemeval_harness.py --mode gemini
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from agentic_composite_memory import (
    AppMemoryManifest,
    CompositeMemoryService,
    DeterministicHashEmbeddingAdapter,
    DomainPackSpec,
    Event,
    ExtractedEntity,
    ExtractedFact,
    ExtractionPayload,
    FewShotExtractionExample,
    GeminiEmbeddingAdapter,
    GeminiOntologyExtractorAdapter,
    GovernanceTier,
    InMemoryUnifiedAdapter,
    MemoryRetentionPolicy,
    OllamaEmbeddingAdapter,
    OllamaOntologyExtractorAdapter,
    OpenAICompatibleEmbeddingAdapter,
    OpenAICompatibleOntologyExtractorAdapter,
    PlatformOntologyRegistrar,
    RelationRule,
    RuleBasedMockExtractorAdapter,
    types,
)


def _ev(text: str, author: str = "user") -> Event:
    return Event(
        author=author,
        content=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
    )


def _build_adapters(mode: str):
    if mode == "gemini":
        return (
            GeminiEmbeddingAdapter(output_dimensionality=128),
            GeminiOntologyExtractorAdapter(),
        )
    if mode == "openai":
        return (
            OpenAICompatibleEmbeddingAdapter(
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                output_dimensionality=128,
            ),
            OpenAICompatibleOntologyExtractorAdapter(
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            ),
        )
    if mode == "ollama":
        return (
            OllamaEmbeddingAdapter(output_dimensionality=128),
            OllamaOntologyExtractorAdapter(),
        )

    ext = RuleBasedMockExtractorAdapter()
    ext.add_keyword_rule(
        "alice works on project apollo",
        ExtractionPayload(
            entities=[
                ExtractedEntity(canonical_name="Alice", entity_class="Person", summary="Staff Engineer Alice"),
                ExtractedEntity(canonical_name="Project-Apollo", entity_class="Project", summary="Payments modernization project"),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Alice",
                    source_class="Person",
                    predicate="WORKS_ON",
                    target_entity="Project-Apollo",
                    target_class="Project",
                    fact_summary="Alice works on Project-Apollo",
                    valid_from_iso="2026-01-10T09:00:00Z",
                )
            ],
        ),
    )
    ext.add_keyword_rule(
        "project apollo depends on ledger-db-eu",
        ExtractionPayload(
            entities=[
                ExtractedEntity(canonical_name="Project-Apollo", entity_class="Project", summary="Payments modernization project"),
                ExtractedEntity(canonical_name="Ledger-DB-EU", entity_class="System", summary="EU Ledger Database"),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Project-Apollo",
                    source_class="Project",
                    predicate="DEPENDS_ON",
                    target_entity="Ledger-DB-EU",
                    target_class="System",
                    fact_summary="Project-Apollo depends on Ledger-DB-EU",
                    valid_from_iso="2026-01-10T09:05:00Z",
                )
            ],
        ),
    )
    ext.add_keyword_rule(
        "alice is based in boston",
        ExtractionPayload(
            entities=[
                ExtractedEntity(canonical_name="Alice", entity_class="Person", summary="Staff Engineer Alice"),
                ExtractedEntity(canonical_name="Boston", entity_class="p:SpatialLocation", summary="Boston Office"),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Alice",
                    source_class="Person",
                    predicate="CURRENT_OFFICE",
                    target_entity="Boston",
                    target_class="p:SpatialLocation",
                    fact_summary="Alice is currently based in Boston",
                    valid_from_iso="2026-02-01T09:00:00Z",
                )
            ],
        ),
    )
    ext.add_keyword_rule(
        "alice relocated to zurich",
        ExtractionPayload(
            entities=[
                ExtractedEntity(canonical_name="Alice", entity_class="Person", summary="Staff Engineer Alice"),
                ExtractedEntity(canonical_name="Zurich", entity_class="p:SpatialLocation", summary="Zurich Office"),
            ],
            facts=[
                ExtractedFact(
                    source_entity="Alice",
                    source_class="Person",
                    predicate="CURRENT_OFFICE",
                    target_entity="Zurich",
                    target_class="p:SpatialLocation",
                    fact_summary="Alice relocated to Zurich",
                    valid_from_iso="2026-06-01T09:00:00Z",
                )
            ],
        ),
    )
    return DeterministicHashEmbeddingAdapter(dim=128), ext


async def run_locomo_longmemeval_harness(mode: str = "deterministic") -> dict[str, Any]:
    t0 = time.perf_counter()
    store = InMemoryUnifiedAdapter()
    embedder, extractor = _build_adapters(mode)
    service = CompositeMemoryService(
        registry_adapter=store,
        seed_store_adapter=store,
        graph_adapter=store,
        embedding_adapter=embedder,
        extractor_adapter=extractor,
    )
    registrar = PlatformOntologyRegistrar(store)
    registrar.register_domain_pack(
        DomainPackSpec(
            pack_id="enterprise_work@v1.0",
            domain_name="enterprise_work",
            classes={
                "Person": "p:Actor",
                "Project": "p:Concept",
                "System": "p:Object",
            },
            relations={
                "WORKS_ON": RelationRule(predicate="WORKS_ON", domain_class="Person", range_class="Project", is_functional=False),
                "DEPENDS_ON": RelationRule(predicate="DEPENDS_ON", domain_class="Project", range_class="System", is_functional=False, is_transitive=True),
                "CURRENT_OFFICE": RelationRule(predicate="CURRENT_OFFICE", domain_class="Person", range_class="p:SpatialLocation", is_functional=True),
            },
        )
    )
    await registrar.register_adk_app(
        AppMemoryManifest(
            tenant_id="bench",
            app_name="eval_agent",
            governance_tier=GovernanceTier.TIER2_DOMAIN_SHARED,
            subscribed_domain_packs=["enterprise_work@v1.0"],
            retention_policy=MemoryRetentionPolicy(hypothesis_ttl_days=14, completed_goal_ttl_days=30),
            custom_extraction_topics=["Employee locations", "Project system dependencies", "Dietary and travel preferences"],
            few_shot_examples=[
                FewShotExtractionExample(
                    input_text="Bob moved to Tokyo office.",
                    expected_entities=[{"canonical_name": "Bob", "entity_class": "Person"}],
                    expected_facts=[{"source_entity": "Bob", "predicate": "CURRENT_OFFICE", "target_entity": "Tokyo"}],
                    notes="Functional office assignment",
                )
            ],
        )
    )

    app = "bench:eval_agent"
    user = "alice"

    # Ingest benchmark dialogue stream
    turns = [
        "Alice works on Project Apollo for Q3 payments modernization.",
        "Project Apollo depends on Ledger-DB-EU for settlement storage.",
        "Alice is based in Boston office as of February 2026.",
        "Alice relocated to Zurich office in June 2026.",
        "My preference is vegetarian meals on all business flights.",
        "I switched from vegetarian meals to pescatarian meals on business flights.",
        "Pending goal: complete SOC2 evidence review for Ledger-DB-EU by Friday.",
        "Completed SOC2 evidence review for Ledger-DB-EU ahead of schedule.",
    ]
    for txt in turns:
        await service.add_events_to_memory(app_name=app, user_id=user, events=[_ev(txt)])

    # Add an already-expired TTL memory to test retention sweep
    past_exp = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    await service.add_events_to_memory(
        app_name=app,
        user_id=user,
        events=[_ev("Temporary promo code TEMP-999 valid only today.")],
        custom_metadata={"expires_at": past_exp},
    )
    maint = await service.run_memory_maintenance(min_episodes_to_compact=20)

    # Pin a Letta-style Core Working Memory Block
    service.upsert_working_memory_block(
        app_name=app,
        user_id=user,
        label="human_profile",
        value="Alice — Staff Engineer leading Project-Apollo (Zurich HQ).",
    )

    # 1. Single-Hop Factual Recall
    r1 = await service.search_memory(app_name=app, user_id=user, query="Project-Apollo")
    b1 = "\n".join(m.content.parts[0].text for m in r1.memories)
    c1_pass = "Alice" in b1 and "Project-Apollo" in b1

    # 2. Multi-Hop Relational Reasoning (Alice -> Project-Apollo -> Ledger-DB-EU)
    r2 = await service.search_memory(app_name=app, user_id=user, query="Alice", max_hops=2)
    b2 = "\n".join(m.content.parts[0].text for m in r2.memories)
    c2_pass = "Ledger-DB-EU" in b2

    # 3. Temporal State Update & Historical Point-in-Time Travel
    r3_now = await service.search_memory(app_name=app, user_id=user, query="Alice office")
    b3_now = "\n".join(m.content.parts[0].text for m in r3_now.memories)
    r3_past = await service.search_memory(
        app_name=app, user_id=user, query="Alice office", as_of_iso="2026-03-15T00:00:00Z"
    )
    b3_past = "\n".join(m.content.parts[0].text for m in r3_past.memories)
    c3_pass = (
        "(Alice) -[CURRENT_OFFICE]-> (Zurich)" in b3_now
        and "(Alice) -[CURRENT_OFFICE]-> (Boston)" not in b3_now
        and "(Alice) -[CURRENT_OFFICE]-> (Boston)" in b3_past
    )

    # 4. Free-Text Preference & Goal Reconciliation (ADD -> UPDATE -> RESOLVE)
    r4 = await service.search_memory(app_name=app, user_id=user, query="meals on business flights")
    b4 = "\n".join(m.content.parts[0].text for m in r4.memories)
    revs = service.get_memory_revisions(app_name=app, user_id=user, limit=50)
    rev_actions = {rv.action for rv in revs}
    c4_pass = (
        "pescatarian" in b4.lower()
        and "UPDATE" in rev_actions
        and "RESOLVE" in rev_actions
    )

    # 5. Unanswerable Abstention Gate
    ans_un = await service.check_answerable(
        app_name=app, user_id=user, query="What is the lunar gravity constant on Mars?"
    )
    c5_pass = ans_un["status"] == "UNANSWERABLE"

    # 6. TTL Retention Sweep & Working Memory Block Injection
    c6_pass = (
        maint["expired_memories"] >= 1
        and "TEMP-999" not in b1
        and "[WORKING_MEMORY_BLOCKS]" in b1
        and "Staff Engineer leading Project-Apollo" in b1
    )

    scores = {
        "1_single_hop_factual": c1_pass,
        "2_multi_hop_relational": c2_pass,
        "3_temporal_update_and_time_travel": c3_pass,
        "4_preference_and_goal_reconciliation": c4_pass,
        "5_unanswerable_abstention": c5_pass,
        "6_ttl_retention_and_working_blocks": c6_pass,
    }
    passed = sum(1 for v in scores.values() if v)
    return {
        "harness": "Synthetic 6-Category Memory Evaluation (LoCoMo/LongMemEval-Inspired, Not Official Dataset)",
        "mode": mode,
        "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "overall_score_pct": round(100.0 * passed / len(scores), 1),
        "categories_passed": f"{passed}/{len(scores)}",
        "category_results": scores,
        "revision_actions_recorded": sorted(rev_actions),
        "expired_memories_swept": maint["expired_memories"],
        "all_passed": passed == len(scores),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["deterministic", "gemini", "openai", "ollama"],
        default="deterministic",
    )
    args = parser.parse_args()
    report = asyncio.run(run_locomo_longmemeval_harness(mode=args.mode))
    out_path = Path(__file__).resolve().parent.parent / "docs" / "locomo_longmemeval_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    assert report["all_passed"] is True, f"Harness failed: {report}"
