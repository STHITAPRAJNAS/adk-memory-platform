# Local Setup, Docker Stack & PyPI Testing Guide

This guide walks through setting up the complete local multi-backend infrastructure on your machine (`PostgreSQL + pgvector`, `Neo4j`, and `Redis`), running the test suites, and testing the **published PyPI release (`agentic-composite-memory`)** against a realistic 5-agent enterprise simulation.

---

## 1. Local Docker Infrastructure Setup

The repository includes a ready-to-run [`docker-compose.yml`](../docker-compose.yml) that provisions all 3 production storage engines with healthchecks and automatic PostgreSQL Row-Level Security (RLS) + `pgvector` HNSW initialization ([`docs/sql/001_postgres_rls_and_partitions.sql`](./sql/001_postgres_rls_and_partitions.sql)):

| Container | Image | Host Port | Credentials / URI | Role in Platform |
| :--- | :--- | :--- | :--- | :--- |
| **`adk-memory-postgres`** | `pgvector/pgvector:pg16` | `5433 -> 5432` | `postgresql://platform_admin:platform_secret@127.0.0.1:5433/adk_memory_platform` | Tenant Registry, Episodic & Seed Store (`pgvector` HNSW + BM25 GIN + RLS), Transactional Outbox |
| **`adk-memory-neo4j`** | `neo4j:5-community` | `7687` (Bolt), `7474` (HTTP) | `bolt://127.0.0.1:7687` (`neo4j` / `platform_secret`) | Tier-2 Bi-Temporal Knowledge Graph (`t_valid`, `t_invalid`) & Tier-3 Reasoning Traces |
| **`adk-memory-redis`** | `redis:7-alpine` | `6379` | `redis://127.0.0.1:6379/0` | Sub-5ms Tier-3 Procedural Playbook Cache & Working Memory |

### Start the Containers
```bash
# Start all 3 containers and wait until healthchecks pass
docker compose up -d --wait

# Verify container health
docker compose ps
```

---

## 2. Running the Test Suites (Unit + Live Docker Multi-Agent A2A)

### A. Standalone Unit & Platform Integration Suite (No Docker Required)
Tests all 3 ontology formats (JSON, Pydantic, OWL/SHACL `.ttl`), bi-temporal conflict invalidation, point-in-time historical queries (`as_of_iso`), Two-Bucket overflow quarantine, bottom-up schema evolution, PII/Bearer scrubbing, circuit-breaker degradation, GDPR purge, FastAPI + MCP co-hosting, Cloud Auth providers, and LLM/Embedding adapters:

```bash
uv run --extra dev pytest tests/test_composite_memory_platform.py -v
```

### B. Live Docker + Multi-Agent Google ADK A2A End-to-End Suite
Spins up live `adk-memory-postgres`, `adk-memory-neo4j`, and `adk-memory-redis` connections alongside 3 live Google ADK A2A agents (`platform_concierge`, `sre_specialist`, `crm_specialist`) and verifies cross-agent memory persistence, bi-temporal graph mutations, and single-stack Postgres/Redis parity:

```bash
uv run --extra dev pytest tests/test_a2a_docker_e2e.py -v -s
```

---

## 3. Testing the Published PyPI Package in an Isolated Environment

To verify the exact wheel published on **PyPI (`agentic-composite-memory`)** without importing local `src/` files, use `uv run --isolated --no-project`:

```bash
# 1. Ensure Docker containers are running
docker compose up -d --wait

# 2. Run the 5-Agent Enterprise Simulation using ONLY the PyPI wheel from pypi.org
uv run --isolated --no-project --refresh --with "agentic-composite-memory[all]" python -c "
import agentic_composite_memory, agentic_memory_sdk, importlib.metadata, runpy
print('=== PYPI PACKAGE VERIFICATION ===')
print('Installed Version:', importlib.metadata.version('agentic-composite-memory'))
print('agentic_composite_memory.__file__:', agentic_composite_memory.__file__)
assert 'site-packages' in agentic_memory_sdk.__file__, 'Must be imported from PyPI site-packages!'
runpy.run_path('examples/realistic_enterprise_stress_test.py', run_name='__main__')
"
```

> **Tip: Running with a Live Gemini API Key**
> By default, `examples/realistic_enterprise_stress_test.py` runs in deterministic offline mode (`DeterministicHashEmbeddingAdapter` + scripted enterprise extractor) so CI and local runs work without API keys. Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in your environment to switch the benchmark automatically to live **Gemini 2.5 Flash** (`GeminiOntologyExtractorAdapter`) and **`text-embedding-004`** (`GeminiEmbeddingAdapter`):
> ```bash
> export GOOGLE_API_KEY="your-gemini-api-key"
> uv run python examples/realistic_enterprise_stress_test.py
> ```

---

## 4. What `examples/realistic_enterprise_stress_test.py` Simulates & Verifies

The benchmark simulates **5 collaborating and competing agents** operating against live Docker PostgreSQL (`:5433`), Neo4j (`:7687`), and Redis (`:6379`):

1. **`sre_commander` (`t_bench`)**:
   - Ingests multi-hop topology (`api-edge-gw -> payment-router-01 -> aurora-ledger-01`) alongside sibling canary service (`payment-router-02` in `eu-west-1`) and raw credentials (`Bearer eyJ...`, SSN `123-45-6789`).
   - Tests that an unconfirmed proposal (`"Should we drain us-east-1 and fail payment-router-01 over to eu-central-1?"`) does **not** prematurely mutate the graph, whereas user confirmation (`"Yes, fail payment-router-01 over to eu-central-1 immediately"`) closes `t_invalid` on `us-east-1` and activates `eu-central-1`.
2. **`finops_economist` (`t_bench`)**:
   - Onboards with `PlatformOntologyRegistrar` auto-inferring `OWNED_BY_PRIMARY` as a functional relation (`is_functional=True`).
   - Records `alice_vp`'s executive preference (`UTC-Executive-Bullets`) in `user:t_bench:alice_vp` and transfers `payment-router-01` ownership from `CC-104-Core` to `CC-209-Global`.
3. **`secops_auditor` (`t_bench`)**:
   - Ingests an emergent security advisory (`CVE-2026-9981` of class `ZeroDayVulnerability` and predicate `MITIGATED_BY_WAF_RULE`).
   - Triggers `evolve_app_ontology_from_overflow()` to promote `ZeroDayVulnerability` and `MITIGATED_BY_WAF_RULE` from `app:t_bench:secops_auditor` into the governed ontology.
4. **`incident_scribe` (`t_bench`)**:
   - Benchmarks `ExtractionMode.SLEEP_TIME` (<10ms hot-path episodic write) followed by `run_memory_maintenance()` batch extraction, session compaction (`[AUTO_SESSION_SUMMARY]`), and Layer-3 Entity Dossier synthesis (`[Dossier: ...]`).
5. **`spy_agent` (`t_rival`)**:
   - Hostile tenant probe attempting to query `payment-router-01`, `aurora-ledger-01`, and `CC-209-Global` across PostgreSQL RLS, Neo4j, and Redis.

### Verified Benchmark Output ([`docs/maturity_observation_report.json`](./maturity_observation_report.json))

| Verification Check | Result | Measured Value / Behavior |
| :--- | :--- | :--- |
| **`entity_resolution_guard_sibling_separation`** | `true` | `payment-router-01` and `payment-router-02` remain separate vertices |
| **`bitemporal_functional_failover_active_eu_central_1`** | `true` | `us-east-1` closed (`t_invalid`), `eu-central-1` active |
| **`multihop_dependency_preservation`** | `true` | 2-hop chain `api-edge-gw -> payment-router-01 -> aurora-ledger-01` recalled |
| **`cross_app_user_preference_recall`** | `true` | `alice_vp` preference written in FinOps recalled automatically in SRE Commander |
| **`pii_and_bearer_token_redacted`** | `true` | JWT & SSN scrubbed to `[REDACTED_SECRET]` and `[REDACTED_SSN]` |
| **`steward_auto_functional_cost_center_invalidation`** | `true` | `CC-104-Core` superseded by `CC-209-Global` |
| **`bottom_up_ontology_evolution_promoted_classes`** | `true` | `ZeroDayVulnerability` promoted to TBox |
| **`cross_tenant_zero_leakage`** | `true` | `0` bytes leaked to `t_rival` |
| **`matched_procedural_playbook_in_context`** | `true` | `[MATCHED_PROCEDURAL_PLAYBOOKS]` injected at top of context window |
| **`postgres_neo4j_maintenance_compaction_and_dossiers`** | `true` | `compacted_raw_episodes: 3`, `summaries_created: 1`, `synthesized_entity_dossiers: 5` |
| **Sleep-Time Hot-Path Ingest Latency** | **`~8.5 – 10.8 ms`** | Sub-11ms PostgreSQL `pgvector` write with zero LLM blocking |
| **Redis Playbook Match Latency** | **`~4.7 – 4.9 ms`** | Sub-5ms cosine lookup and reinforcement |

---

## 5. Head-to-Head A/B Enterprise Agent Lift Benchmark (`examples/enterprise_agent_lift_benchmark.py`)

To quantify the exact accuracy and reliability **Lift** that enterprise agent applications get from `agentic-composite-memory` compared to a **Baseline Agent (Flat Vector-Only Top-K Chunk RAG)**, run [`examples/enterprise_agent_lift_benchmark.py`](../examples/enterprise_agent_lift_benchmark.py) against the live Docker stack:

```bash
docker compose up -d --wait
uv run python examples/enterprise_agent_lift_benchmark.py
```

The benchmark executes **4 collaborating agent frameworks** (**Google ADK**, **LangGraph `BaseStore` + `StateGraph` Nodes + `StructuredTool`s**, **FastAPI Remote Client**, and **MCP `2026-07-28` Streamable HTTP**) against live **PostgreSQL 16 (`pgvector` + RLS :5433) + Neo4j 5 (`:7687`) + Redis 7 (`:6379`)** and saves the report to **[`docs/enterprise_agent_lift_report.json`](./enterprise_agent_lift_report.json)**:

| Enterprise Evaluation Scenario | Arm A: Baseline Agent (Flat Vector-Only RAG) | Arm B: Enterprise Agent with `agentic-composite-memory` | Measured Agent Lift |
| :--- | :---: | :---: | :--- |
| **1. Active State Precision (Stale-Fact Elimination)** (`ORD-88410` re-routed `FC-Seattle-01` $\rightarrow$ `FC-Reno-02` $\rightarrow$ `FC-Dallas-03`) | **33.3%** *(2 stale chunks retrieved)* | **100.0%** *(0% stale functional edge contamination)* | **`+66.7%` Active State Precision Lift** |
| **2. Point-in-Time Historical Audit Accuracy** (`as_of_iso="2026-11-06T00:00:00Z"`) | **0.0%** | **100.0%** *(Exact `FC-Reno-02` historical state)* | **`+100.0%` Historical Audit Lift** |
| **3. Multi-Hop (2-Hop) Blast-Radius Discovery** (`mobile-checkout-bff` $\rightarrow$ `payment-router-01` $\rightarrow$ `aurora-ledger-01`) | **0.0%** | **100.0%** *(Neo4j 2-hop Cypher traversal)* | **`+100.0%` Multi-Hop Discovery Lift** |
| **4. Sibling Entity Disambiguation** (`payment-router-01` vs. `payment-router-02`) | **50.0%** | **100.0%** *(`EntityResolutionGuard`)* | **`+50.0%` Disambiguation Lift** |
| **5. Cross-App Persona Survival Under Churn** *(12 telemetry turns + session compaction)* | **100.0%** *(Degrades as turn volume grows)* | **100.0%** *(Pinned in `user:{tenant}:{user_id}`)* | **Guaranteed Cross-App Persona Retention** |
| **6. Zero-Shot Procedural Playbook SOP Recall** *(Redis Tier-3 Playbook cache)* | **0.0%** | **100.0%** *(`[MATCHED_PROCEDURAL_PLAYBOOKS]`)* | **`+100.0%` Procedural SOP Adherence Lift** |
| **7. Pre-Flight Hallucination Gate (`check_answerable`)** *(Detects `"return on ad spend"` gap & unanswerable queries)* | **0.0%** | **100.0%** *(`ANSWERABLE` / `PARTIAL` / `UNANSWERABLE`)* | **`+100.0%` Pre-Flight Hallucination Prevention** |
| **8. Autonomous Schema Induction & Evolution** *(SQL DDL + Ossie YAML + Steward Certify + `min_support_count=2` promotion)* | **N/A** | **100.0% Verified** *(Promoted `ColdChainShipment`, synthesized 19 Entity Dossiers)* | **Zero-Touch Ontology Evolution** |
