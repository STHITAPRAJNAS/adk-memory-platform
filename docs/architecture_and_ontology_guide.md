# Enterprise Agentic Memory Platform: Architecture & Ontology Guide

This document distills the architectural foundations for building a **domain-agnostic, multi-tenant Composite Memory SDK & Platform Service** for **Google Agent Development Kit (ADK)** platforms hosting 100s of agents across diverse business domains.

---

## 1. First Principles: Domain Model vs. Ontology (TBox) vs. Graph KB (ABox)

| Layer | Formal Role | What It Defines | Mutability | Representation in `agentic-memory-sdk` | Failure Mode if Omitted |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Domain Model** | Conceptual Boundary | Bounded context, business vocabulary, core entities, and invariants (e.g., SRE, CRM, Clinical Trials, Payments). | Low | `DomainPackSpec` JSON/YAML/Python/Turtle packs in `examples/domain_packs/`. | Context bleeding across business units. |
| **2. Ontology (TBox)** | Terminological Box (Schema & Rulebook) | Formal classes, `subClassOf` hierarchies, allowed predicates, domain/range rules, disjointness, and functional cardinality (`is_functional`). | Versioned & Governed | `CompiledTBox`, `RelationRule`, OWL `.ttl` & SHACL shapes. | Label & predicate explosion (`uses`, `utilizes`, `relies_on`). |
| **3. Graph KB (ABox)** | Assertional Box (Instance State & Facts) | Populated graph of concrete entity vertices, episodic records, and bi-temporal edges (`t_valid`, `t_invalid`, `t_created`). | High (real-time streaming) | Amazon Neptune (`openCypher`), Neo4j (`Bolt`), PostgreSQL (`pgvector`), OpenSearch. | Stateless amnesia or contradictory stale facts. |

---

## 2. Stratified Ontology Architecture (Supporting Any Domain Without Bottlenecks)

To support **100s of arbitrary ADK agent apps** on one shared platform without forcing every team to write formal OWL/SHACL upfront:

```
+-----------------------------------------------------------------------------------+
| LAYER 0: PLATFORM UPPER ONTOLOGY (Mandatory for 100% of Agent Apps, Zero-Config)  |
|   Classes:    p:Actor, p:Object, p:Event, p:Concept, p:CandidateConcept           |
|   Properties: t_valid, t_invalid, t_created, tenant_id, namespace, episode_id     |
+-----------------------------------------+-----------------------------------------+
                                          | rdfs:subClassOf / Extends
          +-------------------------------+-------------------------------+
          v                                                               v
+-----------------------------------------+     +-----------------------------------+
| LAYER 1: FEDERATED DOMAIN PACKS         |     | LAYER 2: APP-LOCAL EXTENSIONS     |
| (Plug-and-Play Shared Business TBoxes)  |     | (Self-Service / Emergent Schema)  |
|   - sre_infra@v1.2                      |     |   - Inline Python Pydantic models |
|   - customer_crm@v2.0                   |     |   - Auto-quarantined overflow in  |
|   - clinical_trials@v1.0                |     |     `app:{tenant}:{app}`          |
|   - payments@v2.1 (.ttl OWL/SHACL)      |     |                                   |
+-----------------------------------------+     +-----------------------------------+
```

### Progressive Governance Tiers

1. **Tier 0 (`tier0_auto`)**: Zero configuration required. Writes land in `app:{tenant}:{app}` using Layer 0 (`UpperConceptType`) classes.
2. **Tier 1 (`tier1_app_typed`)**: Developer passes Python Pydantic models or custom classes in `AppMemoryManifest`. Isolated to `app:{tenant}:{app}`.
3. **Tier 2 (`tier2_domain_shared`)**: App subscribes to one or more `DomainPackSpec` packs (`subscribed_domain_packs=["sre_infra@v1.2"]`). Governed facts are shared in `domain:{tenant}:{domain}`, while non-conforming or low-confidence extractions are routed via the **Two-Bucket Pattern** into `app:{tenant}:{app}` as `p:CandidateConcept`.
4. **Tier 3 (`tier3_enterprise_core`)**: Read-only master reference data in `global:{tenant}:core`.

---

## 3. The 3-Tier Context Graph Topology

1. **Tier 1 — Episodic Subgraph (`EpisodicRecord`)**:
   - Immutable raw conversation turns and tool outputs with dense embeddings.
   - Strictly isolated per `(tenant_id, app_name, user_id)` via PostgreSQL Row-Level Security (RLS) or OpenSearch security filters.
2. **Tier 2 — Bi-Temporal Semantic Subgraph (`GraphVertexUpsert` & `GraphEdgeMutation`)**:
   - Every relationship edge carries `t_valid`, `t_invalid` (`"ACTIVE"` or ISO timestamp), `t_created`, `source_episode_id`, and `confidence`.
   - When an edge predicate is marked `is_functional=True` in the TBox (e.g., `DEPLOYED_IN_PRIMARY`, `SUBSCRIBED_TO_TIER`, `CURRENT_PHASE`), committing a new target automatically sets `t_invalid = new_t_valid` on the superseded active edge.
3. **Tier 3 — Procedural Reasoning Trace Subgraph (`ReasoningTraceStep`)**:
   - Captures `(:AgentRun)-[:EXECUTED_STEP]->(:ToolCall)-[:TOUCHED_ENTITY]->(:Entity)` so agents can audit and reuse successful execution paths.

---

## 4. Production Edge-Case Defense Matrix

| Edge Scenario | SDK Defense Mechanism | Implementation |
| :--- | :--- | :--- |
| **1. Dual-Write Partial Failure** | Transactional Outbox Pattern (`OutboxOperation` + `process_outbox_batch`) | `service.py`, `postgres_pgvector.py` |
| **2. Concurrent Entity Creation Race** | Deterministic SHA-256 canonical vertex ID (`GraphVertexUpsert.deterministic_id`) | `models.py` |
| **3. LLM Schema / SHACL Violation** | Two-Bucket Runtime Validator (`TBoxRuntimeValidator.validate_and_partition`) | `governance.py` |
| **4. Low-Confidence / Prompt Injection** | Confidence threshold gating (`min_domain_confidence=0.85`) | `governance.py` |
| **5. Graph DB Outage / Spike** | `AsyncCircuitBreaker` + graceful fallback to Seed Index summaries | `governance.py`, `service.py` |
| **6. Context Window Overflow** | `TokenBudgetSerializer` ranking active facts by confidence within token budget | `governance.py` |
| **7. PII / Secret Leakage** | Pre-ingestion `PIIScrubber` (SSNs, PANs, API keys, Bearer tokens) | `governance.py` |
| **8. Right-to-be-Forgotten (GDPR)** | Cascading lineage purge (`purge_user_lineage`) across Episodic, Seed, and Graph stores | `service.py`, `adapters/` |
