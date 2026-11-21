# Domain Pack Authoring & Ontology Engineering Guide

This guide explains how **Layer-1 Federated Domain Packs (`DomainPackSpec`)** work in `agentic-composite-memory`, compares all **6 ways** to create one, and provides complete, production-grade blueprints for complex systems including **Omnichannel Retail & Supply Chain** and **Payment Transaction Processing, Clearing & Dispute Resolution**.

---

## 1. Anatomy of a Layer-1 Domain Pack (`DomainPackSpec`)

In `agentic-composite-memory`, the core engine is 100% domain-agnostic. It provides **Layer 0 (Platform Upper Ontology & Cognitive Memory Taxonomy)** with 12 built-in classes (`UpperConceptType`):
- **`p:Actor`**: People, customers, organizations, merchants, banks, carriers, agents, teams.
- **`p:Object`**: Physical or digital assets, SKUs, warehouses, microservices, databases, ledger accounts, payment cards.
- **`p:Event`**: Time-bounded occurrences—orders, shipments, payment transactions, chargeback disputes, outages, deployments.
- **`p:Concept`**: General domain concepts, loyalty tiers, risk scores, status codes, currencies.
- **`p:UserPreference`**: Personal user settings, sizing/fabric preferences, communication styles (auto-routed to `user:{tenant}:{user_id}` and immune to session compaction).
- **`p:PolicyGuardrail`**: Normative safety boundaries, compliance rules, and approval gates (`MUST_NOT_EXECUTE`, `REQUIRES_APPROVAL_FOR`, `CONSTRAINED_BY_POLICY` — pinned against compaction and given `+0.35` salience priority).
- **`p:WorkingHypothesis`**: Epistemic memory for tentative, unconfirmed theories or beliefs (`HYPOTHESIZES_CAUSE`, `ASSUMES_STATE` — tagged `[UNCONFIRMED_HYPOTHESIS]` and blocked from invalidating confirmed functional facts).
- **`p:EpisodicLesson`**: Reflexion/experiential memory for user corrections, postmortem insights, and pitfalls to avoid (`LEARNED_FROM_FAILURE`, `AVOIDS_PITFALL`, `CORRECTED_BY_USER` — pinned against compaction and given `+0.35` salience priority).
- **`p:ProspectiveGoal`**: Prospective memory for active goals, SLAs, pending follow-ups, and deadlines (`HAS_PENDING_GOAL`, functional `HAS_GOAL_STATUS`, `BLOCKED_BY`).
- **`p:QuantitativeMetric`**: Measurable KPIs, thresholds, financial totals, and latency/error readings (functional `MEASURED_AT_VALUE`, `HAS_THRESHOLD`).
- **`p:SpatialLocation`**: Physical or logical venues, addresses, and coordinates (`LOCATED_IN`).
- **`p:CandidateConcept`**: Automatic quarantine class for novel/unvalidated concepts discovered in `app:{tenant}:{app}` (`Bucket 2`).

A **Layer-1 Domain Pack (`DomainPackSpec`)** plugs your business vocabulary into Layer 0:

| Field | Type | Purpose |
| :--- | :--- | :--- |
| `pack_id` | `str` | Versioned identifier (e.g., `"retail_omnichannel@1.0.0"`, `"payment_processing@2.1.0"`). |
| `domain_name` | `str` | Logical domain namespace key (e.g., `"retail_omnichannel"` $\rightarrow$ writes to `domain:{tenant}:retail_omnichannel`). |
| `version` | `str` | Semantic version (`"1.0.0"`), automatically bumped by Steward Review or Bottom-Up Auto-Evolution. |
| `classes` | `dict[str, str]` | Mapping of `ClassName -> ParentClass` (e.g., `{"PaymentTransaction": "p:Event", "Merchant": "p:Actor"}`). |
| `relations` | `dict[str, RelationRule]` | Typed predicates with `domain_class`, `range_class`, `is_functional`, and `is_transitive`. |
| `disjoint_classes` | `list[tuple[str, str]]` | Pairs of classes that can never share an entity instance (e.g., `[("Merchant", "PaymentTransaction")]`). |
| `synonyms` | `dict[str, list[str]]` | Business aliases for classes/metrics used by `check_answerable()` pre-flight coverage checks. |
| `certified` | `bool` | Whether a domain steward has certified the schema via `certify_domain_pack()`. |

---

## 2. The 6 Ways to Create a Domain Pack

You can author or bootstrap a Domain Pack using whichever artifact your team already owns:

| Method | Loader / API / MCP Entrypoint | Best For |
| :--- | :--- | :--- |
| **1. Declarative JSON / Dict** | `DomainPackLoader.from_json_file(path)` / `POST /v1/domains/packs` / MCP `register_domain_pack` | GitOps repositories, CI/CD pipelines, and cross-language teams. |
| **2. Python Pydantic Models** | `DomainPackLoader.from_pydantic_models(...)` | Python, Google ADK, and LangGraph application teams ("Ontology-as-Code"). |
| **3. W3C OWL 2 / RDFS / SHACL Turtle (`.ttl`)** | `DomainPackLoader.from_turtle_file(...)` / `POST /v1/domains/packs/turtle` | Enterprise knowledge graph architects and industry standard ontologies (FIBO, GoodRelations, Schema.org). |
| **4. Relational SQL DDL (`CREATE TABLE`)** | `DomainPackLoader.from_sql_ddl(...)` / `POST /v1/domains/packs/sql-ddl` / MCP `register_domain_pack(sql_ddl=...)` | Bootstrapping directly from existing PostgreSQL, Snowflake, or MySQL OLTP/warehouse schemas (tables, FKs, and junction tables). |
| **5. Apache Ossie / BI Semantic YAML** | `DomainPackLoader.from_ossie_yaml(...)` / `POST /v1/domains/packs/ossie` / MCP `register_domain_pack(ossie_yaml=...)` | Importing `terms`, `synonyms`, `relationships`, and `metrics` from BI semantic layers (Apache Ossie, NVIDIA Auto-Ontology). |
| **6. Cold-Start Autonomous Transcript Induction** | `await service.induce_domain_pack(...)` / `POST /v1/domains/packs/induce` / MCP `induce_domain_pack` | Discovering classes, Layer-0 parents, and empirical `1:1` functional vs. `1:N` relations automatically from raw chat logs, tickets, or stored episodes. |

---

## 3. Golden Modeling Rules for Complex Enterprise Systems

Before creating a Domain Pack for a complex business, apply these three rules:

1. **Rule 1 — Set `is_functional=True` on Single-Valued Mutable State**:
   - Whenever an entity can only have **one current value** for a relationship at a given point in time, set `is_functional=True`.
   - *Retail Examples*: `ENROLLED_IN_TIER` (a customer upgrades from `Silver` to `Ambassador-VIP`), `SHIPPED_FROM_PRIMARY` (an order is re-routed from `WH-Newark-01` to `WH-Dallas-02`).
   - *Payments Examples*: `HAS_SETTLEMENT_STATUS` (`AUTHORIZED` $\rightarrow$ `CAPTURED` $\rightarrow$ `SETTLED`), `ROUTED_VIA_PRIMARY_PROCESSOR` (`Processor-A` $\rightarrow$ `Processor-B`), `HAS_DISPUTE_STATE` (`OPEN` $\rightarrow$ `REPRESENTMENT` $\rightarrow$ `WON`).
   - *What the engine does automatically*: When the new state arrives at $t_2$, the graph adapter automatically sets `t_invalid = t_2` on the old edge and inserts the new edge as `t_invalid = 'ACTIVE'`. Active queries only see the latest state, while `search_memory(..., as_of_iso=t_1)` reconstructs the exact historical state at $t_1$!
2. **Rule 2 — Set `is_functional=False` on Multi-Valued Collections**:
   - When an entity can have many simultaneous targets, keep `is_functional=False`.
   - *Examples*: `CONTAINS_SKU` (an order has 5 line items), `FLAGGED_BY_FRAUD_RULE` (a transaction triggers 3 fraud rules), `DEPENDS_ON`.
3. **Rule 3 — Set `is_transitive=True` on Hierarchical / Dependency Chains**:
   - When $A \rightarrow B$ and $B \rightarrow C$ implies $A \rightarrow^+ C$ across multi-hop graph expansion (`max_hops=2` or `3`), set `is_transitive=True`.
   - *Examples*: `DEPENDS_ON` (`PaymentGateway -> FraudEngine -> RedisCluster`), `PART_OF_MERCHANT_GROUP` (`StoreTerminal -> FranchiseMerchant -> ParentHoldingCorp`).

---

## 4. Blueprint A: Complex Omnichannel Retail & Supply-Chain Business

In a complex retail business, multiple agents collaborate:
- **Personal Stylist Agent**: Manages shopper preferences (`user:{tenant}:{user_id}`) and loyalty tier upgrades.
- **Order & Inventory Fulfillment Agent**: Tracks multi-SKU orders, warehouse inventory, and weather/stockout re-routing.
- **Returns & VIP Exchange Concierge Agent**: Executes instant exchanges using Tier-3 Procedural Playbooks.
- **Merchandising Trend Agent**: Discovers emergent social commerce trends (`Bucket 2` overflow $\rightarrow$ promoted via `evolve_app_ontology_from_overflow`).

Here is how to create the **Omnichannel Retail Domain Pack** using any of the 6 methods:

### Method 1: Declarative JSON ([`examples/domain_packs/retail_omnichannel.json`](../examples/domain_packs/retail_omnichannel.json))
```json
{
  "pack_id": "retail_omnichannel@1.0.0",
  "domain_name": "retail_omnichannel",
  "version": "1.0.0",
  "classes": {
    "RetailCustomer": "p:Actor",
    "CarrierPartner": "p:Actor",
    "ProductSKU": "p:Object",
    "FulfillmentWarehouse": "p:Object",
    "RetailOrder": "p:Event",
    "ReturnAuthorization": "p:Event",
    "LoyaltyTier": "p:Concept"
  },
  "disjoint_classes": [
    ["RetailCustomer", "ProductSKU"],
    ["RetailOrder", "FulfillmentWarehouse"]
  ],
  "synonyms": {
    "RetailOrder": ["purchase_order", "customer_order", "checkout_cart"],
    "FulfillmentWarehouse": ["distribution_center", "depot", "fc"]
  },
  "relations": {
    "ENROLLED_IN_TIER": {
      "predicate": "ENROLLED_IN_TIER",
      "domain_class": "RetailCustomer",
      "range_class": "LoyaltyTier",
      "is_functional": true,
      "description": "Active loyalty tier (functional: tier upgrade invalidates prior tier)"
    },
    "SHIPPED_FROM_PRIMARY": {
      "predicate": "SHIPPED_FROM_PRIMARY",
      "domain_class": "RetailOrder",
      "range_class": "FulfillmentWarehouse",
      "is_functional": true,
      "description": "Primary fulfillment warehouse (functional: re-routing invalidates old warehouse)"
    },
    "CONTAINS_SKU": {
      "predicate": "CONTAINS_SKU",
      "domain_class": "RetailOrder",
      "range_class": "ProductSKU",
      "is_functional": false,
      "description": "Line-item SKUs included in an order (multi-valued)"
    },
    "STOCKED_AT": {
      "predicate": "STOCKED_AT",
      "domain_class": "ProductSKU",
      "range_class": "FulfillmentWarehouse",
      "is_functional": false,
      "description": "Warehouses carrying inventory for this SKU"
    }
  }
}
```

### Method 2: From Your Existing Retail SQL Schema (`DomainPackLoader.from_sql_ddl`)
```python
from agentic_composite_memory import DomainPackLoader, PlatformOntologyRegistrar

retail_sql_ddl = """
CREATE TABLE sales.retail_customers (
    customer_id TEXT PRIMARY KEY,
    loyalty_tier_id TEXT REFERENCES sales.loyalty_tiers(tier_id)
);
CREATE TABLE warehouse.fulfillment_warehouses (
    warehouse_id TEXT PRIMARY KEY,
    region_code TEXT NOT NULL
);
CREATE TABLE catalog.product_skus (
    sku_id TEXT PRIMARY KEY,
    sku_name TEXT NOT NULL
);
CREATE TABLE sales.retail_orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES sales.retail_customers(customer_id),
    primary_warehouse_id TEXT REFERENCES warehouse.fulfillment_warehouses(warehouse_id)
);
-- Many-to-many junction table: automatically induces INCLUDES_PRODUCT_SKU (is_functional=False)
CREATE TABLE sales.order_line_items (
    line_id TEXT PRIMARY KEY,
    FOREIGN KEY (order_id) REFERENCES sales.retail_orders(order_id),
    FOREIGN KEY (sku_id) REFERENCES catalog.product_skus(sku_id)
);
"""

retail_pack = DomainPackLoader.from_sql_ddl("retail_omnichannel@1.0.0", retail_sql_ddl)
```

### Method 3: From Apache Ossie / BI Semantic YAML (`DomainPackLoader.from_ossie_yaml`)
```python
ossie_yaml = """
terms:
  - name: RetailCustomer
    synonyms: [shopper, buyer, member]
  - name: RetailOrder
    synonyms: [purchase_order, sales_order]
  - name: FulfillmentWarehouse
    synonyms: [distribution_center, fulfillment_center]
  - name: LoyaltyTier
    synonyms: [vip_tier, membership_level]
relationships:
  - name: ENROLLED_IN_TIER
    source: RetailCustomer
    target: LoyaltyTier
    cardinality: many_to_one
  - name: SHIPPED_FROM_PRIMARY
    source: RetailOrder
    target: FulfillmentWarehouse
    cardinality: many_to_one
"""
retail_pack = DomainPackLoader.from_ossie_yaml("retail_omnichannel@1.0.0", ossie_yaml)
```

---

## 5. Blueprint B: Complex Payment Transaction Processing, Clearing & Dispute Resolution Business

A payment processing platform involves strict PCI-DSS PII scrubbing (`PIIScrubber` redacting card PANs before embedding/LLM calls), high-stakes state transitions, multi-hop clearing chains, and point-in-time auditability for chargeback disputes:

- **Core Actors (`p:Actor`)**: `Cardholder`, `MerchantAccount`, `AcquiringBank`, `IssuingBank`, `PaymentProcessor`.
- **Core Infrastructure & Financial Objects (`p:Object`)**: `PaymentGateway`, `SettlementLedger`, `PaymentInstrument`, `FraudRuleEngine`.
- **Core Temporal Events (`p:Event`)**: `PaymentTransaction`, `ChargebackDispute`, `SettlementBatch`, `FraudAlertEvent`.
- **Core Governance Concepts (`p:Concept`)**: `TransactionStatus` (`AUTHORIZED`, `CAPTURED`, `SETTLED`, `DECLINED`, `REVERSED`), `DisputeStage` (`OPENED`, `EVIDENCE_SUBMITTED`, `REPRESENTMENT`, `ARBITRATION`, `WON`, `LOST`), `RiskTier`, `MccCategory`.

### Option 1: Authoring the Payment Processing Pack in W3C OWL 2 / SHACL Turtle (`.ttl`)

```turtle
@prefix : <http://platform.example.org/ontologies/payments#> .
@prefix p: <http://platform.example.org/ontologies/upper#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# --- Classes Anchored Under Layer 0 Upper Ontology ---
:Cardholder a owl:Class ; rdfs:subClassOf p:Actor .
:MerchantAccount a owl:Class ; rdfs:subClassOf p:Actor ;
    owl:disjointWith :PaymentTransaction , :SettlementLedger .
:AcquiringBank a owl:Class ; rdfs:subClassOf p:Actor .
:IssuingBank a owl:Class ; rdfs:subClassOf p:Actor .

:PaymentGateway a owl:Class ; rdfs:subClassOf p:Object .
:SettlementLedger a owl:Class ; rdfs:subClassOf p:Object .

:PaymentTransaction a owl:Class ; rdfs:subClassOf p:Event .
:ChargebackDispute a owl:Class ; rdfs:subClassOf p:Event .
:SettlementBatch a owl:Class ; rdfs:subClassOf p:Event .

:TransactionStatus a owl:Class ; rdfs:subClassOf p:Concept .
:DisputeStage a owl:Class ; rdfs:subClassOf p:Concept .
:RiskTier a owl:Class ; rdfs:subClassOf p:Concept .

# --- Functional State Relations (Bi-Temporal Conflict Invalidation Enabled) ---
:HAS_TRANSACTION_STATUS a owl:ObjectProperty , owl:FunctionalProperty ;
    rdfs:label "HAS_TRANSACTION_STATUS" ;
    rdfs:domain :PaymentTransaction ;
    rdfs:range :TransactionStatus .

:ROUTED_VIA_PRIMARY_GATEWAY a owl:ObjectProperty , owl:FunctionalProperty ;
    rdfs:label "ROUTED_VIA_PRIMARY_GATEWAY" ;
    rdfs:domain :PaymentTransaction ;
    rdfs:range :PaymentGateway .

:HAS_DISPUTE_STAGE a owl:ObjectProperty , owl:FunctionalProperty ;
    rdfs:label "HAS_DISPUTE_STAGE" ;
    rdfs:domain :ChargebackDispute ;
    rdfs:range :DisputeStage .

:ASSIGNED_RISK_TIER a owl:ObjectProperty , owl:FunctionalProperty ;
    rdfs:label "ASSIGNED_RISK_TIER" ;
    rdfs:domain :MerchantAccount ;
    rdfs:range :RiskTier .

# --- Structural & Transitive Lineage Relations ---
:PROCESSED_FOR_MERCHANT a owl:ObjectProperty , owl:FunctionalProperty ;
    rdfs:label "PROCESSED_FOR_MERCHANT" ;
    rdfs:domain :PaymentTransaction ;
    rdfs:range :MerchantAccount .

:DISPUTES_TRANSACTION a owl:ObjectProperty , owl:FunctionalProperty ;
    rdfs:label "DISPUTES_TRANSACTION" ;
    rdfs:domain :ChargebackDispute ;
    rdfs:range :PaymentTransaction .

:CLEARED_THROUGH_ACQUIRER a owl:ObjectProperty , owl:FunctionalProperty ;
    rdfs:label "CLEARED_THROUGH_ACQUIRER" ;
    rdfs:domain :MerchantAccount ;
    rdfs:range :AcquiringBank .

:SETTLES_INTO_LEDGER a owl:ObjectProperty , owl:TransitiveProperty ;
    rdfs:label "SETTLES_INTO_LEDGER" ;
    rdfs:domain :AcquiringBank ;
    rdfs:range :SettlementLedger .
```

Load and register it in one call:
```python
from agentic_composite_memory import DomainPackLoader

payments_pack = DomainPackLoader.from_turtle_str("payment_processing@2.1.0", turtle_str)
registrar.register_domain_pack(payments_pack)
```

### Option 2: Authoring the Payment Processing Pack in Python Pydantic ("Ontology-as-Code")

```python
from pydantic import BaseModel
from agentic_composite_memory import (
    DomainPackLoader,
    RelationRule,
    UpperConceptType,
)

class MerchantAccount(BaseModel):
    merchant_id: str
    mcc_code: str

class PaymentGateway(BaseModel):
    gateway_id: str
    region: str

class PaymentTransaction(BaseModel):
    txn_id: str
    amount_cents: int
    currency: str

class ChargebackDispute(BaseModel):
    dispute_id: str
    reason_code: str

class TransactionStatus(BaseModel):
    status_code: str

class DisputeStage(BaseModel):
    stage_code: str

payments_pack = DomainPackLoader.from_pydantic_models(
    pack_id="payment_processing@2.1.0",
    entity_models=[
        (MerchantAccount, UpperConceptType.ACTOR),
        (PaymentGateway, UpperConceptType.OBJECT),
        (PaymentTransaction, UpperConceptType.EVENT),
        (ChargebackDispute, UpperConceptType.EVENT),
        (TransactionStatus, UpperConceptType.CONCEPT),
        (DisputeStage, UpperConceptType.CONCEPT),
    ],
    relations=[
        RelationRule(
            predicate="HAS_TRANSACTION_STATUS",
            domain_class="PaymentTransaction",
            range_class="TransactionStatus",
            is_functional=True,
            description="Current lifecycle status (AUTHORIZED -> CAPTURED -> SETTLED)",
        ),
        RelationRule(
            predicate="ROUTED_VIA_PRIMARY_GATEWAY",
            domain_class="PaymentTransaction",
            range_class="PaymentGateway",
            is_functional=True,
            description="Active payment gateway route (functional failover)",
        ),
        RelationRule(
            predicate="DISPUTES_TRANSACTION",
            domain_class="ChargebackDispute",
            range_class="PaymentTransaction",
            is_functional=True,
            description="Links a chargeback dispute to its underlying transaction",
        ),
        RelationRule(
            predicate="HAS_DISPUTE_STAGE",
            domain_class="ChargebackDispute",
            range_class="DisputeStage",
            is_functional=True,
            description="Current chargeback lifecycle stage (OPENED -> REPRESENTMENT -> WON)",
        ),
    ],
    disjoint_classes=[("MerchantAccount", "PaymentTransaction")],
)
```

### Option 3: Cold-Start Autonomous Induction from Historical Payment Logs (`induce_domain_pack`)

If your team has raw payment operational logs or dispute transcripts and wants the platform to figure out the initial ontology automatically:

```python
induced_pack, audit_entry, stats = await service.induce_domain_pack(
    domain_name="payment_processing",
    transcripts=[
        "Transaction(TXN-7701) routed to payment-gateway-01 and deployed in us-east-1.",
        "Dispute(DSP-301) affects Transaction(TXN-7701) and assigned to Analyst(sarah_risk).",
        "Incident INC-881 affects payment-gateway-01 and mitigated by Patch(TLS-Hotfix-04).",
    ],
    min_support_count=1,
    min_confidence=0.75,
    auto_register=True,
    registrar=registrar,
)
```

---

## 6. The 4-Stage Enterprise Governance Lifecycle

Regardless of which of the 6 methods you use to create a Domain Pack, manage it in production through these 4 stages:

```text
+-----------------------+     +------------------------+     +-----------------------+     +------------------------+
| Stage 1: Draft/Induce | --> | Stage 2: Steward Review| --> | Stage 3: Certify Pack | --> | Stage 4: Auto-Evolve   |
| (JSON/TTL/SQL/Ossie/  |     | (Roots classes, fixes  |     | (Marks certified=True |     | (Promotes recurring    |
|  Pydantic/Transcripts)|     |  functional/transitive)|     |  in immutable Audit)  |     |  Bucket-2 overflow)    |
+-----------------------+     +------------------------+     +-----------------------+     +------------------------+
```

```python
# Stage 1 & 2: Review and refine draft pack via Ontology Intelligence Steward
refined_pack, review_audit = registrar.review_and_refine_domain_pack(
    payments_pack,
    tenant_id="global_pay",
    auto_register=True,
)

# Stage 3: Certify the pack once verified with `check_answerable`
certified_pack, cert_audit = registrar.certify_domain_pack(
    "payment_processing@2.1.0",
    certified_by="chief_payments_architect",
    tenant_id="global_pay",
)

# Stage 4: Periodically promote high-confidence emergent concepts (support >= 3)
evolution_audit = await service.evolve_app_ontology_from_overflow(
    "global_pay:dispute_resolution_agent",
    min_confidence_to_promote=0.85,
    min_support_count=3,
    registrar=registrar,
)
```
