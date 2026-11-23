# `adk-memory-platform` Engineering, Security & Release Rules

## 1. Documentation Must Always Match Codebase Actuals
* Before every commit or release, audit `README.md`, `CHANGELOG.md`, and `docs/*.md` directly against the actual codebase:
  * CLI flags and subcommands (`--backend memory|docker`, `migrate-db`, `--enable-maintenance-worker` — never stale `--mode` flags).
  * REST routes (`/v1/lineage/purge`, `/ui`, `/v1/memory/items`, `/v1/memory/revisions`, `/v1/memory/blocks`, `/v1/metrics`, `/metrics`) and MCP tool/resource/prompt counts.
  * Exact pytest test counts and honest benchmark labels (never present a synthetic deterministic check as an official external dataset like LoCoMo or LongMemEval).
  * Document any breaking changes (such as removed CLI flags) explicitly under `### Breaking Changes` in `CHANGELOG.md`.

## 2. Defense-in-Depth AuthZ & Bring-Your-Own-Auth (BYOA)
* **Separate AuthN from AuthZ**: Treat AuthN as pluggable BYOA (`custom_request_authenticator` for API Gateways/Istio/Kong headers, `custom_token_verifier` for OIDC/JWKS), while enforcing AuthZ and tenant/app/user isolation **inside the service layer** (`CompositeMemoryService` via `ContextVar` `AuthPrincipal`), never in HTTP middleware alone.
* **Zero Cross-Tenant Leaks**:
  * When authenticated, a bare `app_name` (no `tenant:` prefix) must resolve to the caller's own `tenant_id`, never `default_tenant_id` (`t1`).
  * Unfiltered list endpoints (e.g., `GET /v1/memory/revisions`) must force `tenant_filter = principal.tenant_id` and respect `allowed_users`.
  * Validate query params, path params, JSON body fields, MCP tool `arguments` (`/v1/mcp/call`, `/mcp/http`, `/mcp/sse`), and MCP `resources/read` URIs independently; reject conflicting query-vs-body parameters (`403`).
  * Global cross-tenant routes (`/v1/maintenance/run`, `/v1/outbox/flush`, MCP `run_memory_maintenance`) require `PLATFORM_ADMIN` with `tenant_id == "*"`.
  * `READER` principals must be able to list MCP tools/resources/prompts and call read-only MCP tools over `/mcp/http` and `/mcp/sse`.
* **Secrets & Public Bind Hygiene**:
  * Never accept secrets (`jwt_secret`, API keys) as CLI arguments (visible in `ps`/`top`); read strictly from environment variables.
  * Require `exp` claim, non-empty `tenant_id` claim, and `>= 32`-byte HS256 JWT secrets.
  * Refuse to bind `adk-memory serve-api` to `0.0.0.0` without authentication enabled unless `--allow-insecure-public-bind` is explicitly passed.
* **Complete GDPR Purge**: `purge_user_lineage()` must erase user data across **all** stores — Episodic, Seed, Bi-Temporal Graph, Working Memory Blocks, and Memory Revision History (both in-memory and PostgreSQL).
* **Bounded / Distributed Rate Limiting**: API rate limiters must evict stale keys with a hard `max_keys` cap and prefer Redis atomic counters when Redis is configured.

## 3. PostgreSQL Row-Level Security (RLS) & Schema Hygiene
* **Always Test Runtime Flows as Non-Superuser (`platform_app_user`)**: Superuser `platform_admin` bypasses RLS and hides missing `set_config` bugs. Reserve `platform_admin` for `init-db` / `migrate-db` DDL only.
* **Fail-Closed RLS Everywhere**: Every RLS policy (`episodic_memories`, `graph_seed_index`, `memory_revisions`, `working_memory_blocks`) must fail closed (return 0 rows) when `app.tenant_id` is unset.
* **Transaction-Scoped RLS Context**: Every SQL query touching RLS-protected tables must run inside a transaction setting `app.tenant_id` (and `app.app_name` / `app.user_id` where required).
* **Tier-0 Auto-Partitioning & Idempotent Migrations**: Auto-provision tenant partitions on first write so unregistered Tier-0 tenants never throw raw `500` partition errors, and keep `docs/sql/001_postgres_rls_and_partitions.sql` idempotent for `adk-memory migrate-db`.

## 4. Verification, Packaging & CI-Gated Releases
* Run `.\.venv\Scripts\ruff.exe check src/ tests/ examples/` and `.\.venv\Scripts\pytest.exe tests/test_composite_memory_platform.py -v` before every commit.
* On Windows, invoke `.\.venv\Scripts\python.exe` / `pytest.exe` / `ruff.exe` directly (avoiding `uv run` corporate TLS proxy errors) and run Docker Compose + live Docker tests in a single continuous PowerShell session.
* Keep FastAPI version dynamic via `importlib.metadata`, keep `[project.optional-dependencies].all` complete, and ensure `.github/workflows/release.yml` always gates `build` and `publish-pypi` on a passing `verify` job.
