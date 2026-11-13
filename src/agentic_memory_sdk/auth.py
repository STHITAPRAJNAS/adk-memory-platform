from __future__ import annotations

import asyncio
import ssl
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class EnterpriseTlsConfig:
    """
    Enterprise TLS / mTLS configuration for cloud-managed databases
    (AWS RDS/Aurora, ElastiCache, MemoryDB, GCP Cloud SQL/AlloyDB, Azure Postgres, Neo4j Aura).
    """

    enabled: bool = True
    verify_hostname: bool = True
    ca_file: str | None = None
    client_cert_file: str | None = None
    client_key_file: str | None = None
    min_tls_version: int = ssl.TLSVersion.TLSv1_2

    def build_ssl_context(self) -> ssl.SSLContext | bool:
        if not self.enabled:
            return False
        ctx = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=self.ca_file,
        )
        ctx.minimum_version = ssl.TLSVersion(self.min_tls_version)
        if self.verify_hostname:
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
        else:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        if self.client_cert_file:
            ctx.load_cert_chain(
                certfile=self.client_cert_file,
                keyfile=self.client_key_file,
            )
        return ctx


@runtime_checkable
class CloudAuthProvider(Protocol):
    """
    Protocol for enterprise cloud database authentication (IAM tokens, Entra ID,
    GCP Workload Identity, Vault dynamic secrets, or rotating Secrets Manager credentials).
    """

    async def get_credentials(self) -> tuple[str, str]:
        """Returns `(username, password_or_token)` with automatic caching & pre-expiry refresh."""
        ...

    async def get_username(self) -> str:
        """Returns the current database principal/username."""
        ...

    async def get_password(self) -> str:
        """
        Returns the current database password or short-lived IAM/OAuth token.
        Directly compatible with `asyncpg.create_pool(..., password=auth_provider.get_password)`.
        """
        ...

    def invalidate(self) -> None:
        """Forces immediate token/secret rotation on next connection attempt."""
        ...


class CachedTokenAuthProviderBase:
    """
    Base class providing concurrency-safe token caching and proactive pre-expiry rotation
    for short-lived cloud IAM tokens (typically 15–60 minutes TTL).
    """

    def __init__(
        self,
        username: str,
        token_ttl_sec: int = 900,
        refresh_margin_sec: int = 120,
    ):
        self._username = username
        self.token_ttl_sec = max(10, token_ttl_sec)
        self.refresh_margin_sec = min(refresh_margin_sec, self.token_ttl_sec // 2)
        self._cached_username: str | None = None
        self._cached_token: str | None = None
        self._expires_at: float = 0.0
        self._refresh_count: int = 0
        self._lock = asyncio.Lock()

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    def invalidate(self) -> None:
        self._cached_token = None
        self._expires_at = 0.0

    async def _mint_credentials(self) -> tuple[str, str]:
        raise NotImplementedError

    async def get_credentials(self) -> tuple[str, str]:
        now = time.monotonic()
        if (
            self._cached_token is not None
            and self._cached_username is not None
            and now < (self._expires_at - self.refresh_margin_sec)
        ):
            return self._cached_username, self._cached_token

        async with self._lock:
            now = time.monotonic()
            if (
                self._cached_token is not None
                and self._cached_username is not None
                and now < (self._expires_at - self.refresh_margin_sec)
            ):
                return self._cached_username, self._cached_token

            user, token = await self._mint_credentials()
            self._cached_username = user
            self._cached_token = token
            self._expires_at = time.monotonic() + self.token_ttl_sec
            self._refresh_count += 1
            return user, token

    async def get_username(self) -> str:
        user, _ = await self.get_credentials()
        return user

    async def get_password(self) -> str:
        _, token = await self.get_credentials()
        return token


class StaticOrEnvAuthProvider(CachedTokenAuthProviderBase):
    """
    Static or environment-backed credentials provider implementing `CloudAuthProvider`.
    """

    def __init__(self, username: str, password: str):
        super().__init__(username=username, token_ttl_sec=86400 * 365, refresh_margin_sec=0)
        self._password = password

    async def _mint_credentials(self) -> tuple[str, str]:
        return self._username, self._password


class AwsIamDatabaseAuthProvider(CachedTokenAuthProviderBase):
    """
    Enterprise AWS IAM Authentication Provider for:
      - Amazon RDS / Aurora PostgreSQL (`service="rds"`, uses `generate_db_auth_token`)
      - Amazon ElastiCache / MemoryDB for Redis (`service="elasticache"` or `"memorydb"`, SigV4 presigned token)
    Supports optional cross-account IAM Role assumption (`assume_role_arn` + `external_id`)
    and custom token generators for offline/unit testing.
    """

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        region: str = "us-east-1",
        service: str = "rds",
        cluster_name: str | None = None,
        assume_role_arn: str | None = None,
        external_id: str | None = None,
        token_ttl_sec: int = 900,
        refresh_margin_sec: int = 120,
        custom_token_generator: Callable[[], str | Awaitable[str]] | None = None,
    ):
        super().__init__(
            username=username,
            token_ttl_sec=token_ttl_sec,
            refresh_margin_sec=refresh_margin_sec,
        )
        self.host = host
        self.port = port
        self.region = region
        self.service = service.lower()
        self.cluster_name = cluster_name or host
        self.assume_role_arn = assume_role_arn
        self.external_id = external_id
        self.custom_token_generator = custom_token_generator

    def get_boto3_session(self) -> Any:
        import boto3

        if not self.assume_role_arn:
            return boto3.Session(region_name=self.region)

        sts = boto3.client("sts", region_name=self.region)
        assume_kwargs: dict[str, Any] = {
            "RoleArn": self.assume_role_arn,
            "RoleSessionName": "adk-memory-platform-db-auth",
            "DurationSeconds": max(900, self.token_ttl_sec),
        }
        if self.external_id:
            assume_kwargs["ExternalId"] = self.external_id
        creds = sts.assume_role(**assume_kwargs)["Credentials"]
        return boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=self.region,
        )

    def _generate_sync(self) -> str:
        session = self.get_boto3_session()
        if self.service == "rds":
            rds = session.client("rds", region_name=self.region)
            return rds.generate_db_auth_token(
                DBHostname=self.host,
                Port=self.port,
                DBUsername=self. _username,
                Region=self.region,
            )
        # ElastiCache / MemoryDB IAM auth uses a SigV4 presigned URL token
        from botocore.awsrequest import AWSRequest
        from botocore.signers import RequestSigner

        credentials = session.get_credentials().get_frozen_credentials()
        signer = RequestSigner(
            service_name=self.service,
            region_name=self.region,
            signing_name=self.service,
            signature_version="v4",
            credentials=credentials,
            event_emitter=session.events,
        )
        url = signer.generate_presigned_url(
            request_dict={
                "url_path": "/",
                "query_string": {"Action": "connect", "User": self._username},
                "headers": {"Host": self.cluster_name},
                "method": "GET",
                "body": b"",
            },
            region_name=self.region,
            expires_in=self.token_ttl_sec,
            operation_name="connect",
        )
        return url.replace("https://", "").replace("http://", "")

    async def _mint_credentials(self) -> tuple[str, str]:
        if self.custom_token_generator is not None:
            res = self.custom_token_generator()
            token = await res if asyncio.iscoroutine(res) else str(res)
            return self._username, token
        token = await asyncio.to_thread(self._generate_sync)
        return self._username, token


class GcpCloudSqlIamAuthProvider(CachedTokenAuthProviderBase):
    """
    Enterprise Google Cloud IAM Database Authentication Provider for:
      - Cloud SQL for PostgreSQL / AlloyDB (`https://www.googleapis.com/auth/cloud-platform`)
      - Cloud Memorystore for Redis IAM Auth
    Uses Application Default Credentials (ADC) / Workload Identity Federation.
    """

    def __init__(
        self,
        username: str,
        scopes: list[str] | None = None,
        target_service_account: str | None = None,
        token_ttl_sec: int = 3300,
        refresh_margin_sec: int = 300,
        custom_token_generator: Callable[[], str | Awaitable[str]] | None = None,
    ):
        super().__init__(
            username=username,
            token_ttl_sec=token_ttl_sec,
            refresh_margin_sec=refresh_margin_sec,
        )
        self.scopes = scopes or ["https://www.googleapis.com/auth/cloud-platform"]
        self.target_service_account = target_service_account
        self.custom_token_generator = custom_token_generator

    def _generate_sync(self) -> str:
        import google.auth  # type: ignore
        from google.auth.transport.requests import Request  # type: ignore

        creds, _ = google.auth.default(scopes=self.scopes)
        if self.target_service_account:
            from google.auth import impersonated_credentials  # type: ignore

            creds = impersonated_credentials.Credentials(
                source_credentials=creds,
                target_principal=self.target_service_account,
                target_scopes=self.scopes,
                lifetime=self.token_ttl_sec,
            )
        creds.refresh(Request())
        return str(creds.token)

    async def _mint_credentials(self) -> tuple[str, str]:
        if self.custom_token_generator is not None:
            res = self.custom_token_generator()
            token = await res if asyncio.iscoroutine(res) else str(res)
            return self._username, token
        token = await asyncio.to_thread(self._generate_sync)
        return self._username, token


class AzureEntraDatabaseAuthProvider(CachedTokenAuthProviderBase):
    """
    Enterprise Microsoft Entra ID (Azure AD / Managed Identity) Authentication Provider for:
      - Azure Database for PostgreSQL Flexible Server (`scope="https://ossrdbms-aad.database.windows.net/.default"`)
      - Azure Managed Redis / Azure Cache for Redis (`scope="https://redis.azure.com/.default"`)
    """

    POSTGRES_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"
    REDIS_SCOPE = "https://redis.azure.com/.default"

    def __init__(
        self,
        username: str,
        scope: str = POSTGRES_SCOPE,
        managed_identity_client_id: str | None = None,
        token_ttl_sec: int = 3300,
        refresh_margin_sec: int = 300,
        custom_token_generator: Callable[[], str | Awaitable[str]] | None = None,
    ):
        super().__init__(
            username=username,
            token_ttl_sec=token_ttl_sec,
            refresh_margin_sec=refresh_margin_sec,
        )
        self.scope = scope
        self.managed_identity_client_id = managed_identity_client_id
        self.custom_token_generator = custom_token_generator

    def _generate_sync(self) -> str:
        from azure.identity import DefaultAzureCredential  # type: ignore

        cred = DefaultAzureCredential(
            managed_identity_client_id=self.managed_identity_client_id
        )
        token_obj = cred.get_token(self.scope)
        return str(token_obj.token)

    async def _mint_credentials(self) -> tuple[str, str]:
        if self.custom_token_generator is not None:
            res = self.custom_token_generator()
            token = await res if asyncio.iscoroutine(res) else str(res)
            return self._username, token
        token = await asyncio.to_thread(self._generate_sync)
        return self._username, token


class VaultOrSecretManagerAuthProvider(CachedTokenAuthProviderBase):
    """
    Dynamic Secret Rotation Provider for HashiCorp Vault Database Secrets Engine,
    AWS Secrets Manager, or GCP Secret Manager.
    Accepts an async or sync callable `secret_fetcher` returning `(username, password)`.
    """

    def __init__(
        self,
        secret_fetcher: Callable[[], tuple[str, str] | Awaitable[tuple[str, str]]],
        token_ttl_sec: int = 1800,
        refresh_margin_sec: int = 180,
    ):
        super().__init__(
            username="dynamic",
            token_ttl_sec=token_ttl_sec,
            refresh_margin_sec=refresh_margin_sec,
        )
        self.secret_fetcher = secret_fetcher

    async def _mint_credentials(self) -> tuple[str, str]:
        res = self.secret_fetcher()
        if asyncio.iscoroutine(res):
            user, pwd = await res
        else:
            user, pwd = res  # type: ignore
        return str(user), str(pwd)
