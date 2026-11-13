from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agentic_memory_sdk import RemoteCompositeMemoryService

REPO_ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR_AGENTS_DIR = Path(__file__).resolve().parent / "orchestrator"
SRE_AGENTS_DIR = Path(__file__).resolve().parent / "remote_agents_sre"
CRM_AGENTS_DIR = Path(__file__).resolve().parent / "remote_agents_crm"


@dataclass(frozen=True)
class A2ASampleSettings:
    host: str = "127.0.0.1"
    memory_api_port: int = 8095
    orchestrator_port: int = 8090
    sre_agent_port: int = 8091
    crm_agent_port: int = 8092
    tenant_id: str = "t1"
    use_fake_llm: bool = True
    model_name: str = "gemini-2.5-flash"

    @property
    def memory_api_url(self) -> str:
        return os.environ.get(
            "MEMORY_API_URL", f"http://{self.host}:{self.memory_api_port}"
        )

    @property
    def orchestrator_base_url(self) -> str:
        return f"http://{self.host}:{self.orchestrator_port}"

    @property
    def sre_agent_base_url(self) -> str:
        return f"http://{self.host}:{self.sre_agent_port}"

    @property
    def crm_agent_base_url(self) -> str:
        return f"http://{self.host}:{self.crm_agent_port}"

    @property
    def orchestrator_rpc_url(self) -> str:
        return f"{self.orchestrator_base_url}/a2a/platform_concierge"

    @property
    def orchestrator_card_url(self) -> str:
        return f"{self.orchestrator_rpc_url}/.well-known/agent-card.json"

    @property
    def sre_agent_rpc_url(self) -> str:
        return f"{self.sre_agent_base_url}/a2a/sre_specialist"

    @property
    def sre_agent_card_url(self) -> str:
        return f"{self.sre_agent_rpc_url}/.well-known/agent-card.json"

    @property
    def crm_agent_rpc_url(self) -> str:
        return f"{self.crm_agent_base_url}/a2a/crm_specialist"

    @property
    def crm_agent_card_url(self) -> str:
        return f"{self.crm_agent_rpc_url}/.well-known/agent-card.json"


def get_a2a_settings() -> A2ASampleSettings:
    live = os.environ.get("MEMORY_LIVE_GEMINI", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return A2ASampleSettings(use_fake_llm=not live)


def get_remote_memory_client() -> RemoteCompositeMemoryService:
    settings = get_a2a_settings()
    return RemoteCompositeMemoryService(
        base_url=settings.memory_api_url, timeout_sec=15.0
    )
