"""
Platform Concierge Orchestrator (A2A Server + Client on :8090).
Routes SRE requests over A2A to `sre_specialist` (:8091) and CRM requests over A2A to `crm_specialist` (:8092).
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.apps import App, ResumabilityConfig

from a2a_sample.config import get_a2a_settings
from a2a_sample.scripted_llm import build_concierge_model


class TransferableRemoteA2aAgent(RemoteA2aAgent):
    disallow_transfer_to_parent: bool = False
    disallow_transfer_to_peers: bool = False


def build_remote_sre_agent() -> TransferableRemoteA2aAgent:
    settings = get_a2a_settings()
    return TransferableRemoteA2aAgent(
        name="sre_specialist",
        description=(
            "Remote SRE & Cloud Infrastructure specialist reached over A2A. "
            "Handles microservice deployments, regional failovers, and database dependencies."
        ),
        agent_card=settings.sre_agent_card_url,
        timeout=60.0,
    )


def build_remote_crm_agent() -> TransferableRemoteA2aAgent:
    settings = get_a2a_settings()
    return TransferableRemoteA2aAgent(
        name="crm_specialist",
        description=(
            "Remote Customer CRM & Subscription specialist reached over A2A. "
            "Handles customer accounts, subscription tiers, and billing inquiries."
        ),
        agent_card=settings.crm_agent_card_url,
        timeout=60.0,
    )


def build_agent() -> LlmAgent:
    settings = get_a2a_settings()
    model = build_concierge_model() if settings.use_fake_llm else settings.model_name
    return LlmAgent(
        name="platform_concierge",
        model=model,
        description="Front-door A2A orchestrator routing requests to domain-specialized remote A2A agents.",
        instruction=(
            "You are the Platform Concierge. Transfer SRE/infrastructure questions to `sre_specialist` "
            "and Customer/CRM questions to `crm_specialist` over A2A."
        ),
        sub_agents=[build_remote_sre_agent(), build_remote_crm_agent()],
    )


root_agent = build_agent()
app = App(
    name="platform_concierge",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
