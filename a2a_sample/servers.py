from __future__ import annotations

import os
from typing import Any

from .config import (
    CRM_AGENTS_DIR,
    ORCHESTRATOR_AGENTS_DIR,
    SRE_AGENTS_DIR,
    get_a2a_settings,
)


def build_sre_a2a_app() -> Any:
    os.environ.setdefault("ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS", "1")
    from google.adk.cli.fast_api import get_fast_api_app

    settings = get_a2a_settings()
    return get_fast_api_app(
        agents_dir=str(SRE_AGENTS_DIR),
        web=False,
        a2a=True,
        host=settings.host,
        port=settings.sre_agent_port,
    )


def build_crm_a2a_app() -> Any:
    os.environ.setdefault("ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS", "1")
    from google.adk.cli.fast_api import get_fast_api_app

    settings = get_a2a_settings()
    return get_fast_api_app(
        agents_dir=str(CRM_AGENTS_DIR),
        web=False,
        a2a=True,
        host=settings.host,
        port=settings.crm_agent_port,
    )


def build_orchestrator_a2a_app() -> Any:
    os.environ.setdefault("ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS", "1")
    from google.adk.cli.fast_api import get_fast_api_app

    settings = get_a2a_settings()
    return get_fast_api_app(
        agents_dir=str(ORCHESTRATOR_AGENTS_DIR),
        web=False,
        a2a=True,
        host=settings.host,
        port=settings.orchestrator_port,
    )
