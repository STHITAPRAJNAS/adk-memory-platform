import sys
from agentic_memory_sdk import *  # noqa: F401,F403
from agentic_memory_sdk import __all__  # noqa: F401
from agentic_memory_sdk import (
    adapters,
    adk_integration,
    auth,
    cli,
    governance,
    llm_extractors,
    models,
    ontology_loader,
    ports,
    server,
    service,
)

# Register submodule aliases so `from agentic_composite_memory.<submodule> import ...` works seamlessly
for _sub_name, _sub_mod in (
    ("adapters", adapters),
    ("adk_integration", adk_integration),
    ("auth", auth),
    ("cli", cli),
    ("governance", governance),
    ("llm_extractors", llm_extractors),
    ("models", models),
    ("ontology_loader", ontology_loader),
    ("ports", ports),
    ("server", server),
    ("service", service),
):
    sys.modules[f"{__name__}.{_sub_name}"] = _sub_mod
