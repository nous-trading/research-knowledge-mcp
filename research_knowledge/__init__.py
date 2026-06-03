"""Research Knowledge MCP — academic paper RAG.

Public API:
    - ``set_oat_provider`` — optionally enable Claude Code OAuth-token auth.
    - ``get_anthropic_client`` / ``get_async_anthropic_client`` — resolved clients.
"""

from .auth import (
    get_anthropic_client,
    get_async_anthropic_client,
    set_oat_provider,
)

__all__ = [
    "get_anthropic_client",
    "get_async_anthropic_client",
    "set_oat_provider",
]

__version__ = "0.1.0"
