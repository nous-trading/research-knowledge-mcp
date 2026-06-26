"""Anthropic client resolution.

Resolves an Anthropic client from whatever credentials are available, in
priority order:

1. ``ANTHROPIC_API_KEY`` environment variable — the standard path for most
   users. Get one at https://console.anthropic.com/.
2. A Claude Code OAuth (OAT) token, if discoverable. This lets the library run
   under a Claude Code subscription without a separate API key. OAT discovery is
   delegated to an optional provider injected by the host application (see
   ``set_oat_provider``); when no provider is registered, OAT is simply skipped.

This module deliberately has no dependency on any host project. The OAT path is
opt-in via dependency injection so the package stays standalone and reusable.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Protocol

import anthropic

logger = logging.getLogger(__name__)


class OATProvider(Protocol):
    """Minimal interface a host app can register to enable OAT auth.

    Any object exposing these three members works — e.g. the host project's
    ``OATTokenProvider``. Kept structural so this package never imports the host.
    """

    @property
    def access_token(self) -> str | None: ...

    def get_anthropic_client(self) -> anthropic.Anthropic: ...

    def get_async_anthropic_client(self) -> anthropic.AsyncAnthropic: ...


# Optional, host-injected OAT provider factory. Stays None for standalone use.
_oat_provider_factory: Callable[[], OATProvider] | None = None


def set_oat_provider(factory: Callable[[], OATProvider]) -> None:
    """Register a factory that builds an OAT provider (optional).

    Host applications that want OAuth-token auth call this once at startup,
    passing a callable that returns an object satisfying :class:`OATProvider`.
    Standalone users never call this and fall back to ``ANTHROPIC_API_KEY``.
    """
    global _oat_provider_factory
    _oat_provider_factory = factory


def _resolve_oat_provider() -> OATProvider | None:
    if _oat_provider_factory is None:
        return None
    try:
        provider = _oat_provider_factory()
    except Exception as e:  # provider construction is host code — never fatal here
        logger.warning("OAT provider factory failed, falling back to API key: %s", e)
        return None
    return provider if provider.access_token else None


def get_anthropic_client() -> anthropic.Anthropic:
    """Return a synchronous Anthropic client from available credentials."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()
    provider = _resolve_oat_provider()
    if provider is not None:
        return provider.get_anthropic_client()
    raise RuntimeError(
        "No Anthropic credentials found. Set ANTHROPIC_API_KEY "
        "(https://console.anthropic.com/) or register an OAT provider via "
        "research_knowledge.auth.set_oat_provider()."
    )


def get_async_anthropic_client() -> anthropic.AsyncAnthropic:
    """Return an asynchronous Anthropic client from available credentials."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return anthropic.AsyncAnthropic()
    provider = _resolve_oat_provider()
    if provider is not None:
        return provider.get_async_anthropic_client()
    raise RuntimeError(
        "No Anthropic credentials found. Set ANTHROPIC_API_KEY "
        "(https://console.anthropic.com/) or register an OAT provider via "
        "research_knowledge.auth.set_oat_provider()."
    )
