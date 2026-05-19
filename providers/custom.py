"""Generic transports for user-defined custom providers.

Built lazily by :mod:`providers.registry` so importing the registry does not
eager-load these transports. Two protocols are supported, matching the two
transport base classes:

* ``openai_chat``      -> :class:`GenericOpenAIProvider`
* ``anthropic_messages`` -> :class:`GenericAnthropicProvider`
"""

from __future__ import annotations

from typing import Any

from config.custom_providers import CustomProviderRecord
from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import InvalidRequestError
from providers.openai_compat import OpenAIChatTransport

_ANTHROPIC_VERSION = "2023-06-01"


class GenericOpenAIProvider(OpenAIChatTransport):
    """OpenAI-compatible Chat Completions adapter for a custom provider."""

    def __init__(
        self, config: ProviderConfig, *, provider_name: str, base_url: str
    ) -> None:
        super().__init__(
            config,
            provider_name=provider_name,
            base_url=base_url,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        try:
            return build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc


class GenericAnthropicProvider(AnthropicMessagesTransport):
    """Native Anthropic Messages adapter for a custom provider.

    Sends both ``x-api-key`` and ``Authorization: Bearer`` (when a key is set)
    so it works against real Anthropic and Anthropic-compatible gateways.
    """

    def __init__(
        self, config: ProviderConfig, *, provider_name: str, default_base_url: str
    ) -> None:
        super().__init__(
            config,
            provider_name=provider_name,
            default_base_url=default_base_url,
        )

    def _auth_headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {
            "x-api-key": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
        }

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        headers.update(self._auth_headers())
        return headers

    def _model_list_headers(self) -> dict[str, str]:
        headers = self._auth_headers()
        if headers:
            headers["anthropic-version"] = _ANTHROPIC_VERSION
        return headers


def create_custom_provider(
    record: CustomProviderRecord, config: ProviderConfig
) -> BaseProvider:
    """Instantiate the generic transport for ``record``'s protocol."""
    if record.protocol == "anthropic_messages":
        return GenericAnthropicProvider(
            config,
            provider_name=record.provider_id,
            default_base_url=record.base_url,
        )
    return GenericOpenAIProvider(
        config,
        provider_name=record.provider_id,
        base_url=config.base_url or record.base_url,
    )
