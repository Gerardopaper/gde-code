"""Generic transports for user-defined custom providers.

Built lazily by :mod:`providers.registry` so importing the registry does not
eager-load these transports. Two protocols are supported, matching the two
transport base classes:

* ``openai_chat``      -> :class:`GenericOpenAIProvider`
* ``anthropic_messages`` -> :class:`GenericAnthropicProvider`

A custom provider may carry a manual model catalog. When present, the generic
provider uses it for model listing instead of calling the upstream ``/models``
endpoint (some private gateways don't expose one).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from config.custom_providers import CustomProviderModel, CustomProviderRecord
from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import InvalidRequestError
from providers.model_listing import ProviderModelInfo
from providers.openai_compat import OpenAIChatTransport

_ANTHROPIC_VERSION = "2023-06-01"


def _manual_model_infos(
    models: Iterable[CustomProviderModel],
) -> frozenset[ProviderModelInfo]:
    return frozenset(
        ProviderModelInfo(model_id=model.model_id, supports_thinking=None)
        for model in models
    )


class GenericOpenAIProvider(OpenAIChatTransport):
    """OpenAI-compatible Chat Completions adapter for a custom provider."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        base_url: str,
        models: Iterable[CustomProviderModel] = (),
    ) -> None:
        super().__init__(
            config,
            provider_name=provider_name,
            base_url=base_url,
            api_key=config.api_key,
        )
        self._manual_models: tuple[CustomProviderModel, ...] = tuple(models)

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

    async def list_model_ids(self) -> frozenset[str]:
        if self._manual_models:
            return frozenset(model.model_id for model in self._manual_models)
        return await super().list_model_ids()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        if self._manual_models:
            return _manual_model_infos(self._manual_models)
        return await super().list_model_infos()

    async def test_chat(self, model_id: str) -> dict[str, Any]:
        """Send a single-token chat completion to verify connectivity for a model."""
        response = await self._client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            stream=False,
        )
        return {
            "ok": True,
            "model": getattr(response, "model", model_id),
            "response_id": getattr(response, "id", None),
        }


class GenericAnthropicProvider(AnthropicMessagesTransport):
    """Native Anthropic Messages adapter for a custom provider.

    Sends both ``x-api-key`` and ``Authorization: Bearer`` (when a key is set)
    so it works against real Anthropic and Anthropic-compatible gateways.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        default_base_url: str,
        models: Iterable[CustomProviderModel] = (),
    ) -> None:
        super().__init__(
            config,
            provider_name=provider_name,
            default_base_url=default_base_url,
        )
        self._manual_models: tuple[CustomProviderModel, ...] = tuple(models)

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

    async def list_model_ids(self) -> frozenset[str]:
        if self._manual_models:
            return frozenset(model.model_id for model in self._manual_models)
        return await super().list_model_ids()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        if self._manual_models:
            return _manual_model_infos(self._manual_models)
        return await super().list_model_infos()

    async def test_chat(self, model_id: str) -> dict[str, Any]:
        """Send a single-token /messages call to verify connectivity for a model."""
        body = {
            "model": model_id,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        headers.update(self._auth_headers())
        response = await self._client.post("/messages", json=body, headers=headers)
        response.raise_for_status()
        return {
            "ok": True,
            "model": model_id,
            "http_status": response.status_code,
        }


def create_custom_provider(
    record: CustomProviderRecord, config: ProviderConfig
) -> BaseProvider:
    """Instantiate the generic transport for ``record``'s protocol."""
    if record.protocol == "anthropic_messages":
        return GenericAnthropicProvider(
            config,
            provider_name=record.provider_id,
            default_base_url=record.base_url,
            models=record.models,
        )
    return GenericOpenAIProvider(
        config,
        provider_name=record.provider_id,
        base_url=config.base_url or record.base_url,
        models=record.models,
    )
