"""Custom providers wired through registry, settings, and routing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from api.model_router import ModelRouter
from config import custom_providers as cp
from config.settings import Settings
from providers.custom import (
    GenericAnthropicProvider,
    GenericOpenAIProvider,
    create_custom_provider,
)
from providers.exceptions import UnknownProviderTypeError
from providers.registry import build_custom_provider_config, create_provider


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store_path = tmp_path / ".gdec" / "custom_providers.json"
    monkeypatch.setattr(cp, "custom_providers_path", lambda: store_path)
    cp.invalidate_cache()
    yield
    cp.invalidate_cache()


def _settings_mock(**overrides):
    mock = MagicMock()
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.provider_max_concurrency = 5
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 10.0
    mock.enable_model_thinking = True
    mock.log_raw_sse_events = False
    mock.log_api_error_tracebacks = False
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def _register(**kwargs) -> cp.CustomProviderRecord:
    record = cp.CustomProviderRecord(**kwargs)
    cp.save_custom_providers([record])
    return record


def test_build_custom_provider_config_maps_record_fields() -> None:
    record = cp.CustomProviderRecord(
        provider_id="mine",
        display_name="Mine",
        base_url="https://api.mine/v1",
        api_key="sk-1",
    )
    config = build_custom_provider_config(record, _settings_mock())

    assert config.api_key == "sk-1"
    assert config.base_url == "https://api.mine/v1"
    assert config.proxy == ""
    assert config.max_concurrency == 5


def test_create_custom_provider_picks_transport_by_protocol() -> None:
    openai_rec = cp.CustomProviderRecord(
        provider_id="o", display_name="O", base_url="https://o/v1"
    )
    anth_rec = cp.CustomProviderRecord(
        provider_id="a",
        display_name="A",
        base_url="https://a",
        protocol="anthropic_messages",
    )
    with patch("providers.openai_compat.AsyncOpenAI"), patch("httpx.AsyncClient"):
        op = create_custom_provider(
            openai_rec, build_custom_provider_config(openai_rec, _settings_mock())
        )
        ap = create_custom_provider(
            anth_rec, build_custom_provider_config(anth_rec, _settings_mock())
        )

    assert isinstance(op, GenericOpenAIProvider)
    assert isinstance(ap, GenericAnthropicProvider)


def test_generic_anthropic_headers_include_auth_when_key_set() -> None:
    rec = cp.CustomProviderRecord(
        provider_id="a",
        display_name="A",
        base_url="https://a",
        api_key="sk-xyz",
        protocol="anthropic_messages",
    )
    with patch("httpx.AsyncClient"):
        provider = create_custom_provider(
            rec, build_custom_provider_config(rec, _settings_mock())
        )
    assert isinstance(provider, GenericAnthropicProvider)
    headers = provider._request_headers()
    assert headers["x-api-key"] == "sk-xyz"
    assert headers["Authorization"] == "Bearer sk-xyz"
    assert headers["anthropic-version"]


def test_generic_anthropic_headers_omit_auth_without_key() -> None:
    rec = cp.CustomProviderRecord(
        provider_id="a",
        display_name="A",
        base_url="https://a",
        protocol="anthropic_messages",
    )
    with patch("httpx.AsyncClient"):
        provider = create_custom_provider(
            rec, build_custom_provider_config(rec, _settings_mock())
        )
    assert isinstance(provider, GenericAnthropicProvider)
    headers = provider._request_headers()
    assert "x-api-key" not in headers
    assert "Authorization" not in headers


def test_registry_create_provider_resolves_custom() -> None:
    _register(
        provider_id="mycustom",
        display_name="My Custom",
        base_url="https://api.mycustom/v1",
        api_key="k",
    )
    with patch("providers.openai_compat.AsyncOpenAI"), patch("httpx.AsyncClient"):
        provider = create_provider("mycustom", _settings_mock())

    assert isinstance(provider, GenericOpenAIProvider)


def test_registry_unknown_provider_still_raises() -> None:
    with pytest.raises(UnknownProviderTypeError, match="Unknown provider_type"):
        create_provider("does-not-exist", _settings_mock())


def test_settings_accepts_registered_custom_model() -> None:
    _register(provider_id="mycustom", display_name="My Custom", base_url="https://m/v1")
    settings = Settings(model="mycustom/some-model")
    assert settings.provider_type == "mycustom"
    assert settings.model_name == "some-model"


def test_settings_rejects_unregistered_provider() -> None:
    with pytest.raises(ValueError, match="Invalid provider"):
        Settings(model="bogus-provider/x")


def test_model_router_routes_custom_provider_directly() -> None:
    _register(provider_id="mycustom", display_name="My Custom", base_url="https://m/v1")
    settings = Settings(model="mycustom/fallback")
    resolved = ModelRouter(settings).resolve("mycustom/some-model")

    assert resolved.provider_id == "mycustom"
    assert resolved.provider_model == "some-model"
