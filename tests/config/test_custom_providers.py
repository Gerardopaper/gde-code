"""Tests for the JSON-backed custom provider store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import custom_providers as cp


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the store at a temp file and clear the parse cache around each test."""
    store_path = tmp_path / ".gdec" / "custom_providers.json"
    monkeypatch.setattr(cp, "custom_providers_path", lambda: store_path)
    cp.invalidate_cache()
    yield store_path
    cp.invalidate_cache()


def test_record_accepts_valid_fields() -> None:
    rec = cp.CustomProviderRecord(
        provider_id="my-llm_1",
        display_name="My LLM",
        base_url="https://api.example.com/v1",
        api_key="sk-abc",
        protocol="anthropic_messages",
    )
    assert rec.provider_id == "my-llm_1"
    assert rec.protocol == "anthropic_messages"


def test_api_key_is_optional_and_defaults_openai_protocol() -> None:
    rec = cp.CustomProviderRecord(
        provider_id="local",
        display_name="Local",
        base_url="http://localhost:1234/v1",
    )
    assert rec.api_key == ""
    assert rec.protocol == "openai_chat"


@pytest.mark.parametrize("bad_id", ["Has-Caps", "with space", "dot.id", "slash/id", ""])
def test_invalid_provider_id_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        cp.CustomProviderRecord(
            provider_id=bad_id, display_name="X", base_url="http://x"
        )


def test_provider_id_cannot_collide_with_builtin() -> None:
    with pytest.raises(ValidationError, match="collides with a built-in"):
        cp.CustomProviderRecord(
            provider_id="nvidia_nim", display_name="X", base_url="http://x"
        )


def test_blank_display_name_or_base_url_rejected() -> None:
    with pytest.raises(ValidationError):
        cp.CustomProviderRecord(
            provider_id="ok", display_name="  ", base_url="http://x"
        )
    with pytest.raises(ValidationError):
        cp.CustomProviderRecord(provider_id="ok", display_name="X", base_url="")


def test_save_then_load_round_trip(_isolated_store: Path) -> None:
    records = [
        cp.CustomProviderRecord(
            provider_id="alpha",
            display_name="Alpha",
            base_url="https://a.example/v1",
            api_key="k",
            protocol="openai_chat",
        )
    ]
    cp.save_custom_providers(records)

    loaded = cp.load_custom_providers()
    assert set(loaded) == {"alpha"}
    assert loaded["alpha"].base_url == "https://a.example/v1"
    assert _isolated_store.is_file()


def test_missing_file_yields_empty_mapping() -> None:
    assert cp.load_custom_providers() == {}
    assert cp.custom_provider_ids() == ()
    assert cp.get_custom_provider("nope") is None


def test_malformed_json_is_tolerated(_isolated_store: Path) -> None:
    _isolated_store.parent.mkdir(parents=True, exist_ok=True)
    _isolated_store.write_text("{ not json", encoding="utf-8")
    cp.invalidate_cache()
    assert cp.load_custom_providers() == {}


def test_invalid_record_in_file_is_dropped(_isolated_store: Path) -> None:
    _isolated_store.parent.mkdir(parents=True, exist_ok=True)
    _isolated_store.write_text(
        json.dumps({"providers": [{"provider_id": "BAD CAPS"}]}), encoding="utf-8"
    )
    cp.invalidate_cache()
    assert cp.load_custom_providers() == {}


def test_is_known_provider_covers_builtin_and_custom() -> None:
    assert cp.is_known_provider("nvidia_nim") is True
    assert cp.is_known_provider("totally-unknown") is False

    cp.save_custom_providers(
        [
            cp.CustomProviderRecord(
                provider_id="mine", display_name="Mine", base_url="http://m"
            )
        ]
    )
    assert cp.is_known_provider("mine") is True
