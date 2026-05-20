"""User-defined custom provider store (JSON-backed, dynamic at runtime).

Kept free of transport/protocol imports: this module may only depend on
``config`` and pydantic (see ``test_config_does_not_import_non_config_packages``).
Custom providers are a separate runtime layer and are intentionally *not*
merged into ``PROVIDER_CATALOG`` / ``SUPPORTED_PROVIDER_IDS`` (single-source
contract).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .paths import custom_providers_path
from .provider_ids import SUPPORTED_PROVIDER_IDS

CustomProviderProtocol = Literal["openai_chat", "anthropic_messages"]

PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._/:\-]+$")


class CustomProviderModel(BaseModel):
    """A model entry under a custom provider (manual model catalog)."""

    model_id: str
    display_name: str

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        value = value.strip()
        if not MODEL_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "model_id must use only letters, digits, hyphens, underscores, "
                "dots, colons, or slashes"
            )
        return value

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("display_name must not be empty")
        return value


class CustomProviderRecord(BaseModel):
    """One user-defined provider.

    ``provider_id`` is the immutable routing prefix (``provider_id/model``).
    ``api_key`` is optional (local or keyless gateways).
    """

    provider_id: str
    display_name: str
    base_url: str
    api_key: str = ""
    protocol: CustomProviderProtocol = "openai_chat"
    models: list[CustomProviderModel] = Field(default_factory=list)
    bypass_system_proxy: bool = False
    verify_tls: bool = True

    @model_validator(mode="after")
    def _dedupe_models(self) -> CustomProviderRecord:
        seen: dict[str, CustomProviderModel] = {}
        for model in self.models:
            seen[model.model_id] = model
        self.models = list(seen.values())
        return self

    @field_validator("provider_id")
    @classmethod
    def _validate_provider_id(cls, value: str) -> str:
        value = value.strip()
        if not PROVIDER_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "provider_id must use only lowercase letters, digits, hyphens, "
                "and underscores"
            )
        if value in SUPPORTED_PROVIDER_IDS:
            raise ValueError(f"provider_id '{value}' collides with a built-in provider")
        return value

    @field_validator("display_name", "base_url")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("api_key")
    @classmethod
    def _strip_api_key(cls, value: str) -> str:
        return value.strip()


class CustomProviderStore(BaseModel):
    """On-disk shape: a list of provider records."""

    providers: list[CustomProviderRecord] = Field(default_factory=list)


def _records_from_path(
    path_str: str, _mtime_ns: int
) -> dict[str, CustomProviderRecord]:
    """Parse the store file. ``_mtime_ns`` only participates in cache keying."""
    path = Path(path_str)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    try:
        store = CustomProviderStore.model_validate(
            data if isinstance(data, dict) else {"providers": data}
        )
    except ValidationError:
        return {}
    return {record.provider_id: record for record in store.providers}


@lru_cache(maxsize=8)
def _cached_records(path_str: str, mtime_ns: int) -> dict[str, CustomProviderRecord]:
    return _records_from_path(path_str, mtime_ns)


def load_custom_providers() -> dict[str, CustomProviderRecord]:
    """Return ``{provider_id: record}`` for all stored custom providers.

    Missing or malformed files yield an empty mapping (best effort). Results are
    cached by file mtime; call :func:`invalidate_cache` after writing.
    """
    path = custom_providers_path()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}
    return dict(_cached_records(str(path), mtime_ns))


def invalidate_cache() -> None:
    """Drop the mtime-keyed parse cache (call after writing the store)."""
    _cached_records.cache_clear()


def save_custom_providers(records: list[CustomProviderRecord]) -> Path:
    """Atomically write the custom provider store and invalidate the cache."""
    path = custom_providers_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = CustomProviderStore(providers=records).model_dump()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)
    invalidate_cache()
    return path


def custom_provider_ids() -> tuple[str, ...]:
    """Return stored custom provider ids."""
    return tuple(load_custom_providers())


def get_custom_provider(provider_id: str) -> CustomProviderRecord | None:
    """Return one custom provider record, or ``None``."""
    return load_custom_providers().get(provider_id)


def is_known_provider(provider_id: str) -> bool:
    """Whether ``provider_id`` is a built-in or a stored custom provider."""
    return (
        provider_id in SUPPORTED_PROVIDER_IDS or provider_id in load_custom_providers()
    )
