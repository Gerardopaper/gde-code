"""Admin custom-provider CRUD endpoints and UI wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.admin_config import MASKED_SECRET
from api.app import create_app
from config import custom_providers as cp


def _local_client(app):
    return TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for key in ("MODEL", "GDEC_ENV_FILE", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    cp.invalidate_cache()
    yield
    cp.invalidate_cache()


def _app():
    return create_app(lifespan_enabled=False)


def _payload(**overrides):
    base = {
        "provider_id": "my-llm",
        "display_name": "My LLM",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-secret",
        "protocol": "openai_chat",
    }
    base.update(overrides)
    return base


def test_create_list_and_status_roundtrip():
    client = _local_client(_app())

    created = client.post("/admin/api/custom-providers", json=_payload())
    assert created.status_code == 200
    body = created.json()
    assert body["applied"] is True
    assert body["provider"]["api_key"] == MASKED_SECRET
    assert body["provider"]["has_api_key"] is True

    listed = client.get("/admin/api/custom-providers").json()["providers"]
    assert [p["provider_id"] for p in listed] == ["my-llm"]
    assert listed[0]["api_key"] == MASKED_SECRET

    config = client.get("/admin/api/config").json()
    custom = [p for p in config["provider_status"] if p.get("kind") == "custom"]
    assert custom and custom[0]["provider_id"] == "my-llm"


def test_invalid_provider_id_is_rejected():
    client = _local_client(_app())
    resp = client.post(
        "/admin/api/custom-providers", json=_payload(provider_id="Bad Caps")
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is False
    assert body["errors"]


def test_collision_with_builtin_rejected():
    client = _local_client(_app())
    body = client.post(
        "/admin/api/custom-providers", json=_payload(provider_id="nvidia_nim")
    ).json()
    assert body["applied"] is False


def test_update_preserves_key_when_masked():
    client = _local_client(_app())
    client.post("/admin/api/custom-providers", json=_payload())

    updated = client.post(
        "/admin/api/custom-providers",
        json=_payload(
            display_name="Renamed", api_key=MASKED_SECRET, protocol="anthropic_messages"
        ),
    ).json()
    assert updated["applied"] is True

    record = cp.get_custom_provider("my-llm")
    assert record is not None
    assert record.api_key == "sk-secret"
    assert record.display_name == "Renamed"
    assert record.protocol == "anthropic_messages"


def test_delete_custom_provider():
    client = _local_client(_app())
    client.post("/admin/api/custom-providers", json=_payload())

    deleted = client.delete("/admin/api/custom-providers/my-llm").json()
    assert deleted["applied"] is True
    assert client.get("/admin/api/custom-providers").json()["providers"] == []

    missing = client.delete("/admin/api/custom-providers/my-llm").json()
    assert missing["applied"] is False


def test_custom_provider_endpoints_are_loopback_only():
    app = _app()
    remote = TestClient(app, client=("203.0.113.10", 50000))
    assert remote.get("/admin/api/custom-providers").status_code == 403
    assert (
        remote.post("/admin/api/custom-providers", json=_payload()).status_code == 403
    )
    assert remote.delete("/admin/api/custom-providers/x").status_code == 403


def test_admin_ui_exposes_custom_providers_view():
    static_dir = Path(__file__).resolve().parents[2] / "api" / "admin_static"
    index_html = (static_dir / "index.html").read_text(encoding="utf-8")
    admin_js = (static_dir / "admin.js").read_text(encoding="utf-8")

    assert 'data-view="custom_providers"' in index_html
    assert 'id="customProviderAdd"' in index_html
    assert "loadCustomProviders" in admin_js
    assert "/admin/api/custom-providers" in admin_js
