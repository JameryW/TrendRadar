from pathlib import Path

import yaml

from mcp_server.tools.notification import NotificationTools
from trendradar.core.loader import _load_webhook_config
import trendradar.notification.dispatcher as dispatcher_module
from trendradar.notification.senders import send_to_generic_webhook


class FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


def _write_config(root: Path, config: dict) -> None:
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )


def test_discord_webhook_config_prefers_environment(monkeypatch):
    env_url = "https://discord.com/api/webhooks/env/test-token"
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", env_url)

    config = _load_webhook_config(
        {
            "notification": {
                "channels": {
                    "discord": {"webhook_url": "https://discord.com/yaml"},
                }
            }
        }
    )

    assert config["DISCORD_WEBHOOK_URL"] == env_url


def test_generic_discord_webhook_enforces_safe_batch_and_content_payload(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        payload = kwargs["json"]
        calls.append((url, payload))
        if len(payload["content"]) > 2000:
            return FakeResponse(400, '{"content":["Must be 2000 or fewer in length."]}')
        return FakeResponse(204)

    def split_content(*_args, max_bytes, **_kwargs):
        return ["x" * max_bytes]

    monkeypatch.setattr("trendradar.notification.senders.requests.post", fake_post)

    result = send_to_generic_webhook(
        webhook_url="https://discord.com/api/webhooks/123/test-token",
        payload_template='{"title": "{title}", "content": "{content}"}',
        report_data={},
        report_type="当前榜单",
        batch_size=4000,
        split_content_func=split_content,
    )

    assert result is True
    assert len(calls) == 1
    _, payload = calls[0]
    assert set(payload) == {"content"}
    assert len(payload["content"].encode("utf-8")) <= 1900


def test_generic_non_discord_webhook_keeps_configured_batch_size(monkeypatch):
    requested_limits = []
    sent_payloads = []

    def fake_post(_url, **kwargs):
        sent_payloads.append(kwargs["json"])
        return FakeResponse(204)

    def split_content(*_args, max_bytes, **_kwargs):
        requested_limits.append(max_bytes)
        return ["x" * max_bytes]

    monkeypatch.setattr("trendradar.notification.senders.requests.post", fake_post)

    result = send_to_generic_webhook(
        webhook_url="https://hooks.example.test/trendradar",
        payload_template="",
        report_data={},
        report_type="当前榜单",
        batch_size=4000,
        split_content_func=split_content,
    )

    assert result is True
    assert requested_limits == [3800]
    assert len(sent_payloads[0]["content"]) == 3800
    assert sent_payloads[0]["title"] == "当前榜单"


def test_cli_dispatcher_prefers_explicit_discord_and_deduplicates_legacy_generic(
    monkeypatch,
):
    url = "https://discord.com/api/webhooks/123/test-token"
    calls = []

    dispatcher = dispatcher_module.NotificationDispatcher(
        config={
            "ENABLE_NOTIFICATION": True,
            "DISCORD_WEBHOOK_URL": url,
            "GENERIC_WEBHOOK_URL": url,
            "DISPLAY": {"REGIONS": {}},
        },
        get_time_func=lambda: None,
        split_content_func=lambda *_args, **_kwargs: [],
    )

    monkeypatch.setattr(
        dispatcher,
        "_send_discord",
        lambda *_args, **_kwargs: calls.append("discord") or True,
    )
    monkeypatch.setattr(
        dispatcher,
        "_send_generic_webhook",
        lambda *_args, **kwargs: calls.append(
            ("generic", kwargs.get("webhook_urls"))
        )
        or True,
    )

    results = dispatcher.dispatch_all(
        report_data={"stats": [], "new_titles": []},
        report_type="测试",
        skip_translation=True,
    )

    assert results == {"discord": True}
    assert calls == ["discord"]


def test_mcp_discord_notification_splits_message_before_sending(tmp_path, monkeypatch):
    webhook_url = "https://discord.com/api/webhooks/123/test-token"
    _write_config(
        tmp_path,
        {
            "notification": {
                "enabled": True,
                "channels": {"discord": {"webhook_url": webhook_url}},
            },
            "advanced": {"batch_size": {"discord": 1900}},
        },
    )

    payloads = []

    def fake_post(_url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(204)

    monkeypatch.setattr("mcp_server.tools.notification.requests.post", fake_post)

    result = NotificationTools(str(tmp_path)).send_notification(
        message="x" * 3500,
        title="测试",
        channels=["discord"],
    )

    assert result["success"] is True
    assert result["results"]["discord"]["success"] is True
    assert len(payloads) >= 2
    assert all(set(payload) == {"content"} for payload in payloads)
    assert all(len(payload["content"].encode("utf-8")) <= 1900 for payload in payloads)


def test_mcp_legacy_generic_discord_notification_is_also_safe(tmp_path, monkeypatch):
    webhook_url = "https://discord.com/api/webhooks/123/test-token"
    _write_config(
        tmp_path,
        {
            "notification": {
                "enabled": True,
                "channels": {
                    "generic_webhook": {
                        "webhook_url": webhook_url,
                        "payload_template": '{"title": "{title}", "content": "{content}"}',
                    }
                },
            },
            "advanced": {"batch_size": {"default": 4000}},
        },
    )

    payloads = []

    def fake_post(_url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(204)

    monkeypatch.setattr("mcp_server.tools.notification.requests.post", fake_post)

    result = NotificationTools(str(tmp_path)).send_notification(
        message="x" * 3500,
        title="测试",
        channels=["generic_webhook"],
    )

    assert result["success"] is True
    assert len(payloads) >= 2
    assert all(set(payload) == {"content"} for payload in payloads)
    assert all(len(payload["content"].encode("utf-8")) <= 1900 for payload in payloads)


def test_mcp_does_not_send_the_same_discord_webhook_twice(tmp_path, monkeypatch):
    webhook_url = "https://discord.com/api/webhooks/123/test-token"
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("GENERIC_WEBHOOK_URL", raising=False)
    _write_config(
        tmp_path,
        {
            "notification": {
                "enabled": True,
                "channels": {
                    "discord": {"webhook_url": webhook_url},
                    "generic_webhook": {"webhook_url": webhook_url},
                },
            },
        },
    )

    payloads = []

    def fake_post(_url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(204)

    monkeypatch.setattr("mcp_server.tools.notification.requests.post", fake_post)

    result = NotificationTools(str(tmp_path)).send_notification(
        message="test message",
        channels=["generic_webhook", "discord"],
    )

    assert result["success"] is True
    assert result["summary"] == "2/2 个渠道发送成功"
    assert len(payloads) == 1
    assert result["results"]["generic_webhook"]["deduplicated_to"] == "discord"


def test_mcp_duplicate_discord_webhook_propagates_send_failure(tmp_path, monkeypatch):
    webhook_url = "https://discord.com/api/webhooks/123/test-token"
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("GENERIC_WEBHOOK_URL", raising=False)
    _write_config(
        tmp_path,
        {
            "notification": {
                "enabled": True,
                "channels": {
                    "discord": {"webhook_url": webhook_url},
                    "generic_webhook": {"webhook_url": webhook_url},
                },
            },
        },
    )

    payloads = []

    def fake_post(_url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(500, "failed")

    monkeypatch.setattr("mcp_server.tools.notification.requests.post", fake_post)

    result = NotificationTools(str(tmp_path)).send_notification(
        message="test message",
        channels=["discord", "generic_webhook"],
    )

    assert result["success"] is False
    assert result["summary"] == "0/2 个渠道发送成功"
    assert len(payloads) == 1
    assert result["results"]["generic_webhook"]["success"] is False
