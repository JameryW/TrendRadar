from pathlib import Path

import yaml

from mcp_server.tools.notification import NotificationTools
from trendradar.ai.analyzer import AIAnalysisResult
from trendradar.ai.formatter import get_ai_analysis_renderer, render_ai_analysis_discord
from trendradar.core.loader import _load_webhook_config
import trendradar.notification.dispatcher as dispatcher_module
from trendradar.notification.senders import (
    _build_discord_payload,
    _extract_discord_summary_fields,
    send_to_generic_webhook,
)


class FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text

    def json(self):
        import json
        try:
            return json.loads(self.text)
        except Exception:
            return {}


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


def test_generic_discord_webhook_sends_rich_embed_payload(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        payload = kwargs["json"]
        calls.append((url, payload))
        return FakeResponse(204)

    def split_content(*_args, max_bytes, **_kwargs):
        return ["x" * min(max_bytes, 1000)]

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
    assert payload["username"] == "TrendRadar"
    assert "avatar_url" in payload
    assert "embeds" in payload
    assert len(payload["embeds"]) == 1
    embed = payload["embeds"][0]
    assert embed["title"] == "📡 当前榜单"
    assert embed["color"] == 0x5865F2
    assert embed["footer"]["text"] == "TrendRadar 热点雷达"
    assert "timestamp" in embed
    assert len(embed["description"]) <= 4096


def test_generic_discord_webhook_extracts_summary_fields(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        payload = kwargs["json"]
        calls.append((url, payload))
        return FakeResponse(204)

    sample_content = (
        "**概览：** 15 个平台，300 条热点\n"
        "**数据：** 120 条新增\n"
        "**本轮：** 2026-11-01 18:00\n"
        "---\n"
        "## 正文内容详情\n- 条目 1\n- 条目 2"
    )

    def split_content(*_args, **_kwargs):
        return [sample_content]

    monkeypatch.setattr("trendradar.notification.senders.requests.post", fake_post)

    result = send_to_generic_webhook(
        webhook_url="https://discord.com/api/webhooks/123/test-token",
        payload_template="",
        report_data={},
        report_type="热点汇总",
        split_content_func=split_content,
    )

    assert result is True
    assert len(calls) == 1
    embed = calls[0][1]["embeds"][0]
    assert "fields" in embed
    assert len(embed["fields"]) == 3
    assert embed["fields"][0] == {"name": "📋 概览", "value": "15 个平台，300 条热点", "inline": True}
    assert embed["fields"][1] == {"name": "📊 数据", "value": "120 条新增", "inline": True}
    assert embed["fields"][2] == {"name": "🕐 本轮", "value": "2026-11-01 18:00", "inline": True}
    assert embed["description"].startswith("## 正文内容详情")


def test_generic_discord_webhook_embed_fallback_to_plain_content_on_400(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        payload = kwargs["json"]
        calls.append((url, payload))
        # First attempt with embeds returns 400 Bad Request
        if "embeds" in payload:
            return FakeResponse(400, '{"embeds": ["Invalid embed"]}')
        # Fallback with content returns 204 Success
        return FakeResponse(204)

    def split_content(*_args, **_kwargs):
        return ["fallback text body"]

    monkeypatch.setattr("trendradar.notification.senders.requests.post", fake_post)

    result = send_to_generic_webhook(
        webhook_url="https://discord.com/api/webhooks/123/test-token",
        payload_template="",
        report_data={},
        report_type="当前榜单",
        split_content_func=split_content,
    )

    assert result is True
    assert len(calls) == 2
    assert "embeds" in calls[0][1]
    assert calls[1][1] == {"content": "fallback text body"}


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


def test_mcp_discord_notification_sends_embed_card(tmp_path, monkeypatch):
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
        title="测试报告",
        channels=["discord"],
    )

    assert result["success"] is True
    assert result["results"]["discord"]["success"] is True
    assert len(payloads) >= 2
    for payload in payloads:
        assert payload["username"] == "TrendRadar"
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert payload["embeds"][0]["color"] == 0x5865F2
        assert payload["embeds"][0]["title"].startswith("📡")
        assert len(payload["embeds"][0]["description"].encode("utf-8")) <= 4096


def test_mcp_discord_notification_fallback_on_embed_rejection(tmp_path, monkeypatch):
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

    attempts = []

    def fake_post(_url, **kwargs):
        payload = kwargs["json"]
        attempts.append(payload)
        if "embeds" in payload:
            return FakeResponse(400, '{"message": "Invalid Embed"}')
        return FakeResponse(204)

    monkeypatch.setattr("mcp_server.tools.notification.requests.post", fake_post)

    result = NotificationTools(str(tmp_path)).send_notification(
        message="测试降级消息",
        title="测试",
        channels=["discord"],
    )

    assert result["success"] is True
    assert result["results"]["discord"]["success"] is True
    assert len(attempts) == 2
    assert "embeds" in attempts[0]
    assert attempts[1] == {"content": "测试降级消息"}


def test_mcp_legacy_generic_discord_notification_also_sends_embed(tmp_path, monkeypatch):
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
    for payload in payloads:
        assert payload["username"] == "TrendRadar"
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert payload["embeds"][0]["color"] == 0x5865F2


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


def test_extract_discord_summary_fields():
    content = (
        "**概览：** 8个平台\n"
        "**数据：** 50条\n"
        "**本轮：** 12:00\n"
        "---\n"
        "正文开始"
    )
    fields, body = _extract_discord_summary_fields(content)
    assert len(fields) == 3
    assert fields[0]["name"] == "📋 概览"
    assert fields[0]["value"] == "8个平台"
    assert fields[1]["name"] == "📊 数据"
    assert fields[2]["name"] == "🕐 本轮"
    assert body == "正文开始"


def test_render_ai_analysis_discord():
    res = AIAnalysisResult(
        core_trends="核心态势分析内容",
        sentiment_controversy="舆情争议内容",
        signals="信号异动内容",
        rss_insights="RSS观点内容",
        outlook_strategy="策略建议内容",
        standalone_summaries={"财联社": "快讯摘要"},
        success=True,
    )
    setattr(res, "market_regions", "A股/美股走势分化")

    rendered = render_ai_analysis_discord(res)
    assert "## ✨ AI 深度研判" in rendered
    assert "**🎯 核心态势**" in rendered
    assert "**🌏 地域/市场**" in rendered
    assert "**💬 舆论分歧**" in rendered
    assert "**📡 异动信号**" in rendered
    assert "**🧠 专业视角**" in rendered
    assert "**💡 研判策略**" in rendered
    assert "**🌐 独立速览**" in rendered


def test_get_ai_analysis_renderer_discord():
    renderer = get_ai_analysis_renderer("discord")
    assert renderer == render_ai_analysis_discord
