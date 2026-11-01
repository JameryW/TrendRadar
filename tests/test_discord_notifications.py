import os
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
    _get_discord_embed_color,
    _get_discord_embed_thumbnail,
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
        "**最热话题：** 科技创新、低空经济\n"
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
    assert len(embed["fields"]) == 4
    assert embed["fields"][0] == {"name": "📋 概览", "value": "15 个平台，300 条热点", "inline": True}
    assert embed["fields"][1] == {"name": "📊 数据", "value": "120 条新增", "inline": True}
    assert embed["fields"][2] == {"name": "🕐 本轮", "value": "2026-11-01 18:00", "inline": True}
    assert embed["fields"][3] == {"name": "🔥 最热话题", "value": "科技创新、低空经济", "inline": False}
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
        message="x" * 7500,
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
        "**最热话题：** 机器人、量子计算\n"
        "---\n"
        "正文开始"
    )
    fields, body = _extract_discord_summary_fields(content)
    assert len(fields) == 4
    assert fields[0]["name"] == "📋 概览"
    assert fields[0]["value"] == "8个平台"
    assert fields[1]["name"] == "📊 数据"
    assert fields[2]["name"] == "🕐 本轮"
    assert fields[3]["name"] == "🔥 最热话题"
    assert fields[3]["value"] == "机器人、量子计算"
    assert fields[3]["inline"] is False
    assert body == "正文开始"


def test_discord_embed_dynamic_colors():
    assert _get_discord_embed_color("市场异动预警") == 0xED4245
    assert _get_discord_embed_color("突发快讯") == 0xED4245
    assert _get_discord_embed_color("当前报告", "🚨 检测到异动信号") == 0xED4245
    assert _get_discord_embed_color("每日早报") == 0x57F287
    assert _get_discord_embed_color("今日总结") == 0x57F287
    assert _get_discord_embed_color("晚盘复盘") == 0x9B59B6
    assert _get_discord_embed_color("夜盘跟踪") == 0x9B59B6
    assert _get_discord_embed_color("深度分析周报") == 0xFEE75C
    assert _get_discord_embed_color("AI分析报告") == 0x9B59B6
    assert _get_discord_embed_color("智库研判") == 0x9B59B6
    assert _get_discord_embed_color("增量分析报告") == 0x00D2D3
    assert _get_discord_embed_color("RSS订阅报告") == 0xE67E22
    assert _get_discord_embed_color("当前榜单") == 0x5865F2


def test_render_ai_analysis_discord():
    res = AIAnalysisResult(
        core_trends="核心态势分析内容",
        sentiment_controversy="舆情争议内容",
        signals="信号异动内容",
        rss_insights="RSS观点内容",
        outlook_strategy="策略建议内容",
        standalone_summaries={"财联社": "快讯摘要"},
        analyzed_news=45,
        hotlist_analyzed=35,
        rss_analyzed=10,
        success=True,
    )
    setattr(res, "market_regions", "A股/美股走势分化")

    rendered = render_ai_analysis_discord(res)
    assert "## ✨ AI 深度研判" in rendered
    assert "-# 📊 深度研判 45 条焦点热点（热榜 35 ｜ RSS 10）" in rendered
    assert "### 🎯 核心态势" in rendered
    assert "### 🌏 地域/市场" in rendered
    assert "### 💬 舆论分歧" in rendered
    assert "### 📡 异动信号" in rendered
    assert "### 🧠 专业视角" in rendered
    assert "### 💡 研判策略" in rendered
    assert "### 🌐 独立速览" in rendered


def test_get_ai_analysis_renderer_discord():
    renderer = get_ai_analysis_renderer("discord")
    assert renderer == render_ai_analysis_discord


def test_extract_crawler_native_summary_fields():
    content = (
        "**总新闻：** 56 条（新增 8 + 0）\n"
        "**热榜：** 47/255（平台 11/11）\n"
        "**RSS：** 9/40（源 2/2）\n"
        "**AI 分析：** 47 + 9\n"
        "**类型：** 当前榜单\n"
        "**时间：** 2026-11-01 20:50:00\n"
        "**最热话题：** 科技创新(5) | 低空经济(4) | 人工智能(3)\n"
        "\n"
        "📊 **热点新闻统计** (共 47 条)\n"
    )
    fields, body = _extract_discord_summary_fields(content)
    assert len(fields) == 6
    assert fields[0] == {"name": "📋 总新闻", "value": "56 条（新增 8 + 0）", "inline": True}
    assert fields[1] == {"name": "🔥 热榜", "value": "47/255（平台 11/11）", "inline": True}
    assert fields[2] == {"name": "📡 RSS", "value": "9/40（源 2/2）", "inline": True}
    assert fields[3] == {"name": "✨ AI 分析", "value": "47 + 9", "inline": True}
    assert fields[4] == {"name": "🏷️ 类型", "value": "当前榜单", "inline": True}
    assert fields[5] == {"name": "🔥 最热话题", "value": "科技创新(5) | 低空经济(4) | 人工智能(3)", "inline": False}
    assert body.startswith("📊 **热点新闻统计**")


def test_discord_embed_thumbnail():
    assert "1f6a8.png" in _get_discord_embed_thumbnail("市场异动预警")
    assert "2728.png" in _get_discord_embed_thumbnail("AI深度研判")
    assert "2728.png" in _get_discord_embed_thumbnail("当前报告", "## ✨ AI 深度研判\n- 核心内容")
    assert "1f195.png" in _get_discord_embed_thumbnail("增量分析模式")
    assert "1f4f0.png" in _get_discord_embed_thumbnail("RSS订阅统计")
    assert "1f4c8.png" in _get_discord_embed_thumbnail("每日早报")
    assert "1f319.png" in _get_discord_embed_thumbnail("晚盘复盘")
    assert "1f4ca.png" in _get_discord_embed_thumbnail("深度分析周报")
    assert "1f4e1.png" in _get_discord_embed_thumbnail("当前榜单")


def test_discord_embed_author_thumbnail_badges():
    raw_content = (
        "**概览：** 10个平台\n"
        "**数据：** 100条\n"
        "---\n"
        "  1. [今日头条] [新能源车免征购置税延长](https://example.com) [1] - 10:30 (2次)\n"
        "  2. [华尔街见闻] [全球降息周期启动](https://example.com/ws) [2]"
    )
    os.environ["TRENDRADAR_WEB_URL"] = "http://localhost:8081"
    try:
        payload = _build_discord_payload("今日总结", raw_content, batch_index=1, total_batches=2)
        assert payload["username"] == "TrendRadar"
        embed = payload["embeds"][0]
        assert embed["author"]["name"] == "TrendRadar · 实时舆情雷达"
        assert embed["author"]["url"] == "http://localhost:8081"
        assert embed["url"] == "http://localhost:8081"
        assert "1f4c8.png" in embed["thumbnail"]["url"]  # 今日总结 -> 📈
        assert "批次 1/2" in embed["footer"]["text"]
        assert "`今日头条`" in embed["description"]
        assert "`华尔街见闻`" in embed["description"]
        assert len(embed["fields"]) == 2
    finally:
        os.environ.pop("TRENDRADAR_WEB_URL", None)


def test_discord_embed_multi_batch_field_scoping():
    content_b1 = "**概览：** 10个平台\n**数据：** 100条\n---\n正文第一批"
    content_b2 = "**概览：** 10个平台\n**数据：** 100条\n---\n正文第二批"

    payload1 = _build_discord_payload("当前榜单", content_b1, batch_index=1, total_batches=2)
    payload2 = _build_discord_payload("当前榜单", content_b2, batch_index=2, total_batches=2)

    assert "fields" in payload1["embeds"][0]
    # 第 2 批去除了冗余的 fields 展示，但正文中的 header 依然被干净移除
    assert "fields" not in payload2["embeds"][0]
    assert payload2["embeds"][0]["description"] == "正文第二批"


def test_discord_embed_heading_and_topic_beautification():
    raw_content = (
        "**概览：** 5个平台\n"
        "**数据：** 20条\n"
        "**新增：** 5条\n"
        "---\n"
        "📊 **热点新闻统计** (共 20 条)\n"
        "\n"
        "🔥 [1/5] **低空经济** : **8** 条\n"
        "\n"
        "  1. [今日头条] [eVTOL适航取证提速](https://example.com/1) **[1]** - 21:26 (2次)\n"
        "  2. [财联社] [低空经济发展提速](https://example.com/2) **[2]** - 21:26\n"
        "  3. [华尔街见闻] [eVTOL产业链调研](https://example.com/3) **[3]** - 21:26\n"
        "  4. [澎湃新闻] [行业大会即将召开](https://example.com/4) **[4]** - 21:26\n"
        "\n"
        "📰 **RSS 订阅统计** (共 2 条)\n"
        "\n"
        "📌 [1/2] **AI 技术** : 2 条\n"
        "\n"
        "  1. [Hacker News] [Show HN: FastMCP](https://example.com/hn1) - 21:26\n"
        "\n"
        "🆕 **新增热点** (共 1 条)\n"
        "\n"
        "**今日头条** (1 条):\n"
        "\n"
        "  1. [百度热搜] [新型固态电池研发成功](https://example.com/5)\n"
        "\n"
        "📋 **独立展示区** (共 1 条)\n"
        "\n"
        "**华尔街见闻** (1 条):\n"
        "\n"
        "  1. [美股全线收高](https://example.com/6)\n"
        "\n"
        "⚠️ **数据获取失败的平台：**\n"
        "  • 腾讯新闻"
    )
    payload = _build_discord_payload("当前榜单", raw_content, batch_index=1, total_batches=1)
    desc = payload["embeds"][0]["description"]
    assert "## 📊 热点新闻统计 (共 20 条)" in desc
    assert "### 🔥 [1/5] **低空经济** · 8条" in desc
    assert "🥇 **[1]**" in desc
    assert "🥈 **[2]**" in desc
    assert "🥉 **[3]**" in desc
    assert "**[4]**" in desc
    assert " · 21:26" in desc
    assert "## 📰 RSS 订阅统计 (共 2 条)" in desc
    assert "### 📌 [1/2] **AI 技术** · 2条" in desc
    assert "## 🆕 新增热点 (共 1 条)" in desc
    assert "### 📌 **今日头条** · 1条" in desc
    assert "## 📋 独立展示区 (共 1 条)" in desc
    assert "### 📌 **华尔街见闻** · 1条" in desc
    assert "## ⚠️ 数据获取失败的平台" in desc
    assert "`今日头条`" in desc
    assert "`财联社`" in desc
    assert "`华尔街见闻`" in desc

    fields = payload["embeds"][0]["fields"]
    field_names = [f["name"] for f in fields]
    assert "📋 概览" in field_names
    assert "📊 数据" in field_names
    assert "🆕 新增" in field_names
    assert " · 21:26 · 🔁 2次" in desc


def test_discord_embed_advanced_visual_enhancements():
    # 自动提取批次头部并赋给卡片标题与页脚，验证奖牌区间、纯文本奖牌、频次图标与段首来源徽章
    batched_content = (
        "**[第 2/5 批次]**\n\n"
        "  1. [彭博社] [全球芯片产业链动态](https://example.com/c1) **[1 - 3]** - 22:00 (3次)\n"
        "  2. 纯文本要闻 **[2]** - 22:00\n"
        "\n"
        "[格隆汇]:\n港股科技股午后走高\n"
    )
    payload = _build_discord_payload("AI深度研判", batched_content)
    embed = payload["embeds"][0]
    assert embed["title"] == "📡 AI深度研判 · 第 2/5 批"
    assert embed["footer"]["text"] == "TrendRadar 热点雷达 · 批次 2/5"
    assert "fields" not in embed  # 第 2 批次自动不输出概览网格
    desc = embed["description"]
    assert "🥇 **[1 - 3]**" in desc
    assert "纯文本要闻 🥈 **[2]** · 22:00" in desc
    assert " · 22:00 · 🔁 3次" in desc
    assert "`格隆汇`: 港股科技股午后走高" in desc



