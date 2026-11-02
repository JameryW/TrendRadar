# coding=utf-8
"""
测试 AI 内容分析引擎扩展及多渠道富文本渲染
覆盖:
- AIAnalysisResult 中 market_regions 字段支持与解析
- AIAnalyzer 响应解析（含 market_regions 与容错）
- 各渠道格式化器（飞书、钉钉、Telegram、HTML、纯文本、通用Markdown）
- 副标题自适应统计生成器
- MCP 渠道策略指南同步
"""

import json
from unittest.mock import MagicMock

from trendradar.ai.analyzer import AIAnalysisResult, AIAnalyzer
from trendradar.ai.formatter import (
    _format_ai_subtext,
    _format_topics_and_sentiment_line,
    get_ai_analysis_renderer,
    render_ai_analysis_dingtalk,
    render_ai_analysis_discord,
    render_ai_analysis_feishu,
    render_ai_analysis_html_rich,
    render_ai_analysis_markdown,
    render_ai_analysis_plain,
    render_ai_analysis_telegram,
)
from trendradar.notification.senders import _extract_discord_summary_fields
from mcp_server.tools.notification import CHANNEL_FORMAT_GUIDES


def test_ai_analysis_result_dataclass_market_regions():
    """验证 AIAnalysisResult 显式支持 market_regions 字段"""
    res = AIAnalysisResult(
        core_trends="核心态势",
        market_regions="国内A股震荡，港股科技领涨，美股纳指微调",
        sentiment_controversy="舆情争议",
        signals="弱信号",
        rss_insights="RSS观点",
        outlook_strategy="策略建议",
        standalone_summaries={"财联社": "快讯摘要"},
        analyzed_news=30,
        hotlist_analyzed=20,
        rss_analyzed=10,
        success=True,
    )
    assert res.market_regions == "国内A股震荡，港股科技领涨，美股纳指微调"
    assert res.core_trends == "核心态势"
    assert res.success is True


def test_ai_analyzer_parse_response_with_market_regions():
    """验证 AIAnalyzer._parse_response 成功提取 market_regions 字段"""
    analyzer = AIAnalyzer(
        ai_config={"API_KEY": "test-key", "MODEL": "test-model"},
        analysis_config={"MAX_NEWS_FOR_ANALYSIS": 50},
        get_time_func=MagicMock(),
    )

    data = {
        "core_trends": "AI赋能各行业持续发酵",
        "market_regions": "A股算力概念股活跃；美股芯片股调整",
        "sentiment_controversy": "算力成本高昂与降本预期博弈",
        "signals": "某冷门芯片技术贴在技术论坛爆火",
        "rss_insights": "半导体产业周报：先进制程良率爬坡",
        "outlook_strategy": "1. 投资者：关注高性价比算力供应商",
        "standalone_summaries": {"知乎": "社区技术流讨论偏理性"},
    }

    raw_json = json.dumps(data, ensure_ascii=False)
    result = analyzer._parse_response(f"```json\n{raw_json}\n```")

    assert result.success is True
    assert result.core_trends == data["core_trends"]
    assert result.market_regions == data["market_regions"]
    assert result.sentiment_controversy == data["sentiment_controversy"]
    assert result.signals == data["signals"]
    assert result.rss_insights == data["rss_insights"]
    assert result.outlook_strategy == data["outlook_strategy"]
    assert result.standalone_summaries == {"知乎": "社区技术流讨论偏理性"}


def test_ai_analyzer_parse_response_without_market_regions():
    """验证当模型未输出 market_regions 时，字段缺省为空字符串不报错"""
    analyzer = AIAnalyzer(
        ai_config={"API_KEY": "test-key", "MODEL": "test-model"},
        analysis_config={"MAX_NEWS_FOR_ANALYSIS": 50},
        get_time_func=MagicMock(),
    )

    data = {
        "core_trends": "核心热点",
        "sentiment_controversy": "舆情分歧",
    }
    result = analyzer._parse_response(json.dumps(data))
    assert result.success is True
    assert result.core_trends == "核心热点"
    assert result.market_regions == ""


def test_format_ai_subtext():
    """验证 _format_ai_subtext 自适应统计行生成逻辑"""
    assert _format_ai_subtext(None) == ""
    assert _format_ai_subtext(AIAnalysisResult(analyzed_news=0)) == ""

    res_hotlist_only = AIAnalysisResult(analyzed_news=25, hotlist_analyzed=25)
    assert _format_ai_subtext(res_hotlist_only) == "📊 深度研判 25 条焦点热点（热榜 25）"

    res_full = AIAnalysisResult(
        analyzed_news=60,
        hotlist_analyzed=40,
        rss_analyzed=15,
        standalone_analyzed=5,
    )
    assert (
        _format_ai_subtext(res_full)
        == "📊 深度研判 60 条焦点热点（热榜 40 ｜ RSS 15 ｜ 独立源 5）"
    )


def test_render_ai_analysis_feishu():
    """验证飞书卡片 Markdown 渲染效果（含灰色统计行与 HTML 实体转义）"""
    res = AIAnalysisResult(
        core_trends="核心态势分析内容",
        market_regions="A股港股联动观察",
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
    rendered = render_ai_analysis_feishu(res)
    assert "**✨ AI 深度研判**" in rendered
    assert "<font color='grey'>📊 深度研判 45 条焦点热点（热榜 35 ｜ RSS 10）</font>" in rendered
    assert "**🎯 核心热点态势**" in rendered
    assert "**🌏 地域与市场**" in rendered
    assert "A股港股联动观察" in rendered
    assert "**💬 舆论风向争议**" in rendered
    assert "**📡 异动与弱信号**" in rendered
    assert "**🧠 RSS 深度洞察**" in rendered
    assert "**💡 研判策略建议**" in rendered
    assert "**🌐 独立源点速览**" in rendered
    assert "&#91;财联社&#93;:\n快讯摘要" in rendered


def test_render_ai_analysis_dingtalk():
    """验证钉钉 Markdown 渲染效果（#### 级标题与 > 引用块统计）"""
    res = AIAnalysisResult(
        core_trends="核心态势分析内容",
        market_regions="国内与海外科技股表现",
        sentiment_controversy="多空分歧",
        signals="冷门赛道异动",
        rss_insights="海外行研观点",
        outlook_strategy="布局建议",
        standalone_summaries={"36氪": "创投速递"},
        analyzed_news=30,
        hotlist_analyzed=30,
        success=True,
    )
    rendered = render_ai_analysis_dingtalk(res)
    assert "### ✨ AI 深度研判" in rendered
    assert "> 📊 深度研判 30 条焦点热点（热榜 30）" in rendered
    assert "#### 🎯 核心热点态势" in rendered
    assert "#### 🌏 地域与市场" in rendered
    assert "国内与海外科技股表现" in rendered
    assert "#### 💬 舆论风向争议**" not in rendered  # 钉钉是 #### 无 **
    assert "#### 💬 舆论风向争议" in rendered
    assert "#### 📡 异动与弱信号" in rendered
    assert "#### 🧠 RSS 深度洞察" in rendered
    assert "#### 💡 研判策略建议" in rendered
    assert "#### 🌐 独立源点速览" in rendered
    assert "[36氪]:\n创投速递" in rendered


def test_render_ai_analysis_telegram():
    """验证 Telegram HTML 渲染效果（<code> 统计行与 <b> 标签）"""
    res = AIAnalysisResult(
        core_trends="核心态势 <敏感词 & 符号>",
        market_regions="港股恒指上行",
        sentiment_controversy="中外观点温差",
        signals="前沿信号",
        rss_insights="深度洞察",
        outlook_strategy="策略要点",
        standalone_summaries={"Hacker News": "Tech Discussions"},
        analyzed_news=20,
        hotlist_analyzed=15,
        rss_analyzed=5,
        success=True,
    )
    rendered = render_ai_analysis_telegram(res)
    assert "<b>✨ AI 深度研判</b>" in rendered
    assert "<code>📊 深度研判 20 条焦点热点（热榜 15 ｜ RSS 5）</code>" in rendered
    assert "<b>🎯 核心热点态势</b>" in rendered
    assert "&lt;敏感词 &amp; 符号&gt;" in rendered
    assert "<b>🌏 地域与市场</b>" in rendered
    assert "港股恒指上行" in rendered
    assert "<b>💬 舆论风向争议</b>" in rendered
    assert "<b>📡 异动与弱信号</b>" in rendered
    assert "<b>🧠 RSS 深度洞察</b>" in rendered
    assert "<b>💡 研判策略建议</b>" in rendered
    assert "<b>🌐 独立源点速览</b>" in rendered


def test_render_ai_analysis_html_rich():
    """验证 HTML 丰富卡片渲染效果（副标题样式与网格块）"""
    res = AIAnalysisResult(
        core_trends="热点格局变化",
        market_regions="A股芯片板块走强",
        sentiment_controversy="估值讨论",
        signals="新专利公布",
        rss_insights="学术前沿",
        outlook_strategy="谨慎追高",
        standalone_summaries={"雪球": "大V观点"},
        analyzed_news=18,
        hotlist_analyzed=18,
        success=True,
    )
    html = render_ai_analysis_html_rich(res)
    assert '<div class="ai-section-title">✨ AI 深度研判</div>' in html
    assert '<div class="ai-section-subtitle">📊 深度研判 18 条焦点热点（热榜 18）</div>' in html
    assert '<div class="ai-block-title">🎯 核心热点态势</div>' in html
    assert '<div class="ai-block-title">🌏 地域与市场</div>' in html
    assert "A股芯片板块走强" in html
    assert '<div class="ai-block-title">💬 舆论风向争议</div>' in html
    assert '<div class="ai-block-title">📡 异动与弱信号</div>' in html
    assert '<div class="ai-block-title">🧠 RSS 深度洞察</div>' in html
    assert '<div class="ai-block-title">💡 研判策略建议</div>' in html
    assert '<div class="ai-block-title">🌐 独立源点速览</div>' in html


def test_render_ai_analysis_error_and_skipped():
    """验证失败与跳过状态在各渠道的正确呈现"""
    err_res = AIAnalysisResult(success=False, error="网络连接超时")
    skip_res = AIAnalysisResult(success=False, skipped=True, error="本轮无新增")

    assert "⚠️ AI 分析失败: 网络连接超时" in render_ai_analysis_feishu(err_res)
    assert "ℹ️ 本轮无新增" in render_ai_analysis_feishu(skip_res)

    assert "⚠️ AI 分析失败: 网络连接超时" in render_ai_analysis_dingtalk(err_res)
    assert "ℹ️ 本轮无新增" in render_ai_analysis_dingtalk(skip_res)

    assert "⚠️ AI 分析失败: 网络连接超时" in render_ai_analysis_telegram(err_res)
    assert "ℹ️ 本轮无新增" in render_ai_analysis_telegram(skip_res)

    assert "AI 分析失败: 网络连接超时" in render_ai_analysis_html_rich(err_res)
    assert "ℹ️ 本轮无新增" in render_ai_analysis_html_rich(skip_res)


def test_channel_format_guides_ai_support():
    """验证 MCP 渠道策略指南中对 AI 深度研判格式的支持说明已更新"""
    assert any("AI 深度研判" in item for item in CHANNEL_FORMAT_GUIDES["feishu"]["supported"])
    assert any("AI 深度研判" in item for item in CHANNEL_FORMAT_GUIDES["dingtalk"]["supported"])
    assert any("AI 深度研判" in item for item in CHANNEL_FORMAT_GUIDES["telegram"]["supported"])
    assert any("态势基调" in item or "舆情基调" in item for item in CHANNEL_FORMAT_GUIDES["discord"]["supported"])


def test_ai_analysis_result_sentiment_and_topics():
    """验证 AIAnalysisResult 支持 market_sentiment 与 highlight_topics 属性"""
    res = AIAnalysisResult(
        market_sentiment="积极偏多",
        highlight_topics=["AI算力", "半导体", "美联储利率"],
        core_trends="算力板块走强",
        success=True,
    )
    assert res.market_sentiment == "积极偏多"
    assert res.highlight_topics == ["AI算力", "半导体", "美联储利率"]


def test_ai_analyzer_parse_response_with_sentiment_and_topics():
    """验证 AIAnalyzer._parse_response 正确解析 market_sentiment 与列表/字符串形式 highlight_topics"""
    analyzer = AIAnalyzer(
        ai_config={"API_KEY": "test-key", "MODEL": "test-model"},
        analysis_config={"MAX_NEWS_FOR_ANALYSIS": 50},
        get_time_func=MagicMock(),
    )

    # 1. 数组形式 highlight_topics
    data_list = {
        "market_sentiment": "中性分歧",
        "highlight_topics": ["#AI算力", "人形机器人", "国产芯片"],
        "core_trends": "科技领域多点开花",
    }
    res_list = analyzer._parse_response(json.dumps(data_list))
    assert res_list.market_sentiment == "中性分歧"
    assert res_list.highlight_topics == ["AI算力", "人形机器人", "国产芯片"]

    # 2. 字符串逗号/空格分隔形式 highlight_topics
    data_str = {
        "market_sentiment": "避险防守",
        "highlight_topics": "#黄金, 避险资产、 美元指数",
        "core_trends": "避险情绪升温",
    }
    res_str = analyzer._parse_response(json.dumps(data_str))
    assert res_str.market_sentiment == "避险防守"
    assert res_str.highlight_topics == ["黄金", "避险资产", "美元指数"]


def test_format_topics_and_sentiment_line():
    """验证 _format_topics_and_sentiment_line 多渠道样式输出"""
    res = AIAnalysisResult(
        market_sentiment="积极偏多",
        highlight_topics=["AI算力", "国产芯片"],
    )
    # Discord: 行内代码徽章
    discord_line = _format_topics_and_sentiment_line(res, style="discord")
    assert "> 🌡️ **态势基调**: 积极偏多  🏷️ **焦点标签**: `#AI算力` `#国产芯片`" == discord_line

    # Markdown / Feishu / Dingtalk: 引用块与分隔符
    markdown_line = _format_topics_and_sentiment_line(res, style="markdown")
    assert "> 🌡️ **态势基调**: 积极偏多 ｜ 🏷️ **焦点标签**: #AI算力 #国产芯片" == markdown_line

    # Telegram: HTML 标签
    tg_line = _format_topics_and_sentiment_line(res, style="telegram")
    assert "<code>🌡️ <b>态势基调:</b> 积极偏多 ｜ 🏷️ <b>焦点:</b> #AI算力 #国产芯片</code>" == tg_line

    # Plain: 方括号纯文本
    plain_line = _format_topics_and_sentiment_line(res, style="plain")
    assert "[🌡️ 态势基调: 积极偏多 ｜ 🏷️ 焦点标签: #AI算力 #国产芯片]" == plain_line

    # 空数据安全返回空串
    assert _format_topics_and_sentiment_line(None) == ""
    assert _format_topics_and_sentiment_line(AIAnalysisResult()) == ""


def test_render_ai_analysis_with_sentiment_and_topics_across_channels():
    """验证各渠道渲染器完整包含态势基调与焦点标签"""
    res = AIAnalysisResult(
        market_sentiment="谨慎观望",
        highlight_topics=["低空经济", "量子计算"],
        core_trends="前沿赛道博弈",
        analyzed_news=20,
        hotlist_analyzed=20,
        success=True,
    )

    # Discord
    discord_out = render_ai_analysis_discord(res)
    assert "## ✨ AI 深度研判" in discord_out
    assert "-# 📊 深度研判 20 条焦点热点（热榜 20）" in discord_out
    assert "> 🌡️ **态势基调**: 谨慎观望  🏷️ **焦点标签**: `#低空经济` `#量子计算`" in discord_out

    # Feishu
    feishu_out = render_ai_analysis_feishu(res)
    assert "**✨ AI 深度研判**" in feishu_out
    assert "> 🌡️ **态势基调**: 谨慎观望 ｜ 🏷️ **焦点标签**: #低空经济 #量子计算" in feishu_out

    # Dingtalk
    dingtalk_out = render_ai_analysis_dingtalk(res)
    assert "### ✨ AI 深度研判" in dingtalk_out
    assert "> 🌡️ **态势基调**: 谨慎观望 ｜ 🏷️ **焦点标签**: #低空经济 #量子计算" in dingtalk_out

    # Telegram
    tg_out = render_ai_analysis_telegram(res)
    assert "<b>✨ AI 深度研判</b>" in tg_out
    assert "<code>🌡️ <b>态势基调:</b> 谨慎观望 ｜ 🏷️ <b>焦点:</b> #低空经济 #量子计算</code>" in tg_out

    # HTML Rich
    html_out = render_ai_analysis_html_rich(res)
    assert '<span class="ai-sentiment-badge">🌡️ 谨慎观望</span>' in html_out
    assert '<span class="ai-topic-tag">#低空经济</span>' in html_out
    assert '<span class="ai-topic-tag">#量子计算</span>' in html_out


def test_discord_extract_summary_fields_with_sentiment_and_topics():
    """验证 Discord Embed 概览解析器能够提取舆情基调与焦点标签为独立字段"""
    raw_content = (
        "**总新闻：** 120条\n"
        "**舆情基调：** 积极偏多\n"
        "**焦点标签：** #AI算力 #人形机器人\n"
        "**热榜：** 80条\n"
        "---\n"
        "正文内容从这里开始..."
    )
    fields, body = _extract_discord_summary_fields(raw_content)
    names = [f["name"] for f in fields]
    assert "📋 总新闻" in names
    assert "🌡️ 舆情基调" in names
    assert "🏷️ 焦点标签" in names
    assert "🔥 热榜" in names

    # 焦点标签独占整行
    tag_field = next(f for f in fields if f["name"] == "🏷️ 焦点标签")
    assert tag_field["inline"] is False
    assert tag_field["value"] == "#AI算力 #人形机器人"
    assert "正文内容从这里开始" in body
