# coding=utf-8
"""
消息分批处理模块

提供消息内容分批拆分功能，确保消息大小不超过各平台限制
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Callable

from trendradar.report.formatter import format_title_for_platform
from trendradar.report.helpers import format_rank_display
from trendradar.utils.time import DEFAULT_TIMEZONE, format_iso_time_friendly, convert_time_for_display
from trendradar.notification.batch import truncate_at_line_boundary


NOTIFICATION_MAX_TITLES_PER_GROUP = 5
NOTIFICATION_MAX_NEW_TITLES_PER_SOURCE = 3
NOTIFICATION_MAX_STANDALONE_ITEMS_PER_SOURCE = 3


# === 分批安全辅助函数 ===

def _split_content_by_lines(
    content: str, footer: str, max_bytes: int, base_header: str
) -> List[str]:
    """将超长内容按行边界拆分成多个完整批次（每个批次带 footer）

    不会丢弃任何内容，溢出部分自动分配到后续批次。

    Args:
        content: 正文内容（不含 footer，可能含 base_header）
        footer: 尾部内容（更新时间等）
        max_bytes: 单批次最大字节数
        base_header: 后续批次的头部

    Returns:
        完整批次列表（每个元素 = 正文 + footer，大小 ≤ max_bytes）
    """
    footer_size = len(footer.encode("utf-8"))
    result_batches = []
    lines = content.split("\n")

    current = ""
    for line in lines:
        candidate = current + line + "\n"
        if len(candidate.encode("utf-8")) + footer_size > max_bytes and current.strip():
            result_batches.append(current + footer)
            current = base_header + line + "\n"
        else:
            current = candidate

    if current.strip():
        result_batches.append(current + footer)

    return result_batches


def _safe_append_batch(
    batches: List[str], content: str, footer: str, max_bytes: int,
    base_header: str = ""
) -> None:
    """安全追加批次，超限时按行拆分成多个批次（不丢弃内容）

    Args:
        batches: 批次列表（原地修改）
        content: 正文内容（不含 footer）
        footer: 尾部内容（更新时间等）
        max_bytes: 最大字节数
        base_header: 溢出时后续批次的头部
    """
    full = content + footer
    if len(full.encode("utf-8")) <= max_bytes:
        batches.append(full)
        return

    split_batches = _split_content_by_lines(content, footer, max_bytes, base_header)
    if split_batches:
        batches.extend(split_batches)
    else:
        # 极端情况：单行就超限，强制截断
        batches.append(truncate_at_line_boundary(full, max_bytes))


def _safe_new_batch(
    new_content: str, footer: str, max_bytes: int, base_header: str,
    batches: List[str] = None
) -> str:
    """安全创建新批次，超限时将溢出内容拆分到 batches 中，返回最后一段作为 current_batch

    Args:
        new_content: 新批次完整内容（含 base_header + section_header + ...）
        footer: 尾部内容
        max_bytes: 最大字节数
        base_header: 基础头部
        batches: 批次列表，溢出部分追加到此（可选）

    Returns:
        可安全继续追加内容的 current_batch（大小 + footer ≤ max_bytes）
    """
    if len((new_content + footer).encode("utf-8")) <= max_bytes:
        return new_content

    if batches is None:
        # 无法拆分到 batches，退回行边界截断
        footer_size = len(footer.encode("utf-8"))
        available = max_bytes - footer_size
        header_size = len(base_header.encode("utf-8"))
        if available <= header_size:
            return base_header
        return truncate_at_line_boundary(new_content, available)

    # 拆分：前面的部分存入 batches，最后一段作为 current_batch 返回
    split_batches = _split_content_by_lines(new_content, footer, max_bytes, base_header)
    if len(split_batches) <= 1:
        # 无法再拆，直接返回（由后续 _safe_append_batch 兜底）
        return new_content

    # 前 N-1 个批次存入 batches
    batches.extend(split_batches[:-1])
    # 最后一个批次去掉 footer 作为 current_batch（后续还会追加内容）
    last = split_batches[-1]
    if last.endswith(footer):
        return last[: -len(footer)]
    return last


# === 通知内容精简辅助函数 ===

def _normalize_title_for_merge(title: str) -> str:
    """归一化标题，用于通知去重合并"""
    if not title:
        return ""
    text = str(title).lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[\[\]【】（）()《》<>「」\"'“”‘’]", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text.strip()


def _split_source_names(source_name: str) -> List[str]:
    """拆分合并后的来源名称"""
    if not source_name:
        return []
    parts = re.split(r"[/,，、|｜]+", str(source_name))
    return [p.strip() for p in parts if p and p.strip()]


def _merge_source_names(existing: str, incoming: str) -> str:
    """合并来源名称，避免重复且控制长度"""
    sources = []
    for source in _split_source_names(existing) + _split_source_names(incoming):
        if source not in sources:
            sources.append(source)
    if len(sources) <= 3:
        return "/".join(sources)
    return "/".join(sources[:3]) + f"等{len(sources)}源"


def _safe_int(value, default: int = 1) -> int:
    """安全转换整数"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _best_rank(item: Dict) -> int:
    """获取条目的最佳排名，越小价值越高"""
    ranks = item.get("ranks", [])
    numeric_ranks = [
        r for r in ranks
        if isinstance(r, int) and r > 0
    ]
    rank = item.get("rank")
    if isinstance(rank, int) and rank > 0:
        numeric_ranks.append(rank)
    return min(numeric_ranks) if numeric_ranks else 99


def _item_value_score(item: Dict) -> int:
    """通知高价值排序分数"""
    best_rank = _best_rank(item)
    count = item.get("count", 1)
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 1

    source_count = len(
        _split_source_names(item.get("source_name", ""))
        or _split_source_names(item.get("feed_name", ""))
    )
    has_url = bool(item.get("url") or item.get("mobile_url") or item.get("mobileUrl"))
    is_new = bool(item.get("is_new"))
    title_len = len(str(item.get("title", "")))

    score = 0
    score += max(0, 120 - min(best_rank, 120)) * 4
    score += min(max(count, 1), 20) * 8
    score += min(source_count, 5) * 10
    score += 25 if is_new else 0
    score += 5 if has_url else 0
    if 8 <= title_len <= 80:
        score += 8
    return score


def _has_rank_signal(item: Dict) -> bool:
    """判断条目是否带有热榜排名信号"""
    ranks = item.get("ranks", [])
    rank = item.get("rank")
    return bool(ranks) or (isinstance(rank, int) and rank > 0)


def _is_high_value_item(item: Dict) -> bool:
    """过滤明显低价值的热榜长尾"""
    if not _has_rank_signal(item):
        # RSS 等无排名内容无法按热榜长尾判断，交给排序和数量限制处理。
        return True

    count = _safe_int(item.get("count", 1), 1)
    source_count = len(
        _split_source_names(item.get("source_name", ""))
        or _split_source_names(item.get("feed_name", ""))
    )
    if _best_rank(item) > 50 and count <= 1 and source_count <= 1 and not item.get("is_new"):
        return False
    return True


def _merge_duplicate_items(items: List[Dict]) -> List[Dict]:
    """合并重复标题，保留价值最高的展示形态"""
    merged_by_key: Dict[str, Dict] = {}
    order = 0

    for item in items or []:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")
        key = _normalize_title_for_merge(title)
        if not key:
            continue

        candidate = item.copy()
        candidate["_original_index"] = order
        order += 1

        existing = merged_by_key.get(key)
        if not existing:
            merged_by_key[key] = candidate
            continue

        if _item_value_score(candidate) > _item_value_score(existing):
            base = candidate
            other = existing
        else:
            base = existing
            other = candidate

        base = base.copy()
        base["source_name"] = _merge_source_names(
            existing.get("source_name", ""), candidate.get("source_name", "")
        )
        base["feed_name"] = _merge_source_names(
            existing.get("feed_name", ""), candidate.get("feed_name", "")
        )
        base["count"] = _safe_int(existing.get("count", 1), 1) + _safe_int(
            candidate.get("count", 1), 1
        )
        base["is_new"] = bool(existing.get("is_new")) or bool(candidate.get("is_new"))

        ranks = []
        for rank in (existing.get("ranks", []) or []) + (candidate.get("ranks", []) or []):
            if isinstance(rank, int) and rank not in ranks:
                ranks.append(rank)
        if ranks:
            base["ranks"] = sorted(ranks)

        if not base.get("url") and other.get("url"):
            base["url"] = other.get("url")
        if not base.get("mobile_url") and other.get("mobile_url"):
            base["mobile_url"] = other.get("mobile_url")
        if not base.get("mobileUrl") and other.get("mobileUrl"):
            base["mobileUrl"] = other.get("mobileUrl")

        first_times = [
            t for t in (existing.get("first_time"), candidate.get("first_time")) if t
        ]
        last_times = [
            t for t in (existing.get("last_time"), candidate.get("last_time")) if t
        ]
        if first_times:
            base["first_time"] = min(first_times)
        if last_times:
            base["last_time"] = max(last_times)

        timelines = []
        for timeline in (existing.get("rank_timeline"), candidate.get("rank_timeline")):
            if isinstance(timeline, list):
                timelines.extend(timeline)
        if timelines:
            base["rank_timeline"] = timelines

        base["_original_index"] = min(
            existing.get("_original_index", 0), candidate.get("_original_index", 0)
        )
        merged_by_key[key] = base

    return list(merged_by_key.values())


def _select_high_value_items(items: List[Dict], limit: int) -> List[Dict]:
    """合并重复并选择高价值条目"""
    merged = _merge_duplicate_items(items)
    high_value = [item for item in merged if _is_high_value_item(item)]
    pool = high_value if high_value else merged
    ranked = sorted(
        pool,
        key=lambda item: (-_item_value_score(item), _best_rank(item), item.get("_original_index", 0)),
    )
    selected = ranked[:limit]
    for item in selected:
        item.pop("_original_index", None)
    return selected


def _compact_stat_groups(stats: List[Dict], limit: int) -> List[Dict]:
    """精简关键词统计分组"""
    compacted = []
    for stat in stats or []:
        titles = _select_high_value_items(stat.get("titles", []), limit)
        if not titles:
            continue
        item = stat.copy()
        original_count = item.get("count", len(stat.get("titles", [])))
        try:
            original_count = max(int(original_count), len(stat.get("titles", [])))
        except (TypeError, ValueError):
            original_count = len(stat.get("titles", []))
        item["titles"] = titles
        item["original_count"] = original_count
        item["display_count"] = len(titles)
        compacted.append(item)
    return compacted


def _format_count_display(original_count: int, display_count: int) -> str:
    """显示精选数量"""
    if original_count > display_count:
        return f"{display_count}/{original_count}"
    return str(display_count)


def _format_group_header(
    format_type: str,
    icon: str,
    sequence_display: str,
    label: str,
    count_text: str,
    count: int,
) -> str:
    """格式化紧凑分组标题"""
    if format_type == "feishu":
        seq = f"<font color='grey'>{sequence_display}</font>"
        if count >= 10:
            count_part = f"<font color='red'>{count_text}</font>"
        elif count >= 5:
            count_part = f"<font color='orange'>{count_text}</font>"
        else:
            count_part = count_text
        return f"{icon} {seq} **{label}** · {count_part}条\n"

    if format_type == "telegram":
        return f"{icon} {sequence_display} {label} · {count_text}条\n"

    if format_type == "slack":
        label_part = f"*{label}*"
        count_part = f"*{count_text}*" if count >= 5 else count_text
        return f"{icon} {sequence_display} {label_part} · {count_part}条\n"

    if format_type == "generic_webhook":
        label_part = f"**{label}**"
        count_part = f"**{count_text}**" if count >= 5 else count_text
        return f"{icon} {sequence_display} {label_part} · {count_part}条\n"

    label_part = f"**{label}**"
    count_part = f"**{count_text}**" if count >= 5 else count_text
    return f"{icon} {sequence_display} {label_part} · {count_part}条\n"


def _format_source_header(
    format_type: str,
    source_name: str,
    count_text: str,
) -> str:
    """格式化紧凑来源标题"""
    if format_type == "telegram":
        return f"▸ {source_name} · {count_text}条\n"
    if format_type == "slack":
        return f"▸ *{source_name}* · {count_text}条\n"
    if format_type == "generic_webhook":
        return f"▸ **{source_name}** · {count_text}条\n"
    return f"▸ **{source_name}** · {count_text}条\n"


def _format_section_header(
    format_type: str,
    icon: str,
    title: str,
    count_text: str,
) -> str:
    """格式化紧凑区块标题"""
    if format_type == "telegram":
        return f"{icon} {title} · {count_text}条\n"
    if format_type == "slack":
        return f"{icon} *{title}* · {count_text}条\n"
    if format_type == "generic_webhook":
        return f"{icon} **{title}** · {count_text}条\n"
    return f"{icon} **{title}** · {count_text}条\n"


def _prefix_section_header(
    section_header: str,
    format_type: str,
    feishu_separator: str,
    add_separator: bool,
    has_content: bool,
) -> str:
    """根据当前位置为区块标题添加紧凑分隔"""
    if not add_separator or not has_content:
        return section_header
    if format_type == "feishu":
        return f"\n{feishu_separator}\n{section_header}"
    if format_type == "dingtalk":
        return f"\n---\n{section_header}"
    return f"\n{section_header}"


def _compact_notification_layout(content: str) -> str:
    """压缩通知排版，保留分段但移除列表内大空行"""
    if not content:
        return content

    # 列表项之间不留空行，减少滚动成本。
    content = re.sub(r"(?m)^(\s*\d+\. .+)\n{2,}(?=\s*\d+\. )", r"\1\n", content)

    # 标题后最多一行空白。
    content = re.sub(r"([：:])\n{3,}", r"\1\n\n", content)

    # 全局最多保留一个空行。
    content = re.sub(r"\n{4,}", "\n\n", content)
    content = re.sub(r"\n{3}", "\n\n", content)
    content = re.sub(r"(?m)^(\s*\d+\. .+)\n\n(?=▸ )", r"\1\n", content)
    return content.strip("\n")


# 默认批次大小配置
DEFAULT_BATCH_SIZES = {
    "dingtalk": 20000,
    "feishu": 29000,
    "ntfy": 3800,
    "generic_webhook": 2000,
    "default": 4000,
}

# 默认区域顺序
DEFAULT_REGION_ORDER = ["hotlist", "rss", "new_items", "standalone", "ai_analysis"]


def split_content_into_batches(
    report_data: Dict,
    format_type: str,
    update_info: Optional[Dict] = None,
    max_bytes: Optional[int] = None,
    mode: str = "daily",
    batch_sizes: Optional[Dict[str, int]] = None,
    feishu_separator: str = "---",
    region_order: Optional[List[str]] = None,
    get_time_func: Optional[Callable[[], datetime]] = None,
    rss_items: Optional[list] = None,
    rss_new_items: Optional[list] = None,
    timezone: str = DEFAULT_TIMEZONE,
    display_mode: str = "keyword",
    ai_content: Optional[str] = None,
    standalone_data: Optional[Dict] = None,
    rank_threshold: int = 10,
    ai_stats: Optional[Dict] = None,
    report_type: str = "热点分析报告",
    show_new_section: bool = True,
) -> List[str]:
    """分批处理消息内容，确保词组标题+至少第一条新闻的完整性（支持热榜+RSS合并+AI分析+独立展示区）

    热榜统计与RSS统计并列显示，热榜新增与RSS新增并列显示。
    region_order 控制各区域的显示顺序。
    AI分析内容根据 region_order 中的位置显示。
    独立展示区根据 region_order 中的位置显示。

    Args:
        report_data: 报告数据字典，包含 stats, new_titles, failed_ids, total_new_count
        format_type: 格式类型 (feishu, dingtalk, wework, telegram, ntfy, bark, slack)
        update_info: 版本更新信息（可选）
        max_bytes: 最大字节数（可选，如果不指定则使用默认配置）
        mode: 报告模式 (daily, incremental, current)
        batch_sizes: 批次大小配置字典（可选）
        feishu_separator: 飞书消息分隔符
        region_order: 区域显示顺序列表
        get_time_func: 获取当前时间的函数（可选）
        rss_items: RSS 统计条目列表（按源分组，用于合并推送）
        rss_new_items: RSS 新增条目列表（可选，用于新增区块）
        timezone: 时区名称（用于 RSS 时间格式化）
        display_mode: 显示模式 (keyword=按关键词分组, platform=按平台分组)
        ai_content: AI 分析内容（已渲染的字符串，可选）
        standalone_data: 独立展示区数据（可选），包含 platforms 和 rss_feeds 列表
        ai_stats: AI 分析统计数据（可选），包含 total_news, analyzed_news, max_news_limit 等

    Returns:
        分批后的消息内容列表
    """
    if region_order is None:
        region_order = DEFAULT_REGION_ORDER
    # 合并批次大小配置
    sizes = {**DEFAULT_BATCH_SIZES, **(batch_sizes or {})}

    if max_bytes is None:
        if format_type == "dingtalk":
            max_bytes = sizes.get("dingtalk", 20000)
        elif format_type == "feishu":
            max_bytes = sizes.get("feishu", 29000)
        elif format_type == "ntfy":
            max_bytes = sizes.get("ntfy", 3800)
        elif format_type == "generic_webhook":
            max_bytes = sizes.get("generic_webhook", 2000)
        else:
            max_bytes = sizes.get("default", 4000)

    batches = []

    total_hotlist_count = sum(
        len(stat["titles"]) for stat in report_data["stats"] if stat["count"] > 0
    )
    total_titles = total_hotlist_count
    
    # 累加 RSS 条目数
    if rss_items:
        total_titles += sum(stat.get("count", 0) for stat in rss_items)

    now = get_time_func() if get_time_func else datetime.now()

    # 构建头部信息
    base_header = ""

    # 格式化粗体标记
    if format_type == "slack":
        b_s, b_e = "*", "*"
    elif format_type == "telegram":
        b_s, b_e = "", ""
    else:
        b_s, b_e = "**", "**"

    # 提取统计数据
    hotlist_total = report_data.get("hotlist_total", total_hotlist_count)
    new_count = report_data.get("total_new_count", 0)
    platform_total = report_data.get("platform_total", 0)
    failed_count = len(report_data.get("failed_ids", []))
    platform_success = platform_total - failed_count if platform_total else 0
    rss_matched = report_data.get("rss_matched_count", 0)
    rss_total_items = report_data.get("rss_total_count", 0)
    rss_source_total = report_data.get("rss_source_total", 0)
    rss_source_failed = report_data.get("rss_source_failed", 0)
    rss_source_success = max(0, rss_source_total - rss_source_failed)

    # === 顶部摘要：压缩为 3 行 ===
    rss_new_count = sum(len(stat.get("titles", [])) for stat in (rss_new_items or []))
    total_new = new_count + rss_new_count

    overview_parts = [f"总 {total_titles}"]
    if total_new > 0:
        overview_parts.append(f"新 {new_count}+{rss_new_count}")
    base_header += f"{b_s}概览：{b_e}" + " | ".join(overview_parts) + "\n"

    data_parts = [f"热榜 {total_hotlist_count}/{hotlist_total}"]
    if platform_total > 0:
        data_parts.append(f"平台 {platform_success}/{platform_total}")
    if rss_source_total > 0:
        data_parts.append(f"RSS {rss_matched}/{rss_total_items}")
        data_parts.append(f"RSS源 {rss_source_success}/{rss_source_total}")

    if standalone_data:
        sa_platform_count = sum(len(p.get("items", [])) for p in standalone_data.get("platforms", []))
        sa_rss_count = sum(len(f.get("items", [])) for f in standalone_data.get("rss_feeds", []))
        sa_total = sa_platform_count + sa_rss_count
        if sa_total > 0:
            sa_parts = []
            if sa_platform_count > 0:
                sa_parts.append(f"热榜 {sa_platform_count}")
            if sa_rss_count > 0:
                sa_parts.append(f"RSS {sa_rss_count}")
            data_parts.append(f"独立 {sa_total}({' + '.join(sa_parts)})")

    standalone_analyzed = ai_stats.get("standalone_analyzed", 0) if ai_stats else 0
    ai_has_data = ai_stats and (ai_stats.get("analyzed_news", 0) > 0 or standalone_analyzed > 0)
    if ai_has_data:
        hotlist_analyzed = _safe_int(ai_stats.get("hotlist_analyzed", 0), 0)
        rss_analyzed = _safe_int(ai_stats.get("rss_analyzed", 0), 0)
        standalone_analyzed = _safe_int(standalone_analyzed, 0)
        analyzed_total = _safe_int(ai_stats.get("analyzed_news", 0), 0)
        ai_mode_val = ai_stats.get("ai_mode", "")

        if hotlist_analyzed == 0 and rss_analyzed == 0 and standalone_analyzed == 0 and analyzed_total > 0:
            hotlist_analyzed = analyzed_total

        ai_parts = []
        if hotlist_analyzed > 0:
            ai_parts.append(f"热榜{hotlist_analyzed}")
        if ai_stats.get("include_rss", True) and rss_analyzed > 0:
            ai_parts.append(f"RSS{rss_analyzed}")
        if ai_stats.get("include_standalone", False) and standalone_analyzed > 0:
            ai_parts.append(f"独立{standalone_analyzed}")
        ai_display = " + ".join(ai_parts) if ai_parts else str(analyzed_total or 0)

        mode_suffix = ""
        if ai_mode_val and ai_mode_val != mode:
            mode_map = {"daily": "全天汇总", "current": "当前榜单", "incremental": "增量分析"}
            mode_suffix = f" [{mode_map.get(ai_mode_val, ai_mode_val)}]"

        data_parts.append(f"AI {ai_display}{mode_suffix}")

    base_header += f"{b_s}数据：{b_e}" + " | ".join(data_parts) + "\n"

    meta_parts = [report_type, now.strftime("%m-%d %H:%M")]
    top_words = report_data.get("stats", [])[:3]
    if top_words:
        topics = " | ".join(f"{s['word']}({s['count']})" for s in top_words)
        meta_parts.append(f"热词 {topics}")

    base_header += f"{b_s}本轮：{b_e}" + " | ".join(meta_parts) + "\n"

    if format_type in ("feishu", "dingtalk"):
        base_header += "---\n"
    else:
        base_header += "\n"

    base_footer = ""
    if format_type in ("wework", "bark"):
        base_footer = f"\n\n\n> 更新时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"
        if update_info:
            base_footer += f"\n> TrendRadar 发现新版本 **{update_info['remote_version']}**，当前 **{update_info['current_version']}**"
    elif format_type == "telegram":
        base_footer = f"\n\n更新时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"
        if update_info:
            base_footer += f"\nTrendRadar 发现新版本 {update_info['remote_version']}，当前 {update_info['current_version']}"
    elif format_type == "ntfy":
        base_footer = f"\n\n> 更新时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"
        if update_info:
            base_footer += f"\n> TrendRadar 发现新版本 **{update_info['remote_version']}**，当前 **{update_info['current_version']}**"
    elif format_type == "feishu":
        base_footer = f"\n\n<font color='grey'>更新时间：{now.strftime('%Y-%m-%d %H:%M:%S')}</font>"
        if update_info:
            base_footer += f"\n<font color='grey'>TrendRadar 发现新版本 {update_info['remote_version']}，当前 {update_info['current_version']}</font>"
    elif format_type == "dingtalk":
        base_footer = f"\n\n> 更新时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"
        if update_info:
            base_footer += f"\n> TrendRadar 发现新版本 **{update_info['remote_version']}**，当前 **{update_info['current_version']}**"
    elif format_type == "slack":
        base_footer = f"\n\n_更新时间：{now.strftime('%Y-%m-%d %H:%M:%S')}_"
        if update_info:
            base_footer += f"\n_TrendRadar 发现新版本 *{update_info['remote_version']}*，当前 *{update_info['current_version']}_"

    # 根据 display_mode 选择统计标题
    stats_title = "热词精选" if display_mode == "keyword" else "热点精选"
    stats_header = ""
    if report_data["stats"]:
        stats_header = _format_section_header(
            format_type, "📊", stats_title, str(total_hotlist_count)
        )

    current_batch = base_header
    current_batch_has_content = False

    # 当没有热榜数据时的处理
    # 注意：如果有 ai_content，不应该返回"暂无匹配"消息，而应该继续处理 AI 内容
    if (
        not report_data["stats"]
        and not report_data["new_titles"]
        and not report_data["failed_ids"]
        and not ai_content  # 有 AI 内容时不返回"暂无匹配"
        and not rss_items  # 有 RSS 内容时也不返回
        and not standalone_data  # 有独立展示区数据时也不返回
    ):
        if mode == "incremental":
            mode_text = "增量模式下暂无新增匹配的热点词汇"
        elif mode == "current":
            mode_text = "当前榜单模式下暂无匹配的热点词汇"
        else:
            mode_text = "暂无匹配的热点词汇"
        simple_content = f"📭 {mode_text}\n\n"
        final_content = base_header + simple_content + base_footer
        batches.append(final_content)
        return [_compact_notification_layout(batch) for batch in batches]

    # 定义处理热点词汇统计的函数
    def process_stats_section(current_batch, current_batch_has_content, batches, add_separator=True):
        """处理热点词汇统计"""
        display_stats = _compact_stat_groups(
            report_data.get("stats", []), NOTIFICATION_MAX_TITLES_PER_GROUP
        )
        if not display_stats:
            return current_batch, current_batch_has_content, batches

        total_count = len(display_stats)

        # 根据 add_separator 决定是否添加前置分割线
        actual_stats_header = _prefix_section_header(
            stats_header, format_type, feishu_separator, add_separator, current_batch_has_content
        )

        # 添加统计标题
        test_content = current_batch + actual_stats_header
        if (
            len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
            < max_bytes
        ):
            current_batch = test_content
            current_batch_has_content = True
        else:
            if current_batch_has_content:
                _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
            current_batch = _safe_new_batch(
                base_header + stats_header, base_footer, max_bytes, base_header, batches
            )
            current_batch_has_content = True

        # 逐个处理词组（确保词组标题+第一条新闻的原子性）
        for i, stat in enumerate(display_stats):
            word = stat["word"]
            count = stat.get("original_count", stat.get("count", len(stat.get("titles", []))))
            display_count = stat.get("display_count", len(stat.get("titles", [])))
            count_text = _format_count_display(count, display_count)
            sequence_display = f"[{i + 1}/{total_count}]"

            icon = "🔥" if count >= 10 else "📈" if count >= 5 else "📌"
            word_header = _format_group_header(
                format_type, icon, sequence_display, word, count_text, count
            )

            # 构建第一条新闻
            # display_mode: keyword=显示来源, platform=显示关键词
            show_source = display_mode == "keyword"
            show_keyword = display_mode == "platform"
            first_news_line = ""
            if stat["titles"]:
                first_title_data = stat["titles"][0]
                if format_type in ("wework", "bark", "generic_webhook"):
                    formatted_title = format_title_for_platform(
                        "wework", first_title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "telegram":
                    formatted_title = format_title_for_platform(
                        "telegram", first_title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "ntfy":
                    formatted_title = format_title_for_platform(
                        "ntfy", first_title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "feishu":
                    formatted_title = format_title_for_platform(
                        "feishu", first_title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "dingtalk":
                    formatted_title = format_title_for_platform(
                        "dingtalk", first_title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "slack":
                    formatted_title = format_title_for_platform(
                        "slack", first_title_data, show_source=show_source, show_keyword=show_keyword
                    )
                else:
                    formatted_title = f"{first_title_data['title']}"

                first_news_line = f"  1. {formatted_title}\n"

            # 原子性检查：词组标题+第一条新闻必须一起处理
            word_with_first_news = word_header + first_news_line
            test_content = current_batch + word_with_first_news

            if (
                len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
                >= max_bytes
            ):
                if current_batch_has_content:
                    _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                current_batch = _safe_new_batch(
                    base_header + stats_header + word_with_first_news,
                    base_footer, max_bytes, base_header, batches
                )
                current_batch_has_content = True
                start_index = 1
            else:
                current_batch = test_content
                current_batch_has_content = True
                start_index = 1

            # 处理剩余新闻条目
            for j in range(start_index, len(stat["titles"])):
                title_data = stat["titles"][j]
                if format_type in ("wework", "bark", "generic_webhook"):
                    formatted_title = format_title_for_platform(
                        "wework", title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "telegram":
                    formatted_title = format_title_for_platform(
                        "telegram", title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "ntfy":
                    formatted_title = format_title_for_platform(
                        "ntfy", title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "feishu":
                    formatted_title = format_title_for_platform(
                        "feishu", title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "dingtalk":
                    formatted_title = format_title_for_platform(
                        "dingtalk", title_data, show_source=show_source, show_keyword=show_keyword
                    )
                elif format_type == "slack":
                    formatted_title = format_title_for_platform(
                        "slack", title_data, show_source=show_source, show_keyword=show_keyword
                    )
                else:
                    formatted_title = f"{title_data['title']}"

                news_line = f"  {j + 1}. {formatted_title}\n"

                test_content = current_batch + news_line
                if (
                    len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
                    >= max_bytes
                ):
                    if current_batch_has_content:
                        _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                    current_batch = _safe_new_batch(
                        base_header + stats_header + word_header + news_line,
                        base_footer, max_bytes, base_header, batches
                    )
                    current_batch_has_content = True
                else:
                    current_batch = test_content
                    current_batch_has_content = True

            # 词组间分隔符
            if i < len(display_stats) - 1:
                separator = "\n"

                test_content = current_batch + separator
                if (
                    len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
                    < max_bytes
                ):
                    current_batch = test_content

        return current_batch, current_batch_has_content, batches

    # 定义处理新增新闻的函数
    def process_new_titles_section(current_batch, current_batch_has_content, batches, add_separator=True):
        """处理新增新闻"""
        if not show_new_section or not report_data["new_titles"]:
            return current_batch, current_batch_has_content, batches

        display_sources = []
        for source_data in report_data.get("new_titles", []):
            titles = _select_high_value_items(
                source_data.get("titles", []), NOTIFICATION_MAX_NEW_TITLES_PER_SOURCE
            )
            if not titles:
                continue
            display_sources.append({
                **source_data,
                "titles": titles,
                "original_count": len(source_data.get("titles", [])),
                "display_count": len(titles),
            })
        if not display_sources:
            return current_batch, current_batch_has_content, batches

        original_total = report_data.get("total_new_count", 0)
        display_total = sum(len(source.get("titles", [])) for source in display_sources)
        count_text = _format_count_display(original_total, display_total)
        new_header = _prefix_section_header(
            _format_section_header(format_type, "🆕", "新增热点", count_text),
            format_type,
            feishu_separator,
            add_separator,
            current_batch_has_content,
        )

        test_content = current_batch + new_header
        if (
            len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
            >= max_bytes
        ):
            if current_batch_has_content:
                _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
            current_batch = _safe_new_batch(
                base_header + new_header, base_footer, max_bytes, base_header, batches
            )
            current_batch_has_content = True
        else:
            current_batch = test_content
            current_batch_has_content = True

        # 逐个处理新增新闻来源
        for source_data in display_sources:
            original_count = source_data.get("original_count", len(source_data["titles"]))
            display_count = source_data.get("display_count", len(source_data["titles"]))
            count_text = _format_count_display(original_count, display_count)
            source_header = _format_source_header(
                format_type, source_data["source_name"], count_text
            )

            # 构建第一条新增新闻
            first_news_line = ""
            if source_data["titles"]:
                first_title_data = source_data["titles"][0]
                title_data_copy = first_title_data.copy()
                title_data_copy["is_new"] = False

                if format_type in ("wework", "bark", "generic_webhook"):
                    formatted_title = format_title_for_platform(
                        "wework", title_data_copy, show_source=False
                    )
                elif format_type == "telegram":
                    formatted_title = format_title_for_platform(
                        "telegram", title_data_copy, show_source=False
                    )
                elif format_type == "feishu":
                    formatted_title = format_title_for_platform(
                        "feishu", title_data_copy, show_source=False
                    )
                elif format_type == "dingtalk":
                    formatted_title = format_title_for_platform(
                        "dingtalk", title_data_copy, show_source=False
                    )
                elif format_type == "slack":
                    formatted_title = format_title_for_platform(
                        "slack", title_data_copy, show_source=False
                    )
                else:
                    formatted_title = f"{title_data_copy['title']}"

                first_news_line = f"  1. {formatted_title}\n"

            # 原子性检查：来源标题+第一条新闻
            source_with_first_news = source_header + first_news_line
            test_content = current_batch + source_with_first_news

            if (
                len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
                >= max_bytes
            ):
                if current_batch_has_content:
                    _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                current_batch = _safe_new_batch(
                    base_header + new_header + source_with_first_news,
                    base_footer, max_bytes, base_header, batches
                )
                current_batch_has_content = True
                start_index = 1
            else:
                current_batch = test_content
                current_batch_has_content = True
                start_index = 1

            # 处理剩余新增新闻
            for j in range(start_index, len(source_data["titles"])):
                title_data = source_data["titles"][j]
                title_data_copy = title_data.copy()
                title_data_copy["is_new"] = False

                if format_type in ("wework", "generic_webhook"):
                    formatted_title = format_title_for_platform(
                        "wework", title_data_copy, show_source=False
                    )
                elif format_type == "telegram":
                    formatted_title = format_title_for_platform(
                        "telegram", title_data_copy, show_source=False
                    )
                elif format_type == "feishu":
                    formatted_title = format_title_for_platform(
                        "feishu", title_data_copy, show_source=False
                    )
                elif format_type == "dingtalk":
                    formatted_title = format_title_for_platform(
                        "dingtalk", title_data_copy, show_source=False
                    )
                elif format_type == "slack":
                    formatted_title = format_title_for_platform(
                        "slack", title_data_copy, show_source=False
                    )
                else:
                    formatted_title = f"{title_data_copy['title']}"

                news_line = f"  {j + 1}. {formatted_title}\n"

                test_content = current_batch + news_line
                if (
                    len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
                    >= max_bytes
                ):
                    if current_batch_has_content:
                        _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                    current_batch = _safe_new_batch(
                        base_header + new_header + source_header + news_line,
                        base_footer, max_bytes, base_header, batches
                    )
                    current_batch_has_content = True
                else:
                    current_batch = test_content
                    current_batch_has_content = True

            current_batch += "\n"

        return current_batch, current_batch_has_content, batches

    # 定义处理 AI 分析的函数
    def process_ai_section(current_batch, current_batch_has_content, batches, add_separator=True):
        """处理 AI 分析内容"""
        nonlocal ai_content
        if not ai_content:
            return current_batch, current_batch_has_content, batches

        # 根据 add_separator 决定是否添加前置分割线
        ai_separator = ""
        if add_separator and current_batch_has_content:
            # 需要添加分割线
            if format_type == "feishu":
                ai_separator = f"\n{feishu_separator}\n"
            elif format_type == "dingtalk":
                ai_separator = "\n---\n"
            else:
                ai_separator = "\n"
        # 如果不需要分割线，ai_separator 保持为空字符串

        # 尝试将 AI 内容添加到当前批次
        test_content = current_batch + ai_separator + ai_content
        if (
            len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
            < max_bytes
        ):
            current_batch = test_content
            current_batch_has_content = True
        else:
            if current_batch_has_content:
                _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)

            # AI 内容可能很长，按行拆分成多个批次
            footer_size = len(base_footer.encode("utf-8"))
            header_size = len(base_header.encode("utf-8"))
            available = max_bytes - footer_size - header_size

            ai_lines = ai_content.split("\n")
            current_batch = base_header
            current_batch_has_content = False

            for line in ai_lines:
                test_line = line + "\n" if not line.endswith("\n") else line
                test_content = current_batch + test_line
                if len(test_content.encode("utf-8")) + footer_size >= max_bytes and current_batch_has_content:
                    _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                    current_batch = base_header + test_line
                else:
                    current_batch = test_content
                current_batch_has_content = True

        return current_batch, current_batch_has_content, batches

    # 定义处理独立展示区的函数
    def process_standalone_section_wrapper(current_batch, current_batch_has_content, batches, add_separator=True):
        """处理独立展示区"""
        if not standalone_data:
            return current_batch, current_batch_has_content, batches
        return _process_standalone_section(
            standalone_data, format_type, feishu_separator, base_header, base_footer,
            max_bytes, current_batch, current_batch_has_content, batches, timezone,
            rank_threshold, add_separator
        )

    # 定义处理 RSS 统计的函数
    def process_rss_stats_wrapper(current_batch, current_batch_has_content, batches, add_separator=True):
        """处理 RSS 统计"""
        if not rss_items:
            return current_batch, current_batch_has_content, batches
        return _process_rss_stats_section(
            rss_items, format_type, feishu_separator, base_header, base_footer,
            max_bytes, current_batch, current_batch_has_content, batches, timezone,
            add_separator
        )

    # 定义处理 RSS 新增的函数
    def process_rss_new_wrapper(current_batch, current_batch_has_content, batches, add_separator=True):
        """处理 RSS 新增"""
        if not rss_new_items:
            return current_batch, current_batch_has_content, batches
        return _process_rss_new_titles_section(
            rss_new_items, format_type, feishu_separator, base_header, base_footer,
            max_bytes, current_batch, current_batch_has_content, batches, timezone,
            add_separator
        )

    # 按 region_order 顺序处理各区域
    # 记录是否已有区域内容（用于决定是否添加分割线）
    has_region_content = False

    for region in region_order:
        # 记录处理前的状态，用于判断该区域是否产生了内容
        batch_before = current_batch
        has_content_before = current_batch_has_content
        batches_len_before = len(batches)

        # 决定是否需要添加分割线（第一个有内容的区域不需要）
        add_separator = has_region_content

        if region == "hotlist":
            # 处理热榜统计
            current_batch, current_batch_has_content, batches = process_stats_section(
                current_batch, current_batch_has_content, batches, add_separator
            )
        elif region == "rss":
            # 处理 RSS 统计
            current_batch, current_batch_has_content, batches = process_rss_stats_wrapper(
                current_batch, current_batch_has_content, batches, add_separator
            )
        elif region == "new_items":
            # 处理热榜新增
            current_batch, current_batch_has_content, batches = process_new_titles_section(
                current_batch, current_batch_has_content, batches, add_separator
            )
            # 处理 RSS 新增（跟随 new_items，继承 add_separator 逻辑）
            # 如果热榜新增产生了内容，RSS 新增需要分割线
            new_batch_changed = (
                current_batch != batch_before or
                current_batch_has_content != has_content_before or
                len(batches) != batches_len_before
            )
            rss_new_separator = new_batch_changed or has_region_content
            current_batch, current_batch_has_content, batches = process_rss_new_wrapper(
                current_batch, current_batch_has_content, batches, rss_new_separator
            )
        elif region == "standalone":
            # 处理独立展示区
            current_batch, current_batch_has_content, batches = process_standalone_section_wrapper(
                current_batch, current_batch_has_content, batches, add_separator
            )
        elif region == "ai_analysis":
            # 处理 AI 分析
            current_batch, current_batch_has_content, batches = process_ai_section(
                current_batch, current_batch_has_content, batches, add_separator
            )

        # 检查该区域是否产生了内容
        region_produced_content = (
            current_batch != batch_before or
            current_batch_has_content != has_content_before or
            len(batches) != batches_len_before
        )
        if region_produced_content:
            has_region_content = True

    if report_data["failed_ids"]:
        failed_header = _prefix_section_header(
            _format_section_header(
                format_type, "⚠️", "失败平台", str(len(report_data["failed_ids"]))
            ),
            format_type,
            feishu_separator,
            True,
            current_batch_has_content,
        )

        test_content = current_batch + failed_header
        if (
            len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
            >= max_bytes
        ):
            if current_batch_has_content:
                _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
            current_batch = _safe_new_batch(
                base_header + failed_header, base_footer, max_bytes, base_header, batches
            )
            current_batch_has_content = True
        else:
            current_batch = test_content
            current_batch_has_content = True

        for i, id_value in enumerate(report_data["failed_ids"], 1):
            if format_type == "feishu":
                failed_line = f"  • <font color='red'>{id_value}</font>\n"
            elif format_type == "dingtalk":
                failed_line = f"  • **{id_value}**\n"
            else:
                failed_line = f"  • {id_value}\n"

            test_content = current_batch + failed_line
            if (
                len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8"))
                >= max_bytes
            ):
                if current_batch_has_content:
                    _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                current_batch = _safe_new_batch(
                    base_header + failed_header + failed_line,
                    base_footer, max_bytes, base_header, batches
                )
                current_batch_has_content = True
            else:
                current_batch = test_content
                current_batch_has_content = True

    # 完成最后批次
    if current_batch_has_content:
        _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)

    return [_compact_notification_layout(batch) for batch in batches]


def _process_rss_stats_section(
    rss_stats: list,
    format_type: str,
    feishu_separator: str,
    base_header: str,
    base_footer: str,
    max_bytes: int,
    current_batch: str,
    current_batch_has_content: bool,
    batches: List[str],
    timezone: str = DEFAULT_TIMEZONE,
    add_separator: bool = True,
) -> tuple:
    """处理 RSS 统计区块（按关键词分组，与热榜统计格式一致）

    Args:
        rss_stats: RSS 关键词统计列表，格式与热榜 stats 一致：
            [{"word": "AI", "count": 5, "titles": [...]}]
        format_type: 格式类型
        feishu_separator: 飞书分隔符
        base_header: 基础头部
        base_footer: 基础尾部
        max_bytes: 最大字节数
        current_batch: 当前批次内容
        current_batch_has_content: 当前批次是否有内容
        batches: 已完成的批次列表
        timezone: 时区名称
        add_separator: 是否在区块前添加分割线（第一个区域时为 False）

    Returns:
        (current_batch, current_batch_has_content, batches) 元组
    """
    if not rss_stats:
        return current_batch, current_batch_has_content, batches

    display_stats = _compact_stat_groups(
        rss_stats, NOTIFICATION_MAX_TITLES_PER_GROUP
    )
    if not display_stats:
        return current_batch, current_batch_has_content, batches

    # 计算总条目数
    total_items = sum(stat["count"] for stat in rss_stats)
    total_keywords = len(display_stats)
    display_items = sum(len(stat.get("titles", [])) for stat in display_stats)
    rss_count_text = _format_count_display(total_items, display_items)

    # RSS 统计区块标题（根据 add_separator 决定是否添加前置分割线）
    rss_header = _prefix_section_header(
        _format_section_header(format_type, "📰", "RSS精选", rss_count_text),
        format_type,
        feishu_separator,
        add_separator,
        current_batch_has_content,
    )

    # 添加 RSS 标题
    test_content = current_batch + rss_header
    if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) < max_bytes:
        current_batch = test_content
        current_batch_has_content = True
    else:
        if current_batch_has_content:
            _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
        current_batch = _safe_new_batch(
            base_header + rss_header, base_footer, max_bytes, base_header, batches
        )
        current_batch_has_content = True

    # 逐个处理关键词组（与热榜一致）
    for i, stat in enumerate(display_stats):
        word = stat["word"]
        count = stat.get("original_count", stat.get("count", len(stat.get("titles", []))))
        display_count = stat.get("display_count", len(stat.get("titles", [])))
        count_text = _format_count_display(count, display_count)
        sequence_display = f"[{i + 1}/{total_keywords}]"

        icon = "🔥" if count >= 10 else "📈" if count >= 5 else "📌"
        word_header = _format_group_header(
            format_type, icon, sequence_display, word, count_text, count
        )

        # 构建第一条新闻（使用 format_title_for_platform）
        first_news_line = ""
        if stat["titles"]:
            first_title_data = stat["titles"][0]
            if format_type in ("wework", "bark", "generic_webhook"):
                formatted_title = format_title_for_platform("wework", first_title_data, show_source=True)
            elif format_type == "telegram":
                formatted_title = format_title_for_platform("telegram", first_title_data, show_source=True)
            elif format_type == "ntfy":
                formatted_title = format_title_for_platform("ntfy", first_title_data, show_source=True)
            elif format_type == "feishu":
                formatted_title = format_title_for_platform("feishu", first_title_data, show_source=True)
            elif format_type == "dingtalk":
                formatted_title = format_title_for_platform("dingtalk", first_title_data, show_source=True)
            elif format_type == "slack":
                formatted_title = format_title_for_platform("slack", first_title_data, show_source=True)
            else:
                formatted_title = f"{first_title_data['title']}"

            first_news_line = f"  1. {formatted_title}\n"

        # 原子性检查：关键词标题 + 第一条新闻必须一起处理
        word_with_first_news = word_header + first_news_line
        test_content = current_batch + word_with_first_news

        if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
            if current_batch_has_content:
                _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
            current_batch = _safe_new_batch(
                base_header + rss_header + word_with_first_news,
                base_footer, max_bytes, base_header, batches
            )
            current_batch_has_content = True
            start_index = 1
        else:
            current_batch = test_content
            current_batch_has_content = True
            start_index = 1

        # 处理剩余新闻条目
        for j in range(start_index, len(stat["titles"])):
            title_data = stat["titles"][j]
            if format_type in ("wework", "bark", "generic_webhook"):
                formatted_title = format_title_for_platform("wework", title_data, show_source=True)
            elif format_type == "telegram":
                formatted_title = format_title_for_platform("telegram", title_data, show_source=True)
            elif format_type == "ntfy":
                formatted_title = format_title_for_platform("ntfy", title_data, show_source=True)
            elif format_type == "feishu":
                formatted_title = format_title_for_platform("feishu", title_data, show_source=True)
            elif format_type == "dingtalk":
                formatted_title = format_title_for_platform("dingtalk", title_data, show_source=True)
            elif format_type == "slack":
                formatted_title = format_title_for_platform("slack", title_data, show_source=True)
            else:
                formatted_title = f"{title_data['title']}"

            news_line = f"  {j + 1}. {formatted_title}\n"

            test_content = current_batch + news_line
            if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
                if current_batch_has_content:
                    _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                current_batch = _safe_new_batch(
                    base_header + rss_header + word_header + news_line,
                    base_footer, max_bytes, base_header, batches
                )
                current_batch_has_content = True
            else:
                current_batch = test_content
                current_batch_has_content = True

        # 关键词间分隔符
        if i < len(display_stats) - 1:
            separator = "\n"

            test_content = current_batch + separator
            if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) < max_bytes:
                current_batch = test_content

    return current_batch, current_batch_has_content, batches


def _process_rss_new_titles_section(
    rss_new_stats: list,
    format_type: str,
    feishu_separator: str,
    base_header: str,
    base_footer: str,
    max_bytes: int,
    current_batch: str,
    current_batch_has_content: bool,
    batches: List[str],
    timezone: str = DEFAULT_TIMEZONE,
    add_separator: bool = True,
) -> tuple:
    """处理 RSS 新增区块（按来源分组，与热榜新增格式一致）

    Args:
        rss_new_stats: RSS 新增关键词统计列表，格式与热榜 stats 一致：
            [{"word": "AI", "count": 5, "titles": [...]}]
        format_type: 格式类型
        feishu_separator: 飞书分隔符
        base_header: 基础头部
        base_footer: 基础尾部
        max_bytes: 最大字节数
        current_batch: 当前批次内容
        current_batch_has_content: 当前批次是否有内容
        batches: 已完成的批次列表
        timezone: 时区名称
        add_separator: 是否在区块前添加分割线（第一个区域时为 False）

    Returns:
        (current_batch, current_batch_has_content, batches) 元组
    """
    if not rss_new_stats:
        return current_batch, current_batch_has_content, batches

    # 从关键词分组中提取所有条目，重新按来源分组
    source_map = {}
    for stat in rss_new_stats:
        for title_data in stat.get("titles", []):
            source_name = title_data.get("source_name", "未知来源")
            if source_name not in source_map:
                source_map[source_name] = []
            source_map[source_name].append(title_data)

    if not source_map:
        return current_batch, current_batch_has_content, batches

    display_source_map = {}
    source_original_counts = {}
    for source_name, titles in source_map.items():
        selected = _select_high_value_items(titles, NOTIFICATION_MAX_NEW_TITLES_PER_SOURCE)
        if selected:
            display_source_map[source_name] = selected
            source_original_counts[source_name] = len(titles)

    if not display_source_map:
        return current_batch, current_batch_has_content, batches

    # 计算总条目数
    total_items = sum(len(titles) for titles in source_map.values())
    display_total = sum(len(titles) for titles in display_source_map.values())
    count_text = _format_count_display(total_items, display_total)

    # RSS 新增区块标题（根据 add_separator 决定是否添加前置分割线）
    new_header = _prefix_section_header(
        _format_section_header(format_type, "🆕", "RSS新增", count_text),
        format_type,
        feishu_separator,
        add_separator,
        current_batch_has_content,
    )

    # 添加 RSS 新增标题
    test_content = current_batch + new_header
    if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
        if current_batch_has_content:
            _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
        current_batch = _safe_new_batch(
            base_header + new_header, base_footer, max_bytes, base_header, batches
        )
        current_batch_has_content = True
    else:
        current_batch = test_content
        current_batch_has_content = True

    # 按来源分组显示（与热榜新增格式一致）
    source_list = list(display_source_map.items())
    for i, (source_name, titles) in enumerate(source_list):
        count = source_original_counts.get(source_name, len(titles))
        count_text = _format_count_display(count, len(titles))

        source_header = _format_source_header(format_type, source_name, count_text)

        # 构建第一条新闻（不显示来源，禁用 new emoji）
        first_news_line = ""
        if titles:
            first_title_data = titles[0].copy()
            first_title_data["is_new"] = False
            if format_type in ("wework", "bark", "generic_webhook"):
                formatted_title = format_title_for_platform("wework", first_title_data, show_source=False)
            elif format_type == "telegram":
                formatted_title = format_title_for_platform("telegram", first_title_data, show_source=False)
            elif format_type == "ntfy":
                formatted_title = format_title_for_platform("ntfy", first_title_data, show_source=False)
            elif format_type == "feishu":
                formatted_title = format_title_for_platform("feishu", first_title_data, show_source=False)
            elif format_type == "dingtalk":
                formatted_title = format_title_for_platform("dingtalk", first_title_data, show_source=False)
            elif format_type == "slack":
                formatted_title = format_title_for_platform("slack", first_title_data, show_source=False)
            else:
                formatted_title = f"{first_title_data['title']}"

            first_news_line = f"  1. {formatted_title}\n"

        # 原子性检查：来源标题 + 第一条新闻必须一起处理
        source_with_first_news = source_header + first_news_line
        test_content = current_batch + source_with_first_news

        if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
            if current_batch_has_content:
                _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
            current_batch = _safe_new_batch(
                base_header + new_header + source_with_first_news,
                base_footer, max_bytes, base_header, batches
            )
            current_batch_has_content = True
            start_index = 1
        else:
            current_batch = test_content
            current_batch_has_content = True
            start_index = 1

        # 处理剩余新闻条目（禁用 new emoji）
        for j in range(start_index, len(titles)):
            title_data = titles[j].copy()
            title_data["is_new"] = False
            if format_type in ("wework", "bark", "generic_webhook"):
                formatted_title = format_title_for_platform("wework", title_data, show_source=False)
            elif format_type == "telegram":
                formatted_title = format_title_for_platform("telegram", title_data, show_source=False)
            elif format_type == "ntfy":
                formatted_title = format_title_for_platform("ntfy", title_data, show_source=False)
            elif format_type == "feishu":
                formatted_title = format_title_for_platform("feishu", title_data, show_source=False)
            elif format_type == "dingtalk":
                formatted_title = format_title_for_platform("dingtalk", title_data, show_source=False)
            elif format_type == "slack":
                formatted_title = format_title_for_platform("slack", title_data, show_source=False)
            else:
                formatted_title = f"{title_data['title']}"

            news_line = f"  {j + 1}. {formatted_title}\n"

            test_content = current_batch + news_line
            if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
                if current_batch_has_content:
                    _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                current_batch = _safe_new_batch(
                    base_header + new_header + source_header + news_line,
                    base_footer, max_bytes, base_header, batches
                )
                current_batch_has_content = True
            else:
                current_batch = test_content
                current_batch_has_content = True

        # 来源间添加空行（与热榜新增格式一致）
        current_batch += "\n"

    return current_batch, current_batch_has_content, batches


def _format_rss_item_line(
    item: Dict,
    index: int,
    format_type: str,
    timezone: str = DEFAULT_TIMEZONE,
) -> str:
    """格式化单条 RSS 条目

    Args:
        item: RSS 条目字典
        index: 序号
        format_type: 格式类型
        timezone: 时区名称

    Returns:
        格式化后的条目行字符串
    """
    title = item.get("title", "")
    url = item.get("url", "")
    published_at = item.get("published_at", "")

    # 使用友好时间格式
    if published_at:
        friendly_time = format_iso_time_friendly(published_at, timezone, include_date=True)
    else:
        friendly_time = ""

    # 构建条目行
    if format_type == "feishu":
        if url:
            item_line = f"  {index}. [{title}]({url})"
        else:
            item_line = f"  {index}. {title}"
        if friendly_time:
            item_line += f" <font color='grey'>- {friendly_time}</font>"
    elif format_type == "telegram":
        if url:
            item_line = f"  {index}. {title} ({url})"
        else:
            item_line = f"  {index}. {title}"
        if friendly_time:
            item_line += f" - {friendly_time}"
    else:
        if url:
            item_line = f"  {index}. [{title}]({url})"
        else:
            item_line = f"  {index}. {title}"
        if friendly_time:
            item_line += f" `{friendly_time}`"

    item_line += "\n"
    return item_line


def _process_standalone_section(
    standalone_data: Dict,
    format_type: str,
    feishu_separator: str,
    base_header: str,
    base_footer: str,
    max_bytes: int,
    current_batch: str,
    current_batch_has_content: bool,
    batches: List[str],
    timezone: str = DEFAULT_TIMEZONE,
    rank_threshold: int = 10,
    add_separator: bool = True,
) -> tuple:
    """处理独立展示区区块

    独立展示区显示指定平台的完整热榜或 RSS 源内容，不受关键词过滤影响。
    热榜按原始排名排序，RSS 按发布时间排序。

    Args:
        standalone_data: 独立展示数据，格式：
            {
                "platforms": [{"id": "zhihu", "name": "知乎热榜", "items": [...]}],
                "rss_feeds": [{"id": "hacker-news", "name": "Hacker News", "items": [...]}]
            }
        format_type: 格式类型
        feishu_separator: 飞书分隔符
        base_header: 基础头部
        base_footer: 基础尾部
        max_bytes: 最大字节数
        current_batch: 当前批次内容
        current_batch_has_content: 当前批次是否有内容
        batches: 已完成的批次列表
        timezone: 时区名称
        rank_threshold: 排名高亮阈值
        add_separator: 是否在区块前添加分割线（第一个区域时为 False）

    Returns:
        (current_batch, current_batch_has_content, batches) 元组
    """
    if not standalone_data:
        return current_batch, current_batch_has_content, batches

    platforms = standalone_data.get("platforms", [])
    rss_feeds = standalone_data.get("rss_feeds", [])

    if not platforms and not rss_feeds:
        return current_batch, current_batch_has_content, batches

    display_platforms = []
    for platform in platforms:
        items = _select_high_value_items(
            platform.get("items", []), NOTIFICATION_MAX_STANDALONE_ITEMS_PER_SOURCE
        )
        if items:
            display_platforms.append({
                **platform,
                "items": items,
                "original_count": len(platform.get("items", [])),
                "display_count": len(items),
            })

    display_rss_feeds = []
    for feed in rss_feeds:
        items = _select_high_value_items(
            feed.get("items", []), NOTIFICATION_MAX_STANDALONE_ITEMS_PER_SOURCE
        )
        if items:
            display_rss_feeds.append({
                **feed,
                "items": items,
                "original_count": len(feed.get("items", [])),
                "display_count": len(items),
            })

    if not display_platforms and not display_rss_feeds:
        return current_batch, current_batch_has_content, batches

    # 计算总条目数
    total_platform_items = sum(len(p.get("items", [])) for p in platforms)
    total_rss_items = sum(len(f.get("items", [])) for f in rss_feeds)
    total_items = total_platform_items + total_rss_items
    display_items = (
        sum(len(p.get("items", [])) for p in display_platforms)
        + sum(len(f.get("items", [])) for f in display_rss_feeds)
    )
    count_text = _format_count_display(total_items, display_items)

    # 独立展示区标题（根据 add_separator 决定是否添加前置分割线）
    section_header = _prefix_section_header(
        _format_section_header(format_type, "📋", "独立速览", count_text),
        format_type,
        feishu_separator,
        add_separator,
        current_batch_has_content,
    )

    # 添加区块标题
    test_content = current_batch + section_header
    if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) < max_bytes:
        current_batch = test_content
        current_batch_has_content = True
    else:
        if current_batch_has_content:
            _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
        current_batch = _safe_new_batch(
            base_header + section_header, base_footer, max_bytes, base_header, batches
        )
        current_batch_has_content = True

    # 处理热榜平台
    for platform in display_platforms:
        platform_name = platform.get("name", platform.get("id", ""))
        items = platform.get("items", [])
        if not items:
            continue
        count_text = _format_count_display(
            platform.get("original_count", len(items)),
            platform.get("display_count", len(items)),
        )

        platform_header = _format_source_header(format_type, platform_name, count_text)

        # 构建第一条新闻
        first_item_line = ""
        if items:
            first_item_line = _format_standalone_platform_item(items[0], 1, format_type, rank_threshold)

        # 原子性检查
        platform_with_first = platform_header + first_item_line
        test_content = current_batch + platform_with_first

        if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
            if current_batch_has_content:
                _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
            current_batch = _safe_new_batch(
                base_header + section_header + platform_with_first,
                base_footer, max_bytes, base_header, batches
            )
            current_batch_has_content = True
            start_index = 1
        else:
            current_batch = test_content
            current_batch_has_content = True
            start_index = 1

        # 处理剩余条目
        for j in range(start_index, len(items)):
            item_line = _format_standalone_platform_item(items[j], j + 1, format_type, rank_threshold)

            test_content = current_batch + item_line
            if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
                if current_batch_has_content:
                    _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                current_batch = _safe_new_batch(
                    base_header + section_header + platform_header + item_line,
                    base_footer, max_bytes, base_header, batches
                )
                current_batch_has_content = True
            else:
                current_batch = test_content
                current_batch_has_content = True

        current_batch += "\n"

    # 处理 RSS 源
    for feed in display_rss_feeds:
        feed_name = feed.get("name", feed.get("id", ""))
        items = feed.get("items", [])
        if not items:
            continue
        count_text = _format_count_display(
            feed.get("original_count", len(items)),
            feed.get("display_count", len(items)),
        )

        feed_header = _format_source_header(format_type, feed_name, count_text)

        # 构建第一条 RSS
        first_item_line = ""
        if items:
            first_item_line = _format_standalone_rss_item(items[0], 1, format_type, timezone)

        # 原子性检查
        feed_with_first = feed_header + first_item_line
        test_content = current_batch + feed_with_first

        if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
            if current_batch_has_content:
                _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
            current_batch = _safe_new_batch(
                base_header + section_header + feed_with_first,
                base_footer, max_bytes, base_header, batches
            )
            current_batch_has_content = True
            start_index = 1
        else:
            current_batch = test_content
            current_batch_has_content = True
            start_index = 1

        # 处理剩余条目
        for j in range(start_index, len(items)):
            item_line = _format_standalone_rss_item(items[j], j + 1, format_type, timezone)

            test_content = current_batch + item_line
            if len(test_content.encode("utf-8")) + len(base_footer.encode("utf-8")) >= max_bytes:
                if current_batch_has_content:
                    _safe_append_batch(batches, current_batch, base_footer, max_bytes, base_header)
                current_batch = _safe_new_batch(
                    base_header + section_header + feed_header + item_line,
                    base_footer, max_bytes, base_header, batches
                )
                current_batch_has_content = True
            else:
                current_batch = test_content
                current_batch_has_content = True

        current_batch += "\n"

    return current_batch, current_batch_has_content, batches


def _format_standalone_platform_item(item: Dict, index: int, format_type: str, rank_threshold: int = 10) -> str:
    """格式化独立展示区的热榜条目（复用热点词汇统计区样式）

    Args:
        item: 热榜条目，包含 title, url, rank, ranks, first_time, last_time, count
        index: 序号
        format_type: 格式类型
        rank_threshold: 排名高亮阈值

    Returns:
        格式化后的条目行字符串
    """
    title = item.get("title", "")
    url = item.get("url", "") or item.get("mobileUrl", "")
    ranks = item.get("ranks", [])
    rank = item.get("rank", 0)
    first_time = item.get("first_time", "")
    last_time = item.get("last_time", "")
    count = item.get("count", 1)

    # 使用 format_rank_display 格式化排名（复用热点词汇统计区逻辑）
    # 如果没有 ranks 列表，用单个 rank 构造
    if not ranks and rank > 0:
        ranks = [rank]
    rank_timeline = item.get("rank_timeline")
    rank_display = format_rank_display(ranks, rank_threshold, format_type, rank_timeline=rank_timeline) if ranks else ""

    # 构建时间显示（用 ~ 连接范围，与热点词汇统计区一致）
    # 将 HH-MM 格式转换为 HH:MM 格式
    time_display = ""
    if first_time and last_time and first_time != last_time:
        first_time_display = convert_time_for_display(first_time)
        last_time_display = convert_time_for_display(last_time)
        time_display = f"{first_time_display}~{last_time_display}"
    elif first_time:
        time_display = convert_time_for_display(first_time)

    # 构建次数显示（格式为 (N次)，与热点词汇统计区一致）
    count_display = f"({count}次)" if count > 1 else ""

    # 根据格式类型构建条目行（复用热点词汇统计区样式）
    if format_type == "feishu":
        if url:
            item_line = f"  {index}. [{title}]({url})"
        else:
            item_line = f"  {index}. {title}"
        if rank_display:
            item_line += f" {rank_display}"
        if time_display:
            item_line += f" <font color='grey'>- {time_display}</font>"
        if count_display:
            item_line += f" <font color='green'>{count_display}</font>"

    elif format_type == "dingtalk":
        if url:
            item_line = f"  {index}. [{title}]({url})"
        else:
            item_line = f"  {index}. {title}"
        if rank_display:
            item_line += f" {rank_display}"
        if time_display:
            item_line += f" - {time_display}"
        if count_display:
            item_line += f" {count_display}"

    elif format_type == "telegram":
        if url:
            item_line = f"  {index}. {title} ({url})"
        else:
            item_line = f"  {index}. {title}"
        if rank_display:
            item_line += f" {rank_display}"
        if time_display:
            item_line += f" - {time_display}"
        if count_display:
            item_line += f" {count_display}"

    elif format_type == "slack":
        if url:
            item_line = f"  {index}. <{url}|{title}>"
        else:
            item_line = f"  {index}. {title}"
        if rank_display:
            item_line += f" {rank_display}"
        if time_display:
            item_line += f" _{time_display}_"
        if count_display:
            item_line += f" {count_display}"

    else:
        # wework, bark, ntfy
        if url:
            item_line = f"  {index}. [{title}]({url})"
        else:
            item_line = f"  {index}. {title}"
        if rank_display:
            item_line += f" {rank_display}"
        if time_display:
            item_line += f" - {time_display}"
        if count_display:
            item_line += f" {count_display}"

    item_line += "\n"
    return item_line


def _format_standalone_rss_item(
    item: Dict, index: int, format_type: str, timezone: str = "Asia/Shanghai"
) -> str:
    """格式化独立展示区的 RSS 条目

    Args:
        item: RSS 条目，包含 title, url, published_at, author
        index: 序号
        format_type: 格式类型
        timezone: 时区名称

    Returns:
        格式化后的条目行字符串
    """
    title = item.get("title", "")
    url = item.get("url", "")
    published_at = item.get("published_at", "")
    author = item.get("author", "")

    # 使用友好时间格式
    friendly_time = ""
    if published_at:
        friendly_time = format_iso_time_friendly(published_at, timezone, include_date=True)

    # 构建元信息
    meta_parts = []
    if friendly_time:
        meta_parts.append(friendly_time)
    if author:
        meta_parts.append(author)
    meta_str = ", ".join(meta_parts)

    # 根据格式类型构建条目行
    if format_type == "feishu":
        if url:
            item_line = f"  {index}. [{title}]({url})"
        else:
            item_line = f"  {index}. {title}"
        if meta_str:
            item_line += f" <font color='grey'>- {meta_str}</font>"
    elif format_type == "telegram":
        if url:
            item_line = f"  {index}. {title} ({url})"
        else:
            item_line = f"  {index}. {title}"
        if meta_str:
            item_line += f" - {meta_str}"
    elif format_type == "slack":
        if url:
            item_line = f"  {index}. <{url}|{title}>"
        else:
            item_line = f"  {index}. {title}"
        if meta_str:
            item_line += f" _{meta_str}_"
    else:
        # wework, bark, ntfy, dingtalk
        if url:
            item_line = f"  {index}. [{title}]({url})"
        else:
            item_line = f"  {index}. {title}"
        if meta_str:
            item_line += f" `{meta_str}`"

    item_line += "\n"
    return item_line
