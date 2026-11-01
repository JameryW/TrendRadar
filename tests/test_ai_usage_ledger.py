# coding=utf-8
"""用量台账（output/ai_usage.jsonl）的记账行为测试。

背景：TrendRadar 此前完全不记录 token 用量，只能靠日志数调用次数、靠估算报花费。
client.chat() 是所有 AI 调用的唯一入口，记账放在那里。
"""
import json
from types import SimpleNamespace

import trendradar.ai.client as client_module
from trendradar.ai.client import AIClient, default_usage_ledger, record_usage


def _response(content="ok", usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


def _usage(prompt=1000, completion_tokens=200, cached=None):
    details = (
        SimpleNamespace(cached_tokens=cached) if cached is not None else None
    )
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion_tokens,
        total_tokens=prompt + completion_tokens,
        prompt_tokens_details=details,
    )


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_chat_writes_one_usage_row(monkeypatch, tmp_path):
    ledger = tmp_path / "ai_usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LEDGER", str(ledger))
    captured = {}

    def fake_completion(**params):
        captured.update(params)
        return _response(usage=_usage(cached=512))

    monkeypatch.setattr(client_module, "completion", fake_completion)
    client = AIClient({"MODEL": "openai/mimo-v2.6-flash", "API_KEY": "k"})

    assert client.chat([{"role": "user", "content": "hi"}], purpose="analysis") == "ok"

    rows = _read_rows(ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["in"] == 1000
    assert row["out"] == 200
    assert row["tot"] == 1200
    assert row["cache"] == 512
    assert row["model"] == "openai/mimo-v2.6-flash"
    assert row["purpose"] == "analysis"
    assert row["fallback"] is False
    assert isinstance(row["dur"], float) and row["dur"] >= 0


def test_purpose_never_reaches_the_model(monkeypatch):
    """purpose 是记账标签，绝不能混进 litellm 参数（否则 API 会 400）"""
    captured = {}

    def fake_completion(**params):
        captured.update(params)
        return _response(usage=_usage())

    monkeypatch.setattr(client_module, "completion", fake_completion)
    client = AIClient({"MODEL": "deepseek/deepseek-chat", "API_KEY": "k"})

    client.chat([{"role": "user", "content": "hi"}], purpose="translation")

    assert "purpose" not in captured
    assert captured["model"] == "deepseek/deepseek-chat"


def test_fallback_call_records_the_model_actually_used(monkeypatch, tmp_path):
    ledger = tmp_path / "ai_usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LEDGER", str(ledger))

    def fake_completion(**params):
        if params["model"] == "openai/primary":
            raise RuntimeError("primary down")
        return _response(usage=_usage())

    monkeypatch.setattr(client_module, "completion", fake_completion)
    client = AIClient(
        {
            "MODEL": "openai/primary",
            "API_KEY": "k",
            "FALLBACK_MODELS": [{"model": "openai/backup", "api_key": "other-k"}],
        }
    )

    assert client.chat([{"role": "user", "content": "hi"}], purpose="analysis") == "ok"

    rows = _read_rows(ledger)
    assert len(rows) == 1
    assert rows[0]["model"] == "openai/backup"
    assert rows[0]["fallback"] is True


def test_missing_usage_object_does_not_crash(monkeypatch, tmp_path):
    ledger = tmp_path / "ai_usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LEDGER", str(ledger))

    monkeypatch.setattr(
        client_module, "completion", lambda **params: _response(usage=None)
    )
    client = AIClient({"MODEL": "deepseek/deepseek-chat", "API_KEY": "k"})

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    rows = _read_rows(ledger)
    assert rows[0]["in"] == 0 and rows[0]["out"] == 0
    assert rows[0]["purpose"] == ""  # 未标注的调用同样入账，只是没有标签


def test_ledger_failure_never_breaks_the_call(monkeypatch, tmp_path):
    """台账目录不可写 → 依然正常返回，只打印告警"""
    bad = tmp_path / "nope"
    bad.write_text("x", encoding="utf-8")  # 文件占位，mkdir 必然失败
    monkeypatch.setattr(
        client_module, "completion", lambda **params: _response(usage=_usage())
    )
    client = AIClient({"MODEL": "deepseek/deepseek-chat", "API_KEY": "k"})

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"

    row = record_usage(
        _response(usage=_usage()), "m", purpose="x", ledger=bad / "sub" / "u.jsonl"
    )
    assert row["in"] == 1000


def test_default_ledger_follows_env(monkeypatch, tmp_path):
    target = tmp_path / "custom.jsonl"
    monkeypatch.setenv("AI_USAGE_LEDGER", str(target))
    assert default_usage_ledger() == target
