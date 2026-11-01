import pytest


@pytest.fixture(autouse=True)
def _redirect_usage_ledger(tmp_path, monkeypatch):
    """测试期间把用量台账指到临时目录，绝不污染生产 output/ai_usage.jsonl。"""
    monkeypatch.setenv("AI_USAGE_LEDGER", str(tmp_path / "ai_usage.jsonl"))
