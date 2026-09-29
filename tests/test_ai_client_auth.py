from types import SimpleNamespace

import trendradar.ai.client as client_module
import trendradar.ai.translator as translator_module
from trendradar.ai.client import AIClient
from trendradar.ai.translator import AITranslator
from trendradar.core.loader import _load_ai_config


def _response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_bearer_token_uses_authorization_header_not_litellm_api_key(monkeypatch):
    captured = {}

    def fake_completion(**params):
        captured.update(params)
        return _response()

    monkeypatch.setattr(client_module, "completion", fake_completion)
    client = AIClient(
        {
            "MODEL": "anthropic/DeepSeek-V4-Flash-0731",
            "AUTH_TOKEN": "test-bearer-token",
            "API_BASE": "https://api.scnet.cn/api/llm/anthropic",
        }
    )

    assert client.validate_config() == (True, "")
    assert client.chat([{"role": "user", "content": "ping"}]) == "ok"
    assert "api_key" not in captured
    assert captured["extra_headers"] == {
        "Authorization": "Bearer test-bearer-token"
    }


def test_native_api_key_keeps_api_key_semantics(monkeypatch):
    captured = {}
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-bearer-must-not-win")

    def fake_completion(**params):
        captured.update(params)
        return _response()

    monkeypatch.setattr(client_module, "completion", fake_completion)
    client = AIClient(
        {
            "MODEL": "deepseek/deepseek-chat",
            "API_KEY": "native-api-key",
        }
    )

    client.chat([{"role": "user", "content": "ping"}])
    assert captured["api_key"] == "native-api-key"
    assert "extra_headers" not in captured


def test_ambient_bearer_is_used_when_config_has_no_credential(monkeypatch):
    captured = {}

    def fake_completion(**params):
        captured.update(params)
        return _response()

    monkeypatch.setattr(client_module, "completion", fake_completion)
    monkeypatch.setenv("AI_AUTH_TOKEN", "ambient-bearer")
    monkeypatch.delenv("AI_API_KEY", raising=False)

    client = AIClient({"MODEL": "anthropic/test-model"})
    client.chat([{"role": "user", "content": "ping"}])

    assert "api_key" not in captured
    assert captured["extra_headers"] == {
        "Authorization": "Bearer ambient-bearer"
    }


def test_loader_reads_ai_auth_token_without_aliasing_it_to_api_key(monkeypatch):
    monkeypatch.setenv("AI_AUTH_TOKEN", "bearer-only")
    monkeypatch.delenv("AI_API_KEY", raising=False)

    config = _load_ai_config({"ai": {}})

    assert config["AUTH_TOKEN"] == "bearer-only"
    assert config["API_KEY"] == ""


def test_loader_native_key_ignores_ambient_anthropic_token(monkeypatch):
    monkeypatch.delenv("AI_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("AI_API_KEY", "native-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-bearer")

    config = _load_ai_config({"ai": {}})

    assert config["API_KEY"] == "native-api-key"
    assert config["AUTH_TOKEN"] == ""


def test_translator_accepts_bearer_only_credentials(monkeypatch):
    monkeypatch.setattr(
        translator_module,
        "load_prompt_template",
        lambda *_args, **_kwargs: (
            "Translate accurately.",
            "Translate to {target_language}: {content}",
        ),
    )
    translator = AITranslator(
        {"ENABLED": True, "LANGUAGE": "English"},
        {
            "MODEL": "anthropic/DeepSeek-V4-Flash-0731",
            "AUTH_TOKEN": "test-bearer-token",
        },
    )

    monkeypatch.setattr(translator, "_call_ai", lambda _prompt: "translated")
    single = translator.translate("待翻译")
    assert single.success is True
    assert single.translated_text == "translated"

    monkeypatch.setattr(
        translator,
        "_call_ai",
        lambda _prompt: "[1] first\n[2] second",
    )
    batch = translator.translate_batch(["一", "二"])
    assert batch.success_count == 2
    assert [item.translated_text for item in batch.results] == ["first", "second"]


# ---------------------------------------------------------------------------
# 降级链（mimo 主模型 -> deepseek-flash）
# ---------------------------------------------------------------------------


def _client(monkeypatch, completion_stub, **config_overrides):
    monkeypatch.setattr(client_module, "completion", completion_stub)
    config = {
        "MODEL": "openai/mimo-v2.6-flash",
        "API_KEY": "primary-key",
        "API_BASE": "https://token-plan-cn.xiaomimimo.com/v1",
        "FALLBACK_MODELS": [
            {
                "model": "openai/deepseek-flash",
                "api_base": "https://api.deepseek.com/v1",
                "api_key_env": "DEEPSEEK_API_KEY",
            }
        ],
    }
    config.update(config_overrides)
    return AIClient(config)


def test_primary_success_never_touches_fallback(monkeypatch):
    calls = []

    def fake_completion(**params):
        calls.append(params)
        return _response("primary-ok")

    client = _client(monkeypatch, fake_completion)
    assert client.chat([{"role": "user", "content": "ping"}]) == "primary-ok"
    assert len(calls) == 1
    assert calls[0]["model"] == "openai/mimo-v2.6-flash"


def test_primary_failure_falls_back_with_its_own_endpoint_and_key(monkeypatch):
    calls = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    def fake_completion(**params):
        calls.append(params)
        if params["model"] == "openai/mimo-v2.6-flash":
            raise RuntimeError("simulated primary outage")
        return _response("fallback-ok")

    client = _client(monkeypatch, fake_completion)
    assert client.chat([{"role": "user", "content": "ping"}]) == "fallback-ok"

    assert [call["model"] for call in calls] == [
        "openai/mimo-v2.6-flash",
        "openai/deepseek-flash",
    ]
    primary, fallback = calls
    # 主模型用 api_key 模式，降级模型必须带自己的端点和 key，不能沿用主模型的
    assert primary["api_key"] == "primary-key"
    assert primary["api_base"] == "https://token-plan-cn.xiaomimimo.com/v1"
    assert fallback["api_base"] == "https://api.deepseek.com/v1"
    assert fallback["api_key"] == "deepseek-key"
    assert "extra_headers" not in fallback


def test_fallback_without_credential_never_reuses_primary_key(monkeypatch):
    """换了服务商却没给 key 时，宁可不带 key 也不把主模型 key 发过去"""
    calls = []

    def fake_completion(**params):
        calls.append(params)
        if params["model"] == "openai/mimo-v2.6-flash":
            raise RuntimeError("simulated primary outage")
        return _response("fallback-ok")

    client = _client(
        monkeypatch,
        fake_completion,
        FALLBACK_MODELS=[
            {"model": "openai/deepseek-flash", "api_base": "https://api.deepseek.com/v1"}
        ],
    )
    client.chat([{"role": "user", "content": "ping"}])

    fallback = calls[-1]
    assert "api_key" not in fallback
    assert "extra_headers" not in fallback


def test_string_fallback_inherits_primary_endpoint_and_credentials(monkeypatch):
    calls = []

    def fake_completion(**params):
        calls.append(params)
        if params["model"].endswith("mimo-v2.6-flash"):
            raise RuntimeError("simulated primary outage")
        return _response("fallback-ok")

    client = _client(
        monkeypatch, fake_completion, FALLBACK_MODELS=["openai/mimo-v2.6-flash-v2"]
    )
    client.chat([{"role": "user", "content": "ping"}])

    fallback = calls[-1]
    assert fallback["api_key"] == "primary-key"
    assert fallback["api_base"] == "https://token-plan-cn.xiaomimimo.com/v1"


def test_every_model_failing_raises_with_full_attempt_list(monkeypatch):
    def fake_completion(**params):
        raise RuntimeError(f"boom:{params['model']}")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    client = _client(monkeypatch, fake_completion)

    try:
        client.chat([{"role": "user", "content": "ping"}])
    except RuntimeError as error:  # noqa: PT011 - 断言错误信息可定位
        message = str(error)
        assert "openai/mimo-v2.6-flash" in message
        assert "openai/deepseek-flash" in message
        assert "boom:openai/deepseek-flash" in message
    else:
        raise AssertionError("全部模型失败时应当抛错")


def test_loader_passes_fallback_dicts_through(monkeypatch):
    monkeypatch.delenv("AI_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("AI_API_KEY", "primary-key")

    fallbacks = [
        {
            "model": "openai/deepseek-flash",
            "api_base": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
        }
    ]
    config = _load_ai_config({"ai": {"model": "openai/mimo-v2.6-flash",
                                     "fallback_models": fallbacks}})

    assert config["FALLBACK_MODELS"] == fallbacks
    assert config["MODEL"] == "openai/mimo-v2.6-flash"
    assert config["API_KEY"] == "primary-key"
    assert config["AUTH_TOKEN"] == ""
