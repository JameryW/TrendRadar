# coding=utf-8
"""
AI 客户端模块

基于 LiteLLM 的统一 AI 模型接口
支持 100+ AI 提供商（OpenAI、DeepSeek、Gemini、Claude、国内模型等）
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from litellm import completion


def default_usage_ledger() -> Path:
    """用量台账路径：环境变量 ``AI_USAGE_LEDGER`` > 仓库 ``output/ai_usage.jsonl``。

    容器内 ``/app/output`` 与宿主 ``trendradar/output`` 是同一个 bind mount
    （output/.gitignore 忽略 ``*``），所以宿主侧的 ``llm-call-monitor.py``
    不需要 ``podman exec`` 就能读到逐次用量。
    """
    env_path = os.environ.get("AI_USAGE_LEDGER")
    if env_path:
        return Path(env_path)
    # __file__ = <root>/trendradar/trendradar/ai/client.py
    # parents[2] = <root>/trendradar（容器内即 /app）
    return Path(__file__).resolve().parents[2] / "output" / "ai_usage.jsonl"


def _usage_value(usage: Any, *names: str) -> int:
    """从 litellm 的 usage 对象（对象或 dict）里取第一个存在的字段"""
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def record_usage(
    response: Any,
    model: str,
    purpose: Optional[str] = None,
    duration: Optional[float] = None,
    fallback: bool = False,
    ledger: Optional[Path] = None,
) -> Dict[str, Any]:
    """把一次成功调用的 token 用量追加进 JSONL 台账，并打印一行日志。

    任何异常都吞掉 —— 记账绝不能反过来打断真实调用。
    """
    usage = getattr(response, "usage", None)
    in_tok = _usage_value(usage, "prompt_tokens", "input_tokens")
    out_tok = _usage_value(usage, "completion_tokens", "output_tokens")
    total = _usage_value(usage, "total_tokens") or (in_tok + out_tok)
    details = None
    if usage is not None:
        details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
    cached = _usage_value(details, "cached_tokens")
    if not cached and usage is not None:
        cached = _usage_value(usage, "cache_read_input_tokens", "cached_tokens")

    row: Dict[str, Any] = {
        "ts": time.time(),
        "model": model,
        "purpose": purpose or "",
        "in": in_tok,
        "out": out_tok,
        "tot": total,
        "cache": cached,
        "dur": round(duration, 3) if duration is not None else None,
        "fallback": bool(fallback),
    }
    print(
        "[AI] 用量: in={:,} out={:,}{} ({}, {})".format(
            in_tok,
            out_tok,
            " cache={:,}".format(cached) if cached else "",
            model,
            purpose or "未标注",
        )
    )
    try:
        path = Path(ledger) if ledger else default_usage_ledger()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - 记账失败只告警不抛
        print(f"[AI] 用量台账写入失败: {type(exc).__name__}: {exc}")
    return row


class AIClient:
    """统一的 AI 客户端（基于 LiteLLM）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 AI 客户端

        Args:
            config: AI 配置字典
                - MODEL: 模型标识（格式: provider/model_name）
                - API_KEY: API 密钥
                - AUTH_TOKEN: Bearer 鉴权令牌（与 API_KEY 互斥）
                - API_BASE: API 基础 URL（可选）
                - TEMPERATURE: 采样温度
                - MAX_TOKENS: 最大生成 token 数
                - TIMEOUT: 请求超时时间（秒）
                - NUM_RETRIES: 重试次数（可选）
                - FALLBACK_MODELS: 备用模型列表（可选）
        """
        self.model = config.get("MODEL", "deepseek/deepseek-chat")
        configured_api_key = config.get("API_KEY") or ""
        configured_auth_token = config.get("AUTH_TOKEN") or ""
        if configured_api_key or configured_auth_token:
            # A caller-provided credential choice is authoritative.  In
            # particular, an ambient ANTHROPIC_AUTH_TOKEN inherited by a
            # self-hosted runner must not silently turn a native API-key
            # request into Bearer authentication.
            self.api_key = configured_api_key
            self.auth_token = configured_auth_token
        else:
            self.api_key = os.environ.get("AI_API_KEY", "")
            self.auth_token = (
                os.environ.get("AI_AUTH_TOKEN", "")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            )
        self.api_base = config.get("API_BASE", "")
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.max_tokens = config.get("MAX_TOKENS", 5000)
        self.timeout = config.get("TIMEOUT", 120)
        self.num_retries = config.get("NUM_RETRIES", 2)
        self.fallback_models = config.get("FALLBACK_MODELS", [])
        self.extra_params = config.get("EXTRA_PARAMS") or {}
        if not isinstance(self.extra_params, dict):
            print(
                f"[AI] 配置错误: ai.extra_params 应为字典，"
                f"当前为 {type(self.extra_params).__name__}，已忽略"
            )
            self.extra_params = {}

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        调用 AI 模型进行对话

        主模型失败时按 ``fallback_models`` 顺序降级重试。降级不是 LiteLLM
        的能力（``litellm.completion()`` 没有 ``fallbacks`` 参数），因此在这里
        显式实现：每个降级条目可以自带 model / api_base / 凭证，跨服务商时
        必须各用各的 key。

        Args:
            messages: 消息列表，格式: [{"role": "system/user/assistant", "content": "..."}]
            **kwargs: 额外参数，会覆盖默认配置。其中 ``purpose`` 是调用用途标签
                （analysis / translation / repair_json / filter_* …），只用于用量台账，
                不会发给模型。

        Returns:
            str: AI 响应内容

        Raises:
            Exception: 主模型与全部降级模型都失败时抛出（错误信息含各次尝试）
        """
        purpose = kwargs.pop("purpose", None)
        started = time.time()
        attempts: List[str] = [self.model]
        used_model = self.model
        try:
            response = completion(
                **self._build_params(self.model, self.api_key, self.auth_token,
                                     self.api_base, messages, kwargs)
            )
        except Exception as primary_error:
            response = None
            last_error = primary_error
            for entry in self._normalized_fallbacks():
                fallback_model = entry.get("model")
                if not fallback_model or fallback_model in attempts:
                    continue
                attempts.append(fallback_model)
                try:
                    params = self._build_params(
                        fallback_model,
                        self._entry_api_key(entry),
                        self._entry_auth_token(entry),
                        entry.get("api_base") or self.api_base,
                        messages,
                        kwargs,
                    )
                    print(f"[AI] 主模型失败，降级到 {fallback_model} ...")
                    response = completion(**params)
                    used_model = fallback_model
                    break
                except Exception as fallback_error:
                    last_error = fallback_error
                    print(f"[AI] 降级模型 {fallback_model} 也失败: "
                          f"{self._short_error(fallback_error)}")
            if response is None:
                raise RuntimeError(
                    "所有模型均调用失败（已尝试: {}）: {}".format(
                        " -> ".join(attempts), self._short_error(last_error)
                    )
                ) from last_error

        # 记账放在 _extract_content 之前：即使内容为空/结构异常，这次钱已经花了
        record_usage(
            response,
            used_model,
            purpose=purpose,
            duration=time.time() - started,
            fallback=used_model != self.model,
        )
        return self._extract_content(response)

    # ------------------------------------------------------------------
    # 请求构造与降级
    # ------------------------------------------------------------------
    def _build_params(
        self,
        model: str,
        api_key: str,
        auth_token: str,
        api_base: str,
        messages: List[Dict[str, str]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建单次 ``litellm.completion()`` 调用参数"""
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": kwargs.get("timeout", self.timeout),
            "num_retries": kwargs.get("num_retries", self.num_retries),
        }

        # Bearer-compatible gateways (for example SCNet's Anthropic endpoint)
        # must not receive the token through LiteLLM's ``api_key`` argument:
        # that selects Anthropic's native ``x-api-key`` header.  Explicit
        # bearer credentials take precedence while native providers retain
        # their existing API-key semantics.
        if auth_token:
            params["extra_headers"] = {
                "Authorization": f"Bearer {auth_token}",
            }
        elif api_key:
            params["api_key"] = api_key

        if api_base:
            params["api_base"] = api_base

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens and max_tokens > 0:
            params["max_tokens"] = max_tokens

        # 添加 fallback 模型（如果配置了）
        if self.fallback_models and model == self.model:
            params["fallbacks"] = self.fallback_models

        # 合并配置文件中的 extra_params（如自定义请求头 extra_headers），
        # 再用调用方 kwargs 覆盖；两者均不覆盖上方已确定的核心参数，
        # 因此未配置 extra_params 时行为与原先完全一致
        extra_merged = self.extra_params if isinstance(self.extra_params, dict) else {}
        for key, value in {**extra_merged, **kwargs}.items():
            if key not in params:
                params[key] = value
        return params

    def _normalized_fallbacks(self) -> List[Dict[str, Any]]:
        """归一化降级配置：字符串条目按“继承主模型凭证与端点”处理"""
        entries: List[Dict[str, Any]] = []
        for item in self.fallback_models or []:
            if isinstance(item, str) and item:
                entries.append({"model": item})
            elif isinstance(item, dict) and item.get("model"):
                entries.append(item)
        return entries

    def _entry_api_key(self, entry: Dict[str, Any]) -> str:
        """降级条目的 API Key：条目显式值 > 条目指定的环境变量 > 主模型 key"""
        if entry.get("api_key"):
            return entry["api_key"]
        env_name = entry.get("api_key_env")
        if env_name:
            return os.environ.get(env_name, "")
        # 条目换了端点却没给凭证时不要沿用主模型 key（发到别家必然 401）
        if entry.get("api_base") and entry.get("api_base") != self.api_base:
            return ""
        return self.api_key

    def _entry_auth_token(self, entry: Dict[str, Any]) -> str:
        """降级条目的 Bearer Token：优先于 API Key"""
        if entry.get("auth_token"):
            return entry["auth_token"]
        env_name = entry.get("auth_token_env")
        if env_name:
            return os.environ.get(env_name, "")
        if entry.get("api_key") or entry.get("api_key_env"):
            return ""
        if entry.get("api_base") and entry.get("api_base") != self.api_base:
            return ""
        return self.auth_token

    @staticmethod
    def _short_error(error: BaseException) -> str:
        """截断错误信息，避免把整个响应回执塞进报告"""
        text = f"{type(error).__name__}: {error}"
        return text if len(text) <= 300 else text[:300] + "..."

    @staticmethod
    def _extract_content(response: Any) -> str:
        """提取响应内容（部分提供商返回内容块列表）"""
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return content or ""

    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效

        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not self.model:
            return False, "未配置 AI 模型（model）"

        if not self.auth_token and not self.api_key:
            return False, (
                "未配置 AI 凭证，请设置 AI_AUTH_TOKEN/ANTHROPIC_AUTH_TOKEN "
                "或 AI_API_KEY"
            )

        # 验证模型格式（应该包含 provider/model）
        if "/" not in self.model:
            return False, f"模型格式错误: {self.model}，应为 'provider/model' 格式（如 'deepseek/deepseek-chat'）"

        return True, ""
