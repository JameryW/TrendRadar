# coding=utf-8
"""
AI 客户端模块

基于 LiteLLM 的统一 AI 模型接口
支持 100+ AI 提供商（OpenAI、DeepSeek、Gemini、Claude、国内模型等）
"""

import os
from typing import Any, Dict, List

from litellm import completion


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
            **kwargs: 额外参数，会覆盖默认配置

        Returns:
            str: AI 响应内容

        Raises:
            Exception: 主模型与全部降级模型都失败时抛出（错误信息含各次尝试）
        """
        attempts: List[str] = [self.model]
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

        # 合并其他额外参数
        for key, value in kwargs.items():
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
