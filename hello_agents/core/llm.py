"""HelloAgents 统一 LLM 层（7.2 的参考实现，精简版）。

设计理念（对应教程 7.2.3 自动检测机制）：
1. 参数优先，环境变量兜底 —— 传了参数用参数，没传去 .env 里找
2. 自动检测 Provider —— 有 DEEPSEEK_API_KEY 就当 DeepSeek 用
3. 双调用入口 —— invoke() 非流式拿完整回复，stream_invoke() 流式逐字吐
   （后面所有 Agent 的 run() 只依赖 invoke()，这是本层的核心契约）
"""

import os
from typing import Iterator, Optional

from openai import OpenAI

from .exceptions import HelloAgentsException


class HelloAgentsLLM:
    """兼容 OpenAI 接口的统一 LLM 客户端（精简版：聚焦 DeepSeek / OpenAI 兼容服务）。"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
    ):
        # 原则一：参数优先，环境变量兜底
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 原则二：自动检测（精简版只判断 DeepSeek / 其他 OpenAI 兼容服务）
        if not self.model:
            self.model = "deepseek-chat" if self._is_deepseek() else "gpt-3.5-turbo"

        # 校验必要参数：没有 key 和地址就无法调 API，尽早报错
        if not all([self.api_key, self.base_url]):
            raise HelloAgentsException(
                "API密钥和服务地址必须被提供或在 .env 文件中定义（DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL）。"
            )

        # OpenAI SDK 的客户端是通用的：换 base_url 就能连任何兼容服务
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _is_deepseek(self) -> bool:
        """判断当前是否在用 DeepSeek：环境变量或 base_url 里能看出来就行。"""
        return bool(os.getenv("DEEPSEEK_API_KEY")) or "deepseek" in (self.base_url or "")

    def invoke(self, messages: list[dict], **kwargs) -> str:
        """非流式调用：一次性返回完整回复文本。

        这是整个框架最核心的契约 —— 后面所有 Agent 的 run()
        最终都调用 self.llm.invoke(messages)，参数是 to_dict() 出来的字典列表。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            # 把底层千奇百怪的报错统一包装成框架异常，上层好捕获
            raise HelloAgentsException(f"LLM调用失败: {e}")

    def stream_invoke(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """流式调用：逐段返回回复，打字机效果。

        yield 的写法：调用方用 for chunk in llm.stream_invoke(...) 逐段处理。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,  # 关键：开启流式
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            raise HelloAgentsException(f"LLM流式调用失败: {e}")
