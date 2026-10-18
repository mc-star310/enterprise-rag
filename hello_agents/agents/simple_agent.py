"""简单对话 Agent（7.4 第一个具体 Agent）—— 整个框架最短、最该先读懂的一个。

SimpleAgent 的逻辑一句话概括：
    把“系统提示 + 历史 + 当前输入”拼成一个消息列表 → 丢给 LLM → 存进历史 → 返回回复。

它是所有复杂 Agent（ReAct / Reflection / PlanAndSolve）的“最小骨架”。
只要看懂这一个，后面三个只是往 run() 里加循环、加工具、加反思而已。
"""

from typing import Optional, Iterator

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message


class SimpleAgent(Agent):
    """最基础的对话 Agent：一问一答，无工具、无循环。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        # 把公共属性（name/llm/system_prompt/config/历史）交给基类 Agent 初始化
        super().__init__(name, llm, system_prompt, config)

    def _build_messages(self, input_text: str) -> list[dict]:
        """把“系统提示 + 历史 + 当前输入”整合成 LLM 要的字典列表。

        抽成独立方法，是因为 run() 和 stream_run() 都要用同一套拼装逻辑，
        避免复制两遍（DRY：Don't Repeat Yourself）。
        """
        messages: list[dict] = []

        # 1) 系统提示：规定 Agent 的“人设/规则”，每条对话都带上
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 2) 历史：把之前的所有消息原样接上，让模型有“上下文记忆”
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 3) 当前输入：用户这一轮说了什么
        messages.append({"role": "user", "content": input_text})
        return messages

    def run(self, input_text: str, **kwargs) -> str:
        """非流式运行：一次返回完整回复，并写入历史。

        这是 Agent 抽象基类里那个 @abstractmethod run() 的具体实现。
        注意 self.llm.invoke(messages) —— 这正是第七章最早讲的“LLM 层的核心契约”。
        """
        messages = self._build_messages(input_text)

        # 调用 LLM 层（7.2），拿到完整字符串回复
        response = self.llm.invoke(messages, **kwargs)

        # 把这一轮“用户问 + 模型答”写进历史，下一轮才有记忆
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

        return response

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式运行：逐段吐出回复（打字机效果），结束后再存历史。"""
        messages = self._build_messages(input_text)

        full_response = ""
        # 调用 LLM 层的流式入口，逐块累加到 full_response
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk  # 边生成边返回，调用方可以实时显示

        # 全部收完，再把整轮对话存进历史
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
