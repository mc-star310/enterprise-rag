"""Agent 抽象基类（7.3.3）

这是整个框架的“骨架”。所有具体 Agent（SimpleAgent / ReActAgent / ...）都继承它。
基类负责两件通用的事：
1. 统一构造参数（name / llm / system_prompt / config）
2. 统一历史记录管理（add_message / get_history / clear_history）

至于“这个 Agent 怎么思考、怎么调工具”，交给子类用 @abstractmethod 强制实现 run()——
基类只规定“必须有个 run()”，不规定“run 里干啥”。这就是抽象基类的意义。
"""

from __future__ import annotations  # 让下面的类型注解变成“延迟求值”，避免 import 还没写好的 llm.py

from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING
from .message import Message
from .config import Config

# 仅在“类型检查阶段”导入，运行时不需要 —— 这样 Agent 类现在就能单独使用，
# 等后面写好 core/llm.py，这里的类型提示自然就生效，无需改动。
if TYPE_CHECKING:
    from .llm import HelloAgentsLLM


class Agent(ABC):
    """Agent 基类（ABC = Abstract Base Class，抽象基类）"""

    def __init__(
        self,
        name: str,
        llm: "HelloAgentsLLM",
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name                          # Agent 名字，用于日志和区分
        self.llm = llm                            # LLM 实例（地基来自 7.2）
        self.system_prompt = system_prompt        # 系统提示词（定义 Agent 人设/职责）
        self.config = config or Config()         # 没传就用默认配置；"or" 提供兜底默认值
        self._history: list[Message] = []         # 对话历史，用 Message 列表保存，下划线表示“内部使用”

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent —— 抽象方法，子类必须实现。

        @abstractmethod 的作用：如果某个子类忘了写 run()，Python 会直接拒绝实例化它
        （TypeError: Can't instantiate abstract class ...）。这把“接口契约”用代码强制锁死。
        """
        pass

    def add_message(self, message: Message):
        """往历史里加一条消息。"""
        self._history.append(message)

    def clear_history(self):
        """清空历史。每轮新对话前调用，避免上一轮内容串进来。"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """返回历史副本（.copy() 防止外部直接改到内部列表）。"""
        return self._history.copy()

    def __str__(self) -> str:
        # 打印时显示名字和所用 provider，便于调试
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        return self.__str__()
