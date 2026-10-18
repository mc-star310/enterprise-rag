"""消息系统（7.3.1）

Message 是框架里最基础的“数据容器”——它代表一次对话中的一条消息。
无论系统提示、用户输入、模型回复、工具返回，在框架眼里都是一条 Message。
把消息统一成一种结构，后面 Agent 拼历史、发给 LLM、解析回复，才有统一入口。
"""

from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

# 消息角色：用 Literal 把取值锁死在四种之内。
# 好处：如果写 role="userx" 这种无效值，Pydantic 会在创建对象时直接报错，
# 而不是把脏数据带进后续逻辑里悄悄出错。
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """消息类（继承 Pydantic 的 BaseModel，自动获得类型校验与序列化能力）"""

    content: str                       # 消息正文
    role: MessageRole                  # 消息角色，只能是上面四种之一
    timestamp: datetime = None         # 创建时间，默认在 __init__ 里填“当前时刻”
    metadata: Optional[Dict[str, Any]] = None  # 任意附加信息（工具名、调用 ID 等）

    def __init__(self, content: str, role: MessageRole, **kwargs):
        # 重写构造，让 timestamp / metadata 在未传入时给默认值。
        # 注意这里手动调 super().__init__，而不是用 Pydantic 默认的字段默认值，
        # 因为 datetime.now() 需要“每次创建时现算”，写在字段默认值上会是固定的启动时间。
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转成 OpenAI Chat API 需要的字典格式。

        这是 Message 最关键的方法：框架最终要把消息发给 LLM，
        而 OpenAI 接口只认 {"role": ..., "content": ...} 这种 JSON 结构。
        只要所有消息都能 to_dict()，Agent 拼历史时就能直接 list 拼接、整批发给模型。
        """
        return {
            "role": self.role,
            "content": self.content,
        }

    def __str__(self) -> str:
        # 打印/日志时给人看的简洁格式
        return f"[{self.role}] {self.content}"
