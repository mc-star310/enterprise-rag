"""配置管理（7.3.2）

Config 负责“把运行参数集中管理”，避免到处硬编码。
它做两件事：
1. 提供一组带默认值的配置项（模型名、温度、日志级别……）
2. 提供 from_env() 类方法，从环境变量一次性读入，做到“代码不改、只改环境变量就能换配置”
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """HelloAgents 配置类"""

    # —— LLM 相关 ——
    default_model: str = "gpt-3.5-turbo"     # 默认模型（DeepSeek 场景会被 LLM 层的自动检测覆盖）
    default_provider: str = "openai"         # 默认 provider
    temperature: float = 0.7                 # 采样温度：越低越确定，越高越发散
    max_tokens: Optional[int] = None         # 单次回复最大 token，None 表示不限制

    # —— 系统相关 ——
    debug: bool = False                      # 调试开关
    log_level: str = "INFO"                 # 日志级别

    # —— 其他 ——
    max_history_length: int = 100           # 对话历史最多保留多少条，防止无限增长撑爆上下文

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量构建配置。

        用类方法（@classmethod）而不是普通函数，是因为它要返回“当前这个类”的实例，
        cls 指的就是 Config 本身，子类继承后也能直接复用。
        环境变量读出来全是字符串，这里手动做类型转换（lower/==/float/int），
        转换失败就回退到默认值——这是配置系统的标准写法。
        """
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """把配置转成普通字典，方便传给日志、持久化或别的库。

        self.dict() 是 Pydantic BaseModel 自带的方法，会把所有字段序列化成一个 dict。
        """
        return self.dict()
