"""工具基类（7.5 工具系统的基础）

为什么要有 Tool 基类？
因为 Agent 可能调用几十种工具（计算器、搜索、查数据库……），如果每种写法都不一样，
Agent 根本没法统一管理。Tool 基类把“所有工具都长什么样”定死：
    - 都有 name / description（告诉模型“我是干嘛的”）
    - 都有 run(parameters)（真正执行）
    - 都能 to_dict()（把描述塞进提示词，让模型知道有哪些工具可用）
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """工具参数的定义：模型调用工具时，需要知道“要传什么、什么类型、是否必填”"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """所有工具的抽象基类。具体工具必须实现 run() 和 get_parameters()。"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具，返回结果字符串。"""
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """返回该工具需要的参数定义列表（用于生成提示词）。"""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """校验必填参数是否都给了。"""
        required = [p.name for p in self.get_parameters() if p.required]
        return all(p in parameters for p in required)

    def to_dict(self) -> Dict[str, Any]:
        """把工具描述转成字典，方便拼进提示词。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.dict() for p in self.get_parameters()],
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"
