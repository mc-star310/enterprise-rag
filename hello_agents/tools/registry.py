"""工具注册表（7.5 的核心：工具的“户口本”）

为什么需要注册表？
ReActAgent 在思考时要决定“调哪个工具”，它必须先知道“有哪些工具可用”。
ToolRegistry 就是这张清单：
    - register_tool()  把工具登记进来
    - get_tools_description()  把清单变成提示词文本，喂给模型
    - execute_tool()   按名字找到工具并执行
    - get_tool()       按名字查找

设计要点：工具和 Agent 解耦——Agent 只认“注册表”，不认识任何具体工具。
这样你随时加一个新工具，只要 register 进去，Agent 自动就能用，不用改 Agent 代码。
"""

from typing import Optional, Callable
from .base import Tool


class ToolRegistry:
    """工具的注册、管理和执行。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """注册一个 Tool 实例。同名会覆盖。"""
        if tool.name in self._tools:
            print(f"⚠️ 工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def unregister(self, name: str):
        if name in self._tools:
            del self._tools[name]
            print(f"🗑️ 工具 '{name}' 已注销。")
        else:
            print(f"⚠️ 工具 '{name}' 不存在。")

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def execute_tool(self, name: str, input_text: str) -> str:
        """按名字执行工具。找不到或出错都返回带说明的字符串，而不是直接崩。"""
        tool = self._tools.get(name)
        if not tool:
            return f"错误：未找到名为 '{name}' 的工具。"
        try:
            # 关键：ReAct 里的调用是 "工具名[一整段文本]"，而不同工具的参数名不一样
            # （计算器叫 input，搜索叫 query）。这里读工具自己声明的参数表，
            # 把整段文本塞进它的第一个必填参数 —— 这就是"参数映射"。
            params = tool.get_parameters()
            required = [p.name for p in params if p.required]
            primary = required[0] if required else "input"
            return tool.run({primary: input_text})
        except Exception as e:
            return f"错误：执行工具 '{name}' 时发生异常: {e}"

    def get_tools_description(self) -> str:
        """生成给 LLM 看的工具清单文本。"""
        if not self._tools:
            return "暂无可用工具"
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> list[Tool]:
        return list(self._tools.values())


# 全局注册表：方便随手注册、各模块共享
global_registry = ToolRegistry()
