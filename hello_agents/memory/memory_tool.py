"""记忆工具（8.2 记忆系统收官）：把 MemoryManager 包装成 Agent 可直接调用的 Tool。

写它的目的：让 ReActAgent 在思考链里能写出
    Action: memory[记 用户喜欢用Python做数据分析]
    Action: memory[查 用户的专业背景]
记忆系统从此从“裸对话历史”升级为“Agent 可调用的长期记忆”。

三种动作（单一 input 参数，与 calculator 同一接法，兼容 ToolRegistry.execute_tool）：
  - 记 <内容>   : 存入长期情景记忆（内容结尾加 [重要] 提升权重到 0.9）
  - 查 <问题>   : 跨 working + episodic 检索 top-k，带来源类型与重要性
  - 忘 <关键词> : 按 id 或内容子串精准删除（新增的 remove_memory 能力）
"""

from typing import Dict, Any, List

from ..tools.base import Tool, ToolParameter
from .manager import MemoryManager


class MemoryTool(Tool):
    """让 Agent 直接读写记忆系统的工具（8.2 收官）。"""

    def __init__(self, manager: MemoryManager):
        super().__init__(
            name="memory",
            description=(
                "长期记忆管理：记住用户偏好/事实、检索过往记忆、删除指定记忆。"
                "输入格式：'记 <内容>' 保存（结尾加 [重要] 提升权重）；"
                "'查 <问题>' 检索相关记忆；'忘 <关键词或id>' 删除匹配的记忆。"
                "例如：'记 用户是计算机专业学生 [重要]'、'查 用户的专业'、'忘 计算机'"
            ),
        )
        self.manager = manager

    def run(self, parameters: Dict[str, Any]) -> str:
        text = (parameters.get("input") or "").strip()
        if not text:
            return "错误：记忆工具需要一个动作，格式：'记 <内容>' / '查 <问题>' / '忘 <关键词>'。"
        action, _, rest = text.partition(" ")
        if action in ("记", "save", "add", "remember"):
            content = rest.strip()
            if not content:
                return "错误：'记' 后面需要跟要记住的内容。"
            importance = 0.9 if "[重要]" in content else 0.6
            content = content.replace("[重要]", "").strip()
            mid = self.manager.add_memory(
                content, memory_type="episodic", importance=importance
            )
            return f"已记住：{content}（id={mid}，重要性={importance}）"
        if action in ("查", "search", "recall", "query"):
            query = rest.strip()
            if not query:
                return "错误：'查' 后面需要跟要检索的问题。"
            results = self.manager.retrieve_memories(query, limit=3)
            if not results:
                return "没有找到相关记忆。"
            lines = [
                f"{i}. [{r.memory_type}] {r.content}（重要性 {r.importance:.1f}）"
                for i, r in enumerate(results, 1)
            ]
            return "相关记忆：\n" + "\n".join(lines)
        if action in ("忘", "forget", "delete", "remove"):
            key = rest.strip()
            if not key:
                return "错误：'忘' 后面需要跟要删除的关键词或 id。"
            n = self.manager.remove_memory(key)
            return (
                f"已删除 {n} 条匹配 '{key}' 的记忆。"
                if n
                else f"没有找到匹配 '{key}' 的记忆。"
            )
        return "无法识别的动作，请用 '记/查/忘' 开头，例如 '记 用户喜欢Python'。"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="记忆动作：'记 <内容>' 保存、'查 <问题>' 检索、'忘 <关键词>' 删除",
                required=True,
            )
        ]


def register_memory_tool(
    registry=None,
    db_path: str = ":memory:",
    user_id: str = "default_user",
) -> MemoryTool:
    """便捷函数：创建 MemoryTool 并注册进指定 ToolRegistry（默认全局注册表）。

    用法：
        from hello_agents.memory import register_memory_tool
        register_memory_tool(db_path="memory.db")  # 注册进全局 registry
    之后 ReActAgent 的 ToolRegistry 里就多了一个名为 'memory' 的工具。
    """
    from ..tools.registry import global_registry

    reg = registry or global_registry
    manager = MemoryManager(episodic_db=db_path, user_id=user_id)
    tool = MemoryTool(manager)
    reg.register_tool(tool)
    return tool
