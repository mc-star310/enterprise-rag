"""HelloAgents 记忆系统（8.2 轻栈实现）。

提供：
- WorkingMemory：纯内存工作记忆（容量上限 + TTL + 综合评分检索）
- EpisodicMemory：SQLite 持久化情景记忆（重启不丢 + 三因素评分）
- MemoryManager：跨类型统一调度（编码/检索/遗忘/固化）
- MemoryTool：把记忆系统包装成 Agent 可直接调用的 Tool（注册进 ToolRegistry 即可用）
"""

from .working_memory import MemoryItem, WorkingMemory
from .episodic_memory import EpisodicMemory
from .manager import MemoryManager
from .memory_tool import MemoryTool, register_memory_tool

__all__ = [
    "MemoryItem",
    "WorkingMemory",
    "EpisodicMemory",
    "MemoryManager",
    "MemoryTool",
    "register_memory_tool",
]
