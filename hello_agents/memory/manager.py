"""HelloAgents 记忆系统 · 记忆管理器（8.2 轻栈实现）。

MemoryManager 是四种记忆的统一调度入口（对应教材 MemoryManager）：
- add_memory：按 memory_type 分发到对应记忆模块
- retrieve_memories：跨类型合并检索
- forget_memories：三种遗忘策略（重要性 / 时间 / 容量）
- consolidate_memories：短期 → 长期整合（如 working → episodic）
"""

import time

from .working_memory import MemoryItem, WorkingMemory
from .episodic_memory import EpisodicMemory


class MemoryManager:
    """记忆管理器：统一操作接口，内部按类型分发给各记忆模块。"""

    def __init__(
        self,
        enable_working: bool = True,
        enable_episodic: bool = True,
        episodic_db: str = ":memory:",
        user_id: str = "default_user",
    ):
        self.user_id = user_id
        self.working = WorkingMemory() if enable_working else None
        self.episodic = EpisodicMemory(episodic_db) if enable_episodic else None

    # ---------- 编码 + 存储（按类型分发）----------
    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        **metadata,
    ) -> str:
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
        )
        if memory_type == "episodic" and self.episodic:
            return self.episodic.add(item)
        # working 与默认都进工作记忆（工作记忆不持久化）
        if self.working:
            return self.working.add(item)
        if self.episodic:
            return self.episodic.add(item)
        raise RuntimeError("未启用任何记忆模块")

    # ---------- 检索（跨类型合并）----------
    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: list = None,
        min_importance: float = 0.1,
    ) -> list:
        types = memory_types or ["working", "episodic"]
        results = []
        if "working" in types and self.working:
            results += self.working.retrieve(query, limit=limit * 2)
        if "episodic" in types and self.episodic:
            results += self.episodic.retrieve(
                query, limit=limit * 2, min_importance=min_importance
            )
        # 合并后按时间倒序（最近的优先），并去重
        seen = set()
        merged = []
        for it in sorted(results, key=lambda x: x.timestamp, reverse=True):
            if it.id in seen:
                continue
            seen.add(it.id)
            merged.append(it)
        return merged[:limit]

    # ---------- 遗忘（三种策略）----------
    def forget_memories(
        self, strategy="importance_based", threshold=0.2, max_age_days=30
    ) -> int:
        count = 0
        if self.working:
            if strategy in ("importance_based", "time_based"):
                before = len(self.working.items)
                if strategy == "importance_based":
                    self.working.items = [
                        i for i in self.working.items if i.importance >= threshold
                    ]
                else:
                    now = time.time()
                    self.working.items = [
                        i
                        for i in self.working.items
                        if now - i.timestamp < max_age_days * 86400.0
                    ]
                count += before - len(self.working.items)
            # capacity_based 对工作记忆意义不大，跳过
        if self.episodic:
            count += self.episodic.forget(strategy, threshold, max_age_days)
        return count

    # ---------- 精准删除（支持 Agent '忘' 动作）----------
    def remove_memory(self, identifier: str) -> int:
        """精准删除某条记忆：按 id 或内容子串，跨 working + episodic。返回删除条数。"""
        count = 0
        if self.working:
            count += self.working.remove(identifier)
        if self.episodic:
            count += self.episodic.delete(identifier)
        return count

    # ---------- 整合：短期 → 长期 ----------
    def consolidate_memories(
        self, from_type="working", to_type="episodic", importance_threshold=0.7
    ) -> int:
        """把重要的短期记忆固化为长期记忆（教材默认 working→episodic，阈值 0.7）。"""
        if from_type != "working" or not self.working or not self.episodic:
            return 0
        candidates = [
            it for it in self.working.items if it.importance >= importance_threshold
        ]
        for it in candidates:
            it.metadata = {**it.metadata, "consolidated_from": "working"}
            self.episodic.add(it)
        # 从工作记忆移除已固化的项
        self.working.items = [
            it for it in self.working.items if it.importance < importance_threshold
        ]
        return len(candidates)

    def stats(self) -> dict:
        s = {}
        if self.working:
            s["working"] = self.working.stats()
        if self.episodic:
            s["episodic"] = self.episodic.stats()
        return s
