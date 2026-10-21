"""HelloAgents 记忆系统 · 工作记忆（8.2 参考实现，轻栈版）。

教材 8.2 把记忆拆成四种类型 + 一条生命周期（编码 → 存储 → 检索 → 整合 → 遗忘）。
本文件只实现最基础、零依赖的【工作记忆 WorkingMemory】：

- 纯内存存储（对应教材“当前会话临时上下文”），不落盘
- 容量上限 max_capacity 条，满了丢最老的（对应教材“容量限制”）
- TTL 自动过期清理（对应教材“会话结束自动清理”）
- 检索评分 = 相似度 × 重要性 × 时间衰减（对应教材“评分三因素”公式）

后面要做的 EpisodicMemory / SemanticMemory / MemoryManager 都会复用这里的
MemoryItem 数据结构，所以这一层是整章地基，先吃透。
"""

import re
import time
import uuid
from dataclasses import dataclass, field

# 两个轻量分词正则：英文/数字按词，中文按单字。
# 这样同一套相似度算法对中英混合语料都 work（你的企业知识库就是中英混合）。
_WORD = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> set:
    """把文本转成 token 集合：英文/数字词 + 中文单字，统一小写。"""
    text = (text or "").lower()
    tokens = set(_WORD.findall(text))   # 英文单词、数字串
    tokens |= set(_CJK.findall(text))   # 每个汉字单独成 token
    return tokens


@dataclass
class MemoryItem:
    """一条记忆的统一数据结构（教材里就叫 MemoryItem）。

    importance 和 timestamp 是后面所有评分公式的原材料，必须记住：
    - importance：0.0~1.0，模拟大脑对这条信息重要性的主观评估
    - timestamp：写入时间，用于时间衰减（越久的分越低）
    """

    content: str
    memory_type: str = "working"
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class WorkingMemory:
    """工作记忆：纯内存 + TTL 自动清理 + 综合评分检索。"""

    def __init__(self, max_capacity: int = 50, ttl_minutes: int = 60):
        self.max_capacity = max_capacity          # 容量上限
        self.ttl = ttl_minutes * 60               # 过期时间（秒）
        self.items: list[MemoryItem] = []

    # ---------- 生命周期 1：编码 + 存储（add）----------
    def add(self, item: MemoryItem) -> str:
        self._expire()                              # 写入前先清理过期项
        if len(self.items) >= self.max_capacity:    # 容量满了丢最老的
            self.items.pop(0)
        self.items.append(item)
        return item.id

    # ---------- 生命周期 4/5：检索 + （遗忘在 _expire 里）----------
    def retrieve(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """按“相似度 × 重要性 × 时间衰减”综合打分，返回 top-limit 条。"""
        self._expire()
        scored = []
        for it in self.items:
            sim = self._similarity(query, it.content)
            if sim <= 0:
                continue
            importance_factor = 0.5 + 0.5 * it.importance   # 重要的放大
            decay = self._time_decay(it)                    # 越旧衰减
            scored.append((sim * importance_factor * decay, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]

    def clear(self) -> None:
        """清空（对应会话结束）。"""
        self.items.clear()

    def remove(self, identifier: str) -> int:
        """精准删除：移除 id 匹配或 content 含 identifier 的项（支持 Agent '忘' 动作）。"""
        idt = (identifier or "").strip()
        if not idt:
            return 0
        before = len(self.items)
        self.items = [
            it for it in self.items
            if not (it.id == idt or idt in it.content)
        ]
        return before - len(self.items)

    def stats(self) -> dict:
        return {"count": len(self.items), "max_capacity": self.max_capacity}

    # ---------- 内部辅助 ----------
    def _expire(self) -> None:
        """遗忘策略之一：TTL 过期自动删除。"""
        now = time.time()
        self.items = [it for it in self.items if now - it.timestamp < self.ttl]

    @staticmethod
    def _similarity(query: str, content: str) -> float:
        """查询词命中率：query 的 token 有几个出现在 content 里。"""
        q = _tokenize(query)
        if not q:
            return 0.0
        c = _tokenize(content)
        return len(q & c) / len(q)

    @staticmethod
    def _time_decay(item: MemoryItem, half_life: float = 1800.0) -> float:
        """时间衰减：半衰期 30 分钟，最旧也保留 0.5 权重（不让老记忆直接归零）。"""
        age = time.time() - item.timestamp
        return 0.5 + 0.5 * (0.5 ** (age / half_life))
