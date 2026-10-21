"""HelloAgents 记忆系统 · 情景记忆（8.2 轻栈实现）。

教材的 EpisodicMemory 用 SQLite（结构化）+ Qdrant（云向量库）的混合存储，依赖重。
我们按轻栈路线改为：sqlite3（Python 标准库，零依赖）做持久化 + 关键词相似度 +
时间近因性评分。核心能力不变：
- 长期存储具体事件 / 经历（重启不丢，区别于工作记忆）
- 支持按会话 / 重要性检索
- 评分三因素：相似度 × 时间近因性 × 重要性

后续 SemanticMemory 可在此之上加向量路（复用 framework 的 embedding fallback）。
"""

import json
import sqlite3
import threading
import time

from .working_memory import MemoryItem, WorkingMemory


class EpisodicMemory:
    """情景记忆：SQLite 持久化 + 关键词 / 时间混合评分检索。"""

    def __init__(self, db_path: str = ":memory:", recency_half_life_days: float = 7.0):
        self.db_path = db_path
        # check_same_thread=False：retrieve 可能在非创建线程调用
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.recency_half_life = recency_half_life_days * 86400.0
        self._init_schema()

    def _init_schema(self):
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    session_id TEXT,
                    metadata TEXT
                )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_session ON episodes(session_id)"
            )

    # ---------- 生命周期 1：编码 + 存储 ----------
    def add(self, item: MemoryItem) -> str:
        session_id = item.metadata.get("session_id", "default")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO episodes "
                "(id, content, memory_type, importance, timestamp, session_id, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    item.id,
                    item.content,
                    item.memory_type,
                    item.importance,
                    item.timestamp,
                    session_id,
                    json.dumps(item.metadata, ensure_ascii=False),
                ),
            )
        return item.id

    # ---------- 生命周期 4：检索 ----------
    def retrieve(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.1,
        session_id: str = None,
    ) -> list[MemoryItem]:
        rows = self._fetch_all(min_importance, session_id)
        scored = []
        for row in rows:
            item = self._row_to_item(row)
            sim = WorkingMemory._similarity(query, item.content)
            if sim <= 0:
                continue
            recency = self._recency(item.timestamp)
            importance_factor = 0.8 + item.importance * 0.4
            # 教材 episode 评分：(相似度×0.8 + 近因性×0.2) × 重要性权重
            score = (sim * 0.8 + recency * 0.2) * importance_factor
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]

    # ---------- 生命周期 5：遗忘（三种策略）----------
    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.2,
        max_age_days: int = 30,
    ) -> int:
        now = time.time()
        if strategy == "importance_based":
            cond, params = "importance < ?", (threshold,)
        elif strategy == "time_based":
            cond, params = "timestamp < ?", (now - max_age_days * 86400.0,)
        elif strategy == "capacity_based":
            with self._lock:
                total = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                if total == 0:
                    return 0
                offset = int(total * threshold)
                row = self._conn.execute(
                    "SELECT importance FROM episodes ORDER BY importance ASC "
                    "LIMIT 1 OFFSET ?",
                    (offset,),
                ).fetchone()
            cutoff = row[0] if row else 0.0
            cond, params = "importance <= ?", (cutoff,)
        else:
            raise ValueError(f"未知遗忘策略: {strategy}")
        with self._lock, self._conn:
            cur = self._conn.execute(f"DELETE FROM episodes WHERE {cond}", params)
            return cur.rowcount

    def delete(self, identifier: str) -> int:
        """精准删除：按 id 或 content 子串删除一条记忆（支持 Agent '忘' 动作）。"""
        idt = (identifier or "").strip()
        if not idt:
            return 0
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM episodes WHERE id = ? OR content LIKE ?",
                (idt, f"%{idt}%"),
            )
            return cur.rowcount

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            sessions = self._conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM episodes"
            ).fetchone()[0]
        return {"episodes": total, "sessions": sessions, "db_path": self.db_path}

    def close(self):
        self._conn.close()

    # ---------- 内部 ----------
    def _fetch_all(self, min_importance: float, session_id: str | None) -> list:
        with self._lock:
            if session_id:
                rows = self._conn.execute(
                    "SELECT id, content, memory_type, importance, timestamp, "
                    "session_id, metadata FROM episodes "
                    "WHERE importance >= ? AND session_id = ?",
                    (min_importance, session_id),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, content, memory_type, importance, timestamp, "
                    "session_id, metadata FROM episodes WHERE importance >= ?",
                    (min_importance,),
                ).fetchall()
        return rows

    @staticmethod
    def _row_to_item(row) -> MemoryItem:
        id_, content, mtype, importance, ts, sid, meta = row
        return MemoryItem(
            content=content,
            memory_type=mtype,
            importance=importance,
            timestamp=ts,
            id=id_,
            metadata=json.loads(meta) if meta else {},
        )

    def _recency(self, timestamp: float) -> float:
        age = time.time() - timestamp
        return 0.5 + 0.5 * (0.5 ** (age / self.recency_half_life))
