"""语义版 RAG（第 8 章进阶）

和 minimal_rag.py 的骨架**完全一样**：切块 → 检索 → 生成。
唯一的区别是检索零件升级了：

    MinimalRAG  : TF-IDF 字面匹配
    SemanticRAG : 嵌入向量语义匹配（可插拔：Ollama / 云端 API / TF-IDF 兜底）

这个对比本身就是本章最重要的一课：
**RAG 的结构是稳定的，效果差异几乎全在"检索"这一步。**
后面要加的"混合检索 + RRF + 重排"，也都是只动这一步。
"""

import os
import re
from typing import List, Dict, Any

from ..core.llm import HelloAgentsLLM
from .embedding import BaseEmbedding, create_embedding_with_fallback
from .vector_store import VectorStore


class SemanticRAG:
    """语义检索版 RAG。"""

    def __init__(
        self,
        llm: HelloAgentsLLM,
        embedding: BaseEmbedding = None,
        chunk_size: int = 300,
        top_k: int = 3,
        store_path: str = None,
    ):
        self.llm = llm
        # 没传就用工厂函数自动挑一个能用的（Ollama → 云端 → TF-IDF）
        self.embedding = embedding or create_embedding_with_fallback()
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.store = VectorStore()
        self.store_path = store_path
        if store_path and self.store.load(store_path):
            print("（已复用本地向量库，跳过重新嵌入）")

    # ---------- 第 1 步：切块（和 MinimalRAG 一模一样）----------
    def add_text(self, text: str, source: str = "unknown"):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        new_chunks = []
        for para in paragraphs:
            for piece in self._split_long(para):
                new_chunks.append((piece, {"source": source}))

        if not new_chunks:
            return
        # 批量嵌入（比一条条调快得多）
        vectors = self.embedding.embed([c[0] for c in new_chunks])
        self.store.add(
            texts=[c[0] for c in new_chunks],
            vectors=vectors,
            metadatas=[c[1] for c in new_chunks],
        )
        print(f"📄 已加入 {len(new_chunks)} 块（来自 {source}），当前共 {len(self.store)} 块")
        if self.store_path:
            self.store.save(self.store_path)

    def _split_long(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        pieces, buf = [], ""
        for sentence in re.split(r"(?<=[。！？!?.])", text):
            if not sentence.strip():
                continue
            if len(buf) + len(sentence) > self.chunk_size and buf:
                pieces.append(buf.strip())
                buf = sentence
            else:
                buf += sentence
        if buf.strip():
            pieces.append(buf.strip())
        return pieces

    def load_directory(self, dir_path: str):
        if not os.path.isdir(dir_path):
            print(f"⚠️ 目录不存在：{dir_path}")
            return
        for name in os.listdir(dir_path):
            if not name.lower().endswith((".md", ".txt")):
                continue
            with open(os.path.join(dir_path, name), "r", encoding="utf-8") as f:
                self.add_text(f.read(), source=name)

    # ---------- 第 2 步：语义检索 ----------
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        q_vec = self.embedding.embed_query(query)
        hits = self.store.search(q_vec, top_k=top_k or self.top_k)
        # 统一对外字段，和 MinimalRAG 保持一致
        return [
            {"text": h["text"], "source": h["metadata"].get("source", "?"), "score": h["score"]}
            for h in hits
        ]

    # ---------- 第 3 步：生成（和 MinimalRAG 一模一样）----------
    def ask(self, question: str, with_sources: bool = True) -> str:
        hits = self.retrieve(question)
        if not hits:
            return "知识库里没有找到相关内容，无法回答。"

        context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
        prompt = (
            "请只根据下面提供的资料回答问题。如果资料里没有答案，"
            "就明确说'资料中没有提到'，不要编造。\n\n"
            f"资料：\n{context}\n\n"
            f"问题：{question}"
        )
        answer = self.llm.invoke([{"role": "user", "content": prompt}])

        if with_sources:
            sources = "、".join(f"{h['source']}({h['score']:.2f})" for h in hits)
            answer += f"\n\n【出处】{sources}"
        return answer

    def stats(self) -> Dict[str, Any]:
        return {
            "chunks": len(self.store),
            "dim": self.store.vectors.shape[1] if self.store.vectors is not None else 0,
        }
