"""最小可用 RAG（第 8 章入门版）

RAG（Retrieval-Augmented Generation，检索增强生成）的本质只有三步：
    1. 切块（Chunking）：把长文档切成一小段一小段
    2. 检索（Retrieval）：从所有块里找出和问题最相关的几块
    3. 生成（Generation）：把这几块塞进提示词，让模型基于它们回答

这一版刻意做到"零外部依赖"：
    - 检索用 TF-IDF + 余弦相似度（scikit-learn），不装向量数据库、不申请嵌入 API key
    - 中文分词用"字符 n-gram"，不需要 jieba，中文英文都能用
    - 生成还是用你在第七章写的 HelloAgentsLLM（DeepSeek）

先跑通这个闭环，再往上换成"真嵌入模型 + Chroma 向量库"——那只是把第 2 步换掉，
整体结构完全不变。这也是学 RAG 的正确顺序：先懂骨架，再换零件。
"""

import os
import re
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.llm import HelloAgentsLLM


class MinimalRAG:
    """一个能跑起来的最小 RAG 系统。"""

    def __init__(self, llm: HelloAgentsLLM, chunk_size: int = 300, top_k: int = 3):
        self.llm = llm
        self.chunk_size = chunk_size   # 每个块大约多少字
        self.top_k = top_k             # 每次检索返回几块
        self.chunks: List[Dict[str, Any]] = []   # 所有文本块：{"text":..., "source":...}
        self._vectorizer: TfidfVectorizer = None
        self._matrix = None            # 所有块的 TF-IDF 向量矩阵

    # ---------- 第 1 步：切块 ----------
    def add_text(self, text: str, source: str = "unknown"):
        """把一段文本切块后加入知识库。"""
        # 先按空行/标题切成自然段，再把过长的段落按句号继续切
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for para in paragraphs:
            for piece in self._split_long(para):
                self.chunks.append({"text": piece, "source": source})
        self._matrix = None  # 有新内容，索引失效，下次检索时重建
        print(f"📄 已加入 {len(paragraphs)} 段（来自 {source}），当前共 {len(self.chunks)} 块")

    def _split_long(self, text: str) -> List[str]:
        """把超过 chunk_size 的段落按句号切成多个块。"""
        if len(text) <= self.chunk_size:
            return [text]
        pieces, buf = [], ""
        # 按中英文句末标点切句
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
        """把一个目录下的 .md / .txt 文件全部载入知识库。"""
        if not os.path.isdir(dir_path):
            print(f"⚠️ 目录不存在：{dir_path}")
            return
        for name in os.listdir(dir_path):
            if not name.lower().endswith((".md", ".txt")):
                continue
            path = os.path.join(dir_path, name)
            with open(path, "r", encoding="utf-8") as f:
                self.add_text(f.read(), source=name)

    # ---------- 第 2 步：检索 ----------
    def _build_index(self):
        """为所有块建立 TF-IDF 索引。"""
        # analyzer="char" + ngram_range=(1,2)：按字符一二元组切分，中文不需要分词器
        self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """找出和 query 最相关的 top_k 个块。"""
        if not self.chunks:
            return []
        if self._matrix is None:
            self._build_index()

        k = top_k or self.top_k
        q_vec = self._vectorizer.transform([query])
        # 余弦相似度：越接近 1 越相关
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            results.append(
                {
                    "text": self.chunks[i]["text"],
                    "source": self.chunks[i]["source"],
                    "score": float(scores[i]),
                }
            )
        return results

    # ---------- 第 3 步：生成 ----------
    def ask(self, question: str, with_sources: bool = True) -> str:
        """检索 + 生成，返回答案。"""
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
        return {"chunks": len(self.chunks), "sources": len({c['source'] for c in self.chunks})}
