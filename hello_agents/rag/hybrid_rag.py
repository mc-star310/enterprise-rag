"""混合检索 RAG（第 8 章进阶）

前面两版：
    MinimalRAG  : 只有字面检索（TF-IDF），擅长精确关键词，但不懂语义
    SemanticRAG : 只有语义检索（嵌入），懂同义改写，但遇到专有名词反而不准

现实里的生产级 RAG，几乎都是「两路一起上」：
    一路 BM25 / TF-IDF 抓字面精确匹配
    一路 嵌入向量     抓语义相关
    再把两路的排序结果融合成一个更准的排序 —— 这一步叫 RRF（Reciprocal Rank Fusion）

为什么不直接用语义？
    「2026 世界杯冠军」这种问题，字面里带着具体名词，TF-IDF 一眼就中；
    而「知识有哪些更新方式」这种问题，语义路更能懂你想问的是「RAG 动机」那段。
    两条路互补，谁都替代不了谁 —— 这点在昨天的对比实验里已经亲眼看到了。

RRF 最大的好处：它只看「排名」不看「分数大小」。
    语义路分数是余弦相似度（0~1），字面路是 TF-IDF 权重，量纲完全不同；
    但 RRF 把两路都转成「第几名 → 1/(名次+60)」，直接相加即可，
    免去了繁琐的分数归一化。这是它被广泛采用的核心原因。
"""

import os
import re
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.llm import HelloAgentsLLM
from .embedding import BaseEmbedding, create_embedding_with_fallback
from .vector_store import VectorStore


class HybridRAG:
    """字面检索 + 语义检索 + RRF 融合的 RAG。"""

    def __init__(
        self,
        llm: HelloAgentsLLM,
        embedding: BaseEmbedding = None,
        chunk_size: int = 300,
        top_k: int = 3,
        rrf_k: int = 60,
        min_chunk_size: int = 80,
    ):
        self.llm = llm
        # 没传就用工厂函数自动挑一个能用的（Ollama → 云端 → TF-IDF）
        self.embedding = embedding or create_embedding_with_fallback()
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size  # 短于此的碎片并入上一块（代码块/孤行）
        self.top_k = top_k
        self.rrf_k = rrf_k  # RRF 的平滑常数，经验值 60

        # 两路检索共享同一份切块（关键：保证 index 对齐）
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

        # 语义路：向量库（复用已验证的 VectorStore）
        self.store = VectorStore()

        # 字面路：TF-IDF（自己维护，因为要拿到每个块的排名）
        self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        self._tfidf_fitted = False
        self._matrix = None

    # ---------- 第 1 步：切块（和前两版一致）----------
    def add_text(self, text: str, source: str = "unknown"):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        raw_pieces = []
        for para in paragraphs:
            for piece in self._split_long(para):
                raw_pieces.append(piece)
        # 小块合并：代码块、孤行等被空行切出的短碎片（如一行命令）
        # 单独成块时字面/语义信号都太弱，检索排不上来——并入上一块。
        new_chunks = []
        for piece in raw_pieces:
            if len(piece) < self.min_chunk_size and new_chunks:
                new_chunks[-1] += "\n" + piece
            else:
                new_chunks.append(piece)
        if not new_chunks:
            return
        # 两路都记录同一批块
        self.texts.extend(new_chunks)
        self.metadatas.extend([{"source": source}] * len(new_chunks))
        self._tfidf_fitted = False  # 文本变了，TF-IDF 索引失效，下次检索重建
        # 语义路：增量嵌入并加入向量库
        vectors = self.embedding.embed(new_chunks)
        self.store.add(
            texts=new_chunks,
            vectors=vectors,
            metadatas=[{"source": source}] * len(new_chunks),
        )
        print(f"📄 已加入 {len(new_chunks)} 块（来自 {source}），当前共 {len(self.texts)} 块")

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

    # ---------- 第 2 步：混合检索 ----------
    def _lexical_retrieve(self, query: str, candidate_k: int) -> Dict[int, float]:
        """字面路：TF-IDF 余弦相似度，返回 {块index: 分数}。"""
        if not self._tfidf_fitted:
            self._matrix = self._vectorizer.fit_transform(self.texts)
            self._tfidf_fitted = True
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_k]
        # 只保留有正相关的块，避免噪声倒灌
        return {i: float(scores[i]) for i in ranked if scores[i] > 0}

    def _semantic_retrieve(self, query: str, candidate_k: int) -> Dict[int, float]:
        """语义路：向量库余弦相似度，返回 {块index: 分数}。"""
        q_vec = self.embedding.embed_query(query)
        hits = self.store.search(q_vec, top_k=candidate_k)
        # store.search 已带 index 字段，直接拿来用
        return {h["index"]: h["score"] for h in hits}

    def _rrf_merge(self, lexical: Dict[int, float], semantic: Dict[int, float], top_k: int) -> List[Dict[str, Any]]:
        """RRF 融合：每路按分数排好名次，第 rank 名得分 1/(rank + k)，两路相加。"""
        fused: Dict[int, float] = {}
        for route in (lexical, semantic):
            # 同一路内部先按分数排好名次
            ranked = sorted(route.keys(), key=lambda i: route[i], reverse=True)
            for rank, idx in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rank + self.rrf_k)
        # 按融合分取最终 top_k
        final = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]
        return [
            {
                "text": self.texts[i],
                "source": self.metadatas[i].get("source", "?"),
                "score": fused[i],  # 这里是 RRF 融合分，不是原始相似度
            }
            for i in final
        ]

    def retrieve(self, query: str, top_k: int = None, candidate_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        # 候选池要远大于最终返回数，融合才有意义（小语料时取全部块也行）
        cand = candidate_k or max(k * 4, 12)
        lex = self._lexical_retrieve(query, cand)
        sem = self._semantic_retrieve(query, cand)
        return self._rrf_merge(lex, sem, k)

    # ---------- 第 3 步：生成（和前两版一致）----------
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
            sources = "、".join(f"{h['source']}(RRF={h['score']:.4f})" for h in hits)
            answer += f"\n\n【出处】{sources}"
        return answer

    def stats(self) -> Dict[str, Any]:
        return {
            "chunks": len(self.texts),
            "dim": self.embedding.dimension,
        }
