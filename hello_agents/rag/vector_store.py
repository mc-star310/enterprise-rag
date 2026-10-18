"""向量库（第 8 章 RAG 的第二个零件）

向量库负责两件事：
    1. 存：把文本块和它对应的向量存起来
    2. 查：给一个查询向量，找出最相近的 top_k 个块

这一版先用"内存 + numpy"实现，把原理讲透：
    相似度 = 余弦相似度（两个向量夹角的余弦值，越接近 1 越像）
以后换成 Chroma / Milvus，对外接口（add / search）完全不变，
上层 RAG 代码一行都不用改——这就是抽象的价值。
"""

import json
import os
from typing import List, Dict, Any

import numpy as np


class VectorStore:
    """极简向量库：内存存储 + 余弦相似度检索，可选持久化到 json。"""

    def __init__(self):
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = None  # shape: (n_chunks, dim)

    # ---------- 存 ----------
    def add(self, texts: List[str], vectors: List[List[float]], metadatas: List[Dict] = None):
        """把一批文本块及其向量加入库。"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

        new_vecs = np.array(vectors, dtype="float32")
        if self.vectors is None:
            self.vectors = new_vecs
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])

    # ---------- 查 ----------
    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """检索与查询向量最相似的 top_k 个块。"""
        if self.vectors is None or len(self.texts) == 0:
            return []

        q = np.array(query_vector, dtype="float32")
        # 余弦相似度 = (A·B) / (|A| * |B|)
        # 先各自归一化，点积就直接等于余弦相似度
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        v_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)
        scores = v_norm @ q_norm

        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "index": int(i),
                "text": self.texts[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
            }
            for i in ranked
        ]

    # ---------- 持久化（可选）----------
    def save(self, path: str):
        """把向量库存成 json，下次直接加载，省去重新嵌入的时间和算力。"""
        data = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"💾 向量库已保存到 {path}（{len(self.texts)} 块）")

    def load(self, path: str) -> bool:
        """从 json 加载向量库。"""
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data["texts"]
        self.metadatas = data["metadatas"]
        self.vectors = np.array(data["vectors"], dtype="float32")
        print(f"📂 已加载向量库 {path}（{len(self.texts)} 块）")
        return True

    def __len__(self):
        return len(self.texts)
