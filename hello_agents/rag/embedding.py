"""嵌入层（第 8 章 RAG 的核心零件之一）

嵌入（Embedding）是什么？
    把一段文字变成一个数字向量（比如 1024 维的浮点数组）。
    语义相近的文字，算出来的向量在空间中就靠得近。
    所以"检索"就变成了"找和问题的向量最相近的那几个文档向量"。

为什么要有这一层抽象？
    嵌入方案有很多：本地 Ollama、硅基流动 API、阿里百炼 API、本地 sentence-transformers……
    如果代码里写死某一种，换方案就要改遍所有地方。
    所以定义一个统一的接口 BaseEmbedding，具体实现可插拔——
    换嵌入方案时，只需要换传给 RAG 的那个对象，其他代码一行不动。

三种实现：
    OllamaEmbedding            本地免费，需要 ollama serve + bge-m3 模型
    OpenAICompatibleEmbedding  任意 OpenAI 兼容的 /embeddings 接口（硅基流动/百炼/…）
    TFIDFEmbedding             零依赖兜底，纯字面匹配，不算真正的语义嵌入
"""

from abc import ABC, abstractmethod
from typing import List
import os

import requests


class BaseEmbedding(ABC):
    """嵌入模型的统一接口。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """把一批文本转成向量列表。"""
        pass

    def embed_query(self, text: str) -> List[float]:
        """把单个查询转成向量（默认复用 embed）。"""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """向量维度。"""
        vec = self.embed_query("测试")
        return len(vec)


class OllamaEmbedding(BaseEmbedding):
    """用本机 Ollama 做嵌入，完全免费、不联网。

    前置条件：
        1. ollama serve 已在运行（默认 http://localhost:11434）
        2. 已拉取嵌入模型：ollama pull bge-m3
    """

    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


class OpenAICompatibleEmbedding(BaseEmbedding):
    """调用任意 OpenAI 兼容的嵌入接口。

    常见配置：
        硅基流动（免费额度）：base_url=https://api.siliconflow.cn/v1  model=BAAI/bge-m3
        阿里百炼：base_url=https://dashscope.aliyuncs.com/compatible-mode/v1  model=text-embedding-v3
    需要在 .env 里配置对应的 key。
    """

    def __init__(self, model: str, base_url: str, api_key: str = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def embed(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # 接口返回的 data 是按 index 排序的，按 index 取回保证顺序一致
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


class TFIDFEmbedding(BaseEmbedding):
    """兜底方案：用 TF-IDF 生成"伪向量"。

    注意：它只看字面重合，不懂语义。问"优势"匹配不到写的是"动机"的段落。
    存在的意义是：当没有任何嵌入服务可用时，系统仍然能跑起来。
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
        self._fitted = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            # 第一次调用时，把传入的文本当作"语料库"来拟合
            self._vectorizer.fit(texts)
            self._fitted = True
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().tolist()


def create_embedding_with_fallback(prefer: str = "ollama") -> BaseEmbedding:
    """工厂函数：按优先级挑一个能用的嵌入方案。

    这样写的好处：上层代码只管拿一个 embedding 对象来用，
    至于背后是本地 Ollama 还是云端 API，它不需要关心。
    """
    # 允许通过环境变量强制指定嵌入方案（CI / 纯字面基线 / 无 GPU 场景）
    forced = (os.getenv("RAG_EMBEDDING") or "").lower()
    if forced in ("tfidf", "lexical", "none"):
        print("⚠️ 使用 TF-IDF 兜底嵌入（RAG_EMBEDDING=%s）" % forced)
        return TFIDFEmbedding()
    if forced in ("ollama", "bge-m3"):
        prefer = "ollama"

    if prefer == "ollama":
        try:
            emb = OllamaEmbedding()
            emb.embed_query("测试连通性")   # 真的试一次，确认服务和模型都在
            print("✅ 使用本地 Ollama 嵌入（bge-m3）")
            return emb
        except Exception as e:
            print(f"⚠️ Ollama 嵌入不可用（{e}），回退到 TF-IDF")

    # 云端方案：只有配了 key 才尝试
    sf_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("EMBEDDING_API_KEY")
    if sf_key:
        try:
            emb = OpenAICompatibleEmbedding(
                model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
                api_key=sf_key,
            )
            emb.embed_query("测试连通性")
            print("✅ 使用云端嵌入 API")
            return emb
        except Exception as e:
            print(f"⚠️ 云端嵌入不可用（{e}），回退到 TF-IDF")

    print("⚠️ 使用 TF-IDF 兜底嵌入（只匹配字面，不理解语义）")
    return TFIDFEmbedding()
