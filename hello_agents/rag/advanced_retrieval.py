"""高级检索策略（8.3.5 轻栈实现）

教材这一节给了三种互补策略：
    MQE  多查询扩展：用 LLM 把一个问题改写成多种表述，分别检索后合并，提升召回率
    HyDE 假设文档嵌入：用 LLM 先写一段"假设答案"，用这段答案（而非原问题）去检索，
                       缩小"问题(疑问句) ↔ 文档(陈述句)"的语义鸿沟，提升检索精度
    统一框架：扩展-检索-合并 三步流程

教材原版依赖 Qdrant 向量库 + 一堆 filter（重栈）。我们保留"扩展-检索-合并"的核心思想，
但底层复用你已经验证过的 HybridRAG（字面 TF-IDF + 语义 bge-m3 + RRF 融合），做到零新依赖。
这样有双重收益：
    - 有 Ollama 时：HyDE 的假设文档走语义路，直接命中真实文档的语义近邻
    - 无 Ollama（CI / 字面模式）时：MQE 的多角度表述 + HyDE 的术语段落，也能让字面路覆盖更多词
关键改进：教材合并是按 id 去重并"保留单路最高分"；我们用 RRF 累加 —— 多个扩展查询都命中的块
理应排更前（这正是多查询融合的价值），比"取单路最高分"更能体现融合增益。
"""

from typing import List, Dict, Any, Optional


def _prompt_mqe(llm, query: str, n: int) -> List[str]:
    """用 LLM 生成 n 个不同表述的扩展查询（教材 8.3.5 思路，轻栈化：复用传入 llm，不新建）。"""
    try:
        prompt = [
            {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
            {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"},
        ]
        text = llm.invoke(prompt)
        lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
        outs = [ln for ln in lines if ln]
        return outs[:n] or [query]
    except Exception:
        return [query]


def _prompt_hyde(llm, query: str) -> Optional[str]:
    """用 LLM 生成一段假设性答案段落（HyDE），用于替代原问题去检索。"""
    try:
        prompt = [
            {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
            {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"},
        ]
        text = llm.invoke(prompt)
        return text.strip() if text else None
    except Exception:
        return None


class AdvancedRAG:
    """在 HybridRAG 之上叠加 MQE / HyDE 扩展检索。

    用法：
        base = HybridRAG(llm, embedding)
        base.add_text(corpus, source="kb")
        adv = AdvancedRAG(base, enable_mqe=True, enable_hyde=True)
        hits = adv.retrieve("打印机连不上怎么办")   # 自动扩展后融合检索
        print(adv.ask("打印机连不上怎么办"))        # 用扩展检索结果生成答案
    """

    def __init__(
        self,
        base_rag,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        mqe_count: int = 3,
        use_llm: bool = True,
        rrf_k: int = 60,
    ):
        self.base_rag = base_rag
        self.enable_mqe = enable_mqe
        self.enable_hyde = enable_hyde
        self.mqe_count = mqe_count
        self.use_llm = use_llm
        self.rrf_k = rrf_k

    # ---------- 内部：LLM 是否真可用 ----------
    def _has_real_llm(self) -> bool:
        llm = getattr(self.base_rag, "llm", None)
        if llm is None:
            return False
        # 真 LLM（HelloAgentsLLM）有 api_key；_StubLLM / 测试桩无 key，自动降级
        return bool(getattr(llm, "api_key", None))

    def _llm_ready(self) -> bool:
        return self.use_llm and self._has_real_llm()

    # ---------- 第 1 步：扩展 ----------
    def _expand(self, query: str) -> List[str]:
        expansions = [query]
        if self._llm_ready():
            llm = self.base_rag.llm
            if self.enable_mqe:
                expansions.extend(_prompt_mqe(llm, query, self.mqe_count))
            if self.enable_hyde:
                h = _prompt_hyde(llm, query)
                if h:
                    expansions.append(h)
        # 去重保序
        seen, uniq = set(), []
        for e in expansions:
            if e and e not in seen:
                seen.add(e)
                uniq.append(e)
        return uniq

    # ---------- 第 2+3 步：逐查询检索 + RRF 合并 ----------
    def retrieve(self, query: str, top_k: int = None, candidate_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or getattr(self.base_rag, "top_k", 3)
        cand = candidate_k or max(k * 4, 12)
        queries = self._expand(query)
        fused: Dict[tuple, float] = {}          # (source, text) -> RRF 融合分
        meta: Dict[tuple, Dict] = {}
        for q in queries:
            hits = self.base_rag.retrieve(q, top_k=cand)
            for rank, h in enumerate(hits):
                key = (h.get("source", "?"), h.get("text", ""))
                fused[key] = fused.get(key, 0.0) + 1.0 / (rank + self.rrf_k)
                meta[key] = h
        ranked = sorted(fused.keys(), key=lambda key: fused[key], reverse=True)[:k]
        return [
            {"text": meta[key]["text"], "source": meta[key]["source"], "score": fused[key]}
            for key in ranked
        ]

    def ask(self, question: str, with_sources: bool = True) -> str:
        hits = self.retrieve(question)
        return self.base_rag.ask(question, with_sources=with_sources, hits=hits)
