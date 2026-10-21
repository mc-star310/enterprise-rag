"""AdvancedRAG(HyDE + MQE) 评测对比：纯混合检索 vs 混合+高级检索扩展。

前置：.env 中配置了 DEEPSEEK_API_KEY + DEEPSEEK_BASE_URL，且 Ollama 已启动（bge-m3）。
运行：
    cd enterprise-rag
    .venv/Scripts/python.exe -m kb.eval.t_eval_advanced

说明：
  - 同一次运行内先跑纯混合检索 baseline（HybridRAG），再跑 AdvancedRAG（MQE + HyDE），
    直接对比 R@k / MRR@5 的变化，验证高级检索扩展是否进一步修正长文档语义漂移。
  - AdvancedRAG 的扩展查询需要真实 LLM（DeepSeek），因此本脚本不走 CI（CI 用 t_eval_enterprise 的纯字面基线）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from kb.ingest import build_rag
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.rag.advanced_retrieval import AdvancedRAG
from kb.eval.t_eval_enterprise import evaluate


def main():
    corpus = os.path.join(ROOT, "kb", "corpus")
    print("=" * 60)
    print("构建 base HybridRAG（Ollama 语义嵌入 + TF-IDF 字面 + RRF）")
    print("=" * 60)
    base = build_rag(corpus, llm=HelloAgentsLLM())
    st = base.stats()
    print(f"索引：{st['chunks']} 块 | 嵌入维度 dim={st.get('dim')}")

    print("\n" + "=" * 60)
    print("A. 纯混合检索 baseline（HybridRAG）")
    print("=" * 60)
    evaluate(base)

    print("\n" + "=" * 60)
    print("B. AdvancedRAG（MQE 多查询扩展 + HyDE 假设文档嵌入）")
    print("=" * 60)
    adv = AdvancedRAG(base, enable_mqe=True, enable_hyde=True, mqe_count=3)
    evaluate(adv)


if __name__ == "__main__":
    main()
