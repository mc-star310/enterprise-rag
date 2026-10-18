"""端到端冒烟测试：不调 LLM，只验证「文档接入 → 混合检索」是否跑通。

运行：
    cd enterprise-rag
    python -m kb.smoke_test
"""

from kb.ingest import build_rag, ROOT

# 1) 接入语料并建索引（embedding 会自动 fallback：Ollama → TF-IDF，无需联网也能跑）
corpus = str(ROOT / "kb" / "corpus")
rag = build_rag(corpus)
print("索引统计：", rag.stats())

# 2) 跨三个部门的查询，验证检索能命中正确文档
queries = [
    "年假怎么申请",          # HR
    "忘记密码了怎么办",       # IT
    "什么是敏感个人信息",     # 合规
    "打印机连不上怎么处理",    # IT
]

print("\n========== 检索验证 ==========")
for q in queries:
    print(f"\n>>> 查询：{q}")
    hits = rag.retrieve(q, top_k=3)
    if not hits:
        print("  （无命中）")
    for i, h in enumerate(hits, 1):
        snippet = h["text"][:50].replace("\n", " ")
        print(f"  {i}. [{h['source']}] {snippet}...")
