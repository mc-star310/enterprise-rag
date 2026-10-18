"""企业知识库评测集：覆盖 HR / IT / 合规(PIPL) / 合规(自写) / 产品(Python) 五部门。

指标：
  - Recall@k：gold 文档（期望来源）是否出现在检索 top-k 的 source 中
  - MRR@5  ：基于 gold 文档首次出现排名的倒数均值

运行：
    cd enterprise-rag
    .venv/Scripts/python.exe -m kb.eval.t_eval_enterprise

说明：
  - 不调 LLM，纯检索评测，离线可跑。
  - 若 Ollama 已开启（bge-m3），HybridRAG 自动启用语义路，指标会显著优于
    纯 TF-IDF 降级模式——脚本会打印当前是「混合(含语义)」还是「纯 TF-IDF」，
    并列出 top-5 未命中 gold 的 case，便于定位语义漂移问题。
"""
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from kb.ingest import build_rag  # noqa: E402

CORPUS = os.path.join(ROOT, "kb", "corpus")
KS = (1, 3, 5)

# (query, gold 文件名 basename, 部门)
CASES = [
    # ---------------- HR（自写：hr_leave_policy.md） ----------------
    ("年假怎么申请", "hr_leave_policy.md", "HR"),
    ("年假一共有几天", "hr_leave_policy.md", "HR"),
    ("请病假需要提交证明吗", "hr_leave_policy.md", "HR"),
    ("加班可以调休吗怎么算", "hr_leave_policy.md", "HR"),
    ("请假要在哪里提交", "hr_leave_policy.md", "HR"),
    ("离职流程是什么", "hr_leave_policy.md", "HR"),

    # ---------------- IT（自写：it_helpdesk.md） ----------------
    ("忘记密码了怎么办", "it_helpdesk.md", "IT"),
    ("打印机连不上怎么处理", "it_helpdesk.md", "IT"),
    ("网络断了如何报修", "it_helpdesk.md", "IT"),
    ("怎么连接公司 VPN", "it_helpdesk.md", "IT"),
    ("电脑蓝屏了如何处理", "it_helpdesk.md", "IT"),
    ("怎么安装常用办公软件", "it_helpdesk.md", "IT"),

    # ---------------- 合规-PIPL（真实：pip_个人信息保护法.md） ----------------
    ("什么是敏感个人信息", "pip_个人信息保护法.md", "合规-PIPL"),
    ("个人信息处理者有哪些义务", "pip_个人信息保护法.md", "合规-PIPL"),
    ("向境外提供个人信息需要什么条件", "pip_个人信息保护法.md", "合规-PIPL"),
    ("个人在信息处理活动中享有哪些权利", "pip_个人信息保护法.md", "合规-PIPL"),
    ("处理个人信息的合法性基础有哪些", "pip_个人信息保护法.md", "合规-PIPL"),
    ("违法处理个人信息要承担什么法律责任", "pip_个人信息保护法.md", "合规-PIPL"),
    ("处理敏感个人信息需要单独同意吗", "pip_个人信息保护法.md", "合规-PIPL"),
    ("个人信息的保存期限怎么规定", "pip_个人信息保护法.md", "合规-PIPL"),

    # ---------------- 合规-自写（compliance_privacy_basics.md） ----------------
    ("员工处理个人信息要遵守什么原则", "compliance_privacy_basics.md", "合规-自写"),
    ("什么是数据最小化原则", "compliance_privacy_basics.md", "合规-自写"),
    ("发生个人信息泄露应该怎么办", "compliance_privacy_basics.md", "合规-自写"),
    ("员工为什么要做个人信息保护培训", "compliance_privacy_basics.md", "合规-自写"),

    # ---------------- 产品-Python（真实：devref_python_*.md） ----------------
    ("如何创建 Python 虚拟环境", "devref_python_venv.md", "产品-Python"),
    ("venv 虚拟环境怎么激活", "devref_python_venv.md", "产品-Python"),
    ("logging 模块怎么配置输出到文件", "devref_python_logging.md", "产品-Python"),
    ("logging 的日志级别有哪些", "devref_python_logging.md", "产品-Python"),
    ("什么是异步任务 asyncio", "devref_python_asyncio.md", "产品-Python"),
    ("Python 列表推导式怎么写", "devref_python_datastructures.md", "产品-Python"),
    ("Python 字典数据结构怎么用", "devref_python_datastructures.md", "产品-Python"),
    ("Python 入门有哪些基本概念", "devref_python_tutorial_intro.md", "产品-Python"),
]


def evaluate(rag):
    hit = defaultdict(lambda: {k: 0 for k in KS})      # 部门 -> 各 k 命中数
    mrr = defaultdict(lambda: 0.0)                     # 部门 -> MRR@5 累加
    cnt = defaultdict(int)                             # 部门 -> 题数
    overall_hit = {k: 0 for k in KS}
    overall_mrr = 0.0
    failed = []                                        # top-5 未命中 gold 的 case

    for q, gold, dept in CASES:
        hits = rag.retrieve(q, top_k=max(KS))
        sources = [h["source"] for h in hits]
        cnt[dept] += 1
        for k in KS:
            top = sources[:k]
            if gold in top:
                hit[dept][k] += 1
                overall_hit[k] += 1
                rank = top.index(gold) + 1
                mrr[dept] += 1.0 / rank
                if k == max(KS):
                    overall_mrr += 1.0 / rank
        if gold not in sources[: max(KS)]:
            failed.append((dept, q, sources[:3]))

    n_total = len(CASES)
    print(f"\n总题数：{n_total}\n")
    print(f"{'部门':<14}{'题数':>4}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'MRR@5':>9}")
    print("-" * 54)
    for dept in ["HR", "IT", "合规-PIPL", "合规-自写", "产品-Python"]:
        n = cnt[dept]
        r1 = hit[dept][1] / n if n else 0
        r3 = hit[dept][3] / n if n else 0
        r5 = hit[dept][5] / n if n else 0
        mm = mrr[dept] / n if n else 0
        print(f"{dept:<14}{n:>4}{r1:>9.2f}{r3:>9.2f}{r5:>9.2f}{mm:>9.2f}")
    print("-" * 54)
    r1 = overall_hit[1] / n_total
    r3 = overall_hit[3] / n_total
    r5 = overall_hit[5] / n_total
    print(f"{'整体':<14}{n_total:>4}{r1:>9.2f}{r3:>9.2f}{r5:>9.2f}{overall_mrr / n_total:>9.2f}")

    if failed:
        print(f"\n⚠️ Top-5 未命中 gold 的 case（{len(failed)} 条）：")
        for dept, q, top3 in failed:
            print(f"  [{dept}] {q}")
            for s in top3:
                print(f"       误命中 -> {s}")
    else:
        print("\n✅ 全部 case 的 gold 文档均进入 Top-5。")
    return overall_hit, n_total


def main():
    print("========== 构建企业知识库索引 ==========")
    rag = build_rag(CORPUS)
    st = rag.stats()
    mode = "混合(含语义嵌入)" if st.get("dim") and st.get("dim") != 782 else "纯 TF-IDF(已降级)"
    print(f"索引：{st['chunks']} 块 | 嵌入模式：{mode}")
    print("\n========== 检索评测 ==========")
    evaluate(rag)


if __name__ == "__main__":
    main()
