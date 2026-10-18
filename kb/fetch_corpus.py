"""企业知识库语料获取脚本（可复现 + 来源可追溯）。

本脚本把「真实公开文档」下载并清洗为纯 Markdown，落到 kb/corpus/ 下。
所有语料均来自公开可验证来源，文件头标注原始链接与用途，未作篡改。

语料构成（demo 用）：
  - 合规库：
      * 中华人民共和国个人信息保护法（全文）—— 中国人大网（全国人大常委会官网）
  - 产品手册：
      * Python 3.13 官方中文文档 5 个核心章节 —— docs.python.org/zh-cn

运行（需联网）：
    cd enterprise-rag
    python -m kb.fetch_corpus

注意：
  - 抓取受网络环境影响；校园网可能屏蔽部分站点，失败时脚本会跳过并提示。
  - 已落盘的语料可直接用于检索；本脚本仅用于「重新获取 / 溯源」。
  - 自写语料（hr / it / compliance_privacy_basics）不在此脚本内，
    它们位于 kb/corpus/ 根目录，由人工维护。
"""
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPLIANCE_DIR = os.path.join(ROOT, "kb", "corpus", "compliance")
PRODUCT_DIR = os.path.join(ROOT, "kb", "corpus", "product")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; enterprise-rag-corpus-fetcher/1.0)"}


def _get(url: str, timeout: int = 25, retries: int = 2):
    last = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS)
            r.encoding = r.apparent_encoding or "utf-8"
            if r.status_code == 200:
                return r
        except Exception as e:  # noqa
            last = e
        time.sleep(1.5)
    print(f"  ❌ 抓取失败（已重试 {retries} 次）：{url}  {last}")
    return None


def fetch_pip():
    """抓取《个人信息保护法》全文（中国人大网，权威静态页）。"""
    os.makedirs(COMPLIANCE_DIR, exist_ok=True)
    url = "http://www.npc.gov.cn/npc/c2/c30834/202108/t20210820_313088.html"
    r = _get(url)
    if not r:
        return False
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    m = re.search(r"中华人民共和国个人信息保护法[\s\S]*", text)
    body = m.group(0) if m else text
    # 清洗网页导航噪声
    body = re.sub(r"当前位置：[\s\S]*?目　　录", "目　　录", body)
    body = re.sub(r"来源：\s*中国人大网[^\n]*\n", "", body)
    body = re.sub(r"浏览字号：\n大\n中\n小\n", "", body)
    header = (
        "# 中华人民共和国个人信息保护法（全文）\n\n"
        "> 来源：中国人大网（全国人大常委会官网）\n"
        f"> 原文链接：{url}\n"
        "> 说明：本文件为公开法律全文，用于企业知识库合规库 demo 语料，未作任何修改。\n\n"
    )
    out = os.path.join(COMPLIANCE_DIR, "pip_个人信息保护法.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + body + "\n")
    print(f"  ✅ PIPL 全文 {len(body)} 字 -> {os.path.relpath(out, ROOT)}")
    return True


# docs.python.org 页面中的纯导航噪声行（整行匹配才剔除，避免误伤正文）
_NAV_NOISE = {
    "主题", "自动", "明亮", "黑暗", "目录", "上一主题", "下一主题",
    "当前页", "报告代码错误", "改进本页面", "显示源码", "显示翻译源",
    "导航", "索引", "模块", "下一页", "上一页", "| 下一页 |", "| 上一页 |",
    "DOM Tree", "© 版权所有", "Built with Sphinx", "页面内容",
}


def _article_to_text(soup) -> str:
    """把 docs.python.org 的 <article> 抽成文本，保留代码块不拆碎。

    做法：先把每个 <pre> 的原文（含换行缩进）存入 codes，并原地替换成
    哨兵行 @@CODE_i@@；再做整页文本抽取与逐行清洗；最后把哨兵行换回
    Markdown 代码围栏。这样 `python -m venv` 之类命令保持连续完整。
    """
    art = soup.find("article") or soup
    codes: list[str] = []
    for pre in art.find_all("pre"):
        codes.append(pre.get_text())
        pre.replace_with(f"@@CODE_{len(codes) - 1}@@")
    lines = []
    for raw in art.get_text("\n").splitlines():
        line = raw.strip()
        if not line or line in _NAV_NOISE:
            continue
        if line.startswith(("| 下一页", "| 上一页", "©", "Built with")):
            continue
        lines.append(line)
    text = "\n".join(lines)
    # 哨兵行换回代码围栏（保留原始缩进与换行）
    for i, code in enumerate(codes):
        text = text.replace(f"@@CODE_{i}@@", "\n```\n" + code.rstrip() + "\n```\n")
    return text


def fetch_python_pages():
    """抓取 Python 官方中文文档若干核心章节，作为产品手册。"""
    os.makedirs(PRODUCT_DIR, exist_ok=True)
    pages = {
        "devref_python_tutorial_intro": "https://docs.python.org/zh-cn/3/tutorial/introduction.html",
        "devref_python_venv": "https://docs.python.org/zh-cn/3/library/venv.html",
        "devref_python_asyncio": "https://docs.python.org/zh-cn/3/library/asyncio-task.html",
        "devref_python_logging": "https://docs.python.org/zh-cn/3/library/logging.html",
        "devref_python_datastructures": "https://docs.python.org/zh-cn/3/tutorial/datastructures.html",
    }
    ok = 0
    for name, url in pages.items():
        r = _get(url)
        if not r:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        txt = _article_to_text(soup)
        header = (
            "# 示例科技开发者平台 · Python 技术参考（节选自 Python 3 官方中文文档）\n\n"
            f"> 来源：Python 官方中文文档 {url}\n"
            "> 说明：本文件为真实开源文档节选，用于企业知识库产品手册 demo 语料。\n\n"
        )
        out = os.path.join(PRODUCT_DIR, f"{name}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(header + txt + "\n")
        print(f"  ✅ {name} {len(txt)} 字 -> {os.path.relpath(out, ROOT)}")
        ok += 1
    return ok == len(pages)


def main():
    print("========== 抓取真实语料 ==========")
    print("[合规库] 个人信息保护法（中国人大网）")
    fetch_pip()
    print("[产品手册] Python 官方中文文档核心章节")
    fetch_python_pages()
    print("\n完成。自写语料（HR/IT/合规指引）已随仓库提供，无需抓取。")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
