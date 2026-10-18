"""企业知识库 - 多格式文档接入管道。

把 PDF / Word / Markdown / TXT 统一读成纯文本，再喂给框架的 HybridRAG。
框架已经负责：智能分块、混合索引（TF-IDF + 语义 + RRF 融合）、检索、生成。
本模块只解决一件事：把"各种格式的文档"变成"纯文本"，让框架能直接吃。

仓库布局：
    enterprise-rag/
    ├── hello_agents/   # 从零手搓的 Agent/RAG 框架（被本模块直接 import）
    ├── kb/             # 本知识库应用（ingest.py / app.py / 语料）
    └── web/index.html  # 前端聊天页

本文件位于 kb/ingest.py，仓库根（含 hello_agents）是 parents[1]。
"""

import os
import sys
from pathlib import Path

# 让本文件能 import 到仓库根的 hello_agents 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hello_agents.rag.hybrid_rag import HybridRAG
from hello_agents.core.llm import HelloAgentsLLM

# 框架的 llm 层只读 os.getenv，不会自动 load_dotenv，这里手动加载仓库根 .env
from dotenv import load_dotenv
load_dotenv(str(ROOT / ".env"))

# 支持的文档格式
SUPPORTED = (".md", ".txt", ".pdf", ".docx")


def read_md_txt(path: str) -> str:
    """Markdown / 纯文本：直接读取。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    """PDF：用 PyMuPDF 逐页抽取文本。"""
    import fitz  # 按需导入，避免无 PDF 时也强依赖

    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def read_docx(path: str) -> str:
    """Word：用 python-docx 抽取段落文本。"""
    import docx  # 按需导入

    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def read_file(path: str) -> str:
    """按扩展名分派到对应的读取器。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".txt"):
        return read_md_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    return ""


def ingest_file(rag: HybridRAG, path: str, source: str = None):
    """读取单个文件并加入 RAG（source 用于答案出处标注）。"""
    text = read_file(path)
    if not text.strip():
        print(f"  ⚠️ 跳过空文件：{path}")
        return
    src = source or os.path.basename(path)
    rag.add_text(text, source=src)


def ingest_directory(rag: HybridRAG, dir_path: str):
    """递归扫描目录，按格式接入所有文档。"""
    for root, _, files in os.walk(dir_path):
        for name in sorted(files):
            if name.lower().endswith(SUPPORTED):
                ingest_file(rag, os.path.join(root, name), source=name)


def build_rag(corpus_dir: str, llm: HelloAgentsLLM = None, **kwargs) -> HybridRAG:
    """便捷函数：构建并灌满一个企业知识库 RAG。"""
    llm = llm or HelloAgentsLLM()
    rag = HybridRAG(llm, **kwargs)
    ingest_directory(rag, corpus_dir)
    return rag


if __name__ == "__main__":
    corpus = str(ROOT / "kb" / "corpus")
    rag = build_rag(corpus)
    print("索引统计：", rag.stats())
