"""企业知识库 - FastAPI + SSE 流式问答服务（复用框架 + 多格式接入管道）。

运行：
    cd enterprise-rag
    python -m venv .venv && .venv/Scripts/activate
    pip install -r requirements.txt
    cp .env.example .env   # 填入 DEEPSEEK_API_KEY
    .venv/Scripts/python.exe -m uvicorn kb.app:app --host 127.0.0.1 --port 8001

浏览器打开 http://127.0.0.1:8001 即可聊天（答案带引用来源）。

技术点：SSE 逐字流式、meta 首帧返回检索资料、CORS 放开。
语料来自 kb/corpus/（多格式、多部门），检索/生成由框架 HybridRAG 负责，
本文件只做「HTTP 封装 + 前端托管」。
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# 复用接入管道（它已负责 sys.path / load_dotenv / 多格式读取）
from kb.ingest import build_rag, ROOT

CORPUS_DIR = ROOT / "kb" / "corpus"
WEB_INDEX = ROOT / "web" / "index.html"  # 复用仓库根的前端页面

app = FastAPI(title="企业知识库 RAG 流式服务")

# 允许跨域，方便以后接独立前端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 启动时构建 RAG（全局单例，只做一次）----------
rag = build_rag(str(CORPUS_DIR))
_llm = rag.llm  # 复用全局 LLM 做流式生成


RAG_PROMPT = (
    "你是企业知识库助手。请只根据下面提供的资料回答问题。"
    "如果资料里没有答案，就明确说'资料中没有提到'，不要编造。"
    "回答时尽量引用资料中的要点。\n\n"
    "资料：\n{context}\n\n"
    "问题：{question}"
)


class ChatReq(BaseModel):
    query: str


@app.get("/health")
def health():
    """健康检查（前端会拉一次显示知识库大小）。"""
    return {
        "service": "企业知识库 RAG 流式服务",
        "rag_chunks": rag.stats().get("chunks"),
        "endpoints": {"POST /chat": "SSE 流式 RAG 问答，body: {\"query\": \"...\"}"},
    }


@app.get("/")
def index():
    """根路径直接返回前端聊天页面（同源，免 CORS）。"""
    return FileResponse(str(WEB_INDEX), media_type="text/html; charset=utf-8")


@app.post("/chat")
def chat(req: ChatReq):
    """SSE 流式 RAG 问答。

    流帧格式（每一帧都是 json.dumps 包裹，避免片段内含换行/引号破坏 SSE）：
      data: {"type":"meta","sources":[{"source":..,"text":..}]}   # 首帧：检索到的资料
      data: "<文本片段>"                                          # 后续：逐段文本
      data: [DONE]                                               # 结束
    """
    hits = rag.retrieve(req.query, top_k=3)
    meta = {
        "type": "meta",
        "sources": [
            {"source": h.get("source", ""), "text": h.get("text", "")}
            for h in hits
        ],
    }
    context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    prompt = RAG_PROMPT.format(context=context, question=req.query)

    def event_stream():
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        for chunk in _llm.stream_invoke([{"role": "user", "content": prompt}]):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
