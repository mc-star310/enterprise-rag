# Enterprise-RAG · 企业级知识库问答系统

从零搭建的**企业级 RAG 知识库系统**：支持多格式文档接入、混合检索（字符级 TF-IDF + 语义 + RRF 融合重排）、带引用的流式问答，可私有化部署。

> 定位：一个**可接入任意企业文档**的知识库平台——把内部手册 / 制度 / 帮助台 / 合规文档丢进去，就能用自然语言提问并拿到带出处的答案。

---

## 核心能力

| 模块 | 说明 |
|---|---|
| **多格式接入** | PDF / Word / Markdown / TXT → 统一清洗 → 智能分块（`kb/ingest.py`） |
| **混合检索 + 融合重排** | 字符级 TF-IDF（中文鲁棒）+ 语义嵌入（Ollama bge-m3）+ RRF 融合；Ollama 未开启时自动降级为 TF-IDF |
| **带引用问答** | 答案标注出处（哪份文档 / 第几段），严格基于检索资料，拒绝编造 |
| **流式 Web 服务** | FastAPI + SSE 逐字流式输出，前端首帧即显示检索来源 |
| **可量化评测** | 仿 252 题方法论，可跑 Recall@k / MRR / 忠实度（见 `kb/eval/`，规划中） |

---

## 架构

```
用户问题
   │
   ▼
┌─────────────────────────────────────────┐
│  kb/app.py  (FastAPI + SSE 流式服务)      │
│   - /chat  SSE 流式问答（首帧返回引用）    │
│   - /     托管前端聊天页                   │
│   - /health 健康检查                       │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│  kb/ingest.py  (多格式接入管道)           │
│   PDF/Word/MD/TXT → 纯文本 → 分块          │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│  hello_agents/rag/hybrid_rag.py          │
│   混合索引（TF-IDF + 语义）+ 融合检索       │
│   + 生成（DeepSeek，stream_invoke）        │
└─────────────────────────────────────────┘
```

`hello_agents/` 是从零手搓的 Agent / RAG 框架（未使用 LangChain），本系统直接将其作为引擎复用。

---

## 快速开始

```bash
# 1. 准备环境
cd enterprise-rag
python -m venv .venv && .venv/Scripts/activate        # Windows
pip install -r requirements.txt

# 2. 配置密钥
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY（https://platform.deepseek.com）

# 3.（可选）开启语义检索
# 安装并启动 Ollama，拉取嵌入模型：  ollama pull bge-m3
# 否则系统自动降级为纯 TF-IDF，仍可正常运行

# 4. 启动服务
.venv/Scripts/python.exe -m uvicorn kb.app:app --host 127.0.0.1 --port 8001

# 5. 浏览器打开 http://127.0.0.1:8001 即可聊天
```

### 离线验证（不调 LLM，只验检索）

```bash
.venv/Scripts/python.exe -m kb.smoke_test
```

会打印索引统计，并对 HR / IT / 合规 三类查询验证 top 检索命中。

---

## 目录结构

```
enterprise-rag/
├── hello_agents/          # 从零手搓的 Agent/RAG 框架（引擎）
│   ├── core/              # LLM 封装、消息、Agent 基类
│   ├── rag/               # HybridRAG（混合检索）、向量库、嵌入
│   ├── agents/            # Simple / ReAct / Reflection / PlanSolve Agent
│   └── tools/             # 工具注册与内置工具
├── kb/                    # 本知识库应用
│   ├── ingest.py          # 多格式接入管道
│   ├── app.py             # FastAPI + SSE 服务
│   ├── smoke_test.py      # 离线检索验证
│   └── corpus/            # 示例语料（HR / IT / 合规，自写 demo）
├── web/
│   └── index.html         # 单文件前端聊天页（无外部 CDN 依赖）
├── requirements.txt
├── .env.example
└── README.md
```

---

## 示例语料说明

`kb/corpus/` 下当前为**自写的演示语料**（虚构公司「示例科技」的内部制度摘要），用于端到端验证管线：
- `hr_leave_policy.md` — 假期与请假制度
- `it_helpdesk.md` — IT 服务台常见故障处理
- `compliance_privacy_basics.md` — 个人信息保护员工指引

系统的价值在于**接入能力**而非示例内容——可替换为任意真实企业文档（产品手册、HR 制度、IT 帮助台、合规/法律文本等）。后续规划接入真实可验证语料（公开法律全文 + 开源产品文档）以演示跨文档问答。

---

## 简历一句话

> 从零搭建企业级 RAG 知识库系统：支持 PDF/Word/Markdown 多格式文档接入与混合检索（字符级 TF-IDF + 语义嵌入 + RRF 融合重排），答案带引用出处，FastAPI + SSE 流式问答，可私有化部署。

## 差异化亮点（对比 LangChain/LangGraph 方案）

- **工程深度**：检索与融合重排从零实现，而非黑盒调用框架；
- **评测严谨**：可量化评测（Recall@k / MRR / 忠实度），而非「感觉能答」；
- **可私有化**：无外部 SaaS 依赖，文档不出本地即可部署。
