# 企业级 RAG 知识库 - 容器化部署
# 镜像基于官方 Python 3.12 精简镜像，不打包任何密钥，适合私有化部署。
FROM python:3.12-slim

# 关闭交互提示、禁止写 pyc、pip 不缓存——让镜像更小、构建更干净
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 非密钥的默认配置（写在镜像里没问题）：
#   DEEPSEEK_BASE_URL  —— DeepSeek 官方接口地址（构造 LLM 必填）
#   RAG_EMBEDDING=tfidf —— 默认纯字面检索兜底，不依赖任何嵌入服务，
#                          容器在任意环境都能直接跑起来（只匹配字面，不理解语义）
ENV DEEPSEEK_BASE_URL=https://api.deepseek.com/v1 \
    RAG_EMBEDDING=tfidf

WORKDIR /app

# 先单独拷依赖清单并安装，利用 Docker 层缓存（改代码不会重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制全部代码与语料（.env / .venv 等已在 .dockerignore 排除）
COPY . .

# 服务端口（uvicorn 在容器内必须监听 0.0.0.0 才能被宿主机访问）
EXPOSE 8001

# 健康检查：轮询 /health，容器“起得来”才会变 healthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health').status==200 else 1)"

# 启动命令。注意：必须运行时传入 DEEPSEEK_API_KEY，否则应用在构造 LLM 时会报错退出。
CMD ["sh", "-c", "uvicorn kb.app:app --host 0.0.0.0 --port 8001"]
