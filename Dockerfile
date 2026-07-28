# vnotes Dockerfile — 视频笔记工具
# 多阶段构建：base + deps → runtime
#
# 构建：  docker build -t vnotes .
# 运行：  docker run -d --name vnotes -p 7458:7458 \
#           -v $(pwd)/output:/app/output \
#           -v $(pwd)/cache:/app/cache \
#           -v $(pwd)/.env:/app/.env:ro \
#           vnotes
# 进容器： docker exec -it vnotes bash

# ============================================================
#  Stage 1: base — 系统依赖 + Python
# ============================================================
FROM python:3.12-slim AS base

# 系统依赖：ffmpeg（完整版，支持音频编码 + 帧抽取）、yt-dlp 系统级
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp 全局安装（最新版）
RUN pip install --no-cache-dir yt-dlp

WORKDIR /app

# ============================================================
#  Stage 2: deps — Python 依赖缓存层
# ============================================================
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright Chromium（截图必需）
RUN python -m playwright install chromium --with-deps

# ============================================================
#  Stage 3: runtime — 最终镜像
# ============================================================
FROM deps AS runtime

# 复制项目代码
COPY vnotes/        ./vnotes/
COPY run.py         ./
COPY serve.py       ./
COPY download_model.py ./

# 默认配置（可被 .env 覆盖）
ENV VNOTES_SERVER_PORT=7458 \
    VNOTES_TRANSCRIBE_BACKEND=faster-whisper \
    VNOTES_WHISPER_DEVICE=cpu \
    VNOTES_WHISPER_MODEL_ZH=small \
    VNOTES_WHISPER_MODEL_EN=small.en \
    VNOTES_HF_ENDPOINT=https://hf-mirror.com \
    VNOTES_OUTPUT_DIR=/app/output \
    VNOTES_CACHE_DIR=/app/cache

# 创建输出 / 缓存目录（挂载点）
RUN mkdir -p /app/output /app/cache
VOLUME ["/app/output", "/app/cache"]

EXPOSE 7458

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:7458/api/config-status || exit 1

# 启动 Web UI
CMD ["python", "serve.py"]
