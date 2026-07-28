# vnotes — 视频链接 → 离线单页笔记 HTML

把一条视频链接变成一份可离线打开的结构化笔记：自动抓元数据、下载音频、Whisper 转写、大模型拆章整理、动态生成 SVG 图解、抽真实帧截图，最后输出单页 HTML + 整页 PNG + 切片。

## 快速开始

```bash
# 1 安装依赖
pip install -r requirements.txt
python -m playwright install chromium

# 2 配置（复制 .env.example → .env，填入 DeepSeek API Key）
cp .env.example .env
# 编辑 .env，填入 VNOTES_LLM_API_KEY=sk-xxx

# 3 离线自检（验证 渲染→截图→QA→切片 管线，无需网络/Whisper/LLM）
python run.py --self-test

# 4 检查环境
python run.py --check

# 5 正式运行
python run.py "https://www.bilibili.com/video/BVxxxxxxxx"
python run.py "https://www.bilibili.com/video/BVxxxxxxxx" --part 2
```

## 流水线

```
视频链接
  │
  ├─ 1. 元数据 ── yt-dlp + cookies → 标题/UP/分P/简介/标签/封面/时长/章节
  ├─ 2. 音频 ──── 下载最佳音频 → ffprobe 校验时长（偏差>5% 重下）
  ├─ 3. 转写 ──── Whisper（中文 turbo / 英文 small.en+翻译）→ 末段时长校验
  ├─ 4. 分析 ──── DeepSeek 按自然结构拆章 → 问题/陷阱/步骤/结论/引用/锚点
  ├─ 5. SVG ───── 按内容类型动态生成（流程/概念/时间线/对比/风险/数据/因果）
  ├─ 6. 抽帧 ──── 界面演示类视频：ffmpeg 精确抽帧，信息密度打分选最佳帧
  ├─ 7. 渲染 ──── 单页 HTML（内联 CSS+SVG，相对路径图片，离线可看）
  ├─ 8. 截图 ──── Playwright 整页 PNG
  ├─ 9. QA ────── 逐行墨量分析 → 检测空白带/截断/渲染异常
  └─ 10. 切片 ─── PIL crop 竖向切片（~1700px/片，100px 重叠，空白处下刀）
```

## 命令参数

```
python run.py <URL> [选项]

  URL              视频链接（B站/YouTube 等）
  --part N         分 P 号（B站）
  --no-frames      跳过真实帧抽取
  --no-slice       跳过切片
  --stub-transcript 用桩转写（不调 Whisper，用于离线测试渲染管线）
  --self-test      离线自检图像管线
  --check          仅检查环境与依赖
  --batch          批量处理多 P 视频并生成 index.html 聚合页
  --parts 1,3,5    批量模式下指定 P 号（逗号分隔，默认全部）
```

## Web UI

```bash
python serve.py
# 浏览器打开 http://localhost:7458
```

Web UI 支持：
- 粘贴链接一键生成笔记
- 实时进度（SSE 推送 + 流水线阶段可视化）
- 转写后端在线切换（faster-whisper / 阿里云 Paraformer / Groq / openai-whisper）
- DeepSeek / Groq / DashScope API Key 在线填写（留空则用 .env 默认值）
- 批量模式开关（多 P 视频逐 P 生成 + 聚合页）
- 历史笔记浏览

## 批量模式（多 P 视频）

```bash
# 批量处理所有 P
python run.py "https://www.bilibili.com/video/BVxxxxxxxx" --batch

# 只处理指定 P
python run.py "https://www.bilibili.com/video/BVxxxxxxxx" --batch --parts 1,3,5

# 或在 Web UI 中勾选「批量模式」
```

批量模式会：
1. 自动检测视频的分 P 列表
2. 逐 P 生成笔记（每个 P 独立输出目录 P01_xxx/、P02_xxx/…）
3. 生成 `index.html` 聚合页，汇总所有 P 的标题、章节数、时长、链接
4. 支持断点续跑：已完成的 P 自动跳过

## 配置（.env）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VNOTES_LLM_BASE_URL` | `https://api.deepseek.com/v1` | LLM API 地址（OpenAI 兼容） |
| `VNOTES_LLM_API_KEY` | — | API Key（必填） |
| `VNOTES_LLM_MODEL` | `deepseek-chat` | 模型名 |
| `VNOTES_COOKIES_BROWSER` | `chrome` | cookies 浏览器（chrome/edge/firefox/brave） |
| `VNOTES_COOKIES_FILE` | — | Netscape cookies.txt 路径（二选一） |
| `VNOTES_WHISPER_DEVICE` | `cuda` | Whisper 设备（cuda/cpu） |
| `VNOTES_WHISPER_MODEL_ZH` | `turbo` | 中文模型 |
| `VNOTES_WHISPER_MODEL_EN` | `small.en` | 英文模型 |
| `VNOTES_OUTPUT_DIR` | `./output` | 输出目录 |
| `VNOTES_CACHE_DIR` | `./cache` | 缓存目录 |

## 依赖说明

### 核心（必装）
- **Python 3.10+**
- `requests` — HTTP 请求
- `Pillow` — 图像处理（QA/切片/帧打分）
- `playwright` — 整页截图（自带独立 Chromium，不冲突系统浏览器）
  ```bash
  pip install playwright
  python -m playwright install chromium
  ```
- `yt-dlp` — 视频元数据与音频下载
  ```bash
  pip install yt-dlp
  ```

### 转写（三选一，用 VNOTES_TRANSCRIBE_BACKEND 切换）

**方案 1：faster-whisper（推荐，轻量）** — 不装 torch，包仅 ~200MB，模型下载到 D 盘
```bash
pip install faster-whisper
# .env 中设置 VNOTES_TRANSCRIBE_BACKEND=faster-whisper（默认）
# 模型自动下载到 D:/vnotes_models（可用 VNOTES_WHISPER_MODEL_DIR 改路径）
```

**方案 2：Groq 云端 API（零空间）** — 极快，完全不占本地空间
```bash
pip install openai
# 注册 https://console.groq.com 拿免费 API Key
# .env 中设置：
#   VNOTES_TRANSCRIBE_BACKEND=groq
#   VNOTES_GROQ_API_KEY=gsk-你的key
```

**方案 3：openai-whisper（原方案，占空间大）** — 需 torch (~2.5GB)
```bash
pip install openai-whisper
# GPU 用户先装 CUDA 版 torch
# .env 中设置 VNOTES_TRANSCRIBE_BACKEND=openai-whisper
```

| 方案 | 占用空间 | 速度 | 成本 | 隐私 |
|------|---------|------|------|------|
| faster-whisper | ~200MB(包)+1.5GB(模型→D盘) | 中等 | 免费 | 本地 |
| Groq API | 0 | 极快 | 免费额度 | 上传音频 |
| openai-whisper | ~3GB+ | 慢 | 免费 | 本地 |

### ffmpeg（重要）
工具需要 ffmpeg 做音频转换和帧抽取。**TRAE 自带的 ffmpeg 是精简版**，不支持音频编码（mp3）和图像编解码（PNG/JPEG）。

- **音频下载**：自动适配 — mp3 转换失败时保留原始格式（m4a/webm）
- **帧抽取**：需要完整 ffmpeg — [下载地址](https://www.gyan.dev/ffmpeg/builds/)，解压后设置环境变量
- **Whisper 转写**：Whisper 内部用 ffmpeg 解码音频，也需要完整版
- **截图/切片**：不依赖 ffmpeg（用 Playwright + PIL）

### LLM
默认使用 DeepSeek（`deepseek-chat`）。也兼容任何 OpenAI API 格式的服务：
- 通义千问：`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1` `model=qwen-plus`
- OpenAI：`base_url=https://api.openai.com/v1` `model=gpt-4o-mini`

## 输出结构

```
output/
  视频标题/
    notes.html        ← 单页笔记（离线可打开）
    full.png          ← 整页截图
    slices/
      slice_01.png    ← 竖向切片（~1700px/片，100px重叠）
      slice_02.png
      ...
    frames/           ← 真实帧截图（仅界面演示类视频）
      01_章节名_000030.jpg
      ...
    cover.jpg         ← 视频封面
    notes_data.json   ← 完整数据（元数据+分析+QA结果）
```

## SVG 动态生成

每章根据内容类型自动选择图解形式：

| 类型 | 适用场景 | 图形 |
|------|---------|------|
| flow | 流程/步骤/路径 | 方框+箭头串联 |
| concept | 概念关系/分层 | 分层卡片/节点连线 |
| timeline | 时间变化 | 横向时间轴 |
| comparison | 对比 | 矩阵表格 |
| risk | 风险/误区 | 检查表/决策树 |
| data | 数据 | 条形/折线简图 |
| causation | 因果 | 箭头链路 |

每张图由 LLM 根据本章真实内容生成，包含关键词、关系、箭头或标签，禁止装饰图和重复套壳。

## 自检

```bash
python run.py --self-test
```

离线验证完整图像管线（无需网络/Whisper/LLM）：
1. 生成合成数据（2 章 + 内联 SVG + 占位截图）
2. 渲染 HTML
3. Playwright 整页截图
4. QA 墨量检查
5. PIL 切片

输出到 `output/_self_test/`。

## Docker 部署

```bash
# 1 准备配置
cp .env.example .env
# 编辑 .env，填入 VNOTES_LLM_API_KEY

# 2 构建并启动
docker compose up -d --build

# 3 访问
# http://localhost:7458

# 查看日志 / 停止
docker compose logs -f
docker compose down
```

Docker 镜像内置完整 ffmpeg + Playwright Chromium + 所有 Python 依赖。
`output/` 和 `cache/` 通过 Volume 挂载持久化，`.env` 只读挂载。

> GPU 转写：取消 `docker-compose.yml` 中 `deploy.resources` 和 `VNOTES_WHISPER_DEVICE=cuda` 的注释，需安装 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)。

## 项目结构

```
vnotes/
  config.py     — 配置（.env + 工具探测）
  util.py       — 子进程/日志/时间/文件名工具
  metadata.py   — yt-dlp 元数据抓取
  audio.py      — 音频下载 + 时长校验（自适应 ffmpeg）
  transcribe.py — Whisper 转写（faster-whisper / Vosk / Paraformer / Groq）
  llm.py        — LLM 客户端（OpenAI 兼容）
  analyze.py    — 内容分析 + 拆章
  svg.py        — 动态 SVG 生成
  frames.py     — 真实帧抽取
  render.py     — HTML 渲染
  screenshot.py — 整页截图（Playwright 优先）
  qa.py         — QA 检查（墨量/空白/截断）
  crop.py       — 切片（PIL crop）
  batch.py      — 批量多 P 处理 + 聚合页
  lightbox.py   — 笔记灯箱注入
  server.py     — FastAPI Web UI 后端 + 嵌入式前端
run.py          — CLI 入口 + 流水线编排
serve.py        — Web UI 启动入口
Dockerfile      — 容器化部署
docker-compose.yml — 一键编排
```
