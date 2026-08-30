# vnotes

> 把一条视频链接，变成一份结构清晰、可离线阅读的笔记。

粘贴 **B 站 / YouTube** 链接，自动完成：抓取元数据 → 下载音频 → 语音转文字 → AI 拆章整理 → 动态生成图解 → 抽取关键帧 → 渲染单页 HTML，并附带整页长图与竖屏切片。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](docker-compose.yml)

---

## 🖼 在线演示 · 笔记展示

点开下面的链接, 看用 vnotes 自动生成的**示例笔记画廊**（GitHub Pages 静态托管, 可直接分享给任何人）:

### [🚀 打开演示 · vnotes 笔记画廊](https://ofeocc.github.io/vnotes/)

> 里面是导入真实视频后自动生成的笔记: **AI 拆章(问题/陷阱/步骤/结论/关键引用) · 动态 SVG 图解 · 界面关键帧 · 时间戳跳转原视频 · 整页长图与竖屏切片**。
> 纯静态、可离线打开、无后端依赖, 适合把成果公开分享或自己长期存档回看。

---

## ✨ 它能帮你做什么

看教学/干货视频时，你想做笔记却不想反复暂停、截图、整理。vnotes 把整条流程自动化：

- **转写**：语音转文字，[5 种后端可选](#-转写后端)，中英文自动识别
- **拆章**：AI 按内容自然结构分章，提炼问题 / 陷阱 / 步骤 / 结论 / 关键引用
- **图解**：每章根据内容类型动态生成 SVG（流程图 / 时间线 / 对比表 / 因果链 / 风险决策…）
- **抽帧**：界面演示类视频自动抽取关键帧；细致模式逐章多帧
- **排版**：Apple 风格单页 HTML，离线可看，时间戳可直接跳回原视频

## 🚀 快速开始

需要 **Python 3.10+** 与 **ffmpeg**（完整版）。（前端页面无需额外构建。）

```bash
# 1) 克隆
git clone https://github.com/ofeocc/vnotes.git
cd vnotes

# 2) 装依赖
pip install -r requirements.txt
python -m playwright install chromium
pip install yt-dlp

# 3) 配置（填入 DeepSeek 等 API Key）
cp .env.example .env

# 4) 启动
python serve.py
# 浏览器打开 http://localhost:7458
```

> 也可以双击 `install.bat`（Windows 一键装环境）后双击 `start-web.bat` 启动。

### 📌 Docker（服务器/容器部署）

```bash
git clone https://github.com/ofeocc/vnotes.git
cd vnotes
cp .env.example .env
docker compose up -d --build
# 浏览器打开 http://localhost:7458
```

## 📖 用法

### 方式一：Web UI（推荐）

```bash
python serve.py
# http://localhost:7458
```

支持：粘贴链接一键生成、**实时进度推送**、转写后端在线切换、API Key 在线填写、批量模式、历史笔记浏览。

### 方式二：命令行

```bash
# 单个视频
python run.py "https://www.bilibili.com/video/BVxxxxxxxx"

# 指定分 P
python run.py "<URL>" --part 2

# 批量处理多 P 并生成聚合页
python run.py "<URL>" --batch

# 只处理指定 P
python run.py "<URL>" --batch --parts 1,3,5

# 检查环境 / 离线自检（无需网络和 API Key）
python run.py --check
python run.py --self-test
```

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `essence`（默认） | 脉络精华，4–12 章，单帧 | 快速回顾视频要点 |
| `detailed` | 细致笔记，8–20 章，多帧抽取 | 深度学习，需要关键截图 |

```bash
python run.py "<URL>" --mode detailed
```

## 🔁 处理流水线

```
视频链接
  ├─ 元数据 ── yt-dlp 抓取标题 / UP / 分 P / 封面 / 时长
  ├─ 音频 ──── 下载 + ffprobe 时长校验（偏差 >5% 自动重下）
  ├─ 转写 ──── 语音转文字 + 末段时长校验
  ├─ 分析 ──── AI 拆章：问题 / 陷阱 / 步骤 / 结论 / 引用
  ├─ SVG ───── 按内容类型动态生成图解
  ├─ 抽帧 ──── ffmpeg 精确抽帧，信息密度打分选最佳帧
  ├─ 渲染 ──── 单页 HTML（内联 CSS + SVG，离线可看）
  ├─ 截图 ──── Playwright 整页 PNG
  ├─ QA ────── 墨量分析检测空白带 / 截断
  └─ 切片 ──── 竖向切片（~1700px/片，100px 重叠）
```

## 🗣 转写后端

通过 `.env` 的 `VNOTES_TRANSCRIBE_BACKEND` 切换：

| 后端 | 占用空间 | 速度 | 成本 | 适合场景 |
|------|---------|------|------|---------|
| `faster-whisper` | ~200MB+ 模型 | 中 | 免费 | 本地机器（有 GPU 最快） |
| `vosk` | ~50MB | 快 | 免费 | 低内存 CPU 机器 |
| `paraformer` | 0 | 极快 | 按秒计费 | 阿里云服务器（国内低延迟） |
| `groq` | 0 | 极快 | 免费额度 | 网络可直连 Groq 的环境 |
| `openai-whisper` | ~3GB | 慢 | 免费 | 有 torch 的环境 |

> ⚠️ 部署到服务器时**不要默认假设 Groq 可用**（受出口、地区、风控影响，可能 403/超时）。更稳的选择是国内能直连的 `paraformer`（DashScope），或回到本地 `faster-whisper`。
> 自己用优先推荐 Windows 本地运行：B 站 Cookie、浏览器登录态、GPU/模型缓存都在本机，排错成本最低。
> 🚀 **GPU 加速**：本机有 NVIDIA GPU 时，`faster-whisper` 配 `VNOTES_WHISPER_DEVICE=cuda` 可达 **~20× 实时**（如 RTX 4060 上 300s 音频约 15s 完成）；需先 `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`，工具会自动把 CUDA 库挂到 CTranslate2 加载路径（`vnotes/transcribe.py` 的 `_ensure_gpu_libs()`），无需手动配 PATH。

## ⚙️ 配置

复制 `.env.example` 为 `.env` 并按需填写（每项都有注释）：

```env
# LLM（OpenAI 兼容，默认 DeepSeek，也支持通义/OpenAI）
VNOTES_LLM_API_KEY=sk-xxx
VNOTES_LLM_BASE_URL=https://api.deepseek.com/v1
VNOTES_LLM_MODEL=deepseek-chat

# 转写后端
VNOTES_TRANSCRIBE_BACKEND=faster-whisper

# 服务端口
VNOTES_SERVER_PORT=7458
```

| 服务 | base_url | model |
|------|----------|-------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |

> 🔐 **安全提示**：本项目为本机/内网使用设计，Web UI **无登录鉴权**且监听 `0.0.0.0`。请仅在可信局域网内使用；如需公网访问，请自行加反向代理+鉴权。API Key 只存于本地 `.env`（已被 `.gitignore` 排除），不会提交到仓库。

## 📁 输出结构

```
output/<视频标题>/
  notes.html        ← 单页笔记（离线打开，时间戳可跳回原视频）
  full.png          ← 整页长图
  slices/           ← 竖向切片（适合手机阅读）
  frames/           ← 关键帧截图（detailed 模式）
  cover.jpg         ← 视频封面
  notes_data.json   ← 结构化数据
```

## 🚀 把笔记发布到 GitHub Pages

生成的笔记是**纯静态文件**（单页 HTML + 图片），可直接公开分享。仓库自带 `build_pages.py`，一键把 `output/` 导出为 GitHub Pages 友好的站点（**画廊 + 单篇笔记 + 灯箱**）：

```bash
python build_pages.py          # 把 output/ 导出到 docs/
git add docs && git commit -m "update notes" && git push
```

然后在仓库 **Settings → Pages**，Source 选 **`Deploy from a branch`**，分支 `main`、目录 `/docs`，保存即可。站点地址形如：
`https://<用户名>.github.io/<仓库>/`

> 每次新增笔记后重跑 `python build_pages.py` 再推送，Pages 会自动更新。`docs/` 已在 `.gitignore` 用 `!docs/**` 放行图片提交，单文件 ≤100MB、站点 ≈1GB 内没问题。

## 📄 平台支持

| 平台 | 链接格式 | 时间戳跳转 |
|------|---------|-----------|
| B 站 | `bilibili.com/video/BVxxx` | `?p=N&t=秒` |
| B 站短链 | `b23.tv/xxx` | `?p=N&t=秒` |
| YouTube | `youtube.com/watch?v=xxx` | `?t=秒` |
| YouTube 短链 | `youtu.be/xxx` | `?t=秒` |

支持直接粘贴分享文本（含中文标题），自动提取 URL。

> 🌐 **YouTube 须知**：国内直连 YouTube 通常被墙，需在 `.env` 配置代理（`VNOTES_PROXY=http://127.0.0.1:7890`）供 yt-dlp 使用；并确保浏览器已登录 YouTube（`VNOTES_COOKIES_BROWSER`）。若仍报 `Sign in to confirm you're not a bot` / `No video formats found`，多为代理 IP 被标记或需 PO token，请换更稳的代理节点后重试。B 站不受此限制。

## 🧱 技术栈

- **Python**（FastAPI / uvicorn）+ 原生 JS 前端（单页、无构建）
- **yt-dlp**（元数据 / 下载）· **ffmpeg**（音频/抽帧）· **Playwright**（截图）
- **faster-whisper / vosk / paraformer / groq / openai-whisper**（转写）
- **DeepSeek / OpenAI 兼容**（AI 拆章与图解）

## 📁 目录结构

```
vnotes/
├── run.py                 CLI 入口
├── serve.py               Web 启动入口
├── config.py              配置管理
├── metadata.py            元数据抓取
├── audio.py               音频下载
├── transcribe.py          语音转写（5 种后端）
├── llm.py                 LLM 客户端
├── analyze.py             AI 内容分析
├── svg.py                 SVG 图解生成
├── frames.py              关键帧抽取
├── render.py              HTML 渲染
├── screenshot.py          整页截图
├── qa.py                  质量检查
├── crop.py                图片切片
├── batch.py               批量处理
├── lightbox.py            灯箱预览组件
├── util.py                工具函数
├── server.py              Web UI（FastAPI）
├── build_pages.py         静态导出为 GitHub Pages 站点（docs/）
└── download_model.py      模型下载脚本
```

## 📄 License

[MIT](LICENSE). 视频内容版权归原作者所有；生成内容仅供参考，请自行判断。
