# vnotes

把一条视频链接变成一份结构化笔记。

粘贴 B 站 / YouTube 链接，自动抓取元数据、下载音频、语音转文字、AI 拆章整理、生成 SVG 图解、抽取关键帧，最终输出一份可离线打开的单页 HTML 笔记——附带整页长图和竖向切片。

## 它解决什么问题

看教学视频时，你想做笔记但不想手动暂停、截图、整理。vnotes 把整个过程自动化：

- **转写**：语音转文字，中文用 turbo 模型，英文自动翻译
- **拆章**：AI 按内容自然结构分章，提取问题、陷阱、步骤、结论、引用
- **图解**：每章根据内容类型动态生成 SVG（流程图 / 时间线 / 对比表 / 因果链等）
- **抽帧**：界面演示类视频自动抽取关键帧截图
- **排版**：输出 Apple 风格单页 HTML，离线可看，附带时间戳跳转链接

## 快速开始

### 方式一：Docker（推荐服务器部署）

```bash
git clone https://gitee.com/heng-zhenghao/vnotes.git
cd vnotes
cp .env.example .env
# 编辑 .env，填入 VNOTES_LLM_API_KEY=sk-xxx
docker compose up -d --build
# 浏览器打开 http://localhost:7458
```

### 方式二：Windows 桌面

```bash
# 双击 install.bat  安装依赖
# 双击 start-web.bat 启动
# 浏览器打开 http://127.0.0.1:7458
```

### 方式三：手动安装

```bash
pip install -r requirements.txt
python -m playwright install chromium
pip install yt-dlp

cp .env.example .env
# 编辑 .env，填入 VNOTES_LLM_API_KEY=sk-xxx

python run.py --check       # 检查环境
python run.py --self-test   # 离线自检（无需网络/API Key）
python serve.py             # 启动 Web UI
```

## 使用方式

### Web UI

```bash
python serve.py
# http://localhost:7458
```

支持：粘贴链接一键生成、实时进度推送、转写后端在线切换、API Key 在线填写、批量模式、历史笔记浏览。

### 命令行

```bash
# 单个视频
python run.py "https://www.bilibili.com/video/BVxxxxxxxx"

# 指定分 P
python run.py "https://www.bilibili.com/video/BVxxxxxxxx" --part 2

# 批量处理多 P 视频并生成聚合页
python run.py "https://www.bilibili.com/video/BVxxxxxxxx" --batch

# 只处理指定 P
python run.py "https://www.bilibili.com/video/BVxxxxxxxx" --batch --parts 1,3,5
```

### 笔记模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `essence`（默认） | 脉络精华，4-12 章，单帧 | 快速回顾视频要点 |
| `detailed` | 细致笔记，8-20 章，多帧抽取 | 深度学习，需要关键截图 |

```bash
python run.py "URL" --mode detailed
# 或在 Web UI 中切换
```

## 处理流程

```
视频链接
  ├─ 元数据 ── yt-dlp 抓取标题/UP/分P/封面/时长
  ├─ 音频 ──── 下载 + ffprobe 时长校验
  ├─ 转写 ──── 语音转文字 + 末段时长校验
  ├─ 分析 ──── AI 拆章：问题/陷阱/步骤/结论/引用
  ├─ SVG ───── 按内容类型动态生成图解
  ├─ 抽帧 ──── ffmpeg 精确抽帧，信息密度打分选最佳帧
  ├─ 渲染 ──── 单页 HTML（内联 CSS+SVG，离线可看）
  ├─ 截图 ──── Playwright 整页 PNG
  ├─ QA ────── 墨量分析检测空白带/截断
  └─ 切片 ──── 竖向切片（~1700px/片，100px 重叠）
```

## 转写后端

通过 `VNOTES_TRANSCRIBE_BACKEND` 切换，适配不同环境：

| 后端 | 占用空间 | 速度 | 成本 | 适合场景 |
|------|---------|------|------|---------|
| `faster-whisper` | ~200MB+模型 | 中等 | 免费 | 本地 GPU 机器 |
| `vosk` | ~50MB | 快 | 免费 | 低内存 CPU 机器 |
| `paraformer` | 0 | 极快 | 按秒计费 | 阿里云服务器（同区低延迟） |
| `groq` | 0 | 极快 | 免费额度 | 海外服务器 |
| `openai-whisper` | ~3GB | 慢 | 免费 | 有 torch 环境的机器 |

服务器部署建议用 `paraformer`（阿里云同区，速度快），本地机器用 `faster-whisper`（有 GPU 时最快）。

## 配置

复制 `.env.example` 为 `.env`，按需填写：

```env
# LLM（必填，默认 DeepSeek）
VNOTES_LLM_API_KEY=sk-xxx
# 也兼容通义千问/OpenAI，改 base_url 和 model 即可

# 转写后端
VNOTES_TRANSCRIBE_BACKEND=faster-whisper

# 服务端口
VNOTES_SERVER_PORT=7458
```

完整配置项见 `.env.example`，每项都有注释说明。

### LLM 选择

默认 DeepSeek（便宜好用）。也兼容任何 OpenAI API 格式：

| 服务 | base_url | model |
|------|----------|-------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |

## 输出结构

```
output/视频标题/
  notes.html        ← 单页笔记（离线打开，时间戳可跳转原视频）
  full.png          ← 整页长图
  slices/           ← 竖向切片（适合手机阅读）
  frames/           ← 关键帧截图（detailed 模式）
  cover.jpg         ← 视频封面
  notes_data.json   ← 结构化数据
```

## 平台支持

| 平台 | 链接格式 | 时间戳跳转 |
|------|---------|-----------|
| B 站 | `bilibili.com/video/BVxxx` | `?p=N&t=秒` |
| B 站短链 | `b23.tv/xxx` | `?p=N&t=秒` |
| YouTube | `youtube.com/watch?v=xxx` | `?t=秒` |
| YouTube 短链 | `youtu.be/xxx` | `?t=秒` |

支持直接粘贴分享文本（含中文标题），自动提取 URL。

## SVG 图解类型

每章由 AI 根据内容自动选择最合适的图解形式：

| 类型 | 适用场景 | 图形 |
|------|---------|------|
| flow | 流程/步骤 | 方框+箭头串联 |
| concept | 概念关系 | 分层卡片/节点连线 |
| timeline | 时间变化 | 横向时间轴 |
| comparison | 对比 | 矩阵表格 |
| risk | 风险/误区 | 检查表/决策树 |
| data | 数据 | 条形/折线简图 |
| causation | 因果 | 箭头链路 |

## 依赖

- **Python 3.10+**
- **ffmpeg**（完整版，音频解码+帧抽取）
- **yt-dlp**（视频下载）
- **Playwright Chromium**（整页截图）

Docker 镜像已内置上述全部依赖。

## 项目结构

```
vnotes/
  config.py       配置管理
  metadata.py     元数据抓取
  audio.py        音频下载
  transcribe.py   语音转写（5 种后端）
  llm.py          LLM 客户端
  analyze.py      AI 内容分析
  svg.py          SVG 图解生成
  frames.py       关键帧抽取
  render.py       HTML 渲染
  screenshot.py   整页截图
  qa.py           质量检查
  crop.py         图片切片
  batch.py        批量处理
  server.py       Web UI
run.py            CLI 入口
serve.py          Web 启动
```

## License

MIT
