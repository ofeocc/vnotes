# vnotes 小白快速开始

## 1. 安装

双击 `install.bat`，等待依赖安装完成。

如果安装失败，先确认：
- 已安装 Python 3.10 或更高版本。
- 网络能访问 Python 包源。
- 已安装完整 ffmpeg，并能在命令行运行 `ffmpeg -version`。

## 2. 配置 API Key

复制 `.env.example` 为 `.env`，然后填写：

```env
VNOTES_LLM_API_KEY=你的 DeepSeek API Key
VNOTES_SERVER_PORT=7458
```

转写后端默认是 `faster-whisper`。如果本地模型下载慢，可以先在 Web UI 里选择 `Vosk` 或云端转写后端。

## 3. 启动 Web UI

双击 `start-web.bat`，浏览器会打开：

```text
http://127.0.0.1:7458
```

如果提示没有 `venv`，说明还没有跑安装脚本，先双击 `install.bat`。

把 B 站视频链接粘进去，点击“生成笔记”。

## 4. 常见问题

- 端口冲突：修改 `.env` 里的 `VNOTES_SERVER_PORT`。
- B 站 Cookie 读取失败：工具会自动无 Cookie 重试；需要登录权限的视频请导出 `cookies.txt` 并配置 `VNOTES_COOKIES_FILE`。
- YouTube：代码路径支持 YouTube，但当前网络必须能稳定访问 YouTube；国内环境通常需要代理后再完整验收。
- 生成慢：本地转写速度取决于 CPU/GPU；想更快可以选 Groq、Paraformer 等云端后端，并填写对应 API Key。

## 5. 输出在哪里

生成结果在 `output/视频标题/`：

- `notes.html`：可离线打开的笔记页
- `full.png`：整页长图
- `cover.jpg`：封面
- `notes_data.json`：结构化数据
