# 发布前检查清单

## 上传前必须确认

- `.env` 不要提交，只提交 `.env.example`。
- `cache/`、`output/`、`venv/`、`*.log` 不要提交。
- `VNOTES_SERVER_PORT` 默认保持 `7458`。
- `python run.py --check` 能通过。
- `python run.py --self-test` 能生成离线自检页面。
- `python serve.py` 后能打开 `http://127.0.0.1:7458`。
- `LICENSE` 文件存在（MIT）。

## 推荐提交文件

- `vnotes/`                  — 核心包
- `run.py`                   — CLI 入口
- `serve.py`                 — Web UI 启动入口
- `download_model.py`        — 模型下载工具
- `requirements.txt`         — Python 依赖
- `README.md`                — 项目文档
- `QUICKSTART.md`            — 小白快速开始
- `RELEASE_CHECKLIST.md`     — 本文件
- `install.bat`              — Windows 一键安装
- `start-web.bat`            — Windows 一键启动
- `.env.example`             — 环境变量模板
- `.gitignore`               — Git 忽略规则
- `LICENSE`                  — MIT 开源协议
- `Dockerfile`               — 容器化部署
- `docker-compose.yml`       — Docker 编排
- `.dockerignore`            — Docker 构建排除

## 可选提交文件

- `diag_svg.py`              — SVG 调试工具（开发用）
- `run_cached.py`            — 缓存调试工具（开发用）

## 发布说明建议

- B 站链路已验证。
- YouTube 代码路径存在，但需要在可访问 YouTube 的网络或代理环境下完整验收。
- DeepSeek API Key 必填。
- Groq / Paraformer 是可选云端转写后端，需要用户自行填写对应 Key。
- Docker 部署：`docker compose up -d --build`，访问 `http://localhost:7458`。
- Windows 桌面：双击 `install.bat` 安装，双击 `start-web.bat` 启动。
