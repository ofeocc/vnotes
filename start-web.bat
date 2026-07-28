@echo off
chcp 65001 >nul
cd /d "%~dp0"
title vnotes Web UI

if not exist "venv\Scripts\python.exe" (
  echo 未检测到项目虚拟环境 venv。
  echo 请先双击 install.bat 完成依赖安装，然后再运行 start-web.bat。
  pause
  exit /b 1
)

set "PY=%~dp0venv\Scripts\python.exe"

"%PY%" -c "import fastapi, uvicorn, requests, PIL" >nul 2>&1
if errorlevel 1 (
  echo Python Web 依赖不完整，请重新运行 install.bat。
  pause
  exit /b 1
)

if not exist ".env" (
  if exist ".env.example" (
    copy ".env.example" ".env" >nul
    echo 已创建 .env，请先填入 DeepSeek API Key。
    echo 文件位置：%cd%\.env
    pause
  )
)

echo 正在启动 vnotes Web UI...
echo 地址：http://127.0.0.1:7458
start "" "http://127.0.0.1:7458"
"%PY%" serve.py
pause
