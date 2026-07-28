@echo off
chcp 65001 >nul
title vnotes 安装脚本

echo ========================================
echo   vnotes 视频笔记工具 - 一键安装
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查版本（>= 3.10）
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
set /a PYNUM=%PYMAJOR%*100+%PYMINOR%
if %PYNUM% LSS 310 (
    echo [错误] Python 版本过低：%PYVER%，需要 3.10 或更高版本
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [1/6] Python 版本：%PYVER%（符合要求 >= 3.10）

REM 创建虚拟环境
if not exist venv (
    echo [2/6] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
) else (
    echo [2/6] 虚拟环境已存在
)

REM 激活虚拟环境
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 虚拟环境激活失败
    pause
    exit /b 1
)

REM 升级 pip
echo [3/6] 升级 pip 并安装 Python 依赖...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [警告] pip 升级失败，继续使用当前版本
)

REM 安装依赖
pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络或手动安装
    echo 可尝试使用国内镜像：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
)

REM Playwright Chromium
echo [4/6] 安装 Playwright Chromium...
python -m playwright install chromium
if errorlevel 1 (
    echo [警告] Playwright Chromium 安装失败，截图功能可能不可用
    echo 可稍后手动执行：python -m playwright install chromium
) else (
    echo Playwright Chromium 安装完成
)

REM .env 配置文件
echo [5/6] 配置文件...
if not exist .env (
    copy .env.example .env >nul
    echo 已从 .env.example 创建 .env
    echo 请编辑 .env 填入你的 API Key
) else (
    echo .env 已存在
)

REM 完成
echo.
echo [6/6] 安装完成！
echo ========================================
echo  下一步：
echo  1. 编辑 .env 填入 DeepSeek API Key
echo  2. 双击 start-web.bat 启动 Web UI
echo  3. 如需排查环境，再运行 venv\Scripts\python.exe run.py --check
echo ========================================
echo.
pause
