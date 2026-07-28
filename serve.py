#!/usr/bin/env python
"""vnotes Web UI 启动入口。"""
import sys
from pathlib import Path

# 确保能 import vnotes
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vnotes.config import Config

def main():
    cfg = Config.load()
    port = cfg.server_port

    print(f"vnotes Web UI 启动中... http://localhost:{port}")

    import uvicorn
    uvicorn.run(
        "vnotes.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
