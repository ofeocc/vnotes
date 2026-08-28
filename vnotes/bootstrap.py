"""启动守卫：本地项目优先使用自带 venv，避免跑到系统 Python。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_GUARD_FLAG = "VNOTES_VENV_GUARD_DONE"
_SKIP_FLAG = "VNOTES_SKIP_VENV_GUARD"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def project_venv_python(root: Path | None = None) -> Path:
    root = root or project_root()
    if os.name == "nt":
        return root / "venv" / "Scripts" / "python.exe"
    return root / "venv" / "bin" / "python"


def _same_path(a: str | Path, b: str | Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return os.path.normcase(os.fspath(a)) == os.path.normcase(os.fspath(b))


def in_project_venv(root: Path | None = None) -> bool:
    venv_python = project_venv_python(root)
    return venv_python.exists() and _same_path(sys.executable, venv_python)


def runtime_status(root: Path | None = None) -> dict[str, str | bool]:
    root = root or project_root()
    venv_python = project_venv_python(root)
    inside = in_project_venv(root)
    return {
        "python_env": "project-venv" if inside else "system-python",
        "in_project_venv": inside,
        "project_venv_exists": venv_python.exists(),
        "python_executable": sys.executable,
    }


def ensure_project_venv(*, module: str | None = None, script: str | Path | None = None) -> None:
    """如果项目 venv 存在而当前没用它启动，原地重启到 venv。

    `module` 用于 `python -m vnotes.server` 这种入口；`script` 用于 `python serve.py`。
    Docker/服务器镜像通常没有项目 venv，因此不会触发。
    """
    if os.environ.get(_SKIP_FLAG) or os.environ.get(_GUARD_FLAG):
        return

    root = project_root()
    venv_python = project_venv_python(root)
    if not venv_python.exists() or in_project_venv(root):
        return

    env = os.environ
    env[_GUARD_FLAG] = "1"
    if module:
        args = [str(venv_python), "-m", module, *sys.argv[1:]]
    else:
        target = Path(script or sys.argv[0]).resolve()
        args = [str(venv_python), str(target), *sys.argv[1:]]

    print(f"[vnotes] 检测到项目 venv，已自动切换：{venv_python}", flush=True)
    raise SystemExit(subprocess.call(args, env=env))
