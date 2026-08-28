"""通用工具：子进程执行、时间格式化、文件名清洗、日志。"""
from __future__ import annotations

import re
import sys
import json
import subprocess
from pathlib import Path
from typing import Any


# ---- 日志（带 [阶段] 前缀，stderr）----
class Logger:
    def __init__(self) -> None:
        self.verbose = True

    def _emit(self, lvl: str, tag: str, msg: str) -> None:
        if not self.verbose and lvl == "DEBUG":
            return
        print(f"[{lvl}] {tag} {msg}", file=sys.stderr, flush=True)

    def info(self, tag: str, msg: str) -> None:
        self._emit("INFO ", tag, msg)

    def warn(self, tag: str, msg: str) -> None:
        self._emit("WARN ", tag, msg)

    def error(self, tag: str, msg: str) -> None:
        self._emit("ERROR", tag, msg)

    def debug(self, tag: str, msg: str) -> None:
        self._emit("DEBUG", tag, msg)


log = Logger()


# ---- 子进程 ----
def run(cmd: list[str], *, tag: str = "run", check: bool = True,
        capture: bool = True, text: bool = True, timeout: int | None = None,
        input_text: str | None = None) -> subprocess.CompletedProcess:
    """统一执行子进程，失败时抛出带 stderr 的异常。"""
    log.debug(tag, "$ " + " ".join(_quote(c) for c in cmd))
    try:
        cp = subprocess.run(
            cmd,
            capture_output=capture,
            text=text,
            timeout=timeout,
            input=input_text,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        raise RuntimeError(f"命令不存在：{cmd[0]}")
    except subprocess.TimeoutExpired as e:
        secs = timeout if timeout is not None else e.timeout
        raise RuntimeError(f"{tag} 超时（{secs}s）：外部命令长时间没有返回，请检查网络、代理或目标站点限制。")
    if check and cp.returncode != 0:
        err = (cp.stderr or "").strip()[-2000:]
        raise RuntimeError(f"{tag} 失败 (exit={cp.returncode})：{err}")
    return cp


def _looks_like_cookie_db_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "cookie" in text and "database" in text and (
        "could not copy" in text
        or "unable to copy" in text
        or "failed to copy" in text
        or "locked" in text
    )


def run_ytdlp(cfg: Any, args: list[str], *, tag: str = "yt-dlp",
              check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run yt-dlp, falling back without browser cookies if cookie DB copying fails."""
    cookie_args = cfg.cookies_args()
    proxy_args = ["--proxy", cfg.proxy] if getattr(cfg, "proxy", "") else []
    cmd = [cfg.yt_dlp, *proxy_args, *cookie_args, *args]
    try:
        return run(cmd, tag=tag, check=check, timeout=timeout)
    except Exception as e:
        if cookie_args and _looks_like_cookie_db_error(e):
            log.warn(
                tag,
                "浏览器 Cookie 数据库读取失败，已自动改用无 Cookie 模式重试；"
                "如果视频需要登录/大会员权限，请导出 cookies.txt 后配置 VNOTES_COOKIES_FILE。",
            )
            return run([cfg.yt_dlp, *proxy_args, *args], tag=tag, check=check, timeout=timeout)
        raise


def _quote(s: str) -> str:
    return s if re.fullmatch(r"[\w./:=?&%+-]+", s) else f'"{s}"'


# ---- 时间 ----
def fmt_ts(sec: float) -> str:
    """秒 -> M:SS 或 H:MM:SS。"""
    if sec is None or sec < 0:
        return "0:00"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_short(sec: float) -> str:
    """紧凑时间戳，用于文件名：HHMMSS。"""
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}{m:02d}{s:02d}"


# ---- 文件名 ----
_BAD = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def safe_name(s: str, maxlen: int = 40) -> str:
    s = _BAD.sub("_", s).strip().strip("._")
    s = re.sub(r"_+", "_", s)
    return (s or "chapter")[:maxlen]


# ---- JSON ----
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_uri(path: Path) -> str:
    """Windows 友好且空格/中文安全的 file:// URI。"""
    from urllib.parse import quote
    p = str(path.resolve()).replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p
    return "file://" + quote(p, safe="/:")
