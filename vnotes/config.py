"""配置层：环境变量 / .env 读取 + 外部可执行文件自动探测。"""
from __future__ import annotations

import os
import shutil
import dataclasses
from pathlib import Path
from typing import Optional


# ---- .env 加载（零依赖，仅 KEY=VALUE，支持 # 注释与引号）----
def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# ---- 可执行文件探测 ----
_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    str(Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe"),
]
_EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _find(name: str) -> Optional[str]:
    p = shutil.which(name)
    return p if p else None


def detect_chrome() -> Optional[str]:
    p = _find("chrome") or _find("google-chrome")
    if p:
        return p
    for c in _CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    for c in _EDGE_CANDIDATES:  # Edge 同样支持 headless
        if Path(c).exists():
            return c
    return None


@dataclasses.dataclass
class Config:
    # 外部工具
    yt_dlp: str = "yt-dlp"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    chrome: Optional[str] = None
    proxy: str = ""  # 可选 HTTP(S) 代理（如 http://127.0.0.1:7890），透传给 yt-dlp，用于 YouTube 等被墙站点

    # LLM（OpenAI 兼容）
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # cookies
    cookies_browser: str = "chrome"
    cookies_file: str = ""

    # 转写后端：faster-whisper / paraformer / groq / openai-whisper
    transcribe_backend: str = "faster-whisper"
    whisper_device: str = "cuda"
    whisper_model_zh: str = "turbo"
    whisper_model_en: str = "small.en"
    whisper_model_dir: str = ""  # 模型下载目录（空=默认；建议指向 D 盘）
    hf_endpoint: str = "https://hf-mirror.com"
    transcribe_chunk_seconds: int = 300  # 本地转写分片长度，降低长音频内存压力
    vosk_model_dir: str = "D:/vnotes_models/vosk-model-small-cn-0.22"
    # Groq
    groq_api_key: str = ""
    groq_model: str = "whisper-large-v3-turbo"
    # 阿里云 Paraformer
    dashscope_api_key: str = ""
    paraformer_model: str = "paraformer-v2"

    # 长视频优化
    max_video_duration: int = 7200  # 单次处理最大时长（秒），超限则分级摘要
    chunk_analyze_parallel: bool = True  # 分析分块是否并行
    svg_parallel: bool = True  # SVG 生成是否并行
    svg_max_workers: int = 6  # SVG 并行最大线程数
    hierarchical_summary: bool = True  # 启用分级摘要（超长视频先摘要再分析）
    checkpoint_enabled: bool = True  # 启用断点恢复（每块结果存盘，中断后可续）
    analyze_max_workers: int = 4  # 分析分块并行最大线程数
    chunk_summary_target: int = 6000  # 分块摘要目标字符数

    # 路径
    output_dir: Path = Path("./output")
    cache_dir: Path = Path("./cache")

    # 运行参数
    server_port: int = 7458
    duration_tolerance: float = 0.05  # 5%
    segment_tail_tolerance: float = 0.08  # 末段结束与音频时长容差
    slice_height: int = 1700  # 切片目标高度(px)
    slice_overlap: int = 100  # 切片重叠(px)

    # 笔记模式：essence（脉络精华）/ detailed（细致笔记）
    note_mode: str = "essence"

    @classmethod
    def load(cls, env_path: Optional[Path] = None) -> "Config":
        # 优先加载 .env
        candidates = [env_path] if env_path else []
        candidates += [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
        for c in candidates:
            if c:
                load_env_file(c)

        def env(k: str, default: str = "") -> str:
            return os.environ.get(k, default)

        cfg = cls()
        cfg.yt_dlp = _find("yt-dlp") or "yt-dlp"
        cfg.ffmpeg = _find("ffmpeg") or "ffmpeg"
        cfg.ffprobe = _find("ffprobe") or "ffprobe"
        cfg.chrome = detect_chrome()
        cfg.proxy = env("VNOTES_PROXY", "")

        cfg.llm_base_url = env("VNOTES_LLM_BASE_URL", cfg.llm_base_url)
        cfg.llm_api_key = env("VNOTES_LLM_API_KEY", cfg.llm_api_key)
        cfg.llm_model = env("VNOTES_LLM_MODEL", cfg.llm_model)

        cfg.cookies_browser = env("VNOTES_COOKIES_BROWSER", cfg.cookies_browser)
        cfg.cookies_file = env("VNOTES_COOKIES_FILE", cfg.cookies_file)

        cfg.whisper_device = env("VNOTES_WHISPER_DEVICE", cfg.whisper_device)
        cfg.whisper_model_zh = env("VNOTES_WHISPER_MODEL_ZH", cfg.whisper_model_zh)
        cfg.whisper_model_en = env("VNOTES_WHISPER_MODEL_EN", cfg.whisper_model_en)
        cfg.whisper_model_dir = env("VNOTES_WHISPER_MODEL_DIR", cfg.whisper_model_dir)
        cfg.hf_endpoint = env("VNOTES_HF_ENDPOINT", env("HF_ENDPOINT", cfg.hf_endpoint))
        cfg.transcribe_chunk_seconds = int(env("VNOTES_TRANSCRIBE_CHUNK_SECONDS", str(cfg.transcribe_chunk_seconds)))
        cfg.vosk_model_dir = env("VNOTES_VOSK_MODEL_DIR", cfg.vosk_model_dir)
        cfg.transcribe_backend = env("VNOTES_TRANSCRIBE_BACKEND", cfg.transcribe_backend)
        cfg.groq_api_key = env("VNOTES_GROQ_API_KEY", cfg.groq_api_key)
        cfg.groq_model = env("VNOTES_GROQ_MODEL", cfg.groq_model)
        cfg.dashscope_api_key = env("VNOTES_DASHSCOPE_API_KEY", cfg.dashscope_api_key)
        cfg.paraformer_model = env("VNOTES_PARAFORMER_MODEL", cfg.paraformer_model)

        cfg.max_video_duration = int(env("VNOTES_MAX_DURATION", str(cfg.max_video_duration)))
        cfg.svg_max_workers = int(env("VNOTES_SVG_MAX_WORKERS", str(cfg.svg_max_workers)))
        cfg.hierarchical_summary = env("VNOTES_HIERARCHICAL_SUMMARY", "true").lower() == "true"
        cfg.checkpoint_enabled = env("VNOTES_CHECKPOINT_ENABLED", "true").lower() == "true"
        cfg.analyze_max_workers = int(env("VNOTES_ANALYZE_MAX_WORKERS", str(cfg.analyze_max_workers)))
        cfg.chunk_summary_target = int(env("VNOTES_CHUNK_SUMMARY_TARGET", str(cfg.chunk_summary_target)))

        # 模型目录默认指向 D 盘（省 C 盘空间）
        if not cfg.whisper_model_dir:
            d_models = Path("D:/vnotes_models")
            if Path("D:/").exists():
                cfg.whisper_model_dir = str(d_models)

        cfg.output_dir = Path(env("VNOTES_OUTPUT_DIR", str(cfg.output_dir)))
        cfg.cache_dir = Path(env("VNOTES_CACHE_DIR", str(cfg.cache_dir)))
        cfg.note_mode = env("VNOTES_NOTE_MODE", cfg.note_mode)
        cfg.server_port = int(env("VNOTES_SERVER_PORT", str(cfg.server_port)))
        return cfg

    # ---- 便捷 ----
    def cookies_args(self) -> list[str]:
        """返回 yt-dlp 的 cookies 参数。"""
        if self.cookies_file:
            return ["--cookies", self.cookies_file]
        if self.cookies_browser:
            return ["--cookies-from-browser", self.cookies_browser]
        return []

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def check_tools(self, need_whisper: bool = False) -> list[str]:
        """返回缺失项的中文提示列表；为空表示就绪。"""
        missing: list[str] = []
        for name, path in [("yt-dlp", self.yt_dlp), ("ffmpeg", self.ffmpeg), ("ffprobe", self.ffprobe)]:
            if not (path and shutil.which(path) or path and Path(path).exists()):
                missing.append(f"未找到 {name}")
        if not self.chrome:
            missing.append("未找到 Chrome/Edge（用于整页截图兜底）")
        # Playwright（截图主路径）
        try:
            import playwright  # noqa: F401
        except Exception:
            missing.append("未安装 playwright（截图主路径，pip install playwright && python -m playwright install chromium）")
        if need_whisper:
            backend = self.transcribe_backend
            if backend == "groq":
                if not self.groq_api_key:
                    missing.append("未配置 Groq API Key（VNOTES_GROQ_API_KEY）")
            elif backend == "paraformer":
                if not self.dashscope_api_key:
                    missing.append("未配置阿里云 DashScope API Key（VNOTES_DASHSCOPE_API_KEY）")
                try:
                    import dashscope  # noqa: F401
                except Exception:
                    missing.append("未安装 dashscope（pip install dashscope）")
            elif backend == "faster-whisper":
                try:
                    import faster_whisper  # noqa: F401
                except Exception:
                    missing.append("未安装 faster-whisper（pip install faster-whisper）")
            elif backend == "vosk":
                try:
                    import vosk  # noqa: F401
                except Exception:
                    missing.append("未安装 vosk（pip install vosk）")
                if not Path(self.vosk_model_dir).exists():
                    missing.append(f"未找到 Vosk 模型目录：{self.vosk_model_dir}")
            else:  # openai-whisper
                try:
                    import whisper  # noqa: F401
                    import torch  # noqa: F401
                except Exception:
                    missing.append("未安装 openai-whisper/torch（或改用 faster-whisper，见 README）")
        if not self.llm_api_key:
            missing.append("未配置 LLM API Key（VNOTES_LLM_API_KEY）")
        return missing
