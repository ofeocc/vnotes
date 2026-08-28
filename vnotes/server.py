"""vnotes Web UI —— FastAPI 后端 + 嵌入式优雅前端。

启动：python serve.py  或  python -m vnotes.server
默认端口 7458，可通过 VNOTES_SERVER_PORT 环境变量修改。
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

try:
    from .bootstrap import ensure_project_venv, runtime_status
except ImportError:
    _BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))
    from vnotes.bootstrap import ensure_project_venv, runtime_status

ensure_project_venv(module="vnotes.server")

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
    from pydantic import BaseModel
except ImportError:
    raise RuntimeError(
        "Web UI 需要 FastAPI：pip install fastapi uvicorn[standard]"
    )

from .config import Config
from .lightbox import inject_note_lightbox
from .metadata import normalize_video_url
from .qa import note_quality
from .util import log, JobCancelled

# 确保项目根目录在 sys.path（run.py 在根目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

app = FastAPI(title="vnotes", docs_url=None, redoc_url=None)

# ============================================================
#  数据模型
# ============================================================

class GenerateRequest(BaseModel):
    url: str
    transcribe_backend: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    groq_api_key: str = ""
    dashscope_api_key: str = ""
    part: int | None = None
    no_frames: bool = False
    no_slice: bool = False
    batch: bool = False
    note_mode: str = ""  # essence / detailed


# ============================================================
#  任务管理
# ============================================================

_jobs: dict[str, dict] = {}
_JOB_TTL = 3600  # 任务数据保留 1 小时
_JOB_STATE_LOCK = threading.Lock()


def _cleanup_old_jobs() -> None:
    """清理过期的任务数据。"""
    now = time.time()
    expired = [jid for jid, j in _jobs.items()
               if now - j.get("created_at", 0) > _JOB_TTL]
    for jid in expired:
        del _jobs[jid]


def _public_job(job: dict) -> dict[str, Any]:
    return {
        "status": job.get("status"),
        "result": job.get("result"),
        "error": job.get("error"),
        "url": job.get("url"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "logs": job.get("logs", [])[-300:],
        "traceback": job.get("traceback", ""),
    }


def _write_job_snapshot(job_id: str) -> None:
    try:
        cfg = Config.load()
        job_dir = cfg.cache_dir / "jobs"
        job_dir.mkdir(parents=True, exist_ok=True)
        data = _public_job(_jobs[job_id])
        (job_dir / f"{job_id}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _running_job_id() -> str | None:
    for jid, job in _jobs.items():
        if job.get("status") == "running":
            return jid
    return None


def _config_for_request(req: GenerateRequest) -> Config:
    """从 .env 加载配置，再应用本次请求的临时覆盖。"""
    cfg = Config.load()
    if req.llm_api_key:
        cfg.llm_api_key = req.llm_api_key
    if req.llm_base_url:
        cfg.llm_base_url = req.llm_base_url
    if req.llm_model:
        cfg.llm_model = req.llm_model
    if req.transcribe_backend:
        cfg.transcribe_backend = req.transcribe_backend
    if req.groq_api_key:
        cfg.groq_api_key = req.groq_api_key
    if req.dashscope_api_key:
        cfg.dashscope_api_key = req.dashscope_api_key
    if req.note_mode:
        cfg.note_mode = req.note_mode
    return cfg


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _backend_problem(cfg: Config, backend: str | None = None) -> str:
    backend = (backend or cfg.transcribe_backend or "faster-whisper").strip()
    if backend == "faster-whisper":
        if not _module_available("faster_whisper"):
            return "当前 Python 环境没有 faster-whisper。请用项目 venv 启动，或在设置里切到 Vosk/Paraformer。"
    elif backend == "vosk":
        if not _module_available("vosk"):
            return "当前 Python 环境没有 vosk。请用项目 venv 启动，或重新运行 install.bat。"
        if not Path(cfg.vosk_model_dir).exists():
            return f"未找到 Vosk 模型目录：{cfg.vosk_model_dir}"
    elif backend == "groq":
        if not cfg.groq_api_key:
            return "Groq 后端缺少 API Key。"
        if not _module_available("openai"):
            return "Groq 后端需要 openai 包。"
    elif backend == "paraformer":
        if not cfg.dashscope_api_key:
            return "阿里云 Paraformer 后端缺少 DashScope API Key。"
        if not _module_available("dashscope"):
            return "Paraformer 后端需要 dashscope 包。"
    elif backend == "openai-whisper":
        if not (_module_available("whisper") and _module_available("torch")):
            return "openai-whisper 后端需要 whisper/torch。"
    else:
        return f"未知转写后端：{backend}"
    return ""


def _config_status() -> dict[str, Any]:
    """返回非敏感配置状态，供前端解释当前默认路线。"""
    cfg = Config.load()
    backend_problem = _backend_problem(cfg)
    runtime = runtime_status(_PROJECT_ROOT)
    return {
        "default_backend": cfg.transcribe_backend,
        "note_mode": cfg.note_mode,
        "has_llm_api_key": bool(cfg.llm_api_key),
        "has_groq_api_key": bool(cfg.groq_api_key),
        "has_dashscope_api_key": bool(cfg.dashscope_api_key),
        "has_vosk_model": Path(cfg.vosk_model_dir).exists(),
        "has_vosk": _module_available("vosk"),
        "has_faster_whisper": _module_available("faster_whisper"),
        "has_dashscope": _module_available("dashscope"),
        "has_openai_whisper": _module_available("whisper") and _module_available("torch"),
        "backend_ready": not backend_problem,
        "backend_problem": backend_problem,
        "python_env": runtime["python_env"],
        "in_project_venv": runtime["in_project_venv"],
        "project_venv_exists": runtime["project_venv_exists"],
        "server_port": cfg.server_port,
        "llm_model": cfg.llm_model,
        "groq_model": cfg.groq_model,
        "paraformer_model": cfg.paraformer_model,
    }


def _preflight_error(cfg: Config) -> str | None:
    """在启动长任务前给出可操作的错误，避免 SSE 里才模糊失败。"""
    if not cfg.llm_api_key:
        return "缺少 DeepSeek API Key。请在设置里填写 DeepSeek API Key，或写入 .env 的 VNOTES_LLM_API_KEY。"

    backend = (cfg.transcribe_backend or "faster-whisper").strip()
    if backend == "groq":
        if not cfg.groq_api_key:
            return "你选择了 Groq 云端转写，但当前没有 Groq API Key。请在设置里填写 Groq API Key，或写入 .env 的 VNOTES_GROQ_API_KEY。"
        if not _module_available("openai"):
            return "Groq 后端需要 openai 包。请运行：pip install openai"
    elif backend == "paraformer":
        if not cfg.dashscope_api_key:
            return "你选择了阿里云 Paraformer，但当前没有 DashScope API Key。请在设置里填写 DashScope API Key，或写入 .env 的 VNOTES_DASHSCOPE_API_KEY。"
        if not _module_available("dashscope"):
            return "Paraformer 后端需要 dashscope 包。请运行：pip install dashscope"
    elif backend == "vosk":
        if not _module_available("vosk"):
            return "Vosk 后端需要 vosk 包。请运行：pip install vosk"
        if not Path(cfg.vosk_model_dir).exists():
            return f"未找到 Vosk 中文模型目录：{cfg.vosk_model_dir}。请运行：python download_model.py vosk-cn D:/vnotes_models"
    elif backend == "openai-whisper":
        if not (_module_available("whisper") and _module_available("torch")):
            return "openai-whisper 后端需要 whisper/torch。建议改选 faster-whisper、Vosk 或 Groq。"
    elif backend == "faster-whisper":
        problem = _backend_problem(cfg, backend)
        if problem:
            return problem
    elif backend != "faster-whisper":
        return f"未知转写后端：{backend}。请改选 faster-whisper / Vosk / Groq / Paraformer。"
    return None


# ============================================================
#  阶段追踪
# ============================================================

def _history_platform(meta: dict[str, Any]) -> str:
    extractor = str(meta.get("extractor") or "").lower()
    source = str(meta.get("webpage_url") or meta.get("base_url") or "").lower()
    if meta.get("is_youtube") or "youtube" in extractor or "youtu.be" in source or "youtube.com" in source:
        return "YouTube"
    if meta.get("is_bili") or "bili" in extractor or "bilibili.com" in source or "b23.tv" in source:
        return "B站"
    return "其他"


def _history_category(meta: dict[str, Any]) -> str:
    text = " ".join([
        str(meta.get("title") or ""),
        str(meta.get("description") or ""),
        " ".join(str(t) for t in meta.get("tags", [])[:16]),
    ]).lower()
    rules = [
        ("编程", ["python", "编程", "代码", "开发", "程序", "前端", "后端", "算法"]),
        ("影视", ["电影", "剧情", "导演", "演员", "影史", "动作片", "电视剧"]),
        ("体育", ["nba", "詹姆斯", "湖人", "篮球", "足球", "体育"]),
        ("动漫", ["二次元", "动漫", "动画", "美少女", "番剧"]),
        ("知识", ["科普", "知识", "历史", "经济", "风水", "健康", "医学", "睡眠"]),
    ]
    for name, words in rules:
        if any(word in text for word in words):
            return name
    return "未分类"


PIPELINE_STAGES = [
    {"id": "meta", "label": "元数据", "desc": "正在了解这个视频…"},
    {"id": "audio", "label": "音频", "desc": "正在提取音频…"},
    {"id": "transcribe", "label": "转写", "desc": "正在聆听每一句话…"},
    {"id": "analyze", "label": "分析", "desc": "AI 正在理解内容…"},
    {"id": "svg", "label": "图解", "desc": "正在为每章绘制图解…"},
    {"id": "render", "label": "渲染", "desc": "正在排版笔记…"},
    {"id": "screenshot", "label": "截图", "desc": "正在定格画面…"},
    {"id": "crop", "label": "切片", "desc": "正在切分长图…"},
]

_TAG_MAP = {
    "meta": "meta",
    "audio": "audio",
    "transcribe": "transcribe",
    "analyze": "analyze",
    "svg": "svg",
    "screenshot": "screenshot",
    "qa": "screenshot",
    "crop": "crop",
    "selftest": "render",
}


class StageTracker:
    """根据日志 tag 追踪当前流水线阶段。"""

    def __init__(self, emit):
        self.emit = emit
        self.current: str | None = None

    def handle(self, tag: str, msg: str) -> None:
        stage = _TAG_MAP.get(tag)
        if stage is None and tag == "main":
            if "HTML" in msg or "渲染" in msg or "跳过抽帧" in msg:
                stage = "render"
            elif "完成" in msg or "总耗时" in msg or "vnotes" in msg:
                return
        if stage and stage != self.current:
            if self.current:
                self.emit({"type": "stage_done", "stage": self.current})
            self.current = stage
            self.emit({"type": "stage_start", "stage": stage})

    def finish(self) -> None:
        if self.current:
            self.emit({"type": "stage_done", "stage": self.current})
            self.current = None


# ============================================================
#  流水线执行
# ============================================================

def _run_pipeline(job_id: str, req: GenerateRequest) -> None:
    """在后台线程中执行流水线，通过 queue 推送事件。"""
    job = _jobs[job_id]
    q: queue.Queue = job["queue"]

    # 构建配置（从 .env 加载，再用用户输入覆盖）
    cfg = _config_for_request(req)

    def _raise_if_cancelled():
        if job.get("cancel_requested"):
            raise JobCancelled("已取消")

    tracker = StageTracker(lambda ev: q.put(ev))

    # 拦截日志
    original_emit = log._emit

    def patched_emit(lvl: str, tag: str, msg: str) -> None:
        original_emit(lvl, tag, msg)
        tracker.handle(tag, msg)
        job.setdefault("logs", []).append({
            "level": lvl.strip(),
            "tag": tag,
            "message": msg,
            "time": time.time(),
        })
        if len(job["logs"]) > 500:
            del job["logs"][:-500]
        q.put({
            "type": "log",
            "level": lvl.strip(),
            "tag": tag,
            "message": msg,
        })

    log._emit = patched_emit

    try:
        if req.batch:
            from vnotes.batch import batch_pipeline

            def on_progress(completed: int, total: int, title: str):
                q.put({
                    "type": "batch_progress",
                    "completed": completed,
                    "total": total,
                    "title": title,
                })

            html_path = batch_pipeline(
                cfg, req.url,
                no_frames=req.no_frames,
                no_slice=req.no_slice,
                on_progress=on_progress,
                cancel_check=_raise_if_cancelled,
            )
            tracker.finish()

            out_dir = html_path.parent
            rel = out_dir.name
            is_index = html_path.name == "index.html"
            info: dict[str, Any] = {
                "dir": rel,
                "is_index": is_index,
                "title": rel,
            }
            q.put({"type": "done", "result": info})
            job["status"] = "done"
            job["result"] = info
            job["finished_at"] = time.time()
            _write_job_snapshot(job_id)
        else:
            from run import pipeline as run_pipeline

            html_path = run_pipeline(
                cfg, req.url,
                part=req.part,
                no_frames=req.no_frames,
                no_slice=req.no_slice,
                stub_transcript=False,
                cancel_check=_raise_if_cancelled,
            )
            tracker.finish()

            out_dir = html_path.parent
            rel = out_dir.name

            # 读取生成的笔记数据
            info: dict[str, Any] = {"dir": rel}
            data_file = out_dir / "notes_data.json"
            if data_file.exists():
                try:
                    data = json.loads(data_file.read_text(encoding="utf-8"))
                    meta = data.get("meta", {})
                    analysis = data.get("analysis", {})
                    info.update({
                        "title": meta.get("title", rel),
                        "chapters": len(analysis.get("chapters", [])),
                        "duration": meta.get("duration", 0),
                        "has_cover": (out_dir / "cover.jpg").exists(),
                        "has_full": (out_dir / "full.png").exists(),
                        "slices": len(data.get("slices", [])),
                    })
                except Exception:
                    pass

            q.put({"type": "done", "result": info})
            job["status"] = "done"
            job["result"] = info
            job["finished_at"] = time.time()
            _write_job_snapshot(job_id)

    except JobCancelled as e:
        log.warn("main", f"任务已取消：{e}")
        tracker.finish()
        q.put({"type": "cancelled", "message": str(e)})
        job["status"] = "cancelled"
        job["error"] = str(e)
        job["finished_at"] = time.time()
        _write_job_snapshot(job_id)
    except Exception as e:
        tb = traceback.format_exc()
        log.error("main", f"任务失败：{e}")
        tracker.finish()
        q.put({"type": "error", "message": str(e)})
        job["status"] = "error"
        job["error"] = str(e)
        job["traceback"] = tb
        job["finished_at"] = time.time()
        _write_job_snapshot(job_id)
    finally:
        log._emit = original_emit
        q.put(None)  # 结束信号


# ============================================================
#  API 端点
# ============================================================

@app.get("/api/stages")
async def get_stages():
    """返回流水线阶段定义，供前端渲染。"""
    return {"stages": PIPELINE_STAGES}


@app.get("/api/config-status")
async def config_status():
    """返回非敏感配置状态。"""
    return _config_status()


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """启动笔记生成任务，返回 job_id。"""
    with _JOB_STATE_LOCK:
        _cleanup_old_jobs()
        running = _running_job_id()
        if running:
            return JSONResponse({
                "error": f"已有生成任务正在运行（job={running}）。请等它结束，或重启服务后再试；本地转写很吃内存，不能并发跑。",
                "job_id": running,
            }, status_code=409)

        try:
            req.url = normalize_video_url(req.url)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        cfg = _config_for_request(req)
        preflight = _preflight_error(cfg)
        if preflight:
            return JSONResponse({
                "error": preflight,
                "config": _config_status(),
            }, status_code=400)

        job_id = uuid.uuid4().hex[:12]
        _jobs[job_id] = {
            "queue": queue.Queue(),
            "status": "running",
            "result": None,
            "error": None,
            "url": req.url,
            "created_at": time.time(),
            "finished_at": None,
            "logs": [],
            "traceback": "",
            "cancel_requested": False,
        }
        _write_job_snapshot(job_id)

    thread = threading.Thread(target=_run_pipeline, args=(job_id, req), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    """返回任务状态和最近日志，便于失败后排查。"""
    if job_id not in _jobs:
        cfg = Config.load()
        snap = cfg.cache_dir / "jobs" / f"{job_id}.json"
        if snap.exists():
            return JSONResponse(json.loads(snap.read_text(encoding="utf-8")))
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return JSONResponse(_public_job(_jobs[job_id]))


@app.post("/api/job/{job_id}/cancel")
async def job_cancel(job_id: str):
    """请求取消一个正在运行的生成任务。"""
    job = _jobs.get(job_id)
    if not job:
        # 允许对已结束/未知任务返回 ok（无副作用）
        return {"ok": True, "job_id": job_id, "note": "任务不存在或已结束"}
    if job.get("status") != "running":
        return {"ok": True, "job_id": job_id, "note": f"任务状态为 {job.get('status')}，无需取消"}
    with _JOB_STATE_LOCK:
        job["cancel_requested"] = True
    log.info("main", f"收到取消请求：job={job_id}")
    return {"ok": True, "job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    """SSE 端点：实时推送任务进度。"""
    async def event_stream():
        if job_id not in _jobs:
            yield f"data: {json.dumps({'type': 'error', 'message': '任务不存在'}, ensure_ascii=False)}\n\n"
            return

        q: queue.Queue = _jobs[job_id]["queue"]
        while True:
            try:
                event = q.get_nowait()
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.1)
                if _jobs[job_id]["status"] != "running":
                    # 任务已结束，排空剩余事件
                    while True:
                        try:
                            event = q.get_nowait()
                            if event is None:
                                break
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        except queue.Empty:
                            break
                    break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/history")
async def history():
    """列出已生成的笔记。"""
    cfg = Config.load()
    items = []
    if cfg.output_dir.exists():
        for d in sorted(cfg.output_dir.iterdir(),
                        key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
                        reverse=True):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if not (d / "notes.html").exists():
                continue
            item: dict[str, Any] = {
                "name": d.name,
                "title": d.name,
                "uploader": "",
                "duration": 0,
                "chapters": 0,
                "has_cover": (d / "cover.jpg").exists(),
                "has_full": (d / "full.png").exists(),
                "updated_at": int(d.stat().st_mtime),
                "generated_at": int(d.stat().st_mtime),
                "platform": "其他",
                "category": "未分类",
                "source_url": "",
                "upload_date": "",
                "tags": [],
                "quality_status": "check",
                "quality_warnings": [],
                "coverage": 0,
            }
            data_file = d / "notes_data.json"
            if data_file.exists():
                try:
                    data = json.loads(data_file.read_text(encoding="utf-8"))
                    meta = data.get("meta", {})
                    audio = data.get("audio", {})
                    analysis = data.get("analysis", {})
                    quality = data.get("content_qa") or note_quality(
                        meta,
                        audio,
                        analysis,
                        data.get("transcript") or {},
                        data.get("qa") or {},
                    )
                    item.update({
                        "title": meta.get("title", d.name),
                        "uploader": meta.get("uploader", ""),
                        "duration": audio.get("duration_sec") or meta.get("duration", 0),
                        "chapters": len(analysis.get("chapters", [])),
                        "platform": _history_platform(meta),
                        "category": _history_category(meta),
                        "source_url": meta.get("webpage_url") or meta.get("base_url") or "",
                        "upload_date": meta.get("upload_date", ""),
                        "tags": meta.get("tags", [])[:8] if isinstance(meta.get("tags", []), list) else [],
                        "quality_status": quality.get("status", "check"),
                        "quality_warnings": quality.get("warnings", [])[:6],
                        "coverage": quality.get("coverage", 0),
                    })
                except Exception:
                    pass
            items.append(item)
    return {"items": items}


@app.get("/output/{file_path:path}")
async def serve_output(file_path: str):
    """安全地提供 output 目录下的文件。"""
    cfg = Config.load()
    full = (cfg.output_dir / file_path).resolve()
    # 防止路径穿越
    try:
        full.relative_to(cfg.output_dir.resolve())
    except ValueError:
        return JSONResponse({"error": "非法路径"}, status_code=403)
    if full.is_file():
        if full.name.lower() == "notes.html":
            try:
                html_text = full.read_text(encoding="utf-8")
                return HTMLResponse(inject_note_lightbox(html_text))
            except UnicodeDecodeError:
                pass
        return FileResponse(full)
    return JSONResponse({"error": "文件不存在"}, status_code=404)


@app.delete("/api/note/{name}")
async def delete_note(name: str):
    """删除一份已生成的笔记（其输出目录）。"""
    cfg = Config.load()
    safe = Path(name).name  # 取末尾一段，防路径穿越
    full = (cfg.output_dir / safe).resolve()
    try:
        full.relative_to(cfg.output_dir.resolve())
    except ValueError:
        return JSONResponse({"error": "非法路径"}, status_code=403)
    if safe.startswith("_"):
        return JSONResponse({"error": "不允许删除系统目录"}, status_code=403)
    if not full.is_dir():
        return JSONResponse({"error": "笔记不存在"}, status_code=404)
    try:
        shutil.rmtree(full)
    except Exception as e:
        return JSONResponse({"error": f"删除失败: {e}"}, status_code=500)
    log.info("main", f"已删除笔记目录：{safe}")
    return {"ok": True, "deleted": safe}


# ============================================================
#  前端页面（嵌入式单页应用）
# ============================================================

_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>vnotes · 视频笔记工作室</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0' stop-color='%23fff' stop-opacity='.42'/%3E%3Cstop offset='1' stop-color='%23000' stop-opacity='.06'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='64' height='64' rx='14' fill='%23F5C518'/%3E%3Crect width='64' height='64' rx='14' fill='url(%23g)'/%3E%3Cpath d='M11 16 L22 16 L32 40 L42 16 L53 16 L32 50 Z' fill='%23121109'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Noto+Sans+SC:wght@300;400;500;600;700&family=Yellowtail&family=Great+Vibes&display=swap" rel="stylesheet">
<style>
:root{
  color-scheme:light;
  --bg:#fafafa; --bg2:#f0f0f0;
  --surface:#fff; --surface2:#f7f7f7;
  --border:#e5e5e5; --border2:#d0d0d0;
  --text:#1a1a1a; --text2:#5a5a5a; --text3:#767676;
  --accent:#F5C518; --accent2:#e0a800; --accent-dk:#1a1a1a;
  --accent-glow:rgba(245,197,24,.35);
  --accent-soft:#fff8e1;
  --cool:#2ab7ca; --cool-soft:#e7fbfd; --cool-glow:rgba(42,183,202,.18);
  --ink-soft:#eef1f4;
  --amber:#FF8C00; --amber2:#FFA500; --amber-glow:rgba(255,140,0,.18);
  --rig-dk:#0a0a0a; --rig-mid:#1a1a1a;
  --success:#2ecc71; --error:#ff3b30; --warn:#ff6b35;
  --r:18px; --rs:12px;
  --t:.35s cubic-bezier(.4,0,.2,1);
  --t-out:.28s cubic-bezier(.16,1,.3,1);
  --t-spring:.5s cubic-bezier(.34,1.56,.64,1);
  --focus-ring:0 0 0 3px rgba(245,197,24,.45);
  /* 阴影层级系统 */
  --sh-1:0 1px 2px rgba(0,0,0,.04),0 1px 3px rgba(0,0,0,.06);
  --sh-2:0 2px 8px rgba(0,0,0,.05),0 1px 3px rgba(0,0,0,.04);
  --sh-3:0 8px 28px rgba(0,0,0,.08),0 2px 8px rgba(0,0,0,.04);
  --sh-4:0 20px 60px rgba(0,0,0,.12),0 6px 20px rgba(0,0,0,.06);
  --sh-glow:0 8px 32px var(--accent-glow);
  /* 字体 */
  --font-display:"Instrument Serif","Noto Serif SC",Georgia,serif;
  --font-body:"Noto Sans SC",-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;
  --font-script:"Yellowtail","Great Vibes","Brush Script MT",cursive;
  --font-mono:"SF Mono","Cascadia Code","JetBrains Mono","Fira Code",monospace;
}
html[data-theme="night"]{
  color-scheme:dark;
  --bg:#101114; --bg2:#17191e;
  --surface:#1b1d22; --surface2:#23262c;
  --border:#31343c; --border2:#444954;
  --text:#f0f2f5; --text2:#b9c0ca; --text3:#89919e;
  --accent:#F5C518; --accent2:#ffd84a; --accent-dk:#15110a;
  --accent-glow:rgba(245,197,24,.28);
  --accent-soft:rgba(245,197,24,.16);
  --cool:#48c7d8; --cool-soft:rgba(72,199,216,.12); --cool-glow:rgba(72,199,216,.2);
  --ink-soft:#252a31;
  --rig-dk:#050608; --rig-mid:#101218;
  --sh-1:0 1px 2px rgba(0,0,0,.22),0 1px 3px rgba(0,0,0,.28);
  --sh-2:0 2px 8px rgba(0,0,0,.26),0 1px 3px rgba(0,0,0,.22);
  --sh-3:0 10px 32px rgba(0,0,0,.34),0 2px 8px rgba(0,0,0,.24);
  --sh-4:0 24px 70px rgba(0,0,0,.42),0 6px 20px rgba(0,0,0,.28);
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  position:relative;
  background:radial-gradient(ellipse 45% 55% at 50% 0%,#fff3e0 0%,#fdf6ec 15%,#fafafa 45%,var(--bg) 70%);
  color:var(--text);
  font-family:var(--font-body);
  -webkit-font-smoothing:antialiased;line-height:1.6;font-size:15px;
  min-height:100vh;overflow-x:hidden;
}

/* ---- 聚光灯黑色棚顶（fixed 始终笼罩顶部，与光锥一体） ---- */
.spot-rig{
  position:fixed;top:0;left:0;z-index:100;
  width:100%;height:92px;
  background:linear-gradient(180deg,var(--rig-dk) 0%,var(--rig-mid) 65%,#2a2a2a 100%);
  border-bottom:2px solid #333;
  box-shadow:0 8px 28px rgba(0,0,0,.4),inset 0 -1px 0 rgba(255,255,255,.05);
}
/* 棚顶横向金属导轨纹理 */
.spot-rig{
  background-image:
    linear-gradient(180deg,var(--rig-dk) 0%,var(--rig-mid) 65%,#2a2a2a 100%),
    repeating-linear-gradient(90deg,transparent 0,transparent 80px,rgba(255,255,255,.015) 80px,rgba(255,255,255,.015) 82px);
}
/* 棚顶中央铭牌标志（独立 fixed，默认隐藏为彩蛋） */
.rig-badge{
  position:fixed;top:0;left:50%;
  transform:translateX(-50%) translateY(-80px);
  display:flex;align-items:center;gap:8px;
  padding:6px 16px 6px 12px;
  background:linear-gradient(180deg,#1f1f1f 0%,#161616 100%);
  border:1px solid #333;border-radius:20px;
  box-shadow:0 2px 8px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.06),inset 0 -1px 0 rgba(0,0,0,.4);
  z-index:102;opacity:0;
  transition:transform .5s cubic-bezier(.34,1.56,.64,1),opacity .4s ease-out;
}
/* 在页面顶部时微微探出一点边角（暗示有东西） */
.rig-badge.at-top{
  transform:translateX(-50%) translateY(-52px);
  opacity:.25;
}
/* overscroll 拉出时完全弹出 */
.rig-badge.pulled{
  transform:translateX(-50%) translateY(14px);
  opacity:1;
}
/* 铭牌左侧小灯（电源指示） */
.rig-badge::before{
  content:'';width:6px;height:6px;border-radius:50%;
  background:var(--accent);
  box-shadow:0 0 8px var(--accent),0 0 14px rgba(245,197,24,.5);
  animation:badgePulse 2.4s ease-in-out infinite;
  flex-shrink:0;
}
.rig-badge-text{
  font-family:var(--font-script);
  font-size:19px;font-weight:400;letter-spacing:0;
  color:rgba(255,255,255,.85);
  line-height:1;
  display:flex;align-items:baseline;gap:1px;
}
.rig-badge-text em{
  font-style:normal;color:var(--accent);
  text-shadow:0 0 12px rgba(245,197,24,.62);
}
/* 铭牌底部小字 */
.rig-badge-sub{
  font-family:var(--font-body);
  font-size:7px;letter-spacing:.3em;text-transform:uppercase;
  color:rgba(255,255,255,.3);
  margin-left:2px;
}
/* 铭牌右侧螺丝 */
.rig-badge::after{
  content:'';position:absolute;right:6px;top:50%;transform:translateY(-50%);
  width:4px;height:4px;border-radius:50%;
  background:radial-gradient(circle,#666 0%,#333 60%,#222 100%);
  box-shadow:0 1px 1px rgba(0,0,0,.6);
}
.theme-pull{
  --cord-rest:44px;
  --pull-extra:0px;
  position:fixed;
  top:0;
  right:42px;
  z-index:430;
  width:44px;
  height:132px;
  border:0;
  padding:0;
  background:transparent;
  cursor:grab;
  touch-action:none;
}
.theme-pull:active{cursor:grabbing}
.theme-pull .pull-cord{
  position:absolute;
  top:0;
  right:12px;
  width:24px;
  height:calc(var(--cord-rest) + var(--pull-extra));
  transform-origin:top center;
  transition:height .5s cubic-bezier(.22,1,.36,1),filter var(--t-out);
  filter:drop-shadow(0 2px 5px rgba(0,0,0,.2));
}
.theme-pull.dragging .pull-cord{transition:none}
.theme-pull .pull-line{
  position:absolute;
  top:0;
  left:50%;
  width:2px;
  height:calc(100% - 15px);
  transform:translateX(-50%);
  border-radius:2px;
  background:
    repeating-linear-gradient(180deg,rgba(255,255,255,.92) 0 4px,rgba(206,211,217,.92) 4px 8px);
  box-shadow:0 0 0 1px rgba(0,0,0,.06),0 0 12px rgba(245,197,24,.12);
}
.theme-pull .pull-handle{
  position:absolute;
  left:50%;
  bottom:0;
  width:24px;
  height:24px;
  transform:translateX(-50%);
  border:2px solid rgba(255,255,255,.88);
  border-radius:50%;
  background:radial-gradient(circle at 50% 38%,rgba(255,255,255,.84) 0 18%,rgba(245,197,24,.95) 22% 48%,rgba(165,120,12,.92) 49% 100%);
  box-shadow:0 4px 13px rgba(0,0,0,.24),0 0 0 1px rgba(0,0,0,.08),0 0 18px rgba(245,197,24,.24);
  transition:box-shadow var(--t-out),transform var(--t-spring),border-color var(--t-out);
}
.theme-pull:hover .pull-handle,
.theme-pull.armed .pull-handle{
  transform:translateX(-50%) scale(1.07);
  box-shadow:0 7px 20px rgba(0,0,0,.28),0 0 0 1px rgba(0,0,0,.08),0 0 24px rgba(245,197,24,.42);
}
.theme-pull:focus-visible{
  outline:none;
}
.theme-pull:focus-visible .pull-handle{
  box-shadow:0 0 0 4px rgba(245,197,24,.26),0 7px 20px rgba(0,0,0,.28),0 0 24px rgba(245,197,24,.42);
}
html[data-theme="night"] .theme-pull{
  --cord-rest:68px;
}
html[data-theme="night"] .theme-pull .pull-line{
  background:
    repeating-linear-gradient(180deg,rgba(206,215,226,.92) 0 4px,rgba(141,151,165,.92) 4px 8px);
  box-shadow:0 0 0 1px rgba(255,255,255,.06),0 0 16px rgba(245,197,24,.14);
}
html[data-theme="night"] .theme-pull .pull-handle{
  border-color:rgba(245,197,24,.78);
  background:radial-gradient(circle at 50% 38%,rgba(255,246,201,.9) 0 16%,rgba(245,197,24,.95) 20% 46%,rgba(70,58,35,.96) 48% 100%);
  box-shadow:0 7px 20px rgba(0,0,0,.42),0 0 0 1px rgba(245,197,24,.18),0 0 22px rgba(245,197,24,.34);
}
@media(max-width:640px){
  .theme-pull{
    right:9px;
    transform:scale(.88);
    transform-origin:top right;
  }
}
/* 棚顶两侧延伸壁（收窄） */
.rig-wall{
  position:fixed;top:0;width:32px;height:92px;z-index:101;
  background:linear-gradient(180deg,var(--rig-dk) 0%,var(--rig-mid) 70%,#222 100%);
  box-shadow:0 8px 24px rgba(0,0,0,.35);
}
.rig-wall-l{left:0;border-right:1px solid rgba(255,255,255,.04)}
.rig-wall-r{right:0;border-left:1px solid rgba(255,255,255,.04)}
/* 侧壁底部螺栓装饰 */
.rig-wall::after{
  content:'';position:absolute;bottom:8px;left:50%;transform:translateX(-50%);
  width:16px;height:5px;border-radius:3px;
  background:linear-gradient(180deg,#444,#222);
  box-shadow:0 1px 2px rgba(0,0,0,.5);
}
/* 影棚两侧暗角（收窄，不再遮挡内容区） */
.stage-vignette{
  position:fixed;top:0;width:7vw;height:100vh;z-index:99;pointer-events:none;
}
.stage-vignette-l{left:0;background:linear-gradient(90deg,rgba(0,0,0,.07) 0%,transparent 100%)}
.stage-vignette-r{right:0;background:linear-gradient(270deg,rgba(0,0,0,.07) 0%,transparent 100%)}
/* 中央大灯罩凸起（与棚顶无缝衔接） */
.spot-rig::before{
  content:'';position:absolute;top:100%;left:50%;transform:translateX(-50%);
  width:210px;height:40px;margin-top:-2px;
  background:linear-gradient(180deg,#252525 0%,#2a2a2a 30%,#1a1a1a 70%,#0d0d0d 100%);
  border-radius:0 0 105px 105px/0 0 40px 40px;
  border:1.5px solid #2a2a2a;border-top:none;
  box-shadow:0 8px 22px rgba(0,0,0,.4),inset 0 -4px 14px rgba(255,140,0,.08),inset 0 2px 0 rgba(255,255,255,.03);
}
/* 中央大灯泡（橘黄，位于灯罩底部） */
.spot-rig::after{
  content:'';position:absolute;top:100%;left:50%;transform:translateX(-50%);
  width:96px;height:16px;margin-top:24px;
  background:radial-gradient(ellipse at center,#fff 0%,#FFE4B5 12%,var(--amber) 38%,var(--amber2) 62%,transparent 88%);
  border-radius:50%;
  box-shadow:0 0 44px 12px rgba(255,140,0,.7),0 0 110px 22px rgba(255,165,0,.35),0 0 180px 34px rgba(255,140,0,.12);
  animation:bulbFlicker 4s ease-in-out infinite;
}

/* ---- 两侧白色小聚光灯（灯罩与棚顶无缝衔接） ---- */
.lamp{
  position:absolute;top:100%;width:56px;height:30px;margin-top:-2px;
  background:linear-gradient(180deg,#252525 0%,#1a1a1a 55%,#0d0d0d 100%);
  border-radius:0 0 56px 56px/0 0 30px 30px;
  border:1.5px solid #2a2a2a;border-top:none;
  box-shadow:0 6px 16px rgba(0,0,0,.4),inset 0 -3px 8px rgba(0,0,0,.3);
}
/* 灯罩与棚顶连接处的小法兰盘（消除缝隙的视觉过渡） */
.lamp::before{
  content:'';position:absolute;top:-4px;left:50%;transform:translateX(-50%);
  width:40px;height:6px;
  background:linear-gradient(180deg,var(--rig-mid) 0%,#252525 100%);
  border-radius:3px 3px 0 0;
  box-shadow:0 1px 2px rgba(0,0,0,.3);
}
/* 小灯灯泡（白光，位于灯罩底部开口处） */
.lamp::after{
  content:'';position:absolute;top:100%;left:50%;transform:translateX(-50%);
  width:40px;height:12px;margin-top:-4px;
  background:radial-gradient(ellipse at center,
    #fff 0%,#fff 5%,#f0f6ff 20%,#c8d8f0 45%,transparent 80%);
  border-radius:50%;
  box-shadow:
    0 0 36px 10px rgba(255,255,255,.9),
    0 0 72px 20px rgba(220,240,255,.5),
    0 0 120px 30px rgba(200,220,255,.2);
  animation:lampFlicker 5s ease-in-out infinite;
}
/* 四盏侧灯位置（对称分布，统一 left） */
.lamp-l1{left:10%}
.lamp-l1::after{animation-delay:0s}
.lamp-l2{left:26%}
.lamp-l2::after{animation-delay:1.2s}
.lamp-r1{left:74%}
.lamp-r1::after{animation-delay:.6s}
.lamp-r2{left:90%}
.lamp-r2::after{animation-delay:1.8s}

/* 小灯白色光锥（锤子风格：从灯泡位置向下的锐利锥形光束） */
.lamp-cone{
  position:fixed;top:118px;width:200px;height:calc(100vh - 118px);
  pointer-events:none;z-index:2;
  /* 极淡的大气雾感（光束周围的空气散射） */
  background:radial-gradient(ellipse 35% 92% at 50% 0%,
    rgba(255,255,255,.05) 0%,
    rgba(245,250,255,.02) 30%,
    transparent 60%);
}
/* 锐利锥形光束主体（clip-path 切出梯形） */
.lamp-cone::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(180deg,
    rgba(255,255,255,.50) 0%,
    rgba(250,253,255,.32) 12%,
    rgba(240,248,255,.16) 35%,
    rgba(230,242,255,.05) 60%,
    rgba(220,235,255,.01) 80%,
    transparent 92%);
  clip-path:polygon(38% 0%, 62% 0%, 82% 100%, 18% 100%);
  -webkit-clip-path:polygon(38% 0%, 62% 0%, 82% 100%, 18% 100%);
  animation:lampConeBreath 6s ease-in-out infinite;
  filter:blur(.5px) drop-shadow(0 0 8px rgba(255,255,255,.12));
}
/* 地面聚焦光斑（光束打在地面上的亮圆） */
.lamp-cone::after{
  content:'';position:absolute;
  bottom:6%;left:50%;transform:translateX(-50%);
  width:115px;height:28px;
  background:radial-gradient(ellipse 50% 50% at center,
    rgba(255,255,255,.45) 0%,
    rgba(245,250,255,.20) 35%,
    rgba(235,245,255,.06) 60%,
    transparent 75%);
  border-radius:50%;
  filter:blur(4px);
  animation:lampSpotBreath 6s ease-in-out infinite;
}
/* 光锥 left = 灯罩中心(灯罩left + 28px) - 光锥半宽(100px) */
.lamp-cone.l1{left:calc(10% + 28px - 100px)}
.lamp-cone.l1::before,.lamp-cone.l1::after{animation-delay:0s}
.lamp-cone.l2{left:calc(26% + 28px - 100px)}
.lamp-cone.l2::before,.lamp-cone.l2::after{animation-delay:1.2s}
.lamp-cone.r1{left:calc(74% + 28px - 100px)}
.lamp-cone.r1::before,.lamp-cone.r1::after{animation-delay:.6s}
.lamp-cone.r2{left:calc(90% + 28px - 100px)}
.lamp-cone.r2::before,.lamp-cone.r2::after{animation-delay:1.8s}

/* ---- 中央橘黄色光锥（从灯泡位置柔和扩散） ---- */
.spot-cone{
  position:fixed;top:92px;left:50%;transform:translateX(-50%);
  width:80vw;height:calc(100vh - 92px);pointer-events:none;z-index:2;
  background:
    radial-gradient(ellipse 42% 68% at 50% 0%,
      rgba(255,140,0,.26) 0%,
      rgba(255,165,0,.13) 22%,
      rgba(255,140,0,.05) 45%,
      rgba(255,100,0,.015) 65%,
      transparent 85%);
  animation:coneBreath 8s ease-in-out infinite,coneSway 14s ease-in-out infinite;
  transform-origin:50% 0%;
}
/* 光锥中心更亮的核 */
.spot-core{
  position:fixed;top:92px;left:50%;transform:translateX(-50%);
  width:44vw;height:calc(100vh - 92px);pointer-events:none;z-index:2;
  background:radial-gradient(ellipse 30% 50% at 50% 0%,
    rgba(255,190,90,.20) 0%,
    rgba(255,160,0,.08) 35%,
    transparent 65%);
  animation:coneBreath 8s ease-in-out infinite reverse,coneSway 14s ease-in-out infinite reverse;
  transform-origin:50% 0%;
}

body::after{
  content:'';position:fixed;inset:0;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence baseFrequency='.9' numOctaves='4'/></filter><rect width='200' height='200' filter='url(%23n)' opacity='.018'/></svg>");
  pointer-events:none;z-index:0;
}
.app{max-width:640px;margin:0 auto;padding:156px 24px 120px;position:relative;z-index:1}

/* ---- Hero ---- */
.hero{text-align:center;margin-bottom:44px;animation:fadeInDown .9s cubic-bezier(.16,1,.3,1)}
.hero-mark{
  width:5px;height:5px;border-radius:50%;background:var(--accent);
  margin:0 auto 16px;box-shadow:0 0 14px var(--accent-glow);
  animation:dotPulse 3s ease-in-out infinite;
  position:relative;
}
/* 旋转光环装饰 */
.hero-mark::before{
  content:'';position:absolute;top:50%;left:50%;
  width:32px;height:32px;margin:-16px 0 0 -16px;
  border:1px solid var(--accent);border-radius:50%;
  opacity:0;animation:ringExpand 3s ease-out infinite;
}
.hero-mark::after{
  content:'';position:absolute;top:50%;left:50%;
  width:32px;height:32px;margin:-16px 0 0 -16px;
  border:1px solid var(--accent);border-radius:50%;
  opacity:0;animation:ringExpand 3s ease-out infinite 1.5s;
}
.hero h1{
  font-family:var(--font-script);
  font-size:72px;font-weight:400;letter-spacing:0;
  color:var(--text);line-height:1.12;
  position:relative;display:inline-block;
  text-shadow:0 2px 22px rgba(0,0,0,.16);
}
/* 聚光灯穿透标题效果 */
.hero h1::before{
  content:'';position:absolute;top:-20%;left:50%;transform:translateX(-50%);
  width:150%;height:160%;
  background:radial-gradient(ellipse 50% 60% at 50% 50%,rgba(245,197,24,.14) 0%,transparent 70%);
  pointer-events:none;z-index:-1;
}
.hero h1 em{
  font-style:normal;
  color:transparent;text-shadow:none;
  background:linear-gradient(140deg,#FFD93D 0%,#F5C518 45%,#e0a800 100%);
  -webkit-background-clip:text;background-clip:text;
  padding:0 6px;font-weight:400;
  position:relative;
  filter:drop-shadow(0 3px 16px rgba(245,197,24,.42));
  transition:filter var(--t-out),transform var(--t-out);
}
.hero h1 em::after{content:none}
.hero h1 em:hover{
  transform:translateY(-2px) rotate(-1deg) scale(1.02);
  filter:drop-shadow(0 6px 26px rgba(245,197,24,.62));
}
.hero p{
  font-size:12px;color:var(--text2);margin-top:12px;
  letter-spacing:.2em;text-transform:uppercase;font-weight:400;
  display:flex;align-items:center;justify-content:center;gap:14px;
}
/* 副标题装饰线 */
.hero p::before,.hero p::after{
  content:'';width:24px;height:1px;background:var(--border2);
}

/* ---- Input ---- */
.input-card{
  background:rgba(255,255,255,.72);
  backdrop-filter:blur(20px) saturate(1.8);
  -webkit-backdrop-filter:blur(20px) saturate(1.8);
  border:1px solid rgba(255,255,255,.6);
  border-radius:var(--r);padding:22px;
  box-shadow:var(--sh-3),inset 0 1px 0 rgba(255,255,255,.5);
  animation:fadeInUp .9s cubic-bezier(.16,1,.3,1) .1s both;
  transition:border-color var(--t-out),box-shadow var(--t-out);
  position:relative;overflow:hidden;
}
/* 卡片顶部微光带 */
.input-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.8),transparent);
}
.input-card:focus-within{
  border-color:var(--accent);
  box-shadow:var(--sh-4),var(--sh-glow),inset 0 1px 0 rgba(255,255,255,.5);
}
.input-card.settings-open,
.input-card.settings-open:focus-within{
  border-color:rgba(245,197,24,.36);
  box-shadow:0 16px 46px rgba(18,20,23,.11),0 0 0 1px rgba(245,197,24,.16),inset 0 1px 0 rgba(255,255,255,.55);
}
.input-row{display:flex;gap:10px}
.url-input{
  flex:1;background:rgba(247,247,247,.6);border:1px solid var(--border);
  border-radius:var(--rs);padding:14px 16px;color:var(--text);
  font-size:15px;font-family:inherit;outline:none;transition:all var(--t-out);
}
.url-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow),inset 0 1px 3px rgba(0,0,0,.03);background:var(--surface)}
.url-input::placeholder{color:var(--text3)}
.gen-btn{
  background:linear-gradient(135deg,#F5C518 0%,#FFD93D 100%);
  color:var(--accent-dk);border:none;border-radius:var(--rs);
  padding:14px 24px;font-size:14px;font-weight:700;font-family:inherit;
  cursor:pointer;transition:all var(--t-out);white-space:nowrap;
  letter-spacing:.02em;position:relative;overflow:hidden;
  box-shadow:0 2px 8px rgba(245,197,24,.3),inset 0 1px 0 rgba(255,255,255,.4);
}
/* 光泽扫过 */
.gen-btn::before{
  content:'';position:absolute;top:0;left:-100%;width:50%;height:100%;
  background:linear-gradient(105deg,transparent 30%,rgba(255,255,255,.5) 50%,transparent 70%);
  transition:left .6s cubic-bezier(.16,1,.3,1);
}
.gen-btn:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(245,197,24,.4),inset 0 1px 0 rgba(255,255,255,.5)}
.gen-btn:hover::before{left:120%}
.gen-btn:active{transform:translateY(0);box-shadow:0 2px 8px rgba(245,197,24,.3)}
.gen-btn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.gen-btn:disabled::before{display:none}

/* ---- Settings ---- */
.settings-toggle{
  background:none;border:none;color:var(--text2);font-size:12px;
  cursor:pointer;margin-top:14px;padding:4px 0;font-family:inherit;
  transition:color var(--t);letter-spacing:.04em;
  display:flex;align-items:center;gap:6px;
}
.settings-toggle:hover{color:var(--text)}
.settings-toggle .arrow{display:inline-block;transition:transform var(--t);font-size:10px}
.settings-toggle.open .arrow{transform:rotate(180deg)}
.settings{
  max-height:0;overflow:hidden;
  transition:max-height .45s cubic-bezier(.4,0,.2,1),opacity .3s ease;
  display:grid;gap:16px;padding:0 4px;opacity:0;
}
.settings.open{max-height:600px;padding-top:18px;opacity:1}
.sg{display:flex;flex-direction:column;gap:6px}
.sg label{
  font-size:10px;color:var(--text2);letter-spacing:.1em;
  text-transform:uppercase;font-weight:600;
}
.sg select,.sg input{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--rs);padding:11px 14px;color:var(--text);
  font-size:13px;font-family:inherit;outline:none;transition:all var(--t);
}
.sg select:focus,.sg input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow);background:var(--surface)}
.sg select option{background:var(--surface);color:var(--text)}
.hint{font-size:11px;color:var(--text3);line-height:1.5;letter-spacing:.01em}

.settings-row{
  display:flex;
  align-items:center;
  gap:10px;
  margin-top:14px;
  min-width:0;
}
.settings-row .settings-toggle{
  margin-top:0;
  padding:7px 10px;
  border:1px solid var(--border);
  border-radius:10px;
  background:rgba(255,255,255,.58);
  box-shadow:var(--sh-1);
}
.settings-row .settings-toggle:hover{
  border-color:var(--border2);
  background:var(--surface);
}
.settings-summary{
  min-width:0;
  color:var(--text3);
  font-size:12px;
  line-height:1.4;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.settings-backdrop{
  position:fixed;
  inset:0;
  z-index:410;
  background:rgba(12,14,18,.24);
  backdrop-filter:blur(3px);
  -webkit-backdrop-filter:blur(3px);
  opacity:0;
  pointer-events:none;
  transition:opacity var(--t-out);
}
.settings-backdrop.open{
  opacity:1;
  pointer-events:auto;
}
.settings-backdrop[hidden]{display:none}
.settings{
  position:relative;
  z-index:1;
  width:100%;
  max-height:0;
  overflow:hidden;
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:14px 12px;
  padding:0 2px;
  opacity:0;
  visibility:hidden;
  pointer-events:none;
  transform:translateY(-8px);
  border-top:1px solid transparent;
  background:linear-gradient(180deg,rgba(255,255,255,.36),rgba(255,255,255,.12));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.54);
  transition:
    max-height .56s cubic-bezier(.16,1,.3,1),
    padding .42s cubic-bezier(.16,1,.3,1),
    margin .42s cubic-bezier(.16,1,.3,1),
    opacity .28s ease,
    transform .42s cubic-bezier(.16,1,.3,1),
    border-color .42s ease,
    visibility 0s linear .42s;
}
.settings.open{
  margin-top:14px;
  max-height:760px;
  padding:16px 2px 2px;
  opacity:1;
  visibility:visible;
  pointer-events:auto;
  transform:none;
  border-top-color:rgba(26,26,26,.08);
  transition:
    max-height .6s cubic-bezier(.16,1,.3,1),
    padding .42s cubic-bezier(.16,1,.3,1),
    margin .42s cubic-bezier(.16,1,.3,1),
    opacity .28s ease .08s,
    transform .42s cubic-bezier(.16,1,.3,1),
    border-color .42s ease,
    visibility 0s;
}
.settings .hint,
.settings .batch-toggle{
  grid-column:1 / -1;
}
.settings-panel-head{
  display:none;
  align-items:flex-start;
  justify-content:space-between;
  gap:14px;
  padding-bottom:4px;
}
.settings-panel-title{
  display:flex;
  flex-direction:column;
  gap:2px;
  min-width:0;
}
.settings-panel-title strong{
  color:var(--text);
  font-size:15px;
  font-weight:650;
}
.settings-panel-title span{
  color:var(--text3);
  font-size:12px;
}
.settings-close{
  width:30px;
  height:30px;
  display:inline-grid;
  place-items:center;
  flex:0 0 auto;
  border:1px solid var(--border);
  border-radius:10px;
  background:var(--surface2);
  color:var(--text2);
  font-size:20px;
  line-height:1;
  cursor:pointer;
  transition:background var(--t-out),border-color var(--t-out),color var(--t-out),transform var(--t-out);
}
.settings-close:hover{
  background:var(--surface);
  border-color:var(--border2);
  color:var(--text);
  transform:translateY(-1px);
}
body.settings-drawer-open{overflow:hidden}
@media(max-width:640px){
  .settings-row{
    align-items:flex-start;
  }
  .settings-summary{
    padding-top:2px;
    white-space:normal;
    display:-webkit-box;
    -webkit-line-clamp:2;
    -webkit-box-orient:vertical;
  }
  .settings{
    grid-template-columns:1fr;
    max-height:0;
    transform:translateY(-6px);
  }
  .settings.open{
    max-height:980px;
    transform:none;
  }
}

/* ---- Processing ---- */
.processing{margin-top:36px;animation:fadeInUp .5s ease}
.stage-desc{
  text-align:center;font-family:var(--font-display);
  font-size:22px;font-weight:400;color:var(--text2);
  margin-bottom:24px;letter-spacing:-.01em;transition:all var(--t-out);
  min-height:30px;font-style:italic;
}
.stage-desc.active{color:var(--text)}

.pipeline{
  display:flex;align-items:flex-start;justify-content:space-between;
  padding:0 4px;position:relative;
}
.p-node{display:flex;flex-direction:column;align-items:center;gap:10px;position:relative;z-index:2;flex-shrink:0}
.p-circle{
  width:28px;height:28px;border-radius:50%;border:1.5px solid var(--border2);
  background:var(--surface);display:flex;align-items:center;justify-content:center;
  transition:all var(--t-out);position:relative;
  box-shadow:var(--sh-1);
}
.p-circle.done{border-color:var(--accent);background:linear-gradient(135deg,#F5C518 0%,#FFD93D 100%);box-shadow:0 2px 8px var(--accent-glow)}
.p-circle.done::after{content:'';width:8px;height:5px;border-left:2px solid var(--accent-dk);border-bottom:2px solid var(--accent-dk);transform:rotate(-45deg) translate(1px,-1px)}
/* running 节点：旋转光环 */
.p-circle.running{border-color:var(--accent);background:var(--surface);box-shadow:0 0 0 4px var(--accent-glow)}
.p-circle.running::before{
  content:'';position:absolute;inset:-4px;border-radius:50%;
  border:1.5px solid transparent;border-top-color:var(--accent);border-right-color:var(--accent);
  animation:spin 1s linear infinite;
}
.p-circle.running::after{content:'';width:8px;height:8px;border-radius:50%;background:var(--accent);animation:blink 1.2s ease-in-out infinite;box-shadow:0 0 8px var(--accent-glow)}
.p-circle.error{border-color:var(--error);background:var(--error)}
.p-label{font-size:10px;color:var(--text3);text-align:center;transition:color var(--t-out);letter-spacing:.03em;font-weight:400}
.p-label.active{color:var(--accent-dk);font-weight:600}
.p-label.done{color:var(--text2)}
.p-conn{
  flex:1;height:2px;background:var(--border);margin-top:13px;
  position:relative;overflow:hidden;transition:background var(--t-out);min-width:6px;border-radius:1px;
}
.p-conn.done{background:linear-gradient(90deg,var(--accent),#FFD93D)}
.p-conn.running::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(90deg,transparent 20%,var(--accent) 50%,transparent 80%);
  animation:flow 1.5s ease-in-out infinite;
  box-shadow:0 0 6px var(--accent-glow);
}

/* ---- Log ---- */
.log-box{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--rs);padding:14px 16px;margin-top:18px;
  max-height:200px;overflow-y:auto;
  font-family:var(--font-mono);
  font-size:11.5px;line-height:1.8;
}
.log-line{color:var(--text2);animation:slideIn .3s ease;padding:0}
.log-lvl{display:inline-block;width:52px;color:var(--accent-dk);margin-right:8px;font-weight:600;opacity:.7}
.log-line.warn .log-lvl{color:var(--warn)}
.log-line.error .log-lvl{color:var(--error)}
.log-line .log-tag{color:var(--text3);margin-right:8px}

/* ---- Result ---- */
.result-card{
  background:rgba(255,255,255,.8);
  backdrop-filter:blur(16px) saturate(1.6);
  -webkit-backdrop-filter:blur(16px) saturate(1.6);
  border:1px solid rgba(255,255,255,.6);
  border-radius:var(--r);padding:26px;margin-top:24px;
  box-shadow:var(--sh-3),inset 0 1px 0 rgba(255,255,255,.5);
  animation:scaleIn .5s cubic-bezier(.16,1,.3,1);
}
.result-head{display:flex;align-items:center;gap:14px;margin-bottom:14px}
.result-icon{
  width:40px;height:40px;border-radius:50%;
  background:linear-gradient(135deg,#F5C518 0%,#FFD93D 100%);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  box-shadow:0 4px 16px var(--accent-glow),inset 0 1px 0 rgba(255,255,255,.4);
}
.result-icon::after{content:'';width:12px;height:7px;border-left:2.5px solid var(--accent-dk);border-bottom:2.5px solid var(--accent-dk);transform:rotate(-45deg) translate(1px,-1px)}
.result-card h3{font-size:18px;font-weight:600;letter-spacing:-.01em}
.result-meta{color:var(--text2);font-size:13px;margin-bottom:18px;margin-left:54px;line-height:1.5}
.result-actions{display:flex;gap:10px;margin-left:54px;flex-wrap:wrap}
.result-actions a{
  background:linear-gradient(135deg,#F5C518 0%,#FFD93D 100%);
  color:var(--accent-dk);text-decoration:none;
  padding:10px 20px;border-radius:var(--rs);font-size:13px;font-weight:600;
  transition:all var(--t-out);letter-spacing:.02em;position:relative;overflow:hidden;
  box-shadow:0 2px 8px rgba(245,197,24,.25);
}
.result-actions a:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(245,197,24,.4)}
.result-actions a.secondary{background:var(--surface2);border:1px solid var(--border2);color:var(--text);font-weight:500;box-shadow:var(--sh-1)}
.result-actions a.secondary:hover{background:var(--bg2);border-color:var(--border2);box-shadow:var(--sh-2)}

/* ---- Error ---- */
.error-card{
  background:rgba(255,59,48,.04);border:1px solid rgba(255,59,48,.18);
  border-radius:var(--r);padding:22px;margin-top:24px;
  animation:scaleIn .4s ease;
}
.error-card h3{color:var(--error);font-size:15px;margin-bottom:8px;font-weight:600}
.error-card p{color:var(--text2);font-size:13px;word-break:break-word;line-height:1.6}

/* ---- History · 扑克牌堆叠 ---- */
.history{margin-top:64px;content-visibility:auto;contain-intrinsic-size:0 400px}
.history-head{display:flex;align-items:center;gap:10px;margin-bottom:24px}
.history-head h2{
  font-family:var(--font-display);font-size:20px;font-weight:400;
  color:var(--text2);letter-spacing:-.01em;font-style:italic;
}
.history-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}

/* 扑克牌容器 */
.hist-deck{
  position:relative;min-height:220px;
  display:flex;justify-content:center;align-items:flex-start;
  perspective:1200px;
}
/* 单张扑克牌 */
.hist-card{
  position:absolute;top:0;left:50%;
  width:260px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:16px;overflow:hidden;text-decoration:none;color:inherit;
  box-shadow:var(--sh-2);
  transform-origin:bottom center;
  transition:transform .6s cubic-bezier(.34,1.56,.64,1),box-shadow .3s ease,opacity .5s ease,filter .25s ease;
  cursor:pointer;
  /* 默认堆叠状态 */
  opacity:0;
  transform:translateX(-50%) translateY(0) rotate(0deg) scale(1);
}
/* 堆叠态：卡片叠在一起 */
.hist-deck:not(.dealt) .hist-card{
  opacity:0;
  transform:translateX(-50%) translateY(-20px) rotate(0deg) scale(.92);
}
.hist-deck:not(.dealt) .hist-card:nth-child(1){
  opacity:1;transform:translateX(-50%) translateY(0) rotate(0deg) scale(1);
}
/* 发牌后：扇形展开 */
.hist-deck.dealt .hist-card{
  opacity:1;
}
/* 每张牌的展开位置由 JS 内联设置 --tx / --rot */
.hist-deck.dealt .hist-card{
  transform:translateX(calc(-50% + var(--tx,0px))) translateY(var(--ty,0px)) rotate(var(--rot,0deg));
}
/* hover：牌浮起 */
.hist-card:hover{
  z-index:50 !important;
  transform:translateX(calc(-50% + var(--tx,0px))) translateY(calc(var(--ty,0px) - 16px)) rotate(0deg) scale(1.05) !important;
  box-shadow:var(--sh-4);
  border-color:var(--accent);
}
.hist-card:hover .hist-cover{transform:scale(1.08)}
/* hover 时黄色光带 */
.hist-cover-wrap::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent),#FFD93D);
  transform:scaleX(0);transform-origin:left;
  transition:transform .4s cubic-bezier(.16,1,.3,1);z-index:1;
}
.hist-card:hover .hist-cover-wrap::before{transform:scaleX(1)}
/* 点击闪光反馈 */
.hist-card.card-flash{
  filter:brightness(1.3) drop-shadow(0 0 24px rgba(245,197,24,.55));
}
/* 封面渐变叠加 */
.hist-cover-wrap::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(180deg,transparent 60%,rgba(0,0,0,.15) 100%);
  pointer-events:none;
}
.hist-cover-wrap{position:relative;overflow:hidden}
.hist-cover{width:100%;aspect-ratio:16/9;object-fit:cover;background:var(--bg2);display:block;transition:transform .6s cubic-bezier(.16,1,.3,1)}
.hist-cover-placeholder{width:100%;aspect-ratio:16/9;background:var(--bg2);display:flex;align-items:center;justify-content:center}
.hist-cover-placeholder::after{content:'';width:24px;height:24px;border:1.5px solid var(--text3);border-radius:50%;opacity:.3}
.hist-info{padding:13px 15px}
.hist-info h4{font-size:13px;font-weight:600;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;letter-spacing:-.01em}
.hist-info p{font-size:11px;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-empty{
  color:var(--text3);padding:24px 0;text-align:center;
  font-family:var(--font-display);font-style:italic;font-size:15px;
}
/* 扑克牌牌背装饰（堆叠时露出的边缘） */
.hist-deck::before{
  content:'';position:absolute;top:-6px;left:50%;
  transform:translateX(-50%);
  width:250px;height:12px;
  background:linear-gradient(180deg,var(--bg2),var(--border));
  border-radius:12px 12px 0 0;
  opacity:0;transition:opacity .4s ease;
}
.hist-deck:not(.dealt)::before{opacity:.5}
/* 提示文字 */
.hist-deck-hint{
  position:absolute;bottom:-32px;left:50%;transform:translateX(-50%);
  font-family:var(--font-display);font-style:italic;font-size:12px;
  color:var(--text3);white-space:nowrap;
  opacity:0;transition:opacity .4s ease;
}
.hist-deck:not(.dealt) .hist-deck-hint{opacity:.6}

/* 历史笔记改为稳定网格：避免牌组重叠后点击区域互相遮挡 */
.hist-deck{
  min-height:0;
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:16px;
  justify-content:stretch;
  align-items:stretch;
  perspective:none;
}
.hist-card,
.hist-deck:not(.dealt) .hist-card,
.hist-deck:not(.dealt) .hist-card:nth-child(1),
.hist-deck.dealt .hist-card{
  position:relative;
  top:auto;
  left:auto;
  width:100%;
  opacity:1;
  transform:none !important;
}
.hist-card:hover{
  transform:translateY(-4px) !important;
}
.hist-deck::before,
.hist-deck-hint{
  display:none !important;
}

.history{
  margin-top:70px;
}
.history-head{
  align-items:flex-start;
  justify-content:space-between;
  gap:18px;
  margin-bottom:18px;
  flex-wrap:wrap;
}
.history-title-row{
  display:flex;
  align-items:center;
  gap:10px;
  min-width:220px;
  flex:1;
}
.history-tools{
  display:flex;
  align-items:center;
  justify-content:flex-end;
  gap:10px;
  flex-wrap:wrap;
}
.history-search-wrap{
  position:relative;
}
.history-search-wrap::before{
  content:'';
  position:absolute;
  left:12px;
  top:50%;
  width:12px;
  height:12px;
  border:1.5px solid var(--text3);
  border-radius:50%;
  transform:translateY(-55%);
  opacity:.72;
  pointer-events:none;
}
.history-search-wrap::after{
  content:'';
  position:absolute;
  left:23px;
  top:24px;
  width:7px;
  height:1.5px;
  background:var(--text3);
  transform:rotate(45deg);
  opacity:.72;
  pointer-events:none;
}
.history-search{
  width:min(280px,70vw);
  height:38px;
  border:1px solid var(--border);
  border-radius:10px;
  background:var(--surface);
  color:var(--text);
  padding:0 12px 0 34px;
  font:inherit;
  font-size:13px;
  outline:none;
  box-shadow:var(--sh-1);
  transition:border-color var(--t-out),box-shadow var(--t-out),background var(--t-out);
}
.history-search:focus{
  border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-glow);
}
/* 让 native select 及其下拉菜单跟随主题，避免白底白字 / 深底黑字 */
select{color-scheme:inherit}
select option{background:var(--surface);color:var(--text)}
.history-filter{
  height:38px;
  min-width:104px;
  border:1px solid var(--border);
  border-radius:10px;
  background:var(--surface);
  color:var(--text2);
  padding:0 28px 0 11px;
  font:inherit;
  font-size:12px;
  outline:none;
  box-shadow:var(--sh-1);
  cursor:pointer;
  transition:border-color var(--t-out),box-shadow var(--t-out),transform var(--t-out);
}
.history-filter:focus{
  border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-glow);
}
.history-filter:hover{
  transform:translateY(-1px);
}
.history-stats{
  min-height:38px;
  display:flex;
  align-items:center;
  color:var(--text2);
  font-size:12px;
  white-space:nowrap;
}
.hist-card{
  display:flex;
  flex-direction:column;
  border-radius:12px;
  overflow:hidden;
}
.hist-card:focus-within{
  border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-glow),var(--sh-3);
}
.hist-main{
  display:block;
  color:inherit;
  text-decoration:none;
  min-width:0;
}
.hist-cover-wrap{
  border-bottom:1px solid var(--border);
}
.hist-info{
  padding:13px 14px 11px;
}
.hist-info h4{
  white-space:normal;
  display:-webkit-box;
  -webkit-line-clamp:2;
  -webkit-box-orient:vertical;
  min-height:38px;
  line-height:1.45;
}
.hist-meta{
  min-height:18px;
}
.hist-badges{
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  margin-top:9px;
}
.hist-badges span{
  display:inline-flex;
  align-items:center;
  max-width:100%;
  min-height:22px;
  padding:2px 8px;
  border-radius:999px;
  background:var(--surface2);
  border:1px solid var(--border);
  color:var(--text2);
  font-size:11px;
  line-height:1.2;
}
.hist-badges .hist-badge-strong{
  background:var(--accent-soft);
  border-color:rgba(245,197,24,.42);
  color:var(--accent-dk);
  font-weight:600;
}
.hist-actions{
  display:flex;
  gap:8px;
  padding:0 14px 14px;
  margin-top:auto;
  flex-wrap:wrap;
}
.hist-actions a{
  min-height:30px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding:0 10px;
  border-radius:8px;
  background:var(--accent);
  color:var(--accent-dk);
  text-decoration:none;
  font-size:12px;
  font-weight:600;
  border:1px solid transparent;
  transition:transform var(--t-out),box-shadow var(--t-out),background var(--t-out),border-color var(--t-out);
}
.hist-actions a:hover{
  transform:translateY(-1px);
  box-shadow:0 6px 18px rgba(245,197,24,.25);
}
.hist-actions a.secondary{
  background:var(--surface2);
  color:var(--text2);
  border-color:var(--border);
  font-weight:500;
}
.hist-actions a.secondary:hover{
  color:var(--text);
  border-color:var(--border2);
  box-shadow:var(--sh-2);
}
.hist-empty{
  grid-column:1/-1;
  background:var(--surface);
  border:1px dashed var(--border2);
  border-radius:12px;
}
@media (max-width:720px){
  .history-head{display:block}
  .history-title-row{margin-bottom:12px}
  .history-tools{justify-content:stretch}
  .history-search-wrap,.history-search{width:100%}
  .history-filter{flex:1;min-width:0}
  .history-stats{width:100%;justify-content:flex-start}
}

/* ---- Film Workbench history: stable, non-overlapping replacement for the old poker deck ---- */
.url-input.recognized{
  border-color:var(--success);
  background:linear-gradient(180deg,#fff,#f8fff9);
  box-shadow:0 0 0 3px rgba(46,204,113,.12),inset 0 1px 3px rgba(0,0,0,.03);
  animation:inputAccepted .72s cubic-bezier(.16,1,.3,1);
}
.gen-btn.is-loading{
  color:rgba(26,26,26,.82);
  box-shadow:0 10px 32px rgba(245,197,24,.36),inset 0 1px 0 rgba(255,255,255,.5);
}
.gen-btn.is-loading::after{
  content:'';
  position:absolute;
  inset:0;
  background:
    linear-gradient(90deg,transparent,rgba(255,255,255,.55),transparent),
    repeating-linear-gradient(90deg,rgba(26,26,26,.08) 0 1px,transparent 1px 10px);
  transform:translateX(-110%);
  animation:buttonScan 1.35s cubic-bezier(.65,0,.35,1) infinite;
  pointer-events:none;
}
.history{
  position:relative;
  padding:2px 0 8px;
}
.history::before{
  content:'';
  position:absolute;
  left:-20px;
  right:-20px;
  top:56px;
  height:1px;
  background:linear-gradient(90deg,transparent,var(--accent),var(--cool),transparent);
  opacity:.38;
  transform-origin:left center;
}
.history-line{
  position:relative;
  overflow:hidden;
}
.history-line::after{
  content:'';
  position:absolute;
  inset:0 auto 0 -42%;
  width:38%;
  background:linear-gradient(90deg,transparent,var(--cool),var(--accent),transparent);
  animation:railScan 4.8s cubic-bezier(.65,0,.35,1) infinite;
  opacity:.8;
}
.history-stats{
  padding:0 10px;
  border:1px solid rgba(42,183,202,.16);
  border-radius:999px;
  background:linear-gradient(180deg,rgba(255,255,255,.78),rgba(231,251,253,.55));
}
.hist-deck,
.hist-deck.dealt{
  position:relative;
  grid-template-columns:repeat(auto-fit,minmax(258px,1fr));
  gap:18px;
  perspective:1100px;
  isolation:isolate;
}
.hist-deck.film-ready::before{
  content:'';
  display:block !important;
  position:absolute;
  inset:-12px -12px auto -12px;
  height:6px;
  width:auto;
  transform:none;
  border-radius:999px;
  opacity:.72;
  background:
    radial-gradient(circle,rgba(26,26,26,.22) 0 2px,transparent 2.5px) 0 0/18px 6px repeat-x,
    linear-gradient(90deg,transparent,rgba(245,197,24,.38),rgba(42,183,202,.22),transparent);
  pointer-events:none;
}
.hist-card,
.hist-deck:not(.dealt) .hist-card,
.hist-deck.dealt .hist-card{
  --rx:0deg;
  --ry:0deg;
  --mx:50%;
  --my:22%;
  --enter-y:18px;
  --enter-scale:.985;
  --lift:0px;
  opacity:0;
  border-radius:14px;
  border-color:rgba(26,26,26,.08);
  background:linear-gradient(180deg,rgba(255,255,255,.96),rgba(255,255,255,.88));
  box-shadow:0 8px 24px rgba(20,24,28,.07),0 1px 2px rgba(20,24,28,.06);
  transform:perspective(1100px) translateY(calc(var(--enter-y) + var(--lift))) scale(var(--enter-scale)) rotateX(var(--rx)) rotateY(var(--ry)) !important;
  transform-style:preserve-3d;
  transition:
    opacity .46s cubic-bezier(.16,1,.3,1),
    transform .52s cubic-bezier(.16,1,.3,1),
    box-shadow .32s cubic-bezier(.16,1,.3,1),
    border-color .32s cubic-bezier(.16,1,.3,1),
    background .32s cubic-bezier(.16,1,.3,1);
  will-change:transform,opacity;
  contain:paint;
}
.hist-card.is-in{
  --enter-y:0px;
  --enter-scale:1;
  opacity:1;
}
.hist-deck.dealt .hist-card.is-in{
  --enter-y:0px;
  --enter-scale:1;
  opacity:1;
}
.hist-card::after{
  content:'';
  position:absolute;
  inset:0;
  z-index:3;
  pointer-events:none;
  opacity:0;
  background:
    radial-gradient(circle at var(--mx) var(--my),rgba(255,255,255,.62),rgba(255,255,255,.20) 18%,transparent 38%),
    linear-gradient(135deg,transparent 0 42%,rgba(42,183,202,.10) 48%,transparent 56%);
  transition:opacity .28s cubic-bezier(.16,1,.3,1);
}
.hist-card:hover,
.hist-card.pointer-active{
  --lift:-7px;
  z-index:4 !important;
  border-color:rgba(245,197,24,.58);
  background:linear-gradient(180deg,#fff,rgba(255,255,255,.92));
  box-shadow:0 22px 54px rgba(20,24,28,.13),0 10px 26px rgba(245,197,24,.15);
}
.hist-deck.dealt .hist-card:hover,
.hist-deck.dealt .hist-card.pointer-active{
  transform:perspective(1100px) translateY(calc(var(--enter-y) + var(--lift))) scale(var(--enter-scale)) rotateX(var(--rx)) rotateY(var(--ry)) !important;
}
.hist-card:hover::after,
.hist-card.pointer-active::after{
  opacity:1;
}
.hist-card.card-flash{
  filter:none;
  animation:historyCardPulse .72s cubic-bezier(.16,1,.3,1);
}
.hist-cover-wrap{
  background:linear-gradient(180deg,var(--ink-soft),#fff);
}
.hist-cover-wrap::before{
  inset:0;
  height:auto;
  background:linear-gradient(105deg,transparent 0 38%,rgba(255,255,255,.62) 50%,transparent 62%);
  transform:translateX(-120%);
  transition:transform .82s cubic-bezier(.65,0,.35,1);
  z-index:2;
}
.hist-card:hover .hist-cover-wrap::before,
.hist-card.pointer-active .hist-cover-wrap::before{
  transform:translateX(120%);
}
.hist-cover-wrap::after{
  background:
    radial-gradient(circle,rgba(255,255,255,.92) 0 1.8px,transparent 2.4px) 8px 8px/18px 8px repeat-x,
    radial-gradient(circle,rgba(255,255,255,.72) 0 1.8px,transparent 2.4px) 8px calc(100% - 14px)/18px 8px repeat-x,
    linear-gradient(180deg,transparent 58%,rgba(0,0,0,.18) 100%);
}
.hist-cover{
  filter:saturate(1.03) contrast(1.02);
  transform:scale(1);
}
.hist-card:hover .hist-cover,
.hist-card.pointer-active .hist-cover{
  transform:scale(1.055);
}
.hist-info{
  position:relative;
  z-index:1;
}
.hist-info h4{
  font-size:13.5px;
}
.hist-meta{
  color:#6d747c;
}
.hist-badges span:first-child{
  background:var(--cool-soft);
  border-color:rgba(42,183,202,.22);
  color:#237f8c;
}
.hist-actions{
  position:relative;
  z-index:4;
}
.hist-actions a{
  border-radius:9px;
}
.hist-actions a.secondary{
  background:rgba(247,247,247,.76);
}
.hist-empty{
  min-height:120px;
  display:flex;
  align-items:center;
  justify-content:center;
}
@keyframes railScan{
  0%,22%{transform:translateX(0);opacity:0}
  34%{opacity:.85}
  68%{opacity:.85}
  86%,100%{transform:translateX(370%);opacity:0}
}
@keyframes buttonScan{
  0%{transform:translateX(-110%)}
  58%,100%{transform:translateX(115%)}
}
@keyframes inputAccepted{
  0%{transform:translateY(0)}
  32%{transform:translateY(-1px)}
  100%{transform:translateY(0)}
}
@keyframes historyCardPulse{
  0%{box-shadow:0 22px 54px rgba(20,24,28,.13),0 0 0 0 rgba(245,197,24,.42)}
  50%{box-shadow:0 22px 54px rgba(20,24,28,.13),0 0 0 7px rgba(245,197,24,.10)}
  100%{box-shadow:0 8px 24px rgba(20,24,28,.07),0 0 0 0 rgba(245,197,24,0)}
}

/* ---- Notes library polish: unified controls and calmer motion ---- */
.history{
  --history-control-h:44px;
  width:min(920px,calc(100vw - 48px));
  left:50%;
  margin-left:0;
  margin-top:74px;
  padding-bottom:14px;
  transform:translateX(-50%);
}
.history::before{display:none}
.history-head{
  display:block;
  margin-bottom:12px;
}
.history-title-row{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:18px;
  min-width:0;
}
.history-title-row h2{
  font-size:24px;
  font-weight:500;
  letter-spacing:0;
  color:var(--text);
}
.history-subtitle{
  margin-top:2px;
  color:var(--text3);
  font-size:12px;
  letter-spacing:0;
}
.history-stats{
  min-height:32px;
  height:32px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding:0 12px;
  border:1px solid rgba(42,183,202,.18);
  border-radius:999px;
  background:linear-gradient(180deg,rgba(255,255,255,.92),rgba(231,251,253,.58));
  color:#406068;
  font-size:12px;
  letter-spacing:0;
  white-space:nowrap;
  box-shadow:0 1px 2px rgba(20,24,28,.04);
}
.history-toolbar{
  display:grid;
  grid-template-columns:minmax(260px,1fr) auto;
  align-items:center;
  gap:12px;
  margin-bottom:18px;
  padding:10px;
  border:1px solid rgba(26,26,26,.08);
  border-radius:16px;
  background:linear-gradient(180deg,rgba(255,255,255,.78),rgba(255,255,255,.58));
  box-shadow:0 10px 34px rgba(20,24,28,.07),inset 0 1px 0 rgba(255,255,255,.72);
  backdrop-filter:blur(16px) saturate(1.35);
  -webkit-backdrop-filter:blur(16px) saturate(1.35);
}
.history-filter-group{
  display:flex;
  align-items:center;
  justify-content:flex-end;
  gap:8px;
  flex-wrap:wrap;
}
.history-control{
  height:var(--history-control-h);
  display:flex;
  align-items:center;
  min-width:0;
  border:1px solid rgba(26,26,26,.09);
  border-radius:12px;
  background:linear-gradient(180deg,#fff,rgba(250,250,250,.86));
  box-shadow:0 1px 2px rgba(20,24,28,.04);
  transition:border-color var(--t-out),box-shadow var(--t-out),background var(--t-out),transform var(--t-out);
}
.history-control:hover{
  border-color:rgba(26,26,26,.16);
  background:#fff;
}
.history-control:focus-within{
  border-color:rgba(245,197,24,.82);
  box-shadow:0 0 0 3px rgba(245,197,24,.22),0 8px 22px rgba(20,24,28,.06);
  background:#fff;
}
.history-search-wrap{
  position:relative;
  width:100%;
}
.history-search-wrap::before{
  left:15px;
  width:13px;
  height:13px;
  border-color:#7a8289;
  opacity:.76;
}
.history-search-wrap::after{
  left:26px;
  top:26px;
  width:7px;
  background:#7a8289;
  opacity:.76;
}
.history-search{
  width:100%;
  height:100%;
  border:0;
  border-radius:12px;
  background:transparent;
  box-shadow:none;
  padding:0 14px 0 40px;
  color:var(--text);
  font-size:13px;
  letter-spacing:0;
  outline:none;
}
.history-search:focus{
  box-shadow:none;
}
.history-select-wrap{
  position:relative;
  gap:8px;
  min-width:136px;
  padding:0 12px;
}
.history-select-wrap>span{
  color:var(--text3);
  font-size:11px;
  letter-spacing:0;
  white-space:nowrap;
}
.history-select-wrap::after{
  content:'';
  position:absolute;
  right:12px;
  top:50%;
  width:7px;
  height:7px;
  border-right:1.5px solid #6d747c;
  border-bottom:1.5px solid #6d747c;
  transform:translateY(-68%) rotate(45deg);
  pointer-events:none;
}
.history-filter{
  height:100%;
  min-width:0;
  flex:1;
  border:0;
  border-radius:0;
  background:transparent;
  box-shadow:none;
  color:var(--text);
  padding:0 18px 0 0;
  font:inherit;
  font-size:13px;
  letter-spacing:0;
  outline:none;
  cursor:pointer;
  appearance:none;
  -webkit-appearance:none;
}
.history-filter:hover{transform:none}
.history-filter:focus{box-shadow:none}
.history-reset{
  height:var(--history-control-h);
  padding:0 14px;
  border:1px solid rgba(26,26,26,.08);
  border-radius:12px;
  background:rgba(247,247,247,.86);
  color:var(--text2);
  font:inherit;
  font-size:12px;
  letter-spacing:0;
  cursor:pointer;
  transition:background var(--t-out),border-color var(--t-out),color var(--t-out),transform var(--t-out);
}
.history-reset:hover{
  background:#fff;
  border-color:rgba(245,197,24,.55);
  color:var(--text);
  transform:translateY(-1px);
}
.history-reset[hidden]{display:none}
.hist-deck,
.hist-deck.dealt{
  grid-template-columns:repeat(auto-fill,minmax(272px,1fr));
  gap:16px;
}
.hist-deck.film-ready::before{
  opacity:.38;
  height:4px;
}
.hist-card,
.hist-deck:not(.dealt) .hist-card,
.hist-deck.dealt .hist-card{
  border-radius:13px;
  border:1px solid rgba(26,26,26,.08);
  background:rgba(255,255,255,.94);
  box-shadow:0 8px 24px rgba(20,24,28,.065),0 1px 2px rgba(20,24,28,.05);
}
.hist-card:hover,
.hist-card.pointer-active{
  --lift:-5px;
  border-color:rgba(245,197,24,.52);
  box-shadow:0 18px 44px rgba(20,24,28,.12),0 8px 20px rgba(245,197,24,.12);
}
.hist-info{
  padding:14px 15px 10px;
}
.hist-info h4{
  display:-webkit-box;
  min-height:40px;
  margin-bottom:6px;
  overflow:hidden;
  -webkit-line-clamp:2;
  -webkit-box-orient:vertical;
  white-space:normal;
  text-overflow:clip;
  line-height:1.48;
  letter-spacing:0;
}
.hist-meta{
  font-size:11.5px;
  letter-spacing:0;
}
.hist-badges{
  margin-top:10px;
  gap:6px;
}
.hist-badges span{
  min-height:23px;
  display:inline-flex;
  align-items:center;
  padding:0 8px;
  border-radius:999px;
  font-size:11px;
  letter-spacing:0;
}
.hist-card.quality-bad{
  border-color:rgba(255,59,48,.36);
  background:linear-gradient(180deg,rgba(255,255,255,.96),rgba(255,246,245,.92));
}
.hist-card.quality-check{
  border-color:rgba(245,197,24,.36);
}
.hist-badges .hist-quality.ok{
  background:#e9f9ef;
  color:#166534;
}
.hist-badges .hist-quality.check{
  background:#fff8e1;
  color:#8a5b00;
}
.hist-badges .hist-quality.bad{
  background:#fff0ef;
  color:#b42318;
}
.hist-actions{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:7px;
  padding:0 14px 14px;
  margin-top:auto;
}
.hist-actions a, .hist-actions button{
  height:34px;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:0 8px;
  border-radius:10px;
  font-size:12px;
  letter-spacing:0;
  white-space:nowrap;
}
.hist-actions button{
  border:0;
  cursor:pointer;
  font-family:inherit;
  color:var(--text2);
}
.hist-actions a:first-child, .hist-actions button:first-child{
  background:linear-gradient(135deg,#F5C518 0%,#FFD93D 100%);
  color:var(--accent-dk);
}
.hist-actions a.secondary, .hist-actions button.secondary{
  background:rgba(247,247,247,.82);
  color:#444;
}
.hist-actions button.secondary:hover{background:rgba(232,232,232,.9)}
.hist-actions button.secondary.warn{background:rgba(255,140,0,.16);color:#a04e00}
.hist-actions button.secondary.warn:hover{background:rgba(255,140,0,.26)}
.hist-actions button.danger{
  background:rgba(255,83,72,.14);
  color:#c0392b;
}
.hist-actions button.danger:hover{background:rgba(255,83,72,.24)}
/* ---- 卡片更多菜单（⋯）：高斯模糊 + 拟物化弹层 ---- */
.hist-card{position:relative}
.hist-card-more{
  position:absolute;top:10px;right:10px;z-index:3;
  width:30px;height:30px;border-radius:999px;
  display:flex;align-items:center;justify-content:center;
  border:1px solid rgba(26,26,26,.1);
  background:rgba(255,255,255,.92);color:var(--text2);
  font-size:15px;line-height:1;cursor:pointer;
  box-shadow:var(--sh-1);
  backdrop-filter:blur(4px);
  opacity:0;transform:scale(.5);pointer-events:none;
  transition:opacity var(--t-out),transform var(--t-spring),background var(--t-out),box-shadow var(--t-out);
}
.hist-card:hover .hist-card-more,
.hist-card:focus-within .hist-card-more,
.hist-card.menu-open .hist-card-more{
  opacity:1;transform:scale(1);pointer-events:auto;
}
.hist-card-more:hover{background:#fff;box-shadow:var(--sh-2)}
.hist-card.menu-open .hist-main,
.hist-card.menu-open .hist-actions{filter:blur(6px);pointer-events:none}
.hist-card-menu{
  position:absolute;inset:0;z-index:4;
  display:flex;flex-direction:column;gap:10px;
  align-items:center;justify-content:center;padding:16px;
  opacity:0;pointer-events:none;
  transition:opacity var(--t-out);
}
.hist-card.menu-open .hist-card-menu{opacity:1;pointer-events:auto}
.menu-btn{
  min-width:120px;padding:11px 18px;border-radius:14px;
  border:1px solid rgba(26,26,26,.12);
  background:linear-gradient(180deg,#fff 0%,#efefef 100%);
  color:var(--text);font:inherit;font-size:14px;font-weight:600;letter-spacing:0;cursor:pointer;
  box-shadow:0 2px 5px rgba(0,0,0,.06),inset 0 1px 0 rgba(255,255,255,.7);
  transition:transform var(--t-out),box-shadow var(--t-out);
}
.menu-btn:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(0,0,0,.1),inset 0 1px 0 rgba(255,255,255,.7)}
.menu-btn:active{transform:translateY(0)}
.menu-btn.danger{
  background:linear-gradient(180deg,#ffefee 0%,#ffdedb 100%);
  border-color:rgba(255,83,72,.3);color:#b42318;
}
/* ---- 置顶栏 ---- */
.pin-bar{
  display:flex;align-items:center;gap:12px;
  margin:0 0 18px;padding:12px 14px;
  border:1px solid var(--border);border-radius:var(--rs);
  background:linear-gradient(180deg,var(--surface) 0%,var(--surface2) 100%);
  box-shadow:var(--sh-1);
}
.pin-bar.hidden{display:none}
.pin-bar-label{flex:0 0 auto;font-size:12px;font-weight:600;color:var(--text3);border-right:1px solid var(--border);padding-right:12px;letter-spacing:.02em;white-space:nowrap}
.pin-track{display:flex;gap:10px;overflow-x:auto;flex:1;padding-bottom:2px;scrollbar-width:thin}
.pin-card{position:relative;flex:0 0 auto;width:150px;border-radius:12px;overflow:hidden;border:1px solid var(--border);background:var(--surface);box-shadow:var(--sh-1);transition:transform var(--t-out),box-shadow var(--t-out)}
.pin-card:hover{transform:translateY(-2px);box-shadow:var(--sh-2)}
.pin-card-link{display:block;text-decoration:none;color:var(--text)}
.pin-cover{display:block;width:100%;height:70px;object-fit:cover;background:linear-gradient(135deg,var(--surface2),var(--bg2))}
.pin-cover.ph{display:flex;align-items:center;justify-content:center;color:var(--text3)}
.pin-title{display:block;padding:7px 9px 8px;font-size:12px;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pin-card-unpin{position:absolute;top:5px;right:5px;width:20px;height:20px;border:0;border-radius:999px;cursor:pointer;background:rgba(0,0,0,.45);color:#fff;font-size:11px;line-height:1;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity var(--t-out),background var(--t-out)}
.pin-card:hover .pin-card-unpin{opacity:1}
.pin-card-unpin:hover{background:rgba(0,0,0,.7)}
/* ---- Toast 通知 ---- */
.toast-box{
  position:fixed;top:16px;left:50%;transform:translateX(-50%);
  z-index:9999;display:flex;flex-direction:column;align-items:center;gap:8px;
  pointer-events:none;
}
.toast{
  max-width:min(520px,92vw);
  padding:10px 16px;border-radius:14px;
  background:rgba(30,33,39,.94);color:#fff;
  font-size:13px;line-height:1.45;letter-spacing:0;
  box-shadow:0 12px 34px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.1);
  opacity:0;transform:translateY(-10px) scale(.96);
  transition:opacity var(--t-out),transform var(--t-spring);
  backdrop-filter:blur(12px);
  pointer-events:none;
}
.toast.show{opacity:1;transform:translateY(0) scale(1)}
.toast.success{border-color:rgba(46,204,113,.5);box-shadow:0 12px 34px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.08),0 0 20px rgba(46,204,113,.25)}
.toast.error{border-color:rgba(255,59,48,.55);box-shadow:0 12px 34px rgba(0,0,0,.22),0 0 20px rgba(255,59,48,.25)}
.toast.info{border-color:rgba(72,199,216,.5);box-shadow:0 12px 34px rgba(0,0,0,.22),0 0 20px rgba(72,199,216,.28)}
html[data-theme="night"] .toast{background:rgba(20,22,27,.96)}
/* ---- 取消生成按钮 ---- */
.cancel-btn{
  display:block;margin:6px auto 0;padding:8px 20px;border-radius:999px;
  border:1px solid rgba(255,83,72,.4);
  background:rgba(255,83,72,.12);color:var(--error);
  font:inherit;font-size:13px;font-weight:600;cursor:pointer;
  transition:background var(--t-out),transform var(--t-out),box-shadow var(--t-out);
}
.cancel-btn:hover{background:rgba(255,83,72,.22);transform:translateY(-1px);box-shadow:0 6px 16px rgba(255,83,72,.18)}
.cancel-btn:disabled{opacity:.55;cursor:default;transform:none}
/* ---- 触屏/手机：⋯ 与 取消置顶 ✕ 常显，不依赖 hover ---- */
@media (hover: none){
  .hist-card-more{opacity:1;transform:scale(1);pointer-events:auto}
  .pin-card-unpin{opacity:1}
}
.hist-empty{
  border:1px dashed rgba(26,26,26,.12);
  border-radius:16px;
  background:rgba(255,255,255,.54);
  color:var(--text3);
}
.history-more{
  grid-column:1/-1;
  display:flex;
  align-items:center;
  justify-content:center;
  gap:12px;
  margin-top:4px;
  padding:10px 0 2px;
  color:var(--text3);
  font-size:12px;
}
.history-more button{
  height:38px;
  padding:0 18px;
  border:1px solid rgba(26,26,26,.1);
  border-radius:999px;
  background:linear-gradient(180deg,rgba(255,255,255,.92),rgba(247,247,247,.82));
  color:var(--text);
  font:inherit;
  font-size:13px;
  font-weight:600;
  cursor:pointer;
  box-shadow:var(--sh-1);
  transition:transform var(--t-out),box-shadow var(--t-out),border-color var(--t-out),background var(--t-out);
}
.history-more button:hover{
  transform:translateY(-1px);
  border-color:rgba(245,197,24,.54);
  box-shadow:0 8px 24px rgba(20,24,28,.08),0 6px 18px rgba(245,197,24,.16);
}
@media (max-width:860px){
  .history-toolbar{
    grid-template-columns:1fr;
  }
  .history-filter-group{
    justify-content:stretch;
  }
  .history-select-wrap{
    flex:1 1 150px;
  }
}
@media (max-width:640px){
  .history{
    margin-top:52px;
  }
  .history-title-row{
    display:block;
  }
  .history-stats{
    margin-top:10px;
  }
  .history-toolbar{
    padding:8px;
    border-radius:14px;
  }
  .history-filter-group{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:8px;
  }
  .history-select-wrap{
    min-width:0;
  }
  .history-select-wrap:nth-child(3),
  .history-reset{
    grid-column:1 / -1;
  }
  .hist-deck,.hist-deck.dealt{
    grid-template-columns:1fr;
    gap:14px;
  }
  .history::before{
    left:0;
    right:0;
  }
  .history-stats{
    border-radius:10px;
    padding:0 0;
    background:transparent;
    border-color:transparent;
  }
}

/* ---- Animations ---- */
@keyframes fadeInDown{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes scaleIn{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
@keyframes slideIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 var(--accent-glow)}50%{box-shadow:0 0 0 5px transparent}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
@keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-5px)}75%{transform:translateX(5px)}}
@keyframes breath{0%,100%{opacity:1}50%{opacity:.5}}
@keyframes dotPulse{0%,100%{transform:scale(1);opacity:.6}50%{transform:scale(1.4);opacity:1}}
@keyframes flow{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
/* 光环扩散 */
@keyframes ringExpand{
  0%{opacity:.6;transform:scale(.3)}
  100%{opacity:0;transform:scale(2)}
}
/* 光泽扫过 */
@keyframes sheen{
  0%,80%,100%{left:-100%}
  90%{left:120%}
}
/* 光锥微摆动 */
@keyframes coneSway{
  0%,100%{transform:translateX(-50%) rotate(0deg)}
  33%{transform:translateX(-50%) rotate(.4deg)}
  66%{transform:translateX(-50%) rotate(-.4deg)}
}
/* 聚光灯灯泡微闪 */
@keyframes bulbFlicker{
  0%,100%{opacity:1;box-shadow:0 0 40px 10px rgba(255,140,0,.7),0 0 100px 20px rgba(255,165,0,.35),0 0 160px 30px rgba(255,140,0,.12)}
  43%{opacity:.94;box-shadow:0 0 34px 9px rgba(255,140,0,.6),0 0 88px 18px rgba(255,165,0,.28),0 0 140px 26px rgba(255,140,0,.1)}
  47%{opacity:1;box-shadow:0 0 42px 11px rgba(255,140,0,.75),0 0 104px 22px rgba(255,165,0,.38),0 0 170px 32px rgba(255,140,0,.14)}
  51%{opacity:.97;box-shadow:0 0 38px 10px rgba(255,140,0,.66),0 0 96px 19px rgba(255,165,0,.32),0 0 150px 28px rgba(255,140,0,.11)}
}
/* 光锥呼吸（带位移修正） */
@keyframes coneBreath{
  0%,100%{opacity:1;scale:1}
  50%{opacity:.7;scale:1.03}
}
/* 小灯灯泡微闪（白光） */
@keyframes lampFlicker{
  0%,100%{opacity:1;box-shadow:0 0 28px 7px rgba(255,255,255,.7),0 0 60px 14px rgba(220,235,255,.3),0 0 100px 22px rgba(200,220,255,.12)}
  45%{opacity:.88;box-shadow:0 0 24px 6px rgba(255,255,255,.6),0 0 52px 12px rgba(220,235,255,.25),0 0 88px 20px rgba(200,220,255,.1)}
  50%{opacity:1;box-shadow:0 0 32px 8px rgba(255,255,255,.75),0 0 64px 16px rgba(220,235,255,.33),0 0 108px 24px rgba(200,220,255,.14)}
  55%{opacity:.93;box-shadow:0 0 26px 7px rgba(255,255,255,.65),0 0 56px 13px rgba(220,235,255,.28),0 0 94px 21px rgba(200,220,255,.11)}
}
/* 小灯光锥呼吸（不带位移，避免破坏 left 定位） */
@keyframes lampConeBreath{
  0%,100%{opacity:1;transform:scale(1)}
  50%{opacity:.7;transform:scale(1.03)}
}
/* 小灯地面光斑呼吸（仅透明度，不干扰 translateX） */
@keyframes lampSpotBreath{
  0%,100%{opacity:1}
  50%{opacity:.55}
}
/* 铭牌电源灯呼吸 */
@keyframes badgePulse{
  0%,100%{opacity:1;box-shadow:0 0 8px var(--accent),0 0 14px rgba(245,197,24,.5)}
  50%{opacity:.5;box-shadow:0 0 4px var(--accent),0 0 8px rgba(245,197,24,.3)}
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;transition:background var(--t)}
::-webkit-scrollbar-thumb:hover{background:var(--accent)}
.log-box::-webkit-scrollbar{width:3px}
.log-box::-webkit-scrollbar-track{background:transparent}
.log-box::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

/* ---- Selection ---- */
::selection{background:var(--accent);color:var(--accent-dk)}

/* ---- Focus visible (WCAG 2.4.13) ---- */
:focus-visible{
  outline:2px solid rgba(245,197,24,.72);
  outline-offset:3px;
  border-radius:4px;
  box-shadow:0 0 0 4px rgba(245,197,24,.16);
}
.url-input:focus-visible,.sg input:focus-visible,.sg select:focus-visible{
  outline:none;
  box-shadow:var(--focus-ring);
}
.gen-btn:focus-visible,
.result-actions a:focus-visible,
.hist-actions a:focus-visible,
.history-reset:focus-visible{
  outline:none;
  box-shadow:0 0 0 4px rgba(245,197,24,.24),0 8px 24px rgba(245,197,24,.32);
}
.history-filter:focus-visible,
.history-search:focus-visible{
  outline:none;
}

/* ---- Reduced motion (WCAG 2.3.3) ---- */
@media(prefers-reduced-motion:reduce){
  *,*::before,*::after{
    animation-duration:.01ms !important;
    animation-iteration-count:1 !important;
    transition-duration:.01ms !important;
    scroll-behavior:auto !important;
  }
}
/* ---- Skip link (WCAG 2.4.1) ---- */
.skip-link:focus{
  position:fixed !important;
  left:50% !important;
  transform:translateX(-50%);
  top:0;
}
/* ---- Forced colors (WCAG 1.4.12) ---- */
@media(forced-colors:active){
  .gen-btn,.result-actions a{
    border:1px solid ButtonText;
  }
}

/* ---- Reveal on scroll ---- */
.reveal-on-scroll{
  opacity:0;transform:translateY(20px);
  transition:opacity .8s cubic-bezier(.16,1,.3,1),transform .8s cubic-bezier(.16,1,.3,1);
}
.reveal-on-scroll.revealed{opacity:1;transform:translateY(0)}
.history.reveal-on-scroll{transform:translateX(-50%) translateY(20px)}
.history.reveal-on-scroll.revealed{transform:translateX(-50%) translateY(0)}

/* ---- Batch Toggle ---- */
.batch-toggle{
  display:flex;align-items:center;gap:10px;cursor:pointer;
  padding:10px 14px;background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--rs);transition:all var(--t);
}
.batch-toggle:hover{border-color:var(--border2);background:var(--bg2)}
.batch-toggle input{
  appearance:none;-webkit-appearance:none;width:18px;height:18px;
  border:1.5px solid var(--border2);border-radius:5px;
  background:transparent;cursor:pointer;position:relative;
  transition:all var(--t);flex-shrink:0;
}
.batch-toggle input:checked{background:var(--accent);border-color:var(--accent)}
.batch-toggle input:checked::after{
  content:'';position:absolute;top:2px;left:5px;width:5px;height:9px;
  border-right:2px solid var(--accent-dk);border-bottom:2px solid var(--accent-dk);
  transform:rotate(45deg);
}
.batch-label{font-size:13px;color:var(--text2);transition:color var(--t);letter-spacing:.01em}
.batch-toggle input:checked + .batch-label{color:var(--text);font-weight:500}

/* ---- Batch Progress ---- */
.batch-progress{margin-top:20px;animation:fadeInUp .4s ease}
.batch-bar{
  height:4px;background:var(--border);border-radius:2px;overflow:hidden;
  margin-bottom:12px;
}
.batch-bar-fill{
  height:100%;background:var(--accent);border-radius:2px;
  transition:width .5s cubic-bezier(.16,1,.3,1);
  box-shadow:0 0 12px var(--accent-glow);
}
.batch-info{
  display:flex;justify-content:space-between;align-items:center;
  font-size:13px;color:var(--text2);
}
.batch-info .current{
  font-family:var(--font-display);font-style:italic;font-size:16px;
  color:var(--text);
}
.batch-info .count{color:var(--text3);font-size:12px}

/* ---- Responsive ---- */
@media(max-width:640px){
  .app{padding:48px 16px 80px}
  .hero h1{font-size:52px}
  .input-row{flex-direction:column}
  .gen-btn{width:100%}
  .pipeline{flex-wrap:wrap;gap:12px;justify-content:center}
  .p-conn{display:none}
  .result-meta,.result-actions{margin-left:0}
}

/* ---- 电影院开场红帘（增强版） ---- */
/* ---- Refined edit-suite rig: thinner, flatter, less view-blocking ---- */
body{
  background:
    radial-gradient(ellipse 34% 24% at 50% 0%,rgba(245,197,24,.24) 0%,rgba(245,197,24,.10) 34%,transparent 72%),
    radial-gradient(ellipse 65% 38% at 50% 18%,rgba(255,255,255,.76) 0%,rgba(255,255,255,.34) 42%,transparent 76%),
    linear-gradient(180deg,#fbfbfb 0%,#fafafa 54%,var(--bg) 100%);
}
.spot-rig{
  height:44px;
  background:
    linear-gradient(180deg,#111 0%,#171717 66%,#202020 100%),
    repeating-linear-gradient(90deg,transparent 0,transparent 72px,rgba(255,255,255,.03) 72px,rgba(255,255,255,.03) 73px);
  border-bottom:1px solid rgba(255,255,255,.08);
  box-shadow:0 5px 14px rgba(0,0,0,.26),inset 0 -1px 0 rgba(255,255,255,.055);
}
.spot-rig::before{
  top:100%;
  width:162px;
  height:9px;
  margin-top:-1px;
  border-radius:0 0 18px 18px/0 0 8px 8px;
  background:
    linear-gradient(180deg,#292929 0%,#1e1e1e 58%,#151515 100%),
    linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
  border:1px solid rgba(255,255,255,.06);
  border-top:none;
  box-shadow:0 4px 10px rgba(0,0,0,.22),inset 0 -1px 0 rgba(245,197,24,.12);
}
.spot-rig::after{
  top:100%;
  width:112px;
  height:4px;
  margin-top:6px;
  background:linear-gradient(90deg,transparent 0%,rgba(255,226,124,.26) 12%,rgba(255,218,76,1) 50%,rgba(255,226,124,.26) 88%,transparent 100%);
  border-radius:999px;
  box-shadow:0 0 18px 4px rgba(245,197,24,.44),0 0 52px 12px rgba(245,197,24,.20),0 0 96px 22px rgba(245,197,24,.08);
  animation:softLightPulse 5.8s ease-in-out infinite;
}
.rig-wall{
  width:14px;
  height:44px;
  background:linear-gradient(180deg,#101010 0%,#181818 72%,#202020 100%);
  box-shadow:0 5px 14px rgba(0,0,0,.24);
}
.rig-wall::after{bottom:7px;width:9px;height:3px;opacity:.72}
.stage-vignette{width:4vw}
.stage-vignette-l{background:linear-gradient(90deg,rgba(0,0,0,.035) 0%,transparent 100%)}
.stage-vignette-r{background:linear-gradient(270deg,rgba(0,0,0,.035) 0%,transparent 100%)}
.rig-badge{
  transform:translateX(-50%) translateY(-50px);
  padding:5px 13px 5px 10px;
  border-color:rgba(255,255,255,.08);
  box-shadow:0 2px 7px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.06);
}
.rig-badge.at-top{transform:translateX(-50%) translateY(-32px);opacity:.18}
.rig-badge.pulled{transform:translateX(-50%) translateY(8px);opacity:.96}
.lamp-l2,.lamp-r1,.lamp-cone.l2,.lamp-cone.r1{display:none}
.lamp{
  top:100%;
  width:30px;
  height:9px;
  margin-top:-1px;
  border-radius:0 0 15px 15px/0 0 8px 8px;
  border:1px solid rgba(255,255,255,.06);
  border-top:none;
  background:linear-gradient(180deg,#252525 0%,#1a1a1a 62%,#111 100%);
  box-shadow:0 4px 11px rgba(0,0,0,.25),inset 0 -1px 3px rgba(255,255,255,.04);
}
.lamp::before{
  top:-3px;
  width:20px;
  height:3px;
  border-radius:3px 3px 0 0;
  background:linear-gradient(180deg,#202020 0%,#282828 100%);
  box-shadow:none;
}
.lamp::after{
  width:20px;
  height:3px;
  margin-top:-2px;
  background:radial-gradient(ellipse at center,#fff 0%,#f6fbff 18%,#cfdbe8 58%,transparent 82%);
  box-shadow:0 0 18px 5px rgba(255,255,255,.62),0 0 42px 12px rgba(210,230,255,.24);
  animation:precisionLampFlicker 6.4s ease-in-out infinite;
}
.lamp-l1{left:18%}
.lamp-r2{left:82%}
.lamp-cone{
  top:52px;
  width:124px;
  height:calc(100vh - 52px);
  background:radial-gradient(ellipse 26% 82% at 50% 0%,rgba(255,255,255,.024) 0%,rgba(235,245,255,.014) 36%,transparent 70%);
}
.lamp-cone::before{
  background:linear-gradient(180deg,rgba(255,255,255,.24) 0%,rgba(242,248,255,.12) 28%,rgba(232,242,255,.04) 58%,transparent 90%);
  clip-path:polygon(45% 0%,55% 0%,72% 100%,28% 100%);
  -webkit-clip-path:polygon(45% 0%,55% 0%,72% 100%,28% 100%);
  filter:blur(.4px);
  animation:precisionConeBreath 8s ease-in-out infinite;
}
.lamp-cone::after{
  bottom:16%;
  width:58px;
  height:12px;
  background:radial-gradient(ellipse at center,rgba(255,255,255,.18) 0%,rgba(235,245,255,.07) 45%,transparent 75%);
  filter:blur(5px);
}
.lamp-cone.l1{left:calc(18% + 15px - 62px)}
.lamp-cone.r2{left:calc(82% + 15px - 62px)}
.spot-cone{
  top:44px;
  width:min(560px,74vw);
  height:220px;
  background:radial-gradient(ellipse 38% 70% at 50% 0%,rgba(245,197,24,.20) 0%,rgba(245,197,24,.085) 34%,rgba(245,197,24,.026) 58%,transparent 82%);
  animation:softConeDrift 11s ease-in-out infinite;
}
.spot-core{
  top:44px;
  width:min(360px,48vw);
  height:150px;
  background:radial-gradient(ellipse 34% 56% at 50% 0%,rgba(255,216,92,.18) 0%,rgba(245,197,24,.07) 44%,transparent 72%);
  animation:softConeDrift 12s ease-in-out infinite reverse;
}
.app{padding-top:96px}
.hero{margin-bottom:34px}
.hero-mark{margin-bottom:12px;opacity:.86}
.hero h1{font-size:58px}
.hero h1::before{
  top:-18%;
  height:130%;
  background:radial-gradient(ellipse 48% 58% at 50% 50%,rgba(245,197,24,.08) 0%,transparent 70%);
}
.input-card{background:rgba(255,255,255,.78)}
@keyframes softLightPulse{
  0%,100%{opacity:.86;box-shadow:0 0 17px 4px rgba(245,197,24,.38),0 0 50px 11px rgba(245,197,24,.17)}
  50%{opacity:1;box-shadow:0 0 22px 5px rgba(245,197,24,.52),0 0 64px 14px rgba(245,197,24,.23)}
}
@keyframes precisionLampFlicker{
  0%,100%{opacity:.82;box-shadow:0 0 18px 5px rgba(255,255,255,.52),0 0 38px 10px rgba(210,230,255,.21)}
  50%{opacity:1;box-shadow:0 0 22px 6px rgba(255,255,255,.68),0 0 50px 13px rgba(210,230,255,.28)}
}
@keyframes precisionConeBreath{
  0%,100%{opacity:.72;transform:scaleY(1)}
  50%{opacity:.46;transform:scaleY(1.02)}
}
@keyframes softConeDrift{
  0%,100%{opacity:1;transform:translateX(-50%) translateY(0)}
  50%{opacity:.78;transform:translateX(-50%) translateY(3px)}
}
@media(max-width:640px){
  .spot-rig,.rig-wall{height:38px}
  .spot-rig::before{width:124px;height:8px}
  .spot-rig::after{width:80px;margin-top:5px}
  .lamp{width:24px;height:8px}
  .lamp::before{width:17px}
  .lamp::after{width:16px;height:3px}
  .lamp-l1{left:18%}
  .lamp-r2{left:82%}
  .lamp-cone{top:48px;width:92px;height:190px}
  .lamp-cone.l1{left:calc(18% + 12px - 46px)}
  .lamp-cone.r2{left:calc(82% + 12px - 46px)}
  .spot-cone{top:38px;height:185px;width:76vw}
  .spot-core{top:38px;height:130px;width:54vw}
  .app{padding-top:82px}
  .hero{margin-bottom:26px}
  .hero h1{font-size:50px}
}

html[data-theme="night"] body{
  background:
    radial-gradient(ellipse 34% 24% at 50% 0%,rgba(245,197,24,.16) 0%,rgba(245,197,24,.06) 34%,transparent 72%),
    radial-gradient(ellipse 62% 38% at 50% 18%,rgba(72,199,216,.08) 0%,rgba(72,199,216,.025) 42%,transparent 76%),
    linear-gradient(180deg,#0c0d10 0%,#111318 54%,var(--bg) 100%);
}
html[data-theme="night"] .spot-rig{
  background:
    linear-gradient(180deg,#040506 0%,#0c0e13 66%,#141821 100%),
    repeating-linear-gradient(90deg,transparent 0,transparent 72px,rgba(255,255,255,.035) 72px,rgba(255,255,255,.035) 73px);
  border-bottom-color:rgba(245,197,24,.12);
}
html[data-theme="night"] .spot-rig::after{
  background:linear-gradient(90deg,transparent 0%,rgba(255,226,124,.2) 12%,rgba(255,218,76,.86) 50%,rgba(255,226,124,.2) 88%,transparent 100%);
}
html[data-theme="night"] .rig-wall{
  background:linear-gradient(180deg,#040506 0%,#0d0f14 72%,#141821 100%);
}
html[data-theme="night"] .spot-cone{
  background:radial-gradient(ellipse 38% 70% at 50% 0%,rgba(245,197,24,.16) 0%,rgba(245,197,24,.06) 34%,rgba(245,197,24,.02) 58%,transparent 82%);
}
html[data-theme="night"] .spot-core{
  background:radial-gradient(ellipse 34% 56% at 50% 0%,rgba(255,216,92,.13) 0%,rgba(245,197,24,.05) 44%,transparent 72%);
}
html[data-theme="night"] .input-card,
html[data-theme="night"] .result-card{
  background:rgba(27,29,34,.76);
  border-color:rgba(255,255,255,.08);
  box-shadow:var(--sh-3),inset 0 1px 0 rgba(255,255,255,.06);
}
html[data-theme="night"] .input-card:focus-within{
  box-shadow:var(--sh-4),0 0 0 1px rgba(245,197,24,.2),0 0 34px rgba(245,197,24,.12);
}
html[data-theme="night"] .url-input,
html[data-theme="night"] .sg select,
html[data-theme="night"] .sg input{
  background:rgba(35,38,44,.72);
  border-color:rgba(255,255,255,.08);
}
html[data-theme="night"] .settings-row .settings-toggle,
html[data-theme="night"] .history-toolbar,
html[data-theme="night"] .history-control,
html[data-theme="night"] .history-reset,
html[data-theme="night"] .hist-actions a.secondary,
html[data-theme="night"] .hist-actions button.secondary,
html[data-theme="night"] .batch-toggle{
  background:rgba(35,38,44,.72);
  border-color:rgba(255,255,255,.08);
}
html[data-theme="night"] .hist-actions button.secondary{color:var(--text2)}
html[data-theme="night"] .hist-actions button.secondary:hover{background:rgba(45,49,57,.85)}
html[data-theme="night"] .hist-actions button.secondary.warn{background:rgba(255,140,0,.16);color:#ffb25e}
html[data-theme="night"] .hist-actions button.secondary.warn:hover{background:rgba(255,140,0,.28)}
html[data-theme="night"] .hist-actions button.danger{background:rgba(255,83,72,.18);color:#ff8a8a}
html[data-theme="night"] .hist-actions button.danger:hover{background:rgba(255,83,72,.28)}
html[data-theme="night"] .hist-card-more{background:rgba(35,38,44,.8);border-color:rgba(255,255,255,.12);color:var(--text2)}
html[data-theme="night"] .hist-card-more:hover{background:rgba(45,49,57,.9)}
html[data-theme="night"] .menu-btn{
  background:linear-gradient(180deg,#2a2e35 0%,#1b1e24 100%);
  border-color:rgba(255,255,255,.09);color:var(--text);
  box-shadow:0 2px 6px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.06);
}
html[data-theme="night"] .menu-btn:hover{box-shadow:0 6px 18px rgba(0,0,0,.42),inset 0 1px 0 rgba(255,255,255,.06)}
html[data-theme="night"] .menu-btn.danger{background:linear-gradient(180deg,#3a2224 0%,#291b1c 100%);border-color:rgba(255,83,72,.36);color:#ff8a8a}
html[data-theme="night"] .pin-bar{background:linear-gradient(180deg,var(--surface) 0%,var(--surface2) 100%)}
html[data-theme="night"] .pin-card{background:var(--surface);border-color:rgba(255,255,255,.08)}
html[data-theme="night"] .pin-cover{background:linear-gradient(135deg,var(--surface2),var(--bg2))}
html[data-theme="night"] .pin-card-link{color:var(--text)}
html[data-theme="night"] .settings,
html[data-theme="night"] .hist-card,
html[data-theme="night"] .hist-deck:not(.dealt) .hist-card,
html[data-theme="night"] .hist-deck.dealt .hist-card{
  background:linear-gradient(180deg,rgba(30,33,39,.94),rgba(24,26,31,.9));
  border-color:rgba(255,255,255,.08);
  box-shadow:0 10px 34px rgba(0,0,0,.34),0 1px 2px rgba(0,0,0,.28);
}
html[data-theme="night"] .hist-card.quality-bad{
  background:linear-gradient(180deg,rgba(44,29,30,.96),rgba(31,24,26,.92));
  border-color:rgba(255,83,72,.32);
}
html[data-theme="night"] .history-stats{
  background:rgba(35,38,44,.6);
  border-color:rgba(72,199,216,.18);
  color:var(--text2);
}
html[data-theme="night"] .history-more button{
  background:rgba(35,38,44,.8);
  border-color:rgba(255,255,255,.08);
  color:var(--text);
}
html[data-theme="night"] .settings-backdrop{
  background:rgba(3,5,8,.46);
}

.settings-backdrop{display:none!important}
body.settings-drawer-open{overflow:auto}
body:has(.settings.open) .theme-pull{
  z-index:560;
}
html[data-theme="night"] .settings{
  background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.018));
  border-top-color:rgba(255,255,255,.08);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.07);
}
html[data-theme="night"] .input-card.settings-open,
html[data-theme="night"] .input-card.settings-open:focus-within{
  border-color:rgba(245,197,24,.2);
  box-shadow:0 16px 44px rgba(0,0,0,.34),0 0 0 1px rgba(245,197,24,.1),inset 0 1px 0 rgba(255,255,255,.06);
}

/* ---- Final studio polish: symmetric side lamps, richer pull cord, curtain theme wipe ---- */
:root{
  --side-lamp-offset:min(32vw,470px);
  --side-lamp-half:16px;
  --side-cone-half:86px;
}
.lamp-l1,
.lamp-r2{
  width:32px;
}
.lamp-l1{left:calc(50% - var(--side-lamp-offset) - var(--side-lamp-half))}
.lamp-r2{left:calc(50% + var(--side-lamp-offset) - var(--side-lamp-half))}
.lamp-cone{
  --beam-shift:0px;
  top:50px;
  width:172px;
  height:calc(100vh - 50px);
  opacity:.86;
  mix-blend-mode:screen;
  background:transparent;
  filter:saturate(1.05);
}
.lamp-cone.l1{
  left:calc(50% - var(--side-lamp-offset) - var(--side-cone-half));
  transform:skewX(2.5deg);
  transform-origin:50% 0%;
}
.lamp-cone.r2{
  left:calc(50% + var(--side-lamp-offset) - var(--side-cone-half));
  transform:skewX(-2.5deg);
  transform-origin:50% 0%;
}
.lamp-cone::before{
  background:
    radial-gradient(ellipse 13% 8% at 50% 0%,rgba(255,255,255,.56) 0%,rgba(222,238,255,.24) 42%,transparent 78%),
    linear-gradient(180deg,
      rgba(236,246,255,.2) 0%,
      rgba(224,240,255,.12) 22%,
      rgba(214,232,252,.04) 56%,
      rgba(214,232,252,.012) 76%,
      transparent 94%);
  clip-path:polygon(48.5% 0%,51.5% 0%,66% 100%,34% 100%);
  -webkit-clip-path:polygon(48.5% 0%,51.5% 0%,66% 100%,34% 100%);
  filter:blur(2.5px);
  opacity:.5;
  mix-blend-mode:screen;
  mask-image:linear-gradient(180deg,#000 0%,rgba(0,0,0,.92) 22%,rgba(0,0,0,.34) 70%,transparent 100%);
  -webkit-mask-image:linear-gradient(180deg,#000 0%,rgba(0,0,0,.92) 22%,rgba(0,0,0,.34) 70%,transparent 100%);
  animation:refinedBeamBreath 7.8s ease-in-out infinite;
}
.lamp-cone::after{
  bottom:14%;
  width:86px;
  height:18px;
  opacity:.38;
  background:radial-gradient(ellipse at center,rgba(232,244,255,.28) 0%,rgba(216,234,255,.10) 46%,transparent 76%);
  filter:blur(7px);
  animation:refinedLampSpot 7.8s ease-in-out infinite;
}
.lamp-l1::after,
.lamp-r2::after{
  width:19px;
  height:3px;
  background:radial-gradient(ellipse at center,#fff 0%,#f8fbff 20%,#dbe6f2 58%,transparent 82%);
  box-shadow:0 0 16px 4px rgba(255,255,255,.58),0 0 38px 10px rgba(210,230,255,.22);
}
html[data-theme="night"] .lamp-l1::after,
html[data-theme="night"] .lamp-r2::after{
  box-shadow:0 0 22px 5px rgba(255,255,255,.64),0 0 58px 14px rgba(180,214,255,.28);
}
html[data-theme="night"] .lamp-cone{
  opacity:.9;
  background:transparent;
}
html[data-theme="night"] .lamp-cone::before{
  background:
    radial-gradient(ellipse 13% 8% at 50% 0%,rgba(255,255,255,.58) 0%,rgba(212,232,255,.26) 43%,transparent 80%),
    linear-gradient(180deg,
      rgba(224,240,255,.2) 0%,
      rgba(192,220,255,.11) 24%,
      rgba(162,200,255,.042) 58%,
      rgba(142,190,255,.014) 78%,
      transparent 95%);
  opacity:.48;
  filter:blur(3px);
}
html[data-theme="night"] .lamp-cone::after{
  opacity:.32;
}
.theme-pull{
  --cord-rest:46px;
  --pull-extra:0px;
  --pull-sway:0deg;
  --handle-squeeze:1;
  width:54px;
  right:36px;
}
.theme-pull::before{
  content:'';
  position:absolute;
  top:5px;
  right:18px;
  width:18px;
  height:8px;
  border-radius:7px 7px 3px 3px;
  background:linear-gradient(180deg,#35383d,#17191d);
  box-shadow:0 2px 6px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.12);
}
.theme-pull .pull-cord{
  right:15px;
  width:24px;
  transform:rotate(var(--pull-sway));
  transform-origin:50% 0%;
  transition:
    height .64s cubic-bezier(.17,.84,.24,1),
    transform .72s cubic-bezier(.18,1.46,.28,1),
    filter var(--t-out);
}
.theme-pull.dragging .pull-cord{
  transition:none;
}
.theme-pull.returning .pull-cord{
  animation:pullCordSettle .82s cubic-bezier(.18,1.46,.28,1);
}
.theme-pull .pull-line{
  width:4px;
  height:calc(100% - 17px);
  background:
    linear-gradient(90deg,rgba(255,255,255,.42),transparent 38%,rgba(0,0,0,.18) 100%),
    repeating-linear-gradient(135deg,#f8e7b8 0 3px,#b48f3b 3px 6px,#f4d889 6px 9px);
  box-shadow:0 0 0 1px rgba(0,0,0,.08),0 2px 9px rgba(0,0,0,.16),0 0 12px rgba(245,197,24,.16);
}
.theme-pull .pull-line::before,
.theme-pull .pull-line::after{
  content:'';
  position:absolute;
  inset:0;
  border-radius:2px;
  pointer-events:none;
}
.theme-pull .pull-line::before{
  background:linear-gradient(90deg,rgba(255,255,255,.72),transparent 36%);
  opacity:.55;
}
.theme-pull .pull-line::after{
  background:repeating-linear-gradient(45deg,transparent 0 5px,rgba(255,255,255,.18) 5px 6px,transparent 6px 11px);
  opacity:.5;
}
.theme-pull .pull-handle{
  bottom:-2px;
  width:24px;
  height:18px;
  border-radius:9px 9px 10px 10px;
  border:1px solid rgba(68,50,12,.36);
  transform:translateX(-50%) scaleY(var(--handle-squeeze));
  background:
    radial-gradient(ellipse 62% 42% at 44% 18%,rgba(255,255,255,.64) 0%,rgba(255,255,255,.16) 38%,transparent 64%),
    linear-gradient(180deg,#f8d96d 0%,#d6a917 48%,#86600b 100%);
  box-shadow:
    0 5px 12px rgba(0,0,0,.22),
    0 0 0 1px rgba(255,255,255,.28) inset,
    0 -4px 7px rgba(84,55,0,.28) inset,
    0 0 15px rgba(245,197,24,.20);
}
.theme-pull .pull-handle::before{
  content:'';
  position:absolute;
  left:50%;
  top:50%;
  width:10px;
  height:6px;
  transform:translate(-50%,-50%);
  border:1px solid rgba(63,45,8,.38);
  border-radius:999px;
  box-shadow:inset 0 1px 1px rgba(255,255,255,.34);
  background:radial-gradient(ellipse at 50% 35%,rgba(255,255,255,.24),transparent 68%);
}
.theme-pull .pull-handle::after{
  content:'';
  position:absolute;
  left:6px;
  top:3px;
  width:8px;
  height:2px;
  border-radius:999px;
  background:rgba(255,255,255,.46);
  filter:blur(.3px);
}
.theme-pull:hover .pull-handle,
.theme-pull.armed .pull-handle{
  transform:translateX(-50%) scale(1.045) scaleY(var(--handle-squeeze));
  box-shadow:
    0 8px 18px rgba(0,0,0,.26),
    0 0 0 1px rgba(255,255,255,.34) inset,
    0 -4px 8px rgba(84,55,0,.3) inset,
    0 0 22px rgba(245,197,24,.36);
}
.theme-pull.returning .pull-handle{
  animation:pullHandleSettle .78s cubic-bezier(.16,1.36,.28,1);
}
html[data-theme="night"] .theme-pull{
  --cord-rest:70px;
}
html[data-theme="night"] .theme-pull .pull-line{
  background:
    linear-gradient(90deg,rgba(255,255,255,.35),transparent 38%,rgba(0,0,0,.28) 100%),
    repeating-linear-gradient(135deg,#d8dbe3 0 3px,#7f8796 3px 6px,#c4c9d3 6px 9px);
}
html[data-theme="night"] .theme-pull .pull-handle{
  border-color:rgba(245,197,24,.42);
  background:
    radial-gradient(ellipse 62% 42% at 44% 18%,rgba(255,255,255,.7) 0%,rgba(255,255,255,.16) 38%,transparent 64%),
    linear-gradient(180deg,#f6dc7b 0%,#d5ad25 46%,#6f540f 100%);
  box-shadow:
    0 8px 20px rgba(0,0,0,.4),
    0 0 0 1px rgba(255,255,255,.22) inset,
    0 -4px 8px rgba(40,29,0,.36) inset,
    0 0 20px rgba(245,197,24,.28);
}
.hist-deck.film-ready::before{
  height:1px;
  opacity:.26;
  background:linear-gradient(90deg,transparent,rgba(245,197,24,.34),rgba(42,183,202,.18),transparent);
}
.hist-cover-wrap::after{
  background:linear-gradient(180deg,transparent 72%,rgba(0,0,0,.10) 100%) !important;
}
.hist-card:hover .hist-cover-wrap::after,
.hist-card.pointer-active .hist-cover-wrap::after{
  background:
    linear-gradient(112deg,transparent 0 38%,rgba(255,255,255,.22) 48%,transparent 58%),
    linear-gradient(180deg,transparent 72%,rgba(0,0,0,.10) 100%) !important;
}
html[data-theme="night"] .hist-cover-wrap::after{
  background:linear-gradient(180deg,transparent 76%,rgba(0,0,0,.16) 100%) !important;
}
.curtain-overlay.theme-transition,
.curtain-overlay.theme-transition.curtain-done{
  display:block !important;
  opacity:1;
  visibility:visible;
  animation:none;
}
.curtain-overlay.theme-transition::before{
  animation:themeCurtainDim 1.22s cubic-bezier(.42,0,.2,1) both;
}
.curtain-overlay.theme-transition .curtain-stage-glow{
  animation:themeStageGlow 1.22s cubic-bezier(.42,0,.2,1) both;
}
.curtain-overlay.theme-transition .curtain-left{
  animation:themeCurtainLeft 1.22s cubic-bezier(.48,0,.18,1) both;
}
.curtain-overlay.theme-transition .curtain-right{
  animation:themeCurtainRight 1.22s cubic-bezier(.48,0,.18,1) both;
}
.curtain-overlay.theme-transition .curtain-seam{
  animation:themeCurtainSeam 1.22s cubic-bezier(.48,0,.18,1) both;
}
.curtain-overlay.theme-transition .curtain-light-spill{
  animation:themeLightSpill 1.22s cubic-bezier(.48,0,.18,1) both;
}
.curtain-overlay.theme-transition .curtain-dust{
  animation:themeDustExpand 1.22s cubic-bezier(.48,0,.18,1) both;
}
@keyframes refinedBeamBreath{
  0%,100%{opacity:.46;filter:blur(2px)}
  48%{opacity:.62;filter:blur(2.8px)}
}
@keyframes refinedLampSpot{
  0%,100%{opacity:.24;transform:translateX(-50%) scaleX(.96)}
  48%{opacity:.4;transform:translateX(-50%) scaleX(1.08)}
}
@keyframes pullCordSettle{
  0%{transform:rotate(var(--pull-sway))}
  34%{transform:rotate(-4.2deg)}
  58%{transform:rotate(2.2deg)}
  78%{transform:rotate(-.9deg)}
  100%{transform:rotate(0deg)}
}
@keyframes pullHandleSettle{
  0%{transform:translateX(-50%) scale(1.045,.94)}
  38%{transform:translateX(-50%) scale(.98,1.08)}
  64%{transform:translateX(-50%) scale(1.02,.985)}
  100%{transform:translateX(-50%) scale(1)}
}
@keyframes themeCurtainLeft{
  0%{transform:translateX(-101%) skewY(0deg)}
  34%{transform:translateX(0) skewY(.36deg)}
  55%{transform:translateX(0) skewY(0deg)}
  76%{transform:translateX(-7%) skewY(-.14deg)}
  100%{transform:translateX(-101%) skewY(0deg)}
}
@keyframes themeCurtainRight{
  0%{transform:translateX(101%) skewY(0deg)}
  34%{transform:translateX(0) skewY(-.36deg)}
  55%{transform:translateX(0) skewY(0deg)}
  76%{transform:translateX(7%) skewY(.14deg)}
  100%{transform:translateX(101%) skewY(0deg)}
}
@keyframes themeCurtainSeam{
  0%,18%,88%,100%{opacity:0}
  34%,58%{opacity:1}
}
@keyframes themeLightSpill{
  0%,20%{width:0;opacity:0}
  44%{width:7%;opacity:.55}
  62%{width:14%;opacity:.3}
  100%{width:0;opacity:0}
}
@keyframes themeDustExpand{
  0%,20%{width:0;opacity:0}
  44%{width:9%;opacity:.55}
  100%{width:62%;opacity:0}
}
@keyframes themeCurtainDim{
  0%,100%{opacity:0}
  32%,60%{opacity:.86}
}
@keyframes themeStageGlow{
  0%,100%{opacity:0}
  38%,62%{opacity:.34}
}
@media(max-width:640px){
  :root{
    --side-lamp-offset:min(32vw,132px);
    --side-lamp-half:12px;
    --side-cone-half:56px;
  }
  .lamp-l1,.lamp-r2{width:24px}
  .lamp-cone{width:112px}
  .theme-pull{right:4px}
}

.curtain-overlay{
  position:fixed;inset:0;z-index:9999;pointer-events:none;
  overflow:hidden;
}
/* 帘布后舞台暖光（拉开时渐亮） */
.curtain-stage-glow{
  position:absolute;inset:0;
  background:radial-gradient(ellipse 60% 50% at 50% 42%,
    rgba(255,200,100,.10) 0%,
    rgba(255,160,60,.05) 30%,
    transparent 70%);
  opacity:0;
  animation:stageGlowIn 2.8s ease-out forwards;
  animation-delay:1s;
  z-index:0;
}
/* 帘布后暗场（拉开时渐隐） */
.curtain-overlay::before{
  content:'';position:absolute;inset:0;
  background:#080808;
  animation:curtainBgFade 2.6s ease-out forwards;
  animation-delay:1.2s;
  z-index:1;
}
/* 金色幕杆 */
.curtain-rod{
  position:absolute;top:0;left:0;right:0;height:14px;
  background:linear-gradient(180deg,
    #8B6914 0%,#DAA520 18%,#FFD700 35%,
    #DAA520 55%,#8B6914 78%,#5a4a0a 100%);
  box-shadow:0 3px 14px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.35);
  z-index:6;
}
/* 幕杆两端装饰球 */
.curtain-rod::before,.curtain-rod::after{
  content:'';position:absolute;top:50%;transform:translateY(-50%);
  width:24px;height:24px;border-radius:50%;
  background:radial-gradient(circle at 35% 35%,
    #FFE55C 0%,#FFD700 25%,#DAA520 55%,#8B6914 85%,#5a4a0a 100%);
  box-shadow:0 2px 10px rgba(0,0,0,.5),inset 0 1px 2px rgba(255,255,255,.4);
}
.curtain-rod::before{left:-8px}
.curtain-rod::after{right:-8px}
/* 左帘 */
.curtain-left{
  position:absolute;top:14px;left:0;width:50.5%;height:calc(100% - 14px);
  background:
    /* 细褶皱高光丝 */
    repeating-linear-gradient(90deg,
      rgba(255,255,255,.07) 0px,rgba(255,255,255,.07) 1px,
      transparent 1px,transparent 7px),
    /* 主褶皱深谷 */
    repeating-linear-gradient(90deg,
      rgba(0,0,0,.38) 0px,rgba(0,0,0,.38) 3px,
      transparent 3px,transparent 24px,
      rgba(255,255,255,.06) 24px,rgba(255,255,255,.06) 26px,
      transparent 26px,transparent 48px),
    /* 丝绒主色 */
    linear-gradient(90deg,
      #3a0000 0%,#6B0000 10%,#8B0000 25%,
      #a82828 45%,#8B0000 65%,#6B0000 85%,#3a0000 100%);
  box-shadow:inset -16px 0 44px rgba(0,0,0,.55),inset 4px 0 8px rgba(0,0,0,.3);
  transform-origin:left center;
  animation:curtainOpenLeft 2.6s cubic-bezier(.55,0,.2,1) forwards;
  animation-delay:1.1s;
  z-index:4;
}
/* 右帘 */
.curtain-right{
  position:absolute;top:14px;right:0;width:50.5%;height:calc(100% - 14px);
  background:
    repeating-linear-gradient(90deg,
      rgba(255,255,255,.07) 0px,rgba(255,255,255,.07) 1px,
      transparent 1px,transparent 7px),
    repeating-linear-gradient(90deg,
      rgba(0,0,0,.38) 0px,rgba(0,0,0,.38) 3px,
      transparent 3px,transparent 24px,
      rgba(255,255,255,.06) 24px,rgba(255,255,255,.06) 26px,
      transparent 26px,transparent 48px),
    linear-gradient(270deg,
      #3a0000 0%,#6B0000 10%,#8B0000 25%,
      #a82828 45%,#8B0000 65%,#6B0000 85%,#3a0000 100%);
  box-shadow:inset 16px 0 44px rgba(0,0,0,.55),inset -4px 0 8px rgba(0,0,0,.3);
  transform-origin:right center;
  animation:curtainOpenRight 2.6s cubic-bezier(.55,0,.2,1) forwards;
  animation-delay:1.1s;
  z-index:4;
}
/* 帘布底部金色流苏 */
.curtain-left::after,.curtain-right::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:16px;
  background:
    repeating-linear-gradient(90deg,
      #8B6914 0px,#8B6914 2px,
      #DAA520 2px,#DAA520 4px,
      #FFD700 4px,#FFD700 5px,
      #DAA520 5px,#DAA520 7px,
      #6B5210 7px,#6B5210 10px);
  box-shadow:0 -2px 10px rgba(0,0,0,.5);
}
/* 中央接缝暗影 */
.curtain-seam{
  position:absolute;top:14px;left:50%;transform:translateX(-50%);
  width:4px;height:calc(100% - 14px);
  background:linear-gradient(90deg,transparent,rgba(0,0,0,.7),transparent);
  animation:seamFade .8s ease-out forwards;
  animation-delay:1.6s;
  z-index:5;
}
/* 中央漏光（帘布拉开时从缝隙透出暖光） */
.curtain-light-spill{
  position:absolute;top:0;left:50%;transform:translateX(-50%);
  width:0;height:100%;
  background:linear-gradient(90deg,
    transparent 0%,
    rgba(255,200,120,.14) 25%,
    rgba(255,225,160,.22) 50%,
    rgba(255,200,120,.14) 75%,
    transparent 100%);
  filter:blur(2px);
  animation:lightSpill 2.6s ease-out forwards;
  animation-delay:1.1s;
  z-index:3;
}
/* 尘埃微粒容器 */
.curtain-dust{
  position:absolute;top:0;left:50%;transform:translateX(-50%);
  width:0;height:100%;overflow:visible;
  animation:dustExpand 2.6s ease-out forwards;
  animation-delay:1.1s;
  z-index:3;
  pointer-events:none;
}
.curtain-dust span{
  position:absolute;border-radius:50%;
  background:rgba(255,230,180,.6);
  box-shadow:0 0 4px rgba(255,220,150,.5);
  animation:dustFloat linear infinite;
}
@keyframes dustFloat{
  0%{transform:translateY(0) translateX(0);opacity:0}
  10%{opacity:.6}
  90%{opacity:.3}
  100%{transform:translateY(-120vh) translateX(30px);opacity:0}
}
/* 暗场渐隐 */
@keyframes curtainBgFade{
  0%{opacity:1}
  100%{opacity:0}
}
/* 舞台暖光渐亮 */
@keyframes stageGlowIn{
  0%{opacity:0}
  40%{opacity:.3}
  100%{opacity:.7}
}
/* 左帘拉开（带织物波动与重量感） */
@keyframes curtainOpenLeft{
  0%{transform:translateX(0) skewY(0deg)}
  6%{transform:translateX(-.4%) skewY(.12deg)}
  18%{transform:translateX(-2.5%) skewY(.35deg)}
  38%{transform:translateX(-14%) skewY(.55deg)}
  58%{transform:translateX(-38%) skewY(.3deg)}
  78%{transform:translateX(-72%) skewY(.12deg)}
  92%{transform:translateX(-97%) skewY(-.04deg)}
  100%{transform:translateX(-101%) skewY(0deg)}
}
/* 右帘拉开 */
@keyframes curtainOpenRight{
  0%{transform:translateX(0) skewY(0deg)}
  6%{transform:translateX(.4%) skewY(-.12deg)}
  18%{transform:translateX(2.5%) skewY(-.35deg)}
  38%{transform:translateX(14%) skewY(-.55deg)}
  58%{transform:translateX(38%) skewY(-.3deg)}
  78%{transform:translateX(72%) skewY(-.12deg)}
  92%{transform:translateX(97%) skewY(.04deg)}
  100%{transform:translateX(101%) skewY(0deg)}
}
/* 接缝渐隐 */
@keyframes seamFade{
  0%{opacity:1}
  100%{opacity:0}
}
/* 漏光展开 */
@keyframes lightSpill{
  0%{width:0}
  25%{width:6%}
  55%{width:28%}
  100%{width:62%}
}
/* 尘埃区域展开 */
@keyframes dustExpand{
  0%{width:0}
  25%{width:6%}
  55%{width:28%}
  100%{width:62%}
}
/* 动画结束后淡出整个 overlay（由 JS 添加 class 触发） */
.curtain-overlay.curtain-done{
  animation:curtainHide .6s ease-out forwards;
}
@keyframes curtainHide{
  from{opacity:1;visibility:visible}
  to{opacity:0;visibility:hidden}
}
</style>
</head>
<body>
<!-- 电影院开场红帘 -->
<div class="curtain-overlay" id="curtain-overlay">
  <div class="curtain-stage-glow"></div>
  <div class="curtain-rod"></div>
  <div class="curtain-light-spill"></div>
  <div class="curtain-dust" id="curtain-dust"></div>
  <div class="curtain-left"></div>
  <div class="curtain-right"></div>
  <div class="curtain-seam"></div>
</div>
<div class="stage-vignette stage-vignette-l"></div>
<div class="stage-vignette stage-vignette-r"></div>
<div class="rig-wall rig-wall-l"></div>
<div class="rig-wall rig-wall-r"></div>
<div class="spot-rig">
  <div class="lamp lamp-l1"></div>
  <div class="lamp lamp-l2"></div>
  <div class="lamp lamp-r1"></div>
  <div class="lamp lamp-r2"></div>
</div>
<!-- 铭牌彩蛋（overscroll 回弹时弹出） -->
<div class="rig-badge" id="rig-badge">
  <span class="rig-badge-text">v<em>notes</em></span>
  <span class="rig-badge-sub">STUDIO</span>
</div>
<div class="spot-core"></div>
<div class="spot-cone"></div>
<div class="lamp-cone l1"></div>
<div class="lamp-cone l2"></div>
<div class="lamp-cone r1"></div>
<div class="lamp-cone r2"></div>
<a href="#main-content" class="skip-link" style="position:absolute;left:-9999px;top:0;z-index:10000;padding:8px 16px;background:var(--accent);color:var(--accent-dk);border-radius:0 0 8px 8px;text-decoration:none;font-weight:600;font-size:13px">跳到主内容</a>
<main class="app" id="main-content">

  <header class="hero" role="banner">
    <div class="hero-mark" aria-hidden="true"></div>
    <h1>v<em>notes</em></h1>
    <p>视频笔记工作室</p>
  </header>

  <section class="input-section" aria-label="视频链接输入">
    <div class="input-card">
      <div class="input-row">
        <label for="url" class="sr-only" style="position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)">视频链接</label>
        <input type="url" id="url" class="url-input" placeholder="粘贴视频链接（B站 / YouTube 等）…" aria-label="视频链接输入框" />
        <button id="gen-btn" class="gen-btn" aria-label="开始生成视频笔记">生成笔记</button>
      </div>
      <button class="settings-toggle" id="settings-toggle" aria-expanded="false" aria-controls="settings">
        <span>设置</span><span class="arrow" aria-hidden="true">&#9662;</span>
      </button>
      <div class="settings" id="settings" role="region" aria-label="生成设置">
        <div class="sg">
          <label>笔记模式</label>
          <select id="note-mode">
            <option value="">使用 .env 默认</option>
            <option value="essence">脉络精华 · 简洁图解笔记</option>
            <option value="detailed">细致笔记 · 含关键帧画面</option>
          </select>
        </div>
        <div class="sg">
          <label>转写后端</label>
          <select id="backend">
            <option value="">使用 .env 默认</option>
            <option value="faster-whisper">faster-whisper（本地，推荐）</option>
            <option value="vosk">Vosk（低内存离线兜底）</option>
            <option value="paraformer">阿里云 Paraformer（国内直连）</option>
            <option value="groq">Groq（云端，极快）</option>
            <option value="openai-whisper">openai-whisper（原版）</option>
          </select>
        </div>
        <div class="sg">
          <label>DeepSeek API Key</label>
          <input type="password" id="llm-key" placeholder="sk-…（留空则用 .env 中的配置）" />
        </div>
        <div class="sg" id="groq-group" style="display:none">
          <label>Groq API Key</label>
          <input type="password" id="groq-key" placeholder="gsk_…" />
        </div>
        <div class="sg" id="dashscope-group" style="display:none">
          <label>DashScope API Key</label>
          <input type="password" id="dashscope-key" placeholder="sk-…（阿里云）" />
        </div>
        <p class="hint">支持 B站、YouTube 等平台。在 .env 文件中配置的 API Key 会作为默认值，此处可临时覆盖。细致笔记模式会抽取视频关键帧画面并生成更详细的子板块内容。</p>
        <p class="hint" id="config-status">正在读取本机配置状态…</p>
        <label class="batch-toggle">
          <input type="checkbox" id="batch-mode" />
          <span class="batch-label">批量模式 · 多 P / 播放列表逐个生成并聚合</span>
        </label>
      </div>
    </div>
  </section>

  <section class="processing" id="processing" style="display:none">
    <div class="batch-progress" id="batch-progress" style="display:none">
      <div class="batch-bar"><div class="batch-bar-fill" id="batch-bar-fill" style="width:0%"></div></div>
      <div class="batch-info">
        <span class="current" id="batch-current">准备中…</span>
        <span class="count" id="batch-count">0 / 0</span>
      </div>
    </div>
    <div class="stage-desc" id="stage-desc"></div>
    <button type="button" id="cancel-btn" class="cancel-btn" style="display:none">取消生成</button>
    <div class="pipeline" id="pipeline"></div>
    <div class="log-box" id="log-box"></div>
  </section>

  <div id="result-area"></div>

  <section class="history reveal-on-scroll" aria-label="历史笔记">
    <div class="history-head">
      <div class="history-title-row">
        <div>
          <h2>历史笔记</h2>
          <p class="history-subtitle">生成过的笔记、长图和封面</p>
        </div>
        <div class="history-stats" id="history-stats"></div>
      </div>
    </div>
    <div class="pin-bar hidden" id="pin-bar" aria-label="置顶笔记">
      <span class="pin-bar-label">📌 置顶</span>
      <div class="pin-track" id="pin-track"></div>
    </div>
    <div class="history-toolbar" aria-label="历史笔记筛选">
      <label class="history-control history-search-wrap" aria-label="搜索历史笔记">
        <input class="history-search" id="history-search" type="search" placeholder="搜索标题 / UP / 标签" autocomplete="off"/>
      </label>
      <div class="history-filter-group">
        <label class="history-control history-select-wrap">
          <span>平台</span>
          <select class="history-filter" id="history-platform" aria-label="平台筛选">
            <option value="">全部平台</option>
            <option value="B站">B站</option>
            <option value="YouTube">YouTube</option>
            <option value="其他">其他</option>
          </select>
        </label>
        <label class="history-control history-select-wrap">
          <span>分类</span>
          <select class="history-filter" id="history-category" aria-label="分类筛选">
            <option value="">全部分类</option>
          </select>
        </label>
        <label class="history-control history-select-wrap">
          <span>质量</span>
          <select class="history-filter" id="history-quality" aria-label="质量筛选">
            <option value="">全部质量</option>
            <option value="ok">合格</option>
            <option value="check">待检查</option>
            <option value="bad">需重生成</option>
          </select>
        </label>
        <label class="history-control history-select-wrap">
          <span>排序</span>
          <select class="history-filter" id="history-sort" aria-label="排序方式">
            <option value="newest">最近生成</option>
            <option value="oldest">最早生成</option>
            <option value="duration_desc">时长最长</option>
            <option value="duration_asc">时长最短</option>
            <option value="title">标题 A-Z</option>
          </select>
        </label>
      </div>
    </div>
    <div class="hist-deck" id="hist-grid" role="list" aria-label="历史笔记列表"></div>
  </section>

</main>

<script>
// ---- 电影院开场红帘 ----
(function(){
  const overlay = document.getElementById('curtain-overlay');
  if(!overlay) return;
  // 截图工具或 ?nocurtain 参数时跳过
  if(location.search.indexOf('nocurtain') !== -1){
    overlay.style.display = 'none';
    return;
  }
  // 同一会话内只播放一次（刷新不重复）
  if(sessionStorage.getItem('vnotes_curtain_played')){
    overlay.style.display = 'none';
    return;
  }
  sessionStorage.setItem('vnotes_curtain_played','1');
  // 动态生成尘埃微粒
  const dust = document.getElementById('curtain-dust');
  if(dust){
    for(let i=0;i<22;i++){
      const s = document.createElement('span');
      s.style.left = Math.random()*100 + '%';
      s.style.top = (55 + Math.random()*45) + '%';
      s.style.animationDuration = (3.5 + Math.random()*4.5) + 's';
      s.style.animationDelay = (1.6 + Math.random()*2.5) + 's';
      s.style.opacity = (.3 + Math.random()*.4);
      const sz = (2 + Math.random()*3).toFixed(1);
      s.style.width = sz+'px'; s.style.height = sz+'px';
      dust.appendChild(s);
    }
  }
  // 动画结束后淡出并移除 overlay
  setTimeout(function(){
    overlay.classList.add('curtain-done');
    setTimeout(function(){ overlay.style.display = 'none'; }, 700);
  }, 4200);
})();

const STAGES = [
  {id:'meta',label:'元数据',desc:'正在了解这个视频…'},
  {id:'audio',label:'音频',desc:'正在提取音频…'},
  {id:'transcribe',label:'转写',desc:'正在聆听每一句话…'},
  {id:'analyze',label:'分析',desc:'AI 正在理解内容…'},
  {id:'svg',label:'图解',desc:'正在为每章绘制图解…'},
  {id:'render',label:'渲染',desc:'正在排版笔记…'},
  {id:'screenshot',label:'截图',desc:'正在定格画面…'},
  {id:'crop',label:'切片',desc:'正在切分长图…'},
];

let es = null;
let currentJobId = null;
let CONFIG_STATUS = null;

const BACKEND_LABELS = {
  '': '.env 默认',
  'faster-whisper': 'faster-whisper 本地',
  'vosk': 'Vosk 离线',
  'paraformer': '阿里云 Paraformer',
  'groq': 'Groq 云端',
  'openai-whisper': 'openai-whisper 原版',
};

function keyState(ok, label){
  return label + (ok ? '已配置' : '未配置');
}

function renderConfigStatus(){
  const el = document.getElementById('config-status');
  if(!el) return;
  const backend = document.getElementById('backend')?.value || '';
  const effective = backend || (CONFIG_STATUS && CONFIG_STATUS.default_backend) || 'faster-whisper';
  if(!CONFIG_STATUS){
    el.textContent = '正在读取本机配置状态…';
    return;
  }
  const parts = [
    '当前会走：' + (BACKEND_LABELS[effective] || effective),
    '.env 默认：' + (BACKEND_LABELS[CONFIG_STATUS.default_backend] || CONFIG_STATUS.default_backend),
    keyState(CONFIG_STATUS.has_llm_api_key, 'DeepSeek '),
    keyState(CONFIG_STATUS.has_groq_api_key, 'Groq '),
    keyState(CONFIG_STATUS.has_dashscope_api_key, 'DashScope '),
  ];
  if(effective === 'groq' && !CONFIG_STATUS.has_groq_api_key && !document.getElementById('groq-key').value.trim()){
    parts.push('Groq 需要在这里填 Key 才能跑');
  }
  if(effective === 'paraformer' && !CONFIG_STATUS.has_dashscope_api_key && !document.getElementById('dashscope-key').value.trim()){
    parts.push('Paraformer 需要填 DashScope Key');
  }
  el.textContent = parts.join(' · ');
  if(typeof updateSettingsSummary === 'function') updateSettingsSummary();
}

async function loadConfigStatus(){
  try {
    const resp = await fetch('/api/config-status');
    CONFIG_STATUS = await resp.json();
  } catch(e) {
    CONFIG_STATUS = null;
  }
  renderConfigStatus();
}

function frontendPreflight(backend, groqKey, dashscopeKey, llmKey){
  if(CONFIG_STATUS && !CONFIG_STATUS.has_llm_api_key && !llmKey){
    return '缺少 DeepSeek API Key。请在设置里填写 DeepSeek API Key，或写入 .env。';
  }
  const effective = backend || (CONFIG_STATUS && CONFIG_STATUS.default_backend) || '';
  if(effective === 'groq' && CONFIG_STATUS && !CONFIG_STATUS.has_groq_api_key && !groqKey){
    return '你当前选择 Groq 云端转写，但没有 Groq API Key。请在设置里填写 Groq API Key，或切回本地/Vosk。';
  }
  if(effective === 'paraformer' && CONFIG_STATUS && !CONFIG_STATUS.has_dashscope_api_key && !dashscopeKey){
    return '你当前选择阿里云 Paraformer，但没有 DashScope API Key。请在设置里填写 DashScope API Key，或切回本地/Vosk。';
  }
  return '';
}

function backendProblemFor(effective, groqKey, dashscopeKey){
  if(!CONFIG_STATUS) return '';
  if(effective === 'faster-whisper' && !CONFIG_STATUS.has_faster_whisper){
    return '当前 Python 环境没有 faster-whisper。请用项目 venv 启动，或在设置里切到 Vosk/Paraformer。';
  }
  if(effective === 'vosk'){
    if(!CONFIG_STATUS.has_vosk) return '当前 Python 环境没有 vosk。请用项目 venv 启动，或重新运行 install.bat。';
    if(!CONFIG_STATUS.has_vosk_model) return '未找到 Vosk 模型目录，请重新运行 install.bat 或下载 Vosk 中文模型。';
  }
  if(effective === 'groq'){
    if(!CONFIG_STATUS.has_groq_api_key && !groqKey) return 'Groq 后端缺少 API Key。';
  }
  if(effective === 'paraformer'){
    if(!CONFIG_STATUS.has_dashscope_api_key && !dashscopeKey) return '阿里云 Paraformer 后端缺少 DashScope API Key。';
    if(!CONFIG_STATUS.has_dashscope) return 'Paraformer 后端需要 dashscope 包，请重新运行 install.bat。';
  }
  if(effective === 'openai-whisper' && !CONFIG_STATUS.has_openai_whisper){
    return 'openai-whisper 后端需要 whisper/torch；建议改用 faster-whisper、Vosk 或云端后端。';
  }
  return '';
}

function renderConfigStatus(){
  const el = document.getElementById('config-status');
  if(!el) return;
  const backend = document.getElementById('backend')?.value || '';
  const groqKey = document.getElementById('groq-key')?.value.trim() || '';
  const dashscopeKey = document.getElementById('dashscope-key')?.value.trim() || '';
  const effective = backend || (CONFIG_STATUS && CONFIG_STATUS.default_backend) || 'faster-whisper';
  if(!CONFIG_STATUS){
    el.textContent = '正在读取本机配置状态…';
    return;
  }
  const envLabel = CONFIG_STATUS.in_project_venv
    ? '运行环境：项目 venv'
    : (CONFIG_STATUS.project_venv_exists ? '运行环境：系统 Python（建议重启到项目 venv）' : '运行环境：当前 Python');
  const problem = backendProblemFor(effective, groqKey, dashscopeKey);
  const parts = [
    envLabel,
    '当前会走：' + (BACKEND_LABELS[effective] || effective),
    '.env 默认：' + (BACKEND_LABELS[CONFIG_STATUS.default_backend] || CONFIG_STATUS.default_backend),
    keyState(CONFIG_STATUS.has_llm_api_key, 'DeepSeek '),
    keyState(CONFIG_STATUS.has_groq_api_key || !!groqKey, 'Groq '),
    keyState(CONFIG_STATUS.has_dashscope_api_key || !!dashscopeKey, 'DashScope '),
  ];
  if(problem) parts.push('当前后端不可用：' + problem);
  el.textContent = parts.join(' · ');
  if(typeof updateSettingsSummary === 'function') updateSettingsSummary();
}

function frontendPreflight(backend, groqKey, dashscopeKey, llmKey){
  if(CONFIG_STATUS && !CONFIG_STATUS.has_llm_api_key && !llmKey){
    return '缺少 DeepSeek API Key。请在设置里填写 DeepSeek API Key，或写入 .env。';
  }
  const effective = backend || (CONFIG_STATUS && CONFIG_STATUS.default_backend) || '';
  const problem = backendProblemFor(effective, groqKey, dashscopeKey);
  if(problem) return problem;
  return '';
}

// ---- Pipeline ----
function initPipeline(){
  const c = document.getElementById('pipeline');
  let html = '';
  STAGES.forEach((s,i) => {
    html += '<div class="p-node"><div class="p-circle" id="pc-'+s.id+'"></div><span class="p-label" id="pl-'+s.id+'">'+s.label+'</span></div>';
    if(i < STAGES.length-1) html += '<div class="p-conn" id="pcn-'+i+'"></div>';
  });
  c.innerHTML = html;
}

function setStage(id, state){
  const circle = document.getElementById('pc-'+id);
  const label = document.getElementById('pl-'+id);
  if(!circle) return;
  circle.className = 'p-circle ' + state;
  if(label) label.className = 'p-label ' + (state==='done'?'done':state==='running'?'active':'');
  const idx = STAGES.findIndex(s=>s.id===id);
  if(state==='done' && idx>0){
    const conn = document.getElementById('pcn-'+(idx-1));
    if(conn) conn.classList.add('done');
  }
  if(state==='running'){
    // 清除之前所有连接线的 running 状态
    document.querySelectorAll('.p-conn').forEach(c=>c.classList.remove('running'));
    // 给当前阶段之前的连接线添加 done，给当前连接线添加 running
    if(idx > 0){
      const prevConn = document.getElementById('pcn-'+(idx-1));
      if(prevConn) prevConn.classList.add('done');
    }
    const currConn = document.getElementById('pcn-'+idx);
    if(currConn) currConn.classList.add('running');
    const stage = STAGES.find(s=>s.id===id);
    const desc = document.getElementById('stage-desc');
    desc.textContent = stage.desc;
    desc.classList.add('active');
  }
}

// ---- Log ----
function addLog(level, tag, msg){
  const box = document.getElementById('log-box');
  const line = document.createElement('div');
  const cls = level.includes('WARN')?'warn':level.includes('ERROR')?'error':'';
  line.className = 'log-line ' + cls;
  line.innerHTML = '<span class="log-lvl">['+escapeHtml(level)+']</span><span class="log-tag">'+escapeHtml(tag)+'</span>'+escapeHtml(msg);
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
  while(box.children.length > 200) box.removeChild(box.firstChild);
}

function escapeHtml(value){
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  }[ch]));
}

function cleanExtractedUrl(url){
  let v = String(url || '').trim().replace(/^[<'"`]+|[>'"`]+$/g, '');
  const trailingChars = ".,;:!?`'\")]}，。；：！？、”’》】）";
  while(v && trailingChars.includes(v[v.length - 1])){
    v = v.slice(0, -1);
  }
  return v;
}

function extractVideoUrl(raw){
  const text = String(raw || '').trim();
  if(!text) return '';
  const direct = text.match(/https?:\/\/[^\s'"<>\u3000\u4e00-\u9fff`]+/i);
  if(direct) return cleanExtractedUrl(direct[0]);
  const bare = text.match(/(?:^|[^\w@])((?:(?:www\.)?bilibili\.com|b23\.tv|(?:www\.)?youtube\.com|youtu\.be)\/[^\s'"<>\u3000\u4e00-\u9fff`]+)/i);
  if(bare) return 'https://' + cleanExtractedUrl(bare[1]);
  const bvid = text.match(/\b(BV[0-9A-Za-z]{10})\b/i);
  if(bvid){
    let url = 'https://www.bilibili.com/video/' + bvid[1];
    const part = text.match(/[?&]p=(\d+)\b/i);
    if(part) url += '?p=' + part[1];
    return url;
  }
  const avid = text.match(/\b(av\d{3,})\b/i);
  if(avid){
    let url = 'https://www.bilibili.com/video/' + avid[1];
    const part = text.match(/[?&]p=(\d+)\b/i);
    if(part) url += '?p=' + part[1];
    return url;
  }
  return text;
}

function normalizeUrlInput(showPulse){
  const inp = document.getElementById('url');
  if(!inp) return '';
  const raw = inp.value.trim();
  const cleaned = extractVideoUrl(raw);
  if(cleaned && cleaned !== raw && /^https?:\/\//i.test(cleaned)){
    inp.value = cleaned;
    if(showPulse){
      inp.classList.remove('recognized');
      void inp.offsetWidth;
      inp.classList.add('recognized');
      setTimeout(() => inp.classList.remove('recognized'), 1000);
    }
  }
  return cleaned;
}

function outputHref(dir, file){
  return '/output/' + encodeURIComponent(String(dir || '')) + '/' + file;
}

// ---- Generation ----
function setGenerateBusy(isBusy, text){
  const btn = document.getElementById('gen-btn');
  if(!btn) return;
  btn.disabled = !!isBusy;
  btn.classList.toggle('is-loading', !!isBusy);
  btn.textContent = text || '生成笔记';
  const cancel = document.getElementById('cancel-btn');
  if(cancel) cancel.style.display = (isBusy && currentJobId) ? 'block' : 'none';
}

function cancelJob(){
  if(!currentJobId) return;
  const btn = document.getElementById('cancel-btn');
  if(btn){ btn.disabled = true; btn.textContent = '取消中…'; }
  fetch('/api/job/'+currentJobId+'/cancel', { method:'POST' })
    .then(r => r.json())
    .then(() => toast('已请求取消生成…','info'))
    .catch(() => toast('取消失败', 'error'));
}

function toast(msg, type){
  const box = document.getElementById('toast-box');
  if(!box) return;
  const el = document.createElement('div');
  el.className = 'toast ' + (type || 'info');
  el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => el.classList.add('show'), 10);
  setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 300); }, 2600);
}

function getVideoId(url){
  if(!url) return '';
  const bv = String(url).match(/BV[\w]{10}/i);
  if(bv) return bv[0].toUpperCase();
  const yt = String(url).match(/[?&]v=([\w-]{6,})/i);
  if(yt) return yt[1];
  return '';
}

async function startGen(){
  const inp = document.getElementById('url');
  const rawInput = inp.value.trim();
  const url = normalizeUrlInput(true) || extractVideoUrl(rawInput);
  if(!url){
    inp.style.animation = 'shake .35s ease';
    setTimeout(()=>inp.style.animation='',350);
    return;
  }
  if(url !== rawInput) inp.value = url;

  // 防重复：同一视频已生成过则确认
  const vid = getVideoId(url);
  if(vid){
    const dup = (historyItems || []).find(it => getVideoId(it.source_url || '') === vid);
    if(dup && !confirm('已生成过《'+(dup.title||dup.name)+'》，确定要重新生成并覆盖吗？')) return;
  }

  const noteMode = document.getElementById('note-mode').value;
  const backend = document.getElementById('backend').value;
  const llmKey = document.getElementById('llm-key').value.trim();
  const groqKey = document.getElementById('groq-key').value.trim();
  const dashscopeKey = document.getElementById('dashscope-key').value.trim();
  const batchMode = document.getElementById('batch-mode').checked;

  const preflight = frontendPreflight(backend, groqKey, dashscopeKey, llmKey);
  if(preflight){
    showError(preflight);
    return;
  }

  saveSettings(url, noteMode, backend, llmKey, groqKey, dashscopeKey, batchMode);

  document.getElementById('processing').style.display = 'block';
  document.getElementById('result-area').innerHTML = '';
  document.getElementById('log-box').innerHTML = '';
  initPipeline();

  // 批量模式显示进度条
  const batchProg = document.getElementById('batch-progress');
  if(batchMode){
    batchProg.style.display = 'block';
    document.getElementById('batch-bar-fill').style.width = '0%';
    document.getElementById('batch-current').textContent = '准备中…';
    document.getElementById('batch-count').textContent = '0 / 0';
  } else {
    batchProg.style.display = 'none';
  }

  const desc = document.getElementById('stage-desc');
  desc.textContent = '准备开始…';
  desc.style.color = '';
  desc.classList.add('active');

  setGenerateBusy(true, batchMode ? '批量生成中…' : '生成中…');

  document.getElementById('processing').scrollIntoView({behavior:'smooth',block:'start'});

  try {
    const resp = await fetch('/api/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        url, note_mode:noteMode, transcribe_backend:backend,
        llm_api_key:llmKey, groq_api_key:groqKey, dashscope_api_key:dashscopeKey,
        batch:batchMode,
      }),
    });
    const data = await resp.json();
    if(!resp.ok || data.error || !data.job_id){
      throw new Error(data.error || '生成请求失败，请检查视频链接。');
    }

    currentJobId = data.job_id;
    setGenerateBusy(true, '生成中…');
    es = new EventSource('/api/stream/'+data.job_id);
    es.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      handleEvent(ev);
    };
    es.onerror = () => {
      es.close();
      setGenerateBusy(false);
    };
  } catch(err) {
    addLog('ERROR','','连接失败：'+err.message);
    showError(err.message);
    const errDesc = document.getElementById('stage-desc');
    if(errDesc){
      errDesc.textContent = '出了点问题。';
      errDesc.style.color = 'var(--error)';
    }
    setGenerateBusy(false);
  }
}

function handleEvent(ev){
  switch(ev.type){
    case 'stage_start':
      setStage(ev.stage,'running');
      break;
    case 'stage_done':
      setStage(ev.stage,'done');
      break;
    case 'batch_progress':
      updateBatchProgress(ev.completed, ev.total, ev.title);
      break;
    case 'log':
      addLog(ev.level, ev.tag, ev.message);
      break;
    case 'done':
      if(es) es.close();
      document.querySelectorAll('.p-conn').forEach(c=>c.classList.remove('running'));
      showResult(ev.result);
      setGenerateBusy(false);
      currentJobId = null;
      document.getElementById('stage-desc').textContent = '完成。';
      toast('笔记生成完成','success');
      loadHistory();
      break;
    case 'error':
      if(es) es.close();
      document.querySelectorAll('.p-conn').forEach(c=>c.classList.remove('running'));
      showError(ev.message);
      setGenerateBusy(false);
      currentJobId = null;
      const errDesc = document.getElementById('stage-desc');
      errDesc.textContent = '出了点问题。';
      errDesc.style.color = 'var(--error)';
      toast('生成失败：' + ev.message, 'error');
      break;
    case 'cancelled':
      if(es) es.close();
      document.querySelectorAll('.p-conn').forEach(c=>c.classList.remove('running'));
      setGenerateBusy(false);
      currentJobId = null;
      const cDesc = document.getElementById('stage-desc');
      if(cDesc){ cDesc.textContent = '已取消。'; cDesc.style.color = ''; }
      toast('已取消生成','info');
      break;
  }
}

function updateBatchProgress(completed, total, title){
  const pct = total > 0 ? (completed / total * 100) : 0;
  document.getElementById('batch-bar-fill').style.width = pct + '%';
  document.getElementById('batch-current').textContent = title;
  document.getElementById('batch-count').textContent = completed + ' / ' + total;
}

function showResult(r){
  const area = document.getElementById('result-area');
  const fallbackDir = r.dir || '';

  // 批量模式：结果是 index.html 聚合页
  if(r.is_index){
    const indexHref = outputHref(r.dir, 'index.html');
    area.innerHTML =
      '<div class="result-card">' +
        '<div class="result-head"><div class="result-icon"></div><h3>批量笔记生成完成</h3></div>' +
        '<div class="result-meta">' + escapeHtml(r.title || r.dir) + '</div>' +
        '<div class="result-actions"><a href="'+indexHref+'" target="_blank" rel="noopener noreferrer" data-open-link="1">查看合集</a></div>' +
      '</div>';
    area.scrollIntoView({behavior:'smooth',block:'center'});
    return;
  }

  let actions = '<a href="'+outputHref(r.dir, 'notes.html')+'" target="_blank" rel="noopener noreferrer" data-open-link="1">查看笔记</a>';
  if(r.has_full) actions += '<a href="'+outputHref(r.dir, 'full.png')+'" target="_blank" rel="noopener noreferrer" data-open-link="1" class="secondary">长图</a>';
  if(r.has_cover) actions += '<a href="'+outputHref(r.dir, 'cover.jpg')+'" target="_blank" rel="noopener noreferrer" data-open-link="1" class="secondary">封面</a>';

  let meta = '';
  if(r.title) meta += r.title;
  if(r.chapters) meta += (meta?' · ':'') + r.chapters + ' 章';
  if(r.duration) meta += (meta?' · ':'') + Math.round(r.duration/60) + ' 分钟';
  if(r.slices) meta += (meta?' · ':'') + r.slices + ' 切片';

  area.innerHTML =
    '<div class="result-card">' +
      '<div class="result-head"><div class="result-icon"></div><h3>笔记生成完成</h3></div>' +
      '<div class="result-meta">' + escapeHtml(meta || fallbackDir) + '</div>' +
      '<div class="result-actions">' + actions + '</div>' +
    '</div>';
  area.scrollIntoView({behavior:'smooth',block:'center'});
}

function showError(msg){
  const area = document.getElementById('result-area');
  area.innerHTML =
    '<div class="error-card"><h3>生成失败</h3><p>' + escapeHtml(msg) + '</p></div>';
  area.scrollIntoView({behavior:'smooth',block:'center'});
}

// ---- History ----
async function loadHistoryLegacyUnused(){
  try {
    const resp = await fetch('/api/history');
    const data = await resp.json();
    const deck = document.getElementById('hist-grid');
    if(!data.items || data.items.length === 0){
      deck.innerHTML = '<p class="hist-empty">还没有笔记，粘贴链接开始吧</p>';
      return;
    }
    deck.innerHTML = data.items.map(it => {
      const noteHref = outputHref(it.name, 'notes.html');
      const cover = it.has_cover
        ? '<div class="hist-cover-wrap"><img class="hist-cover" src="'+outputHref(it.name, 'cover.jpg')+'" alt="" loading="lazy"/></div>'
        : '<div class="hist-cover-wrap"><div class="hist-cover-placeholder"></div></div>';
      const info = (it.uploader || '') + (it.chapters ? ' · ' + it.chapters + ' 章' : '');
      return '<a class="hist-card" href="'+noteHref+'" target="_blank" rel="noopener noreferrer" data-href="'+noteHref+'">' +
        cover + '<div class="hist-info"><h4>'+escapeHtml(it.title)+'</h4><p>'+escapeHtml(info)+'</p></div></a>';
    }).join('');
    dealCards();
  } catch(e) {
    document.getElementById('hist-grid').innerHTML = '<p class="hist-empty">加载失败</p>';
  }
}

function dealCards(){
  const deck = document.getElementById('hist-grid');
  if(!deck) return;
  const cards = deck.querySelectorAll('.hist-card');
  if(cards.length === 0) return;
  deck.classList.add('dealt','film-ready');
  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  cards.forEach((card, index) => {
    card.style.setProperty('--i', index);
    card.style.transitionDelay = reducedMotion ? '0ms' : Math.min(index * 36, 280) + 'ms';
    requestAnimationFrame(() => {
      card.classList.add('is-in');
      window.setTimeout(() => { card.style.transitionDelay = '0ms'; }, Math.min(index * 36, 280) + 360);
    });
    card.addEventListener('pointermove', function(e){
      if(reducedMotion) return;
      const r = this.getBoundingClientRect();
      const px = (e.clientX - r.left) / Math.max(r.width, 1);
      const py = (e.clientY - r.top) / Math.max(r.height, 1);
      this.style.setProperty('--mx', (px * 100).toFixed(1) + '%');
      this.style.setProperty('--my', (py * 100).toFixed(1) + '%');
      this.style.setProperty('--ry', ((px - .5) * 5).toFixed(2) + 'deg');
      this.style.setProperty('--rx', ((.5 - py) * 4).toFixed(2) + 'deg');
      this.classList.add('pointer-active');
    });
    card.addEventListener('pointerleave', function(){
      this.style.setProperty('--rx','0deg');
      this.style.setProperty('--ry','0deg');
      this.style.setProperty('--mx','50%');
      this.style.setProperty('--my','22%');
      this.classList.remove('pointer-active');
    });
    card.addEventListener('click', function(e){
      this.classList.add('card-flash');
      setTimeout(() => this.classList.remove('card-flash'), 700);
    });
  });
}

let historyItems = [];
const HISTORY_PAGE_SIZE = 12;
let historyVisibleLimit = HISTORY_PAGE_SIZE;

function formatHistoryDuration(sec){
  const seconds = Number(sec || 0);
  if(!Number.isFinite(seconds) || seconds <= 0) return '';
  const minutes = Math.max(1, Math.round(seconds / 60));
  if(minutes < 60) return minutes + ' 分钟';
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return hours + ' 小时' + (rest ? ' ' + rest + ' 分钟' : '');
}

function formatHistoryTime(ts){
  const n = Number(ts || 0);
  if(!Number.isFinite(n) || n <= 0) return '';
  const d = new Date(n * 1000);
  const now = new Date();
  const sameYear = d.getFullYear() === now.getFullYear();
  const opts = sameYear
    ? { month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit' }
    : { year:'numeric', month:'2-digit', day:'2-digit' };
  return new Intl.DateTimeFormat('zh-CN', opts).format(d);
}

function formatUploadDate(v){
  const s = String(v || '').trim();
  if(!/^\d{8}$/.test(s)) return '';
  return s.slice(0,4) + '-' + s.slice(4,6) + '-' + s.slice(6,8);
}

function historyCategories(items){
  const set = new Set();
  items.forEach(it => set.add(it.category || '未分类'));
  return [...set].sort((a,b) => a.localeCompare(b, 'zh-CN'));
}

function syncHistoryCategoryOptions(){
  const select = document.getElementById('history-category');
  if(!select) return;
  const current = select.value;
  const options = ['<option value="">全部分类</option>']
    .concat(historyCategories(historyItems).map(c => '<option value="'+escapeHtml(c)+'">'+escapeHtml(c)+'</option>'));
  select.innerHTML = options.join('');
  if([...select.options].some(o => o.value === current)) select.value = current;
}

function historyStatsText(items, total){
  if(total === 0) return '0 条';
  const prefix = items.length === total ? '共 ' + total + ' 条' : '显示 ' + items.length + ' / ' + total + ' 条';
  const duration = formatHistoryDuration(items.reduce((sum, it) => sum + Number(it.duration || 0), 0));
  return duration ? prefix + ' · 约 ' + duration : prefix;
}

function renderHistoryCard(it){
  const noteHref = outputHref(it.name, 'notes.html');
  const title = it.title || it.name || '未命名笔记';
  const cover = it.has_cover
    ? '<div class="hist-cover-wrap"><img class="hist-cover" src="'+outputHref(it.name, 'cover.jpg')+'" alt="" loading="lazy"/></div>'
    : '<div class="hist-cover-wrap"><div class="hist-cover-placeholder"></div></div>';
  const metaParts = [];
  if(it.uploader) metaParts.push(it.uploader);
  if(it.chapters) metaParts.push(it.chapters + ' 章');
  const duration = formatHistoryDuration(it.duration);
  if(duration) metaParts.push(duration);
  const badges = [];
  if(it.has_full) badges.push('<span>长图</span>');
  if(it.has_cover) badges.push('<span>封面</span>');
  const actions = [
    '<a href="'+noteHref+'" target="_blank" rel="noopener noreferrer" data-open-link="1">笔记</a>'
  ];
  if(it.has_full) actions.push('<a href="'+outputHref(it.name, 'full.png')+'" target="_blank" rel="noopener noreferrer" data-open-link="1" class="secondary">长图</a>');
  if(it.has_cover) actions.push('<a href="'+outputHref(it.name, 'cover.jpg')+'" target="_blank" rel="noopener noreferrer" data-open-link="1" class="secondary">封面</a>');
  return '<article class="hist-card" role="listitem">' +
    '<a class="hist-main" href="'+noteHref+'" target="_blank" rel="noopener noreferrer" data-open-link="1">' +
      cover +
      '<div class="hist-info"><h4>'+escapeHtml(title)+'</h4>' +
        '<p class="hist-meta">'+escapeHtml(metaParts.join(' · ') || '已生成')+'</p>' +
        '<div class="hist-badges">'+badges.join('')+'</div>' +
      '</div>' +
    '</a>' +
    '<div class="hist-actions">' + actions.join('') + '</div>' +
  '</article>';
}

function renderHistory(query){
  const deck = document.getElementById('hist-grid');
  const stats = document.getElementById('history-stats');
  if(!deck) return;
  const q = String(query ?? document.getElementById('history-search')?.value ?? '').trim().toLowerCase();
  const filtered = historyItems.filter(it => {
    if(!q) return true;
    return [it.title, it.uploader, it.name].some(v => String(v || '').toLowerCase().includes(q));
  });
  if(stats) stats.textContent = historyStatsText(filtered, historyItems.length);
  if(historyItems.length === 0){
    deck.innerHTML = '<p class="hist-empty">还没有笔记，粘贴链接开始吧</p>';
    return;
  }
  if(filtered.length === 0){
    deck.innerHTML = '<p class="hist-empty">没有找到匹配的历史笔记</p>';
    return;
  }
  deck.innerHTML = filtered.map(renderHistoryCard).join('');
  dealCards();
}

function bindHistorySearch(){
  const input = document.getElementById('history-search');
  if(!input || input.dataset.bound) return;
  input.dataset.bound = '1';
  input.addEventListener('input', () => renderHistory(input.value));
}

/* Publish-ready history browser overrides. */
function formatHistoryDuration(sec){
  const seconds = Number(sec || 0);
  if(!Number.isFinite(seconds) || seconds <= 0) return '';
  const minutes = Math.max(1, Math.round(seconds / 60));
  if(minutes < 60) return minutes + ' 分钟';
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return hours + ' 小时' + (rest ? ' ' + rest + ' 分钟' : '');
}

function formatHistoryTime(ts){
  const n = Number(ts || 0);
  if(!Number.isFinite(n) || n <= 0) return '';
  const d = new Date(n * 1000);
  const now = new Date();
  const sameYear = d.getFullYear() === now.getFullYear();
  const opts = sameYear
    ? { month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit' }
    : { year:'numeric', month:'2-digit', day:'2-digit' };
  return new Intl.DateTimeFormat('zh-CN', opts).format(d);
}

function formatUploadDate(v){
  const s = String(v || '').trim();
  if(!/^\d{8}$/.test(s)) return '';
  return s.slice(0,4) + '-' + s.slice(4,6) + '-' + s.slice(6,8);
}

function historyCategories(items){
  const set = new Set();
  items.forEach(it => set.add(it.category || '未分类'));
  return [...set].sort((a,b) => a.localeCompare(b, 'zh-CN'));
}

function syncHistoryCategoryOptions(){
  const select = document.getElementById('history-category');
  if(!select) return;
  const current = select.value;
  const options = ['<option value="">全部分类</option>']
    .concat(historyCategories(historyItems).map(c => '<option value="'+escapeHtml(c)+'">'+escapeHtml(c)+'</option>'));
  select.innerHTML = options.join('');
  if([...select.options].some(o => o.value === current)) select.value = current;
}

function historyStatsText(items, total){
  if(total === 0) return '0 条';
  const prefix = items.length === total ? '共 ' + total + ' 条' : '显示 ' + items.length + ' / ' + total + ' 条';
  const duration = formatHistoryDuration(items.reduce((sum, it) => sum + Number(it.duration || 0), 0));
  return duration ? prefix + ' · 约 ' + duration : prefix;
}

function sortHistoryItems(items, mode){
  const list = items.slice();
  const titleOf = it => String(it.title || it.name || '');
  if(mode === 'oldest') return list.sort((a,b) => Number(a.generated_at || a.updated_at || 0) - Number(b.generated_at || b.updated_at || 0));
  if(mode === 'duration_desc') return list.sort((a,b) => Number(b.duration || 0) - Number(a.duration || 0));
  if(mode === 'duration_asc') return list.sort((a,b) => Number(a.duration || 0) - Number(b.duration || 0));
  if(mode === 'title') return list.sort((a,b) => titleOf(a).localeCompare(titleOf(b), 'zh-CN'));
  return list.sort((a,b) => Number(b.generated_at || b.updated_at || 0) - Number(a.generated_at || a.updated_at || 0));
}

function historyQualityStatus(it){
  const s = String(it.quality_status || '').trim();
  if(s === 'ok' || s === 'bad' || s === 'check') return s;
  return 'check';
}

function historyQualityLabel(status){
  if(status === 'ok') return '合格';
  if(status === 'bad') return '需重生成';
  return '待检查';
}

function currentHistoryItems(query){
  const q = String(query ?? document.getElementById('history-search')?.value ?? '').trim().toLowerCase();
  const platform = document.getElementById('history-platform')?.value || '';
  const category = document.getElementById('history-category')?.value || '';
  const quality = document.getElementById('history-quality')?.value || '';
  const sortMode = document.getElementById('history-sort')?.value || 'newest';
  const reset = document.getElementById('history-reset');
  if(reset) reset.hidden = !(q || platform || category || quality || sortMode !== 'newest');
  const filtered = historyItems.filter(it => {
    if(platform && (it.platform || '其他') !== platform) return false;
    if(category && (it.category || '未分类') !== category) return false;
    if(quality && historyQualityStatus(it) !== quality) return false;
    if(!q) return true;
    return [it.title, it.uploader, it.name, it.platform, it.category, ...(it.tags || [])]
      .some(v => String(v || '').toLowerCase().includes(q));
  });
  return sortHistoryItems(filtered, sortMode);
}

function renderHistoryCard(it){
  const noteHref = outputHref(it.name, 'notes.html');
  const title = it.title || it.name || '未命名笔记';
  const qualityStatus = historyQualityStatus(it);
  const cover = it.has_cover
    ? '<div class="hist-cover-wrap"><img class="hist-cover" src="'+outputHref(it.name, 'cover.jpg')+'" alt="" loading="lazy"/></div>'
    : '<div class="hist-cover-wrap"><div class="hist-cover-placeholder"></div></div>';
  const metaParts = [];
  if(it.uploader) metaParts.push(it.uploader);
  if(it.chapters) metaParts.push(it.chapters + ' 章');
  const duration = formatHistoryDuration(it.duration);
  if(duration) metaParts.push(duration);
  const generated = formatHistoryTime(it.generated_at || it.updated_at);
  const uploaded = formatUploadDate(it.upload_date);
  const badges = [
    '<span class="hist-quality '+qualityStatus+'">'+historyQualityLabel(qualityStatus)+'</span>',
    '<span class="hist-badge-strong">'+escapeHtml(it.platform || '其他')+'</span>',
    '<span>'+escapeHtml(it.category || '未分类')+'</span>'
  ];
  if(generated) badges.push('<span>生成 ' + escapeHtml(generated) + '</span>');
  if(uploaded) badges.push('<span>发布 ' + escapeHtml(uploaded) + '</span>');
  if(it.has_full) badges.push('<span>长图</span>');
  const actions = [
    '<a href="'+noteHref+'" target="_blank" rel="noopener noreferrer" data-open-link="1">打开</a>'
  ];
  if(it.has_full) actions.push('<a href="'+outputHref(it.name, 'full.png')+'" target="_blank" rel="noopener noreferrer" data-open-link="1" class="secondary">长图</a>');
  if(it.has_cover) actions.push('<a href="'+outputHref(it.name, 'cover.jpg')+'" target="_blank" rel="noopener noreferrer" data-open-link="1" class="secondary">封面</a>');
  if(it.source_url && qualityStatus !== 'ok'){
    actions.push('<button type="button" class="secondary warn" data-history-action="regen" data-name="'+escapeHtml(it.name)+'" data-url="'+escapeHtml(it.source_url)+'">重生成</button>');
  }
  const menu = [
    '<button type="button" class="menu-btn" data-history-action="pin" data-name="'+escapeHtml(it.name)+'">'+(isPinned(it.name)?'取消置顶':'置顶')+'</button>',
    '<button type="button" class="menu-btn danger" data-history-action="del" data-name="'+escapeHtml(it.name)+'">删除</button>'
  ];
  return '<article class="hist-card quality-'+qualityStatus+'" role="listitem">' +
    '<button type="button" class="hist-card-more" data-history-action="more" aria-label="更多操作">&#x22ee;</button>' +
    '<a class="hist-main" href="'+noteHref+'" target="_blank" rel="noopener noreferrer" data-open-link="1">' +
      cover +
      '<div class="hist-info"><h4>'+escapeHtml(title)+'</h4>' +
        '<p class="hist-meta">'+escapeHtml(metaParts.join(' · ') || '已生成')+'</p>' +
        '<div class="hist-badges">'+badges.join('')+'</div>' +
      '</div>' +
    '</a>' +
    '<div class="hist-actions">' + actions.join('') + '</div>' +
    '<div class="hist-card-menu">' + menu.join('') + '</div>' +
  '</article>';
}

function renderHistoryMore(total, shown){
  if(total <= shown) return '';
  return '<div class="history-more" role="presentation">' +
    '<button type="button" id="history-more-btn">加载更多</button>' +
    '<span>已显示 ' + shown + ' / ' + total + '</span>' +
  '</div>';
}

function bindHistoryMore(){
  const btn = document.getElementById('history-more-btn');
  if(!btn || btn.dataset.bound) return;
  btn.dataset.bound = '1';
  btn.addEventListener('click', () => {
    historyVisibleLimit += HISTORY_PAGE_SIZE;
    renderHistory();
  });
}

function renderHistory(query){
  const deck = document.getElementById('hist-grid');
  const stats = document.getElementById('history-stats');
  if(!deck) return;
  syncHistoryCategoryOptions();
  const filtered = currentHistoryItems(query);
  if(stats) stats.textContent = historyStatsText(filtered, historyItems.length);
  if(historyItems.length === 0){
    deck.innerHTML = '<p class="hist-empty">还没有笔记，粘贴链接开始吧</p>';
    return;
  }
  if(filtered.length === 0){
    deck.innerHTML = '<p class="hist-empty">没有找到匹配的历史笔记</p>';
    return;
  }
  const visible = filtered.slice(0, historyVisibleLimit);
  deck.innerHTML = visible.map(renderHistoryCard).join('') + renderHistoryMore(filtered.length, visible.length);
  dealCards();
  bindHistoryMore();
}

function bindHistoryControls(){
  const controls = [
    document.getElementById('history-search'),
    document.getElementById('history-platform'),
    document.getElementById('history-category'),
    document.getElementById('history-quality'),
    document.getElementById('history-sort'),
  ].filter(Boolean);
  controls.forEach(el => {
    if(el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('input', () => {
      historyVisibleLimit = HISTORY_PAGE_SIZE;
      renderHistory();
    });
    el.addEventListener('change', () => {
      historyVisibleLimit = HISTORY_PAGE_SIZE;
      renderHistory();
    });
  });
  const reset = document.getElementById('history-reset');
  if(reset && !reset.dataset.bound){
    reset.dataset.bound = '1';
    reset.addEventListener('click', () => {
      const search = document.getElementById('history-search');
      const platform = document.getElementById('history-platform');
      const category = document.getElementById('history-category');
      const quality = document.getElementById('history-quality');
      const sort = document.getElementById('history-sort');
      if(search) search.value = '';
      if(platform) platform.value = '';
      if(category) category.value = '';
      if(quality) quality.value = '';
      if(sort) sort.value = 'newest';
      historyVisibleLimit = HISTORY_PAGE_SIZE;
      renderHistory();
    });
  }
}

function bindHistorySearch(){
  bindHistoryControls();
  bindHistoryActions();
}

function bindHistoryActions(){
  const grid = document.getElementById('hist-grid');
  if(grid && !grid.dataset.actionsBound){
    grid.dataset.actionsBound = '1';
    grid.addEventListener('click', (ev) => {
      const t = ev.target.closest && ev.target.closest('[data-history-action]');
      if(t){
        ev.preventDefault(); ev.stopPropagation();
        const name = t.dataset.name || '';
        const url = t.dataset.url || '';
        const act = t.dataset.historyAction;
        if(act === 'more'){ toggleCardMenu(t); }
        else if(act === 'pin'){ togglePin(name); closeCardMenus(); renderPinBar(); renderHistory(); toast(isPinned(name)?'已置顶':'已取消置顶','success'); }
        else if(act === 'regen'){ closeCardMenus(); toast('开始重新生成','info'); regenNote(name, url); }
        else if(act === 'del'){ closeCardMenus(); deleteNote(name); }
        return;
      }
      if(ev.target.classList && ev.target.classList.contains('hist-card-menu')){ closeCardMenus(); return; }
      if(!ev.target.closest('.hist-card')) closeCardMenus();
    });
  }
  const track = document.getElementById('pin-track');
  if(track && !track.dataset.bound){
    track.dataset.bound = '1';
    track.addEventListener('click', (ev) => {
      const unpin = ev.target.closest && ev.target.closest('[data-pin-unpin]');
      if(unpin){ ev.preventDefault(); ev.stopPropagation(); togglePin(unpin.dataset.pinUnpin); renderPinBar(); renderHistory(); }
    });
  }
  if(!document.body.dataset.menuDocBound){
    document.body.dataset.menuDocBound = '1';
    document.addEventListener('click', (e) => { if(!e.target.closest('.hist-card')) closeCardMenus(); });
  }
}

function toggleCardMenu(btn){
  const card = btn.closest('.hist-card');
  const wasOpen = card && card.classList.contains('menu-open');
  closeCardMenus();
  if(!wasOpen && card) card.classList.add('menu-open');
}
function closeCardMenus(){
  document.querySelectorAll('.hist-card.menu-open').forEach(c => c.classList.remove('menu-open'));
}

async function loadHistory(){
  const deck = document.getElementById('hist-grid');
  try {
    bindHistorySearch();
    const resp = await fetch('/api/history');
    const data = await resp.json();
    historyItems = Array.isArray(data.items) ? data.items : [];
    historyVisibleLimit = HISTORY_PAGE_SIZE;
    renderHistory();
    renderPinBar();
  } catch(e) {
    if(deck) deck.innerHTML = '<p class="hist-empty">加载失败</p>';
    const stats = document.getElementById('history-stats');
    if(stats) stats.textContent = '';
  }
}

// ---- 历史笔记操作：重生成 / 删除 ----
async function regenNote(name, url){
  const noteMode = document.getElementById('note-mode') ? document.getElementById('note-mode').value : '';
  const backend = document.getElementById('backend') ? document.getElementById('backend').value : '';
  const llmKey = (document.getElementById('llm-key') || {}).value?.trim() || '';
  const groqKey = (document.getElementById('groq-key') || {}).value?.trim() || '';
  const dashscopeKey = (document.getElementById('dashscope-key') || {}).value?.trim() || '';
  if(!url){ showError('无法重生成：该笔记缺少源链接。'); return; }
  const preflight = frontendPreflight(backend, groqKey, dashscopeKey, llmKey);
  if(preflight){ showError(preflight); return; }
  document.getElementById('processing').style.display = 'block';
  document.getElementById('result-area').innerHTML = '';
  document.getElementById('log-box').innerHTML = '';
  initPipeline();
  const batchProg = document.getElementById('batch-progress');
  if(batchProg) batchProg.style.display = 'none';
  const desc = document.getElementById('stage-desc');
  if(desc){ desc.textContent = '正在重新生成：' + name; desc.style.color = ''; }
  setGenerateBusy(true, '重生成中…');
  document.getElementById('processing').scrollIntoView({behavior:'smooth',block:'start'});
  try {
    const resp = await fetch('/api/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ url, note_mode:noteMode, transcribe_backend:backend, llm_api_key:llmKey, groq_api_key:groqKey, dashscope_api_key:dashscopeKey, batch:false }),
    });
    const data = await resp.json();
    if(!resp.ok || data.error || !data.job_id){ throw new Error(data.error || '重生成请求失败。'); }
    currentJobId = data.job_id;
    setGenerateBusy(true, '重生成中…');
    es = new EventSource('/api/stream/'+data.job_id);
    es.onmessage = (e) => { const ev = JSON.parse(e.data); handleEvent(ev); };
    es.onerror = () => { es.close(); setGenerateBusy(false); };
  } catch(err) {
    addLog('ERROR','','连接失败：'+err.message);
    showError(err.message);
    const d = document.getElementById('stage-desc');
    if(d){ d.textContent = '出了点问题。'; d.style.color = 'var(--error)'; }
    setGenerateBusy(false);
  }
}

async function deleteNote(name){
  if(!name) return;
  if(!confirm('删除该笔记「'+name+'」？此操作不可恢复。')) return;
  try {
    const resp = await fetch('/api/note/'+encodeURIComponent(name), { method:'DELETE' });
    const data = await resp.json();
    if(!resp.ok) throw new Error(data.error || '删除失败');
    if(Array.isArray(historyItems)) historyItems = historyItems.filter(it => it.name !== name);
    renderHistory();
    toast('笔记已删除','success');
  } catch(err) {
    toast('删除失败：' + err.message, 'error');
  }
}

// ---- 置顶状态（localStorage 持久化）----
let pinnedNotes = [];
try { pinnedNotes = JSON.parse(localStorage.getItem('vnotes_pinned') || '[]'); if(!Array.isArray(pinnedNotes)) pinnedNotes = []; } catch(e) { pinnedNotes = []; }
function savePinned(){ try { localStorage.setItem('vnotes_pinned', JSON.stringify(pinnedNotes)); } catch(e){} }
function isPinned(name){ return pinnedNotes.indexOf(name) !== -1; }
function togglePin(name){
  if(!name) return;
  const i = pinnedNotes.indexOf(name);
  if(i === -1) pinnedNotes.unshift(name); else pinnedNotes.splice(i, 1);
  savePinned();
}
function renderPinBar(){
  const bar = document.getElementById('pin-bar');
  const track = document.getElementById('pin-track');
  if(!bar || !track) return;
  const items = pinnedNotes.map(n => (historyItems || []).find(it => it.name === n)).filter(Boolean);
  if(!items.length){ bar.classList.add('hidden'); track.innerHTML = ''; return; }
  bar.classList.remove('hidden');
  track.innerHTML = items.map(it => {
    const href = outputHref(it.name, 'notes.html');
    const cover = it.has_cover
      ? '<img class="pin-cover" src="'+outputHref(it.name, 'cover.jpg')+'" alt="" loading="lazy"/>'
      : '<span class="pin-cover ph"></span>';
    return '<div class="pin-card">' +
      '<a class="pin-card-link" href="'+href+'" target="_blank" rel="noopener noreferrer" data-open-link="1">' +
        cover +
        '<span class="pin-title">'+escapeHtml(it.title || it.name)+'</span>' +
      '</a>' +
      '<button type="button" class="pin-card-unpin" data-pin-unpin="'+escapeHtml(it.name)+'" title="取消置顶" aria-label="取消置顶">&#10005;</button>' +
    '</div>';
  }).join('');
}

// ---- Settings ----
function saveSettings(url, noteMode, backend, llmKey, groqKey, dashscopeKey, batchMode){
  try {
    localStorage.setItem('vnotes_url', url);
    localStorage.setItem('vnotes_note_mode', noteMode);
    localStorage.setItem('vnotes_backend', backend);
    localStorage.removeItem('vnotes_llm_key');
    localStorage.removeItem('vnotes_groq_key');
    localStorage.removeItem('vnotes_dashscope_key');
    if(llmKey) sessionStorage.setItem('vnotes_llm_key', llmKey); else sessionStorage.removeItem('vnotes_llm_key');
    if(groqKey) sessionStorage.setItem('vnotes_groq_key', groqKey); else sessionStorage.removeItem('vnotes_groq_key');
    if(dashscopeKey) sessionStorage.setItem('vnotes_dashscope_key', dashscopeKey); else sessionStorage.removeItem('vnotes_dashscope_key');
    localStorage.setItem('vnotes_batch', batchMode ? '1' : '0');
  } catch(e) {}
}

function loadSettings(){
  try {
    document.getElementById('url').value = localStorage.getItem('vnotes_url') || '';
    document.getElementById('note-mode').value = localStorage.getItem('vnotes_note_mode') || '';
    const backend = localStorage.getItem('vnotes_backend') || '';
    document.getElementById('backend').value = backend;
    document.getElementById('llm-key').value = sessionStorage.getItem('vnotes_llm_key') || '';
    document.getElementById('groq-key').value = sessionStorage.getItem('vnotes_groq_key') || '';
    document.getElementById('dashscope-key').value = sessionStorage.getItem('vnotes_dashscope_key') || '';
    document.getElementById('batch-mode').checked = localStorage.getItem('vnotes_batch') === '1';
    updateBackendFields(backend);
  } catch(e) {}
}

function updateBackendFields(val){
  document.getElementById('groq-group').style.display = val==='groq'?'flex':'none';
  document.getElementById('dashscope-group').style.display = val==='paraformer'?'flex':'none';
  renderConfigStatus();
  updateSettingsSummary();
}

function updateSettingsSummary(){
  const el = document.getElementById('settings-summary');
  if(!el) return;
  const noteMode = document.getElementById('note-mode')?.value || '';
  const backend = document.getElementById('backend')?.value || '';
  const effective = backend || (CONFIG_STATUS && CONFIG_STATUS.default_backend) || '';
  const llmKey = document.getElementById('llm-key')?.value.trim() || '';
  const batch = document.getElementById('batch-mode')?.checked;
  const noteLabel = noteMode === 'detailed' ? '细致笔记' : noteMode === 'essence' ? '脉络精华' : '默认笔记模式';
  const backendLabel = backend
    ? (BACKEND_LABELS[backend] || backend)
    : (effective ? '默认 ' + (BACKEND_LABELS[effective] || effective) : '默认转写后端');
  const keyLabel = llmKey ? '临时 DeepSeek Key' : (CONFIG_STATUS && CONFIG_STATUS.has_llm_api_key ? '.env DeepSeek Key' : '缺 DeepSeek Key');
  el.textContent = [noteLabel, backendLabel, keyLabel, batch ? '批量' : '单条'].join(' · ');
}

function setupSettingsDrawer(){
  const toggle = document.getElementById('settings-toggle');
  const panel = document.getElementById('settings');
  if(!toggle || !panel) return;

  if(!document.getElementById('settings-summary')){
    const row = document.createElement('div');
    row.className = 'settings-row';
    toggle.parentNode.insertBefore(row, toggle);
    row.appendChild(toggle);
    const summary = document.createElement('div');
    summary.className = 'settings-summary';
    summary.id = 'settings-summary';
    summary.setAttribute('aria-live', 'polite');
    row.appendChild(summary);
  }

  document.getElementById('settings-backdrop')?.remove();
  panel.querySelector('.settings-panel-head')?.remove();
  panel.setAttribute('role', 'region');
  panel.removeAttribute('aria-modal');
  panel.removeAttribute('tabindex');

  function setSettingsOpen(open){
    const host = panel.closest('.input-card');
    toggle.classList.toggle('open', open);
    panel.classList.toggle('open', open);
    host?.classList.toggle('settings-open', open);
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    panel.setAttribute('aria-hidden', open ? 'false' : 'true');
  }

  if(!toggle.dataset.drawerBound){
    toggle.dataset.drawerBound = '1';
    toggle.addEventListener('click', () => setSettingsOpen(!panel.classList.contains('open')));
    document.addEventListener('keydown', e => {
      if(e.key === 'Escape' && panel.classList.contains('open')) setSettingsOpen(false);
    });
  }
  setSettingsOpen(panel.classList.contains('open'));
  updateSettingsSummary();
}

function setupThemePull(){
  let pull = document.getElementById('theme-pull');
  if(!pull){
    pull = document.createElement('button');
    pull.id = 'theme-pull';
    pull.className = 'theme-pull';
    pull.type = 'button';
    pull.setAttribute('aria-label', '切换夜间模式');
    pull.innerHTML = '<span class="pull-cord" aria-hidden="true"><span class="pull-line"></span><span class="pull-handle"></span></span>';
    document.body.appendChild(pull);
  }

  function applyTheme(theme, persist){
    const mode = theme === 'night' ? 'night' : 'day';
    document.documentElement.dataset.theme = mode;
    pull.setAttribute('aria-pressed', mode === 'night' ? 'true' : 'false');
    pull.title = mode === 'night' ? '切换白天模式' : '切换夜间模式';
    pull.setAttribute('aria-label', pull.title);
    if(persist) localStorage.setItem('vnotes_theme', mode);
  }

  const saved = localStorage.getItem('vnotes_theme');
  applyTheme(saved === 'night' ? 'night' : 'day', false);

  let dragging = false;
  let didDrag = false;
  let armed = false;
  let suppressClick = false;
  let themeTransitionBusy = false;
  let startY = 0;
  let startX = 0;
  const threshold = 58;
  const maxPull = 90;

  function setPull(px, sway){
    const amount = Math.max(0, Math.min(maxPull, px));
    pull.style.setProperty('--pull-extra', amount.toFixed(0) + 'px');
    pull.style.setProperty('--handle-squeeze', (1 - Math.min(amount, threshold) / threshold * .05).toFixed(3));
    if(Number.isFinite(sway)){
      pull.style.setProperty('--pull-sway', Math.max(-8, Math.min(8, sway)).toFixed(2) + 'deg');
    }
    armed = amount >= threshold;
    pull.classList.toggle('armed', armed);
  }

  function playThemeTransition(nextTheme){
    if(themeTransitionBusy) return;
    const overlay = document.getElementById('curtain-overlay');
    const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if(!overlay || reduceMotion){
      applyTheme(nextTheme, true);
      return;
    }
    themeTransitionBusy = true;
    overlay.style.display = 'block';
    overlay.classList.remove('curtain-done','theme-transition');
    void overlay.offsetWidth;
    overlay.classList.add('theme-transition');
    window.setTimeout(() => applyTheme(nextTheme, true), 520);
    window.setTimeout(() => {
      overlay.classList.remove('theme-transition');
      overlay.classList.add('curtain-done');
      overlay.style.display = 'none';
      themeTransitionBusy = false;
    }, 1260);
  }

  function toggleTheme(){
    const nextTheme = document.documentElement.dataset.theme === 'night' ? 'day' : 'night';
    playThemeTransition(nextTheme);
  }

  if(pull.dataset.themeBound) return;
  pull.dataset.themeBound = '1';
  pull.addEventListener('click', () => {
    if(suppressClick){
      suppressClick = false;
      return;
    }
    toggleTheme();
  });
  pull.addEventListener('pointerdown', e => {
    if(themeTransitionBusy) return;
    dragging = true;
    didDrag = false;
    armed = false;
    startY = e.clientY;
    startX = e.clientX;
    pull.classList.remove('returning');
    pull.classList.add('dragging');
    pull.setPointerCapture?.(e.pointerId);
  });
  pull.addEventListener('pointermove', e => {
    if(!dragging) return;
    const dy = Math.max(0, e.clientY - startY);
    const dx = e.clientX - startX;
    if(dy > 4) didDrag = true;
    setPull(dy, -dx / 7);
  });
  pull.addEventListener('pointerup', e => {
    if(!dragging) return;
    dragging = false;
    if(didDrag) suppressClick = true;
    if(armed) toggleTheme();
    setPull(0, 0);
    pull.classList.remove('dragging','armed');
    pull.classList.add('returning');
    window.setTimeout(() => pull.classList.remove('returning'), 860);
    pull.releasePointerCapture?.(e.pointerId);
  });
  pull.addEventListener('pointercancel', e => {
    dragging = false;
    setPull(0, 0);
    pull.classList.remove('dragging','armed');
    pull.classList.add('returning');
    window.setTimeout(() => pull.classList.remove('returning'), 860);
    pull.releasePointerCapture?.(e.pointerId);
  });
}

// ---- Events ----
setupThemePull();
setupSettingsDrawer();
document.getElementById('gen-btn').addEventListener('click', startGen);
const urlInputEl = document.getElementById('url');
urlInputEl.addEventListener('keydown', e => { if(e.key==='Enter') startGen(); });
urlInputEl.addEventListener('paste', () => window.setTimeout(() => normalizeUrlInput(true), 0));
urlInputEl.addEventListener('blur', () => normalizeUrlInput(false));

document.getElementById('backend').addEventListener('change', e => updateBackendFields(e.target.value));
['llm-key','groq-key','dashscope-key'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => {
    renderConfigStatus();
    updateSettingsSummary();
  });
});
['note-mode','batch-mode'].forEach(id => {
  document.getElementById(id).addEventListener('change', updateSettingsSummary);
});

// ---- 滚动渐入观察器 ----
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if(entry.isIntersecting){
      entry.target.classList.add('revealed');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

function observeReveal(){
  document.querySelectorAll('.reveal-on-scroll:not(.revealed)').forEach(el => {
    revealObserver.observe(el);
  });
}

// ---- 铭牌彩蛋：overscroll 回弹时弹出 ----
  (function(){
    const badge = document.getElementById('rig-badge');
    if(!badge) return;
    let pullTimer = null;
    let isPulling = false;

    function updateTopState(){
      if(window.scrollY <= 0){
        badge.classList.add('at-top');
      } else {
        badge.classList.remove('at-top');
        badge.classList.remove('pulled');
      }
    }

    // 滚动时更新顶部状态
    window.addEventListener('scroll', updateTopState, {passive:true});
    updateTopState();

    // 在顶部继续向上 wheel → 弹出铭牌
    window.addEventListener('wheel', function(e){
      if(window.scrollY <= 0 && e.deltaY < 0){
        if(!isPulling){
          isPulling = true;
          badge.classList.add('pulled');
        }
        clearTimeout(pullTimer);
        pullTimer = setTimeout(function(){
          badge.classList.remove('pulled');
          isPulling = false;
        }, 900);
      } else if(e.deltaY > 0 && isPulling){
        badge.classList.remove('pulled');
        isPulling = false;
      }
    }, {passive:true});

    // 移动端 touch 支持
    let touchStartY = 0;
    window.addEventListener('touchstart', function(e){
      touchStartY = e.touches[0].clientY;
    }, {passive:true});
    window.addEventListener('touchmove', function(e){
      if(window.scrollY <= 0){
        const dy = e.touches[0].clientY - touchStartY;
        if(dy > 10 && !isPulling){
          isPulling = true;
          badge.classList.add('pulled');
        }
      }
    }, {passive:true});
    window.addEventListener('touchend', function(){
      if(isPulling){
        clearTimeout(pullTimer);
        pullTimer = setTimeout(function(){
          badge.classList.remove('pulled');
          isPulling = false;
        }, 600);
      }
    }, {passive:true});
  })();

// ---- 滚动视差：聚光灯随滚动微妙偏移 ----
  (function(){
    let ticking = false;
    const spotRig = document.querySelector('.spot-rig');
    const spotCone = document.querySelector('.spot-cone');
    function onScroll(){
      if(ticking) return;
      ticking = true;
      requestAnimationFrame(function(){
        const y = window.scrollY;
        // 棚顶随滚动微微下移（增强固定感）
        if(spotRig) spotRig.style.boxShadow = '0 ' + (8 + Math.min(y*.02,6)) + 'px 28px rgba(0,0,0,.4),inset 0 -1px 0 rgba(255,255,255,.05)';
        ticking = false;
      });
    }
    window.addEventListener('scroll', onScroll, {passive:true});
  })();

// ---- Init ----
initPipeline();
loadSettings();
loadConfigStatus();
loadHistory();
observeReveal();
</script>
    <div class="toast-box" id="toast-box" aria-live="polite"></div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("VNOTES_SERVER_PORT", "7458"))
    print(f"vnotes Web UI: http://localhost:{port}")
    uvicorn.run("vnotes.server:app", host="0.0.0.0", port=port, reload=False)
