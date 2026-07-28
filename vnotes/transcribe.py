"""转写模块：支持四种后端，用环境变量 VNOTES_TRANSCRIBE_BACKEND 切换。

  - faster-whisper（默认）：轻量，不装 torch，模型下载到 D 盘，CPU 也能跑
  - vosk：低内存离线兜底，中文小模型约 50MB
  - paraformer：阿里云 Paraformer 云端 API，国内直连，按秒计费，速度快
  - groq：云端 API，零空间占用，速度极快，需代理
  - openai-whisper：原方案，需 torch（占空间大）

中文用 turbo(zh)，英文用 small.en 并翻成中文；末段时长校验。
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from .config import Config
from .util import log, write_json, run


# ============================================================
#  Vosk（低内存离线兜底）
# ============================================================
def _find_vosk_model(cfg: Config) -> Path:
    candidates = [
        Path(getattr(cfg, "vosk_model_dir", "") or ""),
        Path(getattr(cfg, "whisper_model_dir", "") or "") / "vosk-model-small-cn-0.22",
        Path("D:/vnotes_models/vosk-model-small-cn-0.22"),
    ]
    for cand in candidates:
        if cand and cand.exists() and (cand / "conf").exists() and (cand / "am").exists():
            return cand
    raise RuntimeError(
        "未找到 Vosk 中文模型。请运行：python download_model.py vosk-cn D:/vnotes_models，"
        "或设置 VNOTES_VOSK_MODEL_DIR。"
    )


def _clean_vosk_text(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def _vosk_segment(payload: dict[str, Any], fallback_start: float,
                  fallback_end: float) -> dict[str, Any] | None:
    text = _clean_vosk_text(payload.get("text", ""))
    if not text:
        return None
    words = payload.get("result") or []
    if words:
        start = float(words[0].get("start", fallback_start))
        end = float(words[-1].get("end", fallback_end))
    else:
        start, end = fallback_start, fallback_end
    return {
        "start": round(start, 2),
        "end": round(max(end, start + 0.1), 2),
        "text": text,
    }


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / max(wf.getframerate(), 1)
    except Exception:
        return None


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _cache_fresh(path: Path, source_mtime: float, min_size: int = 1000) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return stat.st_size > min_size and stat.st_mtime >= source_mtime


def _ensure_vosk_wav(cfg: Config, source: Path, wav_path: Path,
                     expected_dur: float | None) -> Path:
    needs_wav = not _cache_fresh(wav_path, _safe_mtime(source))
    if not needs_wav and expected_dur:
        cached_dur = _wav_duration(wav_path)
        if cached_dur is None or abs(cached_dur - expected_dur) > max(2.0, expected_dur * 0.02):
            log.warn("transcribe", f"Vosk WAV 缓存时长不符（cached={cached_dur}s, audio={expected_dur}s），重新转换")
            needs_wav = True
    if needs_wav:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("transcribe", f"Vosk 兜底：转换 16k 单声道 WAV（{wav_path.name}）")
        run([
            cfg.ffmpeg, "-y", "-loglevel", "error",
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", str(wav_path),
        ], tag="vosk/wav", check=True, timeout=600)
    return wav_path


def _vosk_wav_chunks(cfg: Config, audio_path: Path,
                     audio_dur: float | None) -> list[tuple[float, Path]]:
    chunk_sec = max(30, int(getattr(cfg, "transcribe_chunk_seconds", 300)))
    if not audio_dur or audio_dur <= chunk_sec:
        wav_path = audio_path.with_name(f"{audio_path.stem}.vosk.16k.wav")
        return [(0.0, _ensure_vosk_wav(cfg, audio_path, wav_path, audio_dur))]

    chunk_dir = audio_path.parent / "vosk_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    source_mtime = _safe_mtime(audio_path)
    chunks: list[tuple[float, Path]] = []
    start = 0.0
    idx = 1
    while start < audio_dur:
        dur = min(chunk_sec, audio_dur - start)
        wav_path = chunk_dir / f"chunk_{idx:03d}.wav"
        if not _cache_fresh(wav_path, source_mtime):
            log.info("transcribe", f"Vosk 切分音频 {idx}：{start:.0f}s + {dur:.0f}s")
            run([
                cfg.ffmpeg, "-y", "-loglevel", "error",
                "-ss", str(start), "-t", str(dur), "-i", str(audio_path),
                "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path),
            ], tag="vosk/split", check=True, timeout=180)
        chunks.append((start, wav_path))
        start += chunk_sec
        idx += 1
    return chunks


def _recognize_vosk_wav(model, wav_path: Path, offset: float,
                        total_hint: float | None) -> list[dict[str, Any]]:
    import vosk

    segments: list[dict[str, Any]] = []
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise RuntimeError("Vosk 需要 16-bit 单声道 WAV")
        sample_rate = wf.getframerate()
        rec = vosk.KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)
        bytes_per_sec = sample_rate * wf.getsampwidth() * wf.getnchannels()
        processed = 0
        last_boundary = 0.0
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            processed += len(data)
            cur_sec = processed / bytes_per_sec
            if rec.AcceptWaveform(data):
                payload = json.loads(rec.Result())
                seg = _vosk_segment(payload, last_boundary, cur_sec)
                if seg:
                    segments.append(_offset_segments([seg], offset)[0])
                    last_boundary = seg["end"]
        payload = json.loads(rec.FinalResult())
        total_sec = wf.getnframes() / max(sample_rate, 1)
        seg = _vosk_segment(payload, last_boundary, total_sec)
        if seg:
            segments.append(_offset_segments([seg], offset)[0])
    if total_hint:
        log.info("transcribe", f"Vosk 分片完成：{offset + (_wav_duration(wav_path) or 0):.0f}/{total_hint:.0f}s")
    return segments


def _transcribe_vosk(cfg: Config, audio_path: Path, audio_dur: float | None) -> dict[str, Any]:
    try:
        import vosk
    except Exception as e:
        raise RuntimeError("未安装 vosk 包（pip install vosk）") from e

    model_dir = _find_vosk_model(cfg)
    log.info("transcribe", f"Vosk 加载中文小模型：{model_dir}")
    vosk.SetLogLevel(-1)
    model = vosk.Model(str(model_dir))

    chunks = _vosk_wav_chunks(cfg, audio_path, audio_dur)
    if len(chunks) > 1:
        log.info("transcribe", f"Vosk 分片转写：{len(chunks)} 片")

    ckpt_dir = audio_path.parent / "vosk_chunks"
    segments: list[dict[str, Any]] = []
    for idx, (offset, wav_path) in enumerate(chunks, 1):
        ckpt = ckpt_dir / f"chunk_{idx:03d}.json"
        if _cache_fresh(ckpt, _safe_mtime(wav_path), min_size=10):
            try:
                data = json.loads(ckpt.read_text(encoding="utf-8"))
                segments.extend(data.get("segments", []))
                log.info("transcribe", f"跳过已完成 Vosk 分片 {idx}/{len(chunks)}")
                continue
            except Exception:
                pass
        log.info("transcribe", f"Vosk 转写分片 {idx}/{len(chunks)}（offset={offset:.0f}s）")
        segs = _recognize_vosk_wav(model, wav_path, offset, audio_dur)
        segments.extend(segs)
        if len(chunks) > 1:
            write_json(ckpt, {"offset": offset, "segments": segs})

    if not segments:
        raise RuntimeError("Vosk 转写结果为空，无法继续分析。")

    return {
        "language": "zh",
        "model": f"vosk/{model_dir.name}",
        "device": "cpu",
        "segments": segments,
    }


# ============================================================
#  faster-whisper（本地轻量，推荐）
# ============================================================
_FASTER_MODEL_REPOS: dict[str, list[str]] = {
    "tiny": ["Systran/faster-whisper-tiny"],
    "tiny.en": ["Systran/faster-whisper-tiny.en"],
    "base": ["Systran/faster-whisper-base"],
    "base.en": ["Systran/faster-whisper-base.en"],
    "small": ["Systran/faster-whisper-small"],
    "small.en": ["Systran/faster-whisper-small.en"],
    "medium": ["Systran/faster-whisper-medium"],
    "medium.en": ["Systran/faster-whisper-medium.en"],
    "large-v3": ["Systran/faster-whisper-large-v3"],
    "large-v3-turbo": [
        "Systran/faster-whisper-large-v3-turbo",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    ],
    "turbo": [
        "Systran/faster-whisper-large-v3-turbo",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    ],
}


def _has_ctranslate_model(path: Path) -> bool:
    return (path / "model.bin").exists() and (
        (path / "config.json").exists() or (path / "model.bin").stat().st_size > 10_000_000
    )


def _candidate_repos(model_name: str) -> list[str]:
    if "/" in model_name:
        return [model_name]
    return _FASTER_MODEL_REPOS.get(model_name.lower(), [])


def _find_local_faster_model(model_name: str, model_dir: str | None) -> str | None:
    direct = Path(model_name)
    if direct.exists() and _has_ctranslate_model(direct):
        return str(direct)
    if not model_dir:
        return None

    root = Path(model_dir)
    candidates = [root / model_name]
    for repo in _candidate_repos(model_name):
        repo_dir = root / ("models--" + repo.replace("/", "--"))
        candidates.append(repo_dir / "snapshots" / "main")
        snap_root = repo_dir / "snapshots"
        if snap_root.exists():
            snapshots = sorted(
                [p for p in snap_root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            candidates.extend(snapshots)

    for cand in candidates:
        if cand.exists() and _has_ctranslate_model(cand):
            return str(cand)
    return None


def _looks_like_cuda_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(x in text for x in ("cuda", "cudnn", "cublas", "gpu", "compute capability"))


def _looks_like_memory_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(x in text for x in ("mkl_malloc", "failed to allocate memory", "out of memory", "bad allocation"))


def _available_pagefile_mb() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPageFile // (1024 * 1024))
    except Exception:
        return None
    return None


def _faster_preflight_skip_reason(cfg: Config) -> str | None:
    pagefile_mb = _available_pagefile_mb()
    if cfg.whisper_device == "cuda" and not _cuda_runtime_available():
        detail = f"，可用虚拟内存 {pagefile_mb}MB" if pagefile_mb is not None else ""
        return f"CUDA 运行库不完整{detail}"
    if pagefile_mb is not None and pagefile_mb < 1024:
        return f"可用虚拟内存仅 {pagefile_mb}MB，faster-whisper tiny 也可能无法加载"
    return None


def _cuda_runtime_available() -> bool:
    if os.name != "nt":
        return True
    found: set[str] = set()
    names = {"cublas64_12.dll", "cudnn_ops64_9.dll", "cudnn_ops_infer64_8.dll"}
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if not raw:
            continue
        folder = Path(raw)
        for name in names:
            if (folder / name).exists():
                found.add(name)
    return "cublas64_12.dll" in found and (
        "cudnn_ops64_9.dll" in found or "cudnn_ops_infer64_8.dll" in found
    )


def _transcribe_options(device: str, language: str | None) -> dict[str, Any]:
    if device == "cpu":
        return {
            "beam_size": 1,
            "best_of": 1,
            "language": language,
            "vad_filter": False,
            "chunk_length": 10,
            "condition_on_previous_text": False,
        }
    return {
        "beam_size": 5,
        "language": language,
        "vad_filter": True,
    }


def _faster_model_error_message(cfg: Config, model_name: str, err: Exception) -> str:
    model_dir = cfg.whisper_model_dir or "D:/vnotes_models"
    endpoint = cfg.hf_endpoint or os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
    return (
        f"faster-whisper 模型 {model_name!r} 未能加载。当前本地目录：{model_dir}；"
        f"已尝试通过 {endpoint} 下载/加载，但仍失败。\n"
        f"请先运行：python download_model.py {model_name} {model_dir}\n"
        "或者在页面设置里把转写后端切到 paraformer/Groq 并填写对应 API Key。\n"
        f"原始错误：{err}"
    )


def _faster_runtime_error_message(cuda_runtime_missing: bool, err: Exception) -> str:
    lines = ["faster-whisper 转写失败。"]
    if cuda_runtime_missing:
        lines.append("本机 CUDA 运行库不完整，缺少 cublas/cudnn DLL，已自动改用 CPU。")
    if _looks_like_memory_error(err):
        lines.append("CPU 兜底仍然内存不足，通常是可用内存或 Windows 虚拟内存/分页文件太紧。")
    lines.append("处理建议：关闭占内存程序或增大分页文件后重试；要走 GPU，请安装 CUDA 12/cuDNN 并确保 DLL 在 PATH；也可以切到 paraformer/Groq 云端转写后端。")
    lines.append(f"原始错误：{err}")
    return "\n".join(lines)


def _collect_faster_segments(model, audio_path: Path, device: str, language: str | None,
                             cuda_runtime_missing: bool):
    try:
        segments_iter, info = model.transcribe(str(audio_path), **_transcribe_options(device, language))
        segments = [
            {
                "start": round(float(s.start), 2),
                "end": round(float(s.end), 2),
                "text": (s.text or "").strip(),
            }
            for s in segments_iter
        ]
        return info, segments
    except Exception as e:
        raise RuntimeError(_faster_runtime_error_message(cuda_runtime_missing, e)) from e


def _offset_segments(segments: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(s.get("start", 0)) + offset, 2),
            "end": round(float(s.get("end", 0)) + offset, 2),
            "text": text,
        })
    return out


def _audio_chunks(cfg: Config, audio_path: Path, work_dir: Path,
                  audio_dur: float | None, chunk_sec: int) -> list[tuple[float, Path]]:
    if not audio_dur or audio_dur <= chunk_sec:
        return [(0.0, audio_path)]

    chunk_dir = work_dir / "transcribe_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    source_mtime = _safe_mtime(audio_path)
    chunks: list[tuple[float, Path]] = []
    start = 0.0
    idx = 1
    while start < audio_dur:
        dur = min(chunk_sec, audio_dur - start)
        copied = chunk_dir / f"chunk_{idx:03d}{audio_path.suffix}"
        wav = chunk_dir / f"chunk_{idx:03d}.wav"
        out = copied if _cache_fresh(copied, source_mtime) else wav
        if not _cache_fresh(out, source_mtime):
            log.info("transcribe", f"切分音频 {idx}：{start:.0f}s + {dur:.0f}s")
            try:
                run([
                    cfg.ffmpeg, "-y", "-loglevel", "error",
                    "-ss", str(start), "-t", str(dur), "-i", str(audio_path),
                    "-vn", "-c", "copy", str(copied),
                ], tag="transcribe/split", check=True, timeout=120)
                out = copied
            except Exception as e:
                log.warn("transcribe", f"无损切片失败，改用 WAV 切片：{e}")
                run([
                    cfg.ffmpeg, "-y", "-loglevel", "error",
                    "-ss", str(start), "-t", str(dur), "-i", str(audio_path),
                    "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav),
                ], tag="transcribe/split-wav", check=True, timeout=180)
                out = wav
        chunks.append((start, out))
        start += chunk_sec
        idx += 1
    return chunks


def _transcribe_faster_chunked(model, cfg: Config, audio_path: Path, work_dir: Path,
                               audio_dur: float | None, device: str,
                               cuda_runtime_missing: bool) -> tuple[str, list[dict[str, Any]]]:
    chunk_sec = max(30, int(getattr(cfg, "transcribe_chunk_seconds", 300)))
    chunks = _audio_chunks(cfg, audio_path, work_dir, audio_dur, chunk_sec)
    if len(chunks) <= 1:
        info, segments = _collect_faster_segments(model, audio_path, device, None, cuda_runtime_missing)
        return info.language, segments

    log.info("transcribe", f"长音频分片转写：{len(chunks)} 片，每片约 {chunk_sec}s")
    ckpt_dir = work_dir / "transcribe_chunks"
    all_segments: list[dict[str, Any]] = []
    detected_lang: str | None = None
    force_lang: str | None = None

    for idx, (offset, chunk_path) in enumerate(chunks, 1):
        ckpt = ckpt_dir / f"chunk_{idx:03d}.json"
        if _cache_fresh(ckpt, _safe_mtime(chunk_path), min_size=10):
            try:
                import json
                data = json.loads(ckpt.read_text(encoding="utf-8"))
                lang = data.get("language")
                if lang and not detected_lang:
                    detected_lang = lang
                    force_lang = "zh" if lang.startswith("zh") else ("en" if lang == "en" else None)
                segs = data.get("segments", [])
                all_segments.extend(segs)
                log.info("transcribe", f"跳过已完成转写分片 {idx}/{len(chunks)}")
                continue
            except Exception:
                pass

        lang_arg = None if detected_lang is None else force_lang
        log.info("transcribe", f"转写分片 {idx}/{len(chunks)}（offset={offset:.0f}s）")
        info, segs = _collect_faster_segments(model, chunk_path, device, lang_arg, cuda_runtime_missing)
        if detected_lang is None:
            detected_lang = info.language
            force_lang = "zh" if detected_lang.startswith("zh") else ("en" if detected_lang == "en" else None)
            log.info("transcribe", f"检测语言 = {detected_lang}, 概率 = {info.language_probability:.2f}")
        segs = _offset_segments(segs, offset)
        all_segments.extend(segs)
        write_json(ckpt, {"language": detected_lang or info.language, "offset": offset, "segments": segs})

    return detected_lang or "unknown", all_segments


def _load_faster_model(cfg: Config, model_name: str, device: str,
                       compute_type: str, model_dir: str | None):
    from faster_whisper import WhisperModel

    def load(model_id_or_path: str, *, local_files_only: bool, use_device: str, use_compute: str):
        return WhisperModel(
            model_id_or_path,
            device=use_device,
            compute_type=use_compute,
            download_root=model_dir,
            local_files_only=local_files_only,
            cpu_threads=1 if use_device == "cpu" else 0,
            num_workers=1,
        )

    local_path = _find_local_faster_model(model_name, model_dir)
    if local_path:
        log.info("transcribe", f"使用本地 faster-whisper 模型：{local_path}")
        try:
            return load(local_path, local_files_only=False, use_device=device, use_compute=compute_type), device, compute_type
        except Exception as e:
            if device == "cuda" and _looks_like_cuda_error(e):
                log.warn("transcribe", f"CUDA 加载失败，改用 CPU/int8：{e}")
                return load(local_path, local_files_only=False, use_device="cpu", use_compute="int8"), "cpu", "int8"
            raise RuntimeError(f"本地 faster-whisper 模型加载失败：{local_path}\n原始错误：{e}") from e

    try:
        log.info("transcribe", f"检查本地缓存中的 faster-whisper 模型 {model_name}")
        return load(model_name, local_files_only=True, use_device=device, use_compute=compute_type), device, compute_type
    except Exception:
        log.warn("transcribe", f"本地未找到 faster-whisper 模型 {model_name}，准备联网下载/加载")

    if cfg.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = cfg.hf_endpoint
    if os.environ.get("HF_ENDPOINT"):
        log.info("transcribe", f"使用 Hugging Face 镜像：{os.environ['HF_ENDPOINT']}")

    try:
        return load(model_name, local_files_only=False, use_device=device, use_compute=compute_type), device, compute_type
    except Exception as e:
        if device == "cuda" and _looks_like_cuda_error(e):
            log.warn("transcribe", f"CUDA 加载失败，改用 CPU/int8：{e}")
            try:
                return load(model_name, local_files_only=False, use_device="cpu", use_compute="int8"), "cpu", "int8"
            except Exception as cpu_err:
                raise RuntimeError(_faster_model_error_message(cfg, model_name, cpu_err)) from cpu_err
        raise RuntimeError(_faster_model_error_message(cfg, model_name, e)) from e


def _fallback_model_names(primary: str) -> list[str]:
    names = [primary]
    lower = primary.lower()
    if lower.endswith(".en"):
        names.extend(["tiny.en"])
    else:
        names.extend(["tiny"])
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _load_first_faster_model(cfg: Config, model_names: list[str], device: str,
                             compute_type: str, model_dir: str | None):
    errors: list[str] = []
    for idx, name in enumerate(model_names):
        try:
            model, actual_device, actual_compute = _load_faster_model(cfg, name, device, compute_type, model_dir)
            if idx > 0:
                log.warn("transcribe", f"已自动降级到更小的 faster-whisper 模型：{name}")
            return model, actual_device, actual_compute, name
        except Exception as e:
            errors.append(f"{name}: {e}")
            if idx < len(model_names) - 1:
                reason = "内存不足" if _looks_like_memory_error(e) else "加载失败"
                log.warn("transcribe", f"模型 {name} {reason}，尝试更小模型")
                continue
            raise RuntimeError("faster-whisper 所有候选模型都加载失败：\n" + "\n\n".join(errors)) from e


def _transcribe_faster(cfg: Config, audio_path: Path, audio_dur: float | None) -> dict[str, Any]:
    device = cfg.whisper_device
    # CPU 用 int8（小+快），GPU 用 float16
    compute_type = "float16" if device == "cuda" else "int8"
    cuda_runtime_missing = device == "cuda" and not _cuda_runtime_available()
    if cuda_runtime_missing:
        log.warn("transcribe", "CUDA 运行时 DLL 不完整，自动改用 CPU/float32")
        device = "cpu"
        compute_type = "float32"
    model_dir = cfg.whisper_model_dir or None
    if model_dir:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = model_dir

    # 先用多语言模型做语言检测
    log.info("transcribe", f"faster-whisper 加载模型 {cfg.whisper_model_zh}（device={device}, compute={compute_type}）")
    model_candidates = _fallback_model_names(cfg.whisper_model_zh)
    if device == "cpu" and cfg.whisper_model_zh.lower() not in {"tiny", "tiny.en"}:
        log.warn("transcribe", "CPU 模式内存紧张，直接使用 tiny 模型兜底")
        model_candidates = ["tiny"]
    model = None
    try:
        try:
            model, device, compute_type, model_name_used = _load_first_faster_model(
                cfg, model_candidates, device, compute_type, model_dir
            )
        except Exception as e:
            raise RuntimeError(_faster_runtime_error_message(cuda_runtime_missing, e)) from e

        chunk_sec = max(30, int(getattr(cfg, "transcribe_chunk_seconds", 300)))
        if audio_dur and audio_dur > chunk_sec:
            lang, segments = _transcribe_faster_chunked(
                model, cfg, audio_path, audio_path.parent, audio_dur, device, cuda_runtime_missing
            )
        else:
            # 语言检测 + 转写一次完成。中文不再二次重跑，避免 CPU 模式耗时翻倍。
            info, segments = _collect_faster_segments(model, audio_path, device, None, cuda_runtime_missing)
            lang = info.language
            log.info("transcribe", f"检测语言 = {lang}, 概率 = {info.language_probability:.2f}")

        # 如果短英文音频，换英文模型重转；长音频分片模式优先保持同一模型，避免重复加载。
        if (not audio_dur or audio_dur <= chunk_sec) and lang == "en" and cfg.whisper_model_en != model_name_used:
            log.info("transcribe", f"英文音频，切换到 {cfg.whisper_model_en}")
            del model
            model = None
            en_candidates = _fallback_model_names(cfg.whisper_model_en)
            if device == "cpu" and cfg.whisper_model_en.lower() != "tiny.en":
                log.warn("transcribe", "CPU 模式内存紧张，英文音频直接使用 tiny.en 模型兜底")
                en_candidates = ["tiny.en", "tiny"]
            try:
                model, device, compute_type, model_name_used = _load_first_faster_model(
                    cfg, en_candidates, device, compute_type, model_dir
                )
            except Exception as e:
                raise RuntimeError(_faster_runtime_error_message(cuda_runtime_missing, e)) from e
            info, segments = _collect_faster_segments(model, audio_path, device, "en", cuda_runtime_missing)
    finally:
        if model is not None:
            del model
        gc.collect()

    if not segments:
        raise RuntimeError("转写结果为空，无法继续分析。请换用 paraformer/Groq 云端转写，或检查音频是否有人声。")

    return {
        "language": lang,
        "model": f"faster-whisper/{model_name_used}",
        "device": device,
        "segments": segments,
    }


# ============================================================
#  Groq 云端 API（零空间，极快）
# ============================================================
def _transcribe_groq(cfg: Config, audio_path: Path, audio_dur: float | None) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("未安装 openai 包（pip install openai）")

    client = OpenAI(api_key=cfg.groq_api_key, base_url="https://api.groq.com/openai/v1")
    model = cfg.groq_model
    log.info("transcribe", f"Groq API 转写（model={model}）")

    # Groq 文件限制 25MB；超限则分片
    file_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_mb <= 24:
        return _groq_transcribe_one(client, model, audio_path, 0.0)

    # 分片：用 ffmpeg -ss -c copy 切割（不需编码器）
    log.info("transcribe", f"音频 {file_mb:.1f}MB > 24MB，分片上传")
    return _groq_transcribe_chunked(client, model, audio_path, cfg, audio_dur)


def _obj_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _groq_transcribe_one(client, model: str, audio_path: Path, offset: float) -> dict:
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            temperature=0.0,
        )
    segments = []
    for s in getattr(result, "segments", []) or []:
        segments.append({
            "start": round(float(_obj_value(s, "start", _obj_value(s, "t", 0))) + offset, 2),
            "end": round(float(_obj_value(s, "end", 0)) + offset, 2),
            "text": (_obj_value(s, "text", "") or "").strip(),
        })
    lang = getattr(result, "language", "unknown") or "unknown"
    return {"language": lang, "model": f"groq/{model}", "device": "cloud", "segments": segments}


def _groq_transcribe_chunked(client, model: str, audio_path: Path,
                             cfg: Config, audio_dur: float | None) -> dict:
    """大文件分片：用 ffmpeg 切成 ~20 分钟片段分别转写，时间戳累加偏移。"""
    chunk_sec = 1200  # 20 分钟/片
    total_dur = audio_dur or 3600
    tmpdir = Path(tempfile.mkdtemp(prefix="vnotes_groq_"))
    all_segments: list[dict] = []
    lang = "unknown"

    t = 0.0
    idx = 1
    while t < total_dur:
        chunk_path = tmpdir / f"chunk_{idx:03d}.{audio_path.suffix.lstrip('.')}"
        # ffmpeg -ss -t -c copy 切片（不需编码器）
        try:
            run([cfg.ffmpeg, "-y", "-loglevel", "error",
                 "-ss", str(t), "-t", str(chunk_sec),
                 "-i", str(audio_path), "-c", "copy", str(chunk_path)],
                tag="groq/chunk", check=True, timeout=60)
        except Exception as e:
            log.warn("transcribe", f"切片 {idx} 失败：{e}")
            break
        if not chunk_path.exists() or chunk_path.stat().st_size < 1000:
            break
        log.info("transcribe", f"转写分片 {idx}（offset={t:.0f}s）")
        try:
            r = _groq_transcribe_one(client, model, chunk_path, t)
            if lang == "unknown":
                lang = r["language"]
            all_segments.extend(r["segments"])
        except Exception as e:
            log.warn("transcribe", f"分片 {idx} 转写失败：{e}")
        t += chunk_sec
        idx += 1

    # 清理临时文件
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return {"language": lang, "model": f"groq/{model}", "device": "cloud", "segments": all_segments}


# ============================================================
#  阿里云 Paraformer（国内直连，按秒计费，推荐）
# ============================================================
def _transcribe_paraformer(cfg: Config, audio_path: Path, audio_dur: float | None) -> dict[str, Any]:
    """阿里云 DashScope Paraformer 录音文件识别。
    
    国内直连无需代理，¥0.00008/秒，20分钟音频约¥0.10。
    需安装：pip install dashscope
    需配置：VNOTES_DASHSCOPE_API_KEY
    """
    try:
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
    except ImportError:
        raise RuntimeError(
            "未安装 dashscope 包（pip install dashscope）"
        )

    api_key = cfg.dashscope_api_key
    if not api_key:
        raise RuntimeError("Paraformer 后端需要 VNOTES_DASHSCOPE_API_KEY")

    dashscope.api_key = api_key
    model = cfg.paraformer_model
    log.info("transcribe", f"Paraformer 云端转写（model={model}）")

    # Paraformer 录音文件识别接口
    # 支持本地文件路径或 URL，返回带时间戳的 segments
    import tempfile
    import time as _time

    all_segments: list[dict] = []
    lang = "unknown"

    try:
        # 使用 Recognition 接口（实时流式，但可传文件）
        # 对大文件更稳定的方式是 Paraformer 录音文件识别
        from dashscope.audio.asr import Transcription
        
        # 录音文件识别（异步接口，适合长音频）
        log.info("transcribe", f"提交音频到阿里云（{audio_path.name}）")
        task = Transcription.async_call(
            model=model,
            file_urls=[str(audio_path.resolve())],
            language_hints=["zh", "en"],
            disfluency_removal_enabled=True,  # 去语气词
            param_constraints={
                "audio_format": audio_path.suffix.lstrip("."),
            }
        )
        
        # 轮询等待完成
        max_wait = 600  # 最多等 10 分钟
        waited = 0
        while waited < max_wait:
            result = Transcription.fetch(task.output.task_id)
            if result.status_code == 200:
                task_status = result.output.task_status
                if task_status == "SUCCEEDED":
                    break
                elif task_status == "FAILED":
                    raise RuntimeError(f"Paraformer 转写失败：{result.output}")
            _time.sleep(5)
            waited += 5
            if waited % 30 == 0:
                log.info("transcribe", f"等待转写完成…（{waited}s）")
        else:
            raise RuntimeError("Paraformer 转写超时（10 分钟）")

        # 解析结果
        results = result.output.get("results", [])
        if results:
            transcription_url = results[0].get("transcription_url", "")
            if transcription_url:
                # 下载 JSON 结果
                import requests
                resp = requests.get(transcription_url, timeout=30)
                trans_data = resp.json()
                lang = trans_data.get("language", "zh")
                # Paraformer 返回的格式：sentences 数组
                for sent in trans_data.get("sentences", []):
                    all_segments.append({
                        "start": round(float(sent.get("begin_time", 0)) / 1000, 2),
                        "end": round(float(sent.get("end_time", 0)) / 1000, 2),
                        "text": (sent.get("text", "") or "").strip(),
                    })

    except Exception as e:
        if "async_call" in str(e) or "file_urls" in str(e):
            # 如果异步接口不支持本地文件，回退到 Recognition 实时接口
            log.warn("transcribe", f"异步接口异常，回退到实时识别：{e}")
            return _paraformer_realtime(cfg, audio_path, audio_dur)
        raise

    if not all_segments:
        log.warn("transcribe", "Paraformer 返回空结果，回退到实时识别")
        return _paraformer_realtime(cfg, audio_path, audio_dur)

    return {
        "language": lang,
        "model": f"paraformer/{model}",
        "device": "cloud",
        "segments": all_segments,
    }


def _paraformer_realtime(cfg: Config, audio_path: Path, audio_dur: float | None) -> dict[str, Any]:
    """Paraformer 实时识别（本地文件流式推送），作为异步接口的兜底。"""
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
    import wave
    import json

    dashscope.api_key = cfg.dashscope_api_key
    model = cfg.paraformer_model

    all_segments: list[dict] = []
    lang = "zh"

    class Callback(RecognitionCallback):
        def on_open(self):
            log.info("transcribe", "Paraformer 实时连接已建立")
        def on_complete(self):
            log.info("transcribe", "Paraformer 实时识别完成")
        def on_error(self, result: RecognitionResult):
            log.error("transcribe", f"Paraformer 错误：{result}")
        def on_event(self, result: RecognitionResult):
            if result.is_sentence:
                payload = result.get_sentence()
                all_segments.append({
                    "start": round(float(payload.get("begin_time", 0)) / 1000, 2),
                    "end": round(float(payload.get("end_time", 0)) / 1000, 2),
                    "text": (payload.get("text", "") or "").strip(),
                })

    # 需要先转 wav
    wav_path = audio_path.with_suffix(".wav")
    try:
        run([cfg.ffmpeg, "-y", "-loglevel", "error",
             "-i", str(audio_path), "-ar", "16000", "-ac", "1",
             "-f", "wav", str(wav_path)], tag="paraformer/wav", check=True, timeout=120)
    except Exception as e:
        raise RuntimeError(f"音频转 WAV 失败：{e}")

    recognition = Recognition(
        model=model,
        format="wav",
        sample_rate=16000,
        language_hints=["zh", "en"],
        callback=Callback(),
    )
    recognition.start()

    # 分块读取 WAV 文件推送
    chunk_size = 3200  # 200ms of 16kHz 16-bit mono
    with open(wav_path, "rb") as f:
        # 跳过 WAV 头
        f.read(44)
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            recognition.send_audio_frame(data)
    
    recognition.stop()

    # 清理临时 wav
    try:
        wav_path.unlink()
    except Exception:
        pass

    return {
        "language": lang,
        "model": f"paraformer/{model}",
        "device": "cloud",
        "segments": all_segments,
    }


# ============================================================
#  openai-whisper（原方案，需 torch）
# ============================================================
def _transcribe_openai(cfg: Config, audio_path: Path, audio_dur: float | None) -> dict[str, Any]:
    try:
        import whisper
        import torch
    except Exception as e:
        raise RuntimeError(
            "未安装 openai-whisper / torch。建议改用 faster-whisper（轻量）：\n"
            "  设置 VNOTES_TRANSCRIBE_BACKEND=faster-whisper\n"
            "  pip install faster-whisper\n"
            f"（原始错误：{e}）"
        ) from e

    device = cfg.whisper_device
    if device == "cuda" and not torch.cuda.is_available():
        log.warn("transcribe", "CUDA 不可用，回退到 CPU")
        device = "cpu"

    log.info("transcribe", f"openai-whisper 加载模型 {cfg.whisper_model_zh}（device={device}）")
    detect_model = whisper.load_model(cfg.whisper_model_zh, device=device)
    audio = whisper.pad_or_trim(whisper.load_audio(str(audio_path)))
    mel = whisper.log_mel_spectrogram(audio).to(device)
    _, probs = detect_model.detect_language(mel)
    lang = max(probs, key=probs.get)
    log.info("transcribe", f"检测语言 = {lang}")

    if lang.startswith("zh"):
        model_name, force_lang = cfg.whisper_model_zh, "zh"
        model = detect_model
    elif lang == "en":
        model_name, force_lang = cfg.whisper_model_en, None
        model = whisper.load_model(model_name, device=device)
    else:
        model_name, force_lang = cfg.whisper_model_zh, None
        model = detect_model

    result = model.transcribe(str(audio_path), language=force_lang, verbose=False,
                              fp16=(device != "cpu"))
    segments = [
        {"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
         "text": (s["text"] or "").strip()}
        for s in result.get("segments", [])
    ]
    return {"language": lang, "model": f"openai-whisper/{model_name}", "device": device, "segments": segments}


# ============================================================
#  主入口
# ============================================================
def transcribe(cfg: Config, audio_info: dict, work_dir: Path) -> dict[str, Any]:
    """转写音频。自动选择后端，返回标准结构。"""
    audio_path = Path(audio_info.get("abs_path") or (work_dir / audio_info["path"]))
    audio_dur = audio_info.get("duration_sec")
    backend = cfg.transcribe_backend

    log.info("transcribe", f"后端 = {backend}, 音频 = {audio_path.name}")

    if backend == "groq":
        if not cfg.groq_api_key:
            raise RuntimeError("Groq 后端需要 VNOTES_GROQ_API_KEY")
        result = _transcribe_groq(cfg, audio_path, audio_dur)
    elif backend == "paraformer":
        result = _transcribe_paraformer(cfg, audio_path, audio_dur)
    elif backend == "vosk":
        result = _transcribe_vosk(cfg, audio_path, audio_dur)
    elif backend == "openai-whisper":
        result = _transcribe_openai(cfg, audio_path, audio_dur)
    else:  # faster-whisper（默认）
        skip_reason = _faster_preflight_skip_reason(cfg)
        if skip_reason:
            log.warn("transcribe", f"{skip_reason}，直接使用 Vosk 离线兜底")
            result = _transcribe_vosk(cfg, audio_path, audio_dur)
        else:
            try:
                result = _transcribe_faster(cfg, audio_path, audio_dur)
            except Exception as e:
                if _looks_like_memory_error(e):
                    log.warn("transcribe", f"faster-whisper 内存不足，自动改用 Vosk 离线兜底：{e}")
                    result = _transcribe_vosk(cfg, audio_path, audio_dur)
                else:
                    raise

    segments = result["segments"]
    text = " ".join(s["text"] for s in segments).strip()
    log.info("transcribe", f"转写完成：{len(segments)} 段，{len(text)} 字")

    # 末段结束时间 vs 音频时长校验
    tail_check = {"ok": True, "gap": None, "audio_dur": audio_dur}
    if segments and audio_dur:
        last_end = segments[-1]["end"]
        gap = audio_dur - last_end
        tail_check["gap"] = round(gap, 2)
        tol = max(cfg.segment_tail_tolerance * audio_dur, 6.0)
        if abs(gap) > tol:
            tail_check["ok"] = False
            log.warn("transcribe", f"末段结束({last_end:.1f}s) 与音频时长({audio_dur:.1f}s) 偏差 {gap:.1f}s")
            if audio_dur > 120 and last_end < audio_dur * 0.5:
                raise RuntimeError(
                    f"转写明显不完整：末段只到 {last_end:.1f}s，但音频时长 {audio_dur:.1f}s。"
                    "已停止生成，避免输出残缺笔记。"
                )

    out = {
        "language": result["language"],
        "model": result["model"],
        "device": result["device"],
        "backend": backend,
        "segments": segments,
        "text": text,
        "tail_check": tail_check,
    }
    write_json(work_dir / "transcript.json", out)
    return out
