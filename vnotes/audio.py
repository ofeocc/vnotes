"""音频下载与校验：下载最佳音频，ffprobe 校验时长，差异>5%重下或标记。

自适应 ffmpeg 能力：
  - 优先转 mp3（需完整 ffmpeg）；失败则保留原始格式（m4a/webm 等）
  - ffprobe 探测时长失败时回退到元数据时长
  - Whisper 可直接读取 m4a/webm（需完整 ffmpeg）；精简 ffmpeg 下需手动安装
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import Config
from .util import run, log, run_ytdlp, fmt_ts
from .metadata import _split_part, _build_part_url


def _probe_duration(cfg: Config, audio: Path) -> float | None:
    """ffprobe 读取音频时长（秒）。失败返回 None。"""
    try:
        cp = run([cfg.ffprobe, "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=nw=1:nk=1", str(audio)], tag="audio/probe",
                 check=True, timeout=30)
        val = cp.stdout.strip()
        return float(val) if val else None
    except Exception as e:
        log.warn("audio/probe", f"ffprobe 失败（精简 ffmpeg 可能不支持）：{e}")
        return None


def _clear_audio_products(work_dir: Path) -> None:
    """Remove stale audio and downstream caches before a fresh download."""
    for cand in work_dir.glob("audio.*"):
        try:
            cand.unlink()
        except OSError:
            pass
    for name in ("transcript.json", "analysis.json"):
        try:
            (work_dir / name).unlink()
        except FileNotFoundError:
            pass
    for name in ("vosk_chunks", "transcribe_chunks", "checkpoints"):
        shutil.rmtree(work_dir / name, ignore_errors=True)


def _duration_diff(meta_dur: float | int | None, probe: float | int | None) -> float | None:
    if not meta_dur or not probe:
        return None
    return abs(float(probe) - float(meta_dur)) / float(meta_dur)


def _raise_if_audio_too_short(cfg: Config, meta: dict, probe: float | None) -> None:
    meta_dur = meta.get("duration")
    if not meta_dur or not probe:
        return
    if probe >= float(meta_dur) * (1 - cfg.duration_tolerance):
        return
    title = meta.get("title") or "当前视频"
    raise RuntimeError(
        "音频下载不完整，已停止生成，避免输出残缺笔记。\n"
        f"视频：{title}\n"
        f"元数据时长：{fmt_ts(float(meta_dur))}，实际音频：{fmt_ts(float(probe))}。\n"
        "这通常是 B 站未登录/权限限制/CDN 只返回试看片段导致的；本次日志里浏览器 Cookie 读取失败，"
        "请导出 cookies.txt 后配置 VNOTES_COOKIES_FILE，再重新生成。"
    )


def _download_mp3(cfg: Config, url: str, out: Path) -> bool:
    """尝试下载并转 mp3。成功返回 True。"""
    ffmpeg_dir = str(Path(cfg.ffmpeg).parent) if cfg.ffmpeg else None
    cmd = ["-x", "--audio-format", "mp3", "--audio-quality", "0",
           "-f", "ba/b",
           "--no-playlist", "--no-warnings", "--no-progress",
           "-o", str(out.with_suffix(".%(ext)s"))]
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]
    cmd += [url]
    try:
        run_ytdlp(cfg, cmd, tag="audio/dl-mp3", check=True, timeout=60 * 60)
        # 找到实际产物
        if out.exists():
            return True
        for cand in out.parent.glob("audio.*"):
            if cand.suffix == ".mp3":
                cand.replace(out)
                return True
        return False
    except Exception as e:
        log.warn("audio", f"mp3 转换失败（精简 ffmpeg 无 mp3 编码器）：{e}")
        return False


def _download_raw(cfg: Config, url: str, out: Path) -> Path:
    """下载原始音频（不转码）。返回实际文件路径。"""
    ffmpeg_dir = str(Path(cfg.ffmpeg).parent) if cfg.ffmpeg else None
    cmd = ["-x", "-f", "ba/b",
           "--no-playlist", "--no-warnings", "--no-progress",
           "-o", str(out.with_suffix(".%(ext)s"))]
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]
    cmd += [url]
    run_ytdlp(cfg, cmd, tag="audio/dl-raw", check=True, timeout=60 * 60)
    # 找到实际产物（可能是 m4a/webm/opus 等）
    if out.exists():
        return out
    cands = sorted(out.parent.glob("audio.*"), key=lambda p: p.stat().st_size, reverse=True)
    if cands:
        return cands[0]
    raise RuntimeError("音频下载后未找到产物文件")


def download_audio(cfg: Config, meta: dict, work_dir: Path) -> dict[str, Any]:
    """下载音频并校验时长。返回 audio 信息 dict。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    mp3 = work_dir / "audio.mp3"
    meta_dur = meta.get("duration")

    if meta.get("is_bili"):
        url = _build_part_url(meta["base_url"], meta["selected_part"])
    else:
        url = meta.get("webpage_url")

    result: dict[str, Any] = {"meta_duration": meta_dur}

    _clear_audio_products(work_dir)

    # 尝试 mp3，失败则用原始格式
    log.info("audio", "下载最佳音频")
    audio_path: Path | None = None
    if _download_mp3(cfg, url, mp3):
        audio_path = mp3
        result["format"] = "mp3"
        log.info("audio", "mp3 转换成功")
    else:
        log.warn("audio", "回退到原始音频格式（不转码）")
        audio_path = _download_raw(cfg, url, mp3)
        result["format"] = audio_path.suffix.lstrip(".")
        log.info("audio", f"原始音频：{audio_path.name}")

    result["path"] = str(audio_path.name)
    result["abs_path"] = str(audio_path)

    # 时长校验
    probe = _probe_duration(cfg, audio_path)
    result["probe_duration"] = probe

    anomaly = False
    if meta_dur and probe:
        diff = _duration_diff(meta_dur, probe) or 0.0
        if diff > cfg.duration_tolerance:
            log.warn("audio", f"时长差异 {diff:.1%} > {cfg.duration_tolerance:.0%}，重试")
            _clear_audio_products(work_dir)
            if result["format"] == "mp3":
                if _download_mp3(cfg, url, mp3):
                    audio_path = mp3
                else:
                    audio_path = _download_raw(cfg, url, mp3)
                    result["format"] = audio_path.suffix.lstrip(".")
            else:
                audio_path = _download_raw(cfg, url, mp3)
                result["format"] = audio_path.suffix.lstrip(".")
            result["path"] = str(audio_path.name)
            result["abs_path"] = str(audio_path)
            probe2 = _probe_duration(cfg, audio_path)
            result["probe_duration"] = probe2
            if meta_dur and probe2:
                diff2 = _duration_diff(meta_dur, probe2) or 0.0
                if diff2 > cfg.duration_tolerance:
                    anomaly = True
                    log.warn("audio", f"重下后仍偏差 {diff2:.1%}，标记异常")
                    _raise_if_audio_too_short(cfg, meta, probe2)
            else:
                anomaly = True
        else:
            log.info("audio", f"时长校验通过：probe={probe:.1f}s meta={meta_dur:.1f}s")
    elif meta_dur:
        # ffprobe 不可用时用元数据时长
        log.warn("audio", "ffprobe 不可用，使用元数据时长")
        probe = meta_dur
    else:
        log.warn("audio", "缺少元数据时长，跳过校验")

    result["duration_anomaly"] = anomaly
    result["duration_sec"] = probe or meta_dur or 0.0
    return result
