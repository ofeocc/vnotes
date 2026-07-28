"""QA 检查：读取整页 PNG，计算逐行墨量，检测空白带/异常渲染/截断。

逐行墨量(ink) = 该行非白像素占比(0~1)，供 QA 与切片共用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .config import Config
from .util import log


def compute_ink_profile(png: Path, col_stride: int = 4) -> tuple[int, int, list[float]]:
    """返回 (width, height, rows)，rows[i] 为第 i 行的墨量(0~1)。"""
    with Image.open(png) as im:
        im = im.convert("L")
        w, h = im.size
        data = im.tobytes()
    rows: list[float] = []
    inv_total = 255.0 * (w // col_stride or 1)
    for y in range(h):
        base = y * w
        s = 0
        for x in range(0, w, col_stride):
            s += 255 - data[base + x]
        rows.append(s / inv_total)
    return w, h, rows


def find_blank_bands(rows: list[float], threshold: float = 0.012,
                     min_len: int = 60) -> list[tuple[int, int]]:
    """找出连续低墨量(近似空白)的水平带。"""
    bands, start = [], None
    for i, v in enumerate(rows):
        if v < threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                bands.append((start, i))
            start = None
    if start is not None and len(rows) - start >= min_len:
        bands.append((start, len(rows)))
    return bands


def qa(cfg: Config, png_info: dict, analysis: dict) -> dict[str, Any]:
    png = Path(png_info["path"])
    w, h, rows = compute_ink_profile(png)
    blanks = find_blank_bands(rows)
    warnings: list[str] = []

    if png_info.get("truncated"):
        warnings.append(f"整页疑似被截断：图高 {h}px < 测得 {png_info.get('measured')}px")

    # 空白带：忽略顶部/底部纯白边距，只报中间的大块空白
    inner = [b for b in blanks if b[0] > 80 and b[1] < h - 80]
    for s, e in inner:
        if e - s > 160:
            warnings.append(f"发现大块空白带 y={s}~{e}（{e-s}px），疑似渲染异常或空章节")

    # 整体墨量过低
    mean_ink = sum(rows) / max(len(rows), 1)
    if mean_ink < 0.02:
        warnings.append(f"整页墨量过低({mean_ink:.3f})，可能渲染失败")

    ok = not warnings
    log.info("qa", f"墨量均值={mean_ink:.3f} 空白带={len(inner)} 校验{'通过' if ok else '有问题'}")
    for wmsg in warnings:
        log.warn("qa", wmsg)

    return {
        "ok": ok,
        "warnings": warnings,
        "width": w,
        "height": h,
        "mean_ink": round(mean_ink, 4),
        "blank_bands": blanks,
        "profile_len": len(rows),
    }


def note_quality(meta: dict, audio: dict, analysis: dict,
                 transcript: dict | None = None,
                 screenshot_qa: dict | None = None) -> dict[str, Any]:
    """Return a publish/readiness verdict for one generated note."""
    chapters = analysis.get("chapters") or []
    duration = (
        audio.get("duration_sec")
        or audio.get("probe_duration")
        or meta.get("duration")
        or 0
    )
    meta_duration = meta.get("duration") or 0
    last_end = max([float(ch.get("t_end") or 0) for ch in chapters] or [0.0])
    coverage = (last_end / float(duration)) if duration else 0.0

    warnings: list[str] = []
    severe = False

    if not chapters:
        severe = True
        warnings.append("no_chapters")
    if audio.get("duration_anomaly"):
        severe = True
        warnings.append("audio_duration_anomaly")
    if meta_duration and duration and float(duration) < float(meta_duration) * 0.95:
        severe = True
        warnings.append("audio_shorter_than_metadata")
    if duration and coverage < 0.9:
        severe = True
        warnings.append("chapter_coverage_low")

    for idx, ch in enumerate(chapters, 1):
        start = ch.get("t_start")
        end = ch.get("t_end")
        if start is None or end is None:
            severe = True
            warnings.append(f"chapter_{idx}_missing_time")
            continue
        if float(end) <= float(start):
            severe = True
            warnings.append(f"chapter_{idx}_invalid_time")
        if not (ch.get("key_points") or ch.get("steps") or ch.get("questions")):
            warnings.append(f"chapter_{idx}_thin_content")

    tail = (transcript or {}).get("tail_check") or {}
    if tail and not tail.get("ok", True):
        severe = True
        warnings.append("transcript_tail_check_failed")

    analysis_warnings = analysis.get("warnings") or []
    if analysis_warnings:
        warnings.extend(f"analysis:{w}" for w in analysis_warnings)
        if any("漏章" in str(w) or "远小于" in str(w) for w in analysis_warnings):
            severe = True

    if screenshot_qa and not screenshot_qa.get("ok", True):
        warnings.extend(f"screenshot:{w}" for w in (screenshot_qa.get("warnings") or []))

    status = "bad" if severe else ("check" if warnings else "ok")
    return {
        "ok": status == "ok",
        "status": status,
        "coverage": round(coverage, 4),
        "last_end": round(last_end, 2),
        "duration": round(float(duration or 0), 2),
        "meta_duration": round(float(meta_duration or 0), 2),
        "warnings": warnings,
    }
