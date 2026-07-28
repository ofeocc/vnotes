"""真实帧抽取：界面/演示类视频，按章节时间段挑信息量最大、界面最清晰的帧。

流程：下载视频流 → 每章时间段并行采样候选帧 → 信息密度打分(边缘/清晰度/色彩)
→ 选最佳时间戳 → ffmpeg -ss -q:v 1 高质量导出。文件名带阶段名与秒数。
"""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from .config import Config
from .util import run, log, safe_name, fmt_short, run_ytdlp
from .metadata import _split_part, _build_part_url


def download_video(cfg: Config, meta: dict, work_dir: Path) -> Path | None:
    """下载视频流（用于抽帧），限制 1080p 以下。"""
    video = work_dir / "video.mp4"
    if video.exists() and video.stat().st_size > 1_000_000:
        log.info("frames", "视频已存在，跳过下载")
        return video
    if meta.get("is_bili"):
        url = _build_part_url(meta["base_url"], meta["selected_part"])
    else:
        url = meta.get("webpage_url")
    ffmpeg_dir = str(Path(cfg.ffmpeg).parent) if cfg.ffmpeg else None
    cmd = ["-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
           "--merge-output-format", "mp4",
           "--no-playlist", "--no-warnings", "--no-progress",
           "-o", str(video.with_suffix(".%(ext)s"))]
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]
    cmd += [url]
    try:
        run_ytdlp(cfg, cmd, tag="frames/video", check=True, timeout=60 * 60)
    except Exception as e:
        log.warn("frames", f"视频下载失败，跳过抽帧：{e}")
        return None
    if not video.exists():
        cand = sorted(work_dir.glob("video.*"), key=lambda p: p.stat().st_size, reverse=True)
        if cand:
            cand[0].replace(video)
    return video if video.exists() else None


def _extract_frame(cfg: Config, video: Path, t: float, out: Path, width: int = 0) -> bool:
    """在时间 t 抽一帧。width>0 时缩放（用于候选打分提速）。"""
    cmd = [cfg.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2"]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]
    cmd += [str(out)]
    try:
        run(cmd, tag="frames/extract", check=True, timeout=60)
        return out.exists() and out.stat().st_size > 0
    except Exception:
        return False


def _score(img_path: Path) -> float:
    """信息密度打分：边缘密度 + 清晰度 + 色彩方差；空白帧判负。"""
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        return -1.0
    gs = im.convert("L")
    if statistics.pstdev(gs.getdata()) < 8:  # 近似纯色/空白
        return -1.0
    edges = gs.filter(ImageFilter.FIND_EDGES)
    edge_score = statistics.mean(edges.getdata())  # 0~255
    lap = gs.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 0))
    sharp = statistics.pvariance(lap.getdata())
    sat = im.convert("HSV").split()[1]
    sat_var = statistics.pvariance(sat.getdata())
    return edge_score + sharp / 1000.0 + sat_var / 120.0


def _best_timestamp(cfg: Config, video: Path, t_start: float, t_end: float, tmp: Path) -> float:
    """在 [t_start,t_end] 内采样候选帧并打分，返回最佳时间戳。"""
    dur = max(t_end - t_start, 1.0)
    n = min(8, max(3, int(dur / 12) + 1))  # 候选数
    step = dur / (n + 1)
    cands = [round(t_start + step * (i + 1), 2) for i in range(n)]
    tmp.mkdir(parents=True, exist_ok=True)

    def job(i: int, t: float) -> tuple[int, float, float]:
        p = tmp / f"c{i}.jpg"
        ok = _extract_frame(cfg, video, t, p, width=360)
        return (i, t, _score(p) if ok else -1.0)

    best_t, best_s = cands[0], -1.0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, t, s in ex.map(lambda a: job(*a), [(i, t) for i, t in enumerate(cands)]):
            if s > best_s:
                best_s, best_t = s, t
    log.debug("frames", f"候选 {len(cands)} 帧，最佳 t={best_t}s score={best_s:.1f}")
    return best_t


def extract_frames(cfg: Config, meta: dict, analysis: dict, work_dir: Path,
                   frames_dir: Path) -> dict[str, Any]:
    """为每章抽取一帧（仅 is_ui_demo）。在 chapters 上写入 frame 字段。"""
    if not analysis.get("is_ui_demo"):
        log.info("frames", "非界面演示类视频，跳过抽帧")
        return {"enabled": False}
    video = download_video(cfg, meta, work_dir)
    if not video:
        log.warn("frames", "无视频文件，跳过抽帧")
        return {"enabled": False, "reason": "no_video"}

    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp = work_dir / "frame_candidates"
    n_ok = 0
    for ch in analysis.get("chapters", []):
        ts, te = ch.get("t_start", 0), ch.get("t_end", 0)
        if te <= ts:
            continue
        try:
            t = _best_timestamp(cfg, video, ts, te, tmp)
            name = f"{ch['id']:02d}_{safe_name(ch.get('title',''), 20)}_{fmt_short(t)}.jpg"
            out = frames_dir / name
            if _extract_frame(cfg, video, t, out, width=0):  # 全质量 -q:v 2
                ch["frame"] = {"path": f"frames/{name}", "t": round(t, 2)}
                n_ok += 1
                log.info("frames", f"第{ch['id']}章 抽帧 @ {t:.1f}s → {name}")
            else:
                log.warn("frames", f"第{ch['id']}章 抽帧失败")
        except Exception as e:
            log.warn("frames", f"第{ch['id']}章 抽帧异常：{e}")
    # 清理候选
    for f in tmp.glob("*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass
    log.info("frames", f"完成抽帧：{n_ok}/{len(analysis.get('chapters', []))} 章")
    return {"enabled": True, "count": n_ok}


def extract_frames_detailed(cfg: Config, meta: dict, analysis: dict, work_dir: Path,
                            frames_dir: Path) -> dict[str, Any]:
    """细致模式：按 frame_moments 多帧抽取，适用于所有视频类型。

    策略：
    1. 下载视频（同 extract_frames）
    2. 每章遍历 frame_moments，在指定时间戳附近抽帧
    3. 若无 frame_moments，回退到 key_quotes 时间戳
    4. 每帧用 _best_timestamp 微调（±3s 范围内选信息量最大的帧）
    5. 在 chapters 上写入 frames 列表（多帧）
    """
    video = download_video(cfg, meta, work_dir)
    if not video:
        log.warn("frames", "无视频文件，跳过细致抽帧")
        return {"enabled": False, "reason": "no_video"}

    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp = work_dir / "frame_candidates"
    n_ok = 0
    n_total = 0

    for ch in analysis.get("chapters", []):
        ch.setdefault("frames", [])
        ts, te = ch.get("t_start", 0), ch.get("t_end", 0)

        # 收集目标时间戳：优先 frame_moments，回退 key_quotes
        moments = list(ch.get("frame_moments", []))
        if not moments:
            for q in ch.get("key_quotes", []):
                moments.append({"t": q.get("t", 0), "desc": q.get("text", "")[:40]})

        # 限制每章最多 4 帧（避免过多）
        moments = moments[:4]

        for mi, mom in enumerate(moments):
            n_total += 1
            t = mom.get("t", 0)
            desc = mom.get("desc", "")
            if t <= 0 or (te > 0 and t > te + 5):
                continue
            # 在 t±3s 范围内选最佳帧
            t_lo = max(t - 3, ts if ts > 0 else 0)
            t_hi = min(t + 3, te if te > 0 else t + 6)
            try:
                best_t = _best_timestamp(cfg, video, t_lo, t_hi, tmp)
                name = f"{ch['id']:02d}_{mi:02d}_{safe_name(ch.get('title',''), 16)}_{fmt_short(best_t)}.jpg"
                out = frames_dir / name
                if _extract_frame(cfg, video, best_t, out, width=0):
                    ch["frames"].append({
                        "path": f"frames/{name}",
                        "t": round(best_t, 2),
                        "desc": desc,
                    })
                    n_ok += 1
                    log.info("frames", f"第{ch['id']}章 帧{mi+1} @ {best_t:.1f}s → {name}")
                else:
                    log.warn("frames", f"第{ch['id']}章 帧{mi+1} 抽取失败")
            except Exception as e:
                log.warn("frames", f"第{ch['id']}章 帧{mi+1} 异常：{e}")

    # 清理候选
    for f in tmp.glob("*.jpg") if tmp.exists() else []:
        try:
            f.unlink()
        except Exception:
            pass

    log.info("frames", f"细致抽帧完成：{n_ok}/{n_total} 帧（{len(analysis.get('chapters', []))} 章）")
    return {"enabled": True, "count": n_ok, "detailed": True}
