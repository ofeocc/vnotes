#!/usr/bin/env python
"""从缓存的元数据和音频开始，跳过抓取步骤，直接运行转写→分析→渲染→截图→切片。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# 设置环境
os.environ["PYTHONPATH"] = "D:\\python_libs"
os.environ["TEMP"] = "D:\\temp"
os.environ["TMP"] = "D:\\temp"
os.environ["HF_HOME"] = "D:\\vnotes_models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
# CUDA 库路径
cublas_bin = "D:\\python_libs\\nvidia\\cublas\\bin"
nvrtc_bin = "D:\\python_libs\\nvidia\\nvrtc\\bin"
os.environ["PATH"] = cublas_bin + ";" + nvrtc_bin + ";" + os.environ.get("PATH", "")

sys.path.insert(0, "D:\\python_libs")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vnotes import Config, __version__
from vnotes.util import log, safe_name, write_json, fmt_ts
from vnotes import transcribe as T
from vnotes import analyze as Z
from vnotes import svg as S
from vnotes import frames as F
from vnotes import render as R
from vnotes import screenshot as SH
from vnotes import qa as Q
from vnotes import crop as C
import shutil


def main():
    cfg = Config.load()
    cfg.ensure_dirs()

    work = cfg.cache_dir / "run_7b6f7e1c59"
    log.info("main", f"vnotes {__version__} · 工作目录={work}（使用缓存）")

    # 加载缓存的元数据
    meta_path = work / "meta.json"
    if not meta_path.exists():
        log.error("main", f"缓存元数据不存在: {meta_path}")
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    log.info("main", f"标题='{meta.get('title')}' 时长={meta.get('duration')}s")

    # 音频信息
    audio_path = work / "audio.m4a"
    if not audio_path.exists():
        log.error("main", f"缓存音频不存在: {audio_path}")
        return 1
    audio = {
        "path": "audio.m4a",
        "abs_path": str(audio_path),
        "duration_sec": meta.get("duration"),
        "duration_anomaly": False,
    }
    log.info("main", f"音频={audio_path.name} ({audio_path.stat().st_size/1024/1024:.1f}MB)")

    # 3 转写
    log.info("main", "─── 开始转写 ───")
    t0 = time.time()
    transcript = T.transcribe(cfg, audio, work)
    log.info("main", f"转写耗时 {time.time()-t0:.1f}s · {len(transcript.get('segments',[]))} 段")

    # 4 内容分析
    log.info("main", "─── 内容分析 ───")
    analysis = Z.analyze(cfg, meta, transcript, work)

    # 5 SVG
    log.info("main", "─── SVG 生成 ───")
    svgs = S.generate_svgs(cfg, analysis, work)

    # 6 输出目录 & 帧
    out_dir = cfg.output_dir / safe_name(meta.get("title", "notes"), 50)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    # 封面
    if meta.get("cover") and (work / "cover.jpg").exists():
        shutil.copy2(work / "cover.jpg", out_dir / "cover.jpg")

    frames_info = {"enabled": False}
    if analysis.get("is_ui_demo"):
        frames_info = F.extract_frames(cfg, meta, analysis, work, frames_dir)
    else:
        log.info("main", "跳过抽帧（非界面演示类）")

    # 7 渲染 HTML
    log.info("main", "─── 渲染 HTML ───")
    html = out_dir / "notes.html"
    R.render_html(cfg, meta, transcript, analysis, svgs, audio, html, frames_dir)
    log.info("main", f"HTML 已生成：{html}")

    # 8 整页截图
    log.info("main", "─── 整页截图 ───")
    shot = SH.screenshot(cfg, html, out_dir / "full.png")

    # 9 QA
    log.info("main", "─── QA 检查 ───")
    q = Q.qa(cfg, shot, analysis)

    # 10 切片
    log.info("main", "─── 切片 ───")
    slices: list[str] = []
    sl = C.slice_image(cfg, shot, q, out_dir / "slices")
    slices = [str(p.relative_to(out_dir)) for p in sl]

    # 清理空帧目录
    if frames_dir.exists() and not any(frames_dir.iterdir()):
        frames_dir.rmdir()

    write_json(out_dir / "notes_data.json", {
        "meta": meta, "audio": audio, "analysis": analysis,
        "screenshot": shot, "qa": q, "slices": slices, "frames": frames_info,
    })

    log.info("main", "════ 完成 ════")
    log.info("main", f"标题：{meta.get('title')}")
    log.info("main", f"章节：{len(analysis.get('chapters', []))}  帧截图：{frames_info.get('count', 0)}")
    log.info("main", f"HTML：{html}")
    log.info("main", f"整页截图：{shot['width']}x{shot['height']}  切片：{len(slices)}")
    log.info("main", f"QA：{'通过' if q['ok'] else '有问题'}（{len(q['warnings'])} 条提示）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
