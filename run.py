#!/usr/bin/env python
"""vnotes CLI：视频链接 → 离线单页笔记 HTML。

用法：
  python run.py <视频链接> [--part N] [--no-frames] [--no-slice] [--stub-transcript]
  python run.py --self-test            # 离线验证 渲染→截图→QA→切片（无需网络/Whisper/LLM）
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

from vnotes.bootstrap import ensure_project_venv

ensure_project_venv(script=__file__)

from vnotes import Config, __version__
from vnotes.util import log, safe_name, write_json, fmt_ts
from vnotes import metadata as M
from vnotes import audio as A
from vnotes import transcribe as T
from vnotes import analyze as Z
from vnotes import svg as S
from vnotes import frames as F
from vnotes import render as R
from vnotes import screenshot as SH
from vnotes import qa as Q
from vnotes import crop as C


def _work_dir(cfg: Config, url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    return cfg.cache_dir / f"run_{h}"


def _out_dir(cfg: Config, meta: dict) -> Path:
    name = safe_name(meta.get("title", "notes"), 50) or "notes"
    return cfg.output_dir / name


def pipeline(cfg: Config, url: str, *, part: int | None, no_frames: bool,
             no_slice: bool, stub_transcript: bool,
             out_dir: Path | None = None, cancel_check=None) -> Path:
    cfg.ensure_dirs()
    url = M.normalize_video_url(url)
    work = _work_dir(cfg, url)
    work.mkdir(parents=True, exist_ok=True)
    log.info("main", f"vnotes {__version__} · 工作目录={work}")

    def _check():
        if cancel_check:
            cancel_check()

    # 1 元数据
    _check()
    meta = M.fetch_metadata(cfg, url, part=part, work_dir=work)

    # 2 音频
    _check()
    audio = A.download_audio(cfg, meta, work)

    # 3 转写
    _check()
    if stub_transcript:
        log.warn("main", "使用桩转写（不调用 Whisper）")
        dur = audio.get("duration_sec") or meta.get("duration") or 60.0
        transcript = {
            "language": "zh", "model": "stub", "device": "-",
            "segments": [{"start": 0.0, "end": dur, "text": "（桩转写：用于离线流程验证）"}],
            "text": "（桩转写）", "tail_check": {"ok": True, "gap": 0.0, "audio_dur": dur},
        }
    else:
        transcript = T.transcribe(cfg, audio, work)

    # 4 内容分析
    _check()
    analysis = Z.analyze(cfg, meta, transcript, work)

    # 5 SVG
    _check()
    svgs = S.generate_svgs(cfg, analysis, work)

    # 6 输出目录 & 帧
    if out_dir is None:
        out_dir = _out_dir(cfg, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    # 封面复制到输出目录
    if meta.get("cover") and (work / "cover.jpg").exists():
        shutil.copy2(work / "cover.jpg", out_dir / "cover.jpg")

    frames_info = {"enabled": False}
    is_detailed = getattr(cfg, "note_mode", "essence") == "detailed"
    if is_detailed and not no_frames:
        # 细致模式：所有视频类型都抽帧，按 frame_moments 多帧抽取
        frames_info = F.extract_frames_detailed(cfg, meta, analysis, work, frames_dir)
    elif analysis.get("is_ui_demo") and not no_frames:
        frames_info = F.extract_frames(cfg, meta, analysis, work, frames_dir)
    else:
        log.info("main", "跳过抽帧" + ("（--no-frames）" if no_frames else "（非界面演示类）"))

    # 7 渲染 HTML
    _check()
    html = out_dir / "notes.html"
    R.render_html(cfg, meta, transcript, analysis, svgs, audio, html, frames_dir)
    log.info("main", f"HTML 已生成：{html}")

    # 8 整页截图
    _check()
    shot = SH.screenshot(cfg, html, out_dir / "full.png")

    # 9 QA
    _check()
    q = Q.qa(cfg, shot, analysis)
    content_q = Q.note_quality(meta, audio, analysis, transcript, q)
    for wmsg in content_q.get("warnings", []):
        log.warn("quality", str(wmsg))

    # 10 切片
    slices: list[str] = []
    if not no_slice:
        sl = C.slice_image(cfg, shot, q, out_dir / "slices")
        slices = [str(p.relative_to(out_dir)) for p in sl]

    # 清理无帧目录
    if frames_dir.exists() and not any(frames_dir.iterdir()):
        frames_dir.rmdir()

    write_json(out_dir / "notes_data.json", {
        "meta": meta, "audio": audio, "analysis": analysis,
        "screenshot": shot, "qa": q, "content_qa": content_q,
        "slices": slices, "frames": frames_info,
    })
    _summary(meta, audio, transcript, analysis, html, shot, q, slices, frames_info)
    return html


def _summary(meta, audio, transcript, analysis, html, shot, q, slices, frames_info):
    log.info("main", "──── 完成 ────")
    log.info("main", f"标题：{meta.get('title')}")
    log.info("main", f"章节：{len(analysis.get('chapters', []))}  帧截图：{frames_info.get('count', 0)}")
    log.info("main", f"HTML：{html}")
    log.info("main", f"整页截图：{shot['width']}x{shot['height']}  切片：{len(slices)}")
    log.info("main", f"QA：{'通过' if q['ok'] else '有问题'}（{len(q['warnings'])} 条提示）")


# ---------------- 离线 self-test ----------------
def self_test(cfg: Config) -> Path:
    """离线验证 渲染→截图→QA→切片，不依赖网络/Whisper/LLM。"""
    from PIL import Image, ImageDraw
    cfg.ensure_dirs()
    out_dir = cfg.output_dir / "_self_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    log.info("selftest", "生成合成数据（2 章 + 内联 SVG + 占位截图）…")

    # 占位“视频截图”
    for i, (name, hue) in enumerate([("env", (220, 230, 245)), ("editor", (245, 230, 220))], 1):
        im = Image.new("RGB", (1280, 720), (250, 250, 252))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 1280, 56], fill=hue)
        d.rectangle([40, 90, 820, 140], fill=(220, 224, 230))
        for y in range(170, 660, 44):
            d.rectangle([40, y, 900 + (y % 200), y + 20], fill=(235, 236, 240))
        d.rectangle([940, 90, 1240, 660], fill=hue)
        im.save(frames_dir / f"{i:02d}_{name}_000030.jpg", quality=92)

    meta = {
        "title": "vnotes 自检示例：从想法到上线", "uploader": "vnotes",
        "description": "这是一条用于离线自检的合成视频元数据。", "tags": ["自检", "工具", "笔记"],
        "cover": None, "duration": 320.0, "webpage_url": "https://example.com/v/1",
        "base_url": "https://example.com/v/1", "is_bili": False, "selected_part": 1,
        "parts": [{"index": 1, "title": "全片", "duration": 320, "url": ""}],
        "extractor": "SelfTest", "yt_chapters": [],
    }
    transcript = {"language": "zh", "tail_check": {"ok": True, "gap": 0.0}}
    analysis = {
        "is_ui_demo": True, "video_summary": "演示如何把一个想法快速做成可上线的产品。",
        "warnings": [],
        "chapters": [
            {"id": 1, "title": "环境准备", "t_start": 0, "t_end": 120, "svg_type": "flow",
             "svg_rationale": "展示从安装到启动的步骤链",
             "questions": ["需要装哪些依赖？"], "traps": ["别忘配环境变量"],
             "steps": ["安装运行时", "配置环境变量", "启动服务"], "conclusion": "三步即可就绪",
             "key_points": ["依赖最小化", "配置即代码"], "key_quotes": [{"t": 45, "text": "环境即代码"}],
             "visual_anchors": ["终端窗口"], "frame": {"path": "frames/01_env_000030.jpg", "t": 30}},
            {"id": 2, "title": "编辑器核心概念", "t_start": 120, "t_end": 320, "svg_type": "concept",
             "svg_rationale": "表达画布/图层/工具的分层关系",
             "questions": ["画布和图层什么关系？"], "traps": ["图层顺序搞反"],
             "steps": [], "conclusion": "画布是容器，图层是内容",
             "key_points": ["画布分层", "工具作用于图层"], "key_quotes": [{"t": 200, "text": "图层即内容"}],
             "visual_anchors": ["图层面板"], "frame": {"path": "frames/02_editor_000030.jpg", "t": 30}},
        ],
    }
    svgs = [
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 720 300' style='width:100%;height:auto'>"
        "<rect width='720' height='300' fill='#ffffff'/>"
        "<g font-family='sans-serif' font-size='15' fill='#1d1d1f' text-anchor='middle'>"
        "<rect x='40' y='120' width='140' height='56' rx='12' fill='#eef4ff' stroke='#0071e3'/>"
        "<text x='110' y='152'>安装运行时</text>"
        "<path d='M180 148 L240 148' stroke='#0071e3' stroke-width='2' marker-end='url(#a)'/>"
        "<rect x='240' y='120' width='140' height='56' rx='12' fill='#eef4ff' stroke='#0071e3'/>"
        "<text x='310' y='152'>配置环境变量</text>"
        "<path d='M380 148 L440 148' stroke='#0071e3' stroke-width='2' marker-end='url(#a)'/>"
        "<rect x='440' y='120' width='140' height='56' rx='12' fill='#eef4ff' stroke='#0071e3'/>"
        "<text x='510' y='152'>启动服务</text></g>"
        "<defs><marker id='a' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto'>"
        "<path d='M0 0 L8 3 L0 6 Z' fill='#0071e3'/></marker></defs></svg>",
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 720 320' style='width:100%;height:auto'>"
        "<rect width='720' height='320' fill='#ffffff'/>"
        "<g font-family='sans-serif' font-size='15' fill='#1d1d1f' text-anchor='middle'>"
        "<rect x='250' y='30' width='220' height='48' rx='12' fill='#f5f5f7' stroke='#6e6e73'/>"
        "<text x='360' y='60'>画布(容器)</text>"
        "<rect x='120' y='130' width='200' height='48' rx='12' fill='#eef4ff' stroke='#0071e3'/>"
        "<text x='220' y='160'>图层 A</text>"
        "<rect x='400' y='130' width='200' height='48' rx='12' fill='#fff0e6' stroke='#ff8a3d'/>"
        "<text x='500' y='160'>图层 B</text>"
        "<rect x='260' y='230' width='200' height='48' rx='12' fill='#eafff5' stroke='#34c759'/>"
        "<text x='360' y='260'>工具(作用于图层)</text>"
        "<line x1='360' y1='78' x2='220' y2='130' stroke='#d2d2d7'/>"
        "<line x1='360' y1='78' x2='500' y2='130' stroke='#d2d2d7'/>"
        "<line x1='220' y1='178' x2='330' y2='230' stroke='#d2d2d7'/>"
        "<line x1='500' y1='178' x2='390' y2='230' stroke='#d2d2d7'/></g></svg>",
    ]
    audio = {"duration_sec": 320.0, "duration_anomaly": False}

    html = out_dir / "notes.html"
    R.render_html(cfg, meta, transcript, analysis, svgs, audio, html, frames_dir)
    log.info("selftest", f"HTML 已生成：{html}")

    shot = SH.screenshot(cfg, html, out_dir / "full.png")
    q = Q.qa(cfg, shot, analysis)
    sl = C.slice_image(cfg, shot, q, out_dir / "slices")
    log.info("selftest", f"自检完成：截图 {shot['width']}x{shot['height']}，切片 {len(sl)}，QA {'通过' if q['ok'] else '有问题'}")
    return html


def _try_import(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vnotes", description="视频链接 → 离线单页笔记 HTML")
    ap.add_argument("url", nargs="?", help="视频链接（B站/YouTube 等）")
    ap.add_argument("--part", type=int, default=None, help="分 P 号（B站）")
    ap.add_argument("--no-frames", action="store_true", help="跳过真实帧抽取")
    ap.add_argument("--no-slice", action="store_true", help="跳过切片")
    ap.add_argument("--stub-transcript", action="store_true", help="用桩转写（不调 Whisper）")
    ap.add_argument("--self-test", action="store_true", help="离线自检图像管线")
    ap.add_argument("--check", action="store_true", help="仅检查环境与依赖")
    ap.add_argument("--batch", action="store_true", help="批量处理多 P 视频并生成 index.html 聚合页")
    ap.add_argument("--parts", type=str, default=None, help="批量模式下指定 P 号，逗号分隔（如 1,3,5）；默认处理全部")
    ap.add_argument("--mode", type=str, default=None, choices=["essence", "detailed"],
                    help="笔记模式：essence=脉络精华（默认），detailed=细致笔记（含关键帧图）")
    args = ap.parse_args(argv)

    cfg = Config.load()
    if args.mode:
        cfg.note_mode = args.mode
    if args.check:
        miss = cfg.check_tools(need_whisper=not args.stub_transcript and not args.self_test)
        try:
            import playwright
            pw_status = "已安装 (chromium)"
        except Exception:
            pw_status = "未安装"
        print("环境检查：")
        print(f"  yt-dlp     : {cfg.yt_dlp}")
        print(f"  ffmpeg     : {cfg.ffmpeg}")
        print(f"  ffprobe    : {cfg.ffprobe}")
        print(f"  chrome     : {cfg.chrome}")
        print(f"  playwright : {pw_status}")
        print(f"  LLM        : {cfg.llm_base_url} / {cfg.llm_model}" + (" (无Key)" if not cfg.llm_api_key else ""))
        print(f"  转写后端   : {cfg.transcribe_backend}")
        if cfg.transcribe_backend == "groq":
            print(f"    Groq     : model={cfg.groq_model}" + (" (无Key)" if not cfg.groq_api_key else ""))
        elif cfg.transcribe_backend == "faster-whisper":
            fw_status = "已安装" if _try_import("faster_whisper") else "未安装"
            print(f"    faster-whisper: {fw_status}, model_dir={cfg.whisper_model_dir}")
            print(f"    models   : zh={cfg.whisper_model_zh} en={cfg.whisper_model_en} device={cfg.whisper_device}")
        else:
            ow_status = "已安装" if _try_import("whisper") else "未安装"
            print(f"    openai-whisper: {ow_status}, device={cfg.whisper_device}")
        if miss:
            print("  缺失：")
            for m in miss:
                print(f"    - {m}")
        else:
            print("  全部就绪")
        return 0

    if args.self_test:
        return 0 if self_test(cfg) else 1

    if not args.url:
        ap.error("需要提供视频链接，或使用 --self-test / --check")

    missing = cfg.check_tools(need_whisper=not args.stub_transcript)
    if missing:
        log.error("main", "环境不完整：")
        for m in missing:
            log.error("main", "  - " + m)
        log.error("main", "可先运行 --check 查看详情；离线验证图像管线可用 --self-test")
        return 1

    t0 = time.time()
    try:
        if args.batch:
            from vnotes.batch import batch_pipeline
            parts_list = None
            if args.parts:
                parts_list = [int(x.strip()) for x in args.parts.split(",") if x.strip()]
            html = batch_pipeline(
                cfg, args.url,
                parts=parts_list,
                no_frames=args.no_frames,
                no_slice=args.no_slice,
            )
        else:
            html = pipeline(cfg, args.url, part=args.part, no_frames=args.no_frames,
                            no_slice=args.no_slice, stub_transcript=args.stub_transcript)
    except Exception as e:
        log.error("main", f"流程失败：{e}")
        raise
    log.info("main", f"总耗时 {time.time()-t0:.1f}s · 产出 {html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
