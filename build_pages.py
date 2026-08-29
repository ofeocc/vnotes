#!/usr/bin/env python
"""把 vnotes 生成的笔记静态导出为可发布到 GitHub Pages 的站点（docs/）。

用法:
    python build_pages.py

产物:
    docs/index.html          — 画廊：列出所有笔记（封面+标题+章节+原视频链接）
    docs/<slug>/notes.html   — 每篇笔记（已注入灯箱，离线可看）
    docs/<slug>/full.png     — 整页长图
    docs/<slug>/cover.jpg    — 封面
    docs/<slug>/frames/      — 关键帧（若有）
    docs/<slug>/slices/      — 竖向切片（若有）

注意: notes.html 内部用相对路径引用帧/封面，目录结构保持不变即可，无需重写链接。
"""
from __future__ import annotations

import json
import re
import shutil
import html
from pathlib import Path

from vnotes.config import Config
from vnotes.lightbox import inject_note_lightbox
from vnotes.util import safe_name, fmt_ts


def slug(name: str) -> str:
    """URL 安全目录名：保留中文，替换空格/危险字符。"""
    s = re.sub(r'[\s/\\?%*:|"<>\']+', "_", name).strip("._")
    return (s or "note")[:60]


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def main() -> None:
    cfg = Config.load()
    out = cfg.output_dir
    if not out.exists():
        print("没有 output/，请先生成笔记。")
        return
    docs = Path(__file__).resolve().parent / "docs"
    if docs.exists():
        shutil.rmtree(docs)
    docs.mkdir(parents=True)

    items = []
    for d in sorted(out.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        nh = d / "notes.html"
        if not nh.exists():
            continue
        s = slug(d.name)
        target = docs / s
        target.mkdir(parents=True, exist_ok=True)

        # notes.html（注入灯箱脚本，静态页也能用）
        try:
            target_nh = target / "notes.html"
            target_nh.write_text(
                inject_note_lightbox(nh.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        except Exception:
            continue

        for f in ("full.png", "cover.jpg"):
            if (d / f).exists():
                shutil.copy2(d / f, target / f)
        for sub in ("frames", "slices"):
            if (d / sub).is_dir():
                shutil.copytree(d / sub, target / sub)

        meta, analysis, content_qa = {}, {}, {}
        if (d / "notes_data.json").exists():
            try:
                data = json.loads((d / "notes_data.json").read_text(encoding="utf-8"))
                meta = data.get("meta", {})
                analysis = data.get("analysis", {})
                content_qa = data.get("content_qa", {})
            except Exception:
                pass

        items.append({
            "slug": s,
            "name": d.name,
            "title": meta.get("title") or d.name,
            "uploader": meta.get("uploader") or "",
            "duration": meta.get("duration"),
            "chapters": len(analysis.get("chapters", []) or []),
            "is_ui_demo": bool(analysis.get("is_ui_demo")),
            "cover": (d / "cover.jpg").exists(),
            "url": meta.get("webpage_url") or meta.get("base_url") or "",
            "quality": content_qa.get("status", ""),
        })

    # ---- index.html 画廊 ----
    cards = []
    for it in items:
        href = f"./{it['slug']}/notes.html"
        cover = (
            f'<img class="cv" src="./{esc(it["slug"])}/cover.jpg" alt="" loading="lazy"/>'
            if it["cover"] else '<div class="cv ph">◐</div>'
        )
        meta = []
        if it["uploader"]:
            meta.append(esc(it["uploader"]))
        if it["chapters"]:
            meta.append(f"{it['chapters']} 章")
        if it["duration"]:
            meta.append(fmt_ts(float(it["duration"])))
        src = (
            f'<a class="src" href="{esc(it["url"])}" target="_blank" rel="noopener noreferrer">原视频 ↗</a>'
            if it["url"] else ""
        )
        meta_txt = " · ".join(meta) if meta else "已生成"
        cards.append(
            '<article class="card"><a class="cardlink" href="' + href + '">' + cover + "</a>"
            '<div class="info"><a class="t" href="' + href + '">' + esc(it["title"]) + "</a>"
            '<div class="m">' + meta_txt + "</div>" + src + "</div></article>"
        )

    index = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>vnotes · 视频笔记</title>
<style>
:root{{--bg:#faf7f2;--surface:#fff;--border:#e8e2d8;--text:#1a1a1a;--text2:#6a6a6a;--accent:#e0a800;--gold:#F5C518}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Georgia,"Noto Serif SC","Songti SC",serif;background:var(--bg);color:var(--text)}}
.wrap{{max-width:1180px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:40px;font-weight:600;margin:0 0 8px;letter-spacing:.01em}}
.sub{{color:var(--text2);font-size:14px;margin-bottom:36px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:20px}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:0 8px 26px rgba(20,20,20,.06);transition:transform .25s,box-shadow .25s}}
.card:hover{{transform:translateY(-4px);box-shadow:0 16px 40px rgba(20,20,20,.1)}}
.cardlink{{display:block}}.cv{{width:100%;aspect-ratio:16/10;object-fit:cover;display:block;background:#eee}}
.cv.ph{{display:flex;align-items:center;justify-content:center;font-size:34px;color:var(--accent)}}.info{{padding:14px 15px 16px}}
.t{{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;color:var(--text);text-decoration:none;font-size:16px;font-weight:600;line-height:1.45}}
.m{{margin:8px 0 10px;color:var(--text2);font-size:12px}}
.src{{color:var(--accent);text-decoration:none;font-size:12px}}.src:hover{{text-decoration:underline}}
@media(max-width:640px){{h1{{font-size:28px}}.grid{{grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}}}}
</style></head><body><div class="wrap">
<h1>vnotes · 视频笔记</h1>
<p class="sub">共 {len(items)} 篇笔记 · 由 vnotes 生成 · 点击卡片阅读整篇笔记（可离线打开、含时间戳跳转）</p>
<div class="grid">{''.join(cards)}</div>
</div></body></html>'''

    (docs / "index.html").write_text(index, encoding="utf-8")
    print(f"done: {len(items)} 篇笔记 -> {docs}")


if __name__ == "__main__":
    main()
