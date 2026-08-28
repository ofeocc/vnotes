"""HTML 渲染：单页离线 HTML，内联 CSS + SVG，截图走相对路径，图解/视频时间戳标注。

设计取向：克制、优雅、Apple 风；单一强调色；信息分层清晰，不堆砌荧光色。
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config
from .lightbox import inject_note_lightbox
from .util import fmt_ts


def _esc(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


def _jump_url(meta: dict, sec: float) -> str:
    """构造回跳视频的时间戳链接。"""
    sec = int(round(sec or 0))
    if meta.get("is_bili"):
        return f"{meta.get('base_url','')}?p={meta.get('selected_part',1)}&t={sec}"
    if meta.get("is_youtube"):
        base = meta.get("webpage_url", "")
        # 去掉可能已有的 #t= 后缀
        base = base.split("#")[0]
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}t={sec}"
    return f"{meta.get('webpage_url','')}#t={sec}"


def _chip(text: str, cls: str = "") -> str:
    return f"<span class='chip {cls}'>{_esc(text)}</span>"


def _ts_link(meta: dict, sec: float, label: str | None = None) -> str:
    txt = label or fmt_ts(sec)
    return f"<a class='ts' target='_blank' href='{_esc(_jump_url(meta, sec))}'>⏱ {_esc(txt)}</a>"


def _list_block(items: list, cls: str, marker: str) -> str:
    if not items:
        return ""
    lis = "".join(f"<li><span class='mk'>{marker}</span><span>{_esc(it)}</span></li>" for it in items)
    return f"<ul class='blk {cls}'>{lis}</ul>"


def _chapter_body(ch: dict, meta: dict) -> str:
    parts = []
    # 细致模式：子板块
    if ch.get("detail_sections"):
        ds_html = ""
        for ds in ch["detail_sections"]:
            ds_html += (
                f"<div class='detail-sec'><h5>{_esc(ds.get('title',''))}</h5>"
                f"<p>{_esc(ds.get('content',''))}</p></div>"
            )
        parts.append((
            "",
            "<details class='detail-toggle'>"
            "<summary><span>详细讲解</span><small>展开本章完整拆解</small></summary>"
            f"<div class='detail-block'>{ds_html}</div></details>"
        ))
    if ch.get("questions"):
        parts.append(("<h4>问题</h4>", _list_block(ch["questions"], "q", "?")))
    if ch.get("traps"):
        parts.append(("<h4 class='warn'>陷阱 / 误区</h4>", _list_block(ch["traps"], "trap", "!")))
    if ch.get("steps"):
        steps = "".join(f"<li>{_esc(s)}</li>" for s in ch["steps"])
        parts.append(("<h4>步骤</h4>", f"<ol class='blk steps'>{steps}</ol>"))
    if ch.get("key_points"):
        parts.append(("<h4>要点</h4>", _list_block(ch["key_points"], "kp", "•")))
    if ch.get("conclusion"):
        parts.append(("<h4>结论</h4>", f"<p class='concl'>{_esc(ch['conclusion'])}</p>"))
    if ch.get("key_quotes"):
        qitems = "".join(
            f"<blockquote><p>{_esc(q.get('text',''))}</p>"
            f"<footer>{_ts_link(meta, q.get('t',0))}</footer></blockquote>"
            for q in ch["key_quotes"]
        )
        parts.append(("<h4>关键引用</h4>", f"<div class='quotes'>{qitems}</div>"))
    return "".join(h + b for h, b in parts)


def _chapter(ch: dict, svg: str, meta: dict) -> str:
    cid = f"ch-{ch['id']}"
    title = _esc(ch.get("title", ""))
    rng = f"{fmt_ts(ch.get('t_start',0))} – {fmt_ts(ch.get('t_end',0))}"
    type_label = {"flow": "流程", "concept": "概念", "timeline": "时间线", "comparison": "对比",
                  "risk": "风险决策", "data": "数据", "causation": "因果"}.get(ch.get("svg_type"), "")

    # 正文
    body = _chapter_body(ch, meta)

    # SVG 图解
    svg_card = (
        f"<figure class='media'><figcaption><span class='tag svg'>图解</span>"
        f"<span class='cap'>{type_label} · {_esc(ch.get('svg_rationale',''))}</span></figcaption>"
        f"<div class='svg-wrap'>{svg}</div></figure>"
    )

    # 真实截图（细致模式：多帧 gallery）
    shot = ""
    frames_list = ch.get("frames", [])
    if frames_list:
        items = ""
        for fr in frames_list:
            desc = _esc(fr.get("desc", ""))
            items += (
                f"<figure class='frame-item'>"
                f"<img loading='lazy' src='{_esc(fr['path'])}' alt='{title} 截图'/>"
                f"<figcaption><span class='tag shot'>视频帧</span>"
                f"<span class='cap'>{_ts_link(meta, fr.get('t',0))}"
                f"{' · ' + desc if desc else ''}</span></figcaption></figure>"
            )
        shot = f"<div class='frame-gallery'>{items}</div>"
    elif ch.get("frame"):
        # 精华模式：单帧
        fr = ch["frame"]
        shot = (
            f"<figure class='media'><figcaption><span class='tag shot'>视频时间戳</span>"
            f"<span class='cap'>{_ts_link(meta, fr.get('t',0))}</span></figcaption>"
            f"<img loading='lazy' src='{_esc(fr['path'])}' alt='{title} 截图'/></figure>"
        )

    return f"""
<section class='chapter' id='{cid}'>
  <header class='ch-head'>
    <span class='ch-no'>{ch['id']:02d}</span>
    <div class='ch-titles'>
      <h3>{title}</h3>
      <div class='ch-meta'>{_ts_link(meta, ch.get('t_start',0), rng)}</div>
    </div>
  </header>
  <div class='ch-body'>
    {body}
    {svg_card}
    {shot}
  </div>
</section>"""


def _nav(chapters: list[dict], meta: dict) -> str:
    items = "".join(
        f"<a href='#ch-{c['id']}'><b>{c['id']:02d}</b>"
        f"<span>{_esc((c.get('title',''))[:16])}</span>"
        f"<i>{fmt_ts(c.get('t_start',0))}</i></a>"
        for c in chapters
    )
    watch = _esc(_jump_url(meta, 0))
    return f"<nav class='toc'><div class='toc-in'>{items}</div>" \
           f"<a class='watch' target='_blank' href='{watch}'>原视频 ↗</a></nav>"


def _header(meta: dict, analysis: dict, audio_info: dict) -> str:
    cover = ""
    if meta.get("cover"):
        cover = f"<img class='cover' src='{_esc(meta['cover'])}' alt='封面'/>"
    tags = "".join(_chip(t, "tag-chip") for t in (meta.get("tags") or [])[:12])
    parts = meta.get("parts") or []
    part_line = ""
    if len(parts) > 1:
        base = meta.get("base_url") or meta.get("webpage_url", "")
        pl = "".join(
            f"<a class='p{' cur' if p['index']==meta.get('selected_part') else ''}' "
            f"target='_blank' href='{_esc(base)}?p={p['index']}'>"
            f"P{p['index']} {_esc((p.get('title',''))[:14])}</a>"
            for p in parts[:12]
        )
        part_line = f"<div class='parts'><span class='lbl'>分P</span>{pl}</div>"
    badges = "".join(_chip(b) for b in [
        meta.get("extractor", ""),
        fmt_ts(meta.get("duration", 0)) if meta.get("duration") else "",
        f"当前 P{meta.get('selected_part',1)}" if len(parts) > 1 else "",
    ] if b)
    summary = analysis.get("video_summary", "")
    summary_html = f"<div class='summary'>{_esc(summary)}</div>" if summary else ""
    desc = (meta.get("description") or "").strip()
    desc_html = f"<p class='desc'>{_esc(desc[:600])}{'…' if len(desc)>600 else ''}</p>" if desc else ""
    return f"""
<header class='hd'>
  <div class='hd-cover'>{cover}</div>
  <div class='hd-main'>
    <h1>{_esc(meta.get('title',''))}</h1>
    <div class='sub'>
      <span class='up'>UP · {_esc(meta.get('uploader',''))}</span>
      <span class='badges'>{badges}</span>
    </div>
    {part_line}
    {summary_html}
    {desc_html}
    <div class='tags'>{tags}</div>
  </div>
</header>"""


def _footer(meta: dict, analysis: dict, audio_info: dict, transcript: dict) -> str:
    notes = []
    if audio_info.get("duration_anomaly"):
        notes.append("⚠ 音频时长与元数据偏差超 5%（已重试），可能不完整")
    tc = transcript.get("tail_check", {}) if transcript else {}
    if tc and not tc.get("ok"):
        notes.append(f"⚠ 转写末段与音频时长偏差 {tc.get('gap')}s，可能漏段")
    for w in analysis.get("warnings", []):
        notes.append(f"⚠ {w}")
    note_html = "".join(f"<li>{_esc(n)}</li>" for n in notes)
    warn_html = f"<ul class='ft-warn'>{note_html}</ul>" if note_html else ""
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
<footer class='ft'>
  {warn_html}
  <p>由 vnotes 生成 · {gen} · 模型转写+大模型整理 · 图解为 AI 据章节内容动态生成</p>
</footer>"""


CSS = """
:root{--bg:#fafafa;--card:#fff;--ink:#1a1a1a;--sub:#666;--line:#e0e0e0;--soft:#f5f5f5;
--acc:#F5C518;--acc-dk:#1a1a1a;--acc-soft:#fff8e1;--warn:#ff6b35;--danger:#ff3b30;--ok:#2ecc71;--r:18px}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",Segoe UI,Roboto,sans-serif;
-webkit-font-smoothing:antialiased;line-height:1.7;font-size:16px}
.wrap{max-width:880px;margin:0 auto;padding:0 20px 120px}
/* header */
.hd{display:flex;gap:24px;padding:40px 0 28px;border-bottom:3px solid var(--acc);flex-wrap:wrap}
.hd-cover{flex:0 0 240px}.cover{width:100%;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.12);display:block}
.hd-main{flex:1;min-width:280px}
.hd-main h1{font-size:26px;line-height:1.3;margin:0 0 10px;letter-spacing:-.01em}
.sub{display:flex;gap:14px;align-items:center;color:var(--sub);font-size:14px;flex-wrap:wrap}
.up{font-weight:600;color:var(--ink)}
.badges .chip{background:var(--acc-soft);border-radius:7px;padding:2px 9px;font-size:12px;margin-right:6px;color:var(--acc-dk);font-weight:500}
.parts{margin:14px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.parts .lbl{color:var(--sub);font-size:13px}.parts .p{font-size:12px;padding:4px 10px;border:1px solid var(--line);border-radius:8px;text-decoration:none;color:var(--sub)}
.parts .p.cur{background:var(--acc);color:var(--acc-dk);border-color:var(--acc);font-weight:600}
.summary{margin:14px 0;padding:12px 16px;background:var(--acc-soft);border-left:4px solid var(--acc);border-radius:8px;font-size:15px;font-weight:500}
.desc{color:var(--sub);font-size:13px;white-space:pre-wrap;margin:10px 0}
.tags{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.tag-chip{font-size:12px;color:var(--acc-dk);background:var(--acc-soft);border:1px solid #f0e0a0;border-radius:7px;padding:3px 9px}
/* toc */
.toc{position:sticky;top:0;z-index:5;background:rgba(250,250,250,.8);backdrop-filter:saturate(150%) blur(6px);-webkit-backdrop-filter:saturate(150%) blur(6px);
border-bottom:1px solid var(--line);margin:0 -20px 8px;padding:10px 20px;display:flex;align-items:center;gap:14px}
.toc-in{display:flex;gap:6px;overflow-x:auto;scrollbar-width:thin;flex:1}
.toc-in a{display:flex;flex-direction:column;min-width:84px;padding:6px 10px;border-radius:10px;text-decoration:none;color:var(--sub);font-size:12px;transition:.2s}
.toc-in a:hover{background:var(--acc-soft)}
.toc-in a b{color:var(--acc-dk);font-size:13px}.toc-in a i{font-style:normal;color:var(--sub);font-size:11px}
.watch{font-size:13px;color:var(--acc-dk);text-decoration:none;white-space:nowrap;font-weight:600}
.watch:hover{text-decoration:underline}
/* chapter */
.chapter{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:26px 28px;margin:18px 0;
box-shadow:0 1px 3px rgba(0,0,0,.04)}
.ch-head{display:flex;gap:16px;align-items:flex-start;border-bottom:2px solid var(--acc);padding-bottom:14px;margin-bottom:16px}
.ch-no{font-size:22px;font-weight:800;color:var(--acc-dk);flex:0 0 auto;background:var(--acc);padding:2px 10px;border-radius:8px}
.ch-titles h3{margin:0;font-size:19px;letter-spacing:-.01em}
.ch-meta{font-size:13px;color:var(--sub);margin-top:3px}
.ts{color:var(--acc-dk);text-decoration:none;border-bottom:1px dotted var(--acc-dk);font-weight:500}
.ts:hover{color:var(--warn);border-color:var(--warn)}
.ch-body h4{font-size:14px;color:var(--sub);margin:18px 0 8px;font-weight:600;letter-spacing:.02em;text-transform:uppercase}
.ch-body h4.warn{color:var(--warn)}
.ch-body h5{font-size:15px;color:var(--ink);margin:0 0 4px;font-weight:600}
.blk{list-style:none;padding:0;margin:0 0 6px}.blk li{display:flex;gap:10px;padding:4px 0;font-size:15px}
.blk .mk{flex:0 0 18px;text-align:center;font-weight:700}
.blk.q .mk{color:var(--acc-dk)}.blk.trap .mk{color:var(--warn)}.blk.kp .mk{color:var(--acc-dk)}
.blk.trap li{background:#fff5f0;border-radius:8px;padding:6px 10px;margin-bottom:4px}
.steps{margin:0 0 6px;padding-left:22px}.steps li{padding:4px 0;font-size:15px}
.steps li::marker{color:var(--acc-dk);font-weight:700}
.concl{margin:6px 0;padding:12px 16px;background:var(--acc-soft);border-radius:10px;font-weight:500;border-left:4px solid var(--acc)}
.quotes{display:flex;flex-direction:column;gap:8px}
.quotes blockquote{margin:0;padding:10px 16px;background:#fafafa;border-left:3px solid var(--acc);border-radius:0 8px 8px 0}
.quotes blockquote p{margin:0;font-size:14px}.quotes blockquote footer{margin-top:4px;font-size:12px}
/* detail sections (细致模式) */
.detail-toggle{margin:12px 0;border:1px solid var(--line);border-radius:12px;background:#fff}
.detail-toggle summary{cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 16px;font-size:14px;font-weight:700;color:var(--acc-dk);list-style:none}
.detail-toggle summary::-webkit-details-marker{display:none}
.detail-toggle summary::after{content:"+";width:22px;height:22px;border-radius:999px;background:var(--acc-soft);display:inline-flex;align-items:center;justify-content:center;font-weight:800}
.detail-toggle[open] summary::after{content:"-"}
.detail-toggle summary small{font-size:12px;font-weight:500;color:var(--sub);margin-left:auto}
.detail-toggle .detail-block{padding:0 16px 14px}
.detail-block{display:flex;flex-direction:column;gap:10px;margin:6px 0}
.detail-sec{padding:12px 16px;background:var(--soft);border-radius:10px;border-left:3px solid var(--acc)}
.detail-sec h5{font-size:14px;color:var(--acc-dk);margin:0 0 4px}
.detail-sec p{margin:0;font-size:14px;color:var(--ink);line-height:1.6}
/* media */
.media{margin:18px 0 4px}.media figcaption{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.tag{font-size:11px;font-weight:700;padding:2px 9px;border-radius:6px;letter-spacing:.04em}
.tag.svg{background:var(--acc-soft);color:var(--acc-dk);border:1px solid #f0e0a0}.tag.shot{background:#fff0e6;color:var(--warn)}
.cap{font-size:12px;color:var(--sub)}
.svg-wrap{background:var(--soft);border-radius:12px;padding:14px;border:1px solid var(--line)}
.svg-wrap svg{border-radius:8px}
.media img{width:100%;border-radius:12px;border:1px solid var(--line);display:block}
/* frame gallery (细致模式) */
.frame-gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;margin:18px 0 4px}
.frame-item{margin:0}.frame-item img{width:100%;border-radius:10px;border:1px solid var(--line);display:block;transition:transform .25s}
.frame-item img:hover{transform:scale(1.02)}
.frame-item figcaption{display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap}
/* footer */
.ft{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);color:var(--sub);font-size:12px}
.ft-warn{margin:0 0 10px;padding-left:18px}.ft-warn li{color:var(--warn);margin:3px 0}
@media (max-width:640px){.hd-cover{flex:1 1 100%}.chapter{padding:18px}.hd-main h1{font-size:22px}.frame-gallery{grid-template-columns:1fr}}
@media print{.toc{display:none}.chapter{break-inside:avoid;box-shadow:none}}
"""


def render_html(cfg: Config, meta: dict, transcript: dict, analysis: dict,
                svgs: list[str], audio_info: dict, out_path: Path,
                frames_dir: Path) -> Path:
    chapters = analysis.get("chapters", [])
    # 把 svg 绑到章节
    for ch, svg in zip(chapters, svgs):
        ch["_svg"] = svg
    head = _header(meta, analysis, audio_info)
    nav = _nav(chapters, meta)
    body = "".join(_chapter(ch, ch.get("_svg", ""), meta) for ch in chapters)
    foot = _footer(meta, analysis, audio_info, transcript)
    title = _esc(meta.get("title", "视频笔记"))
    html_doc = f"""<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title} · 视频笔记</title>
<style>{CSS}</style></head>
<body><div class="wrap">
{head}
{nav}
{body}
{foot}
</div></body></html>"""
    html_doc = inject_note_lightbox(html_doc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 截图目录与 HTML 同级（相对路径 frames/...）
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
