"""分 P 批量处理：对多 P 视频逐 P 生成笔记，并生成 index.html 聚合页。

设计要点：
- 复用单 P pipeline，逐 P 串行处理（避免并发冲击 LLM / 转写 API 限额）
- 每个 P 独立输出目录（P01_xxx/, P02_xxx/...），互不干扰
- 最终生成 index.html，汇总所有 P 的标题、章节数、时长、链接
- 支持断点续跑：已完成的 P 跳过（检测 notes.html 是否存在）
- 进度回调：每完成一个 P 推送进度
"""
from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .util import log, safe_name, fmt_ts
from . import metadata as M


def _esc(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


def _part_dir(cfg: Config, base_title: str, part_index: int, part_title: str) -> Path:
    """生成分 P 输出目录名：合集标题/P01_标题前缀/

    各 P 笔记放在合集目录下，使 index.html 中的相对路径天然正确。
    """
    safe_base = safe_name(base_title, 50) or "notes"
    safe = safe_name(part_title or f"P{part_index}", 40) or f"P{part_index}"
    name = f"P{part_index:02d}_{safe}"
    return cfg.output_dir / safe_base / name


def _check_part_done(out_dir: Path) -> bool:
    """检查某个 P 是否已完成（notes.html 存在）。"""
    return (out_dir / "notes.html").exists()


def batch_pipeline(
    cfg: Config,
    url: str,
    *,
    parts: list[int] | None = None,
    no_frames: bool = False,
    no_slice: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """批量处理多 P 视频。

    Args:
        cfg: 配置
        url: 视频 URL（B站多 P 链接）
        parts: 指定要处理的 P 号列表；None 表示处理全部 P
        no_frames: 跳过抽帧
        no_slice: 跳过切片
        on_progress: 进度回调 (completed, total, current_title)

    Returns:
        index.html 的路径
    """
    from run import pipeline as run_pipeline

    cfg.ensure_dirs()

    # 1. 抓取元数据，获取分 P 列表
    log.info("batch", f"批量模式：抓取分 P 列表…")
    meta = M.fetch_metadata(cfg, url, part=1)
    all_parts = meta.get("parts") or []

    if len(all_parts) <= 1:
        log.warn("batch", "视频只有 1 个 P，无需批量处理，退回单 P 模式")
        html_path = run_pipeline(
            cfg, url, part=1, no_frames=no_frames,
            no_slice=no_slice, stub_transcript=False,
        )
        return html_path

    # 筛选要处理的 P
    if parts:
        target_parts = [p for p in all_parts if p["index"] in parts]
    else:
        target_parts = all_parts

    total = len(target_parts)
    log.info("batch", f"共 {total} 个 P 需要处理")

    base_url = meta.get("base_url", url)
    base_title = meta.get("title", "")

    # 收集每个 P 的结果信息
    results: list[dict[str, Any]] = []
    completed = 0

    for part_info in target_parts:
        p_idx = part_info["index"]
        p_title = part_info.get("title", f"P{p_idx}")

        if on_progress:
            on_progress(completed, total, p_title)

        out_dir = _part_dir(cfg, base_title, p_idx, p_title)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 断点续跑：已完成的跳过
        if _check_part_done(out_dir):
            log.info("batch", f"P{p_idx} 已完成，跳过")
            info = _load_part_info(out_dir, p_idx, p_title, part_info)
            results.append(info)
            completed += 1
            continue

        log.info("batch", f"━━━ 开始处理 P{p_idx}/{total}：{p_title} ━━━")
        t0 = time.time()

        try:
            part_url = M._build_part_url(base_url, p_idx)
            html_path = run_pipeline(
                cfg, part_url,
                part=p_idx,
                no_frames=no_frames,
                no_slice=no_slice,
                stub_transcript=False,
                out_dir=out_dir,
            )
            elapsed = time.time() - t0
            log.info("batch", f"P{p_idx} 完成（{elapsed:.0f}s）")

            info = _load_part_info(html_path.parent, p_idx, p_title, part_info)
            info["elapsed"] = round(elapsed, 1)
            results.append(info)

        except Exception as e:
            log.error("batch", f"P{p_idx} 失败：{e}")
            results.append({
                "part": p_idx,
                "title": p_title,
                "dir": out_dir.name,
                "duration": part_info.get("duration"),
                "error": str(e),
                "done": False,
            })

        completed += 1

    if on_progress:
        on_progress(completed, total, "生成聚合页")

    # 2. 生成 index.html 聚合页
    index_path = _generate_index(cfg, meta, results)
    log.info("batch", f"聚合页已生成：{index_path}")
    log.info("batch", f"━━━ 批量完成：{completed}/{total} P 成功 ━━━")

    return index_path


def _load_part_info(out_dir: Path, p_idx: int, p_title: str, part_info: dict) -> dict:
    """从输出目录读取已生成笔记的信息。"""
    import json

    info: dict[str, Any] = {
        "part": p_idx,
        "title": p_title,
        "dir": out_dir.name,
        "duration": part_info.get("duration"),
        "done": (out_dir / "notes.html").exists(),
        "has_cover": (out_dir / "cover.jpg").exists(),
        "chapters": 0,
        "has_full": (out_dir / "full.png").exists(),
        "slices": 0,
    }

    data_file = out_dir / "notes_data.json"
    if data_file.exists():
        try:
            data = json.loads(data_file.read_text(encoding="utf-8"))
            analysis = data.get("analysis", {})
            meta = data.get("meta", {})
            info["chapters"] = len(analysis.get("chapters", []))
            info["duration"] = meta.get("duration", info["duration"])
            info["slices"] = len(data.get("slices", []))
            # 用实际笔记中的标题（可能更准确）
            if meta.get("title"):
                info["title"] = meta["title"]
        except Exception:
            pass

    return info


# ============================================================
#  index.html 聚合页生成
# ============================================================

INDEX_CSS = """
:root{--bg:#fbfbfd;--card:#fff;--ink:#1d1d1f;--sub:#6e6e73;--line:#e5e5ea;--soft:#f5f5f7;
--acc:#0071e3;--acc2:#00b8d9;--warn:#ff8a3d;--danger:#ff3b30;--ok:#34c759;--r:18px}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",Segoe UI,Roboto,sans-serif;
-webkit-font-smoothing:antialiased;line-height:1.7;font-size:16px}
.wrap{max-width:880px;margin:0 auto;padding:0 20px 120px}
/* header */
.hd{padding:48px 0 32px;border-bottom:1px solid var(--line)}
.hd h1{font-size:30px;letter-spacing:-.02em;margin:0 0 8px;line-height:1.3}
.hd .sub{color:var(--sub);font-size:14px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.hd .up{font-weight:600;color:var(--ink)}
.badges .chip{background:var(--soft);border-radius:7px;padding:2px 9px;font-size:12px;margin-right:6px}
.summary{margin:16px 0;padding:14px 18px;background:linear-gradient(135deg,#f0f6ff,#f5fbff);
border-left:3px solid var(--acc);border-radius:8px;font-size:15px;color:var(--ink)}
/* stats */
.stats{display:flex;gap:24px;margin:20px 0;padding:20px 0;border-bottom:1px solid var(--line)}
.stat{text-align:center;flex:1}
.stat .num{font-size:28px;font-weight:700;color:var(--acc);letter-spacing:-.02em}
.stat .lbl{font-size:12px;color:var(--sub);text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
/* parts */
.parts-title{font-size:18px;font-weight:600;margin:28px 0 16px;letter-spacing:-.01em}
.part-card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
padding:0;margin:14px 0;overflow:hidden;transition:box-shadow .25s,transform .25s;
text-decoration:none;color:inherit;display:flex}
.part-card:hover{box-shadow:0 8px 30px rgba(0,0,0,.08);transform:translateY(-2px)}
.part-card.error{border-color:rgba(255,59,48,.2);background:#fff5f5}
.part-cover{flex:0 0 200px;aspect-ratio:16/9;background:var(--soft);overflow:hidden;position:relative}
.part-cover img{width:100%;height:100%;object-fit:cover;display:block}
.part-cover .pno{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.7);color:#fff;
font-size:12px;font-weight:600;padding:2px 8px;border-radius:6px}
.part-cover-placeholder{width:100%;height:100%;display:flex;align-items:center;justify-content:center;
background:var(--soft);color:var(--sub);font-size:13px}
.part-info{flex:1;padding:16px 20px;display:flex;flex-direction:column;justify-content:center}
.part-info h3{margin:0 0 6px;font-size:16px;letter-spacing:-.01em}
.part-meta{display:flex;gap:14px;font-size:13px;color:var(--sub);flex-wrap:wrap}
.part-meta .m{display:flex;align-items:center;gap:4px}
.part-meta .m b{color:var(--ink);font-weight:600}
.part-status{margin-top:8px;font-size:12px}
.part-status.ok{color:var(--ok)}.part-status.err{color:var(--danger)}
.part-arrow{flex:0 0 auto;display:flex;align-items:center;padding:0 20px;color:var(--acc);font-size:20px}
/* footer */
.ft{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);color:var(--sub);font-size:12px}
@media(max-width:640px){
  .part-card{flex-direction:column}
  .part-cover{flex:1 1 100%}
  .part-arrow{display:none}
  .stats{flex-wrap:wrap;gap:16px}
  .stat{min-width:80px}
}
@media print{.part-card{break-inside:avoid}}
"""


def _generate_index(cfg: Config, meta: dict, results: list[dict]) -> Path:
    """生成聚合 index.html。"""
    from datetime import datetime

    base_title = meta.get("title", "视频笔记合集")
    uploader = meta.get("uploader", "")
    total_parts = len(results)
    done_parts = sum(1 for r in results if r.get("done"))
    total_duration = sum(r.get("duration") or 0 for r in results)
    total_chapters = sum(r.get("chapters", 0) for r in results)

    # 封面
    cover_html = ""
    # 找第一个有封面的 P
    for r in results:
        if r.get("has_cover"):
            cover_html = f"<img class='cover' src='{_esc(r['dir'])}/cover.jpg' alt='' style='width:180px;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.1);float:right;margin-left:24px'/>"
            break

    # 统计
    stats_html = f"""
<div class="stats">
  <div class="stat"><div class="num">{total_parts}</div><div class="lbl">分 P 总数</div></div>
  <div class="stat"><div class="num">{done_parts}</div><div class="lbl">已完成</div></div>
  <div class="stat"><div class="num">{total_chapters}</div><div class="lbl">章节</div></div>
  <div class="stat"><div class="num">{fmt_ts(total_duration) if total_duration else '-'}</div><div class="lbl">总时长</div></div>
</div>"""

    # 分 P 卡片
    cards_html = ""
    for r in results:
        p_idx = r["part"]
        title = r.get("title", f"P{p_idx}")
        is_done = r.get("done", False)
        is_error = bool(r.get("error"))

        if is_done:
            cover = ""
            if r.get("has_cover"):
                cover = f"<img src='{_esc(r['dir'])}/cover.jpg' alt=''/>"
            else:
                cover = f"<div class='part-cover-placeholder'>无封面</div>"

            duration = r.get("duration")
            dur_str = fmt_ts(duration) if duration else "-"
            ch_str = str(r.get("chapters", 0))
            sl_str = str(r.get("slices", 0))

            status_cls = "ok"
            status_txt = "✓ 已完成"
            if is_error:
                status_cls = "err"
                status_txt = f"✗ 失败：{r.get('error', '')}"

            cards_html += f"""
<a class="part-card" href="{_esc(r['dir'])}/notes.html">
  <div class="part-cover">
    <span class="pno">P{p_idx}</span>
    {cover}
  </div>
  <div class="part-info">
    <h3>{_esc(title)}</h3>
    <div class="part-meta">
      <span class="m"><b>{dur_str}</b></span>
      <span class="m">章节 <b>{ch_str}</b></span>
      <span class="m">切片 <b>{sl_str}</b></span>
    </div>
    <div class="part-status {status_cls}">{_esc(status_txt)}</div>
  </div>
  <div class="part-arrow">→</div>
</a>"""
        else:
            err = r.get("error", "未生成")
            cards_html += f"""
<a class="part-card error" href="javascript:void(0)">
  <div class="part-cover">
    <span class="pno">P{p_idx}</span>
    <div class="part-cover-placeholder">失败</div>
  </div>
  <div class="part-info">
    <h3>{_esc(title)}</h3>
    <div class="part-meta"><span class="m">处理失败</span></div>
    <div class="part-status err">✗ {_esc(err)}</div>
  </div>
</a>"""

    # 描述
    desc = (meta.get("description") or "").strip()
    desc_html = f"<p style='color:var(--sub);font-size:13px;white-space:pre-wrap;margin:10px 0'>{_esc(desc[:400])}{'…' if len(desc)>400 else ''}</p>" if desc else ""

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    watch_url = meta.get("base_url") or meta.get("webpage_url", "")

    html_doc = f"""<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_esc(base_title)} · 笔记合集</title>
<style>{INDEX_CSS}</style></head>
<body><div class="wrap">
<header class="hd">
  {cover_html}
  <h1>{_esc(base_title)}</h1>
  <div class="sub">
    <span class="up">UP · {_esc(uploader)}</span>
    <span class="badges"><span class="chip">{_esc(meta.get('extractor',''))}</span>"""
    if total_duration:
        html_doc += f"""<span class="chip">{fmt_ts(total_duration)}</span>"""
    html_doc += f"""<span class="chip">{total_parts} P</span></span>
    <a href="{_esc(watch_url)}" target="_blank" style="color:var(--acc);text-decoration:none;font-size:13px">原视频 ↗</a>
  </div>
  {desc_html}
</header>

{stats_html}

<h2 class="parts-title">分 P 笔记</h2>
{cards_html}

<footer class="ft">
  <p>由 vnotes 批量生成 · {gen_time} · 共 {done_parts}/{total_parts} P 成功</p>
</footer>

</div></body></html>"""

    # 输出到 output 目录下的合集目录
    safe = safe_name(base_title, 50) or "notes"
    index_dir = cfg.output_dir / safe
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index.html"
    index_path.write_text(html_doc, encoding="utf-8")
    return index_path
