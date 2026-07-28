"""动态 SVG 生成：按章节内容与 svg_type，让 LLM 产出内联 SVG；清洗校验。

每张图来自本章真实内容，包含关键词/关系/箭头/标签，禁止装饰图与重复套壳。
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import Config
from . import llm
from .util import log

TYPE_GUIDE = {
    "flow": "画步骤/路径：用方框+箭头串联流程节点，标注每步关键动作，分支用菱形决策点。",
    "concept": "画关系/分层：用分层卡片或节点-连线表达概念间的包含、依赖、并列关系，标注关系名。",
    "timeline": "画时间线：横向时间轴，关键节点标时间戳与事件，表现随时间的变化。",
    "comparison": "画矩阵/对比表：多列对比维度，行是对象，单元格写差异关键词。",
    "risk": "画检查表/决策树：误区用警示标记，决策点用分支，列出『是否…』判断与后果。",
    "data": "画数据简图：用条形/折线/占比等最简图形表达本章数据，标注数值与单位。",
    "causation": "画因果链路：用箭头链 A→B→C 表达因果，标注每环的关键词与条件。",
}

PALETTE = """可用配色（克制，勿堆叠荧光色）：
背景 #ffffff / 卡片底 #f5f5f7 / 主墨 #1d1d1f / 次墨 #6e6e73 /
强调蓝 #0071e3 / 强调青 #00b8d9 / 强调橙 #ff8a3d / 警示红 #ff3b30 / 成功绿 #34c759。
连线/边框用 #d2d2d7。"""

SYSTEM = f"""你是数据可视化与信息图专家。根据给定章节内容，生成【一张】自包含的内联 SVG 代码。

铁律：
1. 图必须来自本章真实内容：包含具体关键词、关系、箭头或标签，禁止空洞装饰图、禁止与别章雷同的套壳模板。
2. 只输出 SVG 代码本身（以 <svg 开头、</svg> 结尾），不要 markdown、不要解释、不要 ``` 代码块标记。
3. 根 <svg> 必须设 viewBox="0 0 720 420"，不要在根 <svg> 上写 width/height 属性（由页面 CSS 缩放）。
   但子元素（<rect>、<circle>、<line> 等）的 width/height 必须保留，这是它们的尺寸！
4. 文字用中文，字号 >=13，确保清晰可读；用 <text> 而非位图。
5. {PALETTE}
6. 禁止 <script>、禁止外部资源(href/http)、禁止 on* 事件属性、禁止 base64。
7. 适当留白，元素不重叠、不溢出 viewBox。
8. 布局建议：标题放顶部 y=30-50，主内容区 y=70-380，底部留白。
9. 箭头用 <defs><marker> 定义，连线用 <line> 或 <path>，确保箭头可见。"""


def _sanitize(svg: str) -> str:
    """清洗 SVG：只处理根 <svg> 元素的 width/height，保留子元素尺寸。"""
    svg = svg.strip()
    # 去代码块
    m = re.search(r"```(?:svg|xml|html)?\s*(<svg.*?</svg>)\s*```", svg, re.S | re.I)
    if m:
        svg = m.group(1)
    s = svg.find("<svg")
    e = svg.rfind("</svg>")
    if s != -1 and e != -1:
        svg = svg[s:e + 6]
    elif s != -1:
        # LLM 输出被截断（无 </svg>），尝试补全
        svg = svg[s:] + "</svg>"
    if not svg.lower().startswith("<svg"):
        return ""

    # 移除危险内容
    svg = re.sub(r"<script[\s\S]*?</script>", "", svg, flags=re.I)
    svg = re.sub(r"\son\w+\s*=\s*([\"\']).*?\1", "", svg, flags=re.I)
    svg = re.sub(r"href\s*=\s*([\"\'])\s*https?:.*?\1", "", svg, flags=re.I)

    # 分离根 <svg ...> 标签和内部内容
    # 匹配根 svg 标签（第一个 <svg ...>）
    root_match = re.match(r"(<svg)([^>]*)(>)", svg, re.I)
    if not root_match:
        return ""

    root_attrs = root_match.group(2)
    inner = svg[root_match.end():]  # </svg> 之前的内容

    # 补 xmlns
    if "xmlns" not in root_attrs:
        root_attrs += " xmlns='http://www.w3.org/2000/svg'"
    # 补/规范 viewBox
    if "viewBox" not in root_attrs and "viewbox" not in root_attrs.lower():
        root_attrs += " viewBox='0 0 720 420'"
    # 只去根 <svg> 上的 width/height 固定值，交由 CSS（不影响子元素！）
    root_attrs = re.sub(r"\s*width\s*=\s*([\"\'])[^\"\']*\1", "", root_attrs)
    root_attrs = re.sub(r"\s*height\s*=\s*([\"\'])[^\"\']*\1", "", root_attrs)
    # 加 CSS 响应式样式
    if "style=" not in root_attrs:
        root_attrs += ' style="width:100%;height:auto;display:block"'
    else:
        root_attrs = re.sub(r'style\s*=\s*([\"\'])([^\"\']*)\1',
                            lambda m: f'style="width:100%;height:auto;display:block;{m.group(2)}"',
                            root_attrs, count=1)

    return f"<svg{root_attrs}>{inner}"


def _gen_one(cfg: Config, chapter: dict, retry: int = 0) -> str:
    """生成单章 SVG，带一次重试（重试时简化 prompt 降低截断风险）。"""
    guide = TYPE_GUIDE.get(chapter.get("svg_type", "concept"), TYPE_GUIDE["concept"])
    content = (
        f"章节标题：{chapter.get('title','')}\n"
        f"时间段：{chapter.get('t_start')}s - {chapter.get('t_end')}s\n"
        f"svg_type：{chapter.get('svg_type')}（{guide}）\n"
        f"选图理由：{chapter.get('svg_rationale','')}\n"
        f"要点：{chapter.get('key_points', [])}\n"
        f"步骤：{chapter.get('steps', [])}\n"
        f"问题：{chapter.get('questions', [])}\n"
        f"陷阱：{chapter.get('traps', [])}\n"
        f"结论：{chapter.get('conclusion','')}\n"
        f"关键引用：{chapter.get('key_quotes', [])}\n"
        f"视觉锚点：{chapter.get('visual_anchors', [])}\n"
    )
    if retry == 0:
        user = (
            f"据此生成 SVG：\n{content}\n\n"
            f"要求：只输出以 <svg 开头、</svg> 结尾的纯 SVG 代码。"
            f"根 <svg> 设 viewBox=\"0 0 720 420\"，不写 width/height。"
            f"子元素 rect/circle/line 等的 width/height/x/y 必须写全。"
        )
        max_tok = 4096
    else:
        # 重试：极简 prompt，只给最核心信息，减少 token 消耗
        pts = (chapter.get("key_points") or [])[:4]
        steps = (chapter.get("steps") or [])[:4]
        traps = (chapter.get("traps") or [])[:3]
        brief = f"标题：{chapter.get('title','')}\n要点：{pts}\n步骤：{steps}\n陷阱：{traps}"
        user = (
            f"据此生成简洁 SVG（控制在 1500 字符内）：\n{brief}\n\n"
            f"只输出 <svg viewBox=\"0 0 720 420\">...</svg>，"
            f"子元素必须写全 width/height/x/y。不要解释。"
        )
        max_tok = 3000

    raw = llm.chat(
        cfg,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=0.4, max_tokens=max_tok, timeout=180,
    )
    result = _sanitize(raw)

    # 校验：sanitize 后如果为空或太短，重试一次
    if not result or len(result) < 100:
        if retry < 1:
            log.warn("svg", f"第{chapter['id']}章 SVG 校验失败（长度={len(result)}），重试…")
            return _gen_one(cfg, chapter, retry=retry + 1)
        return ""

    # 校验：必须有 <svg 和 </svg> 且包含 <text> 或 <rect> 等实际元素
    has_content = any(tag in result.lower() for tag in ["<text", "<rect", "<circle", "<line", "<path", "<polygon"])
    if not has_content:
        if retry < 1:
            log.warn("svg", f"第{chapter['id']}章 SVG 无实际图形元素，重试…")
            return _gen_one(cfg, chapter, retry=retry + 1)
        return ""

    return result


def generate_svgs(cfg: Config, analysis: dict, work_dir: Path) -> list[str]:
    """为每章生成 SVG，返回与 chapters 等长的 svg 字符串列表。
    
    支持并行生成（cfg.svg_parallel），多章同时调用 LLM，速度提升 4-5 倍。
    """
    chapters = analysis.get("chapters", [])
    if not chapters:
        return []

    # 并行生成
    if getattr(cfg, "svg_parallel", True) and len(chapters) > 1:
        return _generate_svgs_parallel(cfg, chapters)

    # 串行生成（兜底）
    svgs: list[str] = []
    for ch in chapters:
        log.info("svg", f"第{ch['id']}章 [{ch.get('svg_type')}] {ch.get('title','')}")
        try:
            svg = _gen_one(cfg, ch)
            if not svg:
                log.warn("svg", f"第{ch['id']}章 SVG 生成失败，用兜底占位")
                svg = _fallback(ch)
        except Exception as e:
            log.warn("svg", f"第{ch['id']}章生成异常：{e}，用兜底占位")
            svg = _fallback(ch)
        svgs.append(svg)
    return svgs


def _generate_svgs_parallel(cfg: Config, chapters: list[dict]) -> list[str]:
    """并行生成所有章节的 SVG，保持返回顺序与 chapters 一致。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = min(getattr(cfg, "svg_max_workers", 6), len(chapters))
    log.info("svg", f"并行生成 {len(chapters)} 章 SVG（{max_workers} 线程）")

    results: dict[int, str] = {}  # chapter_id -> svg

    def _worker(ch: dict) -> tuple[int, str]:
        log.info("svg", f"第{ch['id']}章 [{ch.get('svg_type')}] {ch.get('title','')}")
        try:
            svg = _gen_one(cfg, ch)
            if not svg:
                log.warn("svg", f"第{ch['id']}章 SVG 生成失败，用兜底占位")
                svg = _fallback(ch)
            return (ch["id"], svg)
        except Exception as e:
            log.warn("svg", f"第{ch['id']}章生成异常：{e}，用兜底占位")
            return (ch["id"], _fallback(ch))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, ch): ch for ch in chapters}
        for future in as_completed(futures):
            ch_id, svg = future.result()
            results[ch_id] = svg

    # 按原始顺序组装
    return [results.get(ch["id"], _fallback(ch)) for ch in chapters]


def _fallback(chapter: dict) -> str:
    """按 svg_type 生成不同类型的兜底图，确保不空白且有信息量。"""
    title = (chapter.get("title", "") or "本章")[:28]
    svg_type = chapter.get("svg_type", "concept")
    pts = [p for p in (chapter.get("key_points") or []) if p][:5]
    if not pts:
        pts = [chapter.get("conclusion", "暂无要点")]
    steps = [s for s in (chapter.get("steps") or []) if s][:5]
    traps = [t for t in (chapter.get("traps") or []) if t][:3]

    base = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 720 420' "
        f"style='width:100%;height:auto;display:block'>"
        f"<rect width='720' height='420' fill='#ffffff'/>"
        f"<text x='40' y='40' font-size='20' font-weight='700' fill='#1d1d1f'>{_esc_xml(title)}</text>"
        f"<line x1='40' y1='52' x2='680' y2='52' stroke='#d2d2d7' stroke-width='1'/>"
    )

    if svg_type == "flow" and steps:
        # 流程图兜底：竖向步骤 + 箭头
        body = ""
        n = len(steps)
        step_h = min(50, (340 - 20) // max(n, 1))
        gap = 12
        y0 = 70
        for i, step in enumerate(steps):
            y = y0 + i * (step_h + gap)
            body += (
                f"<rect x='80' y='{y}' width='560' height='{step_h}' rx='10' "
                f"fill='#f5f5f7' stroke='#d2d2d7'/>"
                f"<text x='100' y='{y + step_h // 2 + 5}' font-size='15' fill='#1d1d1f'>"
                f"{_esc_xml(str(step)[:40])}</text>"
            )
            if i < n - 1:
                ay = y + step_h
                body += (
                    f"<line x1='360' y1='{ay}' x2='360' y2='{ay + gap}' "
                    f"stroke='#0071e3' stroke-width='2'/>"
                    f"<polygon points='356,{ay+gap-4} 364,{ay+gap-4} 360,{ay+gap}' "
                    f"fill='#0071e3'/>"
                )
        return base + body + "</svg>"

    elif svg_type == "timeline" and pts:
        # 时间线兜底：横向轴线 + 节点
        body = ""
        n = len(pts)
        x0, x1 = 60, 660
        body += f"<line x1='{x0}' y1='210' x2='{x1}' y2='210' stroke='#d2d2d7' stroke-width='2'/>"
        for i, pt in enumerate(pts):
            x = x0 + (x1 - x0) * i // max(n - 1, 1) if n > 1 else (x0 + x1) // 2
            y_text = 190 if i % 2 == 0 else 250
            body += (
                f"<circle cx='{x}' cy='210' r='6' fill='#0071e3'/>"
                f"<text x='{x}' y='{y_text}' text-anchor='middle' font-size='13' fill='#1d1d1f'>"
                f"{_esc_xml(str(pt)[:24])}</text>"
            )
        return base + body + "</svg>"

    elif svg_type == "risk" and traps:
        # 风险检查表兜底
        body = ""
        y = 70
        for trap in traps:
            body += (
                f"<rect x='40' y='{y}' width='640' height='44' rx='8' "
                f"fill='#fff8f3' stroke='#ff8a3d' stroke-width='1'/>"
                f"<text x='60' y='{y+28}' font-size='15' fill='#1d1d1f'>⚠ {_esc_xml(str(trap)[:40])}</text>"
            )
            y += 54
        return base + body + "</svg>"

    else:
        # 默认兜底：要点卡片
        lines = ""
        y0 = 70
        card_h = min(50, (340) // max(len(pts), 1))
        for i, p in enumerate(pts):
            y = y0 + i * (card_h + 8)
            lines += (
                f"<rect x='40' y='{y}' width='640' height='{card_h}' rx='10' "
                f"fill='#f5f5f7' stroke='#d2d2d7'/>"
                f"<circle cx='62' cy='{y + card_h // 2}' r='4' fill='#0071e3'/>"
                f"<text x='80' y='{y + card_h // 2 + 5}' font-size='15' fill='#1d1d1f'>"
                f"{_esc_xml(str(p)[:36])}</text>"
            )
        return base + lines + "</svg>"


def _esc_xml(s: str) -> str:
    """转义 XML 特殊字符。"""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;"))
