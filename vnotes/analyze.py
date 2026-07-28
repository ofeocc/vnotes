"""内容分析：LLM 按视频自然结构拆章，提炼问题/陷阱/步骤/结论，决定 SVG 类型与帧需求。

输出中文笔记（英文视频自动翻译为中文）。对超长转写做分块分析再合并。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config
from . import llm
from .util import log, fmt_ts, write_json, read_json

SVG_TYPES = ["flow", "concept", "timeline", "comparison", "risk", "data", "causation"]

CHUNK_CHARS = 22000  # 每块约 22k 字符（含时间戳），DeepSeek 上下文足够
OVERLAP_CHARS = 2000

SYSTEM = """你是资深内容编辑与技术笔记专家。你会收到一段带时间戳的视频转写文本（可能含元数据），任务是把它整理成结构化中文笔记。

硬性要求：
1. 按视频【自然结构】拆章——以话题切换、讲解阶段变化为准，不要按固定长度机械切割；通常 4~12 章为宜。
2. 章节时间戳 t_start/t_end 必须来自转写文本中 "sec=数字" 的真实秒数，单位秒。禁止把 21:31 写成 21.31；21:31 必须写成 1291。
3. 每章提炼以下内容，写成【白话短句】，只保留该章真正涉及的项（没有就给空数组）：
   - questions：本章提出或回答的问题
   - traps：常见陷阱/误区/坑
   - steps：操作步骤或推理步骤（有序）
   - conclusion：一句话结论
   - key_points：要点（短句列表）
   - key_quotes：关键引用，每条带 t（秒）和 text
   - visual_anchors：视觉锚点（画面里值得注意的关键元素，用于配图定位）
4. 为每章选一个 svg_type，依据内容类型：
   - flow：流程/步骤/路径
   - concept：概念关系/分层结构
   - timeline：随时间变化的过程
   - comparison：对比/矩阵
   - risk：风险/误区检查表/决策树
   - data：数据简图
   - causation：因果链路
   并在 svg_rationale 里一句话说明为什么选它、图里要表现什么关系。
5. 判断 is_ui_demo：视频是否在演示某个界面/工具操作（画布、控制台、剪辑器、文档、软件界面等）。是则为 true。
6. 全部用中文输出；若转写是英文，先理解再翻译成中文笔记。
7. 严格输出 JSON，不要任何额外文字。"""

SCHEMA_HINT = """输出 JSON 结构：
{
  "is_ui_demo": false,
  "video_summary": "一句话概括全片",
  "chapters": [
    {
      "title": "章节标题",
      "t_start": 0.0,
      "t_end": 0.0,
      "svg_type": "flow",
      "svg_rationale": "图里要表现什么",
      "questions": ["..."],
      "traps": ["..."],
      "steps": ["..."],
      "conclusion": "...",
      "key_points": ["..."],
      "key_quotes": [{"t": 0.0, "text": "..."}],
      "visual_anchors": ["..."]
    }
  ]
}"""

SUMMARY_SYSTEM = """你是内容压缩专家。将这段视频转写文本压缩为结构化摘要。
要求：
1. 保留所有时间戳信息（[M:SS-M:SS] 格式）
2. 保留关键论点、步骤、结论、数据
3. 去除重复、口语化、无信息量的内容
4. 输出纯文本，不超过 6000 字符
5. 用中文输出"""


# ---- 细致笔记模式 prompt ----
DETAILED_SYSTEM = """你是资深内容编辑与技术笔记专家。你会收到一段带时间戳的视频转写文本（可能含元数据），任务是把它整理成【细致】的结构化中文笔记。

硬性要求：
1. 按视频【自然结构】拆章——比常规笔记更细，但不要把连续讲同一代码/同一概念的几句话拆成很多碎章；课程/讲解类通常 6~14 章为宜。
2. 章节时间戳 t_start/t_end 必须来自转写文本中 "sec=数字" 的真实秒数，单位秒。禁止把 21:31 写成 21.31；21:31 必须写成 1291。
3. 每章提炼以下内容，写成【白话短句】，尽量详细：
   - questions：本章提出或回答的问题（尽量多列）
   - traps：常见陷阱/误区/坑
   - steps：操作步骤或推理步骤（有序，尽量详细到每一步）
   - conclusion：一句话结论
   - key_points：要点（短句列表，比常规模式更全面）
   - key_quotes：关键引用，每条带 t（秒）和 text——【每章 5~10 条】，尽量覆盖章节内的重要时间点
   - visual_anchors：视觉锚点（画面里值得注意的关键元素）
4. 每章额外生成 detail_sections（子板块）：
   - 把章节内容按子话题拆分，每个子板块有 title 和 content（2~4 句详细解释）
   - 子板块数量 2~5 个，视章节复杂度而定
5. 每章生成 frame_moments（关键帧时刻）：
   - 列出本章中【画面有重要信息】的时间点，每条带 t（秒）和 desc（一句话描述画面内容）
   - 关注：图表出现、代码演示、界面操作、对比展示、关键文字画面等
   - 每章 2~6 个帧时刻
6. 为每章选一个 svg_type（flow/concept/timeline/comparison/risk/data/causation），并在 svg_rationale 里说明。
7. 判断 is_ui_demo。
8. 全部用中文输出；若转写是英文，先理解再翻译成中文笔记。
9. 严格输出 JSON，不要任何额外文字。"""

DETAILED_SCHEMA_HINT = """输出 JSON 结构：
{
  "is_ui_demo": false,
  "video_summary": "一句话概括全片",
  "chapters": [
    {
      "title": "章节标题",
      "t_start": 0.0,
      "t_end": 0.0,
      "svg_type": "flow",
      "svg_rationale": "图里要表现什么",
      "questions": ["..."],
      "traps": ["..."],
      "steps": ["..."],
      "conclusion": "...",
      "key_points": ["..."],
      "key_quotes": [{"t": 0.0, "text": "..."}],
      "visual_anchors": ["..."],
      "detail_sections": [{"title": "子板块标题", "content": "2~4句详细解释"}],
      "frame_moments": [{"t": 0.0, "desc": "画面描述"}]
    }
  ]
}"""


def _build_lines(segments: list[dict]) -> str:
    """转写段 -> 带时间戳的紧凑文本。"""
    lines = []
    for s in segments:
        lines.append(
            f"[sec={s['start']:.2f}-{s['end']:.2f} | {fmt_ts(s['start'])}-{fmt_ts(s['end'])}] {s['text']}"
        )
    return "\n".join(lines)


def _chunks(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, i = [], 0
    while i < len(text):
        end = min(i + CHUNK_CHARS, len(text))
        # 在换行处断开
        if end < len(text):
            nl = text.rfind("\n", i, end)
            if nl > i + OVERLAP_CHARS:
                end = nl
        out.append(text[i:end])
        i = end - OVERLAP_CHARS if end < len(text) else end
        if i <= 0:
            break
    return out


def _looks_like_mmss_decimal(chapters: list[dict], audio_dur: float | None) -> bool:
    """检测 LLM 是否把 MM:SS 写成 MM.SS 小数。"""
    if not audio_dur or audio_dur < 120 or not chapters:
        return False
    ends = [float(ch.get("t_end") or 0) for ch in chapters if ch.get("t_end") is not None]
    if not ends:
        return False
    max_end = max(ends)
    audio_min = audio_dur / 60.0
    return max_end < audio_dur * 0.25 and abs(max_end - audio_min) <= 3.0


def _mmss_decimal_to_seconds(value: Any) -> Any:
    if not isinstance(value, (int, float)):
        return value
    if value <= 0:
        return value
    minutes = int(value)
    # JSON 数字会丢掉 20.10 的末尾 0，用 .2f 还原成两位 MM.SS。
    seconds = int(round((float(value) - minutes) * 100))
    if seconds >= 60:
        return value
    return float(minutes * 60 + seconds)


def _repair_mmss_decimal_times(result: dict, audio_dur: float | None) -> bool:
    """把 21.31 这类 MM.SS 误写修正为秒数 1291。"""
    chapters = result.get("chapters", [])
    if not _looks_like_mmss_decimal(chapters, audio_dur):
        return False

    for ch in chapters:
        for key in ("t_start", "t_end"):
            ch[key] = _mmss_decimal_to_seconds(ch.get(key))
        for q in ch.get("key_quotes", []) or []:
            q["t"] = _mmss_decimal_to_seconds(q.get("t"))
        for m in ch.get("frame_moments", []) or []:
            m["t"] = _mmss_decimal_to_seconds(m.get("t"))
    log.warn("analyze", "检测到章节时间像 MM.SS 小数，已自动换算为秒数，避免只覆盖视频开头几十秒")
    return True


def _analyze_chunk(cfg: Config, meta: dict, chunk: str, idx: int, total: int) -> dict:
    meta_brief = (
        f"标题：{meta.get('title','')}\nUP/频道：{meta.get('uploader','')}\n"
        f"简介：{(meta.get('description','') or '')[:600]}\n"
        f"标签：{', '.join(meta.get('tags', [])[:15])}\n"
        f"平台：{meta.get('extractor','')}\n"
    )
    hint = ""
    if meta.get("yt_chapters"):
        hint = "\n（UP主已标注的章节，可作拆分参考）：" + " | ".join(
            f"{fmt_ts(c['t_start'])} {c['title']}" for c in meta["yt_chapters"]
        )

    # 根据笔记模式选择 prompt
    is_detailed = getattr(cfg, "note_mode", "essence") == "detailed"
    sys_prompt = DETAILED_SYSTEM if is_detailed else SYSTEM
    schema = DETAILED_SCHEMA_HINT if is_detailed else SCHEMA_HINT

    user = (
        f"【元数据】\n{meta_brief}{hint}\n\n"
        f"【转写文本（第 {idx}/{total} 块）】\n{chunk}\n\n"
        f"{schema}\n\n只输出 JSON。"
    )
    log.info("analyze", f"分析第 {idx}/{total} 块（{len(chunk)} 字符，{'细致' if is_detailed else '精华'}模式）…")
    return llm.chat_json(
        cfg,
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        temperature=0.3, max_tokens=8192, timeout=600,
    )


def _merge(chunks: list[dict]) -> dict:
    """合并多块分析结果：拼接 chapters，去重跨块重叠章节。"""
    if len(chunks) == 1:
        return chunks[0]
    merged = dict(chunks[0])
    all_ch = list(chunks[0].get("chapters", []))
    for ck in chunks[1:]:
        for ch in ck.get("chapters", []):
            # 去重：若与上一章时间高度重叠则跳过
            if all_ch and ch.get("t_start") is not None and all_ch[-1].get("t_end") is not None:
                if ch["t_start"] < all_ch[-1]["t_end"] - 1:
                    continue
            all_ch.append(ch)
    merged["chapters"] = all_ch
    return merged


def _validate(result: dict, audio_dur: float | None) -> list[str]:
    warns = []
    chs = result.get("chapters", [])
    if not chs:
        warns.append("未生成任何章节")
        return warns
    for i, ch in enumerate(chs, 1):
        if ch.get("svg_type") not in SVG_TYPES:
            warns.append(f"第{i}章 svg_type 非法：{ch.get('svg_type')}，回退 concept")
            ch["svg_type"] = "concept"
        for f in ("t_start", "t_end"):
            if ch.get(f) is None:
                warns.append(f"第{i}章缺 {f}")
        if ch.get("t_start") is not None and ch.get("t_end") is not None:
            if ch["t_end"] <= ch["t_start"]:
                warns.append(f"第{i}章时间区间非法")
        # 强制字段存在
        for f in ("questions", "traps", "steps", "key_points", "key_quotes", "visual_anchors"):
            ch.setdefault(f, [])
        ch.setdefault("conclusion", "")
        ch.setdefault("svg_rationale", "")
        # 细致模式额外字段
        ch.setdefault("detail_sections", [])
        ch.setdefault("frame_moments", [])
        ch["id"] = i
    if audio_dur and chs[-1].get("t_end") is not None:
        if chs[-1]["t_end"] < audio_dur * 0.9:
            warns.append(f"末章结束({chs[-1]['t_end']:.0f}s) 远小于音频时长({audio_dur:.0f}s)，可能漏章")
    return warns


def _analyze_chunks_parallel(cfg: Config, meta: dict, chunks: list[str], work_dir: Path) -> list[dict]:
    """并行分析所有分块，支持断点恢复。
    
    - 使用 ThreadPoolExecutor 并行调用 _analyze_chunk
    - 每块分析完成后保存到 work_dir/checkpoints/chunk_NNN.json
    - 启动时先加载已有 checkpoint，跳过已完成的块
    - 并行度由 cfg.analyze_max_workers 控制
    - 返回结果列表，顺序与 chunks 一致
    """
    total = len(chunks)
    ckpt_dir = work_dir / "checkpoints"
    use_ckpt = getattr(cfg, "checkpoint_enabled", True)
    if use_ckpt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict | None] = [None] * total

    # 加载已有 checkpoint，跳过已完成的块
    if use_ckpt:
        for i in range(total):
            ckpt_path = ckpt_dir / f"chunk_{i + 1:03d}.json"
            if ckpt_path.exists():
                try:
                    results[i] = read_json(ckpt_path)
                    log.info("analyze", f"跳过已完成的块 {i + 1}/{total}（命中 checkpoint）")
                except Exception:
                    pass

    # 收集待处理的块
    pending = [(i + 1, chunks[i]) for i in range(total) if results[i] is None]

    def _worker(idx: int, chunk: str) -> tuple[int, dict]:
        res = _analyze_chunk(cfg, meta, chunk, idx, total)
        if use_ckpt:
            write_json(ckpt_dir / f"chunk_{idx:03d}.json", res)
        return idx, res

    if pending:
        workers = max(1, min(cfg.analyze_max_workers, len(pending)))
        log.info("analyze", f"并行分析 {len(pending)}/{total} 块（{workers} 线程）…")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, idx, chunk) for idx, chunk in pending]
            for fut in as_completed(futures):
                idx, res = fut.result()
                results[idx - 1] = res

    return [r for r in results if r is not None]


def _hierarchical_analyze(cfg: Config, meta: dict, chunks: list[str], work_dir: Path) -> dict:
    """超长视频的分级摘要分析。
    
    Phase 1 - 摘要：对每个 chunk 调用 LLM 压缩为 ~6000 字符的结构化摘要
    Phase 2 - 分析：将所有摘要拼接后，用标准的 _analyze_chunk 进行章节拆分
    
    摘要 prompt 要求：保留时间戳信息、关键论点、步骤、结论，去掉重复口语内容。
    """
    total = len(chunks)
    target = getattr(cfg, "chunk_summary_target", 6000)

    # Phase 1 - 摘要：逐块压缩
    log.info("analyze", f"分级摘要 Phase 1：压缩 {total} 个分块…")
    summaries: list[str] = []
    for i, c in enumerate(chunks, 1):
        user = (
            f"【元数据】标题：{meta.get('title', '')}\n"
            f"【转写文本（第 {i}/{total} 块，需压缩）】\n{c}\n\n"
            f"请压缩为不超过 {target} 字符的结构化摘要。"
        )
        log.info("analyze", f"压缩第 {i}/{total} 块（{len(c)} 字符）…")
        summary = llm.chat(
            cfg,
            [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.2, max_tokens=8192, timeout=600,
        )
        summaries.append(summary)

    # Phase 2 - 分析：拼接摘要后做标准章节拆分
    merged_text = "\n\n".join(summaries)
    log.info("analyze", f"分级摘要 Phase 2：分析拼接后的摘要（{len(merged_text)} 字符）…")
    result = _analyze_chunk(cfg, meta, merged_text, 1, 1)
    return result


def _refine_merge(cfg: Config, merged: dict, meta: dict) -> dict:
    """对合并后的章节做二次精炼。
    
    - 检测时间重叠的相邻章节
    - 合并内容重复的章节
    - 重新编号 chapter id
    - 如果章节数 > 15，调用 LLM 做一次全局精简（合并相似章节）
    
    注意：这个函数不要过度调用 LLM，只在章节明显过多(>15)或有大量重叠时才调用 LLM。
    简单的时间重叠去重用代码逻辑即可。
    """
    result = dict(merged)
    chapters = list(result.get("chapters", []))
    if not chapters:
        return result

    # 1. 检测并合并时间重叠的相邻章节
    cleaned: list[dict] = []
    for ch in chapters:
        if cleaned:
            prev = cleaned[-1]
            if (ch.get("t_start") is not None and prev.get("t_end") is not None
                    and ch["t_start"] < prev["t_end"] - 1):
                # 时间高度重叠，合并到上一章：延长区间、汇总内容
                prev["t_end"] = max(prev.get("t_end", 0) or 0, ch.get("t_end", 0) or 0)
                for f in ("key_points", "questions", "traps", "steps", "visual_anchors"):
                    prev.setdefault(f, [])
                    prev[f] = list(prev[f]) + list(ch.get(f, []))
                if ch.get("conclusion") and ch["conclusion"] not in (prev.get("conclusion") or ""):
                    prev["conclusion"] = ((prev.get("conclusion") or "") + " " + ch["conclusion"]).strip()
                continue
        cleaned.append(ch)

    # 2. 重新编号 chapter id
    for i, ch in enumerate(cleaned, 1):
        ch["id"] = i
    result["chapters"] = cleaned

    # 3. 章节明显过多时调用 LLM 做一次全局精简
    if len(cleaned) > 15:
        log.info("analyze", f"章节数过多({len(cleaned)} > 15)，调用 LLM 全局精简…")
        compact_system = (
            "你是内容编辑专家。把过多、相似的章节合并精简，输出与输入相同结构的 JSON。"
            "要求：保留真实时间戳，合并同类章节，控制在 8~12 章，不要丢失关键内容。严格输出 JSON。"
        )
        chapters_brief = json.dumps(cleaned, ensure_ascii=False)
        user = (
            f"【视频标题】{meta.get('title', '')}\n"
            f"【当前章节（{len(cleaned)} 章，需精简到 8~12 章）】\n{chapters_brief}\n\n"
            f"{SCHEMA_HINT}\n\n只输出 JSON。"
        )
        try:
            refined = llm.chat_json(
                cfg,
                [{"role": "system", "content": compact_system}, {"role": "user", "content": user}],
                temperature=0.2, max_tokens=8192, timeout=600,
            )
            if refined.get("chapters"):
                result = refined
        except Exception as e:
            log.warn("analyze", f"LLM 全局精简失败，保留代码合并结果：{e}")

    return result


def analyze(cfg: Config, meta: dict, transcript: dict, work_dir: Path) -> dict[str, Any]:
    segments = transcript.get("segments", [])
    if not segments:
        raise RuntimeError("转写为空，无法分析")
    text = _build_lines(segments)
    chunks = _chunks(text)
    total = len(chunks)
    tail = transcript.get("tail_check") or {}
    audio_dur = tail.get("audio_dur") or meta.get("duration", 0) or 0

    log.info("analyze", f"转写 {len(segments)} 段 / {len(text)} 字符，分 {total} 块，时长 {audio_dur:.0f}s")

    # 判断是否需要分级摘要
    use_hierarchical = (
        getattr(cfg, "hierarchical_summary", True)
        and audio_dur > cfg.max_video_duration
        and total > 1
    )

    if use_hierarchical:
        log.info("analyze", f"超长视频({audio_dur:.0f}s > {cfg.max_video_duration}s)，启用分级摘要")
        result = _hierarchical_analyze(cfg, meta, chunks, work_dir)
    elif total > 1 and getattr(cfg, "chunk_analyze_parallel", True):
        # 并行分块分析 + 断点恢复
        parts = _analyze_chunks_parallel(cfg, meta, chunks, work_dir)
        result = _merge(parts)
        # 合并质量提升
        if len(parts) > 1:
            result = _refine_merge(cfg, result, meta)
    else:
        # 串行分析（兜底）
        parts = [_analyze_chunk(cfg, meta, c, i, total) for i, c in enumerate(chunks, 1)]
        result = _merge(parts)
        if len(parts) > 1:
            result = _refine_merge(cfg, result, meta)

    # LLM 有时会把 21:31 写成 21.31；先修正再验证。
    _repair_mmss_decimal_times(result, audio_dur)

    # 验证
    warns = _validate(result, audio_dur)
    for w in warns:
        log.warn("analyze", w)
    result["warnings"] = warns

    write_json(work_dir / "analysis.json", result)
    log.info("analyze", f"完成：{len(result.get('chapters', []))} 章，is_ui_demo={result.get('is_ui_demo')}")
    return result
