"""元数据抓取：yt-dlp + cookies，B站保留标题/UP/分P/简介/标签/封面/时长。"""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from .config import Config
from .util import log, safe_name, write_json, run_ytdlp


_URL_RE = re.compile(r"https?://[^\s'\"<>\u3000\u4e00-\u9fff`]+", re.I)
_BARE_VIDEO_DOMAIN_RE = re.compile(
    r"(?<![\w@])((?:(?:www\.)?bilibili\.com|b23\.tv|(?:www\.)?youtube\.com|youtu\.be)/[^\s'\"<>\u3000\u4e00-\u9fff`]+)",
    re.I,
)
_BVID_RE = re.compile(r"\b(BV[0-9A-Za-z]{10})\b", re.I)
_AVID_RE = re.compile(r"\b(av\d{3,})\b", re.I)
_TRAILING_URL_CHARS = ".,;:!?`'\")]}，。；：！？、”’》】）"


def _clean_extracted_url(url: str) -> str:
    url = url.strip().strip("<>'\"`")
    while url and url[-1] in _TRAILING_URL_CHARS:
        url = url[:-1]
    return url


def _validate_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return url
    raise ValueError("请输入有效的视频链接，或粘贴包含 B 站 BV 号/短链的分享文本。")


def normalize_video_url(raw: str) -> str:
    """Extract a playable video URL from pasted share text or a bare BV/av id."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("请输入视频链接。")

    m = _URL_RE.search(text)
    if m:
        return _validate_http_url(_clean_extracted_url(m.group(0)))

    m = _BARE_VIDEO_DOMAIN_RE.search(text)
    if m:
        return _validate_http_url("https://" + _clean_extracted_url(m.group(1)))

    m = _BVID_RE.search(text)
    if m:
        url = f"https://www.bilibili.com/video/{m.group(1)}"
        p = re.search(r"[?&]p=(\d+)\b", text, re.I)
        if p:
            url = _build_part_url(url, int(p.group(1)))
        return url

    m = _AVID_RE.search(text)
    if m:
        url = f"https://www.bilibili.com/video/{m.group(1)}"
        p = re.search(r"[?&]p=(\d+)\b", text, re.I)
        if p:
            url = _build_part_url(url, int(p.group(1)))
        return url

    return _validate_http_url(text)


def _extract_url(raw: str) -> str:
    """Backward-compatible alias used by the pipeline."""
    return normalize_video_url(raw)


def _split_part(url: str) -> tuple[str, int | None]:
    """从 URL 解析出基础 URL 与分 P 号。"""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    values = qs.pop("p", [])
    part = None
    if values:
        try:
            part = int(values[0])
        except (TypeError, ValueError):
            part = None
    query = urlencode(qs, doseq=True)
    base = urlunparse(parsed._replace(query=query, fragment=""))
    return base, part


def _build_part_url(base_url: str, part: int) -> str:
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs["p"] = [str(part)]
    query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=query, fragment=""))


def _run_ytdlp(cfg: Config, args: list[str], *, tag: str) -> list[dict]:
    cp = run_ytdlp(
        cfg,
        ["--socket-timeout", "15", "--retries", "2", "--extractor-retries", "2", "--no-warnings", *args],
        tag=tag,
        check=True,
        timeout=90,
    )
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _fetch_parts(cfg: Config, base_url: str) -> list[dict]:
    """获取分 P 列表（B站多 P；单 P 视频返回单元素）。"""
    try:
        entries = _run_ytdlp(cfg, ["--flat-playlist", "--dump-json", base_url], tag="meta/parts")
    except Exception as e:
        log.warn("meta/parts", f"分P列表抓取失败，按单P处理：{e}")
        return []
    parts = []
    for i, e in enumerate(entries, 1):
        parts.append({
            "index": i,
            "title": e.get("title") or f"P{i}",
            "duration": e.get("duration"),
            "url": e.get("url") or e.get("webpage_url") or "",
            "id": e.get("id"),
        })
    return parts


def _download_cover(cfg: Config, thumb_url: str, dest: Path) -> str | None:
    if not thumb_url:
        return None
    try:
        import requests
        r = requests.get(thumb_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        return str(dest.name)
    except Exception as e:
        log.warn("meta/cover", f"封面下载失败：{e}")
        return None


def fetch_metadata(cfg: Config, url: str, part: int | None = None,
                   work_dir: Path | None = None) -> dict[str, Any]:
    """抓取视频完整元数据。返回标准化的 meta dict。"""
    # 从用户输入中提取纯 URL（可能带标题文字）
    url = _extract_url(url)
    base_url, url_part = _split_part(url)
    chosen_part = part or url_part or 1

    # 平台检测
    is_bili = "bilibili.com" in base_url or "b23.tv" in base_url
    is_youtube = any(d in base_url for d in ("youtube.com", "youtu.be", "youtube-nocookie.com"))

    main_url = _build_part_url(base_url, chosen_part) if is_bili else url
    log.info("meta", f"抓取元数据：{main_url}")

    main = _run_ytdlp(cfg, ["--dump-json", "--no-playlist", main_url], tag="meta/main")
    if not main:
        raise RuntimeError("yt-dlp 未返回元数据")
    info = main[0]
    canonical_url = info.get("webpage_url") or main_url
    if is_bili and "bilibili.com" in canonical_url:
        canonical_base, canonical_part = _split_part(canonical_url)
        base_url = canonical_base
        if part is None and url_part is None and canonical_part:
            chosen_part = canonical_part

    # 分 P / 播放列表（B站多P 或 YouTube 播放列表）
    parts: list[dict] = []
    if is_bili:
        parts = _fetch_parts(cfg, base_url)
    elif is_youtube and ("list=" in base_url or "playlist" in base_url):
        parts = _fetch_parts(cfg, base_url)
    if not parts:
        parts = [{
            "index": chosen_part,
            "title": info.get("title", ""),
            "duration": info.get("duration"),
            "url": info.get("webpage_url", main_url),
            "id": info.get("id"),
        }]

    # 封面下载到工作目录
    cover_name = None
    if work_dir:
        work_dir.mkdir(parents=True, exist_ok=True)
        thumb = info.get("thumbnail") or ""
        cover_name = _download_cover(cfg, thumb, work_dir / "cover.jpg")

    # yt-dlp 可能已抽取章节标记（B站 UP 主分段）
    yt_chapters = []
    for c in info.get("chapters") or []:
        yt_chapters.append({
            "t_start": round(float(c.get("start_time", 0)), 2),
            "t_end": round(float(c.get("end_time", 0)), 2),
            "title": c.get("title", ""),
        })

    meta = {
        "title": info.get("title", ""),
        "uploader": info.get("uploader") or info.get("channel") or info.get("uploader_id", ""),
        "uploader_id": info.get("uploader_id", ""),
        "description": info.get("description", "") or "",
        "tags": info.get("tags", []) or [],
        "thumbnail_url": info.get("thumbnail", ""),
        "cover": cover_name,
        "duration": float(info["duration"]) if info.get("duration") else None,
        "webpage_url": info.get("webpage_url", main_url),
        "base_url": base_url,
        "video_id": info.get("id", ""),
        "extractor": info.get("extractor_key") or info.get("extractor", ""),
        "upload_date": info.get("upload_date", ""),
        "is_bili": is_bili,
        "is_youtube": is_youtube,
        "selected_part": chosen_part,
        "parts": parts,
        "yt_chapters": yt_chapters,
    }
    if work_dir:
        write_json(work_dir / "meta.json", meta)
    log.info("meta", f"标题={meta['title']!r} UP={meta['uploader']!r} 分P={len(parts)} 时长={meta['duration']}s")
    return meta
