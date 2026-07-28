"""切片：把整页 PNG 切成竖向切片，保留约 100px 重叠，在空白处下刀避免切断标题。

用 PIL crop（不依赖 ffmpeg，因为 TRAE 自带 ffmpeg 是精简版，不支持 PNG/crop filter）。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .config import Config
from .util import log
from .qa import compute_ink_profile

GAP_THRESHOLD = 0.012  # 视为空白行的墨量阈值
MIN_HEIGHT = 520       # 单片最小高度，保证推进


def _best_gap(rows: list[float], lo: int, hi: int, target: int) -> int:
    """在 [lo,hi) 内找最佳下刀行：优先空白行(最接近 target)，否则取局部墨量最小处。"""
    lo = max(0, lo)
    hi = min(len(rows), hi)
    blanks = [i for i in range(lo, hi) if rows[i] < GAP_THRESHOLD]
    if blanks:
        return min(blanks, key=lambda i: abs(i - target))
    # 无空白：在 [lo, target] 区间取墨量最小行（往后少切，避免切到下章标题）
    end = max(lo + 1, min(target, hi))
    return min(range(lo, end), key=lambda i: rows[i])


def slice_image(cfg: Config, png_info: dict, qa_result: dict, slices_dir: Path) -> list[Path]:
    full = Path(png_info["path"])
    W = png_info["width"]
    H = png_info["height"]
    rows = compute_ink_profile(full)[2]  # 复算墨量
    target_h = cfg.slice_height
    overlap = cfg.slice_overlap
    window = 240  # 在目标±240px 范围找 gap

    slices_dir.mkdir(parents=True, exist_ok=True)
    # 清理旧切片
    for old in slices_dir.glob("slice_*.png"):
        old.unlink()

    slices: list[Path] = []
    with Image.open(full) as im:
        y = 0
        prev_cut = -1
        idx = 1
        while y < H:
            target = y + target_h
            if target >= H:
                h = H - y
                if h >= 120:  # 太小的尾片并入上一片
                    slices.append(_crop_pil(im, slices_dir, idx, W, y, h))
                    idx += 1
                break
            lo = max(y + MIN_HEIGHT, target - window)
            hi = min(H, target + window)
            cut = _best_gap(rows, lo, hi, target)
            h = cut - y
            if h < MIN_HEIGHT:
                cut = y + target_h
                h = cut - y
            slices.append(_crop_pil(im, slices_dir, idx, W, y, h))
            idx += 1
            nxt = max(cut - overlap, prev_cut + MIN_HEIGHT + 1)
            if nxt <= y:
                nxt = y + max(MIN_HEIGHT, h - overlap)
            prev_cut = cut
            y = nxt

    log.info("crop", f"切出 {len(slices)} 片，目标高 {target_h}px / 重叠 {overlap}px")
    return slices


def _crop_pil(im: Image.Image, out_dir: Path, idx: int,
              W: int, y: int, h: int) -> Path:
    """用 PIL crop 切片。"""
    out = out_dir / f"slice_{idx:02d}.png"
    cropped = im.crop((0, y, W, y + h))
    cropped.save(out, format="PNG", optimize=True)
    return out
