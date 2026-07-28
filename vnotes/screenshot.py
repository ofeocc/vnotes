"""整页截图：优先 Playwright（独立 Chromium，不冲突），兜底 Chrome/Edge --screenshot。

Playwright 优势：自带独立 Chromium 二进制，不依赖系统 Chrome/Edge，不会因
已有浏览器实例运行而挂起。full_page=True 直接截整页，无需手动测量高度。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .config import Config
from .util import log, file_uri

# ---- Chrome/Edge 兜底候选 ----
_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    str(Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe"),
]
_EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _find_system_browsers() -> list[str]:
    found = []
    for p in _CHROME_CANDIDATES + _EDGE_CANDIDATES:
        if Path(p).exists() and p not in found:
            found.append(p)
    return found


# ---- Playwright 截图 ----
def _playwright_screenshot(html: Path, out_png: Path, width: int = 1280) -> dict | None:
    """用 Playwright 截整页。成功返回 info dict，失败返回 None。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warn("shot", "Playwright 未安装，跳过")
        return None

    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_png.exists():
        out_png.unlink()

    abs_html = html.resolve()
    abs_out = out_png.resolve()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-extensions",
                      "--disable-dev-shm-usage", "--force-device-scale-factor=1"],
            )
            page = browser.new_page(
                viewport={"width": width, "height": 800},
                device_scale_factor=1,
            )
            page.goto(abs_html.as_uri(), wait_until="networkidle", timeout=30000)
            # 等待字体/图片渲染
            page.evaluate(
                """async () => {
                    if (document.fonts && document.fonts.ready) {
                        try { await document.fonts.ready; } catch (e) {}
                    }
                    const imgs = Array.from(document.images || []);
                    imgs.forEach(img => {
                        img.loading = 'eager';
                        img.decoding = 'sync';
                    });
                    await Promise.all(imgs.map(img => {
                        if (img.complete && img.naturalWidth > 0) return Promise.resolve();
                        return new Promise(resolve => {
                            const done = () => resolve();
                            img.addEventListener('load', done, { once: true });
                            img.addEventListener('error', done, { once: true });
                            setTimeout(done, 4000);
                        });
                    }));
                }"""
            )
            page.wait_for_timeout(500)

            # 测量真实高度
            measured = page.evaluate(
                "() => Math.ceil(document.documentElement.scrollHeight)"
            )
            log.info("shot", f"Playwright 测得整页高度 = {measured}px")

            # 截整页
            page.screenshot(path=str(abs_out), full_page=True, type="png")
            browser.close()

        if not abs_out.exists() or abs_out.stat().st_size < 1000:
            log.warn("shot", "Playwright 截图文件无效")
            return None

        # 用 PIL 重新保存为标准 RGB PNG（Playwright 可能输出带 alpha/色彩管理的 PNG，
        # 某些 ffmpeg 版本无法解码）
        tmp_out = abs_out.with_suffix(".tmp.png")
        with Image.open(abs_out) as im:
            w, h = im.size
            im.convert("RGB").save(tmp_out, format="PNG", optimize=True)
        shutil.move(str(tmp_out), str(abs_out))
        truncated = bool(measured and h < measured * 0.9)
        if truncated:
            log.warn("shot", f"截图高度 {h}px < 测得 {measured}px，可能被截断")
        log.info("shot", f"Playwright 截图完成：{w}x{h}")
        return {
            "path": str(abs_out), "width": w, "height": h,
            "method": "playwright", "browser": "chromium",
            "measured": measured, "truncated": truncated,
        }
    except Exception as e:
        log.warn("shot", f"Playwright 截图失败：{e}")
        return None


# ---- Chrome/Edge --screenshot 兜底 ----
def _user_data_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="vnotes_headless_"))


def _run_headless(browser: str, args: list[str], timeout: int = 60) -> tuple[int, str, str]:
    cmd = [browser, *args]
    log.debug("shot", "$ " + " ".join(cmd))
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return cp.returncode, cp.stdout or "", cp.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -2, "", str(e)


def _measure_height_raw(browser: str, html: Path) -> int | None:
    """用 --dump-dom + 注入 JS 测量高度。"""
    measure_script = (
        "<script>window.addEventListener('load',function(){"
        "setTimeout(function(){document.title='VNOTES_H:'+"
        "Math.ceil(document.documentElement.scrollHeight)},300);});</script>"
    )
    tmpdir = Path(tempfile.mkdtemp(prefix="vnotes_meas_"))
    udd = _user_data_dir()
    meas = tmpdir / "measure.html"
    txt = html.read_text(encoding="utf-8")
    meas.write_text(txt.replace("</body>", measure_script + "</body>", 1), encoding="utf-8")
    try:
        _, stdout, _ = _run_headless(browser, [
            "--headless=new", "--disable-gpu", "--no-sandbox",
            "--no-first-run", "--no-default-browser-check", "--disable-extensions",
            f"--user-data-dir={udd}", "--remote-debugging-port=0",
            "--virtual-time-budget=5000", "--dump-dom", file_uri(meas),
        ], timeout=45)
        m = re.search(r"VNOTES_H:(\d+)", stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(udd, ignore_errors=True)
    return None


def _try_raw_screenshot(browser: str, mode: str, common: list[str],
                        win: str, out_png: Path, uri: str) -> bool:
    if out_png.exists():
        out_png.unlink()
    _, _, _ = _run_headless(browser, [
        f"--headless={mode}", *common, f"--window-size={win}",
        f"--screenshot={out_png}", uri,
    ], timeout=60)
    ok = out_png.exists() and out_png.stat().st_size > 1000
    if ok:
        log.info("shot", f"{mode} 截图成功 ({Path(browser).name})")
    return ok


def _raw_browser_screenshot(html: Path, out_png: Path) -> dict | None:
    """Chrome/Edge --screenshot 兜底方案。"""
    browsers = _find_system_browsers()
    if not browsers:
        return None

    uri = file_uri(html)
    measured = _measure_height_raw(browsers[0], html)
    h = min(measured or 8000, 16000)

    for browser in browsers:
        udd = _user_data_dir()
        common = ["--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                  "--force-device-scale-factor=1", "--virtual-time-budget=8000",
                  "--default-background-color=FFFFFFFF", "--no-first-run",
                  "--no-default-browser-check", "--disable-extensions",
                  "--disable-session-crashed-bubble", "--disable-restore-tabs",
                  f"--user-data-dir={udd}", "--remote-debugging-port=0"]
        for mode in ("old", "new"):
            win = f"1280,{measured or 100000}" if mode == "old" else f"1280,{h}"
            if _try_raw_screenshot(browser, mode, common, win, out_png, uri):
                with Image.open(out_png) as im:
                    w, h2 = im.size
                truncated = bool(measured and h2 < measured * 0.9)
                return {
                    "path": str(out_png), "width": w, "height": h2,
                    "method": mode, "browser": Path(browser).name,
                    "measured": measured, "truncated": truncated,
                }
        shutil.rmtree(udd, ignore_errors=True)
    return None


# ---- 主入口 ----
def screenshot(cfg: Config, html: Path, out_png: Path) -> dict:
    """截整页 PNG。优先 Playwright，兜底 Chrome/Edge --screenshot。"""
    # 1) Playwright（主路径）
    info = _playwright_screenshot(html, out_png)
    if info:
        return info

    log.warn("shot", "Playwright 不可用，回退到系统 Chrome/Edge --screenshot")

    # 2) Chrome/Edge --screenshot（兜底）
    info = _raw_browser_screenshot(html, out_png)
    if info:
        return info

    raise RuntimeError("截图失败：Playwright 和系统浏览器均不可用")
