"""手动下载 faster-whisper 模型文件，绕过 xet 协议。"""
import sys
import os
import time
import zipfile
import requests

MB = 1024 * 1024
MIRROR = "https://hf-mirror.com"

# 模型仓库 -> 所需文件列表
SYSTRAN_FILES = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
]

MODELS = {
    "tiny": ("Systran/faster-whisper-tiny", SYSTRAN_FILES),
    "tiny.en": ("Systran/faster-whisper-tiny.en", SYSTRAN_FILES),
    "base": ("Systran/faster-whisper-base", SYSTRAN_FILES),
    "base.en": ("Systran/faster-whisper-base.en", SYSTRAN_FILES),
    "small": ("Systran/faster-whisper-small", SYSTRAN_FILES),
    "small.en": ("Systran/faster-whisper-small.en", SYSTRAN_FILES),
    "large-v3-turbo": ("Systran/faster-whisper-large-v3-turbo", SYSTRAN_FILES),
    "turbo": ("Systran/faster-whisper-large-v3-turbo", SYSTRAN_FILES),
}

VOSK_MODELS = {
    "vosk-cn": (
        "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
        "vosk-model-small-cn-0.22",
    ),
}


def download_file(repo: str, filename: str, dest: str, retries: int = 5):
    """从 hf-mirror 下载单个文件，支持断点续传。"""
    url = f"{MIRROR}/{repo}/resolve/main/{filename}"
    tmp = dest + ".tmp"

    # 已下载的大小
    existing = 0
    if os.path.exists(tmp):
        existing = os.path.getsize(tmp)

    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    for attempt in range(retries):
        try:
            print(f"  下载 {filename} (已有 {existing/MB:.1f}MB, 尝试 {attempt+1}/{retries})...")
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) + existing

            mode = "ab" if existing > 0 else "wb"
            with open(tmp, mode) as f:
                downloaded = existing
                last_print = time.time()
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_print > 5:
                            pct = downloaded / total * 100 if total else 0
                            print(f"    {downloaded/MB:.1f}/{total/MB:.1f}MB ({pct:.0f}%)")
                            last_print = time.time()

            os.rename(tmp, dest)
            print(f"  OK {filename} 完成 ({os.path.getsize(dest)/MB:.1f}MB)")
            return True

        except Exception as e:
            print(f"  FAIL: {e}")
            if os.path.exists(tmp):
                existing = os.path.getsize(tmp)
            time.sleep(3)

    return False


def download_url(url: str, dest: str, retries: int = 5):
    """下载普通 URL 文件，支持断点续传。"""
    tmp = dest + ".tmp"
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
    for attempt in range(retries):
        try:
            print(f"  下载 {os.path.basename(dest)} (已有 {existing/MB:.1f}MB, 尝试 {attempt+1}/{retries})...")
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) + existing
            mode = "ab" if existing > 0 else "wb"
            with open(tmp, mode) as f:
                downloaded = existing
                last_print = time.time()
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_print > 5:
                            pct = downloaded / total * 100 if total else 0
                            print(f"    {downloaded/MB:.1f}/{total/MB:.1f}MB ({pct:.0f}%)")
                            last_print = time.time()
            os.rename(tmp, dest)
            print(f"  OK {os.path.basename(dest)} 完成 ({os.path.getsize(dest)/MB:.1f}MB)")
            return True
        except Exception as e:
            print(f"  FAIL: {e}")
            existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            time.sleep(3)
    return False


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "small"
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "D:/vnotes_models"

    if model_name in VOSK_MODELS:
        url, folder = VOSK_MODELS[model_name]
        os.makedirs(model_dir, exist_ok=True)
        dest_dir = os.path.join(model_dir, folder)
        if os.path.exists(os.path.join(dest_dir, "conf")) and os.path.exists(os.path.join(dest_dir, "am")):
            print(f"OK Vosk 模型已存在：{dest_dir}")
            return 0
        zip_path = os.path.join(model_dir, folder + ".zip")
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            if not download_url(url, zip_path):
                print("FAIL Vosk 模型下载失败")
                return 1
        print("  解压 Vosk 模型...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_dir)
        print(f"\nOK Vosk 模型下载完成：{dest_dir}")
        return 0

    if model_name not in MODELS:
        print(f"未知模型: {model_name}，可选: {list(MODELS.keys()) + list(VOSK_MODELS.keys())}")
        return 1

    repo, files = MODELS[model_name]
    dest_dir = os.path.join(model_dir, "models--" + repo.replace("/", "--"), "snapshots", "main")
    os.makedirs(dest_dir, exist_ok=True)

    # 创建 refs/main
    refs_dir = os.path.join(model_dir, "models--" + repo.replace("/", "--"), "refs")
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, "main"), "w") as f:
        f.write("main")

    print(f"下载模型: {repo}")
    print(f"目标目录: {dest_dir}")

    for fn in files:
        dest = os.path.join(dest_dir, fn)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  OK {fn} 已存在 ({os.path.getsize(dest)/MB:.1f}MB)")
            continue
        if not download_file(repo, fn, dest):
            print(f"  FAIL {fn} 下载失败！")
            return 1

    print(f"\nOK 模型 {model_name} 下载完成！")
    print(f"  路径: {dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
