"""诊断 SVG 生成：直接调 LLM 看返回内容。"""
import sys, os, json
os.environ["PYTHONPATH"] = "D:\\python_libs"
os.environ["HF_HOME"] = "D:\\vnotes_models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PATH"] = "D:\\python_libs\\nvidia\\cublas\\bin;D:\\python_libs\\nvidia\\nvrtc\\bin;" + os.environ.get("PATH", "")
sys.path.insert(0, "D:\\python_libs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vnotes import Config
from vnotes import llm
from vnotes import svg as S

cfg = Config.load()

# 用第一章测试
ch = {
    "id": 1,
    "title": "引言：香港动作电影的黄金时代",
    "t_start": 0, "t_end": 59,
    "svg_type": "timeline",
    "svg_rationale": "展示香港电影从60年代到90年代的发展时间线",
    "key_points": [
        "1961年清水湾片场启用标志香港电影进入大制片厂时代",
        "当时美国新好莱坞运动未开始，日本动漫和韩国偶像产业尚未崛起",
        "香港电影通过体系服务创作的混合生态，实现了工业化与创作性并存"
    ],
    "steps": [],
    "questions": ["香港电影对亚洲乃至全世界意味着什么？"],
    "traps": ["误以为现代电影历史完全由好莱坞书写"],
    "conclusion": "香港电影曾是全球动作电影最严厉的父亲",
    "key_quotes": [{"t": 46, "text": "其实到今天很多年轻的观众已经很难真正理解七八十年代的香港电影"}],
    "visual_anchors": ["1961年清水湾片场启用画面"],
}

print("=== 测试 LLM 返回 ===")
try:
    raw = llm.chat(
        cfg,
        [{"role": "system", "content": S.SYSTEM}, {"role": "user", "content": S._gen_one.__code__.co_consts[3] if len(S._gen_one.__code__.co_consts) > 3 else "test"}],
        temperature=0.4, max_tokens=2048, timeout=180,
    )
    print(f"返回长度: {len(raw)}")
    print(f"前500字符:\n{raw[:500]}")
    print(f"\n后200字符:\n{raw[-200:]}")
    print(f"\n含 <svg: {'<svg' in raw.lower()}")
    print(f"含 </svg>: {'</svg>' in raw.lower()}")
    
    sanitized = S._sanitize(raw)
    print(f"\nsanitize 后长度: {len(sanitized)}")
    print(f"sanitize 后前200字符:\n{sanitized[:200]}")
except Exception as e:
    print(f"失败: {e}")

# 也测试完整的 _gen_one
print("\n=== 测试 _gen_one ===")
try:
    result = S._gen_one(cfg, ch)
    print(f"结果长度: {len(result)}")
    print(f"含 <svg: {'<svg' in result.lower()}")
    print(f"前300字符:\n{result[:300]}")
except Exception as e:
    print(f"_gen_one 失败: {e}")
