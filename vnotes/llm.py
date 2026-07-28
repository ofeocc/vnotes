"""LLM 客户端：OpenAI 兼容（DeepSeek / 通义 / OpenAI / 本地 vLLM）。"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from .config import Config
from .util import log


class LLMError(RuntimeError):
    pass


def _post(cfg: Config, payload: dict, timeout: int = 180) -> dict:
    url = cfg.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.llm_api_key}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(1, 4):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                if resp.status_code in (429, 500, 502, 503):
                    time.sleep(3 * attempt)
                    continue
                raise LLMError(last_err)
            return resp.json()
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(3 * attempt)
    raise LLMError(f"LLM 请求失败（重试3次）：{last_err}")


def chat(cfg: Config, messages: list[dict], *, json_mode: bool = False,
         temperature: float = 0.4, max_tokens: int = 4096, timeout: int = 300) -> str:
    payload: dict[str, Any] = {
        "model": cfg.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = _post(cfg, payload, timeout=timeout)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"LLM 返回结构异常：{data}") from e


def chat_json(cfg: Config, messages: list[dict], **kw) -> dict:
    """调用并解析 JSON，兼容代码块包裹与前置文本。"""
    raw = chat(cfg, messages, json_mode=True, **kw)
    return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    # 去代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if m:
        raw = m.group(1)
    # 取第一个 {...}
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        raw = raw[s:e + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"JSON 解析失败：{e}\n原始：{raw[:800]}")
