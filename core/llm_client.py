from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Generator, Optional, Iterable

import requests
# =============================================================================
# api_requester.py 说明（面向未来的广泛模型调用 / OpenAI-compat 网关）
#
# 目标
# - 统一通过 OpenAI-Compatible 的 /v1/chat/completions 接口调用不同模型/供应商
# - 同时支持：非流式(chat) 与 流式(stream_chat, SSE/JSON-lines)
# - 支持：config.yaml 默认值 + 调用时参数覆盖（便于随时切换 key/model/网关）
# - 支持：extra 扩展字段，未来某些模型需要 tools/reasoning/max_tokens/... 时无需改函数签名
#
# ----------------------------
# 配置来源与优先级（同名字段）
#   函数显式传参 > config.yaml > DEFAULTS/硬编码 fallback
#
# Endpoint(url) 解析优先级：
#   1) stream_chat/chat 参数 url
#   2) config.yaml: chat_completions_url
#   3) config.yaml: base_url + "/chat/completions"
#   4) fallback: "https://cn.api.openai-next.com/v1/chat/completions"
#
# ----------------------------
# DEFAULTS（代码内默认值）
#   base_url: None
#   chat_completions_url: None
#   api_key: None                # 必须提供（配置或参数）
#   model_id: None               # 必须提供（配置或参数）
#   temperature: 0.7
#   timeout: 120                 # 秒
#   verify_ssl: True
#
# ----------------------------
# config.yaml 支持的键（简易 YAML：仅支持顶层 key: value，无嵌套/列表）
#   api_key: "sk-xxxx"                           # 必填（或调用时传参）
#   model_id: "gemini-2.5-pro-deepsearch-async"  # 必填（或调用时传参）
#   base_url: "http://152.53.52.170:3003/v1"     # 可选（二选一）
#   chat_completions_url: "http://.../chat/completions"  # 可选（优先于 base_url）
#   temperature: 0.2
#   timeout: 120
#   verify_ssl: true
#
# 示例 config.yaml：
#   base_url: "http://152.53.52.170:3003/v1"
#   api_key: "sk-xxxx"
#   model_id: "gemini-2.5-pro-deepsearch-async"
#   temperature: 0.2
#   timeout: 120
#   verify_ssl: true
#
# ----------------------------
# 公共函数
# 1) chat(prompt, ...) -> str
#    - 非流式：一次性返回完整文本
#    - 固定 payload["stream"]=False
#
# 2) stream_chat(prompt, ...) -> generator[str]
#    - 流式：逐段 yield 文本
#    - 兼容：
#       - 标准 SSE：行前缀 "data: {...}"，结束 "data: [DONE]"
#       - 某些网关：直接按行输出 JSON（无 data: 前缀）
#
# 3) make_messages(user, system=None) -> list[dict]
#    - 多轮对话扩展用的小工具（当前主流程未强制使用）
#
# ----------------------------
# 关键开关（函数参数）
#   prompt (必填)
#   api_key / model_id / url：不传则从 config.yaml 取
#   temperature / timeout / verify_ssl：不传则从 config.yaml -> DEFAULTS 取
#   system：默认 "You are a helpful assistant."
#   extra：dict，直接 merge 进 payload（可覆盖同名键），用于未来扩展：
#          例如 max_tokens/top_p/tools/response_format/reasoning/web_search 等
# =============================================================================


# ----------------------------
# Config
# ----------------------------
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "config.yaml"

DEFAULTS = {
    "base_url": None,  # e.g. "http://152.53.52.170:3003/v1"
    "chat_completions_url": None,  # e.g. "http://152.53.52.170:3003/v1/chat/completions"
    "api_key": None,
    "model_id": None,
    "temperature": 0.7,
    "timeout": 120,
    "verify_ssl": True,
}


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    """
    Minimal YAML loader for simple `key: value` pairs.
    (No nested structures; good enough for api_key/model_id/base_url.)
    """
    data: Dict[str, Any] = {}
    if not path or not path.exists():
        return data

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue

            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()

            # strip quotes
            if len(v) >= 2 and v[0] in ("'", '"') and v[-1] == v[0]:
                v = v[1:-1]

            # basic type casting
            if v.lower() in ("true", "false"):
                data[k] = v.lower() == "true"
                continue
            try:
                if "." in v:
                    data[k] = float(v)
                else:
                    data[k] = int(v)
                continue
            except Exception:
                pass

            data[k] = v

    return data


def _resolve_endpoint(config: Dict[str, Any], url: Optional[str] = None) -> str:
    """
    Resolve final chat/completions endpoint:
    priority: explicit url > config.chat_completions_url > config.base_url + /chat/completions
    """
    if url:
        return url

    cc = config.get("chat_completions_url")
    if cc:
        return cc

    base = config.get("base_url")
    if base:
        base = str(base).rstrip("/")
        return f"{base}/chat/completions"

    # fallback to your earlier default (kept for compatibility)
    return "https://cn.api.openai-next.com/v1/chat/completions"


def _iter_sse_lines(resp: requests.Response) -> Iterable[bytes]:
    """
    Iterate lines from a streaming response.
    Handles:
    - standard SSE: b"data: {...}"
    - some gateways: raw json line without "data: "
    """
    for line in resp.iter_lines():
        if not line:
            continue
        yield line


def _extract_delta_text(event_obj: Dict[str, Any]) -> str:
    """
    Try best-effort extraction of incremental text from multiple variants.
    OpenAI-style:
      choices[0].delta.content
    Some gateways:
      choices[0].message.content (in stream, less common)
    """
    try:
        if isinstance(event_obj, dict) and event_obj.get("text"):
            return str(event_obj.get("text"))

        choice0 = event_obj.get("choices", [{}])[0]
        delta = choice0.get("delta") or {}
        if isinstance(delta, dict) and delta.get("content"):
            return delta["content"]
        if isinstance(delta, str) and delta:
            return delta
        if isinstance(choice0, dict) and choice0.get("text"):
            return str(choice0.get("text"))

        msg = choice0.get("message") or {}
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
    except Exception:
        pass
    return ""


def _extract_final_text(obj: Dict[str, Any]) -> str:
    """
    Non-stream final response text extraction.
    """
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return ""


# ----------------------------
# Public APIs
# ----------------------------
def chat(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    url: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    verify_ssl: Optional[bool] = None,
    system: str = "You are a helpful assistant.",
    extra: Optional[Dict[str, Any]] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> str:
    """
    Non-stream chat completion. Returns full text.
    """
    cfg = {**DEFAULTS, **_load_simple_yaml(config_path)}
    endpoint = _resolve_endpoint(cfg, url=url)

    api_key = api_key or cfg.get("api_key")
    model_id = model_id or cfg.get("model_id")
    if temperature is None:
        temperature = cfg.get("temperature", 0.7)
    if timeout is None:
        timeout = cfg.get("timeout", 120)
    if verify_ssl is None:
        verify_ssl = cfg.get("verify_ssl", True)

    if not api_key:
        raise ValueError("api_key is required (pass api_key or set it in config.yaml).")
    if not model_id:
        raise ValueError(
            "model_id is required (pass model_id or set it in config.yaml)."
        )

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if extra:
        payload.update(extra)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }

    r = requests.post(
        endpoint, headers=headers, json=payload, timeout=timeout, verify=verify_ssl
    )
    r.raise_for_status()
    obj = r.json()
    return _extract_final_text(obj)


def stream_chat(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    url: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    verify_ssl: Optional[bool] = None,
    system: str = "You are a helpful assistant.",
    extra: Optional[Dict[str, Any]] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Generator[str, None, None]:
    """
    Stream chat completion. Yields incremental text chunks.

    Compatible with:
    - OpenAI SSE: lines start with "data: {...}" and end with "data: [DONE]"
    - Some gateways: plain JSON per line (no "data:")
    """
    cfg = {**DEFAULTS, **_load_simple_yaml(config_path)}
    endpoint = _resolve_endpoint(cfg, url=url)

    api_key = api_key or cfg.get("api_key")
    model_id = model_id or cfg.get("model_id")
    if temperature is None:
        temperature = cfg.get("temperature", 0.7)
    if timeout is None:
        timeout = cfg.get("timeout", 120)
    if verify_ssl is None:
        verify_ssl = cfg.get("verify_ssl", True)

    if not api_key:
        raise ValueError("api_key is required (pass api_key or set it in config.yaml).")
    if not model_id:
        raise ValueError(
            "model_id is required (pass model_id or set it in config.yaml)."
        )

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": True,
    }
    if extra:
        payload.update(extra)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }

    with requests.post(
        endpoint,
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
        verify=verify_ssl,
    ) as resp:
        resp.raise_for_status()

        for raw in _iter_sse_lines(resp):
            line = raw.strip()

            if line.startswith(b"data:"):
                line = line[len(b"data:") :].strip()

            if line in (b"[DONE]", b"done", b"DONE"):
                break

            if line.startswith(b":"):
                continue

            try:
                event = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            text = _extract_delta_text(event)
            if text:
                yield text


def make_messages(user: str, system: Optional[str] = None) -> list[dict]:
    """
    Helper for future extension if you later want multi-turn conversation.
    """
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs
