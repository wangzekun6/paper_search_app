"""
兼容 OpenAI 风格接口的大模型调用工具层。

这个文件把项目里的大模型访问逻辑集中到一起，主要负责：
1. 统一读取 API Key / Base URL / 模型名
2. 兼容 DashScope（百炼）这类 OpenAI-compatible 接口
3. 处理 Windows 代理读取
4. 发送普通对话请求和结构化 JSON 请求
5. 给 Day 2 的 query 改写和 Day 3 的语义卡片生成提供共用能力
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    import winreg
except ImportError:  # pragma: no cover - Windows-only fallback
    winreg = None


DEFAULT_OPENAI_API_KEY = ""
DEFAULT_OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENAI_MODEL_CANDIDATES = ["qwen-plus", "qwen-max", "qwen-turbo"]

# 同时兼容 OpenAI 风格环境变量和 DashScope 自己的命名，
# 这样上层代码只依赖这一层，不需要关心底层到底接的是哪家服务。
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", DEFAULT_OPENAI_API_KEY)
OPENAI_API_BASE = (
    os.environ.get("OPENAI_API_BASE")
    or os.environ.get("DASHSCOPE_API_BASE")
    or DEFAULT_OPENAI_API_BASE
).rstrip("/")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL") or os.environ.get(
    "DASHSCOPE_MODEL",
    DEFAULT_OPENAI_MODEL_CANDIDATES[0],
)
REQUEST_PROXIES: Optional[Dict[str, str]] = None


class OpenAIAPIError(RuntimeError):
    pass


def _normalize_proxy_url(value: str) -> str:
    proxy = value.strip()
    if not proxy:
        return ""
    if "://" not in proxy:
        return f"http://{proxy}"
    return proxy


def _parse_proxy_server(proxy_server: str) -> Dict[str, str]:
    proxy_value = proxy_server.strip()
    if not proxy_value:
        return {}

    if "=" not in proxy_value:
        normalized = _normalize_proxy_url(proxy_value)
        return {"http": normalized, "https": normalized} if normalized else {}

    proxies: Dict[str, str] = {}
    for segment in proxy_value.split(";"):
        if "=" not in segment:
            continue
        scheme, address = segment.split("=", 1)
        normalized = _normalize_proxy_url(address)
        if not normalized:
            continue
        scheme_name = scheme.strip().lower()
        if scheme_name in {"http", "https"}:
            proxies[scheme_name] = normalized

    if "https" not in proxies and "http" in proxies:
        proxies["https"] = proxies["http"]
    if "http" not in proxies and "https" in proxies:
        proxies["http"] = proxies["https"]
    return proxies


def detect_request_proxies() -> Optional[Dict[str, str]]:
    """从当前环境或 Windows 系统代理设置中推断 requests 应该使用的代理。"""

    env_proxy_keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    if any(os.environ.get(key) for key in env_proxy_keys):
        return None

    if os.name != "nt" or winreg is None:
        return None

    try:
        # requests 默认不会自动读取 Windows 的 Internet Settings 代理配置，
        # 所以这里手动把系统代理读出来，避免“浏览器能上网、Python 直连超时”的情况。
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            proxy_enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if not proxy_enabled:
                return None
            proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
    except OSError:
        return None

    proxies = _parse_proxy_server(str(proxy_server))
    return proxies or None


def get_request_proxies() -> Optional[Dict[str, str]]:
    global REQUEST_PROXIES
    if REQUEST_PROXIES is None:
        REQUEST_PROXIES = detect_request_proxies() or {}
    return REQUEST_PROXIES or None


def build_headers(api_key: str = "") -> Dict[str, str]:
    key = api_key or OPENAI_API_KEY
    if not key:
        raise OpenAIAPIError("未提供 OpenAI API Key。")

    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Client-Request-Id": str(uuid.uuid4()),
    }


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text.strip() or f"HTTP {response.status_code}"

    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return response.text.strip() or f"HTTP {response.status_code}"


def _extract_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAIAPIError("OpenAI 返回中缺少 choices。")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        if text_parts:
            return "".join(text_parts).strip()
    raise OpenAIAPIError("OpenAI 返回中缺少文本 content。")


def _is_model_error(message: str) -> bool:
    lowered = message.lower()
    return "model" in lowered and ("not found" in lowered or "does not exist" in lowered or "access" in lowered)


def chat_completion(
    messages: Sequence[Dict[str, str]],
    response_format: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 90,
    api_key: str = "",
    model_candidates: Optional[Sequence[str]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    发送一次兼容 OpenAI 的 chat completion 请求。

    返回值依次是：提取后的文本内容、实际使用的模型名、原始响应 JSON。
    """

    headers = build_headers(api_key)
    # 某些账号未必有首选模型权限，因此这里允许按候选列表依次降级尝试，
    # 提高兼容接口下的可用性。
    candidates = list(model_candidates or ([model] if model else [OPENAI_MODEL] + DEFAULT_OPENAI_MODEL_CANDIDATES[1:]))
    if model and model not in candidates:
        candidates.insert(0, model)

    last_error = "未执行任何模型请求。"
    for current_model in candidates:
        payload: Dict[str, Any] = {
            "model": current_model,
            "messages": list(messages),
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens

        response = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            proxies=get_request_proxies(),
            timeout=timeout,
        )
        if response.status_code == 200:
            data = response.json()
            return _extract_message_text(data), current_model, data

        error_message = _extract_error_message(response)
        last_error = f"{current_model}: {error_message}"
        if response.status_code in {400, 404} and _is_model_error(error_message):
            continue
        raise OpenAIAPIError(last_error)

    raise OpenAIAPIError(last_error)


def structured_chat_completion(
    messages: Sequence[Dict[str, str]],
    schema_name: str,
    schema: Dict[str, Any],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: int = 120,
    api_key: str = "",
) -> Tuple[Dict[str, Any], str]:
    """
    请求结构化 JSON 输出。

    优先尝试严格 json_schema；
    如果兼容服务不支持，再退回普通 json_object 模式。
    """

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }
    try:
        content, used_model, _ = chat_completion(
            messages=messages,
            response_format=response_format,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
        )
        return json.loads(content), used_model
    except OpenAIAPIError:
        # 有些兼容 OpenAI 的服务对严格 json_schema 支持不完整，
        # 这时退回 json_object 模式，尽量保持结构化输出可用。
        fallback_messages = list(messages) + [
            {
                "role": "system",
                "content": "Return valid JSON only. Do not wrap in markdown fences.",
            }
        ]
        content, used_model, _ = chat_completion(
            messages=fallback_messages,
            response_format={"type": "json_object"},
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
        )
        return json.loads(content), used_model


def test_openai_api(api_key: str = "", timeout: int = 30) -> Tuple[bool, str]:
    """用一次最小请求测试当前 API Key、Base URL 和模型配置是否可用。"""

    try:
        content, used_model, _ = chat_completion(
            messages=[
                {"role": "system", "content": "Return exactly OK."},
                {"role": "user", "content": "Reply with OK."},
            ],
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=20,
            timeout=timeout,
            api_key=api_key,
        )
        return True, f"OpenAI Key 可用，模型 `{used_model}` 返回：{content}"
    except Exception as exc:
        return False, f"OpenAI 测试失败: {exc}"


def generate_keywords_via_openai(natural_query: str, api_key: str = "") -> Tuple[str, str]:
    """
    把自然语言检索请求改写成更短、更适合词项检索的英文关键词。

    这个函数主要服务 Day 2 页面里的“自然语言 query 改写”能力。
    """

    query = natural_query.strip()
    if not query:
        return "", ""

    messages = [
        {
            "role": "system",
            "content": (
                "Convert a paper search request into short English search keywords or phrases. "
                "Return a comma-separated list only. Keep it concise and retrieval-friendly."
            ),
        },
        {"role": "user", "content": query},
    ]
    content, used_model, _ = chat_completion(
        messages=messages,
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=80,
        timeout=45,
        api_key=api_key,
    )
    return content.strip(), used_model
