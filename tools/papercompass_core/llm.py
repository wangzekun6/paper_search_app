"""
兼容 OpenAI 风格接口的大模型调用工具层。

这个文件把项目里的大模型访问逻辑集中到一起，主要负责：
1. 统一读取 API Key / Base URL / 模型名
2. 兼容不同 OpenAI-compatible 接口与历史环境变量命名
3. 处理 Windows 代理读取
4. 发送普通对话请求和结构化 JSON 请求
5. 给检索 query 改写和语义卡片生成提供共用能力
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from .config import PROJECT_ROOT, RUNTIME_DIR

try:
    import winreg
except ImportError:  # pragma: no cover - Windows-only fallback
    winreg = None


# 统一维护当前项目支持的 API 默认值和环境变量搜索顺序。
DEFAULT_OPENAI_API_BASE = "http://hjlyywp.com/v1"
DEFAULT_OPENAI_MODEL_CANDIDATES = ["gpt-5.2", "gpt-5", "gpt-5-codex-mini", "gpt-5-codex", "gpt-5.4"]
TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
PRIVATE_ENV_PATHS = (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / ".env.local",
    PROJECT_ROOT / "tools" / ".env",
    PROJECT_ROOT / "tools" / ".env.local",
    PROJECT_ROOT / "tools" / "local.llm.env",
)

REQUEST_PROXIES: Optional[Dict[str, str]] = None
PRIVATE_ENV_VALUES: Optional[Dict[str, str]] = None
STRUCTURED_OUTPUT_MODE_PATH = RUNTIME_DIR / "llm_structured_output_modes.json"
STRUCTURED_OUTPUT_MODE_CACHE: Optional[Dict[str, str]] = None
OPENAI_RUNTIME_AVAILABLE: Optional[bool] = None
OPENAI_RUNTIME_MESSAGE = ""
OPENAI_RUNTIME_LOCK = threading.Lock()
OPENAI_RUNTIME_CONDITION = threading.Condition(OPENAI_RUNTIME_LOCK)
OPENAI_RUNTIME_PROBE_INFLIGHT = False


class OpenAIAPIError(RuntimeError):
    pass


def _structured_output_mode_cache_key(model: Optional[str] = None) -> str:
    resolved_model = str(model or OPENAI_MODEL).strip() or OPENAI_MODEL
    return f"{OPENAI_API_BASE}::{resolved_model}"


def _load_structured_output_mode_cache() -> Dict[str, str]:
    global STRUCTURED_OUTPUT_MODE_CACHE
    if STRUCTURED_OUTPUT_MODE_CACHE is not None:
        return STRUCTURED_OUTPUT_MODE_CACHE
    try:
        payload = json.loads(STRUCTURED_OUTPUT_MODE_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    STRUCTURED_OUTPUT_MODE_CACHE = {
        str(key): str(value)
        for key, value in payload.items()
        if str(value) in {"json_schema", "json_object"}
    }
    return STRUCTURED_OUTPUT_MODE_CACHE


def _persist_structured_output_mode_cache(cache: Dict[str, str]) -> None:
    STRUCTURED_OUTPUT_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRUCTURED_OUTPUT_MODE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def get_structured_output_mode(model: Optional[str] = None) -> str:
    return _load_structured_output_mode_cache().get(_structured_output_mode_cache_key(model), "")


def remember_structured_output_mode(
    mode: str,
    *,
    requested_model: Optional[str] = None,
    used_model: Optional[str] = None,
) -> None:
    if mode not in {"json_schema", "json_object"}:
        return
    cache = _load_structured_output_mode_cache()
    keys = {_structured_output_mode_cache_key(requested_model)}
    if used_model:
        keys.add(_structured_output_mode_cache_key(used_model))
    changed = False
    for key in keys:
        if cache.get(key) != mode:
            cache[key] = mode
            changed = True
    if changed:
        _persist_structured_output_mode_cache(cache)


def _set_openai_runtime_state(available: Optional[bool], message: str) -> None:
    global OPENAI_RUNTIME_AVAILABLE, OPENAI_RUNTIME_MESSAGE
    OPENAI_RUNTIME_AVAILABLE = available
    OPENAI_RUNTIME_MESSAGE = str(message or "").strip()


def _resolve_api_key(api_key: str = "") -> str:
    return str(api_key or OPENAI_API_KEY).strip()


def get_cached_openai_runtime_status(api_key: str = "") -> Tuple[Optional[bool], str]:
    resolved_api_key = _resolve_api_key(api_key)
    if not resolved_api_key:
        return False, "未提供 OpenAI API Key。请检查环境变量或 tools/.env 配置。"
    with OPENAI_RUNTIME_LOCK:
        return OPENAI_RUNTIME_AVAILABLE, OPENAI_RUNTIME_MESSAGE


# 读取用户级/系统级 Windows 环境变量，兼容新配置后未重启终端的场景。
def _read_windows_env(name: str) -> str:
    if os.name != "nt" or winreg is None:
        return ""

    locations = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for root, path in locations:
        try:
            with winreg.OpenKey(root, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# 去掉形如 `"value"` 或 `'value'` 的包裹引号，便于解析 `.env` 文件。
def _strip_matching_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


# 读取私有环境变量文件，允许项目把模型配置放在本地文件中。
def _read_private_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return values

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = _strip_matching_quotes(value)
        if key and value:
            values[key] = value
    return values


# 合并多个候选 `.env` 文件，形成最终的本地配置视图。
def read_private_env_values() -> Dict[str, str]:
    global PRIVATE_ENV_VALUES
    if PRIVATE_ENV_VALUES is not None:
        return PRIVATE_ENV_VALUES

    merged: Dict[str, str] = {}
    for path in PRIVATE_ENV_PATHS:
        if not path.exists():
            continue
        for key, value in _read_private_env_file(path).items():
            merged[key] = value
    PRIVATE_ENV_VALUES = merged
    return merged


# 环境变量读取顺序：项目本地私有文件 -> 当前进程 -> Windows 系统环境变量。
def read_env_value(*names: str, default: str = "") -> str:
    private_values = read_private_env_values()
    for name in names:
        value = private_values.get(name, "")
        if value:
            return value
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
        value = _read_windows_env(name)
        if value:
            return value

    return default


# Prefer the current process environment, but also fall back to Windows
# user/machine variables so newly configured keys work without restarting.
OPENAI_API_KEY = read_env_value(
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "API_KEY",
    default="",
)
OPENAI_API_BASE = read_env_value(
    "OPENAI_API_BASE",
    "DASHSCOPE_API_BASE",
    "BASE_URL",
    default=DEFAULT_OPENAI_API_BASE,
).rstrip("/")
OPENAI_MODEL = read_env_value(
    "OPENAI_MODEL",
    "DASHSCOPE_MODEL",
    "MODEL",
    default=DEFAULT_OPENAI_MODEL_CANDIDATES[0],
).strip()
try:
    CHAT_MAX_RETRIES = max(
        0,
        min(6, int(read_env_value("OPENAI_CHAT_MAX_RETRIES", "DASHSCOPE_CHAT_MAX_RETRIES", default="2"))),
    )
except ValueError:
    CHAT_MAX_RETRIES = 2
CHAT_BACKOFF_BASE_SECONDS = 0.8


# 尝试把不同来源的代理配置规范化为 requests 可直接使用的格式。
def _normalize_proxy_url(value: str) -> str:
    proxy = value.strip()
    if not proxy:
        return ""
    if "://" not in proxy:
        return f"http://{proxy}"
    return proxy


# 解析 Windows Internet Settings 中的代理字符串。
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


# 自动探测请求代理，优先尊重显式环境变量，再兜底读取系统代理。
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


# 代理探测结果会被缓存，避免每次请求都重复读取系统配置。
def get_request_proxies() -> Optional[Dict[str, str]]:
    global REQUEST_PROXIES
    if REQUEST_PROXIES is None:
        REQUEST_PROXIES = detect_request_proxies() or {}
    return REQUEST_PROXIES or None


# 统一构造 OpenAI-compatible 请求头。
def build_headers(api_key: str = "") -> Dict[str, str]:
    key = api_key or OPENAI_API_KEY
    if not key:
        raise OpenAIAPIError("未提供 OpenAI API Key。请设置环境变量，或在 tools/.env 中配置 OPENAI_API_KEY。")

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


def _is_retryable_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUS_CODES


def is_transient_openai_error_message(message: str) -> bool:
    lowered = str(message or "").lower()
    if not lowered:
        return False
    if any(f"http {status_code}" in lowered for status_code in TRANSIENT_HTTP_STATUS_CODES):
        return True
    transient_markers = (
        "request_error",
        "timeout",
        "timed out",
        "connection aborted",
        "connection reset",
        "temporarily unavailable",
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "upstream",
    )
    return any(marker in lowered for marker in transient_markers)


def build_model_candidates(
    preferred_model: Optional[str] = None,
    extra_candidates: Optional[Sequence[str]] = None,
) -> List[str]:
    candidates: List[str] = []
    for item in [preferred_model, *(extra_candidates or []), OPENAI_MODEL, *DEFAULT_OPENAI_MODEL_CANDIDATES]:
        text = str(item or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def _retry_backoff_sleep(attempt: int) -> None:
    # Use bounded exponential backoff to absorb transient network and gateway jitter.
    delay = min(6.0, CHAT_BACKOFF_BASE_SECONDS * (2 ** attempt))
    time.sleep(delay)


def _probe_openai_runtime(api_key: str = "", timeout: int = 30) -> Tuple[bool, str]:
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


def ensure_openai_runtime_available(api_key: str = "", timeout: int = 30) -> Tuple[bool, str]:
    global OPENAI_RUNTIME_PROBE_INFLIGHT

    resolved_api_key = _resolve_api_key(api_key)
    if not resolved_api_key:
        message = "未提供 OpenAI API Key。请检查环境变量或 tools/.env 配置。"
        with OPENAI_RUNTIME_LOCK:
            _set_openai_runtime_state(False, message)
        return False, message

    with OPENAI_RUNTIME_CONDITION:
        if OPENAI_RUNTIME_AVAILABLE is True:
            return True, OPENAI_RUNTIME_MESSAGE
        if OPENAI_RUNTIME_AVAILABLE is False and not is_transient_openai_error_message(OPENAI_RUNTIME_MESSAGE):
            return False, OPENAI_RUNTIME_MESSAGE
        if OPENAI_RUNTIME_PROBE_INFLIGHT:
            while OPENAI_RUNTIME_PROBE_INFLIGHT:
                OPENAI_RUNTIME_CONDITION.wait()
            if OPENAI_RUNTIME_AVAILABLE is True:
                return True, OPENAI_RUNTIME_MESSAGE
            if OPENAI_RUNTIME_AVAILABLE is False and not is_transient_openai_error_message(OPENAI_RUNTIME_MESSAGE):
                return False, OPENAI_RUNTIME_MESSAGE
        OPENAI_RUNTIME_PROBE_INFLIGHT = True

    ok, message = _probe_openai_runtime(api_key=resolved_api_key, timeout=timeout)
    with OPENAI_RUNTIME_CONDITION:
        if ok:
            _set_openai_runtime_state(True, message)
        elif is_transient_openai_error_message(message):
            _set_openai_runtime_state(None, f"{message}；已跳过预检，正式请求时会继续重试。")
            ok = True
        else:
            _set_openai_runtime_state(False, message)
        OPENAI_RUNTIME_PROBE_INFLIGHT = False
        OPENAI_RUNTIME_CONDITION.notify_all()
        return ok, OPENAI_RUNTIME_MESSAGE


# 普通对话接口，供非结构化文本生成场景复用。
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
    """Send one OpenAI-compatible chat completion request."""

    headers = build_headers(api_key)
    # Some compatible endpoints may not grant all model permissions.
    # Try candidates in order, and allow model-level fallback.
    candidates = build_model_candidates(preferred_model=model, extra_candidates=model_candidates)

    last_error = "No model request was executed."
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

        for attempt in range(CHAT_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{OPENAI_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    proxies=get_request_proxies(),
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                last_error = f"{current_model}: request_error: {exc}"
                if attempt < CHAT_MAX_RETRIES:
                    _retry_backoff_sleep(attempt)
                    continue
                break

            if response.status_code == 200:
                data = response.json()
                return _extract_message_text(data), current_model, data

            error_message = _extract_error_message(response)
            last_error = f"{current_model}: {error_message}"
            if response.status_code in {400, 404} and _is_model_error(error_message):
                break
            if _is_retryable_status(response.status_code):
                if attempt < CHAT_MAX_RETRIES:
                    _retry_backoff_sleep(attempt)
                    continue
                break
            raise OpenAIAPIError(last_error)

    raise OpenAIAPIError(last_error)


def _load_json_payload(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(text[start : end + 1])
        else:
            raise

    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Expected JSON object.", text, 0)
    return payload


# 结构化对话接口，要求模型按照给定 JSON Schema 返回结果。
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

    cached_mode = get_structured_output_mode(model)
    mode_order = [cached_mode] if cached_mode in {"json_schema", "json_object"} else []
    for fallback_mode in ("json_schema", "json_object"):
        if fallback_mode not in mode_order:
            mode_order.append(fallback_mode)

    last_error: Exception | None = None
    for mode in mode_order:
        if mode == "json_schema":
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
                payload = _load_json_payload(content)
                remember_structured_output_mode(
                    "json_schema",
                    requested_model=model,
                    used_model=used_model,
                )
                return payload, used_model
            except (OpenAIAPIError, json.JSONDecodeError) as exc:
                last_error = exc
                continue

        fallback_messages = list(messages) + [
            {
                "role": "system",
                "content": "Return valid JSON only. Do not wrap in markdown fences.",
            }
        ]
        try:
            content, used_model, _ = chat_completion(
                messages=fallback_messages,
                response_format={"type": "json_object"},
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
            )
            try:
                payload = _load_json_payload(content)
            except json.JSONDecodeError:
                repair_messages = list(fallback_messages) + [
                    {
                        "role": "assistant",
                        "content": content,
                    },
                    {
                        "role": "system",
                        "content": "Your previous response was invalid JSON. Return one corrected JSON object only.",
                    },
                ]
                repaired_content, repaired_model, _ = chat_completion(
                    messages=repair_messages,
                    response_format={"type": "json_object"},
                    model=model,
                    temperature=0,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key,
                )
                payload = _load_json_payload(repaired_content)
                used_model = repaired_model
            remember_structured_output_mode(
                "json_object",
                requested_model=model,
                used_model=used_model,
            )
            return payload, used_model
        except (OpenAIAPIError, json.JSONDecodeError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise OpenAIAPIError("Structured chat completion failed without a recorded error.")


# 对当前模型配置做一次最小连通性探测。
def test_openai_api(api_key: str = "", timeout: int = 30) -> Tuple[bool, str]:
    """用一次最小请求测试当前 API Key、Base URL 和模型配置是否可用。"""
    ok, message = _probe_openai_runtime(api_key=api_key, timeout=timeout)
    with OPENAI_RUNTIME_LOCK:
        if ok:
            _set_openai_runtime_state(True, message)
        elif is_transient_openai_error_message(message):
            _set_openai_runtime_state(None, f"{message}；已跳过预检，正式请求时会继续重试。")
        else:
            _set_openai_runtime_state(False, message)
    return ok, message


# 用大模型把自然语言问题改写成更适合检索的关键词串。
def generate_keywords_via_openai(natural_query: str, api_key: str = "") -> Tuple[str, str]:
    """
    把自然语言检索请求改写成更短、更适合词项检索的英文关键词。

    这个函数主要服务统一检索入口里的“自然语言 query 改写”能力。
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
