"""
PaperCompass 的意图理解层。

这个模块把自然语言 query 转成结构化 IntentFrame，
并负责：
1. 聚合追问
2. 二轮回复合并
3. 三路 query 生成
4. 意图分析产物导出
"""

from __future__ import annotations

import copy
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import (
    DEMOS_DIR,
    INTENT_ERRORS_PATH,
    INTENT_EVAL_PATH,
    INTENT_PROMPT_PATH,
    PROJECT_ROOT,
    SYSTEM_OUTPUT_DIR,
    ensure_system_layout,
    intent_query_cache_path,
    intent_session_cache_path,
    write_json,
)
from .llm import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OpenAIAPIError,
    structured_chat_completion,
    test_openai_api,
)
OUTPUT_DIR = SYSTEM_OUTPUT_DIR
PROMPT_PATH = INTENT_PROMPT_PATH
PILOT_OUTPUT_PATH = DEMOS_DIR / "pilot_intent_frames.json"
TEST_OUTPUT_PATH = INTENT_EVAL_PATH
MERGE_OUTPUT_PATH = DEMOS_DIR / "intent_frame_merge_examples.json"
FEEDBACK_PATH = SYSTEM_OUTPUT_DIR / "eval" / "intent_feedback.txt"
ERROR_LOG_PATH = INTENT_ERRORS_PATH

PROMPT_VERSION = "intent_v1"
OPENAI_RUNTIME_AVAILABLE: Optional[bool] = None
OPENAI_RUNTIME_MESSAGE = ""
MAX_ERROR_LOG_ENTRIES = 500

SEARCH_SCENE_ENUM = [
    "topic_exploration",
    "survey_lookup",
    "recent_progress",
    "specific_paper_lookup",
    "author_trace",
    "method_constrained_search",
]
SCENE_PRIORITY = {
    "topic_exploration": 1,
    "recent_progress": 2,
    "survey_lookup": 3,
    "method_constrained_search": 4,
    "author_trace": 5,
    "specific_paper_lookup": 6,
}
SLOT_STATUS_ENUM = ["confirmed", "ambiguous", "missing"]
PAPER_TYPE_ENUM = [
    "survey",
    "benchmark",
    "method",
    "empirical_study",
    "application_study",
    "theory",
    "analysis",
]
PREFERENCE_VALUE_ENUM = ["", "yes", "no"]

DEFAULT_INTENT_TEST_QUERIES = [
    "retrieval augmented generation",
    "recent agent memory papers",
    "找最新的 RAG hallucination mitigation 论文",
    "帮我找机器翻译综述",
    "我想看最近两年 speech-to-speech translation 的进展",
    "Self-RAG",
    "Arianna Salazar-Miranda",
    "Dylan Sam 的论文",
    "论文标题里好像有 Riddle Me This",
    "Find papers about tool use with large language models",
    "Looking for survey papers on long context in LLMs",
    "recent benchmark for large language models on reasoning",
    "找用 COMET 做 machine translation quality estimation 的论文",
    "multimodal feedback papers, dataset 不限",
    "我想找 low-resource language Urdu 的论文，最好近三年",
    "有没有关于 scientific data visualization 的经典论文",
    "给我多样一些的 agent evaluation papers，并解释为什么推荐",
    "papers by authors of MALT",
    "Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey",
    "benchmark for large language models",
    "我不确定方法，先看医疗 LLM agent 相关综述",
    "找最近的 graph-based summarization papers, 作者不限",
    "我想看用 early exit 做质量估计的论文",
    "papers on vision-language benchmark with CLIP",
    "帮我找 translation quality estimation，数据集都可以，但要 explainable reason",
]
DEFAULT_MERGE_EXAMPLES = [
    {"initial_query": "帮我找 LLM agent 的论文", "follow_up_reply": "最近两年，综述优先，方法不限"},
    {"initial_query": "找机器翻译论文", "follow_up_reply": "作者不限，最好近三年，用 COMET 或者质量估计相关"},
    {"initial_query": "我想看 RAG", "follow_up_reply": "最好是 survey，并且给我解释为什么推荐"},
    {"initial_query": "找某篇关于 long context 的论文", "follow_up_reply": "标题里可能有 Emulating Retrieval Augmented Generation"},
    {"initial_query": "找 multimodal reasoning papers", "follow_up_reply": "数据集不限，最好结果多样一些"},
]

AMBIGUOUS_MARKERS = [
    "不限",
    "都可以",
    "都行",
    "不确定",
    "没要求",
    "无所谓",
    "any",
    "either",
    "not sure",
    "no preference",
    "don't care",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "best",
    "can",
    "do",
    "find",
    "for",
    "from",
    "give",
    "help",
    "i",
    "in",
    "is",
    "latest",
    "look",
    "looking",
    "me",
    "need",
    "of",
    "on",
    "paper",
    "papers",
    "please",
    "recent",
    "show",
    "the",
    "to",
    "want",
    "with",
    "找",
    "给我",
    "帮我",
    "看看",
    "论文",
    "最近",
    "最新",
    "关于",
    "相关",
    "最好",
    "一些",
    "一下",
}
LOW_SIGNAL_KEYWORD_MARKERS = [
    "帮我找",
    "给我找",
    "的论文",
    "最近",
    "最新",
    "不限",
    "优先",
    "解释",
    "为什么",
    "推荐",
    "论文标题",
    "title",
]
KNOWN_MULTIWORD_PHRASES = [
    "retrieval augmented generation",
    "self-rag",
    "agent memory",
    "machine translation",
    "speech-to-speech translation",
    "quality estimation",
    "hallucination mitigation",
    "long context",
    "data selection",
    "scientific data visualization",
    "low-resource language",
    "tool use",
    "large language model",
    "large language models",
    "multimodal feedback",
    "graph-based summarization",
    "vision-language benchmark",
    "medical llm agent",
    "translation quality estimation",
    "agent evaluation",
]
RAW_QUERY_TERM_TRANSLATIONS = {
    "智能体": "agent",
    "多智能体": "multi-agent",
    "多模态": "multimodal",
    "推理": "reasoning",
    "综述": "survey",
    "美国": "US",
    "美国人": "US authors",
}
CHINESE_PREFIX_NOISE = [
    "帮我找有关",
    "帮我找",
    "给我找有关",
    "给我找",
    "我想找有关",
    "我想找",
    "我想看",
    "想找",
    "想看",
    "有关",
    "关于",
]
CHINESE_SUFFIX_NOISE = [
    "方面的论文",
    "相关论文",
    "的论文",
    "论文",
    "文献",
    "文章",
]
PAPER_TYPE_ALIASES = {
    "survey": "survey",
    "review": "survey",
    "benchmark": "benchmark",
    "method": "method",
    "empirical study": "empirical_study",
    "empirical_study": "empirical_study",
    "application study": "application_study",
    "application_study": "application_study",
    "theory": "theory",
    "analysis": "analysis",
}
PAPER_TYPE_QUERY_LABELS = {
    "survey": "survey papers",
    "benchmark": "benchmark papers",
    "method": "method papers",
    "empirical_study": "empirical studies",
    "application_study": "application studies",
    "theory": "theory papers",
    "analysis": "analysis papers",
}
LOW_SIGNAL_QUERY_TERMS = {
    "",
    "paper",
    "papers",
    "research paper",
    "research papers",
    "study",
    "studies",
    "work",
    "works",
    "survey",
    "review",
    "recent paper",
    "recent papers",
    "latest paper",
    "latest papers",
    "recent work",
    "latest work",
}

DOMAIN_MAP = {
    "machine translation": "machine translation",
    "speech-to-speech translation": "speech processing",
    "speech": "speech processing",
    "multimodal": "multimodal NLP",
    "vision-language": "vision-language modeling",
    "rag": "large language models",
    "retrieval augmented generation": "large language models",
    "tool use": "agent systems",
    "scientific data visualization": "scientific visualization",
    "medical": "medicine",
    "healthcare": "healthcare",
    "clinical": "clinical AI",
    "translation quality estimation": "machine translation",
}
TASK_MAP = {
    "retrieval augmented generation": "retrieval-augmented generation",
    "rag": "retrieval-augmented generation",
    "self-rag": "retrieval-augmented generation",
    "machine translation": "machine translation",
    "speech-to-speech translation": "speech-to-speech translation",
    "quality estimation": "quality estimation",
    "translation quality estimation": "quality estimation",
    "tool use": "tool use",
    "long context": "long-context understanding",
    "scientific data visualization": "scientific data visualization",
    "graph-based summarization": "summarization",
    "agent evaluation": "agent evaluation",
    "multimodal feedback": "multimodal feedback",
    "reasoning": "reasoning",
}
PROBLEM_MAP = {
    "hallucination": "hallucination mitigation",
    "hallucination mitigation": "hallucination mitigation",
    "long context": "long-context understanding",
    "low-resource": "low-resource learning",
    "quality estimation": "quality estimation",
    "agent memory": "memory mechanism",
}
METHOD_MAP = {
    "self-rag": "self-rag",
    "retrieval augmented generation": "retrieval-augmented generation",
    "rag": "retrieval-augmented generation",
    "prompt engineering": "prompt engineering",
    "graph-based": "graph-based method",
    "graph-based summarization": "graph-based summarization",
    "early exit": "early exit",
    "comet": "COMET",
    "agent": "agent architecture",
    "clip": "CLIP",
    "quality estimation": "quality estimation",
}
MODEL_MAP = {
    "llm": "large language model",
    "large language model": "large language model",
    "large language models": "large language model",
    "gpt": "GPT family",
    "bert": "BERT family",
    "t5": "T5 family",
    "transformer": "Transformer",
    "vision-language": "vision-language model",
    "clip": "CLIP",
}
DATASET_MAP = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "wmt": "WMT",
    "squad": "SQuAD",
    "humaneval": "HumanEval",
    "ceval": "C-Eval",
    "c-eval": "C-Eval",
    "math": "MATH",
}
METRIC_MAP = {
    "accuracy": "accuracy",
    "f1": "F1",
    "bleu": "BLEU",
    "rouge": "ROUGE",
    "comet": "COMET",
    "recall": "recall",
    "precision": "precision",
    "correlation": "correlation",
}
MODALITY_MAP = {
    "speech": "speech",
    "audio": "audio",
    "image": "vision",
    "vision": "vision",
    "multimodal": "multimodal",
    "text": "text",
    "video": "video",
}

SLOT_SPECS: Dict[str, Dict[str, Any]] = {
    "search_scene": {"path": ("search_scene",), "kind": "string", "allowed": SEARCH_SCENE_ENUM},
    "research_topic.domain": {"path": ("research_topic", "domain"), "kind": "string"},
    "research_topic.task": {"path": ("research_topic", "task"), "kind": "string"},
    "research_topic.problem": {"path": ("research_topic", "problem"), "kind": "string"},
    "research_topic.keywords": {"path": ("research_topic", "keywords"), "kind": "list"},
    "technical_constraints.method": {"path": ("technical_constraints", "method"), "kind": "string"},
    "technical_constraints.model_family": {"path": ("technical_constraints", "model_family"), "kind": "string"},
    "technical_constraints.dataset": {"path": ("technical_constraints", "dataset"), "kind": "string"},
    "technical_constraints.metric": {"path": ("technical_constraints", "metric"), "kind": "string"},
    "technical_constraints.modality": {"path": ("technical_constraints", "modality"), "kind": "string"},
    "document_attributes.time_range": {"path": ("document_attributes", "time_range"), "kind": "string"},
    "document_attributes.paper_type": {
        "path": ("document_attributes", "paper_type"),
        "kind": "string",
        "allowed": PAPER_TYPE_ENUM,
    },
    "document_attributes.author_name": {"path": ("document_attributes", "author_name"), "kind": "string"},
    "document_attributes.title_hint": {"path": ("document_attributes", "title_hint"), "kind": "string"},
    "result_preferences.prefer_recent": {
        "path": ("result_preferences", "prefer_recent"),
        "kind": "string",
        "allowed": PREFERENCE_VALUE_ENUM,
    },
    "result_preferences.prefer_classic": {
        "path": ("result_preferences", "prefer_classic"),
        "kind": "string",
        "allowed": PREFERENCE_VALUE_ENUM,
    },
    "result_preferences.prefer_survey": {
        "path": ("result_preferences", "prefer_survey"),
        "kind": "string",
        "allowed": PREFERENCE_VALUE_ENUM,
    },
    "result_preferences.prefer_diverse": {
        "path": ("result_preferences", "prefer_diverse"),
        "kind": "string",
        "allowed": PREFERENCE_VALUE_ENUM,
    },
    "result_preferences.need_explainable_reason": {
        "path": ("result_preferences", "need_explainable_reason"),
        "kind": "string",
        "allowed": PREFERENCE_VALUE_ENUM,
    },
}
CLARIFICATION_GROUPS = [
    {
        "label": "search_scene",
        "slots": ["search_scene"],
        "question": "你这次更偏向主题调研、找综述、看最新进展、定位具体论文、追作者，还是带方法约束检索？",
    },
    {
        "label": "research_topic",
        "slots": [
            "research_topic.domain",
            "research_topic.task",
            "research_topic.problem",
            "research_topic.keywords",
        ],
        "question": "请补充你关心的主题信息：领域、任务、问题或关键词。",
    },
    {
        "label": "technical_constraints",
        "slots": [
            "technical_constraints.method",
            "technical_constraints.model_family",
            "technical_constraints.dataset",
            "technical_constraints.metric",
            "technical_constraints.modality",
        ],
        "question": "如果有技术约束，请补充方法、模型家族、数据集、评价指标或模态；没有可以直接说不限。",
    },
    {
        "label": "document_attributes",
        "slots": [
            "document_attributes.time_range",
            "document_attributes.paper_type",
            "document_attributes.author_name",
            "document_attributes.title_hint",
        ],
        "question": "如果有文献属性要求，请补充时间范围、论文类型、作者名或标题线索；没有可以直接说不限。",
    },
    {
        "label": "result_preferences",
        "slots": [
            "result_preferences.prefer_recent",
            "result_preferences.prefer_classic",
            "result_preferences.prefer_survey",
            "result_preferences.prefer_diverse",
            "result_preferences.need_explainable_reason",
        ],
        "question": "如果有结果偏好，请说明是否偏向最新、经典、综述、多样结果，以及是否需要解释命中理由。",
    },
]

SYSTEM_PROMPT = """You are building an IntentFrame for an academic paper retrieval system.

The goal is not keyword expansion only. You must infer user intent into a fixed JSON object.

Rules:
1. Return valid JSON only.
2. Follow the schema exactly.
3. Every slot must contain value, status, source, confidence.
4. status must be one of: confirmed, ambiguous, missing.
5. search_scene must be one of: topic_exploration, survey_lookup, recent_progress, specific_paper_lookup, author_trace, method_constrained_search.
6. document_attributes.paper_type must only use: survey, benchmark, method, empirical_study, application_study, theory, analysis.
7. For preference slots, use value "yes", "no", or "".
8. If the user explicitly says "不限", "都可以", "不确定", or similar, mark the slot as ambiguous and do not keep asking it later.
9. When a prior frame is provided, merge the new reply into the existing state instead of dropping old information.
10. Generate three query groups:
   - coarse_queries: short, broad lexical queries for sparse recall
   - dense_queries: fuller natural-language expressions for semantic retrieval
   - exact_queries: only clear entities or short phrases for exact match
11. missing_slots, clarification_question, and the three query groups are part of the model output itself, not placeholders for downstream rules.
12. clarification_question must ask all still-missing key items in one turn, not one by one.
13. Only mark a slot as missing if it is genuinely unresolved after considering the current text and any provided prior frame.
"""

USER_PROMPT_TEMPLATE = """Mode: {mode}

Current user text:
{user_text}

Previous IntentFrame JSON:
{prior_frame_json}

Return one complete IntentFrame JSON object."""


def ensure_output_dir() -> None:
    ensure_system_layout()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def load_error_log() -> List[Dict[str, Any]]:
    if not ERROR_LOG_PATH.exists():
        return []
    try:
        return json.loads(ERROR_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def append_error_log(entry: Dict[str, Any]) -> None:
    payload = dict(entry)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    errors = load_error_log()
    errors.append(payload)
    if len(errors) > MAX_ERROR_LOG_ENTRIES:
        errors = errors[-MAX_ERROR_LOG_ENTRIES:]
    dump_json(ERROR_LOG_PATH, errors)


def can_use_openai() -> bool:
    global OPENAI_RUNTIME_AVAILABLE, OPENAI_RUNTIME_MESSAGE
    if OPENAI_RUNTIME_AVAILABLE is not None:
        return OPENAI_RUNTIME_AVAILABLE
    ok, message = test_openai_api(OPENAI_API_KEY)
    OPENAI_RUNTIME_AVAILABLE = ok
    OPENAI_RUNTIME_MESSAGE = message
    return ok


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def lowercase_text(value: Any) -> str:
    return clean_text(value).lower()


def clamp_confidence(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, round(number, 4)))


def clean_string_list(values: Iterable[Any], limit: int = 8) -> List[str]:
    items: List[str] = []
    seen = set()
    for value in values:
        text = clean_text(value).strip(" ,;")
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        items.append(text)
        if len(items) >= limit:
            break
    return items


def slot_template(kind: str) -> Dict[str, Any]:
    return {
        "value": [] if kind == "list" else "",
        "status": "missing",
        "source": "",
        "confidence": 0.0,
    }


def blank_intent_frame() -> Dict[str, Any]:
    return {
        "search_scene": slot_template("string"),
        "research_topic": {
            "domain": slot_template("string"),
            "task": slot_template("string"),
            "problem": slot_template("string"),
            "keywords": slot_template("list"),
        },
        "technical_constraints": {
            "method": slot_template("string"),
            "model_family": slot_template("string"),
            "dataset": slot_template("string"),
            "metric": slot_template("string"),
            "modality": slot_template("string"),
        },
        "document_attributes": {
            "time_range": slot_template("string"),
            "paper_type": slot_template("string"),
            "author_name": slot_template("string"),
            "title_hint": slot_template("string"),
        },
        "result_preferences": {
            "prefer_recent": slot_template("string"),
            "prefer_classic": slot_template("string"),
            "prefer_survey": slot_template("string"),
            "prefer_diverse": slot_template("string"),
            "need_explainable_reason": slot_template("string"),
        },
        "missing_slots": [],
        "answered_slots": [],
        "clarification_needed": False,
        "clarification_question": "",
        "coarse_queries": [],
        "dense_queries": [],
        "exact_queries": [],
    }


def slot_schema(value_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": value_schema,
            "status": {"type": "string", "enum": SLOT_STATUS_ENUM},
            "source": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["value", "status", "source", "confidence"],
    }


def string_slot_schema(allowed: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    value_schema: Dict[str, Any] = {"type": "string"}
    if allowed:
        value_schema["enum"] = list(allowed)
    return slot_schema(value_schema)


def list_slot_schema() -> Dict[str, Any]:
    return slot_schema({"type": "array", "items": {"type": "string"}})


INTENT_FRAME_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "search_scene": string_slot_schema([""] + SEARCH_SCENE_ENUM),
        "research_topic": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "domain": string_slot_schema(),
                "task": string_slot_schema(),
                "problem": string_slot_schema(),
                "keywords": list_slot_schema(),
            },
            "required": ["domain", "task", "problem", "keywords"],
        },
        "technical_constraints": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "method": string_slot_schema(),
                "model_family": string_slot_schema(),
                "dataset": string_slot_schema(),
                "metric": string_slot_schema(),
                "modality": string_slot_schema(),
            },
            "required": ["method", "model_family", "dataset", "metric", "modality"],
        },
        "document_attributes": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "time_range": string_slot_schema(),
                "paper_type": string_slot_schema([""] + PAPER_TYPE_ENUM),
                "author_name": string_slot_schema(),
                "title_hint": string_slot_schema(),
            },
            "required": ["time_range", "paper_type", "author_name", "title_hint"],
        },
        "result_preferences": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prefer_recent": string_slot_schema(PREFERENCE_VALUE_ENUM),
                "prefer_classic": string_slot_schema(PREFERENCE_VALUE_ENUM),
                "prefer_survey": string_slot_schema(PREFERENCE_VALUE_ENUM),
                "prefer_diverse": string_slot_schema(PREFERENCE_VALUE_ENUM),
                "need_explainable_reason": string_slot_schema(PREFERENCE_VALUE_ENUM),
            },
            "required": [
                "prefer_recent",
                "prefer_classic",
                "prefer_survey",
                "prefer_diverse",
                "need_explainable_reason",
            ],
        },
        "missing_slots": {"type": "array", "items": {"type": "string"}},
        "answered_slots": {"type": "array", "items": {"type": "string"}},
        "clarification_needed": {"type": "boolean"},
        "clarification_question": {"type": "string"},
        "coarse_queries": {"type": "array", "items": {"type": "string"}},
        "dense_queries": {"type": "array", "items": {"type": "string"}},
        "exact_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "search_scene",
        "research_topic",
        "technical_constraints",
        "document_attributes",
        "result_preferences",
        "missing_slots",
        "answered_slots",
        "clarification_needed",
        "clarification_question",
        "coarse_queries",
        "dense_queries",
        "exact_queries",
    ],
}


def build_messages(user_text: str, prior_frame: Optional[Dict[str, Any]] = None, mode: str = "initial") -> List[Dict[str, str]]:
    prior_frame_json = json.dumps(prior_frame or blank_intent_frame(), ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                mode=mode,
                user_text=user_text.strip(),
                prior_frame_json=prior_frame_json,
            ),
        },
    ]


def write_prompt_file() -> None:
    content = f"""# IntentFrame Prompt

Prompt Version: {PROMPT_VERSION}
Model Default: {OPENAI_MODEL}

## System Prompt
{SYSTEM_PROMPT}

## User Prompt Template
{USER_PROMPT_TEMPLATE}

## Output Schema
```json
{json.dumps(INTENT_FRAME_SCHEMA, ensure_ascii=False, indent=2)}
```
"""
    dump_text(PROMPT_PATH, content)


def get_slot(frame: Dict[str, Any], path: Tuple[str, ...]) -> Dict[str, Any]:
    node: Any = frame
    for key in path:
        node = node[key]
    return node


def set_slot(frame: Dict[str, Any], path: Tuple[str, ...], slot: Dict[str, Any]) -> None:
    node = frame
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = slot


def iter_leaf_slots(frame: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    for path_name, spec in SLOT_SPECS.items():
        yield path_name, get_slot(frame, spec["path"]), spec


def normalize_enum(value: str, allowed: Sequence[str]) -> str:
    text = clean_text(value)
    if text in allowed:
        return text
    lowered_map = {item.lower(): item for item in allowed}
    return lowered_map.get(text.lower(), "")


def normalize_slot(raw_slot: Any, kind: str, allowed: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    slot = slot_template(kind)
    if isinstance(raw_slot, dict):
        slot["source"] = clean_text(raw_slot.get("source"))
        slot["confidence"] = clamp_confidence(raw_slot.get("confidence"), fallback=0.0)
        slot["status"] = normalize_enum(str(raw_slot.get("status", "missing")), SLOT_STATUS_ENUM) or "missing"

        if kind == "list":
            slot["value"] = clean_string_list(raw_slot.get("value", []))
            if slot["status"] == "confirmed" and not slot["value"]:
                slot["status"] = "missing"
        else:
            value = clean_text(raw_slot.get("value"))
            if allowed:
                value = normalize_enum(value, allowed)
            slot["value"] = value
            if slot["status"] == "confirmed" and not slot["value"]:
                slot["status"] = "missing"
            if slot["status"] == "missing":
                slot["value"] = ""

        if slot["status"] == "missing":
            slot["confidence"] = 0.0
            if kind == "list":
                slot["value"] = []
        elif slot["confidence"] == 0.0:
            slot["confidence"] = 0.5 if slot["status"] == "ambiguous" else 0.7
    return slot


def normalize_intent_frame(raw_frame: Any) -> Dict[str, Any]:
    frame = blank_intent_frame()
    if not isinstance(raw_frame, dict):
        raw_frame = {}

    for path_name, spec in SLOT_SPECS.items():
        parent = raw_frame
        for key in spec["path"][:-1]:
            if not isinstance(parent, dict):
                parent = {}
                break
            parent = parent.get(key, {})
        raw_slot = parent.get(spec["path"][-1]) if isinstance(parent, dict) else {}
        set_slot(frame, spec["path"], normalize_slot(raw_slot, spec["kind"], spec.get("allowed")))

    frame["missing_slots"] = clean_slot_name_list(raw_frame.get("missing_slots", []))
    frame["answered_slots"] = clean_slot_name_list(raw_frame.get("answered_slots", []))
    frame["clarification_needed"] = bool(raw_frame.get("clarification_needed"))
    frame["clarification_question"] = clean_text(raw_frame.get("clarification_question", ""))
    frame["coarse_queries"] = clean_string_list(raw_frame.get("coarse_queries", []), limit=5)
    frame["dense_queries"] = clean_string_list(raw_frame.get("dense_queries", []), limit=5)
    frame["exact_queries"] = clean_string_list(raw_frame.get("exact_queries", []), limit=5)
    return frame


def clean_slot_name_list(values: Iterable[Any], limit: Optional[int] = None) -> List[str]:
    items: List[str] = []
    seen = set()
    max_items = limit or len(SLOT_SPECS)
    for value in values:
        text = clean_text(value).strip(" ,;")
        if not text or text not in SLOT_SPECS:
            continue
        if text in seen:
            continue
        seen.add(text)
        items.append(text)
        if len(items) >= max_items:
            break
    return items


def global_ambiguous_reply(text: str) -> bool:
    normalized = lowercase_text(text)
    return normalized in {marker.lower() for marker in AMBIGUOUS_MARKERS}


def contains_any(text: str, patterns: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def extract_quoted_phrases(text: str) -> List[str]:
    matches = re.findall(r"[\"“”'‘’《》](.+?)[\"“”'‘’《》]", text)
    return clean_string_list(matches, limit=5)


def extract_capitalized_name(text: str) -> str:
    match = re.search(r"\b([A-Z][A-Za-z'’.-]+(?: [A-Z][A-Za-z'’.-]+){1,3})\b", text)
    return clean_text(match.group(1)) if match else ""


def extract_author_name(text: str) -> str:
    query = clean_text(text)
    patterns = [
        r"(?:papers by|author|authors of)\s+([A-Z][A-Za-z'’.-]+(?: [A-Z][A-Za-z'’.-]+){1,3})",
        r"(?:作者[是为叫:： ]*)([A-Z][A-Za-z'’.-]+(?: [A-Z][A-Za-z'’.-]+){1,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return clean_text(match.group(1))

    tokens = query.split()
    if 1 < len(tokens) <= 4 and all(re.fullmatch(r"[A-Z][A-Za-z'’.-]+", token) for token in tokens):
        return query
    return extract_capitalized_name(query)


def extract_title_hint(text: str) -> str:
    quoted = extract_quoted_phrases(text)
    if quoted:
        return quoted[0]

    patterns = [
        r"(?:title|titled|named|called)\s*[:： ]\s*(.+)",
        r"(?:标题|论文标题).*?[是为叫:： ]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_text(match.group(1))

    words = clean_text(text).split()
    capitalized = [word for word in words if re.fullmatch(r"[A-Z][A-Za-z'’.-]*", word)]
    if len(words) >= 5 and len(capitalized) >= 4:
        return clean_text(text)
    return ""


def extract_year_range(text: str) -> str:
    query = clean_text(text)
    years = re.findall(r"\b(20\d{2})\b", query)
    if len(years) >= 2:
        return f"{years[0]}-{years[-1]}"
    if len(years) == 1:
        year = years[0]
        lowered = query.lower()
        if contains_any(lowered, ["after", "since", "之后", "以后"]):
            return f">={year}"
        if contains_any(lowered, ["before", "until", "之前"]):
            return f"<={year}"
        return year
    if contains_any(query, ["最近", "最新", "recent", "latest"]):
        return "recent"
    if "近两年" in query:
        return "last 2 years"
    if "近三年" in query:
        return "last 3 years"
    return ""


def match_phrase(text: str, mapping: Dict[str, str]) -> str:
    lowered = text.lower()
    for phrase, value in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase.lower() in lowered:
            return value
    return ""


def tokenize_keywords(text: str) -> List[str]:
    query = lowercase_text(text)
    keywords = extract_quoted_phrases(text)
    keywords.extend([phrase for phrase in KNOWN_MULTIWORD_PHRASES if phrase in query])
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}|[\u4e00-\u9fff]{2,}", query):
        if token in STOPWORDS:
            continue
        if any(marker in token for marker in LOW_SIGNAL_KEYWORD_MARKERS):
            continue
        keywords.append(token)
    return clean_string_list(keywords, limit=8)


def extract_method_hint(text: str) -> str:
    lowered = lowercase_text(text)
    pre_for_match = re.search(r"\b([A-Za-z][A-Za-z0-9+\- ]{1,40})\s+for\b", text, flags=re.IGNORECASE)
    if pre_for_match:
        pre_for_candidate = clean_text(pre_for_match.group(1)).strip(" ,;")
        pre_for_method = match_phrase(pre_for_candidate, METHOD_MAP)
        if pre_for_method:
            return pre_for_method

    patterns = [
        r"(?:using|with|based on|via)\s+([A-Za-z0-9+\- ]{2,60})",
        r"(?:基于|使用|采用)([^，。,.；;]{2,30})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = clean_text(match.group(1))
            candidate = re.split(r"\b(?:for|to|的|并|and)\b", candidate, maxsplit=1)[0].strip(" ,;")
            if candidate:
                return candidate
    fallback_method = match_phrase(text, METHOD_MAP)
    if not fallback_method:
        return ""
    has_explicit_method_context = contains_any(
        lowered,
        ["method", "methods", "approach", "using", "with", "based on", "via", "方法", "基于", "使用", "采用"],
    )
    if has_explicit_method_context or fallback_method in {"self-rag", "early exit", "COMET", "CLIP"}:
        return fallback_method
    return ""


def slot_value_is_empty(slot: Dict[str, Any]) -> bool:
    value = slot.get("value")
    if isinstance(value, list):
        return len(value) == 0
    return not clean_text(value)


def build_slot(value: Any, status: str, source: str, confidence: float) -> Dict[str, Any]:
    slot = {
        "value": value,
        "status": normalize_enum(status, SLOT_STATUS_ENUM) or "missing",
        "source": source,
        "confidence": clamp_confidence(confidence),
    }
    if slot["status"] == "missing":
        slot["confidence"] = 0.0
    return slot


def slot_mark_ambiguous(kind: str, source: str, confidence: float = 0.8) -> Dict[str, Any]:
    return build_slot([] if kind == "list" else "", "ambiguous", source, confidence)


def normalize_paper_type_value(value: Any) -> str:
    normalized = re.sub(r"[_\-\s]+", " ", lowercase_text(value))
    return PAPER_TYPE_ALIASES.get(normalized, "")


def paper_type_query_label(paper_type: str) -> str:
    normalized = normalize_paper_type_value(paper_type)
    return PAPER_TYPE_QUERY_LABELS.get(normalized, "papers")


def is_low_signal_query_term(value: Any, paper_type: str = "") -> bool:
    text = clean_text(value)
    normalized = lowercase_text(text)
    if not normalized:
        return True
    if normalized in LOW_SIGNAL_QUERY_TERMS:
        return True
    normalized_paper_type = normalize_paper_type_value(paper_type)
    if normalized_paper_type and normalize_paper_type_value(text) == normalized_paper_type:
        return True
    return False


def filter_retrieval_terms(values: Iterable[Any], paper_type: str = "", limit: int = 8) -> List[str]:
    filtered: List[str] = []
    for value in clean_string_list(values, limit=max(limit * 2, 8)):
        if is_low_signal_query_term(value, paper_type=paper_type):
            continue
        filtered.append(value)
        if len(filtered) >= limit:
            break
    return filtered


def compact_query_text(values: Iterable[Any], paper_type: str = "", limit: int = 3) -> str:
    return clean_text(" ".join(filter_retrieval_terms(values, paper_type=paper_type, limit=limit)))


def reconcile_slot_semantics(frame: Dict[str, Any]) -> None:
    task_slot = get_slot(frame, SLOT_SPECS["research_topic.task"]["path"])
    normalized_task_paper_type = normalize_paper_type_value(task_slot["value"])
    if task_slot["status"] != "confirmed" or not normalized_task_paper_type:
        return

    paper_type_slot = get_slot(frame, SLOT_SPECS["document_attributes.paper_type"]["path"])
    if paper_type_slot["status"] == "missing":
        set_slot(
            frame,
            SLOT_SPECS["document_attributes.paper_type"]["path"],
            build_slot(
                normalized_task_paper_type,
                "confirmed",
                task_slot["source"] or "derived_from_query",
                max(task_slot["confidence"], 0.78),
            ),
        )

    if normalized_task_paper_type == "survey":
        prefer_survey_slot = get_slot(frame, SLOT_SPECS["result_preferences.prefer_survey"]["path"])
        if prefer_survey_slot["status"] == "missing":
            set_slot(
                frame,
                SLOT_SPECS["result_preferences.prefer_survey"]["path"],
                build_slot("yes", "confirmed", task_slot["source"] or "derived_from_query", 0.78),
            )

    set_slot(frame, SLOT_SPECS["research_topic.task"]["path"], slot_template("string"))


def infer_search_scene_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    title_hint = get_slot(frame, SLOT_SPECS["document_attributes.title_hint"]["path"])
    author_name = get_slot(frame, SLOT_SPECS["document_attributes.author_name"]["path"])
    paper_type = get_slot(frame, SLOT_SPECS["document_attributes.paper_type"]["path"])
    prefer_recent = get_slot(frame, SLOT_SPECS["result_preferences.prefer_recent"]["path"])
    prefer_survey = get_slot(frame, SLOT_SPECS["result_preferences.prefer_survey"]["path"])
    method = get_slot(frame, SLOT_SPECS["technical_constraints.method"]["path"])

    if title_hint["status"] == "confirmed" and not slot_value_is_empty(title_hint):
        return build_slot("specific_paper_lookup", "confirmed", "derived_from_query", 0.92)
    if author_name["status"] == "confirmed" and not slot_value_is_empty(author_name):
        return build_slot("author_trace", "confirmed", "derived_from_query", 0.92)
    if paper_type["value"] == "survey" or prefer_survey["value"] == "yes":
        return build_slot("survey_lookup", "confirmed", "derived_from_query", 0.88)
    if prefer_recent["value"] == "yes":
        return build_slot("recent_progress", "confirmed", "derived_from_query", 0.84)
    if method["status"] == "confirmed" and not slot_value_is_empty(method):
        return build_slot("method_constrained_search", "confirmed", "derived_from_query", 0.8)
    return build_slot("topic_exploration", "confirmed", "derived_from_query", 0.55)


def scene_priority(value: Any) -> int:
    return SCENE_PRIORITY.get(clean_text(value), 0)


def clone_completion_slot(base_slot: Dict[str, Any], candidate_slot: Dict[str, Any]) -> Dict[str, Any]:
    slot = copy.deepcopy(candidate_slot)
    slot["source"] = "rule_completion"
    slot["confidence"] = max(
        clamp_confidence(base_slot.get("confidence"), fallback=0.0),
        min(clamp_confidence(candidate_slot.get("confidence"), fallback=0.0), 0.78),
    )
    return slot


def apply_heuristic_slot_completion(user_text: str, frame: Dict[str, Any]) -> Dict[str, Any]:
    query = clean_text(user_text)
    if not query:
        return finalize_intent_frame(frame)

    completed = copy.deepcopy(finalize_intent_frame(frame))
    fallback = heuristic_parse_text(query, source="rule_completion", infer_defaults=True)

    for path_name, spec in SLOT_SPECS.items():
        current_slot = get_slot(completed, spec["path"])
        fallback_slot = get_slot(fallback, spec["path"])
        if fallback_slot.get("status") != "confirmed":
            continue

        if spec["kind"] == "list":
            current_values = current_slot.get("value", []) if isinstance(current_slot.get("value"), list) else []
            fallback_values = clean_string_list(fallback_slot.get("value", []), limit=8)
            should_fill = (current_slot.get("status") in {"missing", "ambiguous"} or not current_values) and bool(fallback_values)
            if not should_fill:
                continue
            merged_values = clean_string_list(list(current_values) + fallback_values, limit=8)
            if not merged_values:
                continue
            slot = clone_completion_slot(current_slot, fallback_slot)
            slot["value"] = merged_values
            set_slot(completed, spec["path"], slot)
            continue

        current_value = clean_text(current_slot.get("value"))
        fallback_value = clean_text(fallback_slot.get("value"))
        if not fallback_value:
            continue

        should_fill = current_slot.get("status") == "missing" or (current_slot.get("status") == "ambiguous" and not current_value)
        if path_name == "search_scene" and current_slot.get("status") in {"missing", "ambiguous"}:
            should_fill = True
        if (
            path_name == "research_topic.domain"
            and current_slot.get("status") == "confirmed"
            and lowercase_text(current_value) in {"artificial intelligence", "ai", "large language models"}
            and lowercase_text(current_value) != lowercase_text(fallback_value)
        ):
            should_fill = True
        if not should_fill:
            continue
        set_slot(completed, spec["path"], clone_completion_slot(current_slot, fallback_slot))

    lowered = lowercase_text(query)
    scene_slot = get_slot(completed, SLOT_SPECS["search_scene"]["path"])
    author_slot = get_slot(completed, SLOT_SPECS["document_attributes.author_name"]["path"])
    title_hint_slot = get_slot(completed, SLOT_SPECS["document_attributes.title_hint"]["path"])
    if contains_any(lowered, ["papers by authors of", "authors of", "作者的论文", "按作者找论文"]):
        if scene_slot.get("value") != "author_trace":
            set_slot(
                completed,
                SLOT_SPECS["search_scene"]["path"],
                build_slot("author_trace", "confirmed", "rule_completion", max(scene_slot.get("confidence", 0.0), 0.74)),
            )
        if author_slot.get("status") == "missing":
            set_slot(completed, SLOT_SPECS["document_attributes.author_name"]["path"], slot_mark_ambiguous("string", "rule_completion", 0.74))
    elif title_hint_slot.get("status") == "confirmed" and scene_slot.get("status") in {"missing", "ambiguous"}:
        set_slot(
            completed,
            SLOT_SPECS["search_scene"]["path"],
            build_slot("specific_paper_lookup", "confirmed", "rule_completion", max(scene_slot.get("confidence", 0.0), 0.74)),
        )

    return finalize_intent_frame(completed)


def preserve_prior_scene(frame: Dict[str, Any], prior_scene: str) -> None:
    if not prior_scene:
        return
    scene_slot = get_slot(frame, SLOT_SPECS["search_scene"]["path"])
    current_scene = clean_text(scene_slot.get("value"))
    if scene_priority(prior_scene) <= scene_priority(current_scene):
        return

    can_keep = False
    if prior_scene == "method_constrained_search":
        method_slot = get_slot(frame, SLOT_SPECS["technical_constraints.method"]["path"])
        can_keep = method_slot.get("status") == "confirmed" and not slot_value_is_empty(method_slot)
    elif prior_scene == "author_trace":
        author_slot = get_slot(frame, SLOT_SPECS["document_attributes.author_name"]["path"])
        can_keep = author_slot.get("status") == "confirmed" and not slot_value_is_empty(author_slot)
    elif prior_scene == "specific_paper_lookup":
        title_slot = get_slot(frame, SLOT_SPECS["document_attributes.title_hint"]["path"])
        can_keep = title_slot.get("status") == "confirmed" and not slot_value_is_empty(title_slot)
    elif prior_scene == "survey_lookup":
        paper_type_slot = get_slot(frame, SLOT_SPECS["document_attributes.paper_type"]["path"])
        survey_slot = get_slot(frame, SLOT_SPECS["result_preferences.prefer_survey"]["path"])
        can_keep = paper_type_slot.get("value") == "survey" or survey_slot.get("value") == "yes"

    if can_keep:
        set_slot(
            frame,
            SLOT_SPECS["search_scene"]["path"],
            build_slot(prior_scene, "confirmed", "merged", max(scene_slot.get("confidence", 0.0), 0.82)),
        )


def fill_derived_slots(frame: Dict[str, Any]) -> None:
    reconcile_slot_semantics(frame)

    paper_type = get_slot(frame, SLOT_SPECS["document_attributes.paper_type"]["path"])
    prefer_survey = get_slot(frame, SLOT_SPECS["result_preferences.prefer_survey"]["path"])
    if paper_type["value"] == "survey" and prefer_survey["status"] == "missing":
        set_slot(
            frame,
            SLOT_SPECS["result_preferences.prefer_survey"]["path"],
            build_slot("yes", "confirmed", "derived_from_query", 0.78),
        )

    search_scene = get_slot(frame, SLOT_SPECS["search_scene"]["path"])
    prefer_recent = get_slot(frame, SLOT_SPECS["result_preferences.prefer_recent"]["path"])
    if search_scene["value"] == "recent_progress" and prefer_recent["status"] == "missing":
        set_slot(
            frame,
            SLOT_SPECS["result_preferences.prefer_recent"]["path"],
            build_slot("yes", "confirmed", "derived_from_query", 0.76),
        )


def compute_slot_lists(frame: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    missing_slots: List[str] = []
    answered_slots: List[str] = []
    for path_name, slot, _ in iter_leaf_slots(frame):
        if slot["status"] == "missing":
            missing_slots.append(path_name)
        else:
            answered_slots.append(path_name)
    return missing_slots, answered_slots


def build_clarification_question(frame: Dict[str, Any]) -> Tuple[bool, str]:
    questions: List[str] = []
    for group in CLARIFICATION_GROUPS:
        if any(get_slot(frame, SLOT_SPECS[path]["path"])["status"] == "missing" for path in group["slots"]):
            questions.append(group["question"])
    if not questions:
        return False, ""
    lines = [f"{idx}. {question}" for idx, question in enumerate(questions, start=1)]
    return True, "为更准确检索，请一次性补充以下信息：\n" + "\n".join(lines)


def collect_query_values(frame: Dict[str, Any], statuses: Sequence[str] = ("confirmed",)) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {
        "topic": [],
        "constraints": [],
        "attributes": [],
        "preferences": [],
        "exact": [],
    }

    for path_name, slot, _ in iter_leaf_slots(frame):
        if slot["status"] not in statuses:
            continue
        value = slot["value"]
        texts = value if isinstance(value, list) else ([value] if clean_text(value) else [])

        if path_name.startswith("research_topic."):
            values["topic"].extend(texts)
        elif path_name.startswith("technical_constraints."):
            values["constraints"].extend(texts)
            if path_name in {
                "technical_constraints.method",
                "technical_constraints.model_family",
                "technical_constraints.dataset",
            }:
                values["exact"].extend(texts)
        elif path_name.startswith("document_attributes."):
            values["attributes"].extend(texts)
            if path_name in {"document_attributes.author_name", "document_attributes.title_hint"}:
                values["exact"].extend(texts)
        elif path_name.startswith("result_preferences.") and texts:
            values["preferences"].extend(texts)
    paper_type = get_slot(frame, SLOT_SPECS["document_attributes.paper_type"]["path"])["value"]
    values["topic"] = filter_retrieval_terms(values["topic"], paper_type=paper_type, limit=6)
    values["constraints"] = filter_retrieval_terms(values["constraints"], limit=6)
    values["attributes"] = filter_retrieval_terms(values["attributes"], paper_type=paper_type, limit=6)
    values["preferences"] = clean_string_list(
        [value for value in values["preferences"] if clean_text(value) not in {"yes", "no"}],
        limit=6,
    )
    values["exact"] = clean_string_list(values["exact"], limit=6)
    return values


def generate_query_variants(frame: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    values = collect_query_values(frame)
    search_scene = get_slot(frame, SLOT_SPECS["search_scene"]["path"])["value"]
    paper_type = get_slot(frame, SLOT_SPECS["document_attributes.paper_type"]["path"])["value"]
    prefer_recent = get_slot(frame, SLOT_SPECS["result_preferences.prefer_recent"]["path"])["value"]
    prefer_diverse = get_slot(frame, SLOT_SPECS["result_preferences.prefer_diverse"]["path"])["value"]
    paper_label = paper_type_query_label(paper_type)

    topic_text = compact_query_text(values["topic"], paper_type=paper_type, limit=3)
    focused_topic_text = compact_query_text(values["topic"], paper_type=paper_type, limit=2)
    constraint_text = compact_query_text(values["constraints"], limit=2)
    attribute_text = compact_query_text(values["attributes"], paper_type=paper_type, limit=2)
    coarse_queries: List[str] = []
    if topic_text:
        coarse_queries.append(topic_text)
    if focused_topic_text and constraint_text:
        coarse_queries.append(clean_text(f"{focused_topic_text} {constraint_text}"))
    if focused_topic_text and paper_type:
        coarse_queries.append(clean_text(f"{focused_topic_text} {paper_type}"))
    if prefer_recent == "yes":
        coarse_queries.append(clean_text(f"recent {focused_topic_text or constraint_text or attribute_text or paper_type}"))
    if search_scene == "author_trace" and values["exact"]:
        coarse_queries.append(values["exact"][0])
    elif search_scene == "specific_paper_lookup" and values["exact"]:
        coarse_queries.append(values["exact"][0])
    elif not coarse_queries and paper_type:
        coarse_queries.append(paper_type)
    coarse_queries = clean_string_list(coarse_queries, limit=5)

    dense_queries: List[str] = []
    if search_scene == "author_trace" and values["exact"]:
        dense_queries.append(clean_text(f"papers by {values['exact'][0]}"))
    if search_scene == "specific_paper_lookup" and values["exact"]:
        dense_queries.append(clean_text(f"paper titled {values['exact'][0]}"))
    if topic_text:
        dense_queries.append(clean_text(f"{paper_label} on {topic_text}" if paper_type else f"papers on {topic_text}"))
    if focused_topic_text and constraint_text:
        dense_queries.append(clean_text(f"{focused_topic_text} with {constraint_text}"))
    if topic_text and attribute_text:
        dense_queries.append(clean_text(f"{topic_text} {attribute_text}"))
    if constraint_text and not topic_text:
        dense_queries.append(clean_text(f"papers using {constraint_text}"))
    if prefer_recent == "yes" and topic_text:
        dense_queries.append(clean_text(f"recent {paper_label} on {topic_text}" if paper_type else f"recent papers on {topic_text}"))
    if prefer_diverse == "yes" and topic_text:
        dense_queries.append(clean_text(f"diverse {paper_label} on {topic_text}" if paper_type else f"diverse papers on {topic_text}"))
    if not dense_queries and coarse_queries:
        dense_queries.extend(coarse_queries[:2])
    dense_queries = clean_string_list(dense_queries, limit=4)

    exact_queries = clean_string_list(values["exact"], limit=5)
    return coarse_queries, dense_queries, exact_queries


def has_effective_query_groups(frame: Dict[str, Any]) -> bool:
    return bool(
        clean_string_list(frame.get("coarse_queries", []), limit=1)
        or clean_string_list(frame.get("dense_queries", []), limit=1)
        or clean_string_list(frame.get("exact_queries", []), limit=1)
    )


def simplify_chinese_keyword(value: str) -> str:
    text = clean_text(value)
    if not text:
        return ""
    for prefix in CHINESE_PREFIX_NOISE:
        while text.startswith(prefix):
            text = text[len(prefix) :]
    for suffix in CHINESE_SUFFIX_NOISE:
        while text.endswith(suffix):
            text = text[: -len(suffix)]
    text = re.sub(r"^(?:作者要是|作者是|作者要|作者为)", "", text)
    text = text.strip(" 的，,。.;:!?！？")
    return clean_text(text)


def extract_raw_query_terms(user_text: str) -> List[str]:
    query = clean_text(user_text)
    if not query:
        return []

    terms: List[str] = []
    lowered = lowercase_text(query)
    terms.extend(extract_quoted_phrases(query))
    terms.extend(tokenize_keywords(query))

    for token in re.findall(r"[A-Za-z][A-Za-z0-9+\-]{1,}|[\u4e00-\u9fff]{2,}", lowered):
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            simplified = simplify_chinese_keyword(token)
            if simplified:
                terms.append(simplified)
                translated = RAW_QUERY_TERM_TRANSLATIONS.get(simplified)
                if translated:
                    terms.append(translated)
            continue
        terms.append(token)

    for phrase, english in RAW_QUERY_TERM_TRANSLATIONS.items():
        if phrase in query:
            terms.append(english)
    return clean_string_list(terms, limit=12)


def ensure_query_groups_from_user_text(frame: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    normalized = finalize_intent_frame(frame)
    existing_query_text = " ".join(
        clean_string_list(
            normalized.get("coarse_queries", []) + normalized.get("dense_queries", []) + normalized.get("exact_queries", []),
            limit=12,
        )
    )
    raw_terms = extract_raw_query_terms(user_text)
    paper_type = get_slot(normalized, SLOT_SPECS["document_attributes.paper_type"]["path"])["value"]
    filtered_terms = filter_retrieval_terms(raw_terms, paper_type=paper_type, limit=8)
    fallback_terms = clean_string_list(filtered_terms or raw_terms, limit=8)
    existing_query_lower = lowercase_text(existing_query_text)
    missing_fallback_terms = [
        term for term in fallback_terms if term and lowercase_text(term) not in existing_query_lower
    ]
    if has_effective_query_groups(normalized) and not missing_fallback_terms:
        return normalized

    repaired = copy.deepcopy(normalized)
    keyword_path = SLOT_SPECS["research_topic.keywords"]["path"]
    keyword_slot = get_slot(repaired, keyword_path)

    if fallback_terms:
        existing_keywords = clean_string_list(keyword_slot.get("value", []), limit=8) if isinstance(keyword_slot.get("value"), list) else []
        if keyword_slot.get("status") in {"missing", "ambiguous"} or not existing_keywords:
            set_slot(
                repaired,
                keyword_path,
                build_slot(
                    fallback_terms,
                    "confirmed",
                    "raw_query_fallback",
                    max(float(keyword_slot.get("confidence") or 0.0), 0.66),
                ),
            )
        else:
            merged_keywords = clean_string_list(existing_keywords + fallback_terms, limit=8)
            if len(merged_keywords) > len(existing_keywords):
                set_slot(
                    repaired,
                    keyword_path,
                    build_slot(
                        merged_keywords,
                        "confirmed",
                        "raw_query_fallback_merge",
                        max(float(keyword_slot.get("confidence") or 0.0), 0.68),
                    ),
                )

    scene_path = SLOT_SPECS["search_scene"]["path"]
    scene_slot = get_slot(repaired, scene_path)
    if scene_slot.get("status") == "missing":
        set_slot(repaired, scene_path, build_slot("topic_exploration", "confirmed", "raw_query_fallback", 0.6))

    coarse_queries, dense_queries, exact_queries = generate_query_variants(repaired)
    if not coarse_queries and not dense_queries and not exact_queries:
        seed_terms = fallback_terms
        seed_text = compact_query_text(seed_terms, paper_type=paper_type, limit=3) or clean_text(user_text)
        coarse_queries = clean_string_list([seed_text] + seed_terms, limit=5)
        dense_seed = coarse_queries[0] if coarse_queries else seed_text
        if dense_seed:
            dense_hint = dense_seed if re.search(r"[\u4e00-\u9fff]", dense_seed) else f"papers on {dense_seed}"
            dense_queries = clean_string_list([dense_hint] + coarse_queries[:2], limit=4)
        exact_queries = clean_string_list(seed_terms[:3], limit=5)

    repaired["coarse_queries"] = clean_string_list(coarse_queries, limit=5)
    repaired["dense_queries"] = clean_string_list(dense_queries, limit=5)
    repaired["exact_queries"] = clean_string_list(exact_queries, limit=5)
    return finalize_intent_frame(repaired)


def has_meaningful_slots(frame: Dict[str, Any]) -> bool:
    for _, slot, _ in iter_leaf_slots(frame):
        status = slot.get("status")
        if status == "missing":
            continue
        value = slot.get("value")
        if isinstance(value, list):
            if value or status == "ambiguous":
                return True
            continue
        if clean_text(value) or status == "ambiguous":
            return True
    return False


def finalize_intent_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_intent_frame(frame)
    fill_derived_slots(normalized)
    computed_missing_slots, computed_answered_slots = compute_slot_lists(normalized)
    llm_missing_slots = clean_slot_name_list(normalized.get("missing_slots", []))
    if llm_missing_slots:
        missing_slots = clean_slot_name_list(
            [slot for slot in llm_missing_slots if slot in computed_missing_slots] + computed_missing_slots
        )
    else:
        missing_slots = computed_missing_slots

    llm_answered_slots = clean_slot_name_list(normalized.get("answered_slots", []))
    if llm_answered_slots:
        answered_slots = clean_slot_name_list(
            [slot for slot in llm_answered_slots if slot in computed_answered_slots and slot not in missing_slots]
            + [slot for slot in computed_answered_slots if slot not in missing_slots]
        )
    else:
        answered_slots = [slot for slot in computed_answered_slots if slot not in missing_slots]

    fallback_clarification_needed, fallback_clarification_question = build_clarification_question(normalized)
    clarification_needed = bool(missing_slots)
    clarification_question = clean_text(normalized.get("clarification_question", "")) if clarification_needed else ""
    if not clarification_question and clarification_needed:
        clarification_question = fallback_clarification_question if fallback_clarification_needed else ""

    coarse_queries = clean_string_list(normalized.get("coarse_queries", []), limit=5)
    dense_queries = clean_string_list(normalized.get("dense_queries", []), limit=5)
    exact_queries = clean_string_list(normalized.get("exact_queries", []), limit=5)
    if not coarse_queries and not dense_queries and not exact_queries:
        # Exception-only guard: if model output is completely empty, fall back to minimal query variants
        # so retrieval can still proceed. Main path still takes LLM-generated query groups directly.
        fallback_coarse_queries, fallback_dense_queries, fallback_exact_queries = generate_query_variants(normalized)
        coarse_queries = fallback_coarse_queries
        dense_queries = fallback_dense_queries
        exact_queries = fallback_exact_queries

    normalized["missing_slots"] = missing_slots
    normalized["answered_slots"] = answered_slots
    normalized["clarification_needed"] = clarification_needed
    normalized["clarification_question"] = clarification_question
    normalized["coarse_queries"] = clean_string_list(coarse_queries, limit=5)
    normalized["dense_queries"] = clean_string_list(dense_queries, limit=5)
    normalized["exact_queries"] = clean_string_list(exact_queries, limit=5)
    return normalized


def detect_slot_ambiguous(text: str, slot_keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    if not any(keyword.lower() in lowered for keyword in slot_keywords):
        return False
    return any(marker.lower() in lowered for marker in AMBIGUOUS_MARKERS)


def heuristic_parse_text(text: str, source: str, infer_defaults: bool = True) -> Dict[str, Any]:
    query = clean_text(text)
    lowered = query.lower()
    frame = blank_intent_frame()

    keywords = tokenize_keywords(query)
    if keywords:
        set_slot(frame, SLOT_SPECS["research_topic.keywords"]["path"], build_slot(keywords, "confirmed", source, 0.72))

    domain = match_phrase(lowered, DOMAIN_MAP)
    if domain:
        set_slot(frame, SLOT_SPECS["research_topic.domain"]["path"], build_slot(domain, "confirmed", source, 0.72))

    task = match_phrase(lowered, TASK_MAP)
    if task:
        set_slot(frame, SLOT_SPECS["research_topic.task"]["path"], build_slot(task, "confirmed", source, 0.76))

    problem = match_phrase(lowered, PROBLEM_MAP)
    if problem:
        set_slot(frame, SLOT_SPECS["research_topic.problem"]["path"], build_slot(problem, "confirmed", source, 0.72))

    method = extract_method_hint(query)
    if method:
        set_slot(frame, SLOT_SPECS["technical_constraints.method"]["path"], build_slot(method, "confirmed", source, 0.76))
    elif detect_slot_ambiguous(query, ["method", "methods", "方法"]):
        set_slot(frame, SLOT_SPECS["technical_constraints.method"]["path"], slot_mark_ambiguous("string", source))

    model = match_phrase(lowered, MODEL_MAP)
    if model:
        set_slot(frame, SLOT_SPECS["technical_constraints.model_family"]["path"], build_slot(model, "confirmed", source, 0.74))
    elif detect_slot_ambiguous(query, ["model", "模型", "model family"]):
        set_slot(frame, SLOT_SPECS["technical_constraints.model_family"]["path"], slot_mark_ambiguous("string", source))

    dataset = match_phrase(lowered, DATASET_MAP)
    if dataset:
        set_slot(frame, SLOT_SPECS["technical_constraints.dataset"]["path"], build_slot(dataset, "confirmed", source, 0.78))
    elif detect_slot_ambiguous(query, ["dataset", "数据集"]):
        set_slot(frame, SLOT_SPECS["technical_constraints.dataset"]["path"], slot_mark_ambiguous("string", source))

    metric = match_phrase(lowered, METRIC_MAP)
    if metric:
        set_slot(frame, SLOT_SPECS["technical_constraints.metric"]["path"], build_slot(metric, "confirmed", source, 0.72))
    elif detect_slot_ambiguous(query, ["metric", "指标", "评价"]):
        set_slot(frame, SLOT_SPECS["technical_constraints.metric"]["path"], slot_mark_ambiguous("string", source))

    modality = match_phrase(lowered, MODALITY_MAP)
    if modality:
        set_slot(frame, SLOT_SPECS["technical_constraints.modality"]["path"], build_slot(modality, "confirmed", source, 0.7))
    elif detect_slot_ambiguous(query, ["modality", "模态"]):
        set_slot(frame, SLOT_SPECS["technical_constraints.modality"]["path"], slot_mark_ambiguous("string", source))

    time_range = extract_year_range(query)
    if time_range:
        set_slot(frame, SLOT_SPECS["document_attributes.time_range"]["path"], build_slot(time_range, "confirmed", source, 0.8))
    elif detect_slot_ambiguous(query, ["time", "year", "recent", "时间", "年份"]):
        set_slot(frame, SLOT_SPECS["document_attributes.time_range"]["path"], slot_mark_ambiguous("string", source))

    paper_type = ""
    if contains_any(lowered, ["survey", "review", "综述"]):
        paper_type = "survey"
    elif "benchmark" in lowered:
        paper_type = "benchmark"
    elif "theory" in lowered:
        paper_type = "theory"
    elif "analysis" in lowered:
        paper_type = "analysis"
    elif contains_any(lowered, ["method", "approach", "方法"]):
        paper_type = "method"
    if paper_type:
        set_slot(frame, SLOT_SPECS["document_attributes.paper_type"]["path"], build_slot(paper_type, "confirmed", source, 0.8))
    elif detect_slot_ambiguous(query, ["paper type", "论文类型", "综述", "benchmark"]):
        set_slot(frame, SLOT_SPECS["document_attributes.paper_type"]["path"], slot_mark_ambiguous("string", source))

    author_name = extract_author_name(query)
    if author_name and contains_any(lowered, ["author", "papers by", "作者", "的论文"]) or (
        author_name and len(query.split()) <= 4 and not contains_any(lowered, ["survey", "recent", "benchmark"])
    ):
        set_slot(frame, SLOT_SPECS["document_attributes.author_name"]["path"], build_slot(author_name, "confirmed", source, 0.88))
    elif detect_slot_ambiguous(query, ["author", "作者"]):
        set_slot(frame, SLOT_SPECS["document_attributes.author_name"]["path"], slot_mark_ambiguous("string", source))

    title_hint = extract_title_hint(query)
    if title_hint and len(title_hint.split()) >= 2:
        set_slot(frame, SLOT_SPECS["document_attributes.title_hint"]["path"], build_slot(title_hint, "confirmed", source, 0.88))
    elif detect_slot_ambiguous(query, ["title", "标题"]):
        set_slot(frame, SLOT_SPECS["document_attributes.title_hint"]["path"], slot_mark_ambiguous("string", source))

    if contains_any(lowered, ["recent", "latest", "最近", "最新"]):
        set_slot(frame, SLOT_SPECS["result_preferences.prefer_recent"]["path"], build_slot("yes", "confirmed", source, 0.9))
    elif contains_any(lowered, ["不要最新", "not recent"]):
        set_slot(frame, SLOT_SPECS["result_preferences.prefer_recent"]["path"], build_slot("no", "confirmed", source, 0.82))
    elif detect_slot_ambiguous(query, ["recent", "latest", "时间"]):
        set_slot(frame, SLOT_SPECS["result_preferences.prefer_recent"]["path"], slot_mark_ambiguous("string", source))

    if contains_any(lowered, ["classic", "foundational", "seminal", "经典", "奠基"]):
        set_slot(frame, SLOT_SPECS["result_preferences.prefer_classic"]["path"], build_slot("yes", "confirmed", source, 0.88))

    if contains_any(lowered, ["survey", "review", "综述"]):
        set_slot(frame, SLOT_SPECS["result_preferences.prefer_survey"]["path"], build_slot("yes", "confirmed", source, 0.88))
    elif contains_any(lowered, ["不要综述", "not survey"]):
        set_slot(frame, SLOT_SPECS["result_preferences.prefer_survey"]["path"], build_slot("no", "confirmed", source, 0.82))

    if contains_any(lowered, ["diverse", "broad", "多样", "多元", "多一些"]):
        set_slot(frame, SLOT_SPECS["result_preferences.prefer_diverse"]["path"], build_slot("yes", "confirmed", source, 0.85))

    if contains_any(lowered, ["explain", "why", "reason", "解释", "理由", "为什么", "explainable"]):
        set_slot(
            frame,
            SLOT_SPECS["result_preferences.need_explainable_reason"]["path"],
            build_slot("yes", "confirmed", source, 0.92),
        )
    elif contains_any(lowered, ["不用解释", "不要解释", "no need explain"]):
        set_slot(
            frame,
            SLOT_SPECS["result_preferences.need_explainable_reason"]["path"],
            build_slot("no", "confirmed", source, 0.86),
        )

    if infer_defaults:
        if get_slot(frame, SLOT_SPECS["document_attributes.author_name"]["path"])["status"] == "confirmed":
            set_slot(frame, SLOT_SPECS["search_scene"]["path"], build_slot("author_trace", "confirmed", source, 0.92))
        elif get_slot(frame, SLOT_SPECS["document_attributes.title_hint"]["path"])["status"] == "confirmed":
            set_slot(frame, SLOT_SPECS["search_scene"]["path"], build_slot("specific_paper_lookup", "confirmed", source, 0.9))
        elif contains_any(lowered, ["survey", "review", "综述"]):
            set_slot(frame, SLOT_SPECS["search_scene"]["path"], build_slot("survey_lookup", "confirmed", source, 0.86))
        elif contains_any(lowered, ["recent", "latest", "最近", "最新"]):
            set_slot(frame, SLOT_SPECS["search_scene"]["path"], build_slot("recent_progress", "confirmed", source, 0.84))
        elif get_slot(frame, SLOT_SPECS["technical_constraints.method"]["path"])["status"] == "confirmed":
            set_slot(
                frame,
                SLOT_SPECS["search_scene"]["path"],
                build_slot("method_constrained_search", "confirmed", source, 0.8),
            )
    return finalize_intent_frame(frame)


def merge_intent_frames(prior_frame: Dict[str, Any], delta_frame: Dict[str, Any], reply_text: str) -> Dict[str, Any]:
    merged = copy.deepcopy(finalize_intent_frame(prior_frame))
    prior_scene_slot = get_slot(merged, SLOT_SPECS["search_scene"]["path"])
    prior_scene = clean_text(prior_scene_slot.get("value")) if prior_scene_slot.get("status") == "confirmed" else ""
    delta = finalize_intent_frame(delta_frame)

    if global_ambiguous_reply(reply_text):
        for path_name in list(merged.get("missing_slots", [])):
            spec = SLOT_SPECS[path_name]
            set_slot(merged, spec["path"], slot_mark_ambiguous(spec["kind"], "follow_up_reply"))
        return finalize_intent_frame(merged)

    for path_name, spec in SLOT_SPECS.items():
        new_slot = get_slot(delta, spec["path"])
        if new_slot["status"] == "missing":
            continue
        if spec["kind"] == "list":
            old_slot = get_slot(merged, spec["path"])
            merged_list = clean_string_list(list(old_slot.get("value", [])) + list(new_slot.get("value", [])), limit=8)
            combined_slot = copy.deepcopy(new_slot)
            combined_slot["value"] = merged_list
            combined_slot["source"] = "merged"
            combined_slot["confidence"] = max(old_slot.get("confidence", 0.0), new_slot.get("confidence", 0.0))
            if not merged_list and combined_slot["status"] == "confirmed":
                combined_slot["status"] = "missing"
            set_slot(merged, spec["path"], combined_slot)
            continue
        set_slot(merged, spec["path"], new_slot)
    preserve_prior_scene(merged, prior_scene)
    return finalize_intent_frame(merged)


def parse_intent_with_llm(
    user_text: str,
    prior_frame: Optional[Dict[str, Any]] = None,
    mode: str = "initial",
) -> Tuple[Dict[str, Any], str]:
    cache_path = intent_query_cache_path(user_text=user_text, prior_frame=prior_frame, mode=mode)
    cached_payload = None
    if cache_path.exists():
        try:
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached_payload = None
    if isinstance(cached_payload, dict) and isinstance(cached_payload.get("intent_frame"), dict):
        cached_frame = finalize_intent_frame(cached_payload["intent_frame"])
        cached_frame = apply_heuristic_slot_completion(user_text, cached_frame)
        cached_frame = ensure_query_groups_from_user_text(cached_frame, user_text)
        if has_effective_query_groups(cached_frame):
            if cached_frame != cached_payload.get("intent_frame"):
                write_json(
                    cache_path,
                    {"used_model": cached_payload.get("used_model") or OPENAI_MODEL, "intent_frame": cached_frame},
                )
            return cached_frame, cached_payload.get("used_model") or OPENAI_MODEL

    messages = build_messages(user_text=user_text, prior_frame=prior_frame, mode=mode)
    raw_frame, used_model = structured_chat_completion(
        messages=messages,
        schema_name="intent_frame",
        schema=INTENT_FRAME_SCHEMA,
        model=OPENAI_MODEL,
        temperature=0.1,
        max_tokens=2200,
        timeout=90,
        api_key=OPENAI_API_KEY,
    )
    if prior_frame is None:
        final_frame = finalize_intent_frame(raw_frame)
    else:
        final_frame = merge_intent_frames(prior_frame, raw_frame, user_text)
    final_frame = apply_heuristic_slot_completion(user_text, final_frame)
    final_frame = ensure_query_groups_from_user_text(final_frame, user_text)
    write_json(cache_path, {"used_model": used_model, "intent_frame": final_frame})
    return final_frame, used_model


def parse_intent_frame(user_text: str) -> Tuple[Dict[str, Any], Optional[str], str]:
    fast_frame = heuristic_parse_text(user_text, source="heuristic_fast_parse", infer_defaults=True)
    fast_frame = ensure_query_groups_from_user_text(fast_frame, user_text)
    if has_meaningful_slots(fast_frame):
        return fast_frame, None, "heuristic_fast_parse"

    if not can_use_openai():
        fallback_frame = heuristic_parse_text(user_text, source="heuristic_fallback", infer_defaults=True)
        fallback_frame = ensure_query_groups_from_user_text(fallback_frame, user_text)
        append_error_log(
            {
                "mode": "initial",
                "user_text": user_text,
                "error": OPENAI_RUNTIME_MESSAGE,
                "parser": "heuristic_fallback",
            }
        )
        return fallback_frame, None, "heuristic_fallback"
    try:
        frame, used_model = parse_intent_with_llm(user_text, prior_frame=None, mode="initial")
        return frame, used_model, "llm"
    except Exception as exc:
        append_error_log({"mode": "initial", "user_text": user_text, "error": str(exc)})
        fallback_frame = heuristic_parse_text(user_text, source="heuristic_fallback", infer_defaults=True)
        fallback_frame = ensure_query_groups_from_user_text(fallback_frame, user_text)
        return fallback_frame, None, "heuristic_fallback"


def merge_follow_up_reply(prior_frame: Dict[str, Any], reply_text: str) -> Tuple[Dict[str, Any], Optional[str], str]:
    prior_normalized = finalize_intent_frame(prior_frame)
    fast_delta = heuristic_parse_text(reply_text, source="follow_up_fastpath", infer_defaults=False)
    if has_meaningful_slots(fast_delta):
        fast_frame = merge_intent_frames(prior_normalized, fast_delta, reply_text)
        fast_frame = ensure_query_groups_from_user_text(fast_frame, reply_text)
        if fast_frame != prior_normalized:
            return fast_frame, None, "heuristic_fast_merge"

    if not can_use_openai():
        delta = heuristic_parse_text(reply_text, source="heuristic_fallback", infer_defaults=False)
        fallback_frame = merge_intent_frames(prior_normalized, delta, reply_text)
        fallback_frame = ensure_query_groups_from_user_text(fallback_frame, reply_text)
        append_error_log(
            {
                "mode": "follow_up_merge",
                "reply_text": reply_text,
                "error": OPENAI_RUNTIME_MESSAGE,
                "parser": "heuristic_fallback",
            }
        )
        return fallback_frame, None, "heuristic_fallback"
    try:
        frame, used_model = parse_intent_with_llm(reply_text, prior_frame=prior_normalized, mode="follow_up_merge")
        return frame, used_model, "llm"
    except Exception as exc:
        append_error_log({"mode": "follow_up_merge", "reply_text": reply_text, "error": str(exc)})
        delta = heuristic_parse_text(reply_text, source="heuristic_fallback", infer_defaults=False)
        fallback_frame = merge_intent_frames(prior_normalized, delta, reply_text)
        fallback_frame = ensure_query_groups_from_user_text(fallback_frame, reply_text)
        return fallback_frame, None, "heuristic_fallback"


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def save_intent_frame(db_path: Path, query_text: str, intent_frame: Dict[str, Any]) -> int:
    with connect_db(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO search_history (query_text, intent_frame_json)
            VALUES (?, ?)
            """,
            (query_text, json.dumps(intent_frame, ensure_ascii=False)),
        )
        conn.commit()
        history_id = int(cursor.lastrowid)
    write_json(intent_session_cache_path(history_id), {"history_id": history_id, "query_text": query_text, "intent_frame": intent_frame})
    return history_id


def load_intent_frame(db_path: Path, history_id: int) -> Dict[str, Any]:
    with connect_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT intent_frame_json
            FROM search_history
            WHERE id = ?
            """,
            (history_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Intent history not found: {history_id}")
    return finalize_intent_frame(json.loads(row["intent_frame_json"]))


def load_search_history_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    with connect_db(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM search_history").fetchone()[0])


def build_pilot_payload(queries: Sequence[str]) -> List[Dict[str, Any]]:
    results = []
    for query in queries:
        frame, used_model, parser = parse_intent_frame(query)
        results.append(
            {
                "query": query,
                "parser": parser,
                "used_model": used_model,
                "intent_frame": frame,
            }
        )
    return results


def build_merge_examples_payload(examples: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    payload = []
    for item in examples:
        initial_frame, initial_model, initial_parser = parse_intent_frame(item["initial_query"])
        merged_frame, merged_model, merged_parser = merge_follow_up_reply(initial_frame, item["follow_up_reply"])
        payload.append(
            {
                "initial_query": item["initial_query"],
                "follow_up_reply": item["follow_up_reply"],
                "initial_parser": initial_parser,
                "initial_used_model": initial_model,
                "merged_parser": merged_parser,
                "merged_used_model": merged_model,
                "initial_intent_frame": initial_frame,
                "merged_intent_frame": merged_frame,
            }
        )
    return payload


def write_feedback(
    openai_available: bool,
    openai_message: str,
    pilot_payload: Sequence[Dict[str, Any]],
    test_payload: Sequence[Dict[str, Any]],
    merge_payload: Sequence[Dict[str, Any]],
) -> None:
    clarification_count = sum(1 for item in test_payload if item["intent_frame"].get("clarification_needed"))
    merge_clarification_count = sum(
        1 for item in merge_payload if item["merged_intent_frame"].get("clarification_needed")
    )
    content = f"""
意图理解流水线执行完成

项目目标
- 当前项目已具备统一 IntentFrame 意图理解层。
- 支持自然语言 query 解析、聚合追问、二轮回复合并和三路 query 生成。

运行信息
- OpenAI 可用: {openai_available}
- OpenAI 状态说明: {openai_message}
- pilot query 数量: {len(pilot_payload)}
- 测试 query 数量: {len(test_payload)}
- merge 样例数量: {len(merge_payload)}
- 测试集中仍需追问的 query 数量: {clarification_count}
- merge 后仍需追问的样例数量: {merge_clarification_count}

已完成能力
- 固定版 IntentFrame Prompt 已写入 intent_frame.md。
- 每个槽位均带 value / status / source / confidence。
- clarification_question 采用单轮聚合追问，不是一问一答式追问。
- follow-up reply 会与旧状态合并，而不是覆盖旧状态。
- coarse_queries / dense_queries / exact_queries 已同步生成。

交付物
- {PROMPT_PATH.name}
- {PILOT_OUTPUT_PATH.name}
- {TEST_OUTPUT_PATH.name}
- {MERGE_OUTPUT_PATH.name}
"""
    dump_text(FEEDBACK_PATH, content)


def build_intent_assets(
    queries: Optional[Sequence[str]] = None,
    merge_examples: Optional[Sequence[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    ensure_output_dir()
    write_prompt_file()
    openai_available = can_use_openai()
    openai_message = OPENAI_RUNTIME_MESSAGE

    query_list = list(queries or DEFAULT_INTENT_TEST_QUERIES)
    pilot_queries = query_list[:5]
    pilot_payload = build_pilot_payload(pilot_queries)
    test_payload = build_pilot_payload(query_list)
    merge_payload = build_merge_examples_payload(merge_examples or DEFAULT_MERGE_EXAMPLES)

    dump_json(PILOT_OUTPUT_PATH, pilot_payload)
    dump_json(TEST_OUTPUT_PATH, test_payload)
    dump_json(MERGE_OUTPUT_PATH, merge_payload)
    write_feedback(openai_available, openai_message, pilot_payload, test_payload, merge_payload)

    return {
        "openai_available": openai_available,
        "openai_message": openai_message,
        "pilot_count": len(pilot_payload),
        "test_query_count": len(test_payload),
        "merge_example_count": len(merge_payload),
        "output_dir": str(OUTPUT_DIR),
    }
