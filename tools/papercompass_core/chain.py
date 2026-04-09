"""
PaperCompass 核心方法主链路。

负责把前四项能力串起来：
query -> 意图解析 -> 聚合追问 -> 三路检索 -> gap 分析 -> 意图重排 -> 结果解释
"""

from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import intent, retrieval, semantic
from .config import (
    CHAIN_ERRORS_PATH,
    DEMO_RUNS_PATH,
    DEMO_WALKTHROUGH_PATH,
    EXPLANATION_SAMPLES_PATH,
    GAP_REPORTS_PATH,
    LEGACY_QUERY_PAPER_MATCH_PROMPT_PATH,
    PROJECT_ROOT,
    QUERY_PAPER_MATCH_PROMPT_PATH,
    RANKING_EVAL_PATH,
    REGRESSION_REPORT_PATH,
    STANDARD_QUERIES_PATH,
    SYSTEM_OUTPUT_DIR,
    ensure_system_layout,
    query_match_cache_path,
    relative_to_project,
    write_json,
)
from .llm import OPENAI_API_KEY, OPENAI_MODEL, OpenAIAPIError, structured_chat_completion, test_openai_api
OUTPUT_DIR = SYSTEM_OUTPUT_DIR
EXPLANATION_PROMPT_PATH = QUERY_PAPER_MATCH_PROMPT_PATH
LEGACY_EXPLANATION_PROMPT_PATH = LEGACY_QUERY_PAPER_MATCH_PROMPT_PATH
CHAIN_DEMOS_PATH = DEMO_RUNS_PATH
RANK_RESULTS_PATH = RANKING_EVAL_PATH
FEEDBACK_PATH = SYSTEM_OUTPUT_DIR / "eval" / "chain_feedback.txt"
ERROR_LOG_PATH = CHAIN_ERRORS_PATH

# 召回融合和意图打分的权重集中放在这里，便于统一调参。
FUSION_WEIGHTS = {"sparse": 0.45, "dense": 0.35, "exact": 0.20}
INTENT_SCORE_WEIGHTS = {
    "scene_match": 0.12,
    "topic_match": 0.45,
    "constraint_match": 0.25,
    "paper_type_match": 0.08,
    "time_preference_match": 0.06,
    "survey_preference_match": 0.04,
}
DEFAULT_CANDIDATE_POOL_SIZE = 40
DEFAULT_TOP_K = 5
DEFAULT_EXPLAIN_LIMIT = 5
DEFAULT_EXPLANATION_WORKERS = 4
DEFAULT_QUERY_MATCH_BATCH_SIZE = 8
DEFAULT_QUERY_MATCH_SECTION_LIMIT = 2
DEFAULT_QUERY_MATCH_SNIPPET_LIMIT = 2
DEFAULT_LLM_MATCH_MIN_LIMIT = 18
DEFAULT_LLM_MATCH_SCORE_GAP = 0.06
DEFAULT_LLM_MATCH_NEIGHBOR_GAP = 0.025
DEFAULT_STANDARD_QUERY_SEMANTIC_TOP_N = 6
DEFAULT_STANDARD_QUERY_SEMANTIC_MIN_FREQUENCY = 2
QUERY_MATCH_CACHE_VERSION = "query_paper_match_llm_required_v4"
QUERY_MATCH_PROMPT_VERSION = "query_paper_match_v4"
OPENAI_RUNTIME_AVAILABLE: Optional[bool] = None
OPENAI_RUNTIME_MESSAGE = ""
DENSE_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
MAX_ERROR_LOG_ENTRIES = 500
DENSE_INDEX_CACHE_VERSION = "dense_index_v1"
DENSE_INDEX_DISK_CACHE_SUBDIR = "dense_indexes"
DENSE_INDEX_DISK_CACHE_KEEP_PER_DB = 3
STAGE_LABELS = {
    "intent_parse": "LLM 意图解析",
    "intent_follow_up_merge": "追问合并",
    "retrieval_sparse": "稀疏召回",
    "retrieval_dense": "稠密召回",
    "retrieval_exact": "精确召回",
    "retrieval_fusion": "候选融合",
    "candidate_rows_load": "候选详情加载",
    "semantic_card_backfill": "语义卡补全",
    "query_paper_match": "Query-Paper 匹配",
    "rerank_and_explain": "重排与解释",
    "gap_report": "Gap 分析",
    "follow_up_suggestion": "追问建议生成",
    "total": "总耗时",
}

DIMENSION_LABELS = {
    "scene_match": "场景匹配",
    "topic_match": "主题匹配",
    "constraint_match": "约束匹配",
    "paper_type_match": "论文类型匹配",
    "time_preference_match": "时间偏好匹配",
    "survey_preference_match": "综述偏好匹配",
    "scene": "场景匹配",
    "topic": "主题匹配",
    "constraint": "约束匹配",
    "constraints": "约束匹配",
    "paper_type": "论文类型匹配",
    "paper type": "论文类型匹配",
    "time_preference": "时间偏好匹配",
    "time preference": "时间偏好匹配",
    "survey_preference": "综述偏好匹配",
    "survey preference": "综述偏好匹配",
}

SLOT_PATH_LABELS = {
    "search_scene": "检索场景",
    "research_topic.domain": "研究领域",
    "research_topic.task": "研究任务",
    "research_topic.problem": "研究问题",
    "research_topic.keywords": "主题关键词",
    "technical_constraints.method": "方法约束",
    "technical_constraints.model_family": "模型家族约束",
    "technical_constraints.dataset": "数据集约束",
    "technical_constraints.metric": "指标约束",
    "technical_constraints.modality": "模态约束",
    "document_attributes.time_range": "时间范围",
    "document_attributes.paper_type": "论文类型",
    "document_attributes.author_name": "作者",
    "document_attributes.title_hint": "标题线索",
    "result_preferences.prefer_recent": "偏好最新",
    "result_preferences.prefer_classic": "偏好经典",
    "result_preferences.prefer_survey": "偏好综述",
    "result_preferences.prefer_diverse": "偏好多样结果",
    "result_preferences.need_explainable_reason": "需要可解释理由",
}

FOLLOW_UP_TIME_RANGE_LABELS = {
    "recent": "最近",
    "last 2 years": "最近两年",
    "last 3 years": "最近三年",
    "classic": "经典时期",
}

FOLLOW_UP_SEARCH_SCENE_LABELS = {
    "topic_exploration": "主题探索",
    "survey_lookup": "综述检索",
    "recent_progress": "近期进展",
    "specific_paper_lookup": "特定论文定位",
    "author_trace": "作者追踪",
    "method_constrained_search": "方法约束检索",
}

FOLLOW_UP_PAPER_TYPE_LABELS = {
    "survey": "综述",
    "benchmark": "基准/评测",
    "method": "方法论文",
    "empirical_study": "实证研究",
    "application_study": "应用研究",
    "theory": "理论研究",
    "analysis": "分析论文",
}

FOLLOW_UP_SUGGESTION_SYSTEM_PROMPT = """You generate the next Chinese follow-up reply for an academic paper search system.

The reply will be shown to the user as a ready-to-submit suggestion.

Your job is to infer what the user truly wants from the full search context, not to mechanically paraphrase a fallback draft.
Treat the original query and the latest user follow-up reply as the strongest signals. Use the intent snapshots, gap report,
result signal summary, and top result mismatch summaries to decide what single follow-up would most improve retrieval focus.

Requirements:
1. Write exactly one direct Chinese follow-up reply, not a question.
2. Preserve stable confirmed intent. If the latest user reply clearly overrides an earlier preference, follow the latest explicit preference.
3. Strengthen only the constraints that are still missing, ambiguous, or contradicted by current results.
4. If paper type mismatch or main-intent mismatch is the dominant problem, make that constraint explicit and hard.
5. If the query gap is small but evidence gap remains large, sharpen the semantic target instead of repeating generic phrases like 不限.
6. Prefer concrete, user-facing wording such as 研究领域、研究任务、研究问题、论文类型、时间范围、模态、以及是否解释推荐理由.
7. Keep it concise, usually one semicolon-separated sentence within 120 Chinese characters.
8. Do not mention internal field names, JSON, schema, ranking score, Top-K, cache, prompt, or model names.
9. Use fallback_draft_reference only as a weak backup when richer context is insufficient.
10. The rationale must be one short Chinese sentence explaining what uncertainty or mismatch this follow-up is trying to fix.

Return JSON only."""

FOLLOW_UP_SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "follow_up_reply": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["follow_up_reply", "rationale"],
    "additionalProperties": False,
}

STANDARD_QUERY_SPECS = [
    {
        "query": "检索 RAG 综述论文",
        "follow_up_reply": "时间范围 2023-2026；聚焦 RAG 幻觉缓解方向；论文类型以综述为主；模型家族、数据集、指标不限；仅文本模态；偏好多样结果否；并解释每篇论文为何匹配。",
        "expected_intent_slots": {
            "search_scene": "survey_lookup",
            "research_topic.task": "retrieval-augmented generation",
            "research_topic.problem": "hallucination mitigation",
            "document_attributes.paper_type": "survey",
            "result_preferences.prefer_diverse": "no",
            "result_preferences.need_explainable_reason": "yes",
        },
        "expected_clarification_focus": [],
        "expected_top_result_type": "survey",
    },
    {
        "query": "最近的 agent memory 论文",
        "follow_up_reply": "时间范围 2023-2026；关注 LLM Agent 长期记忆机制；论文类型方法/基准优先；模型家族、数据集和指标不限；仅文本模态；作者不限；标题线索不限；偏好综述否；偏好多样结果是；并解释命中理由。",
        "expected_intent_slots": {
            "search_scene": "recent_progress",
            "research_topic.problem": "memory mechanism",
            "document_attributes.paper_type": "method",
            "result_preferences.prefer_survey": "no",
            "result_preferences.prefer_diverse": "yes",
        },
        "expected_clarification_focus": [],
        "expected_top_result_type": "method",
    },
    {
        "query": "找 MALT 作者的论文",
        "follow_up_reply": "MALT 指 Mechanistic Ablation of Lossy Translation；优先该论文作者后续相关工作；时间范围 2023-2026；论文类型 method 与 analysis；偏好最新，不偏好经典，不要求综述；并解释它们之间的关联。",
        "expected_intent_slots": {
            "search_scene": "author_trace",
            "document_attributes.time_range": "2023-2026",
            "document_attributes.paper_type": "method",
        },
        "expected_clarification_focus": [],
        "expected_top_result_type": "author_trace",
    },
    {
        "query": "用 COMET 做质量估计的论文",
        "follow_up_reply": "时间范围 2023-2026；聚焦 COMET 在质量估计中的使用；数据集不限；作者不限；偏好最新，不偏好经典；偏好综述否；偏好多样结果是；需要可解释理由；并说明每篇与 COMET QE 的关系。",
        "expected_intent_slots": {
            "search_scene": "method_constrained_search",
            "technical_constraints.metric": "COMET",
            "result_preferences.need_explainable_reason": "yes",
            "result_preferences.prefer_survey": "no",
        },
        "expected_clarification_focus": [],
        "expected_top_result_type": "method",
    },
    {
        "query": "长上下文论文进展",
        "follow_up_reply": "时间范围 2023-2026；主题聚焦长上下文建模；论文类型不限（可包含综述）；方法约束不限；模型家族、数据集、指标不限；仅文本模态；作者不限；偏好多样结果否；需要可解释理由否。",
        "expected_intent_slots": {
            "search_scene": "recent_progress",
            "document_attributes.time_range": "2023-2026",
            "result_preferences.prefer_diverse": "no",
            "result_preferences.need_explainable_reason": "no",
        },
        "expected_clarification_focus": [],
        "expected_top_result_type": "method",
    },
    {
        "query": "大语言模型推理论文（benchmark 优先）",
        "follow_up_reply": "时间范围 2023-2026；任务聚焦推理评测；benchmark 优先但不限；方法、模型家族、作者、标题线索不限；指标可包含 Pass@1/GSM8K/MATH；偏好多样结果否；需要可解释理由否。",
        "expected_intent_slots": {
            "research_topic.task": "reasoning",
            "document_attributes.paper_type": "benchmark",
            "document_attributes.time_range": "2023-2026",
            "result_preferences.prefer_diverse": "no",
            "result_preferences.need_explainable_reason": "no",
        },
        "expected_clarification_focus": [],
        "expected_top_result_type": "benchmark",
    },
    {
        "query": "multimodal reasoning papers",
        "follow_up_reply": "prefer diverse results",
        "expected_intent_slots": {
            "research_topic.domain": "multimodal NLP",
            "research_topic.task": "reasoning",
        },
        "expected_clarification_focus": ["document_attributes.paper_type"],
        "expected_top_result_type": "method",
    },
    {
        "query": "Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey",
        "expected_intent_slots": {
            "search_scene": "specific_paper_lookup",
        },
        "expected_clarification_focus": [],
        "expected_top_result_type": "specific_paper_lookup",
    },
    {
        "query": "early exit for quality estimation",
        "expected_intent_slots": {
            "technical_constraints.method": "early exit",
            "research_topic.task": "quality estimation",
        },
        "expected_clarification_focus": ["document_attributes.time_range"],
        "expected_top_result_type": "method",
    },
    {
        "query": "translation quality estimation explainable reason",
        "expected_intent_slots": {
            "research_topic.task": "quality estimation",
            "result_preferences.need_explainable_reason": "yes",
        },
        "expected_clarification_focus": ["technical_constraints.dataset", "document_attributes.time_range"],
        "expected_top_result_type": "method",
    },
]

METHOD_LIKE_PAPER_TYPES = {"method", "empirical_study", "application_study", "analysis"}

EXPLANATION_SYSTEM_PROMPT = """You judge whether a candidate paper truly matches the user's academic retrieval intent.

Use only the provided intent frame, semantic card, matched snippets, and retrieval signals.
Do not invent evidence. Return JSON only.

Requirements:
0. Topic specificity dominates. A paper that is broader, adjacent, or only loosely related to the requested topic must not receive a high score just because it matches paper type, recency, or retrieval score.
1. All natural-language output fields must be in Simplified Chinese.
2. `brief_reason` must be concise Chinese in 1-2 sentences.
3. `matched_dimensions` and `unmet_dimensions` should prefer short Chinese human-readable phrases; if you use system dimension ids, only use:
   scene_match, topic_match, constraint_match, paper_type_match, time_preference_match, survey_preference_match.
4. `match_score` should reflect semantic fit to the query intent, not lexical overlap alone.
5. `evidence_sufficiency` should reflect whether the provided evidence is enough to justify the recommendation.
6. If the paper misses the user's core topic, task, or problem, set `main_intent_satisfied=false`, mention the missing topical focus in `unmet_dimensions`, and keep `match_score` conservative.
7. Survey match, paper-type match, or recency match alone must not outweigh topic drift.
8. If the paper is strongly on the requested topic but misses only a paper-type or preference requirement, keep `main_intent_satisfied=false` but preserve a moderate or high `match_score`; reserve very low scores for true topic drift.
9. If the user asks for explanations, rationales, interpretability, or explainable reasons, papers that only use a QE metric, benchmark, confidence score, or evaluation dataset without producing interpretable reasons do not satisfy the main intent.
"""

EXPLANATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "main_intent_satisfied": {"type": "boolean"},
        "matched_dimensions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
        "unmet_dimensions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
        "match_score": {"type": "number"},
        "evidence_sufficiency": {"type": "number"},
        "brief_reason": {"type": "string"},
    },
    "required": [
        "main_intent_satisfied",
        "matched_dimensions",
        "unmet_dimensions",
        "match_score",
        "evidence_sufficiency",
        "brief_reason",
    ],
}

EXPLANATION_BATCH_SYSTEM_PROMPT = """You judge whether each candidate paper truly matches the user's academic retrieval intent.

Use only the provided intent frame, semantic card, matched snippets, and retrieval signals.
Do not invent evidence. Return JSON only.

Requirements:
0. Topic specificity dominates. A paper that is broader, adjacent, or only loosely related to the requested topic must not receive a high score just because it matches paper type, recency, or retrieval score.
1. Evaluate every paper independently.
2. Return exactly one result object per input `paper_id`.
3. Copy each `paper_id` exactly as provided.
4. All natural-language output fields must be in Simplified Chinese.
5. `brief_reason` must be concise Chinese in 1-2 sentences.
6. `matched_dimensions` and `unmet_dimensions` should prefer short Chinese human-readable phrases; if you use system dimension ids, only use:
   scene_match, topic_match, constraint_match, paper_type_match, time_preference_match, survey_preference_match.
7. `match_score` should reflect semantic fit to the query intent, not lexical overlap alone.
8. `evidence_sufficiency` should reflect whether the provided evidence is enough to justify the recommendation.
9. If the paper misses the user's core topic, task, or problem, set `main_intent_satisfied=false`, mention the missing topical focus in `unmet_dimensions`, and keep `match_score` conservative.
10. Survey match, paper-type match, or recency match alone must not outweigh topic drift.
11. If the paper is strongly on the requested topic but misses only a paper-type or preference requirement, keep `main_intent_satisfied=false` but preserve a moderate or high `match_score`; reserve very low scores for true topic drift.
12. If the user asks for explanations, rationales, interpretability, or explainable reasons, papers that only use a QE metric, benchmark, confidence score, or evaluation dataset without producing interpretable reasons do not satisfy the main intent.
"""


# 单条 query-paper 匹配结果的结构化输出 schema。
def build_query_paper_match_item_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "paper_id": {"type": "string"},
            "main_intent_satisfied": {"type": "boolean"},
            "matched_dimensions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "unmet_dimensions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "match_score": {"type": "number"},
            "evidence_sufficiency": {"type": "number"},
            "brief_reason": {"type": "string"},
        },
        "required": [
            "paper_id",
            "main_intent_satisfied",
            "matched_dimensions",
            "unmet_dimensions",
            "match_score",
            "evidence_sufficiency",
            "brief_reason",
        ],
    }


# 批量匹配时根据批大小动态生成整体 schema。
def build_query_paper_match_batch_schema(batch_size: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": build_query_paper_match_item_schema(),
                "minItems": 1,
                "maxItems": max(1, batch_size),
            }
        },
        "required": ["results"],
    }


# 写入提示词、演示和评估产物前统一创建目录。
def ensure_output_dir() -> None:
    ensure_system_layout()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


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


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


# 把内部维度标识转换成面向用户展示的中文标签。
def localize_dimension_label(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    lowered = text.lower()
    return DIMENSION_LABELS.get(text) or DIMENSION_LABELS.get(lowered) or text


def localize_slot_path(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return SLOT_PATH_LABELS.get(text, text)


def localize_user_label(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if text in SLOT_PATH_LABELS:
        return SLOT_PATH_LABELS[text]
    lowered = text.lower()
    if lowered in DIMENSION_LABELS:
        return DIMENSION_LABELS[lowered]
    if text in DIMENSION_LABELS:
        return DIMENSION_LABELS[text]
    if text.startswith("Matched dimension:"):
        _, _, dimension = text.partition(":")
        return "命中维度：" + localize_dimension_label(dimension)
    if text.startswith("命中维度："):
        _, _, dimension = text.partition("：")
        return "命中维度：" + localize_dimension_label(dimension)
    return text


def localize_user_label_list(values: Iterable[Any], limit: int = 8) -> List[str]:
    return clean_string_list((localize_user_label(value) for value in values), limit=limit)


def build_match_reason_fallback(matched_dimensions: Sequence[str], main_intent_satisfied: bool) -> str:
    localized_dimensions = localize_user_label_list(matched_dimensions, limit=3)
    if localized_dimensions:
        prefix = "该论文命中维度："
        suffix = "；整体满足主意图。" if main_intent_satisfied else "；整体仅部分匹配主意图。"
        return prefix + "；".join(localized_dimensions) + suffix
    if main_intent_satisfied:
        return "该论文与当前查询整体一致，现有证据足以支持推荐。"
    return "该论文有一定相关性，但匹配证据仍然有限。"


def coerce_query_match_score(value: Any, default: float = 0.0) -> float:
    text = clean_text(value).lower()
    if not text:
        return clamp_score(default)
    try:
        return clamp_score(float(text))
    except (TypeError, ValueError):
        pass

    score_aliases = {
        "very high": 0.95,
        "high": 0.85,
        "strong": 0.85,
        "good": 0.72,
        "sufficient": 0.72,
        "medium": 0.55,
        "moderate": 0.55,
        "partial": 0.5,
        "mixed": 0.45,
        "low": 0.25,
        "weak": 0.25,
        "insufficient": 0.2,
        "poor": 0.15,
        "none": 0.0,
        "no": 0.0,
    }
    return clamp_score(score_aliases.get(text, default))


def coerce_query_match_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"true", "yes", "y", "1", "high", "strong", "good", "sufficient", "matched", "satisfied"}:
        return True
    if text in {"false", "no", "n", "0", "low", "weak", "insufficient", "none", "partial", "mixed"}:
        return False
    return bool(value)


def has_dimension_marker(values: Sequence[str], marker: str) -> bool:
    return any(marker in clean_text(value) for value in values)


# 统一清洗模型返回的 query-paper 匹配结果，保证下游字段稳定。
def normalize_query_paper_match_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    main_intent_satisfied = coerce_query_match_bool(raw_payload.get("main_intent_satisfied"))
    matched_dimensions = [
        value
        for value in localize_user_label_list(raw_payload.get("matched_dimensions", []), limit=4)
        if contains_chinese(value)
    ]
    unmet_dimensions = [
        value
        for value in localize_user_label_list(raw_payload.get("unmet_dimensions", []), limit=4)
        if contains_chinese(value)
    ]
    if not matched_dimensions and main_intent_satisfied:
        matched_dimensions = ["主题匹配"]
    if not unmet_dimensions and not main_intent_satisfied:
        unmet_dimensions = ["主意图未满足"]

    brief_reason = clean_text(raw_payload.get("brief_reason", ""))
    if not brief_reason or not contains_chinese(brief_reason):
        brief_reason = build_match_reason_fallback(matched_dimensions, main_intent_satisfied)
    match_score = coerce_query_match_score(raw_payload.get("match_score", 0.0))
    evidence_sufficiency = coerce_query_match_score(raw_payload.get("evidence_sufficiency", 0.0))

    # 如果模型一边声称满足主意图，一边又把“主意图未满足”列为缺口，优先按否定判断修复。
    if main_intent_satisfied and has_dimension_marker(unmet_dimensions, "主意图"):
        main_intent_satisfied = False

    if not main_intent_satisfied:
        match_score = min(match_score, 0.78)
        if has_dimension_marker(unmet_dimensions, "论文类型") or has_dimension_marker(unmet_dimensions, "主意图"):
            match_score = min(match_score, 0.68)
        evidence_sufficiency = min(evidence_sufficiency, 0.82)
        if not has_dimension_marker(unmet_dimensions, "主意图"):
            unmet_dimensions = clean_string_list(list(unmet_dimensions) + ["主意图未满足"], limit=4)
    else:
        # 对“满足主意图但分数异常偏低”的情况做一致性修复，避免布尔判断和得分冲突。
        match_score = max(match_score, 0.72)
        evidence_sufficiency = max(evidence_sufficiency, 0.58)
        unmet_dimensions = [value for value in unmet_dimensions if "主意图" not in clean_text(value)]

    return {
        "main_intent_satisfied": main_intent_satisfied,
        "matched_dimensions": matched_dimensions,
        "unmet_dimensions": unmet_dimensions,
        "match_score": match_score,
        "evidence_sufficiency": evidence_sufficiency,
        "brief_reason": brief_reason,
    }


def truncate_text(value: Any, max_chars: int = 320) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def compact_json_value(
    value: Any,
    *,
    max_depth: int = 2,
    max_items: int = 6,
    max_chars: int = 240,
) -> Any:
    if max_depth <= 0:
        if isinstance(value, (dict, list)):
            return [] if isinstance(value, list) else {}
        return truncate_text(value, max_chars=max_chars)

    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            compacted[clean_text(key)] = compact_json_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_chars=max_chars,
            )
        return compacted

    if isinstance(value, list):
        return [
            compact_json_value(item, max_depth=max_depth - 1, max_items=max_items, max_chars=max_chars)
            for item in value[:max_items]
        ]

    if isinstance(value, str):
        return truncate_text(value, max_chars=max_chars)
    return value


def build_query_paper_match_paper_payload(
    evidence_pack: Dict[str, Any],
    rank_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "paper_id": rank_result["paper_id"],
        "title": truncate_text(rank_result.get("title", ""), max_chars=240),
        "authors": truncate_text(rank_result.get("authors_raw", ""), max_chars=240),
        "year_month": clean_text(rank_result.get("year_month", "")),
        "abstract": truncate_text(rank_result.get("abstract", ""), max_chars=1200),
        "semantic_card": compact_json_value(
            evidence_pack.get("semantic_card", {}),
            max_depth=2,
            max_items=6,
            max_chars=240,
        ),
        "matched_sections": [
            truncate_text(section, max_chars=180)
            for section in clean_string_list(
                evidence_pack.get("matched_sections", []),
                limit=DEFAULT_QUERY_MATCH_SECTION_LIMIT,
            )
        ],
        "matched_snippets": [
            {
                "field": clean_text(item.get("field", "")),
                "snippet": truncate_text(item.get("snippet", ""), max_chars=260),
            }
            for item in list(evidence_pack.get("matched_snippets", []))[:DEFAULT_QUERY_MATCH_SNIPPET_LIMIT]
        ],
        "retrieval_signals": {
            "base_score": round(float(rank_result.get("base_score", 0.0) or 0.0), 6),
            "preliminary_score": round(float(rank_result.get("preliminary_score", 0.0) or 0.0), 6),
            "retrieval_sources": clean_string_list(rank_result.get("retrieval_sources", []), limit=4),
            "matched_field": clean_text(rank_result.get("matched_field", "")),
            "exact_match_type": clean_text(rank_result.get("exact_match_type", "")),
        },
    }


# query-paper 匹配结果支持磁盘缓存，减少重复调用 LLM。
def load_cached_query_paper_match(
    intent_frame: Dict[str, Any],
    paper_id: str,
) -> Optional[Tuple[Dict[str, Any], Optional[str]]]:
    cache_path = query_match_cache_path(intent_frame, paper_id)
    if not cache_path.exists():
        return None
    try:
        cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            isinstance(cached_payload, dict)
            and cached_payload.get("cache_version") == QUERY_MATCH_CACHE_VERSION
            and cached_payload.get("prompt_version") == QUERY_MATCH_PROMPT_VERSION
            and cached_payload.get("generator") in {"llm_query_paper_match", "llm_query_paper_match_batch"}
            and isinstance(cached_payload.get("query_paper_match"), dict)
        ):
            normalized_payload = normalize_query_paper_match_payload(cached_payload["query_paper_match"])
            return normalized_payload, cached_payload.get("used_model")
    except Exception:
        return None
    return None


def write_cached_query_paper_match(
    intent_frame: Dict[str, Any],
    paper_id: str,
    payload: Dict[str, Any],
    used_model: Optional[str],
    *,
    generator: str,
) -> None:
    cache_path = query_match_cache_path(intent_frame, paper_id)
    write_json(
        cache_path,
        {
            "cache_version": QUERY_MATCH_CACHE_VERSION,
            "prompt_version": QUERY_MATCH_PROMPT_VERSION,
            "generator": generator,
            "used_model": used_model,
            "query_paper_match": payload,
        },
    )


def normalize_query_paper_match_batch_payload(
    raw_payload: Any,
    expected_paper_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    expected_ids = [clean_text(paper_id) for paper_id in expected_paper_ids if clean_text(paper_id)]
    expected_set = set(expected_ids)
    if isinstance(raw_payload, list):
        raw_results = raw_payload
    elif isinstance(raw_payload, dict):
        raw_results = raw_payload.get("results", raw_payload.get("items", []))
    else:
        raw_results = []
    if not isinstance(raw_results, list) or not raw_results:
        raise OpenAIAPIError("LLM query-paper match batch returned no results.")

    normalized_results: Dict[str, Dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        paper_id = clean_text(item.get("paper_id", ""))
        if not paper_id or paper_id not in expected_set or paper_id in normalized_results:
            continue
        normalized_results[paper_id] = normalize_query_paper_match_payload(item)

    missing_ids = [paper_id for paper_id in expected_ids if paper_id not in normalized_results]
    if missing_ids:
        raise OpenAIAPIError("LLM query-paper match batch omitted results for: " + ", ".join(missing_ids[:5]))
    return normalized_results


def chunk_rank_items(items: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    size = max(1, batch_size)
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def compute_llm_match_limit(
    ranked: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    explain_limit: int,
) -> int:
    if not ranked:
        return 0

    minimum_limit = min(
        len(ranked),
        max(top_k + 2, explain_limit + 1, DEFAULT_LLM_MATCH_MIN_LIMIT),
    )
    hard_limit = min(len(ranked), max(top_k * 3, explain_limit * 2, 8))
    if minimum_limit >= hard_limit:
        return minimum_limit

    limit = minimum_limit
    top_score = float(ranked[0].get("preliminary_score", 0.0) or 0.0)
    previous_score = float(ranked[minimum_limit - 1].get("preliminary_score", 0.0) or 0.0)
    for index in range(minimum_limit, hard_limit):
        current_score = float(ranked[index].get("preliminary_score", 0.0) or 0.0)
        if (top_score - current_score) <= DEFAULT_LLM_MATCH_SCORE_GAP or (
            previous_score - current_score
        ) <= DEFAULT_LLM_MATCH_NEIGHBOR_GAP:
            limit = index + 1
            previous_score = current_score
            continue
        break
    return limit


def comparable_text(value: Any) -> str:
    return clean_text(value).lower().replace("–", "-").replace("—", "-").replace("’", "'")


CANONICAL_SLOT_PATTERNS = {
    "retrieval_augmented_generation": (
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "rag",
        "knowledge-augmented text generation",
    ),
    "quality_estimation": (
        "quality estimation",
        "translation quality estimation",
    ),
    "reasoning": (
        "reasoning",
        "reasoning evaluation",
        "multimodal reasoning",
    ),
    "memory_mechanism": (
        "memory mechanism",
        "agent memory",
        "memory in agent systems",
        "memory construction",
        "long-term memory",
    ),
    "multimodal": (
        "multimodal",
        "multimodal nlp",
    ),
}


def canonical_slot_value(value: Any) -> str:
    text = comparable_text(value)
    if not text:
        return ""

    def _pattern_hit(pattern: str) -> bool:
        if len(pattern) <= 4 and pattern.isalpha():
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", text))
        return pattern in text

    for canonical, patterns in CANONICAL_SLOT_PATTERNS.items():
        if any(_pattern_hit(pattern) for pattern in patterns):
            return canonical
    return text


def token_overlap_ratio(left: Any, right: Any) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", comparable_text(left)))
    right_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", comparable_text(right)))
    left_tokens = {token for token in left_tokens if len(token) >= 2}
    right_tokens = {token for token in right_tokens if len(token) >= 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def slot_value_matches_expected(actual: Any, expected: Any) -> bool:
    expected_text = comparable_text(expected)
    if not expected_text:
        return False
    if isinstance(actual, list):
        return any(slot_value_matches_expected(item, expected) for item in actual)
    actual_text = comparable_text(actual)
    if not actual_text:
        return False
    if actual_text == expected_text:
        return True
    if canonical_slot_value(actual_text) == canonical_slot_value(expected_text):
        return True
    if (expected_text in actual_text or actual_text in expected_text) and min(len(expected_text), len(actual_text)) >= 4:
        return True
    return token_overlap_ratio(actual_text, expected_text) >= 0.6


def clamp_score(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, round(float(value), 6)))


def normalize_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    max_score = max(score_map.values())
    if max_score <= 0:
        return {paper_id: 0.0 for paper_id in score_map}
    return {paper_id: clamp_score(score / max_score) for paper_id, score in score_map.items()}


def tokenized_terms(text: str) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/+:-]*|[\u4e00-\u9fff]{2,}", text.lower())
    tokens: List[str] = []
    for raw_token in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", raw_token):
            tokens.append(raw_token)
            continue
        for token in re.split(r"[-_/+:.]+", raw_token):
            token = token.strip()
            if len(token) < 2:
                continue
            tokens.append(token)
    deduped = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def dense_tokens(text: str) -> List[str]:
    base_tokens = tokenized_terms(text)
    bigrams = [f"{base_tokens[index]} {base_tokens[index + 1]}" for index in range(len(base_tokens) - 1)]
    return base_tokens + bigrams


def can_use_openai() -> bool:
    global OPENAI_RUNTIME_AVAILABLE, OPENAI_RUNTIME_MESSAGE
    if OPENAI_RUNTIME_AVAILABLE is not None:
        return OPENAI_RUNTIME_AVAILABLE
    ok, message = test_openai_api(OPENAI_API_KEY)
    OPENAI_RUNTIME_AVAILABLE = ok
    OPENAI_RUNTIME_MESSAGE = message
    return ok


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


# 把当前 query-paper 匹配提示词和 schema 导出到文件。
def write_prompt_file() -> None:
    content = f"""# Query-Paper 匹配提示词

默认模型: {OPENAI_MODEL}

## 系统提示词
{EXPLANATION_SYSTEM_PROMPT}

## 输出 Schema
```json
{json.dumps(EXPLANATION_SCHEMA, ensure_ascii=False, indent=2)}
```
"""
    dump_text(EXPLANATION_PROMPT_PATH, content)
    if LEGACY_EXPLANATION_PROMPT_PATH != EXPLANATION_PROMPT_PATH:
        dump_text(LEGACY_EXPLANATION_PROMPT_PATH, content)
# 稠密索引缓存需要跟数据库内容绑定，避免错用旧索引。
def dense_index_cache_key(db_path: Path) -> str:
    resolved = db_path.resolve(strict=False)
    try:
        with retrieval.connect_db(resolved) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS paper_count,
                    MIN(paper_id) AS min_paper_id,
                    MAX(paper_id) AS max_paper_id,
                    SUM(LENGTH(embedding_text)) AS embedding_len_sum
                FROM papers
                """
            ).fetchone()
        signature = {
            "db_path": str(resolved),
            "paper_count": int(row["paper_count"] or 0),
            "min_paper_id": clean_text(row["min_paper_id"]),
            "max_paper_id": clean_text(row["max_paper_id"]),
            "embedding_len_sum": int(row["embedding_len_sum"] or 0),
            "version": DENSE_INDEX_CACHE_VERSION,
        }
    except Exception:
        stat = resolved.stat()
        signature = {
            "db_path": str(resolved),
            "db_size": int(stat.st_size),
            "db_mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            "version": DENSE_INDEX_CACHE_VERSION,
        }
    digest = hashlib.sha1(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:20]


def dense_index_cache_dir() -> Path:
    ensure_system_layout()
    path = SYSTEM_OUTPUT_DIR / "cache" / DENSE_INDEX_DISK_CACHE_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def dense_index_cache_path(db_path: Path, cache_key: str) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", db_path.stem)[:48] or "papercompass"
    return dense_index_cache_dir() / f"{safe_stem}_{cache_key}.pkl.gz"


def is_valid_dense_index(index: Any) -> bool:
    if not isinstance(index, dict):
        return False
    if not all(key in index for key in ("postings", "norms", "idf", "titles")):
        return False
    postings = index.get("postings")
    norms = index.get("norms")
    idf = index.get("idf")
    titles = index.get("titles")
    if not isinstance(postings, dict) or not isinstance(norms, dict) or not isinstance(idf, dict) or not isinstance(titles, dict):
        return False
    return True


def normalize_dense_index(index: Dict[str, Any]) -> Dict[str, Any]:
    postings_raw = index.get("postings", {})
    norms_raw = index.get("norms", {})
    idf_raw = index.get("idf", {})
    titles_raw = index.get("titles", {})
    postings: Dict[str, List[Tuple[str, float]]] = {}
    for token, pairs in postings_raw.items():
        token_key = clean_text(token)
        if not token_key:
            continue
        normalized_pairs: List[Tuple[str, float]] = []
        if isinstance(pairs, list):
            for item in pairs:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                paper_id = clean_text(item[0])
                if not paper_id:
                    continue
                try:
                    weight = float(item[1])
                except (TypeError, ValueError):
                    continue
                normalized_pairs.append((paper_id, weight))
        if normalized_pairs:
            postings[token_key] = normalized_pairs
    norms = {clean_text(k): float(v) for k, v in norms_raw.items() if clean_text(k)}
    idf = {clean_text(k): float(v) for k, v in idf_raw.items() if clean_text(k)}
    titles = {clean_text(k): clean_text(v) for k, v in titles_raw.items() if clean_text(k)}
    return {"postings": postings, "norms": norms, "idf": idf, "titles": titles}


def prune_dense_index_cache_files(db_path: Path) -> None:
    cache_dir = dense_index_cache_dir()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", db_path.stem)[:48] or "papercompass"
    pattern = f"{safe_stem}_*.pkl.gz"
    candidates = sorted(cache_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates[DENSE_INDEX_DISK_CACHE_KEEP_PER_DB:]:
        try:
            path.unlink()
        except OSError:
            continue


# 优先从磁盘恢复稠密索引，减少重复构建时间。
def load_dense_index_from_disk(db_path: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    cache_path = dense_index_cache_path(db_path, cache_key)
    if not cache_path.exists():
        return None
    try:
        with gzip.open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != DENSE_INDEX_CACHE_VERSION:
            return None
        if payload.get("cache_key") != cache_key:
            return None
        index = payload.get("index")
        if not is_valid_dense_index(index):
            return None
        return index
    except Exception as exc:
        append_error_log(
            {
                "stage": "dense_index_cache_load",
                "cache_path": str(cache_path),
                "error": str(exc),
            }
        )
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


# 将稠密索引持久化到磁盘，供后续查询复用。
def persist_dense_index_to_disk(db_path: Path, cache_key: str, index: Dict[str, Any]) -> None:
    cache_path = dense_index_cache_path(db_path, cache_key)
    payload = {"version": DENSE_INDEX_CACHE_VERSION, "cache_key": cache_key, "index": index}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f"{cache_path.stem}.",
            suffix=".tmp",
            dir=cache_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=3) as gzip_handle:
                pickle.dump(payload, gzip_handle, protocol=pickle.HIGHEST_PROTOCOL)
        if temp_path is not None:
            temp_path.replace(cache_path)
        prune_dense_index_cache_files(db_path)
    except Exception as exc:
        append_error_log(
            {
                "stage": "dense_index_cache_persist",
                "cache_path": str(cache_path),
                "error": str(exc),
            }
        )
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


# 根据数据库中的论文文本构建轻量稠密检索索引。
def build_dense_index(db_path: Path) -> Dict[str, Any]:
    cache_key = dense_index_cache_key(db_path)
    cached = DENSE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    disk_cached = load_dense_index_from_disk(db_path, cache_key)
    if disk_cached is not None:
        DENSE_INDEX_CACHE[cache_key] = disk_cached
        return disk_cached

    with retrieval.connect_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT paper_id, title, embedding_text
            FROM papers
            ORDER BY paper_id
            """
        ).fetchall()

    documents: Dict[str, Counter[str]] = {}
    doc_freq: Counter[str] = Counter()
    titles: Dict[str, str] = {}
    for row in rows:
        paper_id = row["paper_id"]
        titles[paper_id] = clean_text(row["title"])
        tokens = dense_tokens(clean_text(row["embedding_text"]))
        counter = Counter(tokens)
        documents[paper_id] = counter
        for token in counter.keys():
            doc_freq[token] += 1

    total_docs = max(len(documents), 1)
    idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1.0 for token, freq in doc_freq.items()}
    postings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    norms: Dict[str, float] = {}
    for paper_id, counter in documents.items():
        weighted = {token: (1.0 + math.log(freq)) * idf.get(token, 1.0) for token, freq in counter.items()}
        for token, token_weight in weighted.items():
            postings[token].append((paper_id, token_weight))
        norms[paper_id] = math.sqrt(sum(value * value for value in weighted.values())) or 1.0

    index = {"postings": postings, "norms": norms, "idf": idf, "titles": titles}
    persist_dense_index_to_disk(db_path, cache_key, index)
    DENSE_INDEX_CACHE[cache_key] = index
    if len(DENSE_INDEX_CACHE) > 4:
        oldest_key = next(iter(DENSE_INDEX_CACHE.keys()))
        if oldest_key != cache_key:
            DENSE_INDEX_CACHE.pop(oldest_key, None)
    return index


def dense_query_vector(query: str, idf: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    counter = Counter(dense_tokens(query))
    weighted = {token: (1.0 + math.log(freq)) * idf.get(token, 1.0) for token, freq in counter.items()}
    norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
    return weighted, norm


def cosine_similarity(query_vector: Dict[str, float], query_norm: float, doc_vector: Dict[str, float], doc_norm: float) -> float:
    shared = set(query_vector.keys()) & set(doc_vector.keys())
    if not shared:
        return 0.0
    dot = sum(query_vector[token] * doc_vector[token] for token in shared)
    return dot / (query_norm * doc_norm)


def load_paper_rows(conn: Any, paper_ids: Sequence[str]) -> Dict[str, Any]:
    if not paper_ids:
        return {}
    placeholders = ", ".join("?" for _ in paper_ids)
    rows = conn.execute(
        f"""
        SELECT
            papers.paper_id,
            papers.title,
            papers.authors_raw,
            papers.normalized_authors,
            papers.abstract,
            papers.section_titles,
            papers.embedding_text,
            papers.year_month,
            papers.intro_text,
            papers.methods_text,
            papers.results_text,
            papers.discussion_text,
            paper_semantic_cards.semantic_card_json
        FROM papers
        LEFT JOIN paper_semantic_cards
            ON papers.paper_id = paper_semantic_cards.paper_id
        WHERE papers.paper_id IN ({placeholders})
        ORDER BY papers.paper_id
        """,
        list(paper_ids),
    ).fetchall()
    return {row["paper_id"]: row for row in rows}


# 主链路会按需确保候选论文已经拥有语义卡片。
def ensure_semantic_cards_for_papers(db_path: Path, paper_ids: Sequence[str]) -> Dict[str, Any]:
    target_ids = clean_string_list(paper_ids, limit=max(len(paper_ids), 1))
    if not target_ids:
        return {}
    with retrieval.connect_db(db_path) as conn:
        for paper_id in target_ids:
            semantic.generate_card_for_paper(conn, paper_id, refresh=False)
        return load_paper_rows(conn, target_ids)


def parse_json_field(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(value) if value else fallback
    except Exception:
        return fallback


def infer_paper_type(row: Any, semantic_card: Dict[str, Any]) -> str:
    if semantic_card.get("paper_type"):
        return clean_text(semantic_card.get("paper_type"))
    text = f"{row['title']} {row['abstract']}".lower()
    if "survey" in text or "review" in text:
        return "survey"
    if "benchmark" in text:
        return "benchmark"
    if "theory" in text or "theorem" in text:
        return "theory"
    if "analysis" in text:
        return "analysis"
    if "application" in text or "case study" in text:
        return "application_study"
    if "method" in text or "approach" in text or "framework" in text:
        return "method"
    return "empirical_study"


def semantic_card_text(semantic_card: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for field_name in (
        "domain_tags",
        "task_tags",
        "problem_statement",
        "method_tags",
        "model_tags",
        "dataset_tags",
        "metric_tags",
        "core_contributions",
        "retrieval_keywords_en",
        "retrieval_keywords_zh",
        "survey_signals",
        "likely_user_intents",
    ):
        value = semantic_card.get(field_name, [])
        if isinstance(value, list):
            chunks.extend(clean_string_list(value, limit=12))
        else:
            text = clean_text(value)
            if text:
                chunks.append(text)
    return " ".join(chunks)


# 从 IntentFrame 中收集可用于召回和打分的关键词集合。
def collect_intent_terms(intent_frame: Dict[str, Any]) -> Dict[str, List[str]]:
    terms = {
        "topic": [],
        "constraints": [],
        "exact": [],
        "preferences": [],
        "ambiguous": [],
    }
    for path_name, spec in intent.SLOT_SPECS.items():
        slot = intent.get_slot(intent_frame, spec["path"])
        value = slot["value"]
        texts = value if isinstance(value, list) else ([value] if clean_text(value) else [])
        if slot["status"] == "ambiguous":
            terms["ambiguous"].append(path_name)
        if slot["status"] != "confirmed":
            continue
        if path_name.startswith("research_topic."):
            terms["topic"].extend(texts)
        elif path_name.startswith("technical_constraints."):
            terms["constraints"].extend(texts)
            if path_name in {"technical_constraints.method", "technical_constraints.dataset", "technical_constraints.model_family"}:
                terms["exact"].extend(texts)
        elif path_name in {"document_attributes.author_name", "document_attributes.title_hint"}:
            terms["exact"].extend(texts)
        elif path_name.startswith("result_preferences."):
            terms["preferences"].extend(texts)
    paper_type = intent.get_slot(intent_frame, intent.SLOT_SPECS["document_attributes.paper_type"]["path"])["value"]
    return {
        "topic": intent.filter_retrieval_terms(terms["topic"], paper_type=paper_type, limit=10),
        "constraints": intent.filter_retrieval_terms(terms["constraints"], limit=10),
        "exact": clean_string_list(terms["exact"], limit=10),
        "preferences": clean_string_list(
            [value for value in terms["preferences"] if clean_text(value) not in {"yes", "no"}],
            limit=10,
        ),
        "ambiguous": clean_string_list(terms["ambiguous"], limit=10),
    }


# 第一条召回路径：用关键词和 FTS 做稀疏检索。
def run_sparse_retrieval(intent_frame: Dict[str, Any], db_path: Path, top_k_per_query: int = 60) -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for query in intent_frame.get("coarse_queries", []):
        for result in retrieval.search_basic(query, top_k=top_k_per_query, db_path=db_path):
            existing = aggregated.get(result["paper_id"])
            score = float(result.get("fts_score") or 0.0)
            if existing is None or score > existing["raw_score"]:
                aggregated[result["paper_id"]] = {
                    "paper_id": result["paper_id"],
                    "title": result["title"],
                    "raw_score": score,
                    "matched_field": result.get("matched_field", ""),
                    "matched_snippet": result.get("matched_snippet", ""),
                    "source_query": query,
                }

    normalized_scores = normalize_score_map({paper_id: item["raw_score"] for paper_id, item in aggregated.items()})
    for paper_id, score in normalized_scores.items():
        aggregated[paper_id]["sparse_score"] = score
    return aggregated


# 第二条召回路径：强调标题线索、作者名和短语的精确命中。
def run_exact_retrieval(intent_frame: Dict[str, Any], db_path: Path, top_k_per_query: int = 40) -> Dict[str, Dict[str, Any]]:
    unique_queries = list(dict.fromkeys(clean_string_list(intent_frame.get("exact_queries", []), limit=8)))
    if not unique_queries:
        return {}

    aggregated: Dict[str, Dict[str, Any]] = {}
    for query in unique_queries:
        for result in retrieval.search_exact_matches(query, top_k=top_k_per_query, db_path=db_path):
            existing = aggregated.get(result["paper_id"])
            score = float(result.get("exact_score") or 0.0)
            if existing is None or score > existing["raw_score"]:
                aggregated[result["paper_id"]] = {
                    "paper_id": result["paper_id"],
                    "title": result["title"],
                    "raw_score": score,
                    "matched_field": result.get("matched_field", ""),
                    "matched_snippet": result.get("matched_snippet", ""),
                    "exact_match_type": result.get("match_type", ""),
                    "source_query": query,
                }

    normalized_scores = normalize_score_map({paper_id: item["raw_score"] for paper_id, item in aggregated.items()})
    for paper_id, score in normalized_scores.items():
        aggregated[paper_id]["exact_score"] = score
    return aggregated


# 第三条召回路径：基于轻量稠密向量表示做语义近邻搜索。
def run_dense_retrieval(intent_frame: Dict[str, Any], db_path: Path, top_k_per_query: int = 60) -> Dict[str, Dict[str, Any]]:
    queries = clean_string_list(intent_frame.get("dense_queries", []), limit=8)
    if not queries:
        return {}

    index = build_dense_index(db_path)
    aggregated: Dict[str, Dict[str, Any]] = {}
    postings = index["postings"]
    norms = index["norms"]
    idf = index["idf"]
    titles = index["titles"]

    for query in queries:
        query_vector, query_norm = dense_query_vector(query, idf)
        dot_scores: Dict[str, float] = defaultdict(float)
        for token, query_weight in query_vector.items():
            for paper_id, doc_weight in postings.get(token, []):
                dot_scores[paper_id] += query_weight * doc_weight
        if not dot_scores:
            continue

        scored: List[Tuple[str, float]] = []
        for paper_id, dot_value in dot_scores.items():
            denominator = query_norm * norms.get(paper_id, 1.0)
            if denominator <= 0:
                continue
            score = dot_value / denominator
            if score > 0:
                scored.append((paper_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)

        for paper_id, score in scored[:top_k_per_query]:
            existing = aggregated.get(paper_id)
            if existing is None or score > existing["raw_score"]:
                aggregated[paper_id] = {
                    "paper_id": paper_id,
                    "raw_score": score,
                    "source_query": query,
                }

    normalized_scores = normalize_score_map({paper_id: item["raw_score"] for paper_id, item in aggregated.items()})
    for paper_id, score in normalized_scores.items():
        aggregated[paper_id]["dense_score"] = score
        aggregated[paper_id]["title"] = titles.get(paper_id, "")
    return aggregated


# 将三路召回结果融合成统一候选池，是后续重排的输入。
def fuse_candidate_pool(
    sparse_results: Dict[str, Dict[str, Any]],
    dense_results: Dict[str, Dict[str, Any]],
    exact_results: Dict[str, Dict[str, Any]],
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source_name, results in (
        ("sparse", sparse_results),
        ("dense", dense_results),
        ("exact", exact_results),
    ):
        for paper_id, item in results.items():
            candidate = merged.setdefault(
                paper_id,
                {
                    "paper_id": paper_id,
                    "title": item.get("title", ""),
                    "sparse_score": 0.0,
                    "dense_score": 0.0,
                    "exact_score": 0.0,
                    "matched_field": "",
                    "matched_snippet": "",
                    "exact_match_type": "",
                    "retrieval_sources": [],
                },
            )
            candidate["title"] = candidate["title"] or item.get("title", "")
            if source_name == "sparse":
                candidate["sparse_score"] = item.get("sparse_score", 0.0)
                candidate["matched_field"] = item.get("matched_field", "")
                candidate["matched_snippet"] = item.get("matched_snippet", "")
            elif source_name == "dense":
                candidate["dense_score"] = item.get("dense_score", 0.0)
            elif source_name == "exact":
                candidate["exact_score"] = item.get("exact_score", 0.0)
                candidate["exact_match_type"] = item.get("exact_match_type", "")
                if not candidate["matched_snippet"]:
                    candidate["matched_field"] = item.get("matched_field", "")
                    candidate["matched_snippet"] = item.get("matched_snippet", "")
            candidate["retrieval_sources"] = clean_string_list(candidate["retrieval_sources"] + [source_name], limit=3)

    fused = []
    for candidate in merged.values():
        base_score = (
            FUSION_WEIGHTS["sparse"] * candidate["sparse_score"]
            + FUSION_WEIGHTS["dense"] * candidate["dense_score"]
            + FUSION_WEIGHTS["exact"] * candidate["exact_score"]
        )
        candidate["base_score"] = round(base_score, 6)
        fused.append(candidate)
    fused.sort(
        key=lambda item: (
            -item["base_score"],
            -item["exact_score"],
            -item["dense_score"],
            -item["sparse_score"],
            item["title"],
        )
    )
    return fused[:candidate_pool_size]


def compute_match_ratio(terms: Sequence[str], text: str) -> float:
    normalized_text = clean_text(text).lower()
    if not terms:
        return 0.5
    matched = 0
    for term in terms:
        normalized_term = clean_text(term).lower()
        if normalized_term and normalized_term in normalized_text:
            matched += 1
    return clamp_score(matched / max(len(terms), 1))


def build_paper_text(row: Any, semantic_card: Dict[str, Any]) -> str:
    section_titles = parse_json_field(row["section_titles"], [])
    chunks = [
        row["title"],
        row["abstract"],
        " ".join(section_titles),
        row["embedding_text"],
        semantic_card_text(semantic_card),
    ]
    return " ".join(clean_text(chunk) for chunk in chunks if clean_text(chunk))


def compute_time_match(intent_frame: Dict[str, Any], row: Any) -> Tuple[float, List[str]]:
    conflicts: List[str] = []
    year_month = clean_text(row["year_month"])
    time_slot = intent.get_slot(intent_frame, intent.SLOT_SPECS["document_attributes.time_range"]["path"])
    prefer_recent = intent.get_slot(intent_frame, intent.SLOT_SPECS["result_preferences.prefer_recent"]["path"])
    prefer_classic = intent.get_slot(intent_frame, intent.SLOT_SPECS["result_preferences.prefer_classic"]["path"])

    if prefer_classic["status"] == "confirmed" and prefer_classic["value"] == "yes":
        conflicts.append("The current corpus is skewed toward recent papers, so classic-paper preference cannot be fully satisfied.")
        return 0.0, conflicts
    if time_slot["status"] == "confirmed" and time_slot["value"] not in {"", "recent", "last 2 years", "last 3 years"}:
        if time_slot["value"] not in year_month and not time_slot["value"].startswith(">="):
            conflicts.append(
                f"Paper time metadata `{year_month}` does not fully match the requested range `{time_slot['value']}`."
            )
            return 0.2, conflicts
    if prefer_recent["status"] == "confirmed" and prefer_recent["value"] == "yes":
        return 1.0, conflicts
    if time_slot["status"] == "confirmed" and time_slot["value"]:
        return 0.9, conflicts
    return 0.5, conflicts


def compute_survey_match(intent_frame: Dict[str, Any], paper_type: str) -> Tuple[float, List[str]]:
    conflicts: List[str] = []
    prefer_survey = intent.get_slot(intent_frame, intent.SLOT_SPECS["result_preferences.prefer_survey"]["path"])
    if prefer_survey["status"] == "confirmed":
        if prefer_survey["value"] == "yes":
            if paper_type == "survey":
                return 1.0, conflicts
            conflicts.append("The user prefers survey papers, but this paper is not a survey.")
            return 0.0, conflicts
        if prefer_survey["value"] == "no":
            if paper_type != "survey":
                return 1.0, conflicts
            return 0.2, ["The user does not prefer surveys, but this paper is a survey."]
    return 0.5, conflicts


def compute_paper_type_match(intent_frame: Dict[str, Any], paper_type: str) -> Tuple[float, List[str]]:
    conflicts: List[str] = []
    requested = intent.get_slot(intent_frame, intent.SLOT_SPECS["document_attributes.paper_type"]["path"])
    if requested["status"] != "confirmed" or not requested["value"]:
        return 0.5, conflicts
    if requested["value"] == paper_type:
        return 1.0, conflicts
    conflicts.append(f"The user asked for `{requested['value']}`, but the paper type is `{paper_type}`.")
    return 0.0, conflicts


def compute_scene_match(intent_frame: Dict[str, Any], candidate: Dict[str, Any], paper_type: str, evidence_pack: Dict[str, Any]) -> float:
    scene = intent.get_slot(intent_frame, intent.SLOT_SPECS["search_scene"]["path"])["value"]
    if scene == "survey_lookup":
        return 1.0 if paper_type == "survey" else 0.2
    if scene == "recent_progress":
        return 0.9
    if scene == "specific_paper_lookup":
        return 1.0 if candidate.get("exact_score", 0.0) > 0.5 else 0.2
    if scene == "author_trace":
        return 1.0 if candidate.get("exact_match_type") == "author_match" else 0.1
    if scene == "method_constrained_search":
        return 1.0 if evidence_pack["intent_alignment_candidates"] else 0.3
    return 0.6


def build_matched_sections(section_rows: Sequence[Any], query_texts: Sequence[str], limit: int = 3) -> Tuple[List[str], List[Dict[str, str]]]:
    query_features = []
    for query_text in query_texts:
        normalized_query_text = clean_text(query_text)
        if not normalized_query_text:
            continue
        query_features.append(
            (
                retrieval.normalize_match_text(normalized_query_text),
                retrieval.tokenize_query(normalized_query_text),
            )
        )
    if not query_features:
        return [], []

    scored_sections: List[Tuple[int, str, str]] = []
    for row in section_rows:
        section_title = clean_text(row["section_title"])
        section_snippet = clean_text(row["section_snippet"])
        best_score = 0
        for normalized_query, tokens in query_features:
            score = retrieval.score_text(f"{section_title}\n{section_snippet}", normalized_query, tokens)
            if score > best_score:
                best_score = score
        if best_score > 0:
            scored_sections.append((best_score, section_title, section_snippet))
    scored_sections.sort(key=lambda item: item[0], reverse=True)
    matched_sections = []
    matched_snippets = []
    for _, section_title, section_snippet in scored_sections[:limit]:
        matched_sections.append(section_title)
        matched_snippets.append({"field": "section_snippet", "snippet": section_snippet})
    return matched_sections, matched_snippets


# 为每篇候选论文组织标题、摘要、章节和语义卡片等证据包。
def build_paper_evidence_pack(
    candidate: Dict[str, Any],
    row: Any,
    section_rows: Sequence[Any],
    intent_frame: Dict[str, Any],
    query_texts: Optional[Sequence[str]] = None,
    intent_terms: Optional[Dict[str, List[str]]] = None,
    include_section_matches: bool = True,
) -> Dict[str, Any]:
    normalized_authors = parse_json_field(row["normalized_authors"], [])
    semantic_card = parse_json_field(row["semantic_card_json"], {})
    normalized_query_texts = clean_string_list(
        query_texts
        if query_texts is not None
        else intent_frame.get("coarse_queries", []) + intent_frame.get("dense_queries", []) + intent_frame.get("exact_queries", []),
        limit=12,
    )
    matched_sections: List[str] = []
    extra_snippets: List[Dict[str, str]] = []
    if include_section_matches:
        matched_sections, extra_snippets = build_matched_sections(section_rows, normalized_query_texts)
    matched_snippets = []
    if candidate.get("matched_snippet"):
        matched_snippets.append(
            {
                "field": candidate.get("matched_field", ""),
                "snippet": candidate.get("matched_snippet", ""),
            }
        )
    matched_snippets.extend(extra_snippets)

    paper_text = build_paper_text(row, semantic_card)
    terms = intent_terms or collect_intent_terms(intent_frame)
    paper_text_lower = paper_text.lower()
    intent_alignment_candidates = []
    for term in terms["topic"] + terms["constraints"]:
        normalized_term = clean_text(term).lower()
        if normalized_term and normalized_term in paper_text_lower:
            intent_alignment_candidates.append(term)

    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "normalized_authors": normalized_authors,
        "abstract": clean_text(row["abstract"]),
        "semantic_card": semantic_card,
        "matched_sections": clean_string_list(matched_sections, limit=3),
        "matched_snippets": matched_snippets[:4],
        "intent_alignment_candidates": clean_string_list(intent_alignment_candidates, limit=6),
        "constraint_conflicts": [],
        "section_matches_included": bool(include_section_matches),
    }


# 基于意图维度给候选论文打分，形成第一轮规则化重排。
def score_candidate_against_intent(
    candidate: Dict[str, Any],
    row: Any,
    evidence_pack: Dict[str, Any],
    intent_frame: Dict[str, Any],
    intent_terms: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    semantic_card = evidence_pack["semantic_card"]
    paper_type = infer_paper_type(row, semantic_card)
    paper_text = build_paper_text(row, semantic_card)
    terms = intent_terms or collect_intent_terms(intent_frame)

    topic_terms = clean_string_list(terms["topic"], limit=8)
    if not topic_terms:
        topic_terms = clean_string_list(intent_frame.get("coarse_queries", [])[:1], limit=4)
    topic_match = compute_match_ratio(topic_terms, paper_text)

    constraint_terms = clean_string_list(terms["constraints"], limit=6)
    constraint_match = compute_match_ratio(constraint_terms, paper_text) if constraint_terms else 0.5
    conflicts: List[str] = []
    if constraint_terms and constraint_match < 0.3:
        conflicts.append(
            "The user specified technical constraints, but the paper shows weak evidence on method/model/dataset alignment."
        )

    paper_type_match, paper_type_conflicts = compute_paper_type_match(intent_frame, paper_type)
    conflicts.extend(paper_type_conflicts)
    time_preference_match, time_conflicts = compute_time_match(intent_frame, row)
    conflicts.extend(time_conflicts)
    survey_preference_match, survey_conflicts = compute_survey_match(intent_frame, paper_type)
    conflicts.extend(survey_conflicts)
    scene_match = compute_scene_match(intent_frame, candidate, paper_type, evidence_pack)

    intent_score = clamp_score(
        sum(
            [
                INTENT_SCORE_WEIGHTS["scene_match"] * scene_match,
                INTENT_SCORE_WEIGHTS["topic_match"] * topic_match,
                INTENT_SCORE_WEIGHTS["constraint_match"] * constraint_match,
                INTENT_SCORE_WEIGHTS["paper_type_match"] * paper_type_match,
                INTENT_SCORE_WEIGHTS["time_preference_match"] * time_preference_match,
                INTENT_SCORE_WEIGHTS["survey_preference_match"] * survey_preference_match,
            ]
        )
    )
    conflict_penalty = min(0.25, 0.05 * len(conflicts))
    evidence_pack["constraint_conflicts"] = clean_string_list(conflicts, limit=5)

    return {
        "paper_type": paper_type,
        "scene_match": round(scene_match, 6),
        "topic_match": round(topic_match, 6),
        "constraint_match": round(constraint_match, 6),
        "paper_type_match": round(paper_type_match, 6),
        "time_preference_match": round(time_preference_match, 6),
        "survey_preference_match": round(survey_preference_match, 6),
        "intent_score": round(intent_score, 6),
        "conflict_penalty": round(conflict_penalty, 6),
    }


def ensure_ranking_reasons(
    reasons: Sequence[str],
    evidence_pack: Dict[str, Any],
    rank_result: Dict[str, Any],
) -> List[str]:
    normalized = clean_string_list(reasons, limit=4)
    if normalized:
        return normalized

    snippets = evidence_pack.get("matched_snippets", [])
    if snippets:
        top_snippet = snippets[0]
        return [
            clean_text(
                f"Evidence from `{top_snippet.get('field', '')}`: {top_snippet.get('snippet', '')[:120]}"
            )
        ]

    intent_terms = clean_string_list(evidence_pack.get("intent_alignment_candidates", []), limit=3)
    if intent_terms:
        return ["Matched intent terms: " + "; ".join(intent_terms)]

    retrieval_sources = clean_string_list(rank_result.get("retrieval_sources", []), limit=3)
    if retrieval_sources:
        return ["The result stayed competitive across retrieval and fusion stages."]

    return ["The result remains relevant after retrieval and intent-aware reranking."]


def ensure_reason_list(reasons: Sequence[str], evidence_pack: Dict[str, Any], rank_result: Dict[str, Any]) -> List[str]:
    normalized = clean_string_list(reasons, limit=4)
    if normalized:
        return normalized
    return ensure_ranking_reasons([], evidence_pack, rank_result)


def keep_ranked_result(intent_frame: Dict[str, Any], rank_result: Dict[str, Any]) -> bool:
    query_match = rank_result.get("query_paper_match") or {}
    match_score = float(query_match.get("match_score", 0.0) or 0.0)
    main_intent_satisfied = bool(query_match.get("main_intent_satisfied"))
    scene = clean_text(intent.get_slot(intent_frame, intent.SLOT_SPECS["search_scene"]["path"]).get("value"))
    requested_paper_type_slot = intent.get_slot(intent_frame, intent.SLOT_SPECS["document_attributes.paper_type"]["path"])
    requested_paper_type = clean_text(requested_paper_type_slot.get("value")) if requested_paper_type_slot.get("status") == "confirmed" else ""
    exact_match_type = clean_text(rank_result.get("exact_match_type", ""))
    paper_type = clean_text(rank_result.get("paper_type", ""))

    if main_intent_satisfied:
        return True
    if requested_paper_type and paper_type and paper_type != requested_paper_type:
        return False
    if scene == "survey_lookup" and paper_type != "survey":
        return False
    if scene in {"author_trace", "specific_paper_lookup"}:
        if exact_match_type in {"author_match", "title_hint"} and match_score >= 0.6:
            return True
        return match_score >= 0.72
    return match_score >= 0.62


def paper_type_priority(requested_paper_type: str, scene: str, paper_type: str) -> int:
    normalized_type = clean_text(paper_type)
    if requested_paper_type:
        if normalized_type == requested_paper_type:
            return 2
        return 0 if normalized_type else 1
    if scene == "survey_lookup":
        if normalized_type == "survey":
            return 2
        return 0 if normalized_type else 1
    return 1


# 组装 query-paper 匹配所需的模型输入消息。
def build_query_paper_match_messages(
    intent_frame: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    rank_result: Dict[str, Any],
) -> List[Dict[str, str]]:
    payload = {
        "intent_frame": intent_frame,
        "paper": build_query_paper_match_paper_payload(evidence_pack, rank_result),
    }
    return [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


# 对单篇候选论文生成 query-paper 匹配解释。
def generate_query_paper_match(
    intent_frame: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    rank_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not can_use_openai():
        raise OpenAIAPIError(
            f"LLM query-paper match is required, but the LLM runtime is unavailable: {OPENAI_RUNTIME_MESSAGE}"
        )

    cached_result = load_cached_query_paper_match(intent_frame, rank_result["paper_id"])
    if cached_result is not None:
        return cached_result

    try:
        raw_payload, used_model = structured_chat_completion(
            messages=build_query_paper_match_messages(intent_frame, evidence_pack, rank_result),
            schema_name="query_paper_match",
            schema=EXPLANATION_SCHEMA,
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=600,
            timeout=120,
            api_key=OPENAI_API_KEY,
        )
    except Exception as exc:
        append_error_log(
            {
                "stage": "query_paper_match",
                "paper_id": rank_result.get("paper_id"),
                "title": rank_result.get("title"),
                "error": str(exc),
            }
        )
        raise OpenAIAPIError(
            f"LLM query-paper match failed for {rank_result.get('paper_id')}: {exc}"
        ) from exc

    payload = normalize_query_paper_match_payload(raw_payload)
    write_cached_query_paper_match(
        intent_frame,
        rank_result["paper_id"],
        payload,
        used_model,
        generator="llm_query_paper_match",
    )
    return payload, used_model


# 批量解释时把多篇候选打包到一次结构化请求中。
def build_query_paper_match_batch_messages(
    intent_frame: Dict[str, Any],
    batch_items: Sequence[Dict[str, Any]],
    evidence_packs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    payload = {
        "intent_frame": intent_frame,
        "papers": [
            build_query_paper_match_paper_payload(evidence_packs[item["paper_id"]], item)
            for item in batch_items
        ],
    }
    return [
        {"role": "system", "content": EXPLANATION_BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


# 批量生成 query-paper 匹配结果，降低调用成本。
def generate_query_paper_match_batch(
    intent_frame: Dict[str, Any],
    batch_items: Sequence[Dict[str, Any]],
    evidence_packs: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    if not batch_items:
        return {}, None
    if not can_use_openai():
        raise OpenAIAPIError(
            f"LLM query-paper match is required, but the LLM runtime is unavailable: {OPENAI_RUNTIME_MESSAGE}"
        )

    expected_paper_ids = [item["paper_id"] for item in batch_items]
    try:
        raw_payload, used_model = structured_chat_completion(
            messages=build_query_paper_match_batch_messages(intent_frame, batch_items, evidence_packs),
            schema_name="query_paper_match_batch",
            schema=build_query_paper_match_batch_schema(len(batch_items)),
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=max(1200, 420 * len(batch_items)),
            timeout=150,
            api_key=OPENAI_API_KEY,
        )
        normalized_results = normalize_query_paper_match_batch_payload(raw_payload, expected_paper_ids)
    except Exception as exc:
        append_error_log(
            {
                "stage": "query_paper_match_batch",
                "paper_ids": expected_paper_ids,
                "error": str(exc),
            }
        )
        raise OpenAIAPIError(
            "LLM query-paper match batch failed for: " + ", ".join(expected_paper_ids[:5]) + f" | {exc}"
        ) from exc

    for paper_id, payload in normalized_results.items():
        write_cached_query_paper_match(
            intent_frame,
            paper_id,
            payload,
            used_model,
            generator="llm_query_paper_match_batch",
        )
    return normalized_results, used_model


# 根据最终结果反推当前检索还缺什么，形成 Gap 分析。
def build_gap_report(
    intent_frame: Dict[str, Any],
    ranked_results: Sequence[Dict[str, Any]],
    *,
    follow_up_applied: bool = False,
) -> Dict[str, Any]:
    gap_excluded_slots = {"result_preferences.need_explainable_reason"}
    missing_slots = [slot for slot in list(intent_frame.get("missing_slots", [])) if slot not in gap_excluded_slots]
    ambiguous_dimensions = []
    for path_name, spec in intent.SLOT_SPECS.items():
        if path_name in gap_excluded_slots:
            continue
        slot = intent.get_slot(intent_frame, spec["path"])
        if slot["status"] == "ambiguous":
            ambiguous_dimensions.append(path_name)

    matched_dimensions: List[str] = []
    evidence_gap: List[str] = []
    why_broad: List[str] = []
    improvements: List[str] = []

    if ranked_results:
        top_slice = ranked_results[: min(10, len(ranked_results))]
        avg_topic = sum(item["topic_match"] for item in top_slice) / len(top_slice)
        avg_constraint = sum(item["constraint_match"] for item in top_slice) / len(top_slice)
        main_intent_hits = sum(
            1 for item in top_slice if (item.get("query_paper_match", {}) or {}).get("main_intent_satisfied")
        )
        survey_hits = sum(1 for item in top_slice if item["paper_type"] == "survey")
        paper_type_hits = sum(1 for item in top_slice if item["paper_type_match"] >= 0.9)
        match_scores = [
            item.get("query_paper_match", {}).get("match_score", 0.0)
            for item in top_slice
            if item.get("query_paper_match")
        ]
        llm_matched_dimensions = clean_string_list(
            (
                localize_user_label(dimension)
                for item in top_slice
                for dimension in item.get("query_paper_match", {}).get("matched_dimensions", [])
            ),
            limit=6,
        )
        llm_unmet_dimensions = clean_string_list(
            (
                localize_user_label(dimension)
                for item in top_slice
                for dimension in item.get("query_paper_match", {}).get("unmet_dimensions", [])
            ),
            limit=6,
        )
        llm_matched_dimensions = [value for value in llm_matched_dimensions if contains_chinese(value)]
        llm_unmet_dimensions = [value for value in llm_unmet_dimensions if contains_chinese(value)]

        if avg_topic >= 0.5:
            matched_dimensions.append("topic_match")
        else:
            evidence_gap.append("当前 Top-K 结果在目标主题上的集中度仍不足。")

        if main_intent_hits >= max(1, len(top_slice) // 3):
            matched_dimensions.append("主意图满足")
        else:
            if follow_up_applied:
                evidence_gap.append("当前 Top-K 中多数论文仍未满足主意图，追问后的约束尚未真正收敛。")
            else:
                evidence_gap.append("当前 Top-K 中多数论文仍未满足主意图，说明现有约束还不足以让结果收敛。")

        confirmed_constraints = [
            path_name
            for path_name, spec in intent.SLOT_SPECS.items()
            if path_name.startswith("technical_constraints.")
            and intent.get_slot(intent_frame, spec["path"])["status"] == "confirmed"
        ]
        if confirmed_constraints:
            if avg_constraint >= 0.45:
                matched_dimensions.append("constraint_match")
            else:
                evidence_gap.append("当前 Top-K 对技术约束的覆盖仍不稳定。")

        requested_paper_type = intent.get_slot(intent_frame, intent.SLOT_SPECS["document_attributes.paper_type"]["path"])
        if requested_paper_type["status"] == "confirmed":
            if paper_type_hits >= max(2, len(top_slice) // 3):
                matched_dimensions.append("paper_type_match")
            else:
                evidence_gap.append("当前 Top-K 中目标论文类型占比仍偏低。")

        prefer_survey = intent.get_slot(intent_frame, intent.SLOT_SPECS["result_preferences.prefer_survey"]["path"])
        if prefer_survey["status"] == "confirmed" and prefer_survey["value"] == "yes" and survey_hits == 0:
            evidence_gap.append("用户偏好综述，但当前 Top-K 中未出现综述论文。")

        if match_scores and sum(match_scores) / len(match_scores) < 0.55:
            evidence_gap.append("当前 Top-K 的 LLM query-paper 匹配得分整体偏弱。")
        matched_dimensions.extend(llm_matched_dimensions[:3])
        evidence_gap.extend(llm_unmet_dimensions[:2])

    if missing_slots:
        why_broad.append("部分意图槽位仍缺失，检索与重排需要保持较宽召回。")
        localized_missing_slots = localize_user_label_list(missing_slots, limit=6)
        improvements.append("优先补齐这些缺失槽位：" + "；".join(localized_missing_slots))
    if ambiguous_dimensions:
        why_broad.append("部分槽位仍存在歧义，系统会避免在这些维度过度收敛。")
    if evidence_gap:
        why_broad.extend(evidence_gap[:2])

    if not improvements:
        if evidence_gap:
            improvements.append("可补充方法约束、论文类型或更明确的标题/实体线索，以进一步收敛结果。")
        else:
            improvements.append("当前结果已较为集中，可直接查看 Top-K。")

    return {
        "query_gap": localize_user_label_list(missing_slots, limit=6),
        "evidence_gap": clean_string_list(evidence_gap, limit=6),
        "matched_dimensions": localize_user_label_list(matched_dimensions, limit=6),
        "ambiguous_dimensions": localize_user_label_list(ambiguous_dimensions, limit=6),
        "why_current_results_are_broad": clean_string_list(why_broad, limit=4),
        "what_next_answer_would_improve": clean_string_list(improvements, limit=4),
    }


def localize_follow_up_slot_value(path_name: str, value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if path_name == "search_scene":
        return FOLLOW_UP_SEARCH_SCENE_LABELS.get(text, text)
    if path_name == "document_attributes.time_range":
        return FOLLOW_UP_TIME_RANGE_LABELS.get(text, text)
    if path_name == "document_attributes.paper_type":
        return FOLLOW_UP_PAPER_TYPE_LABELS.get(text, text)
    if path_name.startswith("result_preferences."):
        return {"yes": "是", "no": "否"}.get(text, text)
    return text


def format_follow_up_slot_summary_value(path_name: str, slot: Dict[str, Any]) -> str:
    value = slot.get("value")
    if isinstance(value, list):
        return "、".join(
            clean_string_list(
                (localize_follow_up_slot_value(path_name, item) for item in value if clean_text(item)),
                limit=6,
            )
        )
    return localize_follow_up_slot_value(path_name, value)


def build_follow_up_intent_snapshot(frame: Dict[str, Any]) -> Dict[str, Any]:
    confirmed_slots: List[str] = []
    ambiguous_slots: List[str] = []
    for path_name, slot, _ in intent.iter_leaf_slots(frame):
        label = localize_slot_path(path_name)
        value_text = format_follow_up_slot_summary_value(path_name, slot)
        if slot.get("status") == "confirmed" and value_text:
            confirmed_slots.append(f"{label}={value_text}")
        elif slot.get("status") == "ambiguous":
            ambiguous_slots.append(label)
    return {
        "confirmed_slots": clean_string_list(confirmed_slots, limit=12),
        "ambiguous_slots": clean_string_list(ambiguous_slots, limit=8),
        "missing_slots": localize_user_label_list(frame.get("missing_slots", []), limit=8),
        "answered_slots": localize_user_label_list(frame.get("answered_slots", []), limit=12),
        "clarification_needed": bool(frame.get("clarification_needed")),
        "clarification_question": clean_text(frame.get("clarification_question", "")),
        "query_variants": {
            "coarse": clean_string_list(frame.get("coarse_queries", []), limit=4),
            "dense": clean_string_list(frame.get("dense_queries", []), limit=4),
            "exact": clean_string_list(frame.get("exact_queries", []), limit=4),
        },
    }


def build_follow_up_intent_delta(
    initial_frame: Dict[str, Any],
    final_frame: Dict[str, Any],
    limit: int = 8,
) -> List[str]:
    changes: List[str] = []
    for path_name, spec in intent.SLOT_SPECS.items():
        initial_slot = intent.get_slot(initial_frame, spec["path"])
        final_slot = intent.get_slot(final_frame, spec["path"])
        initial_value = format_follow_up_slot_summary_value(path_name, initial_slot)
        final_value = format_follow_up_slot_summary_value(path_name, final_slot)
        initial_status = clean_text(initial_slot.get("status"))
        final_status = clean_text(final_slot.get("status"))
        if initial_status == final_status and initial_value == final_value:
            continue
        label = localize_slot_path(path_name)
        initial_display = initial_value or {"missing": "待补充", "ambiguous": "有歧义"}.get(initial_status, "未指定")
        final_display = final_value or {"missing": "待补充", "ambiguous": "有歧义"}.get(final_status, "未指定")
        changes.append(f"{label}: {initial_display} -> {final_display}")

    query_groups = [
        ("粗召回 query", clean_string_list(initial_frame.get("coarse_queries", []), limit=4), clean_string_list(final_frame.get("coarse_queries", []), limit=4)),
        ("稠密召回 query", clean_string_list(initial_frame.get("dense_queries", []), limit=4), clean_string_list(final_frame.get("dense_queries", []), limit=4)),
        ("精确召回 query", clean_string_list(initial_frame.get("exact_queries", []), limit=4), clean_string_list(final_frame.get("exact_queries", []), limit=4)),
    ]
    for label, before_items, after_items in query_groups:
        if before_items != after_items:
            before_text = " | ".join(before_items) if before_items else "未生成"
            after_text = " | ".join(after_items) if after_items else "未生成"
            changes.append(f"{label}: {before_text} -> {after_text}")
    return clean_string_list(changes, limit=limit)


def build_follow_up_result_signal_summary(
    ranked_results: Sequence[Dict[str, Any]],
    limit: int = 5,
) -> Dict[str, Any]:
    top_slice = list(ranked_results[:limit])
    if not top_slice:
        return {
            "top_slice_size": 0,
            "main_intent_satisfied_count": 0,
            "avg_match_score": 0.0,
            "paper_type_distribution": [],
            "dominant_matched_dimensions": [],
            "dominant_unmet_dimensions": [],
            "main_intent_unsatisfied_titles": [],
        }

    main_intent_satisfied_count = 0
    match_scores: List[float] = []
    paper_type_counter: Counter[str] = Counter()
    matched_dimensions: List[str] = []
    unmet_dimensions: List[str] = []
    unsatisfied_titles: List[str] = []
    for item in top_slice:
        query_match = item.get("query_paper_match") or {}
        if query_match.get("main_intent_satisfied"):
            main_intent_satisfied_count += 1
        else:
            title = clean_text(item.get("title", ""))
            if title:
                unsatisfied_titles.append(title)
        match_scores.append(coerce_query_match_score(query_match.get("match_score", 0.0)))
        paper_type = clean_text(item.get("paper_type", ""))
        if paper_type:
            paper_type_counter[localize_follow_up_slot_value("document_attributes.paper_type", paper_type)] += 1
        matched_dimensions.extend(localize_user_label_list(query_match.get("matched_dimensions", []), limit=6))
        unmet_dimensions.extend(localize_user_label_list(query_match.get("unmet_dimensions", []), limit=6))

    avg_match_score = round(sum(match_scores) / max(len(match_scores), 1), 3)
    paper_type_distribution = [f"{paper_type} x{count}" for paper_type, count in paper_type_counter.most_common(4)]
    return {
        "top_slice_size": len(top_slice),
        "main_intent_satisfied_count": main_intent_satisfied_count,
        "avg_match_score": avg_match_score,
        "paper_type_distribution": paper_type_distribution,
        "dominant_matched_dimensions": clean_string_list(matched_dimensions, limit=6),
        "dominant_unmet_dimensions": clean_string_list(unmet_dimensions, limit=6),
        "main_intent_unsatisfied_titles": clean_string_list(unsatisfied_titles, limit=3),
    }


def build_follow_up_draft_fallback(intent_frame: Dict[str, Any], gap_report: Dict[str, Any]) -> str:
    query_gap = clean_string_list(gap_report.get("query_gap", []), limit=8)
    evidence_gap = clean_string_list(gap_report.get("evidence_gap", []), limit=8)
    if not query_gap and not evidence_gap and not intent_frame.get("clarification_needed"):
        return ""

    def slot_value(path_name: str) -> str:
        slot = intent.get_slot(intent_frame, intent.SLOT_SPECS[path_name]["path"])
        value = slot.get("value")
        if isinstance(value, list):
            localized_items = [
                localize_follow_up_slot_value(path_name, item)
                for item in value
                if clean_text(item)
            ]
            return "、".join(item for item in localized_items if item)
        return localize_follow_up_slot_value(path_name, value)

    segments: List[str] = []
    if any("研究领域" in item for item in query_gap):
        segments.append(f"研究领域是{slot_value('research_topic.domain') or '大语言模型'}")
    if any("研究任务" in item for item in query_gap):
        segments.append(f"研究任务是{slot_value('research_topic.task') or 'retrieval-augmented generation'}")
    if any("研究问题" in item for item in query_gap):
        segments.append(f"研究问题是{slot_value('research_topic.problem') or '当前查询关注的问题'}")
    if any(any(token in item for token in ("方法", "模型家族", "数据集", "指标", "模态")) for item in query_gap):
        segments.append("方法、模型家族、数据集、指标和模态不限")

    time_range = slot_value("document_attributes.time_range")
    if time_range:
        segments.append(f"时间范围限定为{time_range}")
    elif any("时间范围" in item for item in query_gap):
        segments.append("时间范围限定为最近两年")

    paper_type = slot_value("document_attributes.paper_type")
    paper_type_gap = any("论文类型" in item for item in query_gap + evidence_gap)
    if paper_type:
        if paper_type_gap or slot_value("search_scene") == "survey_lookup":
            segments.append(f"论文类型必须是{paper_type}")
        else:
            segments.append(f"论文类型优先{paper_type}")
    elif any("论文类型" in item for item in query_gap):
        segments.append("论文类型以综述为主")

    if slot_value("result_preferences.need_explainable_reason") == "是":
        segments.append("并解释每篇论文为何匹配")

    if not segments:
        segments.append("请只保留与当前主题直接相关、满足已有约束的论文")

    return "；".join(clean_string_list(segments, limit=6))


def build_follow_up_suggestion_context(ranked_results: Sequence[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for index, item in enumerate(ranked_results[:limit], start=1):
        query_match = item.get("query_paper_match") or {}
        items.append(
            {
                "rank": index,
                "title": clean_text(item.get("title", "")),
                "year_month": clean_text(item.get("year_month", "")),
                "paper_type": localize_follow_up_slot_value("document_attributes.paper_type", item.get("paper_type", "")),
                "final_score": round(coerce_query_match_score(item.get("final_score", 0.0)), 3),
                "match_score": round(coerce_query_match_score(query_match.get("match_score", 0.0)), 3),
                "main_intent_satisfied": bool(query_match.get("main_intent_satisfied")),
                "matched_dimensions": localize_user_label_list(query_match.get("matched_dimensions", []), limit=4),
                "unmet_dimensions": localize_user_label_list(query_match.get("unmet_dimensions", []), limit=4),
                "brief_reason": clean_text(query_match.get("brief_reason", "")),
            }
        )
    return items


def build_follow_up_suggestion_messages(
    query: str,
    follow_up_reply: Optional[str],
    initial_intent_frame: Dict[str, Any],
    final_intent_frame: Dict[str, Any],
    gap_report: Dict[str, Any],
    ranked_results: Sequence[Dict[str, Any]],
    fallback_draft: str,
) -> List[Dict[str, str]]:
    payload = {
        "search_round": "after_follow_up" if clean_text(follow_up_reply) else "initial_search",
        "original_query": clean_text(query),
        "latest_user_follow_up_reply": clean_text(follow_up_reply),
        "initial_intent_snapshot": build_follow_up_intent_snapshot(initial_intent_frame),
        "current_intent_snapshot": build_follow_up_intent_snapshot(final_intent_frame),
        "intent_change_after_follow_up": build_follow_up_intent_delta(initial_intent_frame, final_intent_frame),
        "gap_report": gap_report,
        "result_signal_summary": build_follow_up_result_signal_summary(ranked_results),
        "top_results": build_follow_up_suggestion_context(ranked_results),
        "fallback_draft_reference": fallback_draft,
    }
    return [
        {"role": "system", "content": FOLLOW_UP_SUGGESTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def normalize_follow_up_suggestion_payload(raw_payload: Dict[str, Any], fallback_draft: str) -> Tuple[str, str, bool]:
    draft = clean_text(raw_payload.get("follow_up_reply", ""))
    rationale = clean_text(raw_payload.get("rationale", ""))
    used_fallback = False
    if not draft or not contains_chinese(draft):
        draft = fallback_draft
        used_fallback = True
    if draft and draft[-1] not in "。！？!?":
        draft += "。"
    if not rationale or not contains_chinese(rationale):
        rationale = "基于当前意图缺口和排序偏差生成的建议追问。"
    return draft, rationale, used_fallback


def build_follow_up_suggestion(
    query: str,
    follow_up_reply: Optional[str],
    initial_intent_frame: Dict[str, Any],
    final_intent_frame: Dict[str, Any],
    gap_report: Dict[str, Any],
    ranked_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback_draft = build_follow_up_draft_fallback(final_intent_frame, gap_report)
    if not fallback_draft:
        return {"draft": "", "rationale": "", "generator": "none", "used_model": None}

    if not can_use_openai():
        return {
            "draft": fallback_draft,
            "rationale": "LLM 不可用，已回退到规则生成的建议追问。",
            "generator": "rule",
            "used_model": None,
        }

    try:
        raw_payload, used_model = structured_chat_completion(
            messages=build_follow_up_suggestion_messages(
                query,
                follow_up_reply,
                initial_intent_frame,
                final_intent_frame,
                gap_report,
                ranked_results,
                fallback_draft,
            ),
            schema_name="follow_up_suggestion",
            schema=FOLLOW_UP_SUGGESTION_SCHEMA,
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=320,
            timeout=90,
            api_key=OPENAI_API_KEY,
        )
        draft, rationale, used_fallback = normalize_follow_up_suggestion_payload(raw_payload, fallback_draft)
        return {
            "draft": draft,
            "rationale": rationale,
            "generator": "rule" if used_fallback else "llm",
            "used_model": None if used_fallback else used_model,
        }
    except Exception as exc:
        append_error_log(
            {
                "stage": "follow_up_suggestion",
                "error": str(exc),
            }
        )
        return {
            "draft": fallback_draft,
            "rationale": "LLM 追问建议生成失败，已回退到规则生成的建议追问。",
            "generator": "rule",
            "used_model": None,
        }


# 结合规则得分和 LLM 匹配结果做最终重排。
def rerank_candidates(
    db_path: Path,
    intent_frame: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    paper_rows: Dict[str, Any],
    sections_by_paper: Optional[Dict[str, List[Any]]] = None,
    top_k: int = DEFAULT_TOP_K,
    explain_limit: int = DEFAULT_EXPLAIN_LIMIT,
    stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    sections_by_paper = sections_by_paper or {}
    ranked: List[Dict[str, Any]] = []
    evidence_packs: Dict[str, Dict[str, Any]] = {}
    ranked_by_paper_id: Dict[str, Dict[str, Any]] = {}
    shared_query_texts = clean_string_list(
        intent_frame.get("coarse_queries", []) + intent_frame.get("dense_queries", []) + intent_frame.get("exact_queries", []),
        limit=12,
    )
    shared_intent_terms = collect_intent_terms(intent_frame)
    llm_explain_enabled = can_use_openai()
    scene = clean_text(intent.get_slot(intent_frame, intent.SLOT_SPECS["search_scene"]["path"]).get("value"))
    requested_paper_type_slot = intent.get_slot(intent_frame, intent.SLOT_SPECS["document_attributes.paper_type"]["path"])
    requested_paper_type = clean_text(requested_paper_type_slot.get("value")) if requested_paper_type_slot.get("status") == "confirmed" else ""
    if not llm_explain_enabled:
        raise OpenAIAPIError(f"LLM-led ranking is required, but query-paper match is unavailable: {OPENAI_RUNTIME_MESSAGE}")

    for candidate in candidate_pool:
        row = paper_rows[candidate["paper_id"]]
        evidence_pack = build_paper_evidence_pack(
            candidate,
            row,
            sections_by_paper.get(candidate["paper_id"], []),
            intent_frame,
            query_texts=shared_query_texts,
            intent_terms=shared_intent_terms,
            include_section_matches=False,
        )
        evidence_packs[candidate["paper_id"]] = evidence_pack
        score_payload = score_candidate_against_intent(
            candidate,
            row,
            evidence_pack,
            intent_frame,
            intent_terms=shared_intent_terms,
        )
        candidate_paper_type = clean_text(score_payload.get("paper_type", ""))
        explicit_paper_type_penalty = 0.0
        if requested_paper_type and candidate_paper_type and candidate_paper_type != requested_paper_type:
            explicit_paper_type_penalty += 0.22
        if scene == "survey_lookup" and candidate_paper_type and candidate_paper_type != "survey":
            explicit_paper_type_penalty += 0.12
        preliminary_score = clamp_score(
            0.55 * candidate["base_score"]
            + 0.45 * score_payload["intent_score"]
            - score_payload["conflict_penalty"]
            - explicit_paper_type_penalty,
            maximum=1.0,
        )
        rank_item = {
            "paper_id": candidate["paper_id"],
            "title": candidate["title"],
            "authors_raw": row["authors_raw"],
            "year_month": row["year_month"],
            "abstract": clean_text(row["abstract"]),
            "base_score": candidate["base_score"],
            "intent_score": score_payload["intent_score"],
            "preliminary_score": round(preliminary_score, 6),
            "final_score": round(preliminary_score, 6),
            "sparse_score": candidate["sparse_score"],
            "dense_score": candidate["dense_score"],
            "exact_score": candidate["exact_score"],
            "matched_field": candidate.get("matched_field", ""),
            "matched_snippet": candidate.get("matched_snippet", ""),
            "exact_match_type": candidate.get("exact_match_type", ""),
            "retrieval_sources": candidate.get("retrieval_sources", []),
            **score_payload,
            "query_paper_match": None,
            "ranking_reasons": [],
            "unmet_constraints": [],
            "explanation_adjustment": 0.0,
            "explanation_parser": "pre_rank",
            "used_model": None,
            "paper_type_priority": paper_type_priority(requested_paper_type, scene, candidate_paper_type),
        }
        ranked.append(rank_item)
        ranked_by_paper_id[candidate["paper_id"]] = rank_item

    ranked.sort(
        key=lambda item: (-item["paper_type_priority"], -item["preliminary_score"], -item["intent_score"], -item["base_score"], item["title"])
    )

    # Let LLM inspect a focused slice by default, then widen only when
    # preliminary scores remain tightly packed near the shortlist boundary.
    llm_match_limit = compute_llm_match_limit(
        ranked,
        top_k=top_k,
        explain_limit=explain_limit,
    )
    shortlisted = ranked[:llm_match_limit]
    missing_semantic_card_ids = [
        item["paper_id"]
        for item in shortlisted
        if not evidence_packs[item["paper_id"]].get("semantic_card")
    ]
    if llm_explain_enabled and missing_semantic_card_ids:
        if stage_callback:
            stage_callback(
                {
                    "stage": "semantic_card_backfill",
                    "status": "running",
                    "label": STAGE_LABELS["semantic_card_backfill"],
                    "paper_count": len(missing_semantic_card_ids),
                }
            )
        refreshed_rows = ensure_semantic_cards_for_papers(db_path, missing_semantic_card_ids)
        paper_rows.update(refreshed_rows)
        for paper_id in missing_semantic_card_ids:
            item = ranked_by_paper_id.get(paper_id)
            row = paper_rows.get(paper_id)
            if item is None or row is None:
                continue

            evidence_pack = build_paper_evidence_pack(
                item,
                row,
                sections_by_paper.get(paper_id, []),
                intent_frame,
                query_texts=shared_query_texts,
                intent_terms=shared_intent_terms,
                include_section_matches=False,
            )
            evidence_packs[paper_id] = evidence_pack
            score_payload = score_candidate_against_intent(
                item,
                row,
                evidence_pack,
                intent_frame,
                intent_terms=shared_intent_terms,
            )
            candidate_paper_type = clean_text(score_payload.get("paper_type", ""))
            explicit_paper_type_penalty = 0.0
            if requested_paper_type and candidate_paper_type and candidate_paper_type != requested_paper_type:
                explicit_paper_type_penalty += 0.22
            if scene == "survey_lookup" and candidate_paper_type and candidate_paper_type != "survey":
                explicit_paper_type_penalty += 0.12
            preliminary_score = clamp_score(
                0.55 * item["base_score"]
                + 0.45 * score_payload["intent_score"]
                - score_payload["conflict_penalty"]
                - explicit_paper_type_penalty,
                maximum=1.0,
            )
            item.update(
                {
                    "authors_raw": row["authors_raw"],
                    "year_month": row["year_month"],
                    "abstract": clean_text(row["abstract"]),
                    **score_payload,
                    "intent_score": score_payload["intent_score"],
                    "preliminary_score": round(preliminary_score, 6),
                    "final_score": round(preliminary_score, 6),
                    "query_paper_match": None,
                    "ranking_reasons": [],
                    "unmet_constraints": [],
                    "explanation_adjustment": 0.0,
                    "explanation_parser": "pre_rank",
                    "used_model": None,
                    "paper_type_priority": paper_type_priority(requested_paper_type, scene, candidate_paper_type),
                }
            )
        if stage_callback:
            stage_callback(
                {
                    "stage": "semantic_card_backfill",
                    "status": "completed",
                    "label": STAGE_LABELS["semantic_card_backfill"],
                    "paper_count": len(missing_semantic_card_ids),
                }
            )

    shortlisted.sort(
        key=lambda item: (-item["paper_type_priority"], -item["preliminary_score"], -item["intent_score"], -item["base_score"], item["title"])
    )

    match_results: Dict[str, Dict[str, Any]] = {}
    match_failures: List[str] = []
    if shortlisted and llm_explain_enabled:
        pending_items: List[Dict[str, Any]] = []
        for item in shortlisted:
            cached_result = load_cached_query_paper_match(intent_frame, item["paper_id"])
            if cached_result is None:
                pending_items.append(item)
                continue
            match_payload, used_model = cached_result
            match_results[item["paper_id"]] = {
                "match_payload": match_payload,
                "used_model": used_model,
                "parser_name": "llm_query_paper_match_cache",
            }

        if stage_callback:
            stage_callback(
                {
                    "stage": "query_paper_match",
                    "status": "running",
                    "label": STAGE_LABELS["query_paper_match"],
                    "paper_count": len(shortlisted),
                    "cached_count": len(match_results),
                    "llm_count": len(pending_items),
                }
            )

        pending_ids = [item["paper_id"] for item in pending_items]
        missing_section_ids = [paper_id for paper_id in pending_ids if paper_id not in sections_by_paper]
        if missing_section_ids:
            with retrieval.connect_db(db_path) as conn:
                loaded_sections = retrieval.load_sections_for_papers(conn, missing_section_ids)
            sections_by_paper.update(loaded_sections)

        for item in pending_items:
            row = paper_rows.get(item["paper_id"])
            if row is None:
                continue
            evidence_packs[item["paper_id"]] = build_paper_evidence_pack(
                item,
                row,
                sections_by_paper.get(item["paper_id"], []),
                intent_frame,
                query_texts=shared_query_texts,
                intent_terms=shared_intent_terms,
                include_section_matches=True,
            )

        def resolve_query_paper_match_batch_items(batch_items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            try:
                batch_payloads, used_model = generate_query_paper_match_batch(intent_frame, batch_items, evidence_packs)
                return {
                    paper_id: {
                        "match_payload": payload,
                        "used_model": used_model,
                        "parser_name": "llm_query_paper_match_batch",
                    }
                    for paper_id, payload in batch_payloads.items()
                }
            except Exception as batch_exc:
                resolved: Dict[str, Dict[str, Any]] = {}
                fallback_failures: List[str] = []
                for rank_item in batch_items:
                    evidence_pack = evidence_packs[rank_item["paper_id"]]
                    try:
                        match_payload, used_model = generate_query_paper_match(intent_frame, evidence_pack, rank_item)
                    except Exception as single_exc:
                        fallback_failures.append(f"{rank_item['paper_id']}: {single_exc}")
                        continue
                    resolved[rank_item["paper_id"]] = {
                        "match_payload": match_payload,
                        "used_model": used_model,
                        "parser_name": "llm_query_paper_match_single_fallback",
                    }
                if fallback_failures:
                    raise OpenAIAPIError(f"{batch_exc}; single fallback failures: {'; '.join(fallback_failures[:5])}") from batch_exc
                return resolved

        pending_batches = chunk_rank_items(pending_items, DEFAULT_QUERY_MATCH_BATCH_SIZE)
        workers = min(DEFAULT_EXPLANATION_WORKERS, len(pending_batches))
        if workers <= 1:
            for batch_items in pending_batches:
                try:
                    match_results.update(resolve_query_paper_match_batch_items(batch_items))
                except Exception as exc:
                    batch_ids = ", ".join(item["paper_id"] for item in batch_items[:4])
                    match_failures.append(f"{batch_ids}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(resolve_query_paper_match_batch_items, batch_items): [item["paper_id"] for item in batch_items]
                    for batch_items in pending_batches
                }
                for future in as_completed(future_map):
                    batch_ids = future_map[future]
                    try:
                        match_results.update(future.result())
                    except Exception as exc:
                        match_failures.append(f"{', '.join(batch_ids[:4])}: {exc}")

        if stage_callback:
            query_match_models = clean_string_list(
                (
                    match_results[item["paper_id"]].get("used_model")
                    for item in shortlisted
                    if item["paper_id"] in match_results
                ),
                limit=4,
            )
            stage_callback(
                {
                    "stage": "query_paper_match",
                    "status": "completed",
                    "label": STAGE_LABELS["query_paper_match"],
                    "paper_count": len(shortlisted),
                    "cached_count": len([item for item in shortlisted if item["paper_id"] in match_results and match_results[item["paper_id"]].get("parser_name") == "llm_query_paper_match_cache"]),
                    "llm_count": len([item for item in shortlisted if item["paper_id"] in match_results and match_results[item["paper_id"]].get("parser_name") != "llm_query_paper_match_cache"]),
                    "used_model": ", ".join(query_match_models),
                }
            )

    if match_failures:
        raise OpenAIAPIError(
            "LLM query-paper match is required for ranking, but it failed for: "
            + "; ".join(match_failures[:5])
        )

    for item in shortlisted:
        evidence_pack = evidence_packs[item["paper_id"]]
        resolved = match_results.get(item["paper_id"])
        if not resolved:
            raise OpenAIAPIError(f"LLM query-paper match did not return a result for {item['paper_id']}.")
        match_payload = resolved["match_payload"]
        used_model = resolved.get("used_model")
        parser_name = resolved.get("parser_name") or "llm_query_paper_match"
        matched_dimensions = match_payload.get("matched_dimensions", [])
        reasons = []
        if match_payload.get("brief_reason"):
            reasons.append(match_payload["brief_reason"])
        reasons.extend(f"命中维度：{dimension}" for dimension in matched_dimensions[:2])
        paper_type_mismatch_penalty = 0.0
        if requested_paper_type and clean_text(item.get("paper_type")) != requested_paper_type:
            paper_type_mismatch_penalty = 0.34
        elif scene == "survey_lookup" and clean_text(item.get("paper_type")) not in {"", "survey"}:
            paper_type_mismatch_penalty = 0.26
        main_intent_bonus = 0.12 if match_payload["main_intent_satisfied"] else -0.22
        main_intent_penalty = 0.12 if (not match_payload["main_intent_satisfied"] and match_payload["match_score"] < 0.65) else 0.0
        final_score = clamp_score(
            0.04 * item["base_score"]
            + 0.14 * item["intent_score"]
            + 0.50 * match_payload["match_score"]
            + 0.18 * match_payload["evidence_sufficiency"]
            + main_intent_bonus
            - main_intent_penalty,
            maximum=1.0,
        )
        final_score = clamp_score(final_score - paper_type_mismatch_penalty, maximum=1.0)
        item["query_paper_match"] = match_payload
        item["ranking_reasons"] = ensure_reason_list(reasons, evidence_pack, item)
        item["unmet_constraints"] = clean_string_list(
            match_payload.get("unmet_dimensions", []) or evidence_pack.get("constraint_conflicts", []),
            limit=4,
        )
        item["explanation_parser"] = parser_name
        item["used_model"] = used_model
        item["final_score"] = round(final_score, 6)
        item["explanation_adjustment"] = round(final_score - item["preliminary_score"], 6)

    shortlisted.sort(
        key=lambda item: (
            -int(bool((item.get("query_paper_match", {}) or {}).get("main_intent_satisfied"))),
            -item["paper_type_priority"],
            -item["final_score"],
            -item["intent_score"],
            -item["base_score"],
            item["title"],
        )
    )
    filtered_items = [item for item in shortlisted if keep_ranked_result(intent_frame, item)]
    top_items = filtered_items[:top_k]
    top_ids = {item["paper_id"] for item in top_items}
    top_missing_section_ids = [
        paper_id
        for paper_id in top_ids
        if paper_id not in sections_by_paper
        or not evidence_packs.get(paper_id, {}).get("section_matches_included")
    ]
    if top_missing_section_ids:
        unresolved_ids = [paper_id for paper_id in top_missing_section_ids if paper_id not in sections_by_paper]
        if unresolved_ids:
            with retrieval.connect_db(db_path) as conn:
                loaded_sections = retrieval.load_sections_for_papers(conn, unresolved_ids)
            sections_by_paper.update(loaded_sections)

        for paper_id in top_missing_section_ids:
            row = paper_rows.get(paper_id)
            item = ranked_by_paper_id.get(paper_id)
            if row is None or item is None:
                continue
            evidence_packs[paper_id] = build_paper_evidence_pack(
                item,
                row,
                sections_by_paper.get(paper_id, []),
                intent_frame,
                query_texts=shared_query_texts,
                intent_terms=shared_intent_terms,
                include_section_matches=True,
            )
    return top_items, {paper_id: evidence_packs[paper_id] for paper_id in top_ids}


# 以下一组函数用于标准查询回放和回归评估。
def evaluate_intent_slot_checks(final_frame: Dict[str, Any], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for path_name, expected_value in (spec.get("expected_intent_slots") or {}).items():
        slot = intent.get_slot(final_frame, intent.SLOT_SPECS[path_name]["path"])
        actual_value = slot.get("value")
        checks.append(
            {
                "slot": path_name,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "actual_status": slot.get("status"),
                "pass": slot.get("status") in {"confirmed", "ambiguous"} and slot_value_matches_expected(actual_value, expected_value),
            }
        )
    return checks


def evaluate_clarification_focus(final_frame: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    expected_focus = clean_string_list(spec.get("expected_clarification_focus", []), limit=8)
    actual_missing_slots = clean_string_list(final_frame.get("missing_slots", []), limit=20)
    if not expected_focus:
        return {
            "expected_clarification_focus": [],
            "actual_missing_slots": actual_missing_slots,
            "clarification_needed": bool(final_frame.get("clarification_needed")),
            "checks": [],
            "pass": True,
            "note": "No clarification focus was configured for this spec; skipped.",
        }

    checks = []
    for path_name in expected_focus:
        slot = intent.get_slot(final_frame, intent.SLOT_SPECS[path_name]["path"])
        value = slot.get("value")
        has_value = bool(value) if isinstance(value, list) else bool(clean_text(value))
        needs_clarification = path_name in actual_missing_slots or slot.get("status") == "ambiguous"
        checks.append(
            {
                "slot": path_name,
                "actual_status": slot.get("status"),
                "actual_value": value,
                "pass": needs_clarification or has_value,
            }
        )
    return {
        "expected_clarification_focus": expected_focus,
        "actual_missing_slots": actual_missing_slots,
        "clarification_needed": bool(final_frame.get("clarification_needed")),
        "checks": checks,
        "pass": all(item["pass"] for item in checks),
    }


def evaluate_top_result_type(top_result: Optional[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    expected_type = clean_text(spec.get("expected_top_result_type", ""))
    if not expected_type:
        return {"expected_top_result_type": "", "actual_top_result_type": "", "pass": True}
    if not top_result:
        return {
            "expected_top_result_type": expected_type,
            "actual_top_result_type": "",
            "pass": False,
            "reason": "No top result was returned.",
        }

    actual_paper_type = clean_text(top_result.get("paper_type", ""))
    actual_exact_match_type = clean_text(top_result.get("exact_match_type", ""))
    if expected_type == "author_trace":
        passed = actual_exact_match_type in {"author_match", "title_hint"}
        actual_type = actual_exact_match_type or actual_paper_type
    elif expected_type == "specific_paper_lookup":
        passed = bool(top_result.get("exact_score")) or clean_text(top_result.get("matched_field", "")) == "title"
        actual_type = actual_exact_match_type or actual_paper_type
    elif expected_type == "method":
        passed = actual_paper_type in METHOD_LIKE_PAPER_TYPES
        if not passed and actual_paper_type == "survey":
            query_match = top_result.get("query_paper_match") or {}
            passed = bool(query_match.get("main_intent_satisfied")) and float(query_match.get("match_score", 0.0)) >= 0.65
        actual_type = actual_paper_type
    else:
        passed = comparable_text(actual_paper_type) == comparable_text(expected_type)
        actual_type = actual_paper_type
    return {
        "expected_top_result_type": expected_type,
        "actual_top_result_type": actual_type,
        "paper_id": top_result.get("paper_id", ""),
        "title": top_result.get("title", ""),
        "pass": passed,
    }


def evaluate_top_3_quality(top_results: Sequence[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    inspected = list(top_results[:3])
    expected_type = clean_text(spec.get("expected_top_result_type", ""))
    if not inspected:
        return {
            "inspected_count": 0,
            "average_match_score": 0.0,
            "strong_match_count": 0,
            "theme_drift_count": 0,
            "pass": False,
            "reason": "No top-3 results were returned.",
            "items": [],
        }

    required_strong = 1 if expected_type in {"author_trace", "specific_paper_lookup"} else min(2, len(inspected))
    min_average_match = 0.45 if required_strong == 1 else 0.58
    max_theme_drift = max(0, len(inspected) - required_strong)

    strong_match_count = 0
    theme_drift_count = 0
    total_match = 0.0
    items: List[Dict[str, Any]] = []
    for item in inspected:
        query_match = item.get("query_paper_match") or {}
        match_score = float(query_match.get("match_score", 0.0) or 0.0)
        main_intent_satisfied = bool(query_match.get("main_intent_satisfied"))
        strong_match = main_intent_satisfied or match_score >= 0.55
        theme_drift = (not main_intent_satisfied and match_score < 0.55) or match_score < 0.4
        strong_match_count += 1 if strong_match else 0
        theme_drift_count += 1 if theme_drift else 0
        total_match += match_score
        items.append(
            {
                "paper_id": item.get("paper_id", ""),
                "title": item.get("title", ""),
                "paper_type": item.get("paper_type", ""),
                "match_score": round(match_score, 6),
                "main_intent_satisfied": main_intent_satisfied,
                "theme_drift": theme_drift,
            }
        )

    average_match_score = total_match / len(inspected)
    passed = (
        strong_match_count >= required_strong
        and average_match_score >= min_average_match
        and theme_drift_count <= max_theme_drift
    )
    return {
        "inspected_count": len(inspected),
        "average_match_score": round(average_match_score, 6),
        "required_strong_match_count": required_strong,
        "strong_match_count": strong_match_count,
        "max_theme_drift": max_theme_drift,
        "theme_drift_count": theme_drift_count,
        "pass": passed,
        "items": items,
    }


# 抽取解释样例，便于人工查看 query-paper 匹配质量。
def build_explanation_samples(demo_runs: Sequence[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for run in demo_runs:
        samples.append(
            {
                "query": run["query"],
                "follow_up_reply": run.get("follow_up_reply"),
                "top_k_results": [
                    {
                        "paper_id": item["paper_id"],
                        "title": item["title"],
                        "final_score": item["final_score"],
                        "query_paper_match": item.get("query_paper_match"),
                        "ranking_reasons": item.get("ranking_reasons", []),
                    }
                    for item in run.get("top_k_results", [])[:top_n]
                ],
            }
        )
    return samples


# 汇总排序效果评估结果。
def build_ranking_eval(demo_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranking_eval: List[Dict[str, Any]] = []
    for run in demo_runs:
        top_result = (run.get("top_k_results") or [None])[0]
        ranking_eval.append(
            {
                "query": run["query"],
                "follow_up_reply": run.get("follow_up_reply"),
                "candidate_pool_size": run.get("candidate_pool_size", 0),
                "top_result": {
                    "paper_id": top_result.get("paper_id", ""),
                    "title": top_result.get("title", ""),
                    "paper_type": top_result.get("paper_type", ""),
                    "final_score": top_result.get("final_score", 0.0),
                    "query_paper_match": top_result.get("query_paper_match"),
                    "ranking_reasons": top_result.get("ranking_reasons", []),
                }
                if top_result
                else None,
                "top_k_results": [
                    {
                        "paper_id": item["paper_id"],
                        "title": item["title"],
                        "paper_type": item.get("paper_type", ""),
                        "year_month": item.get("year_month", ""),
                        "final_score": item.get("final_score", 0.0),
                        "intent_score": item.get("intent_score", 0.0),
                        "paper_type_match": item.get("paper_type_match", 0.0),
                        "time_preference_match": item.get("time_preference_match", 0.0),
                        "query_paper_match": item.get("query_paper_match"),
                        "ranking_reasons": item.get("ranking_reasons", []),
                        "unmet_constraints": item.get("unmet_constraints", []),
                    }
                    for item in run.get("top_k_results", [])
                ],
            }
        )
    return ranking_eval


# 汇总标准查询的整体回归结果，判断链路是否稳定。
def build_regression_report(
    db_path: Path,
    specs: Sequence[Dict[str, Any]],
    demo_runs: Sequence[Dict[str, Any]],
    openai_available: bool,
    openai_message: str,
) -> Dict[str, Any]:
    run_map = {(item["query"], item.get("follow_up_reply")): item for item in demo_runs}
    query_reports: List[Dict[str, Any]] = []
    for spec in specs:
        key = (spec["query"], spec.get("follow_up_reply"))
        run = run_map.get(key, {})
        final_frame = run.get("final_intent_frame", {})
        top_result = (run.get("top_k_results") or [None])[0]
        top_3_quality_check = evaluate_top_3_quality(run.get("top_k_results", []), spec)
        intent_slot_checks = evaluate_intent_slot_checks(final_frame, spec)
        clarification_check = evaluate_clarification_focus(final_frame, spec)
        top_result_type_check = evaluate_top_result_type(top_result, spec)
        overall_pass = (
            all(item["pass"] for item in intent_slot_checks)
            and clarification_check["pass"]
            and top_result_type_check["pass"]
            and top_3_quality_check["pass"]
        )
        query_reports.append(
            {
                "query": spec["query"],
                "follow_up_reply": spec.get("follow_up_reply"),
                "expected_intent_slots": spec.get("expected_intent_slots", {}),
                "expected_clarification_focus": spec.get("expected_clarification_focus", []),
                "expected_top_result_type": spec.get("expected_top_result_type", ""),
                "intent_slot_checks": intent_slot_checks,
                "clarification_check": clarification_check,
                "top_result_type_check": top_result_type_check,
                "top_3_quality_check": top_3_quality_check,
                "top_result_title": top_result.get("title", "") if top_result else "",
                "top_result_score": top_result.get("final_score", 0.0) if top_result else 0.0,
                "pass": overall_pass,
            }
        )

    return {
        "db_path": str(db_path),
        "openai_available": openai_available,
        "openai_message": openai_message,
        "standard_query_count": len(specs),
        "passed_query_count": sum(1 for item in query_reports if item["pass"]),
        "failed_query_count": sum(1 for item in query_reports if not item["pass"]),
        "query_reports": query_reports,
    }


# 生成标准查询演示说明文档。
def build_demo_walkthrough(specs: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Core Chain Walkthrough",
        "",
        "## Fixed Pipeline",
        "`query -> IntentFrame -> follow-up merge -> hybrid recall -> evidence pack -> query-paper match -> gap report -> top-K explanation`",
        "",
        "## Key Output Files",
        f"- Standard queries: `{relative_to_project(STANDARD_QUERIES_PATH)}`",
        f"- Demo runs: `{relative_to_project(CHAIN_DEMOS_PATH)}`",
        f"- Gap reports: `{relative_to_project(GAP_REPORTS_PATH)}`",
        f"- Ranking eval: `{relative_to_project(RANK_RESULTS_PATH)}`",
        f"- Explanation samples: `{relative_to_project(EXPLANATION_SAMPLES_PATH)}`",
        f"- Regression report: `{relative_to_project(REGRESSION_REPORT_PATH)}`",
        "",
        "## Standard Query Notes",
    ]
    for index, spec in enumerate(specs, start=1):
        slot_summary = "; ".join(
            f"{localize_slot_path(path_name)}={clean_text(expected_value)}"
            for path_name, expected_value in (spec.get("expected_intent_slots") or {}).items()
        ) or "none"
        clarification_summary = "; ".join(
            localize_slot_path(path_name) for path_name in spec.get("expected_clarification_focus", [])
        ) or "none"
        lines.extend(
            [
                f"{index}. Query: `{spec['query']}`",
                f"   Follow-up: {spec.get('follow_up_reply') or 'none'}",
                f"   Expected intent slots: {slot_summary}",
                f"   Expected clarification focus: {clarification_summary}",
                f"   Expected top result type: {spec.get('expected_top_result_type') or 'unspecified'}",
            ]
        )
    return "\n".join(lines)


# 统计标准查询的候选集情况，辅助分析召回覆盖率。
def collect_standard_query_candidate_stats(
    db_path: Path,
    specs: Sequence[Dict[str, Any]],
    *,
    candidate_pool_size: int,
) -> Dict[str, Any]:
    occurrence_counter: Counter[str] = Counter()
    priority_counter: Counter[str] = Counter()
    must_include_ids: List[str] = []
    must_include_seen: set[str] = set()
    query_summaries: List[Dict[str, Any]] = []
    query_errors: List[Dict[str, Any]] = []

    for spec in specs:
        query = spec["query"]
        follow_up_reply = spec.get("follow_up_reply")
        try:
            initial_frame, _, _ = intent.parse_intent_frame(query)
            final_frame = initial_frame
            if follow_up_reply:
                final_frame, _, _ = intent.merge_follow_up_reply(initial_frame, follow_up_reply)

            sparse_results = run_sparse_retrieval(final_frame, db_path=db_path)
            dense_results = run_dense_retrieval(final_frame, db_path=db_path)
            exact_results = run_exact_retrieval(final_frame, db_path=db_path)
            candidate_pool = fuse_candidate_pool(
                sparse_results=sparse_results,
                dense_results=dense_results,
                exact_results=exact_results,
                candidate_pool_size=candidate_pool_size,
            )
        except Exception as exc:
            query_errors.append(
                {
                    "query": query,
                    "follow_up_reply": follow_up_reply,
                    "error": str(exc),
                }
            )
            continue

        candidate_ids = [item["paper_id"] for item in candidate_pool]
        for rank, paper_id in enumerate(candidate_ids):
            occurrence_counter[paper_id] += 1
            priority_counter[paper_id] += max(1, candidate_pool_size - rank)

        for paper_id in candidate_ids[:DEFAULT_STANDARD_QUERY_SEMANTIC_TOP_N]:
            if paper_id in must_include_seen:
                continue
            must_include_seen.add(paper_id)
            must_include_ids.append(paper_id)

        query_summaries.append(
            {
                "query": query,
                "follow_up_reply": follow_up_reply,
                "candidate_count": len(candidate_ids),
                "top_candidate_ids": candidate_ids[:10],
            }
        )

    frequent_ids = [
        paper_id
        for paper_id, count in sorted(
            occurrence_counter.items(),
            key=lambda item: (-item[1], -priority_counter[item[0]], item[0]),
        )
        if count >= DEFAULT_STANDARD_QUERY_SEMANTIC_MIN_FREQUENCY and paper_id not in must_include_seen
    ]
    target_ids = must_include_ids + frequent_ids
    return {
        "candidate_paper_ids": target_ids,
        "candidate_paper_count": len(target_ids),
        "must_include_count": len(must_include_ids),
        "frequent_candidate_count": len(frequent_ids),
        "query_error_count": len(query_errors),
        "query_errors": query_errors,
        "paper_occurrences": [
            {
                "paper_id": paper_id,
                "query_hit_count": occurrence_counter[paper_id],
                "priority_score": int(priority_counter[paper_id]),
            }
            for paper_id in target_ids
        ],
        "query_summaries": query_summaries,
    }


# 先为标准查询相关论文预热语义卡片，减少正式运行时延迟。
def prewarm_semantic_cards_for_standard_queries(
    db_path: Path,
    specs: Sequence[Dict[str, Any]],
    *,
    candidate_pool_size: int,
) -> Dict[str, Any]:
    stats = collect_standard_query_candidate_stats(
        db_path,
        specs,
        candidate_pool_size=candidate_pool_size,
    )
    target_ids = stats["candidate_paper_ids"]
    with retrieval.connect_db(db_path) as conn:
        before_count = semantic.current_card_count(conn)
    if target_ids:
        ensured_rows = ensure_semantic_cards_for_papers(db_path, target_ids)
        ensured_ids = sorted(ensured_rows.keys())
    else:
        ensured_ids = []
    with retrieval.connect_db(db_path) as conn:
        after_count = semantic.current_card_count(conn)
    stats.update(
        {
            "semantic_cards_before": before_count,
            "semantic_cards_after": after_count,
            "new_semantic_cards_generated": max(0, after_count - before_count),
            "ensured_paper_count": len(ensured_ids),
            "ensured_paper_ids": ensured_ids[:50],
        }
    )
    return stats


# 主链路入口：完成意图解析、三路召回、重排、解释和 Gap 分析。
def run_core_chain(
    query: str,
    db_path: Path,
    follow_up_reply: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    explain_limit: int = DEFAULT_EXPLAIN_LIMIT,
    stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    stage_timings: Dict[str, float] = {}
    stage_events: List[Dict[str, Any]] = []

    def relay_stage_event(event: Dict[str, Any]) -> None:
        stage_events.append(event)
        if stage_callback:
            stage_callback(dict(event))

    def emit_stage(stage: str, status: str, **payload: Any) -> None:
        event = {
            "stage": stage,
            "status": status,
            "label": STAGE_LABELS.get(stage, stage),
            **payload,
        }
        relay_stage_event(event)

    emit_stage("intent_parse", "running")
    stage_start = time.perf_counter()
    initial_frame, initial_intent_model, initial_intent_parser = intent.parse_intent_frame(query)
    stage_timings["intent_parse"] = time.perf_counter() - stage_start
    emit_stage(
        "intent_parse",
        "completed",
        duration=round(stage_timings["intent_parse"], 4),
        parser=initial_intent_parser,
        used_model=initial_intent_model,
        missing_slots=len(initial_frame.get("missing_slots", [])),
    )
    final_frame = initial_frame
    follow_up_intent_model: Optional[str] = None
    follow_up_intent_parser: Optional[str] = None
    if follow_up_reply:
        emit_stage("intent_follow_up_merge", "running")
        stage_start = time.perf_counter()
        final_frame, follow_up_intent_model, follow_up_intent_parser = intent.merge_follow_up_reply(
            initial_frame,
            follow_up_reply,
        )
        stage_timings["intent_follow_up_merge"] = time.perf_counter() - stage_start
        emit_stage(
            "intent_follow_up_merge",
            "completed",
            duration=round(stage_timings["intent_follow_up_merge"], 4),
            parser=follow_up_intent_parser,
            used_model=follow_up_intent_model,
            missing_slots=len(final_frame.get("missing_slots", [])),
        )

    emit_stage("retrieval_sparse", "running")
    stage_start = time.perf_counter()
    sparse_results = run_sparse_retrieval(final_frame, db_path=db_path)
    stage_timings["retrieval_sparse"] = time.perf_counter() - stage_start
    emit_stage("retrieval_sparse", "completed", duration=round(stage_timings["retrieval_sparse"], 4), result_count=len(sparse_results))

    emit_stage("retrieval_dense", "running")
    stage_start = time.perf_counter()
    dense_results = run_dense_retrieval(final_frame, db_path=db_path)
    stage_timings["retrieval_dense"] = time.perf_counter() - stage_start
    emit_stage("retrieval_dense", "completed", duration=round(stage_timings["retrieval_dense"], 4), result_count=len(dense_results))

    emit_stage("retrieval_exact", "running")
    stage_start = time.perf_counter()
    exact_results = run_exact_retrieval(final_frame, db_path=db_path)
    stage_timings["retrieval_exact"] = time.perf_counter() - stage_start
    emit_stage("retrieval_exact", "completed", duration=round(stage_timings["retrieval_exact"], 4), result_count=len(exact_results))

    emit_stage("retrieval_fusion", "running")
    stage_start = time.perf_counter()
    candidate_pool = fuse_candidate_pool(
        sparse_results=sparse_results,
        dense_results=dense_results,
        exact_results=exact_results,
        candidate_pool_size=candidate_pool_size,
    )
    stage_timings["retrieval_fusion"] = time.perf_counter() - stage_start
    emit_stage("retrieval_fusion", "completed", duration=round(stage_timings["retrieval_fusion"], 4), candidate_pool_size=len(candidate_pool))

    emit_stage("candidate_rows_load", "running")
    stage_start = time.perf_counter()
    with retrieval.connect_db(db_path) as conn:
        paper_rows = load_paper_rows(conn, [item["paper_id"] for item in candidate_pool])
    stage_timings["candidate_rows_load"] = time.perf_counter() - stage_start
    emit_stage("candidate_rows_load", "completed", duration=round(stage_timings["candidate_rows_load"], 4), row_count=len(paper_rows))

    emit_stage("rerank_and_explain", "running")
    stage_start = time.perf_counter()
    ranked_results, evidence_packs = rerank_candidates(
        db_path=db_path,
        intent_frame=final_frame,
        candidate_pool=candidate_pool,
        paper_rows=paper_rows,
        sections_by_paper={},
        top_k=top_k,
        explain_limit=explain_limit,
        stage_callback=relay_stage_event,
    )
    stage_timings["rerank_and_explain"] = time.perf_counter() - stage_start
    emit_stage("rerank_and_explain", "completed", duration=round(stage_timings["rerank_and_explain"], 4), result_count=len(ranked_results))

    emit_stage("gap_report", "running")
    stage_start = time.perf_counter()
    gap_report = build_gap_report(final_frame, ranked_results, follow_up_applied=bool(follow_up_reply))
    stage_timings["gap_report"] = time.perf_counter() - stage_start
    emit_stage("gap_report", "completed", duration=round(stage_timings["gap_report"], 4))

    follow_up_suggestion = {"draft": "", "rationale": "", "generator": "none", "used_model": None}
    if (
        gap_report.get("query_gap")
        or gap_report.get("evidence_gap")
        or final_frame.get("clarification_needed")
    ):
        emit_stage("follow_up_suggestion", "running")
        stage_start = time.perf_counter()
        follow_up_suggestion = build_follow_up_suggestion(
            query,
            follow_up_reply,
            initial_frame,
            final_frame,
            gap_report,
            ranked_results,
        )
        stage_timings["follow_up_suggestion"] = time.perf_counter() - stage_start
        emit_stage(
            "follow_up_suggestion",
            "completed",
            duration=round(stage_timings["follow_up_suggestion"], 4),
            generator=follow_up_suggestion.get("generator"),
            used_model=follow_up_suggestion.get("used_model"),
        )

    stage_timings["total"] = time.perf_counter() - started_at
    emit_stage("total", "completed", duration=round(stage_timings["total"], 4))

    return {
        "query": query,
        "follow_up_reply": follow_up_reply,
        "initial_intent_frame": initial_frame,
        "final_intent_frame": final_frame,
        "initial_intent_parser": initial_intent_parser,
        "initial_intent_model": initial_intent_model,
        "follow_up_intent_parser": follow_up_intent_parser,
        "follow_up_intent_model": follow_up_intent_model,
        "pipeline_mode": {
            "intent_analysis": "llm_required",
            "query_paper_match": "llm_required",
            "ranking": "llm_led",
        },
        "candidate_pool_size": len(candidate_pool),
        "sparse_results": list(sparse_results.values())[: min(20, len(sparse_results))],
        "dense_results": list(dense_results.values())[: min(20, len(dense_results))],
        "exact_results": list(exact_results.values())[: min(20, len(exact_results))],
        "intent_gap_report": gap_report,
        "follow_up_suggestion": follow_up_suggestion,
        "stage_events": stage_events,
        "stage_timings": {key: round(value, 4) for key, value in stage_timings.items()},
        "top_k_results": ranked_results,
        "paper_evidence_packs": {paper_id: evidence_packs[paper_id] for paper_id in [item["paper_id"] for item in ranked_results]},
    }


# 输出主链路构建反馈报告。
def write_chain_feedback(
    db_path: Path,
    demo_runs: Sequence[Dict[str, Any]],
    openai_available: bool,
    openai_message: str,
) -> None:
    clarification_runs = sum(1 for item in demo_runs if item["initial_intent_frame"].get("clarification_needed"))
    runs_with_follow_up = sum(1 for item in demo_runs if item.get("follow_up_reply"))
    content = "\n".join(
        [
            "核心链路构建完成",
            "",
            "运行信息",
            f"- 数据库文件: {db_path}",
            f"- OpenAI 可用: {openai_available}",
            f"- OpenAI 状态说明: {openai_message}",
            f"- 演示 query 数量: {len(demo_runs)}",
            f"- 带 follow-up 的演示数量: {runs_with_follow_up}",
            f"- 首轮需要追问的演示数量: {clarification_runs}",
            "",
            "输出文件",
            f"- 提示词: {relative_to_project(EXPLANATION_PROMPT_PATH)}",
            f"- 演示结果: {relative_to_project(CHAIN_DEMOS_PATH)}",
            f"- gap 报告: {relative_to_project(GAP_REPORTS_PATH)}",
            f"- 排序评估: {relative_to_project(RANK_RESULTS_PATH)}",
            f"- 解释样例: {relative_to_project(EXPLANATION_SAMPLES_PATH)}",
            f"- 标准查询: {relative_to_project(STANDARD_QUERIES_PATH)}",
            f"- 回归报告: {relative_to_project(REGRESSION_REPORT_PATH)}",
            f"- 演示说明: {relative_to_project(DEMO_WALKTHROUGH_PATH)}",
        ]
    )
    dump_text(FEEDBACK_PATH, content)


# 生成主链路相关的全部演示、评估和提示词资产。
def build_core_chain_assets(
    db_path: Path,
    demos: Optional[Sequence[Dict[str, Any]]] = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    explain_limit: int = DEFAULT_EXPLAIN_LIMIT,
) -> Dict[str, Any]:
    ensure_output_dir()
    write_prompt_file()
    openai_available = can_use_openai()
    openai_message = OPENAI_RUNTIME_MESSAGE

    standard_specs = list(STANDARD_QUERY_SPECS)
    demo_items = list(demos or standard_specs)
    semantic_prewarm_summary = prewarm_semantic_cards_for_standard_queries(
        db_path=db_path,
        specs=demo_items,
        candidate_pool_size=candidate_pool_size,
    )
    demo_runs = [
        run_core_chain(
            query=item["query"],
            follow_up_reply=item.get("follow_up_reply"),
            db_path=db_path,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            explain_limit=explain_limit,
        )
        for item in demo_items
    ]

    gap_reports = [
        {
            "query": item["query"],
            "follow_up_reply": item.get("follow_up_reply"),
            "intent_gap_report": item["intent_gap_report"],
        }
        for item in demo_runs
    ]
    ranking_eval = build_ranking_eval(demo_runs)
    explanation_samples = build_explanation_samples(demo_runs, top_n=min(top_k, 3))
    regression_specs = [
        item
        for item in demo_items
        if any(
            key in item
            for key in ("expected_intent_slots", "expected_clarification_focus", "expected_top_result_type")
        )
    ]
    if not regression_specs:
        regression_specs = standard_specs
    regression_report = build_regression_report(
        db_path=db_path,
        specs=regression_specs,
        demo_runs=demo_runs,
        openai_available=openai_available,
        openai_message=openai_message,
    )
    walkthrough = build_demo_walkthrough(standard_specs)

    dump_json(CHAIN_DEMOS_PATH, demo_runs)
    dump_json(GAP_REPORTS_PATH, gap_reports)
    dump_json(RANK_RESULTS_PATH, ranking_eval)
    dump_json(STANDARD_QUERIES_PATH, standard_specs)
    dump_json(REGRESSION_REPORT_PATH, regression_report)
    dump_json(EXPLANATION_SAMPLES_PATH, explanation_samples)
    dump_text(DEMO_WALKTHROUGH_PATH, walkthrough)
    write_chain_feedback(db_path, demo_runs, openai_available, openai_message)
    return {
        "db_path": str(db_path),
        "openai_available": openai_available,
        "openai_message": openai_message,
        "semantic_prewarm": semantic_prewarm_summary,
        "demo_query_count": len(demo_runs),
        "standard_query_count": len(standard_specs),
        "explanation_sample_count": len(explanation_samples),
        "output_dir": str(OUTPUT_DIR),
    }
