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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

FUSION_WEIGHTS = {"sparse": 0.45, "dense": 0.35, "exact": 0.20}
INTENT_SCORE_WEIGHTS = {
    "scene_match": 0.20,
    "topic_match": 0.30,
    "constraint_match": 0.25,
    "paper_type_match": 0.10,
    "time_preference_match": 0.10,
    "survey_preference_match": 0.05,
}
DEFAULT_CANDIDATE_POOL_SIZE = 40
DEFAULT_TOP_K = 5
DEFAULT_EXPLAIN_LIMIT = 5
DEFAULT_EXPLANATION_WORKERS = 4
OPENAI_RUNTIME_AVAILABLE: Optional[bool] = None
OPENAI_RUNTIME_MESSAGE = ""
DENSE_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
MAX_ERROR_LOG_ENTRIES = 500
DENSE_INDEX_CACHE_VERSION = "dense_index_v1"
DENSE_INDEX_DISK_CACHE_SUBDIR = "dense_indexes"
DENSE_INDEX_DISK_CACHE_KEEP_PER_DB = 3

DIMENSION_LABELS = {
    "scene_match": "检索场景匹配",
    "topic_match": "主题匹配",
    "constraint_match": "技术约束匹配",
    "paper_type_match": "论文类型匹配",
    "time_preference_match": "时间偏好匹配",
    "survey_preference_match": "综述偏好匹配",
    "scene": "检索场景匹配",
    "topic": "主题匹配",
    "constraint": "技术约束匹配",
    "constraints": "技术约束匹配",
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
    "technical_constraints.model_family": "模型族约束",
    "technical_constraints.dataset": "数据集约束",
    "technical_constraints.metric": "指标约束",
    "technical_constraints.modality": "模态约束",
    "document_attributes.time_range": "时间范围",
    "document_attributes.paper_type": "论文类型",
    "document_attributes.author_name": "作者",
    "document_attributes.title_hint": "标题线索",
    "result_preferences.prefer_recent": "偏好最新论文",
    "result_preferences.prefer_classic": "偏好经典论文",
    "result_preferences.prefer_survey": "偏好综述",
    "result_preferences.prefer_diverse": "偏好多样结果",
    "result_preferences.need_explainable_reason": "需要可解释理由",
}

STANDARD_QUERY_SPECS = [
    {
        "query": "retrieval augmented generation survey",
        "follow_up_reply": "recent two years, explain why each paper matches",
        "expected_intent_slots": {
            "search_scene": "survey_lookup",
            "research_topic.task": "retrieval-augmented generation",
            "document_attributes.paper_type": "survey",
        },
        "expected_clarification_focus": ["document_attributes.time_range"],
        "expected_top_result_type": "survey",
    },
    {
        "query": "recent agent memory papers",
        "expected_intent_slots": {
            "search_scene": "recent_progress",
            "research_topic.problem": "memory mechanism",
        },
        "expected_clarification_focus": ["document_attributes.paper_type"],
        "expected_top_result_type": "method",
    },
    {
        "query": "papers by authors of MALT",
        "follow_up_reply": "explain why they are related",
        "expected_intent_slots": {"search_scene": "author_trace"},
        "expected_clarification_focus": ["document_attributes.author_name"],
        "expected_top_result_type": "author_trace",
    },
    {
        "query": "quality estimation with COMET",
        "follow_up_reply": "prefer recent work",
        "expected_intent_slots": {
            "search_scene": "method_constrained_search",
            "technical_constraints.metric": "COMET",
        },
        "expected_clarification_focus": ["technical_constraints.dataset"],
        "expected_top_result_type": "method",
    },
    {
        "query": "long context survey papers",
        "expected_intent_slots": {
            "search_scene": "survey_lookup",
            "document_attributes.paper_type": "survey",
        },
        "expected_clarification_focus": [],
        "expected_top_result_type": "survey",
    },
    {
        "query": "benchmark for large language models on reasoning",
        "expected_intent_slots": {
            "research_topic.task": "reasoning",
            "document_attributes.paper_type": "benchmark",
        },
        "expected_clarification_focus": ["document_attributes.time_range"],
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
        "expected_intent_slots": {"search_scene": "specific_paper_lookup"},
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

EXPLANATION_SYSTEM_PROMPT = """你负责判断候选论文是否匹配用户的学术检索意图。

只能使用给定的 intent frame、semantic card、matched snippets 和排序特征。
不要编造证据，只返回 JSON。
其中：
1. brief_reason 必须使用简体中文，控制在 1 到 2 句话。
2. matched_dimensions 和 unmet_dimensions 优先使用简体中文短语；若使用系统维度标识，只能从以下集合中选择：
   scene_match, topic_match, constraint_match, paper_type_match, time_preference_match, survey_preference_match。
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
    return text


def localize_user_label_list(values: Iterable[Any], limit: int = 8) -> List[str]:
    return clean_string_list((localize_user_label(value) for value in values), limit=limit)


def build_match_reason_fallback(matched_dimensions: Sequence[str], main_intent_satisfied: bool) -> str:
    localized_dimensions = localize_user_label_list(matched_dimensions, limit=3)
    if localized_dimensions:
        prefix = "该论文与当前查询在以下维度上较为匹配："
        suffix = "，整体满足主要检索意图。" if main_intent_satisfied else "，但仍建议结合约束条件进一步判断。"
        return prefix + "、".join(localized_dimensions) + suffix
    if main_intent_satisfied:
        return "该论文与当前查询整体较为匹配，现有证据基本支持当前推荐。"
    return "该论文与当前查询存在一定相关性，但匹配证据仍然有限。"


def normalize_query_paper_match_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    matched_dimensions = localize_user_label_list(raw_payload.get("matched_dimensions", []), limit=4)
    unmet_dimensions = localize_user_label_list(raw_payload.get("unmet_dimensions", []), limit=4)
    brief_reason = clean_text(raw_payload.get("brief_reason", ""))
    main_intent_satisfied = bool(raw_payload.get("main_intent_satisfied"))
    if not brief_reason or not contains_chinese(brief_reason):
        brief_reason = build_match_reason_fallback(matched_dimensions, main_intent_satisfied)
    return {
        "main_intent_satisfied": main_intent_satisfied,
        "matched_dimensions": matched_dimensions,
        "unmet_dimensions": unmet_dimensions,
        "match_score": clamp_score(raw_payload.get("match_score", 0.0)),
        "evidence_sufficiency": clamp_score(raw_payload.get("evidence_sufficiency", 0.0)),
        "brief_reason": brief_reason,
    }


def comparable_text(value: Any) -> str:
    return clean_text(value).lower().replace("–", "-").replace("—", "-")


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
        conflicts.append("当前语料库以近期论文为主，难以满足经典论文偏好。")
        return 0.0, conflicts
    if time_slot["status"] == "confirmed" and time_slot["value"] not in {"", "recent", "last 2 years", "last 3 years"}:
        if time_slot["value"] not in year_month and not time_slot["value"].startswith(">="):
            conflicts.append(f"当前论文时间信息 `{year_month}` 与期望时间范围 `{time_slot['value']}` 不完全匹配。")
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
            conflicts.append("用户偏好综述，但该论文不是 survey。")
            return 0.0, conflicts
        if prefer_survey["value"] == "no":
            return (1.0, conflicts) if paper_type != "survey" else (0.2, ["用户不偏好综述，但该论文是 survey。"])
    return 0.5, conflicts


def compute_paper_type_match(intent_frame: Dict[str, Any], paper_type: str) -> Tuple[float, List[str]]:
    conflicts: List[str] = []
    requested = intent.get_slot(intent_frame, intent.SLOT_SPECS["document_attributes.paper_type"]["path"])
    if requested["status"] != "confirmed" or not requested["value"]:
        return 0.5, conflicts
    if requested["value"] == paper_type:
        return 1.0, conflicts
    conflicts.append(f"用户希望 `{requested['value']}`，但论文类型为 `{paper_type}`。")
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
    }


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
        conflicts.append("用户明确给了技术约束，但该论文在方法/模型/数据集相关证据上较弱。")

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
                f"召回证据来自 `{top_snippet.get('field', '')}`：{top_snippet.get('snippet', '')[:120]}"
            )
        ]

    intent_terms = clean_string_list(evidence_pack.get("intent_alignment_candidates", []), limit=3)
    if intent_terms:
        return ["命中了关键意图词：" + "、".join(intent_terms)]

    retrieval_sources = clean_string_list(rank_result.get("retrieval_sources", []), limit=3)
    if retrieval_sources:
        return ["该结果在初始召回和融合排序阶段保持了较高相关性。"]

    return ["当前结果在召回与意图重排阶段保持了较高相关性。"]


def ensure_reason_list(reasons: Sequence[str], evidence_pack: Dict[str, Any], rank_result: Dict[str, Any]) -> List[str]:
    normalized = clean_string_list(reasons, limit=4)
    if normalized:
        return normalized
    return ensure_ranking_reasons([], evidence_pack, rank_result)


def build_fallback_query_paper_match(rank_result: Dict[str, Any], evidence_pack: Dict[str, Any], error_message: str = "") -> Dict[str, Any]:
    matched_dimensions: List[str] = []
    unmet_dimensions: List[str] = []

    dimension_scores = {
        "scene_match": float(rank_result.get("scene_match", 0.0) or 0.0),
        "topic_match": float(rank_result.get("topic_match", 0.0) or 0.0),
        "constraint_match": float(rank_result.get("constraint_match", 0.0) or 0.0),
        "paper_type_match": float(rank_result.get("paper_type_match", 0.0) or 0.0),
        "time_preference_match": float(rank_result.get("time_preference_match", 0.0) or 0.0),
        "survey_preference_match": float(rank_result.get("survey_preference_match", 0.0) or 0.0),
    }
    for dimension, score in dimension_scores.items():
        if score >= 0.66:
            matched_dimensions.append(dimension)
        elif score <= 0.35:
            unmet_dimensions.append(dimension)

    evidence_sufficiency = clamp_score(
        0.30
        + 0.25 * min(len(evidence_pack.get("matched_sections", [])), 2) / 2.0
        + 0.25 * min(len(evidence_pack.get("matched_snippets", [])), 2) / 2.0
        + 0.20 * min(len(evidence_pack.get("intent_alignment_candidates", [])), 3) / 3.0
    )
    match_score = clamp_score(
        0.58 * float(rank_result.get("intent_score", 0.0) or 0.0)
        + 0.12 * dimension_scores["topic_match"]
        + 0.10 * dimension_scores["constraint_match"]
        + 0.10 * dimension_scores["paper_type_match"]
        + 0.10 * dimension_scores["time_preference_match"]
    )

    if evidence_pack.get("constraint_conflicts"):
        unmet_dimensions.extend(clean_string_list(evidence_pack["constraint_conflicts"], limit=2))

    reason = "该结果由启发式语义匹配兜底生成，综合了意图分数与命中证据。"
    if error_message:
        reason = f"{reason}（LLM 解释暂不可用：{clean_text(error_message)[:120]}）"

    return {
        "main_intent_satisfied": bool(match_score >= 0.62),
        "matched_dimensions": clean_string_list(matched_dimensions, limit=4),
        "unmet_dimensions": clean_string_list(unmet_dimensions, limit=4),
        "match_score": round(match_score, 6),
        "evidence_sufficiency": round(evidence_sufficiency, 6),
        "brief_reason": reason,
    }


def build_query_paper_match_messages(
    intent_frame: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    rank_result: Dict[str, Any],
) -> List[Dict[str, str]]:
    payload = {
        "intent_frame": intent_frame,
        "paper": {
            "paper_id": rank_result["paper_id"],
            "title": rank_result["title"],
            "authors": rank_result.get("authors_raw", ""),
            "year_month": rank_result.get("year_month", ""),
            "abstract": rank_result.get("abstract", ""),
        },
        "semantic_card": evidence_pack.get("semantic_card", {}),
        "matched_sections": evidence_pack.get("matched_sections", []),
        "matched_snippets": evidence_pack.get("matched_snippets", []),
        "rank_features": {
            "base_score": rank_result["base_score"],
            "intent_score": rank_result["intent_score"],
            "scene_match": rank_result["scene_match"],
            "topic_match": rank_result["topic_match"],
            "constraint_match": rank_result["constraint_match"],
            "paper_type_match": rank_result["paper_type_match"],
            "time_preference_match": rank_result["time_preference_match"],
            "survey_preference_match": rank_result["survey_preference_match"],
        },
    }
    return [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def generate_query_paper_match(
    intent_frame: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    rank_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not can_use_openai():
        raise OpenAIAPIError(f"核心语义能力不可用：query-paper 匹配依赖 LLM。{OPENAI_RUNTIME_MESSAGE}")

    cache_path = query_match_cache_path(intent_frame, rank_result["paper_id"])
    if cache_path.exists():
        try:
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            normalized_payload = normalize_query_paper_match_payload(cached_payload["query_paper_match"])
            return normalized_payload, cached_payload.get("used_model")
        except Exception:
            pass

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
        raise OpenAIAPIError(f"核心语义能力不可用：为 {rank_result.get('paper_id')} 生成 query-paper 匹配评分失败。{exc}") from exc

    payload = normalize_query_paper_match_payload(raw_payload)
    write_json(cache_path, {"used_model": used_model, "query_paper_match": payload})
    return payload, used_model


def build_gap_report(intent_frame: Dict[str, Any], ranked_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    missing_slots = list(intent_frame.get("missing_slots", []))
    ambiguous_dimensions = []
    for path_name, spec in intent.SLOT_SPECS.items():
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
        survey_hits = sum(1 for item in top_slice if item["paper_type"] == "survey")
        paper_type_hits = sum(1 for item in top_slice if item["paper_type_match"] >= 0.9)
        match_scores = [
            item.get("query_paper_match", {}).get("match_score", 0.0)
            for item in top_slice
            if item.get("query_paper_match")
        ]
        llm_matched_dimensions = clean_string_list(
            [
                dimension
                for item in top_slice
                for dimension in item.get("query_paper_match", {}).get("matched_dimensions", [])
            ],
            limit=6,
        )
        llm_unmet_dimensions = clean_string_list(
            [
                dimension
                for item in top_slice
                for dimension in item.get("query_paper_match", {}).get("unmet_dimensions", [])
            ],
            limit=6,
        )

        if avg_topic >= 0.5:
            matched_dimensions.append("topic_match")
        else:
            evidence_gap.append("top-K 在主题维度上的集中度仍然偏弱。")

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
                evidence_gap.append("top-K 对方法 / 模型 / 数据集等技术约束的覆盖还不够稳定。")

        requested_paper_type = intent.get_slot(intent_frame, intent.SLOT_SPECS["document_attributes.paper_type"]["path"])
        if requested_paper_type["status"] == "confirmed":
            if paper_type_hits >= max(2, len(top_slice) // 3):
                matched_dimensions.append("paper_type_match")
            else:
                evidence_gap.append("top-K 中符合目标论文类型的论文比例仍然偏低。")

        prefer_survey = intent.get_slot(intent_frame, intent.SLOT_SPECS["result_preferences.prefer_survey"]["path"])
        if prefer_survey["status"] == "confirmed" and prefer_survey["value"] == "yes" and survey_hits == 0:
            evidence_gap.append("用户偏好综述，但当前 top-K 中没有 survey。")

        if match_scores and sum(match_scores) / len(match_scores) < 0.55:
            evidence_gap.append("当前 top 结果的 query-paper 匹配分仍然偏弱。")
        matched_dimensions.extend(llm_matched_dimensions[:3])
        evidence_gap.extend(llm_unmet_dimensions[:2])

    if missing_slots:
        why_broad.append("用户意图仍有缺失槽位，导致召回和重排需要保持较宽覆盖。")
        improvements.append("优先补充以下缺失信息：" + "、".join(missing_slots[:6]))
    if ambiguous_dimensions:
        why_broad.append("部分槽位被标记为 ambiguous，系统默认不会继续追问这些维度。")
    if evidence_gap:
        why_broad.extend(evidence_gap[:2])

    if not improvements:
        if evidence_gap:
            improvements.append("优先补充方法、论文类型或标题线索，可显著收窄结果。")
        else:
            improvements.append("当前结果已较集中，可直接查看 top-K 解释。")

    return {
        "query_gap": localize_user_label_list(missing_slots, limit=6),
        "evidence_gap": clean_string_list(evidence_gap, limit=6),
        "matched_dimensions": localize_user_label_list(matched_dimensions, limit=6),
        "ambiguous_dimensions": localize_user_label_list(ambiguous_dimensions, limit=6),
        "why_current_results_are_broad": clean_string_list(why_broad, limit=4),
        "what_next_answer_would_improve": clean_string_list(improvements, limit=4),
    }


def rerank_candidates(
    db_path: Path,
    intent_frame: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    paper_rows: Dict[str, Any],
    sections_by_paper: Optional[Dict[str, List[Any]]] = None,
    top_k: int = DEFAULT_TOP_K,
    explain_limit: int = DEFAULT_EXPLAIN_LIMIT,
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
    need_explainable_slot = intent.get_slot(
        intent_frame,
        intent.SLOT_SPECS["result_preferences.need_explainable_reason"]["path"],
    )
    llm_explain_enabled = need_explainable_slot.get("value") == "yes"
    if llm_explain_enabled and not can_use_openai():
        llm_explain_enabled = False

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
        preliminary_score = clamp_score(
            0.55 * candidate["base_score"] + 0.45 * score_payload["intent_score"] - score_payload["conflict_penalty"],
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
        }
        ranked.append(rank_item)
        ranked_by_paper_id[candidate["paper_id"]] = rank_item

    ranked.sort(
        key=lambda item: (-item["preliminary_score"], -item["intent_score"], -item["base_score"], item["title"])
    )

    # Limit expensive LLM matching to the slice that can affect final UI output.
    llm_match_limit = min(len(ranked), max(top_k, explain_limit, 3))
    shortlisted = ranked[:llm_match_limit]
    missing_semantic_card_ids = [
        item["paper_id"]
        for item in shortlisted
        if not evidence_packs[item["paper_id"]].get("semantic_card")
    ]
    if llm_explain_enabled and missing_semantic_card_ids:
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
            preliminary_score = clamp_score(
                0.55 * item["base_score"] + 0.45 * score_payload["intent_score"] - score_payload["conflict_penalty"],
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
                }
            )

    shortlisted.sort(
        key=lambda item: (-item["preliminary_score"], -item["intent_score"], -item["base_score"], item["title"])
    )

    if llm_explain_enabled:
        shortlisted_ids = [item["paper_id"] for item in shortlisted]
        missing_section_ids = [paper_id for paper_id in shortlisted_ids if paper_id not in sections_by_paper]
        if missing_section_ids:
            with retrieval.connect_db(db_path) as conn:
                loaded_sections = retrieval.load_sections_for_papers(conn, missing_section_ids)
            sections_by_paper.update(loaded_sections)

        for item in shortlisted:
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

    def resolve_query_paper_match(rank_item: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str], str, str]:
        evidence_pack = evidence_packs[rank_item["paper_id"]]
        try:
            match_payload, used_model = generate_query_paper_match(intent_frame, evidence_pack, rank_item)
            return rank_item["paper_id"], match_payload, used_model, "llm_query_paper_match", ""
        except Exception as exc:
            error_message = str(exc)
            fallback_payload = build_fallback_query_paper_match(rank_item, evidence_pack, error_message=error_message)
            return (
                rank_item["paper_id"],
                fallback_payload,
                None,
                "heuristic_query_paper_match_fallback",
                error_message,
            )

    match_results: Dict[str, Dict[str, Any]] = {}
    if shortlisted and llm_explain_enabled:
        workers = min(DEFAULT_EXPLANATION_WORKERS, len(shortlisted))
        if workers <= 1:
            for item in shortlisted:
                paper_id, match_payload, used_model, parser_name, error_message = resolve_query_paper_match(item)
                match_results[paper_id] = {
                    "match_payload": match_payload,
                    "used_model": used_model,
                    "parser_name": parser_name,
                    "error_message": error_message,
                }
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {executor.submit(resolve_query_paper_match, item): item["paper_id"] for item in shortlisted}
                for future in as_completed(future_map):
                    paper_id = future_map[future]
                    try:
                        resolved = future.result()
                    except Exception as exc:  # defensive: should not happen because worker already handles errors
                        item = next((row for row in shortlisted if row["paper_id"] == paper_id), None)
                        evidence_pack = evidence_packs.get(paper_id, {})
                        fallback_payload = build_fallback_query_paper_match(item or {"paper_id": paper_id}, evidence_pack, error_message=str(exc))
                        resolved = (paper_id, fallback_payload, None, "heuristic_query_paper_match_fallback", str(exc))
                    resolved_paper_id, match_payload, used_model, parser_name, error_message = resolved
                    match_results[resolved_paper_id] = {
                        "match_payload": match_payload,
                        "used_model": used_model,
                        "parser_name": parser_name,
                        "error_message": error_message,
                    }
    elif shortlisted:
        for item in shortlisted:
            evidence_pack = evidence_packs[item["paper_id"]]
            match_results[item["paper_id"]] = {
                "match_payload": build_fallback_query_paper_match(item, evidence_pack, error_message=""),
                "used_model": None,
                "parser_name": "heuristic_query_paper_match_fastpath",
                "error_message": "",
            }

    for item in shortlisted:
        evidence_pack = evidence_packs[item["paper_id"]]
        resolved = match_results.get(item["paper_id"], {})
        match_payload = resolved.get("match_payload") or build_fallback_query_paper_match(item, evidence_pack, error_message="missing_match_result")
        used_model = resolved.get("used_model")
        parser_name = resolved.get("parser_name") or "heuristic_query_paper_match_fallback"
        error_message = clean_text(resolved.get("error_message", ""))
        if parser_name == "heuristic_query_paper_match_fallback":
            append_error_log(
                {
                    "stage": "query_paper_match_fallback",
                    "paper_id": item.get("paper_id"),
                    "title": item.get("title"),
                    "error": error_message or "fallback_without_error",
                }
            )
        matched_dimensions = match_payload.get("matched_dimensions", [])
        reasons = []
        if match_payload.get("brief_reason"):
            reasons.append(match_payload["brief_reason"])
        reasons.extend(f"命中维度：{dimension}" for dimension in matched_dimensions[:2])
        final_score = clamp_score(
            0.20 * item["base_score"]
            + 0.20 * item["intent_score"]
            + 0.35 * match_payload["match_score"]
            + 0.10 * match_payload["evidence_sufficiency"]
            + 0.10 * item["paper_type_match"]
            + 0.05 * item["time_preference_match"]
            + (0.03 if match_payload["main_intent_satisfied"] else 0.0),
            maximum=1.0,
        )
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

    shortlisted.sort(key=lambda item: (-item["final_score"], -item["intent_score"], -item["base_score"], item["title"]))
    top_items = shortlisted[:top_k]
    top_ids = {item["paper_id"] for item in top_items}
    return top_items, {paper_id: evidence_packs[paper_id] for paper_id in top_ids}


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
            "note": "未配置追问焦点断言，跳过该项校验。",
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
            "reason": "没有返回 top 结果。",
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
        intent_slot_checks = evaluate_intent_slot_checks(final_frame, spec)
        clarification_check = evaluate_clarification_focus(final_frame, spec)
        top_result_type_check = evaluate_top_result_type(top_result, spec)
        overall_pass = (
            all(item["pass"] for item in intent_slot_checks)
            and clarification_check["pass"]
            and top_result_type_check["pass"]
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


def build_demo_walkthrough(specs: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# 核心链路演示说明",
        "",
        "## 固定链路",
        "`query -> IntentFrame -> 聚合追问 -> hybrid recall -> evidence pack -> query-paper match -> gap report -> top-K explanation`",
        "",
        "## 关键输出文件",
        f"- 标准查询集：`{relative_to_project(STANDARD_QUERIES_PATH)}`",
        f"- 演示结果：`{relative_to_project(CHAIN_DEMOS_PATH)}`",
        f"- gap 报告：`{relative_to_project(GAP_REPORTS_PATH)}`",
        f"- 排序评估：`{relative_to_project(RANK_RESULTS_PATH)}`",
        f"- 解释样例：`{relative_to_project(EXPLANATION_SAMPLES_PATH)}`",
        f"- 回归报告：`{relative_to_project(REGRESSION_REPORT_PATH)}`",
        "",
        "## 标准查询说明",
    ]
    for index, spec in enumerate(specs, start=1):
        slot_summary = "；".join(
            f"{localize_slot_path(path_name)}={clean_text(expected_value)}"
            for path_name, expected_value in (spec.get("expected_intent_slots") or {}).items()
        ) or "无"
        clarification_summary = "；".join(
            localize_slot_path(path_name) for path_name in spec.get("expected_clarification_focus", [])
        ) or "无需追问"
        lines.extend(
            [
                f"{index}. 查询：`{spec['query']}`",
                f"   补充回复：{spec.get('follow_up_reply') or '无'}",
                f"   预期关键槽位：{slot_summary}",
                f"   预期追问方向：{clarification_summary}",
                f"   预期 top 结果类型：{spec.get('expected_top_result_type') or '未指定'}",
            ]
        )
    return "\n".join(lines)


def run_core_chain(
    query: str,
    db_path: Path,
    follow_up_reply: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    explain_limit: int = DEFAULT_EXPLAIN_LIMIT,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    stage_timings: Dict[str, float] = {}

    stage_start = time.perf_counter()
    initial_frame, _, _ = intent.parse_intent_frame(query)
    stage_timings["intent_parse"] = time.perf_counter() - stage_start
    final_frame = initial_frame
    if follow_up_reply:
        stage_start = time.perf_counter()
        final_frame, _, _ = intent.merge_follow_up_reply(initial_frame, follow_up_reply)
        stage_timings["intent_follow_up_merge"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    sparse_results = run_sparse_retrieval(final_frame, db_path=db_path)
    stage_timings["retrieval_sparse"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    dense_results = run_dense_retrieval(final_frame, db_path=db_path)
    stage_timings["retrieval_dense"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    exact_results = run_exact_retrieval(final_frame, db_path=db_path)
    stage_timings["retrieval_exact"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    candidate_pool = fuse_candidate_pool(
        sparse_results=sparse_results,
        dense_results=dense_results,
        exact_results=exact_results,
        candidate_pool_size=candidate_pool_size,
    )
    stage_timings["retrieval_fusion"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    with retrieval.connect_db(db_path) as conn:
        paper_rows = load_paper_rows(conn, [item["paper_id"] for item in candidate_pool])
    stage_timings["candidate_rows_load"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    ranked_results, evidence_packs = rerank_candidates(
        db_path=db_path,
        intent_frame=final_frame,
        candidate_pool=candidate_pool,
        paper_rows=paper_rows,
        sections_by_paper={},
        top_k=top_k,
        explain_limit=explain_limit,
    )
    stage_timings["rerank_and_explain"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    gap_report = build_gap_report(final_frame, ranked_results)
    stage_timings["gap_report"] = time.perf_counter() - stage_start
    stage_timings["total"] = time.perf_counter() - started_at

    return {
        "query": query,
        "follow_up_reply": follow_up_reply,
        "initial_intent_frame": initial_frame,
        "final_intent_frame": final_frame,
        "candidate_pool_size": len(candidate_pool),
        "sparse_results": list(sparse_results.values())[: min(20, len(sparse_results))],
        "dense_results": list(dense_results.values())[: min(20, len(dense_results))],
        "exact_results": list(exact_results.values())[: min(20, len(exact_results))],
        "intent_gap_report": gap_report,
        "stage_timings": {key: round(value, 4) for key, value in stage_timings.items()},
        "top_k_results": ranked_results,
        "paper_evidence_packs": {paper_id: evidence_packs[paper_id] for paper_id in [item["paper_id"] for item in ranked_results]},
    }


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
        "demo_query_count": len(demo_runs),
        "standard_query_count": len(standard_specs),
        "explanation_sample_count": len(explanation_samples),
        "output_dir": str(OUTPUT_DIR),
    }
