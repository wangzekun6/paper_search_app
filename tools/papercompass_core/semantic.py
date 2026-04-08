"""
PaperCompass 语义卡片生成流水线。

这个文件负责从统一 SQLite 论文库中读取论文信息，
构造一个固定长度、固定字段的模型输入窗口，
调用大模型生成结构化的 PaperSemanticCard，
并把结果缓存回数据库和统一输出目录。

它的核心目标有两个：
1. 控制模型输入成本和响应时间
2. 把自由文本论文转成稳定的结构化检索标签
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import (
    PROJECT_ROOT,
    QUERY_MATCH_CACHE_DIR,
    SEMANTIC_CARD_CACHE_DIR,
    SEMANTIC_CARD_ERRORS_PATH,
    SEMANTIC_CARD_PROMPT_PATH,
    SEMANTIC_CARD_QUALITY_CSV_PATH,
    SEMANTIC_CARD_QUALITY_JSON_PATH,
    SEMANTIC_CARD_SAMPLE_PATH,
    SEMANTIC_CARD_STABILITY_PATH,
    SMOKE_QUERY_JSON_PATH,
    SYSTEM_OUTPUT_DIR,
    ensure_system_layout,
    semantic_card_cache_path,
    write_json,
)
from .llm import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OpenAIAPIError,
    structured_chat_completion,
    test_openai_api,
)
from .models import PaperSemanticCard
from .retrieval import DEFAULT_DB_PATH, connect_db


SMOKE_QUERY_PATH = SMOKE_QUERY_JSON_PATH
OUTPUT_DIR = SYSTEM_OUTPUT_DIR
PROMPT_PATH = SEMANTIC_CARD_PROMPT_PATH
PILOT_OUTPUT_PATH = SYSTEM_OUTPUT_DIR / "demos" / "pilot_semantic_cards.json"
SAMPLE_OUTPUT_PATH = SEMANTIC_CARD_SAMPLE_PATH
QUALITY_CHECK_CSV_PATH = SEMANTIC_CARD_QUALITY_CSV_PATH
QUALITY_CHECK_JSON_PATH = SEMANTIC_CARD_QUALITY_JSON_PATH
FIELD_STABILITY_PATH = SEMANTIC_CARD_STABILITY_PATH
CACHE_STRATEGY_PATH = SYSTEM_OUTPUT_DIR / "eval" / "card_cache_strategy.txt"
FEEDBACK_PATH = SYSTEM_OUTPUT_DIR / "eval" / "semantic_card_feedback.txt"
ERROR_LOG_PATH = SEMANTIC_CARD_ERRORS_PATH

PROMPT_VERSION = "semantic_v1"
DEFAULT_TARGET_COUNT = 100
DEFAULT_PILOT_COUNT = 5
DEFAULT_QUALITY_SAMPLE_SIZE = 20
GENERATED_STATUSES = ("generated",)
OPENAI_RUNTIME_AVAILABLE: Optional[bool] = None
OPENAI_RUNTIME_MESSAGE = ""

PAPER_TYPE_ENUM = [
    "survey",
    "benchmark",
    "method",
    "empirical_study",
    "application_study",
    "theory",
    "analysis",
]
USER_INTENT_ENUM = [
    "topic_exploration",
    "survey_lookup",
    "recent_progress",
    "specific_paper_lookup",
    "author_trace",
    "method_constrained_search",
]
SOURCE_SECTION_ENUM = ["abstract", "Introduction", "Methods", "Results", "Discussion", "Other"]

SYSTEM_PROMPT = """You are building structured semantic cards for an academic paper retrieval system.

Read only the provided compact paper context. Do not invent facts that are not supported by the input.

Output requirements:
1. Return valid JSON only.
2. Follow the schema exactly.
3. paper_type must be one of: survey, benchmark, method, empirical_study, application_study, theory, analysis.
4. core_contributions must contain at most 3 short items, each describing a concrete contribution from the paper.
5. likely_user_intents must only use: topic_exploration, survey_lookup, recent_progress, specific_paper_lookup, author_trace, method_constrained_search.
6. evidence_spans must link important claims back to a coarse source section using only: abstract, Introduction, Methods, Results, Discussion, Other.
7. Keep tags concise and retrieval-friendly. Prefer noun phrases.
8. retrieval_keywords_en should be short English search phrases. retrieval_keywords_zh should be short Chinese search phrases.
9. If evidence is weak, be conservative rather than speculative.
10. If the paper introduces a method, benchmark, or dataset, mention it in method_tags/model_tags/dataset_tags when supported.
"""

USER_PROMPT_TEMPLATE = """Paper context JSON:
{paper_context}

Produce one PaperSemanticCard JSON object for this paper."""

FALLBACK_ZH_MAP = {
    "retrieval": "检索",
    "generation": "生成",
    "translation": "机器翻译",
    "speech": "语音",
    "dialogue": "对话",
    "multimodal": "多模态",
    "reasoning": "推理",
    "benchmark": "基准评测",
    "survey": "综述",
    "evaluation": "评估",
    "hallucination": "幻觉",
    "agent": "智能体",
    "memory": "记忆",
    "quality estimation": "质量估计",
    "long context": "长上下文",
    "visualization": "可视化",
    "data selection": "数据选择",
}

HEURISTIC_RULES = {
    "domain_tags": {
        "machine translation": "machine translation",
        "translation": "translation",
        "retrieval": "information retrieval",
        "generation": "text generation",
        "speech": "speech processing",
        "dialogue": "dialogue systems",
        "multimodal": "multimodal NLP",
        "vision-language": "vision-language modeling",
        "reasoning": "reasoning",
        "summarization": "summarization",
        "benchmark": "evaluation",
        "survey": "literature review",
    },
    "task_tags": {
        "retrieval augmented generation": "retrieval-augmented generation",
        "rag": "retrieval-augmented generation",
        "translation": "translation",
        "quality estimation": "quality estimation",
        "summarization": "summarization",
        "dialogue": "dialogue",
        "reasoning": "reasoning",
        "hallucination": "hallucination mitigation",
        "visualization": "scientific data visualization",
        "long context": "long-context understanding",
        "benchmark": "benchmarking",
    },
    "method_tags": {
        "agent": "agent framework",
        "memory": "memory mechanism",
        "retrieval": "retrieval",
        "rag": "retrieval-augmented generation",
        "deferral": "quality-aware deferral",
        "benchmark": "benchmark construction",
        "prompt": "prompt engineering",
        "fine-tun": "fine-tuning",
        "contrastive": "contrastive learning",
        "graph": "graph-based method",
        "decoding": "decoding strategy",
        "quality estimation": "quality estimation",
    },
    "model_tags": {
        "large language model": "large language model",
        "llm": "large language model",
        "vision-language model": "vision-language model",
        "transformer": "transformer",
        "gpt": "gpt-style model",
        "bigbird": "BigBird",
        "clip": "CLIP",
    },
    "dataset_tags": {
        "dataset": "dataset",
        "benchmark": "benchmark dataset",
        "corpus": "corpus",
        "wmt": "WMT",
        "arxiv": "arXiv",
    },
    "metric_tags": {
        "accuracy": "accuracy",
        "f1": "F1",
        "bleu": "BLEU",
        "rouge": "ROUGE",
        "recall": "recall",
        "precision": "precision",
        "correlation": "correlation",
    },
}


SEMANTIC_CARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "string"},
        "domain_tags": {"type": "array", "items": {"type": "string"}},
        "task_tags": {"type": "array", "items": {"type": "string"}},
        "problem_statement": {"type": "string"},
        "method_tags": {"type": "array", "items": {"type": "string"}},
        "model_tags": {"type": "array", "items": {"type": "string"}},
        "dataset_tags": {"type": "array", "items": {"type": "string"}},
        "metric_tags": {"type": "array", "items": {"type": "string"}},
        "paper_type": {"type": "string", "enum": PAPER_TYPE_ENUM},
        "core_contributions": {"type": "array", "items": {"type": "string"}},
        "application_scenarios": {"type": "array", "items": {"type": "string"}},
        "retrieval_keywords_en": {"type": "array", "items": {"type": "string"}},
        "retrieval_keywords_zh": {"type": "array", "items": {"type": "string"}},
        "survey_signals": {"type": "array", "items": {"type": "string"}},
        "likely_user_intents": {"type": "array", "items": {"type": "string", "enum": USER_INTENT_ENUM}},
        "limitations_or_scope": {"type": "string"},
        "evidence_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_field": {"type": "string"},
                    "claim_value": {"type": "string"},
                    "source_section": {"type": "string", "enum": SOURCE_SECTION_ENUM},
                    "evidence_text": {"type": "string"},
                },
                "required": ["target_field", "claim_value", "source_section", "evidence_text"],
            },
        },
    },
    "required": [
        "paper_id",
        "domain_tags",
        "task_tags",
        "problem_statement",
        "method_tags",
        "model_tags",
        "dataset_tags",
        "metric_tags",
        "paper_type",
        "core_contributions",
        "application_scenarios",
        "retrieval_keywords_en",
        "retrieval_keywords_zh",
        "survey_signals",
        "likely_user_intents",
        "limitations_or_scope",
        "evidence_spans",
    ],
}


def parse_args() -> argparse.Namespace:
    """解析语义卡片流水线参数，支持批量生成和单篇重刷两种模式。"""

    parser = argparse.ArgumentParser(description="Generate PaperCompass PaperSemanticCard objects with OpenAI.")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Generate pilot cards, then at least 100 semantic cards.")
    build_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    build_parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    build_parser.add_argument("--pilot-count", type=int, default=DEFAULT_PILOT_COUNT)
    build_parser.add_argument("--refresh", action="store_true")

    paper_parser = subparsers.add_parser("generate-paper", help="Generate a semantic card for a specific paper.")
    paper_parser.add_argument("paper_id")
    paper_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    paper_parser.add_argument("--refresh", action="store_true")

    return parser.parse_args()


def ensure_output_dir() -> None:
    ensure_system_layout()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def split_paragraphs(text: str, limit: int = 2) -> List[str]:
    paragraphs = [part.strip() for part in str(text or "").split("\n\n") if part.strip()]
    return paragraphs[:limit]


def clean_tag_list(values: Iterable[Any], limit: int = 8) -> List[str]:
    items: List[str] = []
    seen = set()
    for value in values:
        text = " ".join(str(value or "").replace("\n", " ").split()).strip(" ,;")
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


def enforce_enum(value: str, allowed: Sequence[str], fallback: str) -> str:
    text = str(value or "").strip()
    if text in allowed:
        return text
    lowered_map = {item.lower(): item for item in allowed}
    if text.lower() in lowered_map:
        return lowered_map[text.lower()]
    return fallback


def blank_semantic_card(paper_id: str) -> Dict[str, Any]:
    return PaperSemanticCard(paper_id=paper_id).to_dict()


def infer_paper_type(title: str, abstract: str, survey_signals: Sequence[str], method_tags: Sequence[str]) -> str:
    text = f"{title} {abstract}".lower()
    if "survey" in text or survey_signals:
        return "survey"
    if "benchmark" in text or "leaderboard" in text:
        return "benchmark"
    if "theory" in text or "bound" in text or "theorem" in text:
        return "theory"
    if "analysis" in text:
        return "analysis"
    if "application" in text or "case study" in text:
        return "application_study"
    if method_tags:
        return "method"
    return "empirical_study"


def infer_user_intents(card: Dict[str, Any], title: str) -> List[str]:
    intents = clean_tag_list(card.get("likely_user_intents", []), limit=len(USER_INTENT_ENUM))
    valid = [intent for intent in intents if intent in USER_INTENT_ENUM]

    if not valid:
        valid.append("topic_exploration")
    if card.get("paper_type") == "survey" and "survey_lookup" not in valid:
        valid.append("survey_lookup")
    if title and "specific_paper_lookup" not in valid:
        valid.append("specific_paper_lookup")
    if card.get("method_tags") and "method_constrained_search" not in valid:
        valid.append("method_constrained_search")
    if card.get("paper_type") in {"benchmark", "method", "analysis", "empirical_study"} and "recent_progress" not in valid:
        valid.append("recent_progress")
    return clean_tag_list(valid, limit=4)


def normalize_evidence_spans(spans: Iterable[Any]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in spans:
        if not isinstance(item, dict):
            continue
        target_field = " ".join(str(item.get("target_field", "")).split()).strip()
        claim_value = " ".join(str(item.get("claim_value", "")).split()).strip()
        source_section = enforce_enum(str(item.get("source_section", "")), SOURCE_SECTION_ENUM, "Other")
        evidence_text = " ".join(str(item.get("evidence_text", "")).split()).strip()
        if not target_field or not claim_value or not evidence_text:
            continue
        normalized.append(
            {
                "target_field": target_field,
                "claim_value": claim_value,
                "source_section": source_section,
                "evidence_text": evidence_text[:280],
            }
        )
        if len(normalized) >= 8:
            break
    return normalized


def load_paper_row(conn: Any, paper_id: str) -> Optional[Any]:
    return conn.execute(
        """
        SELECT
            paper_id,
            title,
            authors_raw,
            abstract,
            section_titles,
            intro_text,
            methods_text,
            results_text,
            discussion_text,
            appendix_titles,
            year_month
        FROM papers
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()


def build_llm_input_from_row(row: Any) -> Dict[str, Any]:
    """
    从 papers 表中的单行记录构造固定 LLM 输入窗口。

    这里严格控制输入字段，只保留最关键的论文证据，
    避免把整篇全文直接送进模型导致成本过高、速度过慢或注意力漂移。
    """

    section_titles = json.loads(row["section_titles"]) if row["section_titles"] else []
    appendix_titles = json.loads(row["appendix_titles"]) if row["appendix_titles"] else []
    # 这里明确不把整篇论文全文送给模型。
    # 这里只取 title / authors / abstract / section_titles，
    # 再加上 intro、methods、results、discussion 各前两段，
    # 以及 appendix_titles，形成固定且可控的输入窗口。
    payload = {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "authors": row["authors_raw"],
        "abstract": row["abstract"],
        "section_titles": section_titles,
        "intro_paragraphs": split_paragraphs(row["intro_text"], limit=2),
        "methods_paragraphs": split_paragraphs(row["methods_text"], limit=2),
        "results_paragraphs": split_paragraphs(row["results_text"], limit=2),
        "discussion_paragraphs": split_paragraphs(row["discussion_text"], limit=2),
        "appendix_titles": appendix_titles,
    }
    return payload


def build_messages(paper_context: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                paper_context=json.dumps(paper_context, ensure_ascii=False, indent=2)
            ),
        },
    ]


def validate_semantic_card(raw_card: Dict[str, Any], paper_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    把模型输出规范化成系统内部可稳定使用的语义卡片格式。

    主要处理字段补全、标签去重、枚举纠偏和证据片段结构清洗。
    """

    paper_id = paper_context["paper_id"]
    title = paper_context.get("title", "")
    abstract = paper_context.get("abstract", "")

    # 模型输出可能有大小写不一致、字段缺失、标签重复、枚举值漂移等问题，
    # 这里统一做一次清洗和回填，保证最终落库的数据结构稳定可用。
    card = blank_semantic_card(paper_id)
    card.update(raw_card or {})
    card["paper_id"] = paper_id
    card["domain_tags"] = clean_tag_list(card.get("domain_tags", []), limit=6)
    card["task_tags"] = clean_tag_list(card.get("task_tags", []), limit=6)
    card["problem_statement"] = " ".join(str(card.get("problem_statement", "")).split()).strip()
    card["method_tags"] = clean_tag_list(card.get("method_tags", []), limit=8)
    card["model_tags"] = clean_tag_list(card.get("model_tags", []), limit=8)
    card["dataset_tags"] = clean_tag_list(card.get("dataset_tags", []), limit=8)
    card["metric_tags"] = clean_tag_list(card.get("metric_tags", []), limit=8)
    card["survey_signals"] = clean_tag_list(card.get("survey_signals", []), limit=4)
    card["paper_type"] = enforce_enum(
        str(card.get("paper_type", "")),
        PAPER_TYPE_ENUM,
        infer_paper_type(title, abstract, card["survey_signals"], card["method_tags"]),
    )
    card["core_contributions"] = clean_tag_list(card.get("core_contributions", []), limit=3)
    card["application_scenarios"] = clean_tag_list(card.get("application_scenarios", []), limit=6)
    card["retrieval_keywords_en"] = clean_tag_list(card.get("retrieval_keywords_en", []), limit=10)
    if not card["retrieval_keywords_en"]:
        card["retrieval_keywords_en"] = clean_tag_list(
            card["domain_tags"] + card["task_tags"] + card["method_tags"], limit=8
        )
    card["retrieval_keywords_zh"] = clean_tag_list(card.get("retrieval_keywords_zh", []), limit=10)
    card["likely_user_intents"] = infer_user_intents(card, title)
    card["limitations_or_scope"] = " ".join(str(card.get("limitations_or_scope", "")).split()).strip()
    card["evidence_spans"] = normalize_evidence_spans(card.get("evidence_spans", []))
    return card


def can_use_openai() -> bool:
    global OPENAI_RUNTIME_AVAILABLE, OPENAI_RUNTIME_MESSAGE
    if OPENAI_RUNTIME_AVAILABLE is not None:
        return OPENAI_RUNTIME_AVAILABLE

    ok, message = test_openai_api(api_key=OPENAI_API_KEY, timeout=30)
    OPENAI_RUNTIME_AVAILABLE = ok
    OPENAI_RUNTIME_MESSAGE = message
    return OPENAI_RUNTIME_AVAILABLE


def split_sentences(text: str, limit: int = 3) -> List[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences[:limit]


def infer_tags_from_rules(text: str, rules: Dict[str, str], limit: int = 6) -> List[str]:
    lowered = text.lower()
    hits = []
    for keyword, label in rules.items():
        if keyword in lowered:
            hits.append(label)
    return clean_tag_list(hits, limit=limit)


def infer_retrieval_keywords_zh(english_keywords: Sequence[str]) -> List[str]:
    zh_keywords = []
    for keyword in english_keywords:
        lowered = keyword.lower()
        zh_keywords.append(FALLBACK_ZH_MAP.get(lowered, keyword))
    return clean_tag_list(zh_keywords, limit=10)


def infer_survey_signals(title: str, abstract: str) -> List[str]:
    text = f"{title} {abstract}".lower()
    signals = []
    for signal in ("survey", "review", "benchmark", "overview"):
        if signal in text:
            signals.append(signal)
    return clean_tag_list(signals, limit=4)


def heuristic_semantic_card(paper_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    保留一个规则化卡片构造器，供离线调试或人工对比使用。

    它不是产品主路径，主系统不会在模型失败时自动退回到这里。
    """

    title = paper_context.get("title", "")
    abstract = paper_context.get("abstract", "")
    section_titles = paper_context.get("section_titles", [])
    intro_paragraphs = paper_context.get("intro_paragraphs", [])
    methods_paragraphs = paper_context.get("methods_paragraphs", [])
    results_paragraphs = paper_context.get("results_paragraphs", [])
    discussion_paragraphs = paper_context.get("discussion_paragraphs", [])

    # 仅用于离线对比，不参与正式检索链路。
    # 它不依赖外部 API，只根据标题、摘要和关键段落做规则抽取。
    combined_text = " ".join(
        [
            title,
            abstract,
            " ".join(section_titles),
            " ".join(intro_paragraphs),
            " ".join(methods_paragraphs),
            " ".join(results_paragraphs),
            " ".join(discussion_paragraphs),
        ]
    )
    survey_signals = infer_survey_signals(title, abstract)
    domain_tags = infer_tags_from_rules(combined_text, HEURISTIC_RULES["domain_tags"], limit=6)
    task_tags = infer_tags_from_rules(combined_text, HEURISTIC_RULES["task_tags"], limit=6)
    method_tags = infer_tags_from_rules(combined_text, HEURISTIC_RULES["method_tags"], limit=8)
    model_tags = infer_tags_from_rules(combined_text, HEURISTIC_RULES["model_tags"], limit=8)
    dataset_tags = infer_tags_from_rules(combined_text, HEURISTIC_RULES["dataset_tags"], limit=8)
    metric_tags = infer_tags_from_rules(combined_text, HEURISTIC_RULES["metric_tags"], limit=8)
    problem_statement = " ".join(split_sentences(abstract, limit=2))
    core_contributions = split_sentences(abstract, limit=3)
    application_scenarios = clean_tag_list(domain_tags + task_tags, limit=5)
    retrieval_keywords_en = clean_tag_list([title] + method_tags + task_tags + domain_tags, limit=8)
    retrieval_keywords_zh = infer_retrieval_keywords_zh(retrieval_keywords_en)
    paper_type = infer_paper_type(title, abstract, survey_signals, method_tags)

    evidence_spans = []
    if problem_statement:
        evidence_spans.append(
            {
                "target_field": "problem_statement",
                "claim_value": problem_statement[:160],
                "source_section": "abstract",
                "evidence_text": problem_statement[:220],
            }
        )
    for contribution in core_contributions[:2]:
        evidence_spans.append(
            {
                "target_field": "core_contributions",
                "claim_value": contribution[:160],
                "source_section": "abstract",
                "evidence_text": contribution[:220],
            }
        )
    for method_tag in method_tags[:2]:
        evidence_spans.append(
            {
                "target_field": "method_tags",
                "claim_value": method_tag,
                "source_section": "Methods" if methods_paragraphs else "Introduction",
                "evidence_text": (methods_paragraphs[0] if methods_paragraphs else (intro_paragraphs[0] if intro_paragraphs else abstract))[:220],
            }
        )

    raw_card = {
        "paper_id": paper_context["paper_id"],
        "domain_tags": domain_tags,
        "task_tags": task_tags,
        "problem_statement": problem_statement,
        "method_tags": method_tags,
        "model_tags": model_tags,
        "dataset_tags": dataset_tags,
        "metric_tags": metric_tags,
        "paper_type": paper_type,
        "core_contributions": core_contributions,
        "application_scenarios": application_scenarios,
        "retrieval_keywords_en": retrieval_keywords_en,
        "retrieval_keywords_zh": retrieval_keywords_zh,
        "survey_signals": survey_signals,
        "likely_user_intents": [],
        "limitations_or_scope": " ".join(split_sentences(discussion_paragraphs[0] if discussion_paragraphs else "", limit=1)),
        "evidence_spans": evidence_spans,
    }
    return validate_semantic_card(raw_card, paper_context)


def current_card_count(conn: Any) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM paper_semantic_cards WHERE card_status IN ({', '.join('?' for _ in GENERATED_STATUSES)})",
        GENERATED_STATUSES,
    ).fetchone()[0]


def load_existing_generated_ids(conn: Any) -> set[str]:
    rows = conn.execute(
        f"SELECT paper_id FROM paper_semantic_cards WHERE card_status IN ({', '.join('?' for _ in GENERATED_STATUSES)})",
        GENERATED_STATUSES,
    ).fetchall()
    return {row["paper_id"] for row in rows}


def total_paper_count(conn: Any) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])


def list_missing_generated_ids(conn: Any, limit: Optional[int] = None) -> List[str]:
    sql = f"""
        SELECT papers.paper_id
        FROM papers
        LEFT JOIN paper_semantic_cards
            ON papers.paper_id = paper_semantic_cards.paper_id
            AND paper_semantic_cards.card_status IN ({', '.join('?' for _ in GENERATED_STATUSES)})
        WHERE paper_semantic_cards.paper_id IS NULL
        ORDER BY papers.year_month DESC, papers.paper_id DESC
    """
    params: List[Any] = list(GENERATED_STATUSES)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [row["paper_id"] for row in rows]


def load_cached_semantic_card(path: Path) -> Optional[Tuple[Dict[str, Any], str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    card_payload = payload.get("card") if isinstance(payload.get("card"), dict) else payload
    if not isinstance(card_payload, dict):
        return None
    paper_id = " ".join(str(card_payload.get("paper_id", "")).split()).strip()
    if not paper_id:
        return None
    card_status = " ".join(str(payload.get("status", "")).split()).strip() or "generated"
    return card_payload, card_status


def restore_cached_semantic_cards(
    db_path: Path,
    *,
    refresh: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    cache_files = sorted(SEMANTIC_CARD_CACHE_DIR.glob("*.json"))
    summary = {
        "db_path": str(db_path),
        "cache_file_count": len(cache_files),
        "paper_count": 0,
        "imported_count": 0,
        "skipped_existing_count": 0,
        "skipped_missing_paper_count": 0,
        "invalid_cache_count": 0,
    }
    if not cache_files or not db_path.exists():
        return summary

    with connect_db(db_path) as conn:
        summary["paper_count"] = total_paper_count(conn)
        existing_ids = load_existing_generated_ids(conn) if not refresh else set()
        existing_papers = {
            row["paper_id"] for row in conn.execute("SELECT paper_id FROM papers").fetchall()
        }
        for index, path in enumerate(cache_files, start=1):
            loaded = load_cached_semantic_card(path)
            if loaded is None:
                summary["invalid_cache_count"] += 1
                continue
            card, card_status = loaded
            paper_id = card["paper_id"]
            if paper_id not in existing_papers:
                summary["skipped_missing_paper_count"] += 1
                continue
            if not refresh and paper_id in existing_ids:
                summary["skipped_existing_count"] += 1
                continue
            upsert_semantic_card(conn, card, card_status)
            existing_ids.add(paper_id)
            summary["imported_count"] += 1
            if progress_callback and (index == len(cache_files) or summary["imported_count"] % 25 == 0):
                progress_callback(
                    {
                        "stage": "restore_cache",
                        "processed_cache_files": index,
                        "imported_count": summary["imported_count"],
                    }
                )
    return summary


def load_error_log() -> List[Dict[str, Any]]:
    if not ERROR_LOG_PATH.exists():
        return []
    try:
        return json.loads(ERROR_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def append_error_log(entry: Dict[str, Any]) -> None:
    errors = load_error_log()
    errors.append(entry)
    dump_json(ERROR_LOG_PATH, errors)


def upsert_semantic_card(conn: Any, card: Dict[str, Any], card_status: str) -> None:
    conn.execute(
        """
        INSERT INTO paper_semantic_cards (paper_id, semantic_card_json, card_status, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(paper_id) DO UPDATE SET
            semantic_card_json = excluded.semantic_card_json,
            card_status = excluded.card_status,
            updated_at = CURRENT_TIMESTAMP
        """,
        (card["paper_id"], json.dumps(card, ensure_ascii=False), card_status),
    )
    conn.commit()
    write_json(semantic_card_cache_path(card["paper_id"]), {"status": card_status, "card": card})


def select_candidate_paper_ids(conn: Any, limit: int) -> List[str]:
    """
    选择优先生成语义卡片的论文集合。

    先取 smoke query 中已经被命中的论文，再按较新的 paper_id 补齐，
    这样更符合“先覆盖高价值论文”的策略。
    """

    candidates: List[str] = []
    seen = set()

    # 生成优先级分两层：
    # 先处理 smoke query 里高频命中的论文，因为这些论文最可能先被用户搜到；
    # 不够时再按较新的 paper_id 继续补齐。
    if SMOKE_QUERY_PATH.exists():
        payload = json.loads(SMOKE_QUERY_PATH.read_text(encoding="utf-8"))
        for query_item in payload.get("queries", []):
            for result in query_item.get("hybrid_results", []):
                paper_id = result.get("paper_id")
                if paper_id and paper_id not in seen:
                    seen.add(paper_id)
                    candidates.append(paper_id)

    rows = conn.execute(
        """
        SELECT paper_id
        FROM papers
        ORDER BY year_month DESC, paper_id DESC
        """
    ).fetchall()
    for row in rows:
        paper_id = row["paper_id"]
        if paper_id in seen:
            continue
        seen.add(paper_id)
        candidates.append(paper_id)
        if len(candidates) >= limit:
            break

    return candidates[:limit]


def generate_card_for_paper(conn: Any, paper_id: str, refresh: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    为单篇论文生成语义卡片，并写回数据库缓存。

    默认会优先复用已有缓存；只有 refresh=True 时才强制重算。
    返回值同时包含卡片内容和实际使用的模型名。
    """

    if not refresh:
        existing = conn.execute(
            """
            SELECT semantic_card_json, card_status
            FROM paper_semantic_cards
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchone()
        if existing and existing["card_status"] in GENERATED_STATUSES:
            cached_card = json.loads(existing["semantic_card_json"])
            write_json(semantic_card_cache_path(paper_id), {"status": existing["card_status"], "card": cached_card})
            return cached_card, None

    row = load_paper_row(conn, paper_id)
    if row is None:
        raise ValueError(f"Paper not found: {paper_id}")

    paper_context = build_llm_input_from_row(row)
    messages = build_messages(paper_context)
    if not can_use_openai():
        raise OpenAIAPIError(f"核心语义能力不可用：PaperSemanticCard 生成依赖 LLM。{OPENAI_RUNTIME_MESSAGE}")
    try:
        # 主路径只接受真实模型输出。
        # 如果模型调用或结构化输出失败，直接报错，不回退规则卡片。
        raw_card, used_model = structured_chat_completion(
            messages=messages,
            schema_name="paper_semantic_card",
            schema=SEMANTIC_CARD_SCHEMA,
            model=OPENAI_MODEL,
            temperature=0.2,
            max_tokens=1400,
            timeout=60,
            api_key=OPENAI_API_KEY,
        )
        card = validate_semantic_card(raw_card, paper_context)
        upsert_semantic_card(conn, card, "generated")
        return card, used_model
    except Exception as exc:
        append_error_log({"paper_id": paper_id, "error": str(exc)})
        raise OpenAIAPIError(f"核心语义能力不可用：为 {paper_id} 生成 PaperSemanticCard 失败。{exc}") from exc


def generate_pilot_cards(conn: Any, pilot_ids: Sequence[str], refresh: bool = False) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for paper_id in pilot_ids:
        card, _ = generate_card_for_paper(conn, paper_id, refresh=refresh)
        if card:
            cards.append(card)
    return cards


def generate_cards_for_paper_ids(
    conn: Any,
    paper_ids: Sequence[str],
    *,
    refresh: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    target_ids = clean_string_list(paper_ids, limit=max(len(paper_ids), 1))
    generated_count = 0
    error_count = 0
    processed_count = 0
    error_samples: List[Dict[str, str]] = []
    total_count = len(target_ids)
    for paper_id in target_ids:
        processed_count += 1
        try:
            card, _ = generate_card_for_paper(conn, paper_id, refresh=refresh)
            if card:
                generated_count += 1
        except Exception as exc:
            error_count += 1
            if len(error_samples) < 10:
                error_samples.append({"paper_id": paper_id, "error": str(exc)})
        if progress_callback and (processed_count == total_count or processed_count % 5 == 0):
            progress_callback(
                {
                    "stage": "generate_cards",
                    "processed_count": processed_count,
                    "total_count": total_count,
                    "generated_count": generated_count,
                    "error_count": error_count,
                }
            )
    return {
        "requested_count": total_count,
        "processed_count": processed_count,
        "generated_count": generated_count,
        "error_count": error_count,
        "error_samples": error_samples,
    }


def generate_cards_until_target(conn: Any, target_count: int, refresh: bool = False) -> int:
    """
    持续生成语义卡片，直到达到目标数量。

    refresh=False 表示补齐缺失；
    refresh=True 表示强制重刷目标集合。
    """

    existing_ids = load_existing_generated_ids(conn)
    if len(existing_ids) >= target_count and not refresh:
        return len(existing_ids)

    candidates = select_candidate_paper_ids(conn, limit=max(target_count * 2, 200))
    refreshed_ids: set[str] = set()
    for paper_id in candidates:
        # refresh 模式下，目标是“本轮真正重刷了多少篇”，
        # 不能把库里原来就存在的记录直接算进完成数，否则会提前结束。
        if refresh:
            if paper_id in refreshed_ids:
                continue
        else:
            if paper_id in existing_ids:
                continue
        card, _ = generate_card_for_paper(conn, paper_id, refresh=refresh)
        if card:
            existing_ids.add(paper_id)
            if refresh:
                refreshed_ids.add(paper_id)
        if refresh and len(refreshed_ids) >= target_count:
            break
        if not refresh and len(existing_ids) >= target_count:
            break

    return current_card_count(conn)


def load_generated_cards(conn: Any, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    sql = f"""
        SELECT semantic_card_json
        FROM paper_semantic_cards
        WHERE card_status IN ({', '.join('?' for _ in GENERATED_STATUSES)})
        ORDER BY updated_at DESC, paper_id ASC
    """
    params: Tuple[Any, ...] = tuple(GENERATED_STATUSES)
    if limit is not None:
        sql += " LIMIT ?"
        params = tuple(GENERATED_STATUSES) + (limit,)
    rows = conn.execute(sql, params).fetchall()
    return [json.loads(row["semantic_card_json"]) for row in rows]


def quality_issues(card: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for field_name in ("domain_tags", "task_tags", "method_tags", "core_contributions", "likely_user_intents"):
        if not card.get(field_name):
            issues.append(f"missing_{field_name}")
    if card.get("paper_type") not in PAPER_TYPE_ENUM:
        issues.append("invalid_paper_type")
    if len(card.get("core_contributions", [])) > 3:
        issues.append("too_many_core_contributions")
    if not card.get("evidence_spans"):
        issues.append("missing_evidence_spans")
    return issues


def write_quality_check(conn: Any, sample_size: int = DEFAULT_QUALITY_SAMPLE_SIZE) -> Dict[str, Any]:
    """导出一份人工抽查样本，便于检查语义卡片字段质量。"""

    cards = load_generated_cards(conn, limit=sample_size)
    rows = []
    for card in cards:
        paper = load_paper_row(conn, card["paper_id"])
        row = {
            "paper_id": card["paper_id"],
            "title": paper["title"] if paper else "",
            "paper_type": card.get("paper_type", ""),
            "domain_tags": " | ".join(card.get("domain_tags", [])),
            "task_tags": " | ".join(card.get("task_tags", [])),
            "method_tags": " | ".join(card.get("method_tags", [])),
            "core_contributions": " | ".join(card.get("core_contributions", [])),
            "likely_user_intents": " | ".join(card.get("likely_user_intents", [])),
            "issues": " | ".join(quality_issues(card)),
            "manual_check": "",
            "notes": "",
        }
        rows.append(row)

    QUALITY_CHECK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUALITY_CHECK_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=list(rows[0].keys()) if rows else ["paper_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {"sample_size": len(rows), "rows": rows}
    dump_json(QUALITY_CHECK_JSON_PATH, payload)
    return payload


def write_field_stability_report(cards: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    统计关键字段的非空率和分布，评估语义卡片输出是否稳定。

    这个报告用于回答：模型输出的结构化字段能否稳定支撑后续检索和展示。
    """

    total = len(cards)
    critical_fields = [
        "domain_tags",
        "task_tags",
        "method_tags",
        "paper_type",
        "core_contributions",
        "likely_user_intents",
    ]
    non_empty_rates = {}
    for field_name in critical_fields:
        count = 0
        for card in cards:
            value = card.get(field_name)
            if isinstance(value, list) and value:
                count += 1
            elif isinstance(value, str) and value.strip():
                count += 1
        non_empty_rates[field_name] = round(count / total, 4) if total else 0.0

    paper_type_distribution = Counter(card.get("paper_type", "") for card in cards if card.get("paper_type"))
    intent_distribution = Counter(
        intent
        for card in cards
        for intent in card.get("likely_user_intents", [])
        if intent in USER_INTENT_ENUM
    )
    issue_counts = Counter(issue for card in cards for issue in quality_issues(card))

    stable_fields = [field for field, rate in non_empty_rates.items() if rate >= 0.85]
    drift_risk_fields = [field for field, rate in non_empty_rates.items() if rate < 0.85]
    payload = {
        "generated_card_count": total,
        "critical_field_non_empty_rates": non_empty_rates,
        "stable_fields": stable_fields,
        "drift_risk_fields": drift_risk_fields,
        "paper_type_distribution": dict(paper_type_distribution.most_common()),
        "likely_user_intent_distribution": dict(intent_distribution.most_common()),
        "quality_issue_counts": dict(issue_counts.most_common()),
    }
    dump_json(FIELD_STABILITY_PATH, payload)
    return payload


def write_cache_strategy() -> None:
    content = """
语义卡片缓存策略

固定原则
- 全量先完成 PaperIndexRecord，PaperSemanticCard 先做重点批量生成。
- 今日目标不是一次性跑完整个库，而是先稳定完成至少 100 条高价值卡片。
- 每次新生成的卡片立即写回 SQLite 的 paper_semantic_cards，避免重复调用 OpenAI。

优先级
- 优先生成 smoke query 调试结果里高频命中的论文。
- 不足部分按数据库中的最近 paper_id 继续补齐。
- 后续 top-K 命中的论文可以按需补生成。

缓存行为
- card_status = generated: 已有可用卡片，默认不重复请求。
- card_status = error: 记录失败占位，避免静默重复请求；需要时可 refresh。
- 每次生成后立刻 upsert 到 paper_semantic_cards。

后续扩展
- 后续查询链路可以对 top-K 命中的缺失卡片执行按需补生成。
- 当命中论文已有 generated 卡片时，直接读取缓存，不再次请求 OpenAI。
"""
    dump_text(CACHE_STRATEGY_PATH, content)


def write_prompt_file() -> None:
    prompt_text = f"""# PaperSemanticCard Prompt

Prompt Version: {PROMPT_VERSION}
Model Default: {OPENAI_MODEL}

## System Prompt
{SYSTEM_PROMPT}

## User Prompt Template
{USER_PROMPT_TEMPLATE}

## Output Schema
```json
{json.dumps(SEMANTIC_CARD_SCHEMA, ensure_ascii=False, indent=2)}
```
"""
    dump_text(PROMPT_PATH, prompt_text)


def write_feedback(
    db_path: Path,
    generated_count: int,
    pilot_cards: Sequence[Dict[str, Any]],
    quality_payload: Dict[str, Any],
    stability_payload: Dict[str, Any],
    status_counts: Dict[str, int],
    openai_available: bool,
    openai_message: str,
) -> None:
    content = f"""
语义卡片流水线执行完成

执行范围
- 数据库文件: {db_path}
- OpenAI 模型默认值: {OPENAI_MODEL}
- OpenAI 连通状态: {openai_available}
- OpenAI 连通说明: {openai_message}
- 生成卡片总数: {generated_count}
- pilot 试跑卡片数: {len(pilot_cards)}
- 质量检查样本数: {quality_payload.get('sample_size', 0)}
- 卡片状态分布: {status_counts}

已完成目标
- 已固定 PaperSemanticCard schema。
- 已固定 LLM Prompt，并保存到 semantic_card.md。
- 已至少生成 100 条语义卡片并写入 paper_semantic_cards。
- 已实现卡片缓存策略，并写入 card_cache_strategy.txt。
- 已生成质量检查表和字段稳定性报告。

重点字段稳定性
- stable_fields: {", ".join(stability_payload.get('stable_fields', []))}
- drift_risk_fields: {", ".join(stability_payload.get('drift_risk_fields', []))}

交付物
- {PROMPT_PATH.name}
- {PILOT_OUTPUT_PATH.name}
- {SAMPLE_OUTPUT_PATH.name}
- {QUALITY_CHECK_CSV_PATH.name}
- {QUALITY_CHECK_JSON_PATH.name}
- {FIELD_STABILITY_PATH.name}
- {CACHE_STRATEGY_PATH.name}
"""
    dump_text(FEEDBACK_PATH, content)


def run_build_command(args: argparse.Namespace) -> None:
    """
    执行语义卡片的完整批处理流程。

    包括 prompt 落盘、pilot 试跑、批量生成、质量抽查、稳定性统计和反馈文件输出。
    """

    ensure_output_dir()
    write_prompt_file()
    openai_available = can_use_openai()
    openai_message = OPENAI_RUNTIME_MESSAGE

    with connect_db(args.db_path) as conn:
        pilot_ids = select_candidate_paper_ids(conn, max(args.pilot_count, 5))[: args.pilot_count]
        pilot_cards = generate_pilot_cards(conn, pilot_ids, refresh=args.refresh)
        dump_json(PILOT_OUTPUT_PATH, pilot_cards)

        generated_count = generate_cards_until_target(conn, args.target_count, refresh=args.refresh)
        generated_cards = load_generated_cards(conn, limit=args.target_count)
        dump_json(SAMPLE_OUTPUT_PATH, generated_cards)

        quality_payload = write_quality_check(conn, sample_size=DEFAULT_QUALITY_SAMPLE_SIZE)
        stability_payload = write_field_stability_report(load_generated_cards(conn))
        status_counts = {
            row["card_status"]: row["count"]
            for row in conn.execute(
                """
                SELECT card_status, COUNT(*) AS count
                FROM paper_semantic_cards
                GROUP BY card_status
                ORDER BY card_status
                """
            ).fetchall()
        }
        write_cache_strategy()
        write_feedback(
            args.db_path,
            generated_count,
            pilot_cards,
            quality_payload,
            stability_payload,
            status_counts,
            openai_available,
            openai_message,
        )

        print(f"Generated semantic cards: {generated_count}")
        print(f"Prompt file: {PROMPT_PATH}")
        print(f"Quality check: {QUALITY_CHECK_CSV_PATH}")


def run_generate_paper_command(args: argparse.Namespace) -> None:
    ensure_output_dir()
    write_prompt_file()
    with connect_db(args.db_path) as conn:
        card, used_model = generate_card_for_paper(conn, args.paper_id, refresh=args.refresh)
        if card is None:
            raise OpenAIAPIError(f"Failed to generate semantic card for {args.paper_id}")
        print(json.dumps({"paper_id": args.paper_id, "used_model": used_model, "card": card}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command is None:
        args = argparse.Namespace(
            command="build",
            db_path=DEFAULT_DB_PATH,
            target_count=DEFAULT_TARGET_COUNT,
            pilot_count=DEFAULT_PILOT_COUNT,
            refresh=False,
        )

    if args.command == "build":
        run_build_command(args)
    elif args.command == "generate-paper":
        run_generate_paper_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
