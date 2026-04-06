"""
PaperCompass Day 5 核心方法主链路。

负责把前四项能力串起来：
query -> 意图解析 -> 聚合追问 -> 三路检索 -> gap 分析 -> 意图重排 -> 结果解释
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import day2_pipeline as day2
import papercompass_intent as intent
from openai_helpers import OPENAI_API_KEY, OPENAI_MODEL, structured_chat_completion, test_openai_api


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "day5_outputs"
EXPLANATION_PROMPT_PATH = OUTPUT_DIR / "ranking_explanation_prompt.md"
CHAIN_DEMOS_PATH = OUTPUT_DIR / "core_chain_demo_runs.json"
GAP_REPORTS_PATH = OUTPUT_DIR / "intent_gap_reports.json"
RANK_RESULTS_PATH = OUTPUT_DIR / "intent_aware_rank_results_topk.json"
FEEDBACK_PATH = OUTPUT_DIR / "day5_feedback.txt"
ERROR_LOG_PATH = OUTPUT_DIR / "ranking_explanation_errors.json"

FUSION_WEIGHTS = {"sparse": 0.45, "dense": 0.35, "exact": 0.20}
INTENT_SCORE_WEIGHTS = {
    "scene_match": 0.20,
    "topic_match": 0.30,
    "constraint_match": 0.25,
    "paper_type_match": 0.10,
    "time_preference_match": 0.10,
    "survey_preference_match": 0.05,
}
DEFAULT_CANDIDATE_POOL_SIZE = 100
DEFAULT_TOP_K = 10
DEFAULT_EXPLAIN_LIMIT = 20
OPENAI_RUNTIME_AVAILABLE: Optional[bool] = None
OPENAI_RUNTIME_MESSAGE = ""
DENSE_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}

DEFAULT_CHAIN_DEMOS = [
    {
        "query": "我想看 RAG",
        "follow_up_reply": "最近两年，综述优先，最好解释为什么推荐",
    },
    {
        "query": "帮我找 LLM agent 的论文",
        "follow_up_reply": "最近两年，综述优先，方法不限",
    },
    {
        "query": "找机器翻译论文",
        "follow_up_reply": "作者不限，最好近三年，用 COMET 或者质量估计相关",
    },
    {
        "query": "papers by authors of MALT",
        "follow_up_reply": "方法不限，解释一下为什么这些结果相关",
    },
    {
        "query": "Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey",
    },
    {
        "query": "Looking for survey papers on long context in LLMs",
    },
    {
        "query": "找用 COMET 做 machine translation quality estimation 的论文",
        "follow_up_reply": "最近的优先，数据集不限",
    },
    {
        "query": "给我多样一些的 agent evaluation papers，并解释为什么推荐",
    },
    {
        "query": "找 multimodal reasoning papers",
        "follow_up_reply": "数据集不限，最好结果多样一些",
    },
    {
        "query": "我想看用 early exit 做质量估计的论文",
    },
]

EXPLANATION_SYSTEM_PROMPT = """You are explaining why a paper is ranked for an academic retrieval system.

You must only use the provided evidence pack and ranking features.
Do not invent facts beyond the evidence.

Return JSON only with:
- ranking_reasons: 2 to 4 concise reasons
- unmet_constraints: concise unmet constraints
- explanation_adjustment: a number between -0.03 and 0.03
"""

EXPLANATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ranking_reasons": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "unmet_constraints": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
        "explanation_adjustment": {"type": "number"},
    },
    "required": ["ranking_reasons", "unmet_constraints", "explanation_adjustment"],
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    errors = load_error_log()
    errors.append(entry)
    dump_json(ERROR_LOG_PATH, errors)


def write_prompt_file() -> None:
    content = f"""# Ranking Explanation Prompt

Model Default: {OPENAI_MODEL}

## System Prompt
{EXPLANATION_SYSTEM_PROMPT}
"""
    dump_text(EXPLANATION_PROMPT_PATH, content)


def build_dense_index(db_path: Path) -> Dict[str, Any]:
    cache_key = str(db_path.resolve())
    cached = DENSE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with day2.connect_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT paper_id, embedding_text
            FROM papers
            ORDER BY paper_id
            """
        ).fetchall()

    documents: Dict[str, Counter[str]] = {}
    doc_freq: Counter[str] = Counter()
    for row in rows:
        paper_id = row["paper_id"]
        tokens = dense_tokens(clean_text(row["embedding_text"]))
        counter = Counter(tokens)
        documents[paper_id] = counter
        for token in counter.keys():
            doc_freq[token] += 1

    total_docs = max(len(documents), 1)
    idf = {token: math.log((total_docs + 1) / (freq + 1)) + 1.0 for token, freq in doc_freq.items()}
    vectors: Dict[str, Dict[str, float]] = {}
    norms: Dict[str, float] = {}
    for paper_id, counter in documents.items():
        weighted = {token: (1.0 + math.log(freq)) * idf.get(token, 1.0) for token, freq in counter.items()}
        vectors[paper_id] = weighted
        norms[paper_id] = math.sqrt(sum(value * value for value in weighted.values())) or 1.0

    index = {"vectors": vectors, "norms": norms, "idf": idf}
    DENSE_INDEX_CACHE[cache_key] = index
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
        for result in day2.search_basic(query, top_k=top_k_per_query, db_path=db_path):
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
    aggregated: Dict[str, Dict[str, Any]] = {}
    for query in intent_frame.get("exact_queries", []):
        for result in day2.search_exact_matches(query, top_k=top_k_per_query, db_path=db_path):
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
    vectors = index["vectors"]
    norms = index["norms"]
    idf = index["idf"]

    for query in queries:
        query_vector, query_norm = dense_query_vector(query, idf)
        scored: List[Tuple[str, float]] = []
        for paper_id, doc_vector in vectors.items():
            score = cosine_similarity(query_vector, query_norm, doc_vector, norms[paper_id])
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
    with day2.connect_db(db_path) as conn:
        titles = {row["paper_id"]: row["title"] for row in conn.execute("SELECT paper_id, title FROM papers").fetchall()}
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
    scored_sections: List[Tuple[int, str, str]] = []
    for row in section_rows:
        section_title = clean_text(row["section_title"])
        section_snippet = clean_text(row["section_snippet"])
        best_score = 0
        for query_text in query_texts:
            normalized_query = day2.normalize_match_text(query_text)
            tokens = day2.tokenize_query(query_text)
            score = day2.score_text(f"{section_title}\n{section_snippet}", normalized_query, tokens)
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
) -> Dict[str, Any]:
    normalized_authors = parse_json_field(row["normalized_authors"], [])
    semantic_card = parse_json_field(row["semantic_card_json"], {})
    query_texts = clean_string_list(
        intent_frame.get("coarse_queries", []) + intent_frame.get("dense_queries", []) + intent_frame.get("exact_queries", []),
        limit=12,
    )
    matched_sections, extra_snippets = build_matched_sections(section_rows, query_texts)
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
    terms = collect_intent_terms(intent_frame)
    intent_alignment_candidates = []
    for term in terms["topic"] + terms["constraints"]:
        normalized_term = clean_text(term).lower()
        if normalized_term and normalized_term in paper_text.lower():
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
) -> Dict[str, Any]:
    semantic_card = evidence_pack["semantic_card"]
    paper_type = infer_paper_type(row, semantic_card)
    paper_text = build_paper_text(row, semantic_card)
    terms = collect_intent_terms(intent_frame)

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


def heuristic_ranking_reasons(intent_frame: Dict[str, Any], evidence_pack: Dict[str, Any], rank_result: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if evidence_pack["intent_alignment_candidates"]:
        reasons.append("命中了关键意图词：" + "、".join(evidence_pack["intent_alignment_candidates"][:3]))
    if evidence_pack["matched_snippets"]:
        top_snippet = evidence_pack["matched_snippets"][0]
        reasons.append(f"召回证据来自 `{top_snippet['field']}`：{top_snippet['snippet'][:120]}")
    if rank_result["paper_type_match"] >= 0.9:
        reasons.append(f"论文类型与用户预期一致：{rank_result['paper_type']}")
    if rank_result["constraint_match"] >= 0.6:
        reasons.append("方法/模型/数据集等约束与当前论文较为一致。")
    elif evidence_pack["constraint_conflicts"]:
        reasons.append("存在未完全满足的约束：" + "；".join(evidence_pack["constraint_conflicts"][:2]))
    if rank_result["scene_match"] >= 0.8 and intent.get_slot(intent_frame, intent.SLOT_SPECS["search_scene"]["path"])["value"]:
        reasons.append("符合当前检索场景：" + intent.get_slot(intent_frame, intent.SLOT_SPECS["search_scene"]["path"])["value"])
    return clean_string_list(reasons, limit=4)


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


def build_explanation_messages(intent_frame: Dict[str, Any], evidence_pack: Dict[str, Any], rank_result: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = {
        "intent_frame": intent_frame,
        "evidence_pack": evidence_pack,
        "rank_features": {
            "base_score": rank_result["base_score"],
            "intent_score": rank_result["intent_score"],
            "final_score": rank_result["final_score"],
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


def generate_ranking_explanation(
    intent_frame: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    rank_result: Dict[str, Any],
) -> Tuple[List[str], List[str], float, Optional[str], str]:
    heuristic_reasons = ensure_ranking_reasons(
        heuristic_ranking_reasons(intent_frame, evidence_pack, rank_result),
        evidence_pack,
        rank_result,
    )
    heuristic_unmet_constraints = clean_string_list(evidence_pack.get("constraint_conflicts", []), limit=4)
    if not can_use_openai():
        return (
            heuristic_reasons,
            heuristic_unmet_constraints,
            0.0,
            None,
            "heuristic",
        )

    try:
        raw_payload, used_model = structured_chat_completion(
            messages=build_explanation_messages(intent_frame, evidence_pack, rank_result),
            schema_name="ranking_explanation",
            schema=EXPLANATION_SCHEMA,
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=500,
            timeout=120,
            api_key=OPENAI_API_KEY,
        )
    except Exception as exc:
        append_error_log(
            {
                "stage": "ranking_explanation",
                "paper_id": rank_result.get("paper_id"),
                "title": rank_result.get("title"),
                "error": str(exc),
            }
        )
        return heuristic_reasons, heuristic_unmet_constraints, 0.0, None, "heuristic"

    reasons = clean_string_list(raw_payload.get("ranking_reasons", []), limit=4)
    if len(reasons) < 2:
        reasons = heuristic_reasons
    reasons = ensure_ranking_reasons(reasons, evidence_pack, rank_result)

    unmet_constraints = clean_string_list(raw_payload.get("unmet_constraints", []), limit=4)
    if not unmet_constraints:
        unmet_constraints = heuristic_unmet_constraints

    adjustment = clamp_score(
        raw_payload.get("explanation_adjustment", 0.0),
        minimum=-0.03,
        maximum=0.03,
    )
    return reasons, unmet_constraints, adjustment, used_model, "llm"


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
                evidence_gap.append("top-K 中符合目标 paper type 的论文比例仍然偏低。")

        prefer_survey = intent.get_slot(intent_frame, intent.SLOT_SPECS["result_preferences.prefer_survey"]["path"])
        if prefer_survey["status"] == "confirmed" and prefer_survey["value"] == "yes" and survey_hits == 0:
            evidence_gap.append("用户偏好综述，但当前 top-K 中没有 survey。")

    if missing_slots:
        why_broad.append("用户意图仍有缺失槽位，导致召回和重排需要保持较宽覆盖。")
        improvements.append("优先补充以下缺失信息：" + "、".join(missing_slots[:6]))
    if ambiguous_dimensions:
        why_broad.append("部分槽位被标记为 ambiguous，系统默认不会继续追问这些维度。")
    if evidence_gap:
        why_broad.extend(evidence_gap[:2])

    if not improvements:
        if evidence_gap:
            improvements.append("优先补充方法、paper type 或标题线索，可显著收窄结果。")
        else:
            improvements.append("当前结果已较集中，可直接查看 top-K 解释。")

    return {
        "query_gap": missing_slots,
        "evidence_gap": clean_string_list(evidence_gap, limit=6),
        "matched_dimensions": clean_string_list(matched_dimensions, limit=6),
        "ambiguous_dimensions": clean_string_list(ambiguous_dimensions, limit=6),
        "why_current_results_are_broad": clean_string_list(why_broad, limit=4),
        "what_next_answer_would_improve": clean_string_list(improvements, limit=4),
    }


def rerank_candidates(
    intent_frame: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    paper_rows: Dict[str, Any],
    sections_by_paper: Dict[str, List[Any]],
    top_k: int = DEFAULT_TOP_K,
    explain_limit: int = DEFAULT_EXPLAIN_LIMIT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    ranked: List[Dict[str, Any]] = []
    evidence_packs: Dict[str, Dict[str, Any]] = {}

    for candidate in candidate_pool:
        row = paper_rows[candidate["paper_id"]]
        evidence_pack = build_paper_evidence_pack(candidate, row, sections_by_paper.get(candidate["paper_id"], []), intent_frame)
        evidence_packs[candidate["paper_id"]] = evidence_pack
        score_payload = score_candidate_against_intent(candidate, row, evidence_pack, intent_frame)
        final_score = clamp_score(0.6 * candidate["base_score"] + 0.4 * score_payload["intent_score"] - score_payload["conflict_penalty"], maximum=2.0)
        ranked.append(
            {
                "paper_id": candidate["paper_id"],
                "title": candidate["title"],
                "base_score": candidate["base_score"],
                "intent_score": score_payload["intent_score"],
                "final_score": round(final_score, 6),
                "sparse_score": candidate["sparse_score"],
                "dense_score": candidate["dense_score"],
                "exact_score": candidate["exact_score"],
                "matched_field": candidate.get("matched_field", ""),
                "matched_snippet": candidate.get("matched_snippet", ""),
                "exact_match_type": candidate.get("exact_match_type", ""),
                "retrieval_sources": candidate.get("retrieval_sources", []),
                **score_payload,
                "ranking_reasons": [],
                "unmet_constraints": [],
                "explanation_adjustment": 0.0,
                "explanation_parser": "heuristic",
                "used_model": None,
            }
        )

    ranked.sort(key=lambda item: (-item["final_score"], -item["intent_score"], -item["base_score"], item["title"]))

    for item in ranked[: max(top_k, explain_limit)]:
        evidence_pack = evidence_packs[item["paper_id"]]
        reasons, unmet_constraints, adjustment, used_model, parser = generate_ranking_explanation(
            intent_frame,
            evidence_pack,
            item,
        )
        item["ranking_reasons"] = reasons
        item["unmet_constraints"] = clean_string_list(unmet_constraints, limit=4)
        item["explanation_adjustment"] = round(adjustment, 6)
        item["explanation_parser"] = parser
        item["used_model"] = used_model
        item["final_score"] = round(item["final_score"] + adjustment, 6)

    ranked.sort(key=lambda item: (-item["final_score"], -item["intent_score"], -item["base_score"], item["title"]))

    for item in ranked[:top_k]:
        if item["ranking_reasons"]:
            continue
        evidence_pack = evidence_packs[item["paper_id"]]
        reasons, unmet_constraints, adjustment, used_model, parser = generate_ranking_explanation(
            intent_frame,
            evidence_pack,
            item,
        )
        item["ranking_reasons"] = reasons
        item["unmet_constraints"] = clean_string_list(unmet_constraints, limit=4)
        item["explanation_adjustment"] = round(adjustment, 6)
        item["explanation_parser"] = parser
        item["used_model"] = used_model
        item["final_score"] = round(item["final_score"] + adjustment, 6)

    ranked.sort(key=lambda item: (-item["final_score"], -item["intent_score"], -item["base_score"], item["title"]))

    for item in ranked[:top_k]:
        if item["ranking_reasons"]:
            continue
        evidence_pack = evidence_packs[item["paper_id"]]
        reasons, unmet_constraints, _, used_model, parser = generate_ranking_explanation(
            intent_frame,
            evidence_pack,
            item,
        )
        item["ranking_reasons"] = reasons
        item["unmet_constraints"] = clean_string_list(unmet_constraints, limit=4)
        item["explanation_parser"] = parser
        item["used_model"] = used_model

    return ranked[:top_k], evidence_packs


def run_core_chain(
    query: str,
    db_path: Path,
    follow_up_reply: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    explain_limit: int = DEFAULT_EXPLAIN_LIMIT,
) -> Dict[str, Any]:
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

    with day2.connect_db(db_path) as conn:
        paper_rows = load_paper_rows(conn, [item["paper_id"] for item in candidate_pool])
        sections_by_paper = day2.load_sections_for_papers(conn, [item["paper_id"] for item in candidate_pool])

    ranked_results, evidence_packs = rerank_candidates(
        intent_frame=final_frame,
        candidate_pool=candidate_pool,
        paper_rows=paper_rows,
        sections_by_paper=sections_by_paper,
        top_k=top_k,
        explain_limit=explain_limit,
    )
    gap_report = build_gap_report(final_frame, ranked_results)

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
        "top_k_results": ranked_results,
        "paper_evidence_packs": {paper_id: evidence_packs[paper_id] for paper_id in [item["paper_id"] for item in ranked_results]},
    }


def write_feedback(
    db_path: Path,
    demo_runs: Sequence[Dict[str, Any]],
    openai_available: bool,
    openai_message: str,
) -> None:
    clarification_runs = sum(1 for item in demo_runs if item["initial_intent_frame"].get("clarification_needed"))
    runs_with_follow_up = sum(1 for item in demo_runs if item.get("follow_up_reply"))
    content = f"""
Day 5 执行完成

核心方法链路
- query -> 意图解析 -> 聚合追问 -> 三路检索 -> gap 分析 -> 意图重排 -> 结果解释

运行信息
- 数据库文件: {db_path}
- OpenAI 可用: {openai_available}
- OpenAI 状态说明: {openai_message}
- 演示 query 数量: {len(demo_runs)}
- 带 follow-up 的演示数量: {runs_with_follow_up}
- 首轮需要追问的演示数量: {clarification_runs}

已完成能力
- sparse + dense + exact 三路召回已接通。
- 三路候选已融合为 top 100 候选池。
- 每篇候选论文都能构造 PaperEvidencePack。
- 系统可输出 IntentGapReport。
- 系统可做意图感知重排并输出 IntentAwareRankResult。
- top-K 结果带证据和解释，而不是只有排序列表。

交付物
- {EXPLANATION_PROMPT_PATH.name}
- {CHAIN_DEMOS_PATH.name}
- {GAP_REPORTS_PATH.name}
- {RANK_RESULTS_PATH.name}
"""
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

    demo_items = list(demos or DEFAULT_CHAIN_DEMOS)
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
    top_rank_results = [
        {
            "query": item["query"],
            "follow_up_reply": item.get("follow_up_reply"),
            "top_k_results": item["top_k_results"],
        }
        for item in demo_runs
    ]

    dump_json(CHAIN_DEMOS_PATH, demo_runs)
    dump_json(GAP_REPORTS_PATH, gap_reports)
    dump_json(RANK_RESULTS_PATH, top_rank_results)
    write_feedback(db_path, demo_runs, openai_available, openai_message)
    return {
        "db_path": str(db_path),
        "openai_available": openai_available,
        "openai_message": openai_message,
        "demo_query_count": len(demo_runs),
        "output_dir": str(OUTPUT_DIR),
    }
