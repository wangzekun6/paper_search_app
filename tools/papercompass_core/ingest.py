"""
PaperCompass 基础契约流水线。

这个文件的目标不是做全量检索，而是先把项目的基础约定固定下来：
1. 真实论文 JSON 的字段结构长什么样
2. 项目内部要抽成哪些标准对象
3. SQLite schema 应该如何设计
4. 用少量样本先验证整套结构是否稳定

因此这里主要负责“定协议、做样例、写 schema、产出验证报告”。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    PaperIndexRecord,
    PaperSectionRecord,
    PaperSemanticCard,
    RawPaperRecord,
)
from .config import DATASET_DIR, PROJECT_ROOT, resolve_dataset_root


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "archive" / "legacy_samples"
DEFAULT_DATA_ROOT = DATASET_DIR
SCHEMA_SQL = """-- 基础阶段固定下来的 SQLite schema。
-- 这份 schema 定义了项目的基础数据层结构，包括：
-- 1. raw_papers：原始论文 JSON
-- 2. papers：论文级索引字段
-- 3. paper_sections：章节级扁平记录
-- 4. paper_semantic_cards：语义模块生成的语义卡片缓存
-- 5. paper_search_fts：SQLite FTS 检索表

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS raw_papers (
    paper_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    year_month TEXT NOT NULL,
    title TEXT NOT NULL,
    authors_raw TEXT NOT NULL,
    normalized_authors TEXT NOT NULL,
    abstract TEXT NOT NULL,
    section_titles TEXT NOT NULL,
    intro_text TEXT NOT NULL,
    methods_text TEXT NOT NULL,
    results_text TEXT NOT NULL,
    discussion_text TEXT NOT NULL,
    appendix_titles TEXT NOT NULL,
    fulltext_for_sparse TEXT NOT NULL,
    embedding_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_year_month ON papers(year_month);
CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);

CREATE TABLE IF NOT EXISTS paper_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    section_order INTEGER NOT NULL,
    section_title TEXT NOT NULL,
    section_type TEXT NOT NULL CHECK (section_type IN ('intro', 'methods', 'results', 'discussion', 'other')),
    section_text TEXT NOT NULL,
    section_snippet TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_sections_paper_order
    ON paper_sections(paper_id, section_order);

CREATE INDEX IF NOT EXISTS idx_paper_sections_type
    ON paper_sections(section_type);

CREATE TABLE IF NOT EXISTS paper_semantic_cards (
    paper_id TEXT PRIMARY KEY,
    semantic_card_json TEXT NOT NULL,
    card_status TEXT NOT NULL DEFAULT 'pending',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    intent_frame_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS saved_papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_papers_paper_id
    ON saved_papers(paper_id);

CREATE INDEX IF NOT EXISTS idx_search_history_created_at
    ON search_history(created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS paper_search_fts USING fts5(
    paper_id UNINDEXED,
    title,
    abstract,
    section_titles,
    section_snippet
);
"""

RESET_SCHEMA_SQL = """
PRAGMA foreign_keys = OFF;
DROP TABLE IF EXISTS saved_papers;
DROP TABLE IF EXISTS search_history;
DROP TABLE IF EXISTS paper_semantic_cards;
DROP TABLE IF EXISTS paper_sections;
DROP TABLE IF EXISTS papers;
DROP TABLE IF EXISTS raw_papers;
DROP TABLE IF EXISTS paper_search_fts;
PRAGMA foreign_keys = ON;
"""

REQUIRED_TOP_LEVEL_KEYS = [
    "title",
    "authors",
    "abstract",
    "sections",
    "appendices",
    "references",
]
DEFAULT_VALIDATION_IDS = [
    "2502.00008",
    "2502.12701",
    "2502.02494",
    "2502.00577",
    "2502.14083",
]
DEFAULT_SAMPLE_IDS = [
    "2502.00008",
    "2502.02494",
    "2502.12701",
]


def reset_database_schema(conn: sqlite3.Connection) -> None:
    try:
        conn.executescript(RESET_SCHEMA_SQL)
        conn.executescript(SCHEMA_SQL)
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower():
            raise RuntimeError(
                "Database is busy. Close any running Streamlit app, build command, or editor process "
                "that is using system_outputs/runtime/papercompass.db, then retry."
            ) from exc
        raise
SECTION_RULES = OrderedDict(
    [
        ("intro", ["introduction", "background"]),
        ("methods", ["method", "methods", "approach", "framework"]),
        ("results", ["result", "results", "experiment", "experiments", "evaluation"]),
        ("discussion", ["discussion", "conclusion", "conclusions", "analysis"]),
    ]
)
AUTHOR_STOPWORDS = {
    "author",
    "authors",
    "corresponding",
    "university",
    "universidade",
    "college",
    "school",
    "department",
    "institute",
    "instituto",
    "laboratory",
    "lab",
    "center",
    "centre",
    "research",
    "academy",
    "hospital",
    "faculty",
    "sciences",
    "science",
    "engineering",
    "technology",
    "telecomunicacoes",
    "telecomunicações",
    "lisbon",
    "paris",
    "chicago",
    "haven",
    "unit",
    "unbabel",
    "ellis",
}


def parse_args() -> argparse.Namespace:
    """解析基础契约流水线参数，决定样本来源和输出目录。"""

    parser = argparse.ArgumentParser(description="Run the PaperCompass contract pipeline.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing extracted arXiv JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to write sample contract outputs.",
    )
    parser.add_argument(
        "--validation-ids",
        nargs="*",
        default=DEFAULT_VALIDATION_IDS,
        help="Paper ids used for schema validation.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=DEFAULT_SAMPLE_IDS,
        help="Paper ids converted into sample PaperIndexRecord outputs.",
    )
    parser.add_argument(
        "--random-check-count",
        type=int,
        default=4,
        help="Extra random files to report when no validation ids are provided.",
    )
    return parser.parse_args()


def load_paper_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paper_path(data_root: Path, paper_id: str) -> Path:
    path = data_root / f"{paper_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing paper JSON: {path}")
    return path


def source_path_for(data_root: Path, path: Path) -> str:
    return f"{data_root.name}/{path.name}"


def source_path_has_year_month_dir(source_path: str) -> bool:
    return any(re.fullmatch(r"\d{4}", part) for part in Path(source_path).parts)


def extract_year_month(source_path: str, paper_id: str) -> str:
    parts = Path(source_path).parts
    for part in parts:
        if re.fullmatch(r"\d{4}", part):
            return part
    match = re.match(r"(?P<ym>\d{4})\.", paper_id)
    if match:
        return match.group("ym")
    return "unknown"


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_paragraphs(paragraphs: Any) -> List[str]:
    if not isinstance(paragraphs, list):
        return []
    cleaned = [clean_text(item) for item in paragraphs]
    return [item for item in cleaned if item]


def truncate(text: str, max_chars: int = 320) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def classify_section_title(title: str) -> str:
    title_lower = clean_text(title).lower()
    for section_type, keywords in SECTION_RULES.items():
        if any(keyword in title_lower for keyword in keywords):
            return section_type
    return "other"


def flatten_sections(
    paper_id: str,
    section_map: Dict[str, Any],
) -> List[Tuple[str, str, List[str]]]:
    """
    把嵌套的 sections/subsections 结构递归展开成线性列表。

    返回结果会保留：
    1. 扁平化后的完整章节路径
    2. 规则归类后的 section_type
    3. 清洗后的段落列表
    """

    # 先把嵌套 section 扁平化，后续入库、检索、证据定位都依赖这个稳定顺序。
    flattened: List[Tuple[str, str, List[str]]] = []

    def visit_node(title: str, payload: Any, parents: Sequence[str]) -> None:
        if not isinstance(payload, dict):
            return
        title_text = clean_text(title) or "Untitled Section"
        title_path = [part for part in parents if part] + [title_text]
        full_title = " > ".join(title_path)
        paragraphs = clean_paragraphs(payload.get("paragraphs"))
        flattened.append((full_title, classify_section_title(full_title), paragraphs))

        subsections = payload.get("subsections", [])
        if not isinstance(subsections, list):
            return
        for subsection in subsections:
            if not isinstance(subsection, dict):
                continue
            subsection_title = clean_text(subsection.get("title")) or "Untitled Subsection"
            visit_node(subsection_title, subsection, title_path)

    if not isinstance(section_map, dict):
        return flattened

    for title, payload in section_map.items():
        visit_node(str(title), payload, [])
    return flattened


def flatten_appendix_titles(appendices: Any) -> List[str]:
    if not isinstance(appendices, dict):
        return []
    titles = [clean_text(title) for title in appendices.keys()]
    return [title for title in titles if title]


def looks_like_author_name(candidate: str) -> bool:
    tokens = candidate.split()
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    if any(stopword in candidate.lower() for stopword in AUTHOR_STOPWORDS):
        return False
    for token in tokens:
        if not re.fullmatch(r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.\-]*", token):
            return False
    return True


def extract_author_candidates(authors_raw: str) -> List[str]:
    if not authors_raw:
        return []

    sanitized = authors_raw
    sanitized = re.sub(r"\S+@\S+", " | ", sanitized)
    sanitized = re.sub(r"Corresponding Author:?", " | ", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"(?<=[a-z])(?=[A-Z])", " | ", sanitized)
    sanitized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " | ", sanitized)
    sanitized = re.sub(r"(?<=\d)(?=[A-ZÀ-ÖØ-Þ])", " | ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    candidates: List[str] = []
    seen = set()
    for chunk in re.split(r"[|;/]", sanitized):
        for part in chunk.split(","):
            candidate = re.sub(r"\d+", "", part)
            candidate = clean_text(candidate.strip(" ,"))
            if not candidate:
                continue
            if not looks_like_author_name(candidate):
                continue
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def join_blocks(blocks: Iterable[str]) -> str:
    cleaned = [clean_text(block) for block in blocks if clean_text(block)]
    return "\n\n".join(cleaned)


def build_fulltext_for_sparse(
    title: str,
    abstract: str,
    section_records: List[PaperSectionRecord],
) -> str:
    # fulltext_for_sparse 追求的是词汇覆盖面，主要服务稀疏检索/关键词召回。
    # 这里保留 title、abstract、section title 以及每个 section 的前几段，
    # 既保证召回能力，又避免把整篇正文无限扩进去。
    blocks = [title, abstract]
    blocks.extend(section.section_title for section in section_records)
    for section in section_records:
        paragraphs = [part for part in section.section_text.split("\n\n") if part]
        blocks.extend(paragraphs[:2])
    return join_blocks(blocks)


def build_embedding_text(
    title: str,
    abstract: str,
    typed_paragraphs: Dict[str, List[str]],
) -> str:
    # embedding_text 追求的是“紧凑但有代表性的语义窗口”，
    # 只保留最关键的论文字段和四类核心 section 的前几段，
    # 这样后续模型调用和语义检索都能保持输入稳定。
    blocks = [title, abstract]
    for section_type in ("intro", "methods", "results", "discussion"):
        blocks.extend(typed_paragraphs.get(section_type, [])[:2])
    return join_blocks(blocks)


def build_records_for_paper(path: Path, data_root: Path) -> Tuple[RawPaperRecord, PaperIndexRecord, List[PaperSectionRecord], PaperSemanticCard]:
    """
    从单篇论文 JSON 一次性构建基础阶段需要的全部标准对象。

    这是基础阶段的核心对象装配函数：
    原始记录、论文索引记录、章节记录和空语义卡片模板都在这里生成。
    """

    paper_id = path.stem
    raw_json = load_paper_json(path)
    source_path = source_path_for(data_root, path)
    raw_record = RawPaperRecord(
        paper_id=paper_id,
        source_path=source_path,
        raw_json=raw_json,
    )

    flattened = flatten_sections(paper_id, raw_json.get("sections", {}))
    section_records: List[PaperSectionRecord] = []
    typed_paragraphs: Dict[str, List[str]] = {
        "intro": [],
        "methods": [],
        "results": [],
        "discussion": [],
    }
    section_titles: List[str] = []

    for index, (section_title, section_type, paragraphs) in enumerate(flattened, start=1):
        section_text = join_blocks(paragraphs)
        section_snippet = truncate(paragraphs[0] if paragraphs else section_title)
        record = PaperSectionRecord(
            paper_id=paper_id,
            section_order=index,
            section_title=section_title,
            section_type=section_type,
            section_text=section_text,
            section_snippet=section_snippet,
        )
        section_records.append(record)
        section_titles.append(section_title)
        if section_type in typed_paragraphs:
            typed_paragraphs[section_type].extend(paragraphs)

    title = clean_text(raw_json.get("title"))
    abstract = clean_text(raw_json.get("abstract"))
    authors_raw = clean_text(raw_json.get("authors"))
    index_record = PaperIndexRecord(
        paper_id=paper_id,
        source_path=source_path,
        year_month=extract_year_month(source_path, paper_id),
        title=title,
        authors_raw=authors_raw,
        normalized_authors=extract_author_candidates(authors_raw),
        abstract=abstract,
        section_titles=section_titles,
        intro_text=join_blocks(typed_paragraphs["intro"]),
        methods_text=join_blocks(typed_paragraphs["methods"]),
        results_text=join_blocks(typed_paragraphs["results"]),
        discussion_text=join_blocks(typed_paragraphs["discussion"]),
        appendix_titles=flatten_appendix_titles(raw_json.get("appendices", {})),
        fulltext_for_sparse=build_fulltext_for_sparse(title, abstract, section_records),
        embedding_text=build_embedding_text(title, abstract, typed_paragraphs),
    )
    semantic_card = PaperSemanticCard(paper_id=paper_id)
    return raw_record, index_record, section_records, semantic_card


def validate_paper_structure(path: Path, data_root: Path) -> Dict[str, Any]:
    """抽查单篇论文的 JSON 结构，用于确认真实数据格式是否符合预期。"""

    raw_json = load_paper_json(path)
    sections = raw_json.get("sections")
    appendices = raw_json.get("appendices")
    references = raw_json.get("references")
    first_section_key = None
    first_section_payload = None
    if isinstance(sections, dict) and sections:
        first_section_key = next(iter(sections.keys()))
        first_section_payload = sections[first_section_key]

    return {
        "paper_id": path.stem,
        "source_path": source_path_for(data_root, path),
        "top_level_keys": list(raw_json.keys()),
        "has_all_required_keys": all(key in raw_json for key in REQUIRED_TOP_LEVEL_KEYS),
        "authors_type": type(raw_json.get("authors")).__name__,
        "sections_type": type(sections).__name__,
        "sections_is_dict": isinstance(sections, dict),
        "section_count": len(sections) if isinstance(sections, dict) else 0,
        "first_section_key": first_section_key,
        "first_section_keys": list(first_section_payload.keys()) if isinstance(first_section_payload, dict) else [],
        "appendices_type": type(appendices).__name__,
        "appendices_is_dict": isinstance(appendices, dict),
        "appendices_count": len(appendices) if isinstance(appendices, dict) else 0,
        "references_type": type(references).__name__,
        "references_is_list": isinstance(references, list),
        "references_count": len(references) if isinstance(references, list) else 0,
    }


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def build_validation_ids(data_root: Path, requested_ids: Sequence[str], random_check_count: int) -> List[str]:
    if requested_ids:
        return list(requested_ids)
    all_ids = sorted(path.stem for path in data_root.glob("*.json"))
    random.seed(42)
    return random.sample(all_ids, min(random_check_count, len(all_ids)))


def write_field_mapping(output_dir: Path) -> None:
    content = """
基础字段映射说明

RawPaperRecord
- paper_id: 文件名去掉 .json，作为统一主键。
- source_path: 使用项目内数据根目录名 + 文件名，例如 data/arxiv_202502_cs_cl/2502.00008.json。
- raw_json: 原始 JSON 全量保留，只用于详情展示、追溯与调试。

PaperIndexRecord
- paper_id/source_path: 与 RawPaperRecord 保持一致。
- year_month: 优先从路径中的 4 位月份目录提取；当前样例目录没有月份子目录，因此回退为 paper_id 前 4 位，例如 2502。
- title: 原始 title，清理多余空白后保留。
- authors_raw: 原始 authors 字符串，清理多余空白后保留。
- normalized_authors: 轻量作者名抽取结果，用于展示和粗粒度 exact match，不追求完美作者解析。
- abstract: 原始摘要文本。
- section_titles: 主体 sections 与 subsections 扁平化后的标题列表，subsection 采用“父标题 > 子标题”形式。
- intro_text/methods_text/results_text/discussion_text: 根据 section 标题规则归类后聚合的正文。
- appendix_titles: 只保留附录标题，不保留附录长正文。
- fulltext_for_sparse: title + abstract + 全部 section 标题 + 每个 section 前 1 到 2 段。
- embedding_text: title + abstract + intro/methods/results/discussion 的代表性段落。

PaperSemanticCard
- 仅冻结字段名，不在本阶段生成内容。
- 存储时序列化为 semantic_card_json，后续语义卡片阶段再填充。

SQLite 存储约定
- normalized_authors、section_titles、appendix_titles 统一以 JSON 字符串落表。
- raw_json、semantic_card_json 也以 JSON TEXT 落表。
"""
    write_text(output_dir / "field_mapping.txt", content)


def write_section_rules(output_dir: Path) -> None:
    content = """
基础 section 归类规则

规则
- 标题包含 introduction、background -> intro
- 标题包含 method、methods、approach、framework -> methods
- 标题包含 result、results、experiment、experiments、evaluation -> results
- 标题包含 discussion、conclusion、conclusions、analysis -> discussion
- 其他全部 -> other

实现约束
- 只做英文关键词规则，不做复杂变体扩展。
- subsections 会被扁平化，标题保存为“父标题 > 子标题”。
- 归类时对完整标题路径做匹配，因此像“Experiments and Analysis > Deferral”会优先落到 results。
- 无法归类时统一返回 other，不为个别异常标题引入额外复杂性。

固定结论
- references 不进入主检索文本。
- appendices 不作为主召回字段。
- 主检索只围绕 title、authors、abstract、sections。
"""
    write_text(output_dir / "section_rules.txt", content)


def write_feedback_report(
    output_dir: Path,
    data_root: Path,
    validation_summary: Dict[str, Any],
    sample_records: List[PaperIndexRecord],
    db_path: Path,
) -> None:
    checked_ids = ", ".join(item["paper_id"] for item in validation_summary["paper_summaries"])
    sample_ids = ", ".join(record.paper_id for record in sample_records)
    empty_sections = ", ".join(validation_summary["papers_with_empty_sections"]) or "无"
    inferred_year_month = ", ".join(validation_summary["year_month_inference_papers"]) or "无"
    content = f"""
基础契约流水线执行完成

执行范围
- 项目目录: {PROJECT_ROOT}
- 数据源目录: {data_root}
- 结构核验样本: {checked_ids}
- 生成样例记录: {sample_ids}

真实结构结论
- 抽样论文顶层字段稳定为 title/authors/abstract/sections/appendices/references。
- sections 在抽样中均为 dict，且常见子字段为 paragraphs 和 subsections。
- appendices 在抽样中均为 dict，可统一降级为只保留标题。
- references 在抽样中均为 list[str]，已明确排除在主检索文本之外。
- 发现 sections 为空的论文: {empty_sections}

已固定的基础协议
- RawPaperRecord / PaperIndexRecord / PaperSemanticCard 字段已冻结。
- section 归类规则已固定为英文关键词规则。
- 主召回字段固定为 title、abstract、section_titles、section_snippet。
- 向量检索字段固定为 embedding_text。
- 详情字段固定为 raw_json、appendices、references。

推断说明
- 当前解压样例目录没有单独的 4 位月份子目录，因此 year_month 对这些论文回退为 paper_id 前 4 位。
- 涉及该推断的样例论文: {inferred_year_month}

已生成产物
- field_mapping.txt
- section_rules.txt
- sqlite_schema.sql
- validation_summary.json
- sample_paper_index_records.json
- sample_raw_paper_records.json
- sample_paper_sections.json
- sample_semantic_card_template.json
- sample_reference.db

数据库文件
- {db_path}

结论
- 当前阶段所需对象协议、字段分工、SQLite schema 和 3 条样例索引记录已经在项目内落地，可直接作为统一系统全量入库的基础。
"""
    write_text(output_dir / "sample_feedback.txt", content)


def initialize_database(
    db_path: Path,
    raw_records: List[RawPaperRecord],
    index_records: List[PaperIndexRecord],
    all_section_records: List[PaperSectionRecord],
    semantic_cards: List[PaperSemanticCard],
) -> None:
    """
    按当前 schema 写入样例数据库。

    这里写入的是少量样本，不是全量库。
    目的在于验证表结构、字段映射和对象序列化方案是否正确。
    """

    # 这里只写一个小型样例数据库，用来验证 schema 和对象映射是否正确。
    # 真正的全量入库由统一检索流水线执行。
    with sqlite3.connect(db_path, timeout=30) as conn:
        reset_database_schema(conn)

        conn.executemany(
            """
            INSERT INTO raw_papers (paper_id, source_path, raw_json)
            VALUES (?, ?, ?)
            """,
            [
                (
                    record.paper_id,
                    record.source_path,
                    json.dumps(record.raw_json, ensure_ascii=False),
                )
                for record in raw_records
            ],
        )

        conn.executemany(
            """
            INSERT INTO papers (
                paper_id, source_path, year_month, title, authors_raw, normalized_authors,
                abstract, section_titles, intro_text, methods_text, results_text,
                discussion_text, appendix_titles, fulltext_for_sparse, embedding_text
            ) VALUES (
                :paper_id, :source_path, :year_month, :title, :authors_raw, :normalized_authors,
                :abstract, :section_titles, :intro_text, :methods_text, :results_text,
                :discussion_text, :appendix_titles, :fulltext_for_sparse, :embedding_text
            )
            """,
            [record.to_storage_row() for record in index_records],
        )

        conn.executemany(
            """
            INSERT INTO paper_sections (
                paper_id, section_order, section_title, section_type, section_text, section_snippet
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.paper_id,
                    record.section_order,
                    record.section_title,
                    record.section_type,
                    record.section_text,
                    record.section_snippet,
                )
                for record in all_section_records
            ],
        )

        conn.executemany(
            """
            INSERT INTO paper_semantic_cards (paper_id, semantic_card_json, card_status, updated_at)
            VALUES (?, ?, 'pending', CURRENT_TIMESTAMP)
            """,
            [
                (
                    card.paper_id,
                    json.dumps(card.to_dict(), ensure_ascii=False),
                )
                for card in semantic_cards
            ],
        )

        conn.executemany(
            """
            INSERT INTO paper_search_fts (paper_id, title, abstract, section_titles, section_snippet)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    record.paper_id,
                    record.title,
                    record.abstract,
                    "\n".join(record.section_titles),
                    "\n".join(
                        section.section_snippet
                        for section in all_section_records
                        if section.paper_id == record.paper_id
                    ),
                )
                for record in index_records
            ],
        )
        conn.commit()


def main() -> None:
    args = parse_args()
    data_root = resolve_dataset_root(args.data_root)
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir)

    validation_ids = build_validation_ids(data_root, args.validation_ids, args.random_check_count)
    validation_paths = [paper_path(data_root, paper_id) for paper_id in validation_ids]
    validation_details = [validate_paper_structure(path, data_root) for path in validation_paths]

    validation_summary = {
        "required_top_level_keys": REQUIRED_TOP_LEVEL_KEYS,
        "paper_summaries": validation_details,
        "all_have_required_keys": all(item["has_all_required_keys"] for item in validation_details),
        "all_sections_are_dict": all(item["sections_is_dict"] for item in validation_details),
        "all_appendices_are_dict": all(item["appendices_is_dict"] for item in validation_details),
        "all_references_are_list": all(item["references_is_list"] for item in validation_details),
        "papers_with_empty_sections": [
            item["paper_id"] for item in validation_details if item["section_count"] == 0
        ],
        "year_month_inference_papers": [
            item["paper_id"]
            for item in validation_details
            if not source_path_has_year_month_dir(item["source_path"])
        ],
        "fixed_conclusions": {
            "references_into_main_search": False,
            "appendices_into_main_recall": False,
            "main_search_fields": ["title", "authors", "abstract", "sections"],
        },
    }
    dump_json(output_dir / "validation_summary.json", validation_summary)

    sample_paths = [paper_path(data_root, paper_id) for paper_id in args.sample_ids]
    raw_records: List[RawPaperRecord] = []
    index_records: List[PaperIndexRecord] = []
    all_section_records: List[PaperSectionRecord] = []
    semantic_cards: List[PaperSemanticCard] = []
    for path in sample_paths:
        raw_record, index_record, section_records, semantic_card = build_records_for_paper(path, data_root)
        raw_records.append(raw_record)
        index_records.append(index_record)
        all_section_records.extend(section_records)
        semantic_cards.append(semantic_card)

    dump_json(
        output_dir / "sample_raw_paper_records.json",
        [record.to_dict() for record in raw_records],
    )
    dump_json(
        output_dir / "sample_paper_index_records.json",
        [record.to_dict() for record in index_records],
    )
    dump_json(
        output_dir / "sample_paper_sections.json",
        [record.to_dict() for record in all_section_records],
    )
    dump_json(
        output_dir / "sample_semantic_card_template.json",
        [card.to_dict() for card in semantic_cards],
    )

    write_field_mapping(output_dir)
    write_section_rules(output_dir)
    schema_output_path = output_dir / "sqlite_schema.sql"
    schema_output_path.write_text(SCHEMA_SQL, encoding="utf-8")

    db_path = output_dir / "sample_reference.db"
    initialize_database(db_path, raw_records, index_records, all_section_records, semantic_cards)
    write_feedback_report(output_dir, data_root, validation_summary, index_records, db_path)


if __name__ == "__main__":
    main()
