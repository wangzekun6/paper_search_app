"""
PaperCompass 全量入库与检索流水线。

这个文件负责把完整论文库写入 SQLite，并在其上提供三种检索能力：
1. FTS 关键词检索
2. 标题 / 作者 / 短语的 exact match
3. 把两者合并后的 hybrid 检索

它是当前项目的主检索后端，也是语义卡片生成阶段的数据库基础。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pickle
import re
import sqlite3
import tarfile
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .models import PaperIndexRecord, PaperSectionRecord, RawPaperRecord
from .ingest import (
    build_embedding_text,
    build_fulltext_for_sparse,
    classify_section_title,
    clean_paragraphs,
    clean_text,
    extract_author_candidates,
    extract_year_month,
    flatten_appendix_titles,
    join_blocks,
    truncate,
    reset_database_schema,
)
from .config import (
    BUILD_FEEDBACK_PATH,
    DATASET_DIR,
    PROJECT_ROOT,
    SMOKE_QUERY_JSON_PATH,
    SMOKE_QUERY_TEXT_PATH,
    SYSTEM_DB_PATH,
    SYSTEM_OUTPUT_DIR,
    ensure_system_layout,
    get_default_dataset_source,
    get_active_runtime_db_path,
    is_supported_dataset_archive,
    resolve_dataset_root,
)


DEFAULT_OUTPUT_DIR = SYSTEM_OUTPUT_DIR
DEFAULT_DB_PATH = get_active_runtime_db_path()
DEFAULT_QUERY_JSON_PATH = SMOKE_QUERY_JSON_PATH
DEFAULT_QUERY_TEXT_PATH = SMOKE_QUERY_TEXT_PATH
DEFAULT_FEEDBACK_PATH = BUILD_FEEDBACK_PATH
DEFAULT_BUILD_BATCH_SIZE = 1000
DEFAULT_BUILD_MMAP_SIZE = 1073741824
SQLITE_VARIABLE_CHUNK_SIZE = 900
SECTION_SNIPPET_LIMIT = 320
FTS_FIELDS = ("title", "abstract", "section_titles", "section_snippet")
HOT_FTS_FIELDS = ("title", "authors", "abstract", "section_titles")
RUNTIME_METADATA_TABLE = "runtime_metadata"
PAPER_CONTENT_SIGNATURE_KEY = "paper_content_signature_v1"
HOT_FTS_TABLE = "paper_search_fts_hot_v2"
HOT_FTS_SIGNATURE_KEY = "paper_search_fts_hot_signature_v2"
HOT_FTS_VOCAB_TABLE = "paper_search_fts_hot_v2_vocab"
BUILD_SOURCE_FINGERPRINT_KEY = "build_source_fingerprint_v1"
BUILD_DESCRIPTOR_KEY = "build_descriptor_v1"
BUILD_DESCRIPTOR_VERSION = "build_layout_v2"
EXACT_INDEX_CACHE_VERSION = "exact_index_v1"
EXACT_INDEX_DISK_CACHE_SUBDIR = "exact_indexes"
FIELD_LABELS = {
    "title": "标题",
    "authors": "作者",
    "abstract": "摘要",
    "section_titles": "章节标题",
    "section_snippet": "章节片段",
}
MATCH_TYPE_LABELS = {
    "title_hint": "标题线索命中",
    "author_match": "作者命中",
    "phrase_match": "短语命中",
    "fts": "全文检索召回",
}
QUERY_STOPWORDS = {
    "a",
    "an",
    "about",
    "and",
    "by",
    "for",
    "from",
    "in",
    "latest",
    "new",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "recent",
    "the",
    "to",
    "with",
}
DEFAULT_DEBUG_QUERIES = [
    "retrieval augmented generation",
    "recent agent memory papers",
    "quality-aware deferral",
    "machine translation",
    "speech-to-speech translation",
    "hallucination mitigation",
    "long context",
    "data selection for language model pretraining",
    "MALT",
    "Riddle Me This",
    "Arianna Salazar-Miranda",
    "Dylan Sam",
    "form-based codes",
    "benchmark for large language models",
    "multimodal feedback",
    "quality estimation",
    "context length scaling",
    "scientific data visualization",
    "low-resource language urdu",
    "tool use with large language models",
]
EXACT_SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}


# 统一解析建库、检索和调试命令所需参数。
def parse_args() -> argparse.Namespace:
    """解析统一检索流水线参数，支持建库、单次检索和批量调试查询。"""

    parser = argparse.ArgumentParser(description="构建并查询 PaperCompass 的 SQLite 检索数据库。")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="构建完整 SQLite 数据库并生成调试查询日志。")
    build_parser.add_argument("--data-root", type=Path, default=get_default_dataset_source())
    build_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    build_parser.add_argument("--query-json-path", type=Path, default=DEFAULT_QUERY_JSON_PATH)
    build_parser.add_argument("--query-text-path", type=Path, default=DEFAULT_QUERY_TEXT_PATH)
    build_parser.add_argument("--feedback-path", type=Path, default=DEFAULT_FEEDBACK_PATH)
    build_parser.add_argument("--top-k", type=int, default=10)
    build_parser.add_argument("--skip-debug-queries", action="store_true")

    search_parser = subparsers.add_parser("search", help="在 PaperCompass 数据库上执行一次检索。")
    search_parser.add_argument("query", help="检索问题")
    search_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument(
        "--mode",
        choices=["basic", "exact", "hybrid"],
        default="hybrid",
        help="指定检索模式。",
    )

    debug_parser = subparsers.add_parser("debug-queries", help="运行默认的 20 条调试查询。")
    debug_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    debug_parser.add_argument("--query-json-path", type=Path, default=DEFAULT_QUERY_JSON_PATH)
    debug_parser.add_argument("--query-text-path", type=Path, default=DEFAULT_QUERY_TEXT_PATH)
    debug_parser.add_argument("--top-k", type=int, default=10)

    return parser.parse_args()


# 写文件前统一确保父目录存在。
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def localize_field_label(value: Any) -> str:
    text = clean_text(value)
    return FIELD_LABELS.get(text, text)


def localize_match_type_label(value: Any) -> str:
    text = clean_text(value)
    return MATCH_TYPE_LABELS.get(text, text)


# 统一规范化源文件路径，便于数据库内稳定存储。
def normalize_source_path(source_path: str | Path) -> str:
    path = Path(source_path)
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def source_path_for(data_root: Path, path: Path) -> str:
    root_name = data_root.name
    try:
        relative = path.resolve(strict=False).relative_to(data_root.resolve(strict=False))
    except ValueError:
        relative = Path(path.name)
    return f"{root_name}/{relative.as_posix()}"


def archive_root_name(archive_path: Path) -> str:
    name = archive_path.name
    lowered = name.lower()
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return archive_path.stem


def archive_source_path_for(archive_path: Path, member_name: str) -> str:
    root_name = archive_root_name(archive_path)
    parts = [part for part in PurePosixPath(member_name).parts if part not in ("", ".", "/")]
    if root_name in parts:
        relative_parts = parts[parts.index(root_name) :]
    elif len(parts) >= 2 and re.fullmatch(r"\d{4}", parts[-2]):
        relative_parts = [root_name, parts[-2], parts[-1]]
    else:
        relative_parts = [root_name, parts[-1] if parts else member_name]
    return "/".join(relative_parts)


def iter_directory_json_records(data_root: Path) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    for path in sorted(candidate for candidate in data_root.rglob("*.json") if candidate.is_file()):
        raw_json = load_raw_json(path)
        source_path = source_path_for(data_root, path)
        yield path.stem, source_path, raw_json


def iter_archive_json_records(archive_path: Path) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".json"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            source_path = archive_source_path_for(archive_path, member.name)
            paper_id = paper_id_from_source_path(source_path)
            try:
                with io.TextIOWrapper(extracted, encoding="utf-8") as handle:
                    raw_json = json.load(handle)
            except Exception as exc:
                raise ValueError(f"Failed to parse JSON from archive member: {member.name}") from exc
            yield paper_id, source_path, raw_json


def iter_dataset_json_records(data_source: Path) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    if data_source.is_dir():
        yield from iter_directory_json_records(data_source)
        return
    if is_supported_dataset_archive(data_source):
        yield from iter_archive_json_records(data_source)
        return
    raise ValueError(
        f"Unsupported dataset source: {data_source}. Only directories and .tar/.tar.gz/.tgz archives are supported."
    )


def paper_id_from_source_path(source_path: str | Path) -> str:
    return Path(source_path).stem


# 把层级章节展开成线性条目，方便构造章节级索引记录。
def iter_section_entries(
    section_map: Dict[str, Any],
    parents: Optional[Sequence[str]] = None,
) -> List[tuple[str, str, List[str]]]:
    flattened: List[tuple[str, str, List[str]]] = []
    parent_titles = list(parents or [])

    if not isinstance(section_map, dict):
        return flattened

    for title, payload in section_map.items():
        if not isinstance(payload, dict):
            continue

        title_text = clean_text(str(title)) or "Untitled Section"
        title_path = [part for part in parent_titles if part] + [title_text]
        full_title = " > ".join(title_path)
        paragraphs = clean_paragraphs(payload.get("paragraphs"))
        section_type = classify_section_title(full_title)
        flattened.append((full_title, section_type, paragraphs))

        subsections = payload.get("subsections", [])
        if not isinstance(subsections, list):
            continue

        for subsection in subsections:
            if not isinstance(subsection, dict):
                continue
            subsection_title = clean_text(subsection.get("title")) or "Untitled Subsection"
            flattened.extend(iter_section_entries({subsection_title: subsection}, title_path))

    return flattened


# 从原始 JSON 中提取扁平化章节记录。
def extract_sections(raw_json: Dict[str, Any], paper_id: str = "") -> List[PaperSectionRecord]:
    section_records: List[PaperSectionRecord] = []
    flattened = iter_section_entries(raw_json.get("sections", {}))

    for section_order, (section_title, section_type, paragraphs) in enumerate(flattened, start=1):
        section_text = join_blocks(paragraphs)
        section_snippet = truncate(section_text or section_title, max_chars=SECTION_SNIPPET_LIMIT)
        section_records.append(
            PaperSectionRecord(
                paper_id=paper_id,
                section_order=section_order,
                section_title=section_title,
                section_type=section_type,
                section_text=section_text,
                section_snippet=section_snippet,
            )
        )

    return section_records


# 把单篇原始论文解析成可直接入库的论文级索引记录。
def parse_paper_json(
    raw_json: Dict[str, Any],
    source_path: str | Path,
    *,
    section_records: Optional[Sequence[PaperSectionRecord]] = None,
) -> PaperIndexRecord:
    source_path_text = normalize_source_path(source_path)
    paper_id = paper_id_from_source_path(source_path_text)
    resolved_section_records = list(section_records) if section_records is not None else extract_sections(raw_json, paper_id)
    typed_paragraphs: Dict[str, List[str]] = defaultdict(list)
    section_titles: List[str] = []

    # 在全量入库时就提前拆出 intro/methods/results/discussion，
    # 这样语义卡片模块可以直接复用这些字段构造固定长度的 LLM 输入窗口。
    for section_record in resolved_section_records:
        section_titles.append(section_record.section_title)
        paragraphs = [part for part in section_record.section_text.split("\n\n") if part]
        if section_record.section_type in {"intro", "methods", "results", "discussion"}:
            typed_paragraphs[section_record.section_type].extend(paragraphs)

    title = clean_text(raw_json.get("title"))
    abstract = clean_text(raw_json.get("abstract"))
    authors_raw = clean_text(raw_json.get("authors"))

    return PaperIndexRecord(
        paper_id=paper_id,
        source_path=source_path_text,
        year_month=extract_year_month(source_path_text, paper_id),
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
        fulltext_for_sparse=build_fulltext_for_sparse(title, abstract, resolved_section_records),
        embedding_text=build_embedding_text(title, abstract, typed_paragraphs),
    )


# 统一配置 SQLite 连接，使用 Row 访问风格方便后续处理。
def connect_db(db_path: Path) -> sqlite3.Connection:
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def apply_build_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA locking_mode = EXCLUSIVE;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -262144;")
    conn.execute("PRAGMA cache_spill = OFF;")
    conn.execute(f"PRAGMA mmap_size = {DEFAULT_BUILD_MMAP_SIZE};")


def ensure_runtime_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUNTIME_METADATA_TABLE} (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def get_runtime_metadata(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    ensure_runtime_metadata_table(conn)
    row = conn.execute(
        f"SELECT value FROM {RUNTIME_METADATA_TABLE} WHERE key = ?",
        (key,),
    ).fetchone()
    return str(row["value"]) if row and row["value"] is not None else default


def set_runtime_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    ensure_runtime_metadata_table(conn)
    conn.execute(
        f"""
        INSERT INTO {RUNTIME_METADATA_TABLE} (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def build_descriptor(*, store_raw_json: bool = False) -> Dict[str, Any]:
    return {
        "version": BUILD_DESCRIPTOR_VERSION,
        "store_raw_json": bool(store_raw_json),
    }


def write_build_runtime_metadata(
    conn: sqlite3.Connection,
    *,
    source_fingerprint: Dict[str, Any],
    store_raw_json: bool = False,
) -> None:
    set_runtime_metadata(
        conn,
        BUILD_SOURCE_FINGERPRINT_KEY,
        json.dumps(source_fingerprint, ensure_ascii=False, sort_keys=True),
    )
    set_runtime_metadata(
        conn,
        BUILD_DESCRIPTOR_KEY,
        json.dumps(build_descriptor(store_raw_json=store_raw_json), ensure_ascii=False, sort_keys=True),
    )


def load_build_runtime_metadata(db_path: str | Path | None = None) -> Dict[str, Any]:
    database_path = resolve_db_path(db_path)
    if not database_path.exists():
        return {}
    try:
        with connect_db(database_path) as conn:
            source_payload = get_runtime_metadata(conn, BUILD_SOURCE_FINGERPRINT_KEY, "")
            descriptor_payload = get_runtime_metadata(conn, BUILD_DESCRIPTOR_KEY, "")
            paper_count = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] or 0)
    except Exception:
        return {}
    if not source_payload or paper_count <= 0:
        return {}
    try:
        source_fingerprint = json.loads(source_payload)
    except Exception:
        return {}
    try:
        descriptor = json.loads(descriptor_payload) if descriptor_payload else {}
    except Exception:
        descriptor = {}
    return {
        "db_path": str(database_path),
        "paper_count": paper_count,
        "source_fingerprint": source_fingerprint,
        "build_descriptor": descriptor,
    }


def database_matches_build(
    db_path: str | Path | None,
    *,
    source_fingerprint: Dict[str, Any],
    store_raw_json: bool = False,
) -> bool:
    metadata = load_build_runtime_metadata(db_path)
    if not metadata:
        return False
    if metadata.get("source_fingerprint") != source_fingerprint:
        return False
    existing_descriptor = metadata.get("build_descriptor") or {}
    existing_version = clean_text(existing_descriptor.get("version"))
    if existing_version and existing_version != BUILD_DESCRIPTOR_VERSION:
        return False
    existing_store_raw_json = bool(existing_descriptor.get("store_raw_json"))
    if store_raw_json and not existing_store_raw_json:
        return False
    return True


def compute_paper_content_signature(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS paper_count,
            MIN(paper_id) AS min_paper_id,
            MAX(paper_id) AS max_paper_id,
            SUM(LENGTH(fulltext_for_sparse)) AS sparse_len_sum,
            SUM(LENGTH(embedding_text)) AS embedding_len_sum
        FROM papers
        """
    ).fetchone()
    payload = {
        "paper_count": int(row["paper_count"] or 0),
        "min_paper_id": clean_text(row["min_paper_id"]),
        "max_paper_id": clean_text(row["max_paper_id"]),
        "sparse_len_sum": int(row["sparse_len_sum"] or 0),
        "embedding_len_sum": int(row["embedding_len_sum"] or 0),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def get_paper_content_signature(
    db_path: str | Path | None = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> str:
    if conn is not None:
        cached = get_runtime_metadata(conn, PAPER_CONTENT_SIGNATURE_KEY, "")
        if cached:
            return cached
        signature = compute_paper_content_signature(conn)
        set_runtime_metadata(conn, PAPER_CONTENT_SIGNATURE_KEY, signature)
        conn.commit()
        return signature

    database_path = resolve_db_path(db_path)
    with connect_db(database_path) as local_conn:
        return get_paper_content_signature(conn=local_conn)


def exact_index_cache_dir() -> Path:
    ensure_system_layout()
    path = SYSTEM_OUTPUT_DIR / "cache" / EXACT_INDEX_DISK_CACHE_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def exact_index_cache_path(signature: str) -> Path:
    safe_signature = re.sub(r"[^A-Za-z0-9._-]+", "_", signature)[:48] or "missing"
    return exact_index_cache_dir() / f"exact_{EXACT_INDEX_CACHE_VERSION}_{safe_signature}.pkl.gz"


def load_exact_index_from_disk(signature: str) -> Optional[Dict[str, Any]]:
    cache_path = exact_index_cache_path(signature)
    if not cache_path.exists():
        return None
    try:
        with gzip.open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != EXACT_INDEX_CACHE_VERSION:
            return None
        if payload.get("signature") != signature:
            return None
        index = payload.get("index")
        if not isinstance(index, dict) or not isinstance(index.get("papers"), list):
            return None
        return index
    except Exception:
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def persist_exact_index_to_disk(signature: str, index: Dict[str, Any]) -> None:
    cache_path = exact_index_cache_path(signature)
    payload = {"version": EXACT_INDEX_CACHE_VERSION, "signature": signature, "index": index}
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
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# 初始化空数据库并确保基础 schema 完整。
def initialize_database(db_path: Path) -> sqlite3.Connection:
    conn = connect_db(db_path)
    apply_build_pragmas(conn)
    reset_database_schema(conn)
    return conn


# 在重建数据库前备份运行期表，避免历史和收藏信息丢失。
def backup_runtime_tables(db_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not db_path.exists():
        return {"search_history": [], "saved_papers": []}

    try:
        with connect_db(db_path) as conn:
            search_history = [
                {
                    "query_text": row["query_text"],
                    "intent_frame_json": row["intent_frame_json"],
                    "created_at": row["created_at"],
                }
                for row in conn.execute(
                    """
                    SELECT query_text, intent_frame_json, created_at
                    FROM search_history
                    ORDER BY id
                    """
                ).fetchall()
            ]
            saved_papers = [
                {
                    "paper_id": row["paper_id"],
                    "saved_at": row["saved_at"],
                }
                for row in conn.execute(
                    """
                    SELECT paper_id, saved_at
                    FROM saved_papers
                    ORDER BY id
                    """
                ).fetchall()
            ]
    except sqlite3.Error:
        return {"search_history": [], "saved_papers": []}

    return {"search_history": search_history, "saved_papers": saved_papers}


# 建库完成后把运行期表恢复回来，延续用户历史状态。
def restore_runtime_tables(conn: sqlite3.Connection, backup_payload: Dict[str, List[Dict[str, Any]]]) -> None:
    search_history = backup_payload.get("search_history", [])
    if search_history:
        conn.executemany(
            """
            INSERT INTO search_history (query_text, intent_frame_json, created_at)
            VALUES (?, ?, ?)
            """,
            [
                (item["query_text"], item["intent_frame_json"], item["created_at"])
                for item in search_history
            ],
        )

    saved_papers = backup_payload.get("saved_papers", [])
    if saved_papers:
        # Skip stale saved papers missing from the rebuilt corpus to avoid FK errors.
        existing_paper_ids = {
            row["paper_id"]
            for row in conn.execute("SELECT paper_id FROM papers")
        }
        conn.executemany(
            """
            INSERT OR IGNORE INTO saved_papers (paper_id, saved_at)
            VALUES (?, ?)
            """,
            [
                (item["paper_id"], item["saved_at"])
                for item in saved_papers
                if item["paper_id"] in existing_paper_ids
            ],
        )
    conn.commit()


def insert_raw_paper(conn: sqlite3.Connection, raw_record: RawPaperRecord) -> None:
    conn.execute(
        """
        INSERT INTO raw_papers (paper_id, source_path, raw_json)
        VALUES (?, ?, ?)
        """,
        (
            raw_record.paper_id,
            raw_record.source_path,
            json.dumps(raw_record.raw_json, ensure_ascii=False),
        ),
    )


def insert_paper(conn: sqlite3.Connection, index_record: PaperIndexRecord) -> None:
    conn.execute(
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
        index_record.to_storage_row(),
    )


def insert_raw_papers(conn: sqlite3.Connection, raw_records: Sequence[RawPaperRecord]) -> None:
    if not raw_records:
        return
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


def insert_papers(conn: sqlite3.Connection, index_records: Sequence[PaperIndexRecord]) -> None:
    if not index_records:
        return
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


def insert_sections(conn: sqlite3.Connection, section_records: Sequence[PaperSectionRecord]) -> None:
    if not section_records:
        return

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
            for record in section_records
        ],
    )


# 把论文标题、摘要和章节片段写入 FTS 表，供全文检索使用。
def build_fts_index(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM paper_search_fts")
    conn.execute(
        """
        INSERT INTO paper_search_fts (paper_id, title, abstract, section_titles, section_snippet)
        SELECT
            papers.paper_id,
            papers.title,
            papers.abstract,
            COALESCE(papers.section_titles, ''),
            COALESCE(section_snippets.section_snippet, '')
        FROM papers
        LEFT JOIN (
            SELECT
                paper_id,
                GROUP_CONCAT(section_snippet, '\n') AS section_snippet
            FROM paper_sections
            GROUP BY paper_id
        ) AS section_snippets
            ON section_snippets.paper_id = papers.paper_id
        ORDER BY papers.paper_id
        """
    )
    conn.commit()


def ensure_hot_fts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS paper_search_fts_hot_v2 USING fts5(
            paper_id UNINDEXED,
            title,
            authors,
            abstract,
            section_titles
        )
        """
    )


def ensure_hot_fts_vocab_table(conn: sqlite3.Connection) -> None:
    ensure_hot_fts_table(conn)
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {HOT_FTS_VOCAB_TABLE}
        USING fts5vocab({HOT_FTS_TABLE}, 'row')
        """
    )


def lookup_hot_fts_doc_counts(conn: sqlite3.Connection, tokens: Sequence[str]) -> Dict[str, int]:
    clean_tokens: List[str] = []
    seen = set()
    for token in tokens:
        text = clean_text(str(token))
        if not text or text in seen:
            continue
        seen.add(text)
        clean_tokens.append(text)
    if not clean_tokens:
        return {}
    ensure_hot_fts_vocab_table(conn)
    counts: Dict[str, int] = {}
    for start in range(0, len(clean_tokens), SQLITE_VARIABLE_CHUNK_SIZE):
        chunk = clean_tokens[start : start + SQLITE_VARIABLE_CHUNK_SIZE]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT term, doc FROM {HOT_FTS_VOCAB_TABLE} WHERE term IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            counts[str(row["term"])] = int(row["doc"] or 0)
    return counts


def select_informative_hot_fts_terms(
    conn: sqlite3.Connection,
    query: str,
    *,
    max_terms: int = 4,
) -> List[str]:
    tokens = tokenize_query(query)
    if not tokens:
        return []
    doc_counts = lookup_hot_fts_doc_counts(conn, tokens)
    present_terms = [token for token in tokens if doc_counts.get(token, 0) > 0]
    candidate_terms = present_terms or tokens
    ranked = sorted(
        candidate_terms,
        key=lambda token: (
            doc_counts.get(token, 10**9),
            -len(token),
            token,
        ),
    )
    return ranked[: max_terms or len(ranked)]


def build_hot_fts_match_query(
    conn: sqlite3.Connection,
    query: str,
    *,
    max_terms: int = 4,
) -> Tuple[str, List[str]]:
    selected_terms = select_informative_hot_fts_terms(conn, query, max_terms=max_terms)
    if not selected_terms:
        return "", []
    if len(selected_terms) == 1:
        return selected_terms[0], selected_terms
    return " AND ".join(selected_terms), selected_terms


def build_dense_hot_fts_queries(
    conn: sqlite3.Connection,
    query: str,
    *,
    max_terms: int = 4,
) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    selected_terms = select_informative_hot_fts_terms(conn, query, max_terms=max_terms)
    if not selected_terms:
        return [], []
    if len(selected_terms) == 1:
        return [(selected_terms[0], [selected_terms[0]])], selected_terms

    queries: List[Tuple[str, List[str]]] = []
    seen = set()
    strict_query = " AND ".join(selected_terms)
    queries.append((strict_query, list(selected_terms)))
    seen.add(strict_query)
    if len(selected_terms) <= 3:
        combo_sizes = list(range(len(selected_terms), 1, -1))
    else:
        combo_sizes = [3, 2]
    for combo_size in combo_sizes:
        for combo in combinations(selected_terms, combo_size):
            match_query = " AND ".join(combo)
            if match_query in seen:
                continue
            seen.add(match_query)
            queries.append((match_query, list(combo)))
    return queries[:8], selected_terms


def build_exact_hot_fts_queries(
    conn: sqlite3.Connection,
    query: str,
    *,
    max_terms: int = 6,
) -> List[str]:
    phrase_tokens = tokenize_query(query)
    selected_terms = select_informative_hot_fts_terms(conn, query, max_terms=max_terms)
    queries: List[str] = []
    seen = set()

    if 1 < len(phrase_tokens) <= 8:
        phrase_query = '"' + " ".join(phrase_tokens) + '"'
        queries.append(phrase_query)
        seen.add(phrase_query)

    if selected_terms:
        strict_query = " AND ".join(selected_terms)
        if strict_query not in seen:
            queries.append(strict_query)
            seen.add(strict_query)
        if len(selected_terms) >= 3:
            top_terms = selected_terms[: min(4, len(selected_terms))]
            for combo in combinations(top_terms, len(top_terms) - 1):
                combo_query = " AND ".join(combo)
                if combo_query in seen:
                    continue
                queries.append(combo_query)
                seen.add(combo_query)
    return queries[:6]


def hot_fts_select_sql(*, include_authors: bool = False) -> str:
    author_columns = ""
    if include_authors:
        author_columns = """
                papers.authors_raw,
                papers.normalized_authors,
        """
    return f"""
            SELECT
                {HOT_FTS_TABLE}.paper_id,
                papers.title,
                {author_columns}
                papers.abstract,
                papers.section_titles,
                bm25({HOT_FTS_TABLE}, 1.8, 0.8, 1.2, 0.8) AS bm25_score
            FROM {HOT_FTS_TABLE}
            JOIN papers ON papers.paper_id = {HOT_FTS_TABLE}.paper_id
            WHERE {HOT_FTS_TABLE} MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
    """


def build_hot_fts_index(conn: sqlite3.Connection, *, force_rebuild: bool = False) -> None:
    ensure_hot_fts_table(conn)
    paper_signature = get_paper_content_signature(conn=conn)
    cached_signature = get_runtime_metadata(conn, HOT_FTS_SIGNATURE_KEY, "")
    hot_row_count = int(conn.execute(f"SELECT COUNT(*) FROM {HOT_FTS_TABLE}").fetchone()[0] or 0)
    if not force_rebuild and cached_signature == paper_signature and hot_row_count > 0:
        return

    conn.execute(f"DELETE FROM {HOT_FTS_TABLE}")
    conn.execute(
        f"""
        INSERT INTO {HOT_FTS_TABLE} (paper_id, title, authors, abstract, section_titles)
        SELECT
            paper_id,
            title,
            COALESCE(authors_raw, ''),
            abstract,
            COALESCE(section_titles, '')
        FROM papers
        ORDER BY paper_id
        """
    )
    set_runtime_metadata(conn, HOT_FTS_SIGNATURE_KEY, paper_signature)
    conn.commit()


def ensure_query_runtime_artifacts(
    db_path: str | Path | None = None,
    *,
    build_exact_cache: bool = False,
) -> Dict[str, Any]:
    database_path = resolve_db_path(db_path)
    with connect_db(database_path) as conn:
        paper_signature = get_paper_content_signature(conn=conn)
        build_hot_fts_index(conn, force_rebuild=False)
        ensure_hot_fts_vocab_table(conn)

    exact_ready = False
    if build_exact_cache:
        exact_ready = True

    return {
        "db_path": str(database_path),
        "paper_signature": paper_signature,
        "hot_fts_ready": True,
        "hot_fts_vocab_ready": True,
        "exact_ready": exact_ready,
    }


def load_raw_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 全量扫描数据集并重建整个检索数据库。
def build_database(
    data_root: Path,
    db_path: Path,
    *,
    store_raw_json: bool = False,
    source_fingerprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    把全量论文 JSON 写入 SQLite，并构建 FTS 检索表。

    返回值是建库后的核心统计信息，方便命令行和反馈文件直接复用。
    """

    ensure_system_layout()
    data_source = resolve_dataset_root(data_root)
    runtime_backup = backup_runtime_tables(db_path)
    EXACT_SEARCH_CACHE.clear()

    raw_batch: List[RawPaperRecord] = []
    index_batch: List[PaperIndexRecord] = []
    section_batch: List[PaperSectionRecord] = []

    def flush_batches(conn: sqlite3.Connection) -> None:
        if not raw_batch and not index_batch and not section_batch:
            return
        if store_raw_json and raw_batch:
            insert_raw_papers(conn, raw_batch)
        insert_papers(conn, index_batch)
        insert_sections(conn, section_batch)
        conn.commit()
        raw_batch.clear()
        index_batch.clear()
        section_batch.clear()

    conn = initialize_database(db_path)
    try:
        for paper_id, source_path, raw_json in iter_dataset_json_records(data_source):
            section_records = extract_sections(raw_json, paper_id)
            index_record = parse_paper_json(raw_json, source_path, section_records=section_records)
            if store_raw_json:
                raw_batch.append(
                    RawPaperRecord(
                        paper_id=paper_id,
                        source_path=source_path,
                        raw_json=raw_json,
                    )
                )
            index_batch.append(index_record)
            section_batch.extend(section_records)
            if len(index_batch) >= DEFAULT_BUILD_BATCH_SIZE:
                flush_batches(conn)

        flush_batches(conn)
        build_fts_index(conn)
        paper_signature = get_paper_content_signature(conn=conn)
        set_runtime_metadata(conn, PAPER_CONTENT_SIGNATURE_KEY, paper_signature)
        if source_fingerprint is not None:
            write_build_runtime_metadata(
                conn,
                source_fingerprint=source_fingerprint,
                store_raw_json=store_raw_json,
            )
        build_hot_fts_index(conn, force_rebuild=True)
        restore_runtime_tables(conn, runtime_backup)
        stats = {
            "papers": conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "sections": conn.execute("SELECT COUNT(*) FROM paper_sections").fetchone()[0],
            "fts_rows": conn.execute("SELECT COUNT(*) FROM paper_search_fts").fetchone()[0],
        }
        if store_raw_json:
            stats["raw_papers"] = conn.execute("SELECT COUNT(*) FROM raw_papers").fetchone()[0]
    finally:
        conn.close()

    return stats


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is None:
        return get_active_runtime_db_path()
    return Path(db_path)


def database_exists(db_path: str | Path | None = None) -> bool:
    return resolve_db_path(db_path).exists()


def load_database_stats(db_path: str | Path | None = None) -> Dict[str, int]:
    database_path = resolve_db_path(db_path)
    if not database_path.exists():
        return {"papers": 0, "sections": 0, "fts_rows": 0}

    with connect_db(database_path) as conn:
        return {
            "papers": conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "sections": conn.execute("SELECT COUNT(*) FROM paper_sections").fetchone()[0],
            "fts_rows": conn.execute("SELECT COUNT(*) FROM paper_search_fts").fetchone()[0],
        }


def normalize_match_text(text: str) -> str:
    return clean_text(text).lower()


# 查询会先被拆成关键词 token，供多种检索策略复用。
def tokenize_query(query: str) -> List[str]:
    """
    把用户 query 规整成适合 SQLite FTS 的 token 列表。

    这里会统一转小写、拆分连字符方法名、去掉停用词，并做去重。
    """

    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/+:-]*", query.lower())
    tokens: List[str] = []
    for raw_token in raw_tokens:
        # 像 self-rag / graph-based 这类方法名会先切成更细的 token，
        # 让 SQLite FTS 更容易命中不同写法。
        for token in re.split(r"[-_/+:.]+", raw_token):
            token = token.strip()
            if len(token) < 2 or token in QUERY_STOPWORDS:
                continue
            tokens.append(token)

    seen = set()
    deduped: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


# 把自然语言查询改写成更适合 SQLite FTS 的表达式。
def prepare_fts_query(query: str) -> str:
    """
    把自然 query 转成 SQLite FTS 的 MATCH 表达式。

    当前实现是严格的 AND 逻辑，因此更适合短关键词，不适合很长的自然语言句子。
    """

    tokens = tokenize_query(query)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    # 当前采用严格的词项交集检索：
    # 多个 token 会被拼成 AND 查询，要求同一篇论文同时命中这些词。
    # 这样精度更高，但长自然语言 query 容易变得过严。
    return " AND ".join(tokens)


def first_match_index(text: str, normalized_query: str, tokens: Sequence[str]) -> int:
    text_lower = text.lower()
    if normalized_query:
        index = text_lower.find(normalized_query)
        if index >= 0:
            return index
    for token in tokens:
        index = text_lower.find(token)
        if index >= 0:
            return index
    return -1


# 为检索命中结果构造可展示的证据片段。
def build_snippet(text: str, normalized_query: str, tokens: Sequence[str], max_chars: int = 320) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    index = first_match_index(cleaned, normalized_query, tokens)
    if index < 0:
        return truncate(cleaned, max_chars=max_chars)

    start = max(0, index - max_chars // 4)
    end = min(len(cleaned), start + max_chars)
    window = cleaned[start:end].strip()

    if start > 0:
        window = "..." + window
    if end < len(cleaned):
        window = window + "..."
    return window


def score_text(text: str, normalized_query: str, tokens: Sequence[str]) -> int:
    text_norm = normalize_match_text(text)
    return score_normalized_text(text_norm, normalized_query, tokens)


def score_normalized_text(text_norm: str, normalized_query: str, tokens: Sequence[str]) -> int:
    if not text_norm:
        return 0

    score = 0
    if normalized_query and normalized_query in text_norm:
        score += 100
    score += sum(1 for token in tokens if token in text_norm)
    return score


# 批量读取论文章节，供证据展示和 rerank 解释使用。
def load_sections_for_papers(
    conn: sqlite3.Connection,
    paper_ids: Sequence[str],
) -> Dict[str, List[sqlite3.Row]]:
    if not paper_ids:
        return {}

    sections_by_paper: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for start in range(0, len(paper_ids), SQLITE_VARIABLE_CHUNK_SIZE):
        chunk = list(paper_ids[start : start + SQLITE_VARIABLE_CHUNK_SIZE])
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT paper_id, section_title, section_snippet
            FROM paper_sections
            WHERE paper_id IN ({placeholders})
            ORDER BY paper_id, section_order
            """,
            chunk,
        ).fetchall()
        for row in rows:
            sections_by_paper[row["paper_id"]].append(row)
    return sections_by_paper


def _exact_search_cache_key(database_path: Path) -> str:
    return get_paper_content_signature(database_path)


# 构建标题、作者和短语匹配用的内存索引，加速 exact match。
def _build_exact_search_index(database_path: Path) -> Dict[str, Any]:
    with connect_db(database_path) as conn:
        paper_rows = conn.execute(
            """
            SELECT paper_id, title, authors_raw, normalized_authors, abstract, section_titles
            FROM papers
            ORDER BY paper_id
            """
        ).fetchall()

    papers: List[Dict[str, Any]] = []
    for row in paper_rows:
        paper_id = row["paper_id"]
        title = clean_text(row["title"])
        authors_raw = clean_text(row["authors_raw"])
        abstract = clean_text(row["abstract"])
        try:
            normalized_authors = json.loads(row["normalized_authors"])
        except json.JSONDecodeError:
            normalized_authors = []
        try:
            section_titles = [clean_text(str(item)) for item in json.loads(row["section_titles"])]
        except json.JSONDecodeError:
            section_titles = []

        papers.append(
            {
                "paper_id": paper_id,
                "title": title,
                "title_norm": normalize_match_text(title),
                "authors_raw": authors_raw,
                "authors_norm": normalize_match_text(authors_raw),
                "normalized_authors": normalized_authors,
                "abstract": abstract,
                "abstract_norm": normalize_match_text(abstract),
                "section_titles": section_titles,
                "section_title_norms": [normalize_match_text(item) for item in section_titles],
            }
        )
    return {"papers": papers}


def _load_exact_search_index(database_path: Path) -> Dict[str, Any]:
    cache_key = _exact_search_cache_key(database_path)
    cached = EXACT_SEARCH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    disk_cached = load_exact_index_from_disk(cache_key)
    if disk_cached is not None:
        EXACT_SEARCH_CACHE.clear()
        EXACT_SEARCH_CACHE[cache_key] = disk_cached
        return disk_cached

    payload = _build_exact_search_index(database_path)
    persist_exact_index_to_disk(cache_key, payload)
    EXACT_SEARCH_CACHE.clear()
    EXACT_SEARCH_CACHE[cache_key] = payload
    return payload


def prepare_exact_search_index(db_path: str | Path | None = None) -> Dict[str, Any]:
    database_path = resolve_db_path(db_path)
    return _load_exact_search_index(database_path)


# 从命中的字段里挑选最适合展示给用户的证据摘要。
def pick_match_evidence(
    paper_row: sqlite3.Row,
    section_rows: Sequence[sqlite3.Row],
    query: str,
) -> tuple[str, str]:
    normalized_query = normalize_match_text(query)
    tokens = tokenize_query(query)

    title = clean_text(paper_row["title"])
    authors_raw = clean_text(paper_row["authors_raw"]) if "authors_raw" in paper_row.keys() else ""
    abstract = clean_text(paper_row["abstract"])
    try:
        section_titles = json.loads(paper_row["section_titles"])
    except json.JSONDecodeError:
        section_titles = []

    # 命中后优先选一个“最能解释为什么命中”的短证据片段，
    # 方便在 UI 中直接展示命中字段和对应文本。
    candidates: List[tuple[int, str, str]] = [
        (
            score_text(title, normalized_query, tokens) + 20,
            "title",
            build_snippet(title, normalized_query, tokens),
        ),
        (
            score_text(abstract, normalized_query, tokens) + 10,
            "abstract",
            build_snippet(abstract, normalized_query, tokens),
        ),
    ]
    if authors_raw:
        candidates.append(
            (
                score_text(authors_raw, normalized_query, tokens) + 15,
                "authors",
                build_snippet(authors_raw, normalized_query, tokens),
            )
        )

    best_section_title = ""
    best_section_title_score = 0
    for section_title in section_titles:
        section_title_text = clean_text(str(section_title))
        current_score = score_text(section_title_text, normalized_query, tokens)
        if current_score > best_section_title_score:
            best_section_title_score = current_score
            best_section_title = section_title_text

    if best_section_title:
        candidates.append(
            (
                best_section_title_score + 8,
                "section_titles",
                build_snippet(best_section_title, normalized_query, tokens),
            )
        )

    best_section_snippet = ""
    best_section_snippet_score = 0
    for row in section_rows:
        combined_text = "\n".join([clean_text(row["section_title"]), clean_text(row["section_snippet"])])
        current_score = score_text(combined_text, normalized_query, tokens)
        if current_score > best_section_snippet_score:
            best_section_snippet_score = current_score
            best_section_snippet = clean_text(row["section_snippet"]) or clean_text(row["section_title"])

    if best_section_snippet:
        candidates.append(
            (
                best_section_snippet_score + 6,
                "section_snippet",
                build_snippet(best_section_snippet, normalized_query, tokens),
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    for score_value, matched_field, matched_snippet in candidates:
        if score_value > 0 and matched_snippet:
            return matched_field, matched_snippet

    return "title", title or abstract or "暂无可用片段。"


# 稀疏检索：基于 SQLite FTS 返回文本命中结果。
def search_basic(
    query: str,
    top_k: int = 10,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    执行纯 FTS 检索。

    该路径召回速度快、覆盖面广，但本质上仍然是词项匹配，
    对很长或很多限定条件的 query 会比较敏感。
    """

    database_path = resolve_db_path(db_path)
    fts_query = prepare_fts_query(query)
    if not database_path.exists() or not fts_query:
        return []

    with connect_db(database_path) as conn:
        ensure_hot_fts_table(conn)
        rows = conn.execute(
            """
            SELECT
                paper_search_fts.paper_id,
                papers.title,
                papers.abstract,
                papers.section_titles,
                bm25(paper_search_fts, 1.8, 1.2, 0.8, 0.6) AS bm25_score
            FROM paper_search_fts
            JOIN papers ON papers.paper_id = paper_search_fts.paper_id
            WHERE paper_search_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()

        sections_by_paper = load_sections_for_papers(conn, [row["paper_id"] for row in rows])
        results: List[Dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            matched_field, matched_snippet = pick_match_evidence(
                row,
                sections_by_paper.get(row["paper_id"], []),
                query,
            )
            raw_score = row["bm25_score"] if row["bm25_score"] is not None else 0.0
            results.append(
                {
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "matched_field": matched_field,
                    "matched_snippet": matched_snippet,
                    "fts_score": round(-float(raw_score), 6),
                    "rank": rank,
                }
            )

    return results


def search_basic_fast(
    query: str,
    top_k: int = 10,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    database_path = resolve_db_path(db_path)
    if not database_path.exists():
        return []

    with connect_db(database_path) as conn:
        build_hot_fts_index(conn, force_rebuild=False)
        fts_query, _ = build_hot_fts_match_query(conn, query, max_terms=4)
        if not fts_query:
            return []
        rows = conn.execute(hot_fts_select_sql(include_authors=True), (fts_query, top_k)).fetchall()

        results: List[Dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            matched_field, matched_snippet = pick_match_evidence(
                row,
                [],
                query,
            )
            raw_score = row["bm25_score"] if row["bm25_score"] is not None else 0.0
            results.append(
                {
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "matched_field": matched_field,
                    "matched_snippet": matched_snippet,
                    "fts_score": round(-float(raw_score), 6),
                    "rank": rank,
                }
            )
    return results


def search_dense_fast(
    query: str,
    top_k: int = 10,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    database_path = resolve_db_path(db_path)
    if not database_path.exists():
        return []

    with connect_db(database_path) as conn:
        build_hot_fts_index(conn, force_rebuild=False)
        subqueries, selected_terms = build_dense_hot_fts_queries(conn, query, max_terms=4)
        if not subqueries:
            return []

        aggregated: Dict[str, Dict[str, Any]] = {}
        candidate_limit = max(top_k * 2, 40)
        total_term_count = max(len(selected_terms), 1)
        for match_query, matched_terms in subqueries:
            rows = conn.execute(
                hot_fts_select_sql(include_authors=False),
                (match_query, candidate_limit),
            ).fetchall()
            coverage = len(matched_terms) / total_term_count
            for row in rows:
                raw_score = -float(row["bm25_score"] or 0.0)
                combined_score = raw_score + (coverage * 3.0)
                existing = aggregated.get(row["paper_id"])
                if existing is None or combined_score > existing["dense_raw_score"]:
                    aggregated[row["paper_id"]] = {
                        "paper_id": row["paper_id"],
                        "title": row["title"],
                        "dense_raw_score": round(combined_score, 6),
                        "matched_terms": list(matched_terms),
                    }

    results = sorted(
        aggregated.values(),
        key=lambda item: (-float(item["dense_raw_score"]), item["title"]),
    )
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results[:top_k]


def pick_exact_author_match(authors: Sequence[str], query: str, tokens: Sequence[str]) -> str:
    normalized_query = normalize_match_text(query)
    best_author = ""
    best_score = 0
    for author in authors:
        author_text = clean_text(str(author))
        author_norm = normalize_match_text(author_text)
        score = 0
        if normalized_query and normalized_query == author_norm:
            score = 260
        elif normalized_query and normalized_query in author_norm:
            score = 230
        elif tokens and all(token in author_norm for token in tokens):
            score = 210
        if score > best_score:
            best_score = score
            best_author = author_text
    return best_author


# 精确匹配：优先覆盖标题线索、作者名和短语命中场景。
def search_exact_matches(
    query: str,
    top_k: int = 10,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    执行基于标题、作者和短语命中的精确匹配。

    它更适合作者名、标题 hint、方法名、数据集名等高精度查询。
    """

    database_path = resolve_db_path(db_path)
    normalized_query = normalize_match_text(query)
    tokens = tokenize_query(query)
    if not database_path.exists() or not normalized_query:
        return []

    with connect_db(database_path) as conn:
        build_hot_fts_index(conn, force_rebuild=False)
        candidate_queries = build_exact_hot_fts_queries(conn, query, max_terms=6)
        if not candidate_queries:
            return []

        candidate_rows: Dict[str, Dict[str, Any]] = {}
        candidate_limit = max(top_k * 8, 120)
        for match_query in candidate_queries:
            rows = conn.execute(
                hot_fts_select_sql(include_authors=True),
                (match_query, candidate_limit),
            ).fetchall()
            for row in rows:
                if row["paper_id"] in candidate_rows:
                    continue
                try:
                    normalized_authors = json.loads(row["normalized_authors"])
                except json.JSONDecodeError:
                    normalized_authors = []
                try:
                    section_titles = [clean_text(str(item)) for item in json.loads(row["section_titles"])]
                except json.JSONDecodeError:
                    section_titles = []
                candidate_rows[row["paper_id"]] = {
                    "paper_id": row["paper_id"],
                    "title": clean_text(row["title"]),
                    "title_norm": normalize_match_text(row["title"]),
                    "authors_raw": clean_text(row["authors_raw"]),
                    "authors_norm": normalize_match_text(row["authors_raw"]),
                    "normalized_authors": normalized_authors,
                    "abstract": clean_text(row["abstract"]),
                    "abstract_norm": normalize_match_text(row["abstract"]),
                    "section_titles": section_titles,
                    "section_title_norms": [normalize_match_text(item) for item in section_titles],
                }
        paper_rows = list(candidate_rows.values())

    results: List[Dict[str, Any]] = []
    for row in paper_rows:
        title = row["title"]
        title_norm = row["title_norm"]
        authors_raw = row["authors_raw"]
        abstract = row["abstract"]
        normalized_authors = row["normalized_authors"]
        section_titles = row["section_titles"]
        section_title_norms = row["section_title_norms"]

        matches: List[Dict[str, Any]] = []

        if normalized_query == title_norm:
            matches.append(
                {
                    "match_type": "title_hint",
                    "matched_field": "title",
                    "matched_snippet": title,
                    "exact_score": 300,
                }
            )
        elif normalized_query in title_norm:
            matches.append(
                {
                    "match_type": "title_hint",
                    "matched_field": "title",
                    "matched_snippet": title,
                    "exact_score": 260,
                }
            )
        elif tokens and all(token in title_norm for token in tokens):
            matches.append(
                {
                    "match_type": "title_hint",
                    "matched_field": "title",
                    "matched_snippet": title,
                    "exact_score": 220,
                }
            )

        best_author = pick_exact_author_match(normalized_authors, query, tokens)
        if best_author:
            matches.append(
                {
                    "match_type": "author_match",
                    "matched_field": "authors",
                    "matched_snippet": best_author,
                    "exact_score": 240,
                }
            )
        elif tokens and len(tokens) >= 2 and all(token in row["authors_norm"] for token in tokens):
            matches.append(
                {
                    "match_type": "author_match",
                    "matched_field": "authors",
                    "matched_snippet": build_snippet(authors_raw, normalized_query, tokens),
                    "exact_score": 200,
                }
            )

        if len(tokens) <= 8:
            if normalized_query in row["abstract_norm"]:
                matches.append(
                    {
                        "match_type": "phrase_match",
                        "matched_field": "abstract",
                        "matched_snippet": build_snippet(abstract, normalized_query, tokens),
                        "exact_score": 200,
                    }
                )

            best_section_title = ""
            best_section_title_score = 0
            for section_title_text, section_title_norm in zip(section_titles, section_title_norms):
                current_score = score_normalized_text(section_title_norm, normalized_query, tokens)
                if current_score > best_section_title_score:
                    best_section_title_score = current_score
                    best_section_title = section_title_text
            if best_section_title_score >= max(2, len(tokens)):
                matches.append(
                    {
                        "match_type": "phrase_match",
                        "matched_field": "section_titles",
                        "matched_snippet": build_snippet(best_section_title, normalized_query, tokens),
                        "exact_score": 190,
                    }
                )

        if not matches:
            continue

        best_match = max(matches, key=lambda item: item["exact_score"])
        results.append(
            {
                "paper_id": row["paper_id"],
                "title": title,
                **best_match,
            }
        )

    results.sort(key=lambda item: (-item["exact_score"], item["title"]))
    return results[:top_k]


# 混合检索：融合稀疏检索和精确匹配结果，得到更稳健的候选集。
def search_hybrid(
    query: str,
    top_k: int = 10,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    合并 FTS 召回和 exact match 加权结果。

    这是当前项目默认的检索入口，兼顾一定召回能力和更好的排序质量。
    """

    basic_results = search_basic(query, top_k=top_k * 3, db_path=db_path)
    exact_results = search_exact_matches(query, top_k=top_k * 3, db_path=db_path)
    merged: Dict[str, Dict[str, Any]] = {}

    # hybrid 的思路是：
    # 先用 FTS 做召回，再用标题 hint、作者名、短语命中去加分，
    # 让“既被关键词召回、又更像用户真正想找的论文”排得更靠前。
    for result in basic_results:
        merged[result["paper_id"]] = {
            **result,
            "match_type": "fts",
            "exact_score": None,
            "hybrid_score": result.get("fts_score", 0.0) or 0.0,
        }

    for result in exact_results:
        existing = merged.get(result["paper_id"])
        if existing is None:
            merged[result["paper_id"]] = {
                **result,
                "fts_score": None,
                "rank": None,
                "hybrid_score": result["exact_score"],
            }
            continue

        existing["match_type"] = result["match_type"]
        existing["matched_field"] = result["matched_field"]
        existing["matched_snippet"] = result["matched_snippet"]
        existing["exact_score"] = result["exact_score"]
        existing["hybrid_score"] = (existing.get("fts_score") or 0.0) + result["exact_score"]

    combined = sorted(
        merged.values(),
        key=lambda item: (
            -(item.get("hybrid_score") or 0.0),
            -(item.get("exact_score") or 0),
            -(item.get("fts_score") or 0.0),
            item["title"],
        ),
    )
    return combined[:top_k]


def dump_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_text(path: Path, content: str) -> None:
    ensure_parent_dir(path)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def format_results_block(results: Sequence[Dict[str, Any]]) -> str:
    if not results:
        return "无结果。"

    lines: List[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}")
        lines.append(f"论文 ID: {result['paper_id']}")
        lines.append(f"命中字段: {localize_field_label(result.get('matched_field', ''))}")
        lines.append(f"命中片段: {result.get('matched_snippet', '')}")
        if result.get("fts_score") is not None:
            lines.append(f"FTS 分数: {result['fts_score']}")
        if result.get("exact_score") is not None:
            lines.append(f"精确匹配分数: {result['exact_score']}")
        if result.get("match_type"):
            lines.append(f"命中类型: {localize_match_type_label(result['match_type'])}")
        lines.append("")
    return "\n".join(lines).strip()


# 批量运行默认调试查询，验证检索质量并产出样例。
def run_debug_queries(
    db_path: Path,
    query_json_path: Path,
    query_text_path: Path,
    top_k: int = 10,
    queries: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """批量运行预设 query，并导出 JSON/TXT 调试结果供人工检查。"""

    query_list = list(queries or DEFAULT_DEBUG_QUERIES)
    results: List[Dict[str, Any]] = []
    text_blocks: List[str] = []

    for query in query_list:
        basic_results = search_basic(query, top_k=top_k, db_path=db_path)
        exact_results = search_exact_matches(query, top_k=top_k, db_path=db_path)
        hybrid_results = search_hybrid(query, top_k=top_k, db_path=db_path)

        results.append(
            {
                "query": query,
                "basic_results": basic_results,
                "exact_results": exact_results,
                "hybrid_results": hybrid_results,
            }
        )

        text_blocks.append(f"查询: {query}")
        text_blocks.append("[FTS]")
        text_blocks.append(format_results_block(basic_results))
        text_blocks.append("")
        text_blocks.append("[精确匹配]")
        text_blocks.append(format_results_block(exact_results))
        text_blocks.append("")
        text_blocks.append("[混合检索]")
        text_blocks.append(format_results_block(hybrid_results))
        text_blocks.append("")
        text_blocks.append("=" * 80)
        text_blocks.append("")

    payload = {
        "db_path": str(db_path),
        "query_count": len(query_list),
        "top_k": top_k,
        "queries": results,
    }
    dump_json(query_json_path, payload)
    dump_text(query_text_path, "\n".join(text_blocks))
    return payload


# 输出建库和调试查询的反馈报告。
def write_feedback(
    db_path: Path,
    stats: Dict[str, int],
    query_payload: Optional[Dict[str, Any]],
    feedback_path: Path,
) -> None:
    query_count = query_payload["query_count"] if query_payload else 0
    content = f"""
统一检索流水线执行完成

执行范围
- 项目目录: {PROJECT_ROOT}
- 数据库文件: {db_path}
- 写入 papers: {stats['papers']}
- 写入 paper_sections: {stats['sections']}
- 写入 FTS 行数: {stats['fts_rows']}
- 调试 query 数量: {query_count}

已完成能力
- tar 数据中的 JSON 已全量解析并写入 SQLite。
- papers 和 paper_sections 已完成全量入库。
- FTS5 已建立并可做基础关键词检索。
- exact match 已支持标题 hint、作者粗匹配、短语级方法/数据集匹配。
- 检索结果默认返回标题、命中字段、命中片段与分数。

产物
- {db_path.name}
- {DEFAULT_QUERY_JSON_PATH.name if query_payload else '未生成 query json'}
- {DEFAULT_QUERY_TEXT_PATH.name if query_payload else '未生成 query text'}

完成标准
- 可以在全量数据库上稳定检索。
- 不再依赖旧的 JSON 逐文件扫描检索。
- 可以返回“标题 + 证据片段 + 命中字段”。
"""
    dump_text(feedback_path, content)


# CLI 构建命令入口。
def run_build_command(args: argparse.Namespace) -> None:
    stats = build_database(args.data_root, args.db_path)
    query_payload = None
    if not args.skip_debug_queries:
        query_payload = run_debug_queries(
            db_path=args.db_path,
            query_json_path=args.query_json_path,
            query_text_path=args.query_text_path,
            top_k=args.top_k,
        )
    write_feedback(args.db_path, stats, query_payload, args.feedback_path)


# CLI 单次检索入口。
def run_search_command(args: argparse.Namespace) -> None:
    search_map = {
        "basic": search_basic,
        "exact": search_exact_matches,
        "hybrid": search_hybrid,
    }
    results = search_map[args.mode](args.query, top_k=args.top_k, db_path=args.db_path)
    print(f"查询: {args.query}")
    print(format_results_block(results))


# CLI 调试查询入口。
def run_debug_command(args: argparse.Namespace) -> None:
    payload = run_debug_queries(
        db_path=args.db_path,
        query_json_path=args.query_json_path,
        query_text_path=args.query_text_path,
        top_k=args.top_k,
    )
    print(f"已将 {payload['query_count']} 条查询日志写入 {args.query_json_path} 和 {args.query_text_path}")


# 脚本入口：根据子命令执行建库、检索或调试。
def main() -> None:
    args = parse_args()
    if args.command is None:
        args = argparse.Namespace(
            command="build",
            data_root=get_default_dataset_source(),
            db_path=DEFAULT_DB_PATH,
            query_json_path=DEFAULT_QUERY_JSON_PATH,
            query_text_path=DEFAULT_QUERY_TEXT_PATH,
            feedback_path=DEFAULT_FEEDBACK_PATH,
            top_k=10,
            skip_debug_queries=False,
        )

    if args.command == "build":
        run_build_command(args)
    elif args.command == "search":
        run_search_command(args)
    elif args.command == "debug-queries":
        run_debug_command(args)
    else:
        raise ValueError(f"不支持的命令：{args.command}")


if __name__ == "__main__":
    main()
