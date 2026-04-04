"""
旧版 JSON 直接检索脚本。

它不依赖 Day 2 的 SQLite 数据库，而是直接扫描本地 JSON 文件做关键词过滤。
现在它已经不是主检索链路，主要保留作快速排查、对比实验和兼容旧用法。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dataset_config import DATASET_DIR, DATASET_LABEL, DATASET_RELATIVE_PATH, PROJECT_ROOT


SEARCH_MODE_AND = "AND"
SEARCH_MODE_OR = "OR"
EXCLUDED_STATUSES = ("Withdraw", "Reject", "Desk Reject")
DEFAULT_FIELDS = ["title", "authors", "abstract", "sections"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _parse_keywords(keyword_str: str) -> List[str]:
    return [k.strip().lower() for k in keyword_str.replace(",", " ").split() if k.strip()]


def _resolve_input_path(input_path: str | Sequence[str]) -> Path:
    if isinstance(input_path, (list, tuple)):
        candidate = Path(*input_path)
    else:
        candidate = Path(str(input_path))
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _normalize_record(payload: Dict[str, Any], source_path: Optional[Path], index: int = 0) -> Dict[str, Any]:
    record = dict(payload)
    if source_path is not None:
        record.setdefault("id", source_path.stem if index == 0 else f"{source_path.stem}#{index}")
        record.setdefault("source_path", source_path.relative_to(PROJECT_ROOT).as_posix())
    else:
        record.setdefault("id", record.get("paper_id", f"record-{index}"))
        record.setdefault("source_path", "")
    record.setdefault("source", DATASET_LABEL)
    return record


def _load_json_file(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        return []
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in file: %s", path)
        return []

    if isinstance(payload, dict):
        return [_normalize_record(payload, path)]
    if isinstance(payload, list):
        normalized: List[Dict[str, Any]] = []
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                normalized.append(_normalize_record(item, path, index))
        return normalized

    logger.warning("Unsupported JSON top-level structure in %s", path)
    return []


def load_data(input_file: str | Sequence[str]) -> Optional[List[Dict[str, Any]]]:
    """加载单个 JSON 文件或整个目录下的 JSON 文件，返回统一的记录列表。"""

    absolute_path = _resolve_input_path(input_file)

    if absolute_path.is_dir():
        records: List[Dict[str, Any]] = []
        for path in sorted(absolute_path.glob("*.json")):
            records.extend(_load_json_file(path))
        return records

    if absolute_path.is_file():
        return _load_json_file(absolute_path)

    logger.error("Input path not found: %s", absolute_path)
    return None


def _match_keyword_in_fields(item: Dict[str, Any], keyword: str, fields: List[str]) -> bool:
    keyword_lower = keyword.lower()
    if not fields:
        try:
            full_text = json.dumps(item, ensure_ascii=False).lower()
            return keyword_lower in full_text
        except Exception:
            return any(keyword_lower in str(item.get(field, "")).lower() for field in DEFAULT_FIELDS)
    return any(keyword_lower in str(item.get(field, "")).lower() for field in fields)


def _filter_by_search_mode(
    items: List[Dict[str, Any]],
    keywords: List[str],
    fields: List[str],
    search_mode: str,
) -> List[Dict[str, Any]]:
    if search_mode.upper() == SEARCH_MODE_AND:
        return [item for item in items if all(_match_keyword_in_fields(item, kw, fields) for kw in keywords)]
    return [item for item in items if any(_match_keyword_in_fields(item, kw, fields) for kw in keywords)]


def filter_data(
    data: List[Dict[str, Any]],
    keyword: str,
    fields_to_search: List[str],
    search_mode: str = SEARCH_MODE_OR,
    include_rejected: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按照关键词、字段范围和 AND/OR 模式过滤旧版 JSON 记录。"""

    if include_rejected:
        status_filtered = data
    else:
        status_filtered = [item for item in data if item.get("status") not in EXCLUDED_STATUSES]

    keywords = _parse_keywords(keyword)
    if not keywords:
        return status_filtered, []

    filtered = _filter_by_search_mode(status_filtered, keywords, fields_to_search, search_mode)
    return status_filtered, filtered


def count_results(
    data: List[Dict[str, Any]],
    status_filtered: List[Dict[str, Any]],
    filtered: List[Dict[str, Any]],
    keyword: str,
    fields: List[str],
    search_mode: str = SEARCH_MODE_OR,
) -> Dict[str, int]:
    keywords = _parse_keywords(keyword)
    if not keywords:
        return {
            "retrieval_before_status_filter": 0,
            "status_filtered_count": len(status_filtered),
            "retrieval_filtered_count": 0,
        }

    retrieval_before_filter = _filter_by_search_mode(data, keywords, fields, search_mode)
    return {
        "retrieval_before_status_filter": len(retrieval_before_filter),
        "status_filtered_count": len(status_filtered),
        "retrieval_filtered_count": len(filtered),
    }


def main() -> None:
    """旧版 JSON 检索脚本入口，负责解析参数、执行过滤并导出结果。"""

    parser = argparse.ArgumentParser(description="Search local paper JSON files by keyword.")
    parser.add_argument("keyword", help="Keyword(s) to search for. Multiple keywords can be comma or space separated.")
    parser.add_argument(
        "-i",
        "--input_path",
        default=DATASET_RELATIVE_PATH,
        help=f"Input JSON file or directory relative to project root (default: {DATASET_RELATIVE_PATH})",
    )
    parser.add_argument("-o", "--output_file", help="Output JSON filename")
    parser.add_argument(
        "-f",
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help=f"Fields to search in (default: {' '.join(DEFAULT_FIELDS)})",
    )
    parser.add_argument(
        "-m",
        "--search_mode",
        choices=[SEARCH_MODE_AND, SEARCH_MODE_OR],
        default=SEARCH_MODE_OR,
        help=f"{SEARCH_MODE_AND} requires all keywords to match, {SEARCH_MODE_OR} requires any keyword to match",
    )
    parser.add_argument(
        "--include_rejected",
        action="store_true",
        help="Include rejected and withdrawn papers in the results when status fields exist",
    )

    args = parser.parse_args()

    if not args.output_file:
        input_name = os.path.basename(str(args.input_path).rstrip(os.sep)) or "results"
        args.output_file = f"{input_name}-{args.keyword}.json"

    data = load_data(args.input_path)
    if data is None:
        return

    status_filtered, filtered = filter_data(
        data,
        args.keyword,
        args.fields,
        args.search_mode,
        args.include_rejected,
    )
    counts = count_results(data, status_filtered, filtered, args.keyword, args.fields, args.search_mode)

    output_data = {
        "dataset": DATASET_LABEL,
        "input_path": str(args.input_path),
        "retrieval_before_status_filter": counts["retrieval_before_status_filter"],
        "status_filtered_count": counts["status_filtered_count"],
        "retrieval_filtered_count": len(filtered),
        "filtered_papers": filtered,
    }

    with open(args.output_file, "w", encoding="utf-8") as fw:
        json.dump(output_data, fw, ensure_ascii=False, indent=2)
    logger.info("Filtered data has been written to: %s", args.output_file)


if __name__ == "__main__":
    main()
