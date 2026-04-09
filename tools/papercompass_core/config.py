"""
项目级配置与运行时路径中心。

这个文件统一管理数据集位置、运行目录、缓存目录、评估输出路径以及
运行时数据库路径，其他模块尽量只依赖这里暴露的配置入口。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import Request, urlopen


# 下面这组常量定义了项目根目录、默认数据集和运行产物的整体布局。
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_LABEL = "arXiv 2025-02 cs.CL"
DATASET_NAME = "arxiv_202502_cs_cl"
DATASET_PARENT_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATASET_PARENT_DIR / DATASET_NAME
DATASET_RELATIVE_PATH = DATASET_DIR.relative_to(PROJECT_ROOT).as_posix()

BUNDLED_DATA_DIR = PROJECT_ROOT / "bundled_data"
DATASET_TAR_GZ_PATH = BUNDLED_DATA_DIR / f"{DATASET_NAME}.tar.gz"
DATASET_TAR_PATH = BUNDLED_DATA_DIR / f"{DATASET_NAME}.tar"
DATASET_TAR_GZ_PART_PREFIX = BUNDLED_DATA_DIR / f"{DATASET_NAME}.tar.gz.part"
DATASET_TAR_GZ_PART_GLOB = f"{DATASET_NAME}.tar.gz.part*"
DATASET_RELEASE_TAG = "dataset-20260407"
DATASET_RELEASE_BASE_URL = (
    f"https://github.com/wangzekun6/paper_search_app/releases/download/{DATASET_RELEASE_TAG}"
)
DATASET_ARCHIVE_CANDIDATES = (DATASET_TAR_PATH, DATASET_TAR_GZ_PATH)

# 系统输出目录统一收纳运行时数据库、缓存、提示词和评估结果。
SYSTEM_OUTPUT_DIR = PROJECT_ROOT / "system_outputs"
RUNTIME_DIR = SYSTEM_OUTPUT_DIR / "runtime"
CACHE_DIR = SYSTEM_OUTPUT_DIR / "cache"
SEMANTIC_CARD_CACHE_DIR = CACHE_DIR / "semantic_cards"
INTENT_SESSION_CACHE_DIR = CACHE_DIR / "intent_sessions"
QUERY_MATCH_CACHE_DIR = CACHE_DIR / "query_matches"
PROMPTS_DIR = SYSTEM_OUTPUT_DIR / "prompts"
EVAL_DIR = SYSTEM_OUTPUT_DIR / "eval"
DEMOS_DIR = SYSTEM_OUTPUT_DIR / "demos"
SEMANTIC_BACKFILL_STATE_PATH = RUNTIME_DIR / "semantic_backfill_state.json"
SEMANTIC_BACKFILL_LOG_PATH = RUNTIME_DIR / "semantic_backfill.log"

SYSTEM_DB_PATH = RUNTIME_DIR / "papercompass.db"
APP_STATE_PATH = RUNTIME_DIR / "app_state.json"

SMOKE_QUERY_JSON_PATH = EVAL_DIR / "smoke_queries.json"
SMOKE_QUERY_TEXT_PATH = EVAL_DIR / "smoke_queries.txt"
BUILD_FEEDBACK_PATH = EVAL_DIR / "build_feedback.txt"
SEMANTIC_CARD_SAMPLE_PATH = EVAL_DIR / "semantic_cards_sample.json"
SEMANTIC_CARD_QUALITY_JSON_PATH = EVAL_DIR / "semantic_card_quality_check.json"
SEMANTIC_CARD_QUALITY_CSV_PATH = EVAL_DIR / "semantic_card_quality_check.csv"
SEMANTIC_CARD_STABILITY_PATH = EVAL_DIR / "semantic_card_field_stability.json"
SEMANTIC_CARD_ERRORS_PATH = EVAL_DIR / "semantic_card_errors.json"
INTENT_EVAL_PATH = EVAL_DIR / "intent_eval.json"
INTENT_ERRORS_PATH = EVAL_DIR / "intent_frame_errors.json"
RANKING_EVAL_PATH = EVAL_DIR / "ranking_eval.json"
GAP_REPORTS_PATH = EVAL_DIR / "gap_reports.json"
REGRESSION_REPORT_PATH = EVAL_DIR / "regression_report.json"
CHAIN_ERRORS_PATH = EVAL_DIR / "query_paper_match_errors.json"
EXPLANATION_SAMPLES_PATH = EVAL_DIR / "explanation_samples.json"

INTENT_PROMPT_PATH = PROMPTS_DIR / "intent_frame.md"
SEMANTIC_CARD_PROMPT_PATH = PROMPTS_DIR / "semantic_card.md"
QUERY_PAPER_MATCH_PROMPT_PATH = PROMPTS_DIR / "query_paper_match.md"
LEGACY_QUERY_PAPER_MATCH_PROMPT_PATH = PROMPTS_DIR / "ranking_explanation.md"

STANDARD_QUERIES_PATH = DEMOS_DIR / "standard_queries.json"
DEMO_RUNS_PATH = DEMOS_DIR / "demo_runs.json"
DEMO_WALKTHROUGH_PATH = DEMOS_DIR / "demo_walkthrough.md"


# 对外暴露时统一把路径标准化，减少相对路径和大小写差异带来的问题。
def _normalized_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


# 优先检查是否已经有完整压缩包可直接解压。
def _archive_candidates() -> List[Path]:
    return [path for path in DATASET_ARCHIVE_CANDIDATES if path.exists()]


# 兼容分片发布的数据集压缩包。
def _split_archive_candidates() -> List[Path]:
    return sorted(BUNDLED_DATA_DIR.glob(DATASET_TAR_GZ_PART_GLOB))


# 返回所有可能的数据集归档路径，供提示信息或检查逻辑使用。
def dataset_archive_paths() -> List[Path]:
    return list(DATASET_ARCHIVE_CANDIDATES) + _split_archive_candidates()


# 返回相对项目根目录的归档路径，便于输出给用户。
def dataset_archive_relative_paths() -> List[str]:
    relative_paths = [path.relative_to(PROJECT_ROOT).as_posix() for path in DATASET_ARCHIVE_CANDIDATES]
    relative_paths.append(f"{DATASET_TAR_GZ_PART_PREFIX.relative_to(PROJECT_ROOT).as_posix()}*")
    return relative_paths


# 实际可用的归档文件优先返回本地已存在的路径。
def available_dataset_archive_paths() -> List[Path]:
    archives = _archive_candidates()
    if archives:
        return archives
    return _split_archive_candidates()


# 解压前先验证每个成员路径，避免 tar 包越界写入。
def _safe_extract_tar(archive_path: Path, destination_dir: Path) -> None:
    destination_root = destination_dir.resolve(strict=False)
    destination_root_str = str(destination_root)

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_name = member.name.strip()
            if not member_name:
                continue
            if member.islnk() or member.issym():
                raise ValueError(f"Refusing to extract link entry from archive: {member.name}")

            target_path = (destination_root / member_name).resolve(strict=False)
            if os.path.commonpath([destination_root_str, str(target_path)]) != destination_root_str:
                raise ValueError(f"Archive entry escapes extraction directory: {member.name}")

        tar.extractall(path=destination_root)


# 处理 GitHub Release 分片压缩包，先合并再解压。
def _extract_split_archive(split_parts: List[Path], destination_dir: Path) -> None:
    if not split_parts:
        raise FileNotFoundError("No split dataset archive parts were provided for extraction.")

    with tempfile.TemporaryDirectory(prefix=f"{DATASET_NAME}_", suffix="_extract") as temp_dir:
        merged_archive = Path(temp_dir) / f"{DATASET_NAME}.tar.gz"
        with merged_archive.open("wb") as destination_handle:
            for part_path in split_parts:
                with part_path.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
        _safe_extract_tar(merged_archive, destination_dir)


# 分片文件名统一通过这里生成，便于下载和本地查找保持一致。
def _release_archive_part_name(index: int) -> str:
    return f"{DATASET_NAME}.tar.gz.part{index:02d}"


# 当本地没有数据集时，从 GitHub Release 下载分片归档。
def _download_release_split_archives() -> List[Path]:
    BUNDLED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_parts: List[Path] = []

    for index in range(1, 100):
        part_name = _release_archive_part_name(index)
        part_path = BUNDLED_DATA_DIR / part_name
        if part_path.exists():
            downloaded_parts.append(part_path)
            continue

        with tempfile.NamedTemporaryFile(
            prefix=f"{DATASET_NAME}_{index:02d}_",
            suffix=".download",
            delete=False,
        ) as temp_handle:
            temp_path = Path(temp_handle.name)

        request = Request(
            f"{DATASET_RELEASE_BASE_URL}/{part_name}",
            headers={"User-Agent": "PaperCompass dataset bootstrap"},
        )
        try:
            with urlopen(request) as response, temp_path.open("wb") as destination_handle:
                shutil.copyfileobj(response, destination_handle)
            temp_path.replace(part_path)
            downloaded_parts.append(part_path)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            status_code = getattr(exc, "code", None)
            if status_code == 404 and downloaded_parts:
                break
            if status_code == 404:
                raise FileNotFoundError(
                    f"Dataset release assets not found under: {DATASET_RELEASE_BASE_URL}"
                ) from exc
            raise

    if not downloaded_parts:
        raise FileNotFoundError(f"No dataset release assets found under: {DATASET_RELEASE_BASE_URL}")
    return downloaded_parts


# 确保默认数据集可用；如果缺失则尝试从本地归档或远端分片恢复。
def ensure_default_dataset_available() -> Path:
    if DATASET_DIR.exists():
        return DATASET_DIR.resolve()

    archives = _archive_candidates()
    split_archives = _split_archive_candidates()
    if not archives and not split_archives:
        try:
            split_archives = _download_release_split_archives()
        except Exception as exc:
            archive_list = ", ".join(dataset_archive_relative_paths())
            raise FileNotFoundError(
                f"Dataset directory not found: {DATASET_DIR}. "
                f"Expected a local archive at one of: {archive_list}, "
                f"or downloadable release assets under: {DATASET_RELEASE_BASE_URL}."
            ) from exc
        archives = _archive_candidates()
        split_archives = _split_archive_candidates()

    DATASET_PARENT_DIR.mkdir(parents=True, exist_ok=True)
    if archives:
        _safe_extract_tar(archives[0], DATASET_PARENT_DIR)
    else:
        _extract_split_archive(split_archives, DATASET_PARENT_DIR)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Bundled dataset archive was extracted, but dataset directory is still missing: {DATASET_DIR}"
        )
    return DATASET_DIR.resolve()


# 解析数据集根目录；默认会确保打包数据集已经就绪。
def resolve_dataset_root(data_root: str | Path = DATASET_DIR) -> Path:
    requested = _normalized_path(data_root)
    default_dataset = _normalized_path(DATASET_DIR)
    if requested.exists():
        return requested
    if requested == default_dataset:
        return ensure_default_dataset_available()
    raise FileNotFoundError(f"Dataset path not found: {requested}")


# 统一初始化运行目录和缓存目录。
def ensure_system_layout() -> None:
    for directory in (
        SYSTEM_OUTPUT_DIR,
        RUNTIME_DIR,
        CACHE_DIR,
        SEMANTIC_CARD_CACHE_DIR,
        INTENT_SESSION_CACHE_DIR,
        QUERY_MATCH_CACHE_DIR,
        PROMPTS_DIR,
        EVAL_DIR,
        DEMOS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if not APP_STATE_PATH.exists():
        write_json(APP_STATE_PATH, {})


# 把绝对路径转成相对项目根目录的展示形式。
def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# 读 JSON 时提供安全兜底，避免单个文件损坏影响整个流程。
def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# 写 JSON 时统一保证父目录存在并使用 UTF-8 编码。
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 轻量运行态信息统一写入 app_state，便于前端恢复状态。
def merge_app_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    ensure_system_layout()
    state = read_json(APP_STATE_PATH, {})
    state.update(patch)
    write_json(APP_STATE_PATH, state)
    return state


# 当前正在使用的运行期数据库路径统一从 app_state 推断。
def get_active_runtime_db_path() -> Path:
    ensure_system_layout()
    state = read_json(APP_STATE_PATH, {})
    runtime_db_path = state.get("runtime_db_path")
    if runtime_db_path:
        return Path(str(runtime_db_path))
    return SYSTEM_DB_PATH


# 重建数据库时生成带时间戳的新路径，避免覆盖正在使用的旧库。
def create_versioned_runtime_db_path(prefix: str = "papercompass") -> Path:
    ensure_system_layout()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = RUNTIME_DIR / f"{prefix}_{stamp}.db"
    counter = 1
    while candidate.exists():
        candidate = RUNTIME_DIR / f"{prefix}_{stamp}_{counter}.db"
        counter += 1
    return candidate


# 清理旧版本运行库，避免 runtime 目录无限膨胀。
def cleanup_runtime_databases(keep_latest: int = 2, protected_paths: List[Path] | None = None) -> List[str]:
    ensure_system_layout()
    protected = {path.resolve(strict=False) for path in (protected_paths or [])}
    protected.add(SYSTEM_DB_PATH.resolve(strict=False))
    protected.add(get_active_runtime_db_path().resolve(strict=False))

    versioned_paths = sorted(
        RUNTIME_DIR.glob("papercompass_*.db"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    keep_paths = {
        path.resolve(strict=False)
        for path in versioned_paths[: max(keep_latest, 0)]
    }
    keep_paths.update(protected)

    deleted: List[str] = []
    for path in versioned_paths:
        resolved = path.resolve(strict=False)
        if resolved in keep_paths:
            continue
        try:
            path.unlink()
            deleted.append(path.name)
        except OSError:
            continue
    return deleted


# 语义卡片缓存按 paper_id 单独落盘。
def semantic_card_cache_path(paper_id: str) -> Path:
    return SEMANTIC_CARD_CACHE_DIR / f"{paper_id}.json"


# 搜索历史对应的意图缓存路径。
def intent_session_cache_path(history_id: int) -> Path:
    return INTENT_SESSION_CACHE_DIR / f"{history_id}.json"


# 基于查询文本和上下文生成稳定的意图缓存键。
def intent_query_cache_path(user_text: str, prior_frame: Dict[str, Any] | None = None, mode: str = "initial") -> Path:
    digest = hashlib.sha1(
        json.dumps(
            {
                "mode": mode,
                "user_text": user_text,
                "prior_frame": prior_frame or {},
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return INTENT_SESSION_CACHE_DIR / f"query_{digest[:16]}.json"


# query-paper 匹配缓存同时绑定意图框架和论文 id。
def query_match_cache_path(intent_frame: Dict[str, Any], paper_id: str) -> Path:
    digest = hashlib.sha1(
        json.dumps(
            {"paper_id": paper_id, "intent_frame": intent_frame},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return QUERY_MATCH_CACHE_DIR / f"{paper_id}_{digest[:16]}.json"
