"""
PaperCompass 项目级服务层。

这个模块把基础契约、检索和语义卡片等实现细节收敛为统一接口，
让前端、CLI 和后续功能扩展都围绕“完整项目”组织，
而不是直接耦合某个阶段脚本。
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import retrieval
from .config import (
    APP_STATE_PATH,
    STANDARD_QUERIES_PATH,
    SYSTEM_DB_PATH,
    build_dataset_source_fingerprint,
    build_dataset_source_info,
    cleanup_runtime_databases,
    create_versioned_runtime_db_path,
    ensure_system_layout,
    get_active_runtime_db_path,
    get_default_dataset_source,
    materialize_dataset_source_for_build,
    merge_app_state,
    read_json,
    relative_to_project,
    resolve_dataset_root,
)


DEFAULT_DATA_ROOT = get_default_dataset_source()
DEFAULT_DB_PATH = SYSTEM_DB_PATH
DEFAULT_QUERY_JSON_PATH = retrieval.DEFAULT_QUERY_JSON_PATH
DEFAULT_QUERY_TEXT_PATH = retrieval.DEFAULT_QUERY_TEXT_PATH
DEFAULT_SEMANTIC_TARGET_COUNT = 100
DEFAULT_SEMANTIC_TARGET_RATIO = 0.60
DEFAULT_PILOT_COUNT = 5
DEFAULT_SEMANTIC_BACKFILL_MODE = "standard"
DEFAULT_CHAIN_TOP_K = 5
DEFAULT_CHAIN_CANDIDATE_POOL_SIZE = 120
DEFAULT_CHAIN_EXPLAIN_LIMIT = 5
SEMANTIC_GENERATED_STATUSES = ("generated", "generated_fallback")
PROJECT_QUERY_RUNTIME_WARMUP_LOCK = threading.Lock()
PROJECT_QUERY_RUNTIME_WARMUP_STATE: Dict[str, Dict[str, Any]] = {}

# 把检索模式映射到具体实现，方便前端和 CLI 统一透传参数。
SEARCH_MODE_TO_FN: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "basic": retrieval.search_basic,
    "exact": retrieval.search_exact_matches,
    "hybrid": retrieval.search_hybrid,
}
STANDARD_QUERIES_CACHE: Dict[str, Any] = {"mtime_ns": None, "value": None}
DEMO_QUERY_SPECS: List[Dict[str, Any]] = [
    {
        "query": "我想了解 retrieval-augmented generation evaluation 方向的重要论文。",
    },
    {
        "query": "我想系统了解 long-context LLMs 的代表性论文。",
    },
    {
        "query": "我想看近几年 speech-to-speech translation 方向的代表性论文。",
    },
    {
        "query": "我想看 translation quality estimation 方向近几年的重要论文。",
    },
    {
        "query": "我想了解 prompt optimization for large language models 的代表性论文。",
    },
    {
        "query": "我想了解 retrieval-augmented generation for medical QA 的代表性论文。",
    },
]


def _chain_module():
    from . import chain

    return chain


def _intent_module():
    from . import intent

    return intent


def _semantic_module():
    from . import semantic

    return semantic


def _semantic_backfill_module():
    from . import semantic_backfill

    return semantic_backfill


# 默认数据库路径统一从运行态配置读取。
def get_default_db_path() -> Path:
    return get_active_runtime_db_path()


# 统一解析项目数据库路径，允许调用方显式覆盖默认值。
def resolve_project_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is None:
        return get_active_runtime_db_path()
    return Path(db_path)


def project_database_exists(db_path: str | Path | None = None) -> bool:
    return retrieval.database_exists(resolve_project_db_path(db_path))


def require_project_database(db_path: str | Path | None = None) -> Path:
    database_path = resolve_project_db_path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(
            f"Project database not found: {database_path}. Run `python papercompass.py build` first."
        )
    return database_path


def _project_runtime_cache_key(database_path: Path) -> str:
    try:
        size = database_path.stat().st_size
    except OSError:
        size = 0
    return f"{database_path.resolve(strict=False)}::{size}"


# 收藏区和结果卡片都复用这一套作者清洗逻辑。
def extract_authors_for_display(authors_raw: str) -> List[str]:
    text = " ".join(str(authors_raw or "").split()).strip()
    if not text:
        return []

    sanitized = text
    sanitized = re.sub(r"\S+@\S+", " | ", sanitized)
    sanitized = re.sub(r"Corresponding Author:?", " | ", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+(?:and|&)\s+", " | ", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"[|;/]+", " | ", sanitized)
    sanitized = re.sub(r"(?<=[a-z])(?=[A-Z])", " | ", sanitized)
    sanitized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " | ", sanitized)
    sanitized = re.sub(r"(?<=\d)(?=[A-Za-z\u00C0-\u024F\u4E00-\u9FFF])", " | ", sanitized)

    authors: List[str] = []
    seen = set()
    for chunk in sanitized.split("|"):
        for part in re.split(r",(?=\s*[A-Z\u00C0-\u024F\u4E00-\u9FFF])", chunk):
            candidate = re.sub(r"[\*\u2020\u2021]+", "", part)
            candidate = " ".join(re.sub(r"\d+", "", candidate).split()).strip(" ,")
            if not candidate:
                continue
            lowered = candidate.lower().strip(".")
            if lowered in {"and", "&", "et al", "corresponding author"}:
                continue
            if lowered.startswith("corresponding author"):
                continue
            if candidate not in seen:
                seen.add(candidate)
                authors.append(candidate)

    if authors:
        return authors
    return [text]


# 在界面上压缩显示作者列表，避免标题区过长。
def format_authors_for_display(authors_raw: str, max_names: int = 4) -> str:
    authors = extract_authors_for_display(authors_raw)
    if not authors:
        return "-"
    if len(authors) <= max_names:
        return " · ".join(authors)
    return f"{' · '.join(authors[:max_names])} · +{len(authors) - max_names}"


# 统计语义卡片各状态分布，便于查看生成进度。
def _semantic_status_counts(conn: Any) -> Dict[str, int]:
    rows = conn.execute(
        """
        SELECT card_status, COUNT(*) AS count
        FROM paper_semantic_cards
        GROUP BY card_status
        ORDER BY card_status
        """
    ).fetchall()
    return {row["card_status"]: row["count"] for row in rows}


# 汇总项目级统计信息，给状态面板和 CLI 复用。
def load_project_stats(db_path: str | Path | None = None) -> Dict[str, int]:
    database_path = resolve_project_db_path(db_path)
    stats = retrieval.load_database_stats(database_path)
    stats["semantic_cards"] = 0
    stats["intent_histories"] = 0
    stats["saved_papers"] = 0

    if not database_path.exists():
        return stats

    with retrieval.connect_db(database_path) as conn:
        status_placeholders = ", ".join("?" for _ in SEMANTIC_GENERATED_STATUSES)
        stats["semantic_cards"] = int(
            conn.execute(
                f"SELECT COUNT(*) FROM paper_semantic_cards WHERE card_status IN ({status_placeholders})",
                SEMANTIC_GENERATED_STATUSES,
            ).fetchone()[0]
            or 0
        )
        stats["intent_histories"] = int(conn.execute("SELECT COUNT(*) FROM search_history").fetchone()[0])
        stats["saved_papers"] = int(conn.execute("SELECT COUNT(*) FROM saved_papers").fetchone()[0])
    return stats


def _run_project_query_runtime_warmup(database_path: Path, cache_key: str) -> None:
    summary: Dict[str, Any] = {
        "db_path": str(database_path),
        "status": "running",
        "started_at": time.time(),
    }
    try:
        started_at = time.perf_counter()
        query_runtime = retrieval.ensure_query_runtime_artifacts(database_path, build_exact_cache=False)
        summary.update(query_runtime)
        summary["elapsed"] = round(time.perf_counter() - started_at, 4)
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
    with PROJECT_QUERY_RUNTIME_WARMUP_LOCK:
        PROJECT_QUERY_RUNTIME_WARMUP_STATE[cache_key] = summary


def start_project_query_runtime_warmup(db_path: str | Path | None = None) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    cache_key = _project_runtime_cache_key(database_path)
    with PROJECT_QUERY_RUNTIME_WARMUP_LOCK:
        cached = PROJECT_QUERY_RUNTIME_WARMUP_STATE.get(cache_key)
        if isinstance(cached, dict) and cached.get("status") in {"running", "completed"}:
            return dict(cached)
        PROJECT_QUERY_RUNTIME_WARMUP_STATE[cache_key] = {
            "db_path": str(database_path),
            "status": "running",
        }
    thread = threading.Thread(
        target=_run_project_query_runtime_warmup,
        args=(database_path, cache_key),
        daemon=True,
        name="papercompass-query-runtime-warmup",
    )
    thread.start()
    return {"db_path": str(database_path), "status": "running"}


# 从磁盘缓存恢复语义卡片到数据库。
def restore_semantic_card_cache(
    db_path: str | Path | None = None,
    *,
    refresh: bool = False,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    semantic = _semantic_module()
    return semantic.restore_cached_semantic_cards(database_path, refresh=refresh)


# 启动后台语义卡片补全任务。
def start_semantic_backfill(
    db_path: str | Path | None = None,
    *,
    mode: str = DEFAULT_SEMANTIC_BACKFILL_MODE,
    refresh: bool = False,
    target_count: int | None = None,
    target_ratio: float | None = None,
) -> Dict[str, Any]:
    if target_count is None and target_ratio is None:
        target_ratio = DEFAULT_SEMANTIC_TARGET_RATIO
    database_path = require_project_database(db_path)
    semantic_backfill = _semantic_backfill_module()
    return semantic_backfill.start_background_semantic_backfill(
        database_path,
        mode=mode,
        refresh=refresh,
        target_count=target_count,
        target_ratio=target_ratio,
    )


def get_semantic_backfill_status() -> Dict[str, Any]:
    semantic_backfill = _semantic_backfill_module()
    return semantic_backfill.get_semantic_backfill_status()


# 对外暴露统一的项目检索入口。
def search_project(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    if mode not in SEARCH_MODE_TO_FN:
        raise ValueError(f"Unsupported search mode: {mode}")
    database_path = require_project_database(db_path)
    return SEARCH_MODE_TO_FN[mode](query, top_k=top_k, db_path=database_path)


# 生成语义层相关演示、样例和评估产物。
def build_semantic_assets(
    db_path: str | Path | None = None,
    target_count: Optional[int] = None,
    target_ratio: Optional[float] = None,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    refresh: bool = False,
) -> Dict[str, Any]:
    if target_count is None and target_ratio is None:
        target_ratio = DEFAULT_SEMANTIC_TARGET_RATIO
    database_path = require_project_database(db_path)
    chain = _chain_module()
    semantic = _semantic_module()
    semantic.ensure_output_dir()
    semantic.write_prompt_file()
    openai_available = semantic.can_use_openai()
    openai_message = semantic.OPENAI_RUNTIME_MESSAGE
    cache_restore = semantic.restore_cached_semantic_cards(database_path, refresh=False)

    with retrieval.connect_db(database_path) as conn:
        total_papers = semantic.total_paper_count(conn)
        resolved_target_count = semantic.resolve_target_count(
            total_papers,
            target_count=target_count,
            target_ratio=target_ratio,
        )
        if pilot_count > 0:
            pilot_ids = semantic.select_candidate_paper_ids(conn, max(pilot_count, 5))[:pilot_count]
            pilot_cards = semantic.generate_pilot_cards(conn, pilot_ids, refresh=refresh)
        else:
            pilot_cards = []

        semantic.dump_json(semantic.PILOT_OUTPUT_PATH, pilot_cards)

        generated_count = semantic.generate_cards_until_target(
            conn,
            resolved_target_count,
            refresh=refresh,
            allow_heuristic_fallback=True,
        )
        generated_cards = semantic.load_generated_cards(conn, limit=resolved_target_count)
        semantic.dump_json(semantic.SAMPLE_OUTPUT_PATH, generated_cards)

        quality_payload = semantic.write_quality_check(conn, sample_size=semantic.DEFAULT_QUALITY_SAMPLE_SIZE)
        stability_payload = semantic.write_field_stability_report(semantic.load_generated_cards(conn))
        status_counts = _semantic_status_counts(conn)

    standard_query_prewarm = chain.prewarm_semantic_cards_for_standard_queries(
        database_path,
        chain.STANDARD_QUERY_SPECS,
        candidate_pool_size=chain.DEFAULT_CANDIDATE_POOL_SIZE,
    )

    semantic.write_cache_strategy()
    semantic.write_feedback(
        db_path=database_path,
        generated_count=generated_count,
        pilot_cards=pilot_cards,
        quality_payload=quality_payload,
        stability_payload=stability_payload,
        status_counts=status_counts,
        openai_available=openai_available,
        openai_message=openai_message,
    )
    return {
        "db_path": str(database_path),
        "target_count_requested": target_count,
        "target_ratio_requested": target_ratio,
        "resolved_target_count": resolved_target_count,
        "generated_count": generated_count,
        "pilot_cards": len(pilot_cards),
        "quality_sample_size": quality_payload.get("sample_size", 0),
        "status_counts": status_counts,
        "openai_available": openai_available,
        "openai_message": openai_message,
        "cache_restore": cache_restore,
        "standard_query_semantic_prewarm": standard_query_prewarm,
    }


# 构建完整项目：建库、调试查询、可选语义层和后台补全。
def build_project(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    db_path: str | Path | None = None,
    top_k: int = 10,
    generate_query_debug: bool = False,
    build_semantic_layer: bool = False,
    semantic_target_count: int | None = None,
    semantic_target_ratio: float | None = None,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    refresh_semantic_cards: bool = False,
    auto_semantic_backfill: bool = True,
    semantic_backfill_mode: str = DEFAULT_SEMANTIC_BACKFILL_MODE,
    store_raw_json: bool = False,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    if semantic_target_count is None and semantic_target_ratio is None:
        semantic_target_ratio = DEFAULT_SEMANTIC_TARGET_RATIO
    ensure_system_layout()
    data_root_path = resolve_dataset_root(data_root)
    source_fingerprint = build_dataset_source_fingerprint(data_root_path)
    materialized_data_root = data_root_path
    dataset_source = build_dataset_source_info(data_root_path)
    requested_db_path = resolve_project_db_path(db_path)
    active_db_path = get_active_runtime_db_path()
    database_path = requested_db_path
    if active_db_path.exists() and requested_db_path.resolve(strict=False) == active_db_path.resolve(strict=False):
        database_path = create_versioned_runtime_db_path()
    query_payload: Optional[Dict[str, Any]] = None
    reused_existing_build = False

    if requested_db_path.exists() and not retrieval.load_build_runtime_metadata(requested_db_path):
        state = read_json(APP_STATE_PATH, {})
        last_build = state.get("last_build") if isinstance(state, dict) else None
        last_build_db_path = Path(str(last_build.get("db_path"))) if isinstance(last_build, dict) and last_build.get("db_path") else None
        last_build_data_root = (
            Path(str(last_build.get("data_root")))
            if isinstance(last_build, dict) and last_build.get("data_root")
            else None
        )
        if (
            last_build_db_path is not None
            and last_build_data_root is not None
            and last_build_db_path.resolve(strict=False) == requested_db_path.resolve(strict=False)
            and last_build_data_root.resolve(strict=False) == data_root_path.resolve(strict=False)
        ):
            with retrieval.connect_db(requested_db_path) as conn:
                retrieval.write_build_runtime_metadata(
                    conn,
                    source_fingerprint=source_fingerprint,
                    store_raw_json=True,
                )
                conn.commit()

    if not force_rebuild and retrieval.database_matches_build(
        requested_db_path,
        source_fingerprint=source_fingerprint,
        store_raw_json=store_raw_json,
    ):
        database_path = requested_db_path
        database_stats = retrieval.load_database_stats(database_path)
        reused_existing_build = True
    else:
        materialized_data_root = materialize_dataset_source_for_build(data_root_path)
        dataset_source["materialized_path"] = str(materialized_data_root.resolve(strict=False))
        dataset_source["materialized_display_path"] = relative_to_project(materialized_data_root)
        dataset_source["materialized_kind"] = "directory" if materialized_data_root.is_dir() else dataset_source["kind"]
        dataset_source["materialized_from_cache"] = (
            materialized_data_root.resolve(strict=False) != data_root_path.resolve(strict=False)
        )
        database_stats = retrieval.build_database(
            materialized_data_root,
            database_path,
            store_raw_json=store_raw_json,
            source_fingerprint=source_fingerprint,
        )
        if active_db_path.exists() and active_db_path.resolve(strict=False) != database_path.resolve(strict=False):
            runtime_backup = retrieval.backup_runtime_tables(active_db_path)
            with retrieval.connect_db(database_path) as conn:
                retrieval.restore_runtime_tables(conn, runtime_backup)
    if "materialized_path" not in dataset_source:
        dataset_source["materialized_path"] = str(materialized_data_root.resolve(strict=False))
        dataset_source["materialized_display_path"] = relative_to_project(materialized_data_root)
        dataset_source["materialized_kind"] = "directory" if materialized_data_root.is_dir() else dataset_source["kind"]
        dataset_source["materialized_from_cache"] = (
            materialized_data_root.resolve(strict=False) != data_root_path.resolve(strict=False)
        )
    semantic_cache_restore: Dict[str, Any] = {
        "db_path": str(database_path),
        "skipped": True,
        "reason": "semantic-build-not-requested",
    }
    if build_semantic_layer or auto_semantic_backfill:
        semantic = _semantic_module()
        semantic_cache_restore = semantic.restore_cached_semantic_cards(database_path, refresh=False)
    if generate_query_debug:
        query_payload = retrieval.run_debug_queries(
            db_path=database_path,
            query_json_path=DEFAULT_QUERY_JSON_PATH,
            query_text_path=DEFAULT_QUERY_TEXT_PATH,
            top_k=top_k,
        )

    query_index_warm_elapsed = None
    query_index_warm_error = ""
    query_index_warm_summary = None
    try:
        warm_started_at = time.perf_counter()
        query_index_warm_summary = retrieval.ensure_query_runtime_artifacts(database_path, build_exact_cache=True)
        query_index_warm_summary["dense_ready"] = bool(query_index_warm_summary.get("hot_fts_ready"))
        query_index_warm_summary["dense_backend"] = "hot_fts_multi_query"
        query_index_warm_summary["exact_backend"] = "hot_fts_candidate_rescore"
        query_index_warm_elapsed = round(time.perf_counter() - warm_started_at, 4)
    except Exception as exc:
        query_index_warm_error = str(exc)

    semantic_summary = None
    if build_semantic_layer:
        semantic_summary = build_semantic_assets(
            db_path=database_path,
            target_count=semantic_target_count,
            target_ratio=semantic_target_ratio,
            pilot_count=pilot_count,
            refresh=refresh_semantic_cards,
        )

    semantic_backfill_summary = None
    if auto_semantic_backfill:
        semantic_backfill = _semantic_backfill_module()
        semantic_backfill_summary = semantic_backfill.start_background_semantic_backfill(
            database_path,
            mode=semantic_backfill_mode,
            refresh=refresh_semantic_cards,
            target_count=semantic_target_count,
            target_ratio=semantic_target_ratio,
        )

    payload = {
        "data_root": str(data_root_path.resolve()),
        "materialized_data_root": str(materialized_data_root.resolve(strict=False)),
        "source_fingerprint": source_fingerprint,
        "dataset_source": dataset_source,
        "db_path": str(database_path.resolve()),
        "database_stats": database_stats,
        "reused_existing_build": reused_existing_build,
        "force_rebuild": bool(force_rebuild),
        "store_raw_json": bool(store_raw_json),
        "semantic_cache_restore": semantic_cache_restore,
        "query_debug_enabled": generate_query_debug,
        "query_count": query_payload["query_count"] if query_payload else 0,
        "query_index_warmed": query_index_warm_error == "",
        "query_index_warm_summary": query_index_warm_summary,
        "query_index_warm_elapsed": query_index_warm_elapsed,
        "query_index_warm_error": query_index_warm_error,
        "semantic_layer_enabled": build_semantic_layer,
        "semantic_layer": semantic_summary,
        "semantic_backfill": semantic_backfill_summary,
    }
    merge_app_state(
        {
            "last_build": payload,
            "runtime_db_path": str(database_path.resolve()),
            "dataset_source": dataset_source,
        }
    )
    deleted_runtime_dbs = cleanup_runtime_databases(
        keep_latest=2,
        protected_paths=[database_path, active_db_path],
    )
    if deleted_runtime_dbs:
        payload["deleted_old_runtime_dbs"] = deleted_runtime_dbs
        merge_app_state({"last_build": payload})
    return payload


# 单篇论文语义卡片生成接口，供 CLI 或调试场景调用。
def generate_semantic_card_for_paper(
    paper_id: str,
    db_path: str | Path | None = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    semantic = _semantic_module()
    semantic.ensure_output_dir()
    semantic.write_prompt_file()

    with retrieval.connect_db(database_path) as conn:
        card, used_model = semantic.generate_card_for_paper(conn, paper_id, refresh=refresh)

    if card is None:
        raise semantic.OpenAIAPIError(f"Failed to generate semantic card for {paper_id}")

    return {
        "db_path": str(database_path),
        "paper_id": paper_id,
        "used_model": used_model,
        "card": card,
    }


# 查询意图解析入口，支持首轮解析和 follow-up 合并。
def analyze_query_intent(
    user_text: str,
    db_path: str | Path | None = None,
    history_id: Optional[int] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    database_path = resolve_project_db_path(db_path)
    intent = _intent_module()
    prior_frame = None
    prior_history_id = history_id
    if history_id is not None:
        require_project_database(database_path)
        prior_frame = intent.load_intent_frame(database_path, history_id)
        frame, used_model, parser = intent.merge_follow_up_reply(prior_frame, user_text)
    else:
        frame, used_model, parser = intent.parse_intent_frame(user_text)

    new_history_id = None
    if persist:
        require_project_database(database_path)
        new_history_id = intent.save_intent_frame(database_path, user_text, frame)

    payload = {
        "db_path": str(database_path),
        "history_id": new_history_id,
        "prior_history_id": prior_history_id,
        "parser": parser,
        "used_model": used_model,
        "pipeline_mode": {"intent_analysis": "llm_required"},
        "intent_frame": frame,
    }
    merge_app_state({"last_intent_query": user_text, "last_intent_history_id": new_history_id})
    return payload


# 生成意图模块的提示词与评估资产。
def build_intent_assets(
    db_path: str | Path | None = None,
    queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    database_path = resolve_project_db_path(db_path)
    intent = _intent_module()
    if database_path.exists():
        require_project_database(database_path)
    summary = intent.build_intent_assets(queries=queries)
    if database_path.exists():
        summary["intent_histories"] = intent.load_search_history_count(database_path)
    return summary


# 执行一次完整主链路，但不负责写入历史。
def run_project_chain(
    query: str,
    db_path: str | Path | None = None,
    follow_up_reply: Optional[str] = None,
    top_k: int = DEFAULT_CHAIN_TOP_K,
    candidate_pool_size: int = DEFAULT_CHAIN_CANDIDATE_POOL_SIZE,
    explain_limit: int = DEFAULT_CHAIN_EXPLAIN_LIMIT,
    stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    chain = _chain_module()
    payload = chain.run_core_chain(
        query=query,
        follow_up_reply=follow_up_reply,
        db_path=database_path,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explain_limit=explain_limit,
        stage_callback=stage_callback,
    )
    merge_app_state({"last_chain_query": query, "last_chain_result_count": len(payload.get("top_k_results", []))})
    return payload


# 在主链路之上补充历史落库，形成用户会话级接口。
def run_project_chain_session(
    query: str,
    db_path: str | Path | None = None,
    follow_up_reply: Optional[str] = None,
    top_k: int = DEFAULT_CHAIN_TOP_K,
    candidate_pool_size: int = DEFAULT_CHAIN_CANDIDATE_POOL_SIZE,
    explain_limit: int = DEFAULT_CHAIN_EXPLAIN_LIMIT,
    persist_history: bool = True,
    stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    payload = run_project_chain(
        query=query,
        db_path=db_path,
        follow_up_reply=follow_up_reply,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explain_limit=explain_limit,
        stage_callback=stage_callback,
    )
    history_id = None
    if persist_history:
        database_path = require_project_database(db_path)
        history_text = query if not follow_up_reply else f"{query}\nFollow-up: {follow_up_reply}"
        intent = _intent_module()
        history_id = intent.save_intent_frame(database_path, history_text, payload["final_intent_frame"])
    payload["history_id"] = history_id
    return payload


# 最近查询历史列表，供前端管理区和 CLI 使用。
def list_search_history(db_path: str | Path | None = None, limit: int = 20) -> List[Dict[str, Any]]:
    database_path = require_project_database(db_path)
    with retrieval.connect_db(database_path) as conn:
        rows = conn.execute(
            """
            SELECT id, query_text, intent_frame_json, created_at
            FROM search_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "query_text": row["query_text"],
            "created_at": row["created_at"],
            "intent_frame": json.loads(row["intent_frame_json"]),
        }
        for row in rows
    ]


# 收藏/取消收藏接口直接操作数据库表。
def save_paper(paper_id: str, db_path: str | Path | None = None) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    with retrieval.connect_db(database_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO saved_papers (paper_id)
            VALUES (?)
            """,
            (paper_id,),
        )
        conn.commit()
    merge_app_state({"last_saved_paper": paper_id})
    return {"paper_id": paper_id, "saved": True}


def unsave_paper(paper_id: str, db_path: str | Path | None = None) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    with retrieval.connect_db(database_path) as conn:
        conn.execute("DELETE FROM saved_papers WHERE paper_id = ?", (paper_id,))
        conn.commit()
    return {"paper_id": paper_id, "saved": False}


# 读取收藏列表时顺带补齐展示友好的作者信息。
def list_saved_papers(db_path: str | Path | None = None, limit: int = 50) -> List[Dict[str, Any]]:
    database_path = require_project_database(db_path)
    with retrieval.connect_db(database_path) as conn:
        rows = conn.execute(
            """
            SELECT saved_papers.paper_id, saved_papers.saved_at, papers.title, papers.authors_raw, papers.year_month
            FROM saved_papers
            JOIN papers ON papers.paper_id = saved_papers.paper_id
            ORDER BY saved_papers.saved_at DESC, saved_papers.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "authors_raw": row["authors_raw"],
            "authors": extract_authors_for_display(row["authors_raw"]),
            "authors_display": format_authors_for_display(row["authors_raw"]),
            "year_month": row["year_month"],
            "saved_at": row["saved_at"],
        }
        for row in rows
    ]


def get_saved_paper_ids(db_path: str | Path | None = None) -> List[str]:
    database_path = require_project_database(db_path)
    with retrieval.connect_db(database_path) as conn:
        rows = conn.execute("SELECT paper_id FROM saved_papers ORDER BY id DESC").fetchall()
    return [row["paper_id"] for row in rows]


# 单篇详情接口聚合论文主记录、章节和语义卡片。
def get_paper_detail(paper_id: str, db_path: str | Path | None = None) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    with retrieval.connect_db(database_path) as conn:
        row = conn.execute(
            """
            SELECT papers.*, paper_semantic_cards.semantic_card_json, paper_semantic_cards.card_status
            FROM papers
            LEFT JOIN paper_semantic_cards
                ON papers.paper_id = paper_semantic_cards.paper_id
            WHERE papers.paper_id = ?
            """,
            (paper_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Paper not found: {paper_id}")
        section_rows = retrieval.load_sections_for_papers(conn, [paper_id]).get(paper_id, [])

    semantic_card = json.loads(row["semantic_card_json"]) if row["semantic_card_json"] else None
    sections = [
        {
            "section_title": item["section_title"],
            "section_snippet": item["section_snippet"],
        }
        for item in section_rows
    ]
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "authors_raw": row["authors_raw"],
        "authors": extract_authors_for_display(row["authors_raw"]),
        "authors_display": format_authors_for_display(row["authors_raw"], max_names=8),
        "year_month": row["year_month"],
        "abstract": row["abstract"],
        "section_titles": json.loads(row["section_titles"]),
        "section_count": len(sections),
        "semantic_card": semantic_card,
        "semantic_card_status": row["card_status"] or "",
        "sections": sections,
    }


# 标准查询会做简单缓存，避免前端每次刷新都重复读取磁盘。
def load_standard_queries() -> List[Dict[str, Any]]:
    chain = _chain_module()
    default_value = list(chain.STANDARD_QUERY_SPECS)
    try:
        mtime_ns = STANDARD_QUERIES_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        STANDARD_QUERIES_CACHE["mtime_ns"] = None
        STANDARD_QUERIES_CACHE["value"] = default_value
        return copy.deepcopy(default_value)

    cached_mtime = STANDARD_QUERIES_CACHE.get("mtime_ns")
    cached_value = STANDARD_QUERIES_CACHE.get("value")
    if cached_mtime == mtime_ns and isinstance(cached_value, list):
        return copy.deepcopy(cached_value)

    loaded_value = read_json(STANDARD_QUERIES_PATH, default_value)
    if not isinstance(loaded_value, list):
        loaded_value = default_value
    STANDARD_QUERIES_CACHE["mtime_ns"] = mtime_ns
    STANDARD_QUERIES_CACHE["value"] = loaded_value
    return copy.deepcopy(loaded_value)


def load_demo_queries() -> List[Dict[str, Any]]:
    demo_queries = load_standard_queries()
    sanitized: List[Dict[str, Any]] = []
    for item in demo_queries:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        payload.pop("follow_up_reply", None)
        sanitized.append(payload)
    return sanitized


# app_state 负责记录运行期轻量状态，例如最近一次查询信息。
def load_app_state() -> Dict[str, Any]:
    ensure_system_layout()
    return read_json(APP_STATE_PATH, {})


# 生成主链路演示和评估产物。
def build_chain_assets(
    db_path: str | Path | None = None,
    top_k: int = DEFAULT_CHAIN_TOP_K,
    candidate_pool_size: int = DEFAULT_CHAIN_CANDIDATE_POOL_SIZE,
    explain_limit: int = DEFAULT_CHAIN_EXPLAIN_LIMIT,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    chain = _chain_module()
    return chain.build_core_chain_assets(
        db_path=database_path,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explain_limit=explain_limit,
    )


# 统一格式化状态信息，便于 CLI 直接输出。
def format_status_block(stats: Dict[str, int], db_path: str | Path | None = None) -> str:
    database_path = resolve_project_db_path(db_path)
    semantic = _semantic_module()
    semantic_backfill_status = get_semantic_backfill_status()
    payload = {
        "db_path": str(database_path),
        "papers": stats.get("papers", 0),
        "sections": stats.get("sections", 0),
        "fts_rows": stats.get("fts_rows", 0),
        "semantic_cards": stats.get("semantic_cards", 0),
        "intent_histories": stats.get("intent_histories", 0),
        "saved_papers": stats.get("saved_papers", 0),
        "semantic_card_cache_dir": str(semantic.SEMANTIC_CARD_CACHE_DIR),
        "semantic_backfill": semantic_backfill_status,
        "app_state_path": str(APP_STATE_PATH),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
