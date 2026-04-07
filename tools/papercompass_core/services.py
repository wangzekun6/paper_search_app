"""
PaperCompass 项目级服务层。

这个模块把基础契约、检索和语义卡片等实现细节收敛为统一接口，
让前端、CLI 和后续功能扩展都围绕“完整项目”组织，
而不是直接耦合某个阶段脚本。
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import chain, intent, retrieval, semantic
from .config import (
    APP_STATE_PATH,
    DATASET_DIR,
    STANDARD_QUERIES_PATH,
    SYSTEM_DB_PATH,
    cleanup_runtime_databases,
    create_versioned_runtime_db_path,
    ensure_system_layout,
    get_active_runtime_db_path,
    merge_app_state,
    read_json,
    resolve_dataset_root,
)


DEFAULT_DATA_ROOT = DATASET_DIR
DEFAULT_DB_PATH = SYSTEM_DB_PATH
DEFAULT_QUERY_JSON_PATH = retrieval.DEFAULT_QUERY_JSON_PATH
DEFAULT_QUERY_TEXT_PATH = retrieval.DEFAULT_QUERY_TEXT_PATH
DEFAULT_SEMANTIC_TARGET_COUNT = semantic.DEFAULT_TARGET_COUNT
DEFAULT_PILOT_COUNT = semantic.DEFAULT_PILOT_COUNT

SEARCH_MODE_TO_FN: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "basic": retrieval.search_basic,
    "exact": retrieval.search_exact_matches,
    "hybrid": retrieval.search_hybrid,
}
STANDARD_QUERIES_CACHE: Dict[str, Any] = {"mtime_ns": None, "value": None}


def get_default_db_path() -> Path:
    return get_active_runtime_db_path()


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


def load_project_stats(db_path: str | Path | None = None) -> Dict[str, int]:
    database_path = resolve_project_db_path(db_path)
    stats = retrieval.load_database_stats(database_path)
    stats["semantic_cards"] = 0
    stats["intent_histories"] = 0
    stats["saved_papers"] = 0

    if not database_path.exists():
        return stats

    with retrieval.connect_db(database_path) as conn:
        stats["semantic_cards"] = semantic.current_card_count(conn)
        stats["intent_histories"] = int(conn.execute("SELECT COUNT(*) FROM search_history").fetchone()[0])
        stats["saved_papers"] = int(conn.execute("SELECT COUNT(*) FROM saved_papers").fetchone()[0])
    return stats


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


def build_semantic_assets(
    db_path: str | Path | None = None,
    target_count: int = DEFAULT_SEMANTIC_TARGET_COUNT,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    refresh: bool = False,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    semantic.ensure_output_dir()
    semantic.write_prompt_file()
    openai_available = semantic.can_use_openai()
    openai_message = semantic.OPENAI_RUNTIME_MESSAGE

    with retrieval.connect_db(database_path) as conn:
        if pilot_count > 0:
            pilot_ids = semantic.select_candidate_paper_ids(conn, max(pilot_count, 5))[:pilot_count]
            pilot_cards = semantic.generate_pilot_cards(conn, pilot_ids, refresh=refresh)
        else:
            pilot_cards = []

        semantic.dump_json(semantic.PILOT_OUTPUT_PATH, pilot_cards)

        generated_count = semantic.generate_cards_until_target(conn, target_count, refresh=refresh)
        generated_cards = semantic.load_generated_cards(conn, limit=target_count)
        semantic.dump_json(semantic.SAMPLE_OUTPUT_PATH, generated_cards)

        quality_payload = semantic.write_quality_check(conn, sample_size=semantic.DEFAULT_QUALITY_SAMPLE_SIZE)
        stability_payload = semantic.write_field_stability_report(semantic.load_generated_cards(conn))
        status_counts = _semantic_status_counts(conn)

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
        "generated_count": generated_count,
        "pilot_cards": len(pilot_cards),
        "quality_sample_size": quality_payload.get("sample_size", 0),
        "status_counts": status_counts,
        "openai_available": openai_available,
        "openai_message": openai_message,
    }


def build_project(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    db_path: str | Path | None = None,
    top_k: int = 10,
    generate_query_debug: bool = False,
    build_semantic_layer: bool = False,
    semantic_target_count: int = DEFAULT_SEMANTIC_TARGET_COUNT,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    refresh_semantic_cards: bool = False,
) -> Dict[str, Any]:
    ensure_system_layout()
    data_root_path = resolve_dataset_root(data_root)
    requested_db_path = resolve_project_db_path(db_path)
    active_db_path = get_active_runtime_db_path()
    database_path = requested_db_path
    if active_db_path.exists() and requested_db_path.resolve(strict=False) == active_db_path.resolve(strict=False):
        database_path = create_versioned_runtime_db_path()
    query_payload: Optional[Dict[str, Any]] = None

    database_stats = retrieval.build_database(data_root_path, database_path)
    if active_db_path.exists() and active_db_path.resolve(strict=False) != database_path.resolve(strict=False):
        runtime_backup = retrieval.backup_runtime_tables(active_db_path)
        with retrieval.connect_db(database_path) as conn:
            retrieval.restore_runtime_tables(conn, runtime_backup)
    if generate_query_debug:
        query_payload = retrieval.run_debug_queries(
            db_path=database_path,
            query_json_path=DEFAULT_QUERY_JSON_PATH,
            query_text_path=DEFAULT_QUERY_TEXT_PATH,
            top_k=top_k,
        )

    dense_cache_warm_elapsed = None
    dense_cache_warm_error = ""
    try:
        warm_started_at = time.perf_counter()
        chain.build_dense_index(database_path)
        dense_cache_warm_elapsed = round(time.perf_counter() - warm_started_at, 4)
    except Exception as exc:
        dense_cache_warm_error = str(exc)

    semantic_summary = None
    if build_semantic_layer:
        semantic_summary = build_semantic_assets(
            db_path=database_path,
            target_count=semantic_target_count,
            pilot_count=pilot_count,
            refresh=refresh_semantic_cards,
        )

    payload = {
        "data_root": str(data_root_path.resolve()),
        "db_path": str(database_path.resolve()),
        "database_stats": database_stats,
        "query_debug_enabled": generate_query_debug,
        "query_count": query_payload["query_count"] if query_payload else 0,
        "dense_index_cache_warmed": dense_cache_warm_error == "",
        "dense_index_cache_warm_elapsed": dense_cache_warm_elapsed,
        "dense_index_cache_warm_error": dense_cache_warm_error,
        "semantic_layer_enabled": build_semantic_layer,
        "semantic_layer": semantic_summary,
    }
    merge_app_state({"last_build": payload, "runtime_db_path": str(database_path.resolve())})
    deleted_runtime_dbs = cleanup_runtime_databases(
        keep_latest=2,
        protected_paths=[database_path, active_db_path],
    )
    if deleted_runtime_dbs:
        payload["deleted_old_runtime_dbs"] = deleted_runtime_dbs
        merge_app_state({"last_build": payload})
    return payload


def generate_semantic_card_for_paper(
    paper_id: str,
    db_path: str | Path | None = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
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


def analyze_query_intent(
    user_text: str,
    db_path: str | Path | None = None,
    history_id: Optional[int] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    database_path = resolve_project_db_path(db_path)
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
        "intent_frame": frame,
    }
    merge_app_state({"last_intent_query": user_text, "last_intent_history_id": new_history_id})
    return payload


def build_intent_assets(
    db_path: str | Path | None = None,
    queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    database_path = resolve_project_db_path(db_path)
    if database_path.exists():
        require_project_database(database_path)
    summary = intent.build_intent_assets(queries=queries)
    if database_path.exists():
        summary["intent_histories"] = intent.load_search_history_count(database_path)
    return summary


def run_project_chain(
    query: str,
    db_path: str | Path | None = None,
    follow_up_reply: Optional[str] = None,
    top_k: int = chain.DEFAULT_TOP_K,
    candidate_pool_size: int = chain.DEFAULT_CANDIDATE_POOL_SIZE,
    explain_limit: int = chain.DEFAULT_EXPLAIN_LIMIT,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    payload = chain.run_core_chain(
        query=query,
        follow_up_reply=follow_up_reply,
        db_path=database_path,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explain_limit=explain_limit,
    )
    merge_app_state({"last_chain_query": query, "last_chain_result_count": len(payload.get("top_k_results", []))})
    return payload


def run_project_chain_session(
    query: str,
    db_path: str | Path | None = None,
    follow_up_reply: Optional[str] = None,
    top_k: int = chain.DEFAULT_TOP_K,
    candidate_pool_size: int = chain.DEFAULT_CANDIDATE_POOL_SIZE,
    explain_limit: int = chain.DEFAULT_EXPLAIN_LIMIT,
    persist_history: bool = True,
) -> Dict[str, Any]:
    payload = run_project_chain(
        query=query,
        db_path=db_path,
        follow_up_reply=follow_up_reply,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explain_limit=explain_limit,
    )
    history_id = None
    if persist_history:
        database_path = require_project_database(db_path)
        history_text = query if not follow_up_reply else f"{query}\nFollow-up: {follow_up_reply}"
        history_id = intent.save_intent_frame(database_path, history_text, payload["final_intent_frame"])
    payload["history_id"] = history_id
    return payload


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
        "year_month": row["year_month"],
        "abstract": row["abstract"],
        "section_titles": json.loads(row["section_titles"]),
        "semantic_card": semantic_card,
        "semantic_card_status": row["card_status"] or "",
        "sections": sections,
    }


def load_standard_queries() -> List[Dict[str, Any]]:
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


def load_app_state() -> Dict[str, Any]:
    ensure_system_layout()
    return read_json(APP_STATE_PATH, {})


def build_chain_assets(
    db_path: str | Path | None = None,
    top_k: int = chain.DEFAULT_TOP_K,
    candidate_pool_size: int = chain.DEFAULT_CANDIDATE_POOL_SIZE,
    explain_limit: int = chain.DEFAULT_EXPLAIN_LIMIT,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    return chain.build_core_chain_assets(
        db_path=database_path,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explain_limit=explain_limit,
    )


def format_status_block(stats: Dict[str, int], db_path: str | Path | None = None) -> str:
    database_path = resolve_project_db_path(db_path)
    payload = {
        "db_path": str(database_path),
        "papers": stats.get("papers", 0),
        "sections": stats.get("sections", 0),
        "fts_rows": stats.get("fts_rows", 0),
        "semantic_cards": stats.get("semantic_cards", 0),
        "intent_histories": stats.get("intent_histories", 0),
        "saved_papers": stats.get("saved_papers", 0),
        "app_state_path": str(APP_STATE_PATH),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
