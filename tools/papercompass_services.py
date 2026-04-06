"""
PaperCompass 项目级服务层。

这个模块把 Day1 / Day2 / Day3 的实现细节收敛为统一接口，
让前端、CLI 和后续功能扩展都围绕“完整项目”组织，
而不是直接耦合某个 day 脚本。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import day2_pipeline as day2
import day3_pipeline as day3
import papercompass_chain as chain
import papercompass_intent as intent
from dataset_config import DATASET_DIR


DEFAULT_DATA_ROOT = DATASET_DIR
DEFAULT_DB_PATH = day2.DEFAULT_DB_PATH
DEFAULT_QUERY_JSON_PATH = day2.DEFAULT_QUERY_JSON_PATH
DEFAULT_QUERY_TEXT_PATH = day2.DEFAULT_QUERY_TEXT_PATH
DEFAULT_SEMANTIC_TARGET_COUNT = day3.DEFAULT_TARGET_COUNT
DEFAULT_PILOT_COUNT = day3.DEFAULT_PILOT_COUNT

SEARCH_MODE_TO_FN: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "basic": day2.search_basic,
    "exact": day2.search_exact_matches,
    "hybrid": day2.search_hybrid,
}


def project_database_exists(db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    return day2.database_exists(db_path)


def require_project_database(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    database_path = Path(db_path)
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


def load_project_stats(db_path: str | Path = DEFAULT_DB_PATH) -> Dict[str, int]:
    database_path = Path(db_path)
    stats = day2.load_database_stats(database_path)
    stats["semantic_cards"] = 0
    stats["intent_histories"] = 0

    if not database_path.exists():
        return stats

    with day2.connect_db(database_path) as conn:
        stats["semantic_cards"] = day3.current_card_count(conn)
        stats["intent_histories"] = int(conn.execute("SELECT COUNT(*) FROM search_history").fetchone()[0])
    return stats


def search_project(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    if mode not in SEARCH_MODE_TO_FN:
        raise ValueError(f"Unsupported search mode: {mode}")
    database_path = require_project_database(db_path)
    return SEARCH_MODE_TO_FN[mode](query, top_k=top_k, db_path=database_path)


def build_semantic_assets(
    db_path: str | Path = DEFAULT_DB_PATH,
    target_count: int = DEFAULT_SEMANTIC_TARGET_COUNT,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    refresh: bool = False,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    day3.ensure_output_dir()
    day3.write_prompt_file()
    openai_available = day3.can_use_openai()
    openai_message = day3.OPENAI_RUNTIME_MESSAGE

    with day2.connect_db(database_path) as conn:
        if pilot_count > 0:
            pilot_ids = day3.select_candidate_paper_ids(conn, max(pilot_count, 5))[:pilot_count]
            pilot_cards = day3.generate_pilot_cards(conn, pilot_ids, refresh=refresh)
        else:
            pilot_cards = []

        day3.dump_json(day3.PILOT_OUTPUT_PATH, pilot_cards)

        generated_count = day3.generate_cards_until_target(conn, target_count, refresh=refresh)
        generated_cards = day3.load_generated_cards(conn, limit=target_count)
        day3.dump_json(day3.SAMPLE_OUTPUT_PATH, generated_cards)

        quality_payload = day3.write_quality_check(conn, sample_size=day3.DEFAULT_QUALITY_SAMPLE_SIZE)
        stability_payload = day3.write_field_stability_report(day3.load_generated_cards(conn))
        status_counts = _semantic_status_counts(conn)

    day3.write_cache_strategy()
    day3.write_feedback(
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
    db_path: str | Path = DEFAULT_DB_PATH,
    top_k: int = 10,
    generate_query_debug: bool = True,
    build_semantic_layer: bool = True,
    semantic_target_count: int = DEFAULT_SEMANTIC_TARGET_COUNT,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    refresh_semantic_cards: bool = False,
) -> Dict[str, Any]:
    data_root_path = Path(data_root)
    database_path = Path(db_path)
    query_payload: Optional[Dict[str, Any]] = None

    database_stats = day2.build_database(data_root_path, database_path)
    if generate_query_debug:
        query_payload = day2.run_debug_queries(
            db_path=database_path,
            query_json_path=DEFAULT_QUERY_JSON_PATH,
            query_text_path=DEFAULT_QUERY_TEXT_PATH,
            top_k=top_k,
        )

    semantic_summary = None
    if build_semantic_layer:
        semantic_summary = build_semantic_assets(
            db_path=database_path,
            target_count=semantic_target_count,
            pilot_count=pilot_count,
            refresh=refresh_semantic_cards,
        )

    return {
        "data_root": str(data_root_path.resolve()),
        "db_path": str(database_path.resolve()),
        "database_stats": database_stats,
        "query_debug_enabled": generate_query_debug,
        "query_count": query_payload["query_count"] if query_payload else 0,
        "semantic_layer_enabled": build_semantic_layer,
        "semantic_layer": semantic_summary,
    }


def generate_semantic_card_for_paper(
    paper_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    refresh: bool = False,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    day3.ensure_output_dir()
    day3.write_prompt_file()

    with day2.connect_db(database_path) as conn:
        card, used_model = day3.generate_card_for_paper(conn, paper_id, refresh=refresh)

    if card is None:
        raise day3.OpenAIAPIError(f"Failed to generate semantic card for {paper_id}")

    return {
        "db_path": str(database_path),
        "paper_id": paper_id,
        "used_model": used_model,
        "card": card,
    }


def analyze_query_intent(
    user_text: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    history_id: Optional[int] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    database_path = Path(db_path)
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

    return {
        "db_path": str(database_path),
        "history_id": new_history_id,
        "prior_history_id": prior_history_id,
        "parser": parser,
        "used_model": used_model,
        "intent_frame": frame,
    }


def build_intent_assets(
    db_path: str | Path = DEFAULT_DB_PATH,
    queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if Path(db_path).exists():
        require_project_database(db_path)
    summary = intent.build_intent_assets(queries=queries)
    if Path(db_path).exists():
        summary["intent_histories"] = intent.load_search_history_count(Path(db_path))
    return summary


def run_project_chain(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    follow_up_reply: Optional[str] = None,
    top_k: int = chain.DEFAULT_TOP_K,
    candidate_pool_size: int = chain.DEFAULT_CANDIDATE_POOL_SIZE,
    explain_limit: int = chain.DEFAULT_EXPLAIN_LIMIT,
) -> Dict[str, Any]:
    database_path = require_project_database(db_path)
    return chain.run_core_chain(
        query=query,
        follow_up_reply=follow_up_reply,
        db_path=database_path,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explain_limit=explain_limit,
    )


def build_chain_assets(
    db_path: str | Path = DEFAULT_DB_PATH,
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


def format_status_block(stats: Dict[str, int], db_path: str | Path = DEFAULT_DB_PATH) -> str:
    payload = {
        "db_path": str(Path(db_path)),
        "papers": stats.get("papers", 0),
        "sections": stats.get("sections", 0),
        "fts_rows": stats.get("fts_rows", 0),
        "semantic_cards": stats.get("semantic_cards", 0),
        "intent_histories": stats.get("intent_histories", 0),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
