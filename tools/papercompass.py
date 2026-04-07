"""
PaperCompass 统一项目入口。

对外优先暴露“构建项目 / 搜索项目 / 查看项目状态 / 维护语义卡片”，
把 day 脚本收敛为内部实现细节。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from papercompass_services import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DB_PATH,
    DEFAULT_PILOT_COUNT,
    DEFAULT_SEMANTIC_TARGET_COUNT,
    analyze_query_intent,
    build_chain_assets,
    build_project,
    build_intent_assets,
    build_semantic_assets,
    format_status_block,
    generate_semantic_card_for_paper,
    load_project_stats,
    run_project_chain,
    search_project,
)


def print_json(payload: object) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaperCompass unified project CLI.")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build the PaperCompass project index and semantic layer.")
    build_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    build_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    build_parser.add_argument("--top-k", type=int, default=10)
    build_parser.add_argument("--skip-debug-queries", action="store_true")
    build_parser.add_argument("--skip-semantic-cards", action="store_true")
    build_parser.add_argument("--semantic-target-count", type=int, default=DEFAULT_SEMANTIC_TARGET_COUNT)
    build_parser.add_argument("--pilot-count", type=int, default=DEFAULT_PILOT_COUNT)
    build_parser.add_argument("--refresh-semantic-cards", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search papers from the unified project index.")
    search_parser.add_argument("query")
    search_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--mode", choices=["basic", "exact", "hybrid"], default="hybrid")

    status_parser = subparsers.add_parser("status", help="Show current PaperCompass project status.")
    status_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)

    cards_parser = subparsers.add_parser("cards", help="Build or refresh semantic cards for the project.")
    cards_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    cards_parser.add_argument("--target-count", type=int, default=DEFAULT_SEMANTIC_TARGET_COUNT)
    cards_parser.add_argument("--pilot-count", type=int, default=DEFAULT_PILOT_COUNT)
    cards_parser.add_argument("--refresh", action="store_true")
    cards_parser.add_argument("--paper-id", help="Generate or refresh one semantic card only.")

    intent_parser = subparsers.add_parser("intent", help="Parse a natural-language query into an IntentFrame.")
    intent_parser.add_argument("text")
    intent_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    intent_parser.add_argument("--history-id", type=int, help="If provided, treat text as a follow-up reply and merge it.")
    intent_parser.add_argument("--no-save", action="store_true")

    intent_build_parser = subparsers.add_parser("intent-build", help="Generate Day 4 intent prompt and evaluation outputs.")
    intent_build_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)

    chain_parser = subparsers.add_parser("chain", help="Run the full Day 5 core method chain for one query.")
    chain_parser.add_argument("query")
    chain_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    chain_parser.add_argument("--follow-up", help="Optional aggregated clarification reply to merge before retrieval.")
    chain_parser.add_argument("--top-k", type=int, default=10)
    chain_parser.add_argument("--candidate-pool-size", type=int, default=100)
    chain_parser.add_argument("--explain-limit", type=int, default=20)

    chain_build_parser = subparsers.add_parser("chain-build", help="Generate Day 5 core chain demo outputs.")
    chain_build_parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    chain_build_parser.add_argument("--top-k", type=int, default=10)
    chain_build_parser.add_argument("--candidate-pool-size", type=int, default=100)
    chain_build_parser.add_argument("--explain-limit", type=int, default=20)

    return parser.parse_args()


def run_build_command(args: argparse.Namespace) -> None:
    summary = build_project(
        data_root=args.data_root,
        db_path=args.db_path,
        top_k=args.top_k,
        generate_query_debug=not args.skip_debug_queries,
        build_semantic_layer=not args.skip_semantic_cards,
        semantic_target_count=args.semantic_target_count,
        pilot_count=args.pilot_count,
        refresh_semantic_cards=args.refresh_semantic_cards,
    )
    print_json(summary)


def run_search_command(args: argparse.Namespace) -> None:
    results = search_project(args.query, mode=args.mode, top_k=args.top_k, db_path=args.db_path)
    print(f"Query: {args.query}")
    print_json(results)


def run_status_command(args: argparse.Namespace) -> None:
    stats = load_project_stats(args.db_path)
    print(format_status_block(stats, args.db_path))


def run_cards_command(args: argparse.Namespace) -> None:
    if args.paper_id:
        payload = generate_semantic_card_for_paper(args.paper_id, db_path=args.db_path, refresh=args.refresh)
        print_json(payload)
        return

    summary = build_semantic_assets(
        db_path=args.db_path,
        target_count=args.target_count,
        pilot_count=args.pilot_count,
        refresh=args.refresh,
    )
    print_json(summary)


def run_intent_command(args: argparse.Namespace) -> None:
    payload = analyze_query_intent(
        user_text=args.text,
        db_path=args.db_path,
        history_id=args.history_id,
        persist=not args.no_save,
    )
    print_json(payload)


def run_intent_build_command(args: argparse.Namespace) -> None:
    summary = build_intent_assets(db_path=args.db_path)
    print_json(summary)


def run_chain_command(args: argparse.Namespace) -> None:
    payload = run_project_chain(
        query=args.query,
        db_path=args.db_path,
        follow_up_reply=args.follow_up,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        explain_limit=args.explain_limit,
    )
    print_json(payload)


def run_chain_build_command(args: argparse.Namespace) -> None:
    summary = build_chain_assets(
        db_path=args.db_path,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        explain_limit=args.explain_limit,
    )
    print_json(summary)


def main() -> None:
    args = parse_args()
    if args.command is None:
        args = argparse.Namespace(command="status", db_path=DEFAULT_DB_PATH)

    if args.command == "build":
        run_build_command(args)
    elif args.command == "search":
        run_search_command(args)
    elif args.command == "status":
        run_status_command(args)
    elif args.command == "cards":
        run_cards_command(args)
    elif args.command == "intent":
        run_intent_command(args)
    elif args.command == "intent-build":
        run_intent_build_command(args)
    elif args.command == "chain":
        run_chain_command(args)
    elif args.command == "chain-build":
        run_chain_build_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
