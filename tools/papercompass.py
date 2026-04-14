"""
PaperCompass 统一项目入口。

对外优先暴露“构建项目 / 搜索项目 / 查看项目状态 / 维护语义卡片”，
把分阶段脚本收敛为内部实现细节。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from papercompass_core.llm import OpenAIAPIError
from papercompass_core.services import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PILOT_COUNT,
    DEFAULT_SEMANTIC_BACKFILL_MODE,
    DEFAULT_SEMANTIC_TARGET_COUNT,
    analyze_query_intent,
    build_chain_assets,
    build_project,
    build_intent_assets,
    build_semantic_assets,
    format_status_block,
    generate_semantic_card_for_paper,
    get_default_db_path,
    get_paper_detail,
    get_semantic_backfill_status,
    list_saved_papers,
    list_search_history,
    load_project_stats,
    restore_semantic_card_cache,
    run_project_chain_session,
    save_paper,
    search_project,
    start_semantic_backfill,
    unsave_paper,
)

DEFAULT_CHAIN_TOP_K = 5
DEFAULT_CHAIN_CANDIDATE_POOL_SIZE = 40
DEFAULT_CHAIN_EXPLAIN_LIMIT = 5


# 统一以 UTF-8 输出 JSON，尽量避免 Windows 终端出现中文乱码。
def print_json(payload: object) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# 统一注册所有子命令，把项目能力暴露给终端用户。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaperCompass 统一项目命令行工具。")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="构建 PaperCompass 项目索引与语义层。")
    build_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    build_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    build_parser.add_argument("--top-k", type=int, default=10)
    build_parser.add_argument(
        "--with-debug-queries",
        action="store_true",
        help="构建时额外运行 20 条 smoke/debug 查询。由于会执行完整检索检查，速度会更慢。",
    )
    build_parser.add_argument("--skip-debug-queries", action="store_true", help=argparse.SUPPRESS)
    build_parser.add_argument(
        "--with-semantic-cards",
        action="store_true",
        help="构建时额外生成语义卡片。由于需要调用 LLM，速度会更慢。",
    )
    build_parser.add_argument("--skip-semantic-cards", action="store_true", help=argparse.SUPPRESS)
    build_parser.add_argument("--semantic-target-count", type=int, default=DEFAULT_SEMANTIC_TARGET_COUNT)
    build_parser.add_argument("--pilot-count", type=int, default=DEFAULT_PILOT_COUNT)
    build_parser.add_argument("--refresh-semantic-cards", action="store_true")
    build_parser.add_argument("--semantic-backfill-mode", choices=["standard", "all"], default=DEFAULT_SEMANTIC_BACKFILL_MODE)
    build_parser.add_argument("--skip-semantic-backfill", action="store_true")
    build_parser.add_argument("--store-raw-json", action="store_true", help="将原始 JSON 写入 raw_papers；默认关闭以加快 build。")
    build_parser.add_argument("--force-rebuild", action="store_true", help="即使数据源未变化，也强制全量重建数据库。")

    search_parser = subparsers.add_parser("search", help="从统一项目索引中检索论文。")
    search_parser.add_argument("query")
    search_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--mode", choices=["basic", "exact", "hybrid"], default="hybrid")

    status_parser = subparsers.add_parser("status", help="查看当前 PaperCompass 项目状态。")
    status_parser.add_argument("--db-path", type=Path, default=get_default_db_path())

    cards_parser = subparsers.add_parser("cards", help="为项目构建或刷新语义卡片。")
    cards_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    cards_parser.add_argument("--target-count", type=int, default=DEFAULT_SEMANTIC_TARGET_COUNT)
    cards_parser.add_argument("--target-ratio", type=float, help="目标覆盖率，取值通常为 0-1；提供后优先于 target-count。")
    cards_parser.add_argument("--pilot-count", type=int, default=DEFAULT_PILOT_COUNT)
    cards_parser.add_argument("--refresh", action="store_true")
    cards_parser.add_argument("--paper-id", help="只生成或刷新单篇论文的语义卡片。")

    backfill_parser = subparsers.add_parser("semantic-backfill", help="启动或查看后台语义卡补全任务。")
    backfill_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    backfill_parser.add_argument("--mode", choices=["standard", "all"], default=DEFAULT_SEMANTIC_BACKFILL_MODE)
    backfill_parser.add_argument("--refresh", action="store_true")
    backfill_parser.add_argument("--target-count", type=int, help="补全到指定语义卡数量后停止。")
    backfill_parser.add_argument("--target-ratio", type=float, help="补全到指定覆盖率后停止，取值通常为 0-1。")
    backfill_parser.add_argument("--status", action="store_true")
    backfill_parser.add_argument("--restore-cache-only", action="store_true")

    intent_parser = subparsers.add_parser("intent", help="将自然语言查询解析为 IntentFrame。")
    intent_parser.add_argument("text")
    intent_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    intent_parser.add_argument("--history-id", type=int, help="如果提供该参数，则把 text 当作 follow-up 回复并执行合并。")
    intent_parser.add_argument("--no-save", action="store_true")

    intent_build_parser = subparsers.add_parser("intent-build", help="生成意图提示词与评估产物。")
    intent_build_parser.add_argument("--db-path", type=Path, default=get_default_db_path())

    chain_parser = subparsers.add_parser("chain", help="对单条查询运行完整核心方法链路。")
    chain_parser.add_argument("query")
    chain_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    chain_parser.add_argument("--follow-up", help="可选的聚合澄清回复，会在检索前先合并进意图。")
    chain_parser.add_argument("--top-k", type=int, default=DEFAULT_CHAIN_TOP_K)
    chain_parser.add_argument("--candidate-pool-size", type=int, default=DEFAULT_CHAIN_CANDIDATE_POOL_SIZE)
    chain_parser.add_argument("--explain-limit", type=int, default=DEFAULT_CHAIN_EXPLAIN_LIMIT)

    chain_build_parser = subparsers.add_parser("chain-build", help="生成核心链路演示产物。")
    chain_build_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    chain_build_parser.add_argument("--top-k", type=int, default=DEFAULT_CHAIN_TOP_K)
    chain_build_parser.add_argument(
        "--candidate-pool-size", type=int, default=DEFAULT_CHAIN_CANDIDATE_POOL_SIZE
    )
    chain_build_parser.add_argument("--explain-limit", type=int, default=DEFAULT_CHAIN_EXPLAIN_LIMIT)

    history_parser = subparsers.add_parser("history", help="列出最近的检索历史。")
    history_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    history_parser.add_argument("--limit", type=int, default=20)

    saved_parser = subparsers.add_parser("saved", help="列出已收藏论文。")
    saved_parser.add_argument("--db-path", type=Path, default=get_default_db_path())
    saved_parser.add_argument("--limit", type=int, default=20)

    save_parser = subparsers.add_parser("save", help="收藏一篇论文。")
    save_parser.add_argument("paper_id")
    save_parser.add_argument("--db-path", type=Path, default=get_default_db_path())

    unsave_parser = subparsers.add_parser("unsave", help="取消收藏一篇论文。")
    unsave_parser.add_argument("paper_id")
    unsave_parser.add_argument("--db-path", type=Path, default=get_default_db_path())

    paper_parser = subparsers.add_parser("paper", help="查看单篇论文详情。")
    paper_parser.add_argument("paper_id")
    paper_parser.add_argument("--db-path", type=Path, default=get_default_db_path())

    return parser.parse_args()


# 构建项目数据库、调试查询和可选语义层。
def run_build_command(args: argparse.Namespace) -> None:
    summary = build_project(
        data_root=args.data_root,
        db_path=args.db_path,
        top_k=args.top_k,
        generate_query_debug=args.with_debug_queries and not args.skip_debug_queries,
        build_semantic_layer=args.with_semantic_cards and not args.skip_semantic_cards,
        semantic_target_count=args.semantic_target_count,
        pilot_count=args.pilot_count,
        refresh_semantic_cards=args.refresh_semantic_cards,
        auto_semantic_backfill=not args.skip_semantic_backfill,
        semantic_backfill_mode=args.semantic_backfill_mode,
        store_raw_json=args.store_raw_json,
        force_rebuild=args.force_rebuild,
    )
    print_json(summary)


# 对已有项目数据库执行一次检索。
def run_search_command(args: argparse.Namespace) -> None:
    results = search_project(args.query, mode=args.mode, top_k=args.top_k, db_path=args.db_path)
    print(f"查询: {args.query}")
    print_json(results)


# 查看当前项目状态。
def run_status_command(args: argparse.Namespace) -> None:
    stats = load_project_stats(args.db_path)
    print(format_status_block(stats, args.db_path))


# 生成单篇或批量语义卡片。
def run_cards_command(args: argparse.Namespace) -> None:
    if args.paper_id:
        payload = generate_semantic_card_for_paper(args.paper_id, db_path=args.db_path, refresh=args.refresh)
        print_json(payload)
        return

    summary = build_semantic_assets(
        db_path=args.db_path,
        target_count=args.target_count,
        target_ratio=args.target_ratio,
        pilot_count=args.pilot_count,
        refresh=args.refresh,
    )
    print_json(summary)


# 启动、恢复或查看后台语义补全任务。
def run_semantic_backfill_command(args: argparse.Namespace) -> None:
    if args.status:
        print_json(get_semantic_backfill_status())
        return
    if args.restore_cache_only:
        print_json(restore_semantic_card_cache(args.db_path, refresh=args.refresh))
        return
    print_json(
        start_semantic_backfill(
            args.db_path,
            mode=args.mode,
            refresh=args.refresh,
            target_count=args.target_count,
            target_ratio=args.target_ratio,
        )
    )


# 把自然语言查询解析成结构化意图。
def run_intent_command(args: argparse.Namespace) -> None:
    payload = analyze_query_intent(
        user_text=args.text,
        db_path=args.db_path,
        history_id=args.history_id,
        persist=not args.no_save,
    )
    print_json(payload)


# 生成意图模块的提示词和评估产物。
def run_intent_build_command(args: argparse.Namespace) -> None:
    summary = build_intent_assets(db_path=args.db_path)
    print_json(summary)


# 对单个查询运行完整主链路。
def run_chain_command(args: argparse.Namespace) -> None:
    payload = run_project_chain_session(
        query=args.query,
        db_path=args.db_path,
        follow_up_reply=args.follow_up,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        explain_limit=args.explain_limit,
    )
    print_json(payload)


# 生成主链路演示、评估和回归产物。
def run_chain_build_command(args: argparse.Namespace) -> None:
    summary = build_chain_assets(
        db_path=args.db_path,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        explain_limit=args.explain_limit,
    )
    print_json(summary)


# 查询最近的搜索历史。
def run_history_command(args: argparse.Namespace) -> None:
    print_json(list_search_history(db_path=args.db_path, limit=args.limit))


# 查询收藏论文列表。
def run_saved_command(args: argparse.Namespace) -> None:
    print_json(list_saved_papers(db_path=args.db_path, limit=args.limit))


# 收藏指定论文。
def run_save_command(args: argparse.Namespace) -> None:
    print_json(save_paper(args.paper_id, db_path=args.db_path))


# 取消收藏指定论文。
def run_unsave_command(args: argparse.Namespace) -> None:
    print_json(unsave_paper(args.paper_id, db_path=args.db_path))


# 查看单篇论文详情。
def run_paper_command(args: argparse.Namespace) -> None:
    print_json(get_paper_detail(args.paper_id, db_path=args.db_path))


# 根据子命令把请求分发到对应的服务实现。
def main() -> None:
    try:
        args = parse_args()
        if args.command is None:
            args = argparse.Namespace(command="status", db_path=get_default_db_path())

        if args.command == "build":
            run_build_command(args)
        elif args.command == "search":
            run_search_command(args)
        elif args.command == "status":
            run_status_command(args)
        elif args.command == "cards":
            run_cards_command(args)
        elif args.command == "semantic-backfill":
            run_semantic_backfill_command(args)
        elif args.command == "intent":
            run_intent_command(args)
        elif args.command == "intent-build":
            run_intent_build_command(args)
        elif args.command == "chain":
            run_chain_command(args)
        elif args.command == "chain-build":
            run_chain_build_command(args)
        elif args.command == "history":
            run_history_command(args)
        elif args.command == "saved":
            run_saved_command(args)
        elif args.command == "save":
            run_save_command(args)
        elif args.command == "unsave":
            run_unsave_command(args)
        elif args.command == "paper":
            run_paper_command(args)
        else:
            raise ValueError(f"不支持的命令：{args.command}")
    except (OpenAIAPIError, PermissionError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
