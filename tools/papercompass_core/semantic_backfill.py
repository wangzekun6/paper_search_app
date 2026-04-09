"""
语义卡片后台补全任务管理器。

这个文件负责启动独立后台进程补全语义卡片、记录运行状态、
维护日志文件，并向前端或 CLI 暴露任务状态查询接口。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from . import chain, retrieval, semantic
from .config import (
    SEMANTIC_BACKFILL_LOG_PATH,
    SEMANTIC_BACKFILL_STATE_PATH,
    ensure_system_layout,
    get_active_runtime_db_path,
    read_json,
    write_json,
)

BACKFILL_MODES = ("standard", "all")
TOOLS_ROOT = Path(__file__).resolve().parents[1]


# 统一使用 UTC 时间戳记录后台任务状态和日志。
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 状态文件记录后台补全任务的当前进度和元信息。
def load_semantic_backfill_state() -> Dict[str, Any]:
    return read_json(SEMANTIC_BACKFILL_STATE_PATH, {})


# 所有状态写入都走这个函数，保证目录存在且写法一致。
def write_semantic_backfill_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    ensure_system_layout()
    write_json(SEMANTIC_BACKFILL_STATE_PATH, payload)
    return payload


# 通过 pid 探测后台进程是否仍然存活。
def process_is_alive(pid: Any) -> bool:
    try:
        numeric_pid = int(pid)
    except Exception:
        return False
    if numeric_pid <= 0:
        return False
    try:
        os.kill(numeric_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# 把磁盘状态和进程实际状态合并，得到更可靠的任务视图。
def get_semantic_backfill_status() -> Dict[str, Any]:
    state = load_semantic_backfill_state()
    if not state:
        return {"status": "idle", "running": False}
    status = str(state.get("status", "idle"))
    running = status == "running" and process_is_alive(state.get("pid"))
    if status == "running" and not running:
        state = dict(state)
        state["status"] = "stale"
        running = False
    state["running"] = running
    return state


# 后台任务输出统一落到日志文件，便于前端和 CLI 追踪。
def append_log_line(message: str) -> None:
    ensure_system_layout()
    SEMANTIC_BACKFILL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEMANTIC_BACKFILL_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message.rstrip()}\n")


# 启动一个脱离当前终端的后台进程，用于持续补全语义卡片。
def start_background_semantic_backfill(
    db_path: Path,
    *,
    mode: str = "standard",
    refresh: bool = False,
) -> Dict[str, Any]:
    ensure_system_layout()
    resolved_db_path = db_path.resolve(strict=False)
    state = get_semantic_backfill_status()
    if state.get("running"):
        return {
            "started": False,
            "already_running": True,
            "status": state.get("status"),
            "pid": state.get("pid"),
            "db_path": state.get("db_path"),
            "mode": state.get("mode"),
            "state_path": str(SEMANTIC_BACKFILL_STATE_PATH),
            "log_path": str(SEMANTIC_BACKFILL_LOG_PATH),
        }

    command = [
        sys.executable,
        "-m",
        "papercompass_core.semantic_backfill",
        "--db-path",
        str(resolved_db_path),
        "--mode",
        mode,
    ]
    if refresh:
        command.append("--refresh")

    creationflags = 0
    if os.name == "nt":
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

    SEMANTIC_BACKFILL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEMANTIC_BACKFILL_LOG_PATH.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(TOOLS_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=creationflags,
        )

    payload = {
        "status": "running",
        "running": True,
        "pid": process.pid,
        "db_path": str(resolved_db_path),
        "mode": mode,
        "refresh": bool(refresh),
        "started_at": now_iso(),
        "state_path": str(SEMANTIC_BACKFILL_STATE_PATH),
        "log_path": str(SEMANTIC_BACKFILL_LOG_PATH),
    }
    write_semantic_backfill_state(payload)
    append_log_line(f"Started semantic backfill pid={process.pid} mode={mode} db={resolved_db_path}")
    return {"started": True, **payload}


# 更新运行态时保留已有字段，只覆盖发生变化的部分。
def update_runtime_state(base_state: Dict[str, Any], **patch: Any) -> Dict[str, Any]:
    payload = dict(base_state)
    payload.update(patch)
    payload["updated_at"] = now_iso()
    write_semantic_backfill_state(payload)
    return payload


# 后台 worker 的实际执行入口：恢复缓存、预热标准查询并可选全量补全。
def run_semantic_backfill(db_path: Path, *, mode: str = "standard", refresh: bool = False) -> Dict[str, Any]:
    resolved_db_path = db_path.resolve(strict=False)
    base_state = {
        "status": "running",
        "pid": os.getpid(),
        "db_path": str(resolved_db_path),
        "mode": mode,
        "refresh": bool(refresh),
        "started_at": now_iso(),
        "log_path": str(SEMANTIC_BACKFILL_LOG_PATH),
        "state_path": str(SEMANTIC_BACKFILL_STATE_PATH),
    }
    update_runtime_state(base_state)
    append_log_line(f"Worker pid={os.getpid()} entered run loop mode={mode} db={resolved_db_path}")

    cache_restore_summary = semantic.restore_cached_semantic_cards(resolved_db_path, refresh=False)
    update_runtime_state(base_state, cache_restore=cache_restore_summary)

    with retrieval.connect_db(resolved_db_path) as conn:
        semantic_cards_before = semantic.current_card_count(conn)
        total_papers = semantic.total_paper_count(conn)

    generation_enabled = semantic.can_use_openai()
    prewarm_summary: Dict[str, Any] | None = None
    full_backfill_summary: Dict[str, Any] | None = None

    if generation_enabled:
        prewarm_summary = chain.prewarm_semantic_cards_for_standard_queries(
            db_path=resolved_db_path,
            specs=chain.STANDARD_QUERY_SPECS,
            candidate_pool_size=chain.DEFAULT_CANDIDATE_POOL_SIZE,
        )
        update_runtime_state(
            base_state,
            cache_restore=cache_restore_summary,
            prewarm_summary=prewarm_summary,
        )

        if mode == "all":
            with retrieval.connect_db(resolved_db_path) as conn:
                missing_ids = semantic.list_missing_generated_ids(conn)
                missing_before = len(missing_ids)

            def progress_callback(progress: Dict[str, Any]) -> None:
                update_runtime_state(
                    base_state,
                    cache_restore=cache_restore_summary,
                    prewarm_summary=prewarm_summary,
                    full_backfill_summary={
                        "missing_before": missing_before,
                        **progress,
                    },
                )

            with retrieval.connect_db(resolved_db_path) as conn:
                generation_result = semantic.generate_cards_for_paper_ids(
                    conn,
                    missing_ids,
                    refresh=refresh,
                    progress_callback=progress_callback,
                )
                missing_after = len(semantic.list_missing_generated_ids(conn))
                semantic_cards_after = semantic.current_card_count(conn)

            full_backfill_summary = {
                "missing_before": missing_before,
                "missing_after": missing_after,
                "semantic_cards_after": semantic_cards_after,
                **generation_result,
            }
    else:
        append_log_line("OpenAI runtime unavailable; semantic backfill stayed in cache-restore-only mode")

    with retrieval.connect_db(resolved_db_path) as conn:
        semantic_cards_after = semantic.current_card_count(conn)

    summary = {
        "status": "completed",
        "pid": os.getpid(),
        "db_path": str(resolved_db_path),
        "mode": mode,
        "refresh": bool(refresh),
        "started_at": base_state["started_at"],
        "finished_at": now_iso(),
        "generation_enabled": generation_enabled,
        "total_papers": total_papers,
        "semantic_cards_before": semantic_cards_before,
        "semantic_cards_after": semantic_cards_after,
        "coverage_ratio": round(semantic_cards_after / total_papers, 4) if total_papers else 0.0,
        "cache_restore": cache_restore_summary,
        "prewarm_summary": prewarm_summary,
        "full_backfill_summary": full_backfill_summary,
        "log_path": str(SEMANTIC_BACKFILL_LOG_PATH),
        "state_path": str(SEMANTIC_BACKFILL_STATE_PATH),
    }
    update_runtime_state(base_state, **summary)
    append_log_line(
        f"Semantic backfill completed mode={mode} cards={semantic_cards_after}/{total_papers} coverage={summary['coverage_ratio']}"
    )
    return summary


# 提供独立命令行入口，便于直接调试后台任务。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background semantic-card backfill worker for PaperCompass.")
    parser.add_argument("--db-path", type=Path, default=get_active_runtime_db_path())
    parser.add_argument("--mode", choices=BACKFILL_MODES, default="standard")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


# 脚本入口：根据命令行参数执行一次后台补全任务。
def main() -> None:
    args = parse_args()
    if args.status:
        print(get_semantic_backfill_status())
        return
    if args.background:
        print(start_background_semantic_backfill(args.db_path, mode=args.mode, refresh=args.refresh))
        return
    try:
        print(run_semantic_backfill(args.db_path, mode=args.mode, refresh=args.refresh))
    except Exception:
        failure = {
            "status": "failed",
            "pid": os.getpid(),
            "db_path": str(args.db_path.resolve(strict=False)),
            "mode": args.mode,
            "refresh": bool(args.refresh),
            "failed_at": now_iso(),
            "error": traceback.format_exc(),
            "log_path": str(SEMANTIC_BACKFILL_LOG_PATH),
            "state_path": str(SEMANTIC_BACKFILL_STATE_PATH),
        }
        write_semantic_backfill_state(failure)
        append_log_line(f"Semantic backfill failed:\n{failure['error']}")
        raise


if __name__ == "__main__":
    main()
