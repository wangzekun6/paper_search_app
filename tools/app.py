"""
基于 Streamlit 的项目交互界面。

这个页面主要提供两件事：
1. 直接访问 Day 2 的 SQLite 检索能力
2. 可选地先用大模型把自然语言 query 改写成更适合检索的关键词

它本身不负责建库，只负责把数据库能力包装成可操作的前端页面。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from dataset_config import DATASET_LABEL, DATASET_RELATIVE_PATH
from day2_pipeline import (
    DEFAULT_DB_PATH,
    database_exists,
    load_database_stats,
    search_basic,
    search_exact_matches,
    search_hybrid,
)
from openai_helpers import (
    OPENAI_API_KEY,
    generate_keywords_via_openai,
    test_openai_api,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEARCH_MODE_OPTIONS = {
    "Hybrid": "hybrid",
    "FTS": "basic",
    "Exact Match": "exact",
}


@st.cache_data(show_spinner=False)
def load_search_stats() -> Dict[str, int]:
    return load_database_stats(DEFAULT_DB_PATH)


@st.cache_data(show_spinner=False)
def run_search(query: str, mode: str, top_k: int) -> List[Dict[str, Any]]:
    """根据页面选择的模式，把请求路由到 Day 2 的对应检索函数。"""

    if mode == "basic":
        return search_basic(query, top_k=top_k, db_path=DEFAULT_DB_PATH)
    if mode == "exact":
        return search_exact_matches(query, top_k=top_k, db_path=DEFAULT_DB_PATH)
    return search_hybrid(query, top_k=top_k, db_path=DEFAULT_DB_PATH)


def generate_keywords_via_model(natural_query: str, openai_key: str = "") -> str:
    """
    先尝试用大模型把自然语言 query 改写成关键词。

    如果失败，就退回简单分词，保证页面交互不断。
    """

    if not natural_query.strip():
        return ""

    try:
        result, _ = generate_keywords_via_openai(natural_query, openai_key or OPENAI_API_KEY)
        if result:
            return result.strip()
    except Exception:
        logger.exception("调用兼容大模型查询解析失败")

    # 如果大模型改写失败，就退回到一个最简单的分词版本，
    # 至少保证页面还能继续检索，而不是直接报错中断。
    tokens = [token.strip().lower() for token in natural_query.replace(",", " ").split() if token.strip()]
    return ", ".join(tokens)


def create_search_sidebar(stats: Dict[str, int]) -> Dict[str, Any]:
    """渲染左侧检索配置区，并返回当前用户选择的检索参数。"""

    with st.sidebar:
        st.subheader("当前数据源")
        st.caption(DATASET_LABEL)
        st.code(DATASET_RELATIVE_PATH)

        st.subheader("Day 2 数据库")
        st.code(str(Path(DEFAULT_DB_PATH).relative_to(Path.cwd().parent if Path.cwd().name == "tools" else Path.cwd())))
        st.caption(f"papers: {stats['papers']} | paper_sections: {stats['sections']} | fts_rows: {stats['fts_rows']}")

        st.header("搜索配置")
        keyword = st.text_input(
            "输入 query:",
            value="",
            help="支持普通主题词、长一点的自然语言短句、作者名、标题 hint、方法名和数据集名。",
        )
        search_mode_label = st.radio(
            "检索模式:",
            list(SEARCH_MODE_OPTIONS.keys()),
            horizontal=True,
        )
        top_k = st.slider("返回条数", min_value=5, max_value=20, value=10, step=1)
        show_scores = st.checkbox("显示分数与匹配类型", value=True)
        use_nl = st.checkbox(
            "启用大模型查询解析（百炼兼容 API）",
            value=False,
            help="先把自然语言 query 转成短英文关键词，再执行 Day 2 检索。",
        )

        if use_nl:
            st.info("可选测试当前百炼 API Key 是否可用。")
            if st.button("测试百炼 API Key"):
                ok, message = test_openai_api(OPENAI_API_KEY)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

    return {
        "keyword": keyword,
        "search_mode": SEARCH_MODE_OPTIONS[search_mode_label],
        "top_k": top_k,
        "show_scores": show_scores,
        "use_nl": use_nl,
        "openai_key": OPENAI_API_KEY,
    }


def display_search_results(results: List[Dict[str, Any]], stats: Dict[str, int], search_params: Dict[str, Any]) -> None:
    """把检索结果、统计信息和下载按钮渲染到页面主区域。"""

    keyword = search_params["keyword"]
    show_scores = search_params["show_scores"]

    st.subheader("搜索统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总论文数", stats["papers"])
    with col2:
        st.metric("Section 记录数", stats["sections"])
    with col3:
        st.metric("匹配结果", len(results))

    if keyword:
        st.info(f"当前 query: {keyword}")

    if not results:
        st.info("没有找到符合条件的论文。")
        return

    st.subheader(f"找到 {len(results)} 条匹配结果")

    display_rows: List[Dict[str, Any]] = []
    for item in results:
        row = {
            "paper_id": item["paper_id"],
            "title": item["title"],
            "matched_field": item.get("matched_field", ""),
            "matched_snippet": item.get("matched_snippet", ""),
        }
        if show_scores:
            row["fts_score"] = item.get("fts_score")
            row["exact_score"] = item.get("exact_score")
            row["match_type"] = item.get("match_type")
        display_rows.append(row)

    st.dataframe(display_rows, use_container_width=True)

    output_data = {
        "dataset": DATASET_LABEL,
        "db_path": str(DEFAULT_DB_PATH),
        "query": keyword,
        "mode": search_params["search_mode"],
        "results": results,
    }
    filename = "day2-search-results"
    if keyword:
        filename += f"-{keyword.replace(' ', '_').replace(',', '_')}"
    st.download_button(
        label="下载结果 (JSON)",
        data=json.dumps(output_data, ensure_ascii=False, indent=2),
        file_name=f"{filename}.json",
        mime="application/json",
    )


def main() -> None:
    """Streamlit 页面入口：检查数据库、读取配置、执行检索并展示结果。"""

    st.set_page_config(page_title="PaperCompass", layout="wide")
    st.title("PaperCompass")
    st.caption(f"当前检索数据集：{DATASET_LABEL}")

    if not database_exists(DEFAULT_DB_PATH):
        st.error(
            "Day 2 数据库不存在。请先在 `tools/` 目录运行 `python day2_pipeline.py build`，"
            "生成全量 SQLite 与 FTS 索引。"
        )
        return

    stats = load_search_stats()
    search_params = create_search_sidebar(stats)

    if st.button("搜索论文", type="primary"):
        original_query = search_params.get("keyword", "").strip()
        if not original_query:
            st.warning("请输入 query。")
            return

        if search_params.get("use_nl"):
            # 自然语言改写是可选项。
            # 因为底层检索本质上仍然是关键词 / FTS 检索，长自然语言直接搜往往过严，
            # 先改写成短关键词通常会更容易命中。
            with st.spinner("正在使用兼容大模型解析查询..."):
                nl_keywords = generate_keywords_via_model(original_query, search_params.get("openai_key", ""))
            if nl_keywords:
                st.info(f"系统生成关键词: {nl_keywords}")
                search_params["keyword"] = nl_keywords
            else:
                st.warning("未能生成关键词，将使用原始输入继续检索。")

        with st.spinner("正在检索 SQLite / FTS5 ..."):
            results = run_search(
                search_params["keyword"],
                search_params["search_mode"],
                search_params["top_k"],
            )
        display_search_results(results, stats, search_params)
    else:
        st.info("在左侧配置检索模式与 query 后，点击“搜索论文”。")


if __name__ == "__main__":
    main()
