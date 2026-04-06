from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from dataset_config import DATASET_LABEL, DATASET_RELATIVE_PATH
from openai_helpers import (
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    generate_keywords_via_openai,
    test_openai_api,
)
from papercompass_services import (
    DEFAULT_DB_PATH,
    load_project_stats,
    project_database_exists,
    run_project_chain,
    search_project,
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
    return load_project_stats(DEFAULT_DB_PATH)


@st.cache_data(show_spinner=False)
def run_search(query: str, mode: str, top_k: int) -> List[Dict[str, Any]]:
    return search_project(query, mode=mode, top_k=top_k, db_path=DEFAULT_DB_PATH)


@st.cache_data(show_spinner=False)
def run_chain_search(
    query: str,
    follow_up_reply: str,
    top_k: int,
    candidate_pool_size: int,
    explain_limit: int,
) -> Dict[str, Any]:
    return run_project_chain(
        query=query,
        follow_up_reply=follow_up_reply or None,
        db_path=DEFAULT_DB_PATH,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explain_limit=explain_limit,
    )


def relative_display_path(path: Path) -> str:
    current_dir = Path.cwd()
    base_dir = current_dir.parent if current_dir.name == "tools" else current_dir
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def generate_keywords_via_model(natural_query: str, openai_key: str = "") -> str:
    if not natural_query.strip():
        return ""

    try:
        result, _ = generate_keywords_via_openai(natural_query, openai_key or OPENAI_API_KEY)
        if result:
            return result.strip()
    except Exception:
        logger.exception("Keyword rewrite via OpenAI-compatible API failed.")

    tokens = [token.strip().lower() for token in natural_query.replace(",", " ").split() if token.strip()]
    return ", ".join(tokens)


def render_project_overview(stats: Dict[str, int]) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("论文数", stats["papers"])
    with col2:
        st.metric("Section 数", stats["sections"])
    with col3:
        st.metric("FTS 行数", stats["fts_rows"])
    with col4:
        st.metric("语义卡片", stats["semantic_cards"])
    with col5:
        st.metric("Intent 历史", stats["intent_histories"])


def render_sidebar(stats: Dict[str, int]) -> bool:
    with st.sidebar:
        st.subheader("数据集")
        st.caption(DATASET_LABEL)
        st.code(DATASET_RELATIVE_PATH)

        st.subheader("项目索引")
        st.code(relative_display_path(Path(DEFAULT_DB_PATH)))
        st.caption(
            f"papers: {stats['papers']} | sections: {stats['sections']} | "
            f"fts_rows: {stats['fts_rows']} | semantic_cards: {stats['semantic_cards']} | "
            f"intent_histories: {stats['intent_histories']}"
        )

        st.subheader("API 状态")
        if OPENAI_API_KEY:
            st.success("已检测到 OpenAI-compatible API Key")
        else:
            st.warning("未检测到 API Key。如果你刚配置了环境变量，请重启 Streamlit。")
        st.caption(f"Base URL: {OPENAI_API_BASE}")
        st.caption(f"Model: {OPENAI_MODEL}")
        if st.button("测试 API 连接", use_container_width=True):
            with st.spinner("正在测试 API ..."):
                ok, message = test_openai_api(OPENAI_API_KEY)
            if ok:
                st.success(message)
            else:
                st.error(message)

        show_raw_json = st.checkbox("显示原始 JSON", value=False)
    return show_raw_json


def download_payload_button(label: str, payload: Dict[str, Any], filename: str) -> None:
    st.download_button(
        label=label,
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
    )


def render_string_list(title: str, items: List[str], empty_text: str = "无") -> None:
    st.markdown(f"**{title}**")
    if not items:
        st.caption(empty_text)
        return
    for item in items:
        st.write(f"- {item}")


def display_search_results(
    results: List[Dict[str, Any]],
    query: str,
    original_query: str,
    mode: str,
    show_scores: bool,
    show_raw_json: bool,
) -> None:
    st.subheader("直接检索结果")
    st.caption(f"当前 query: {query}")
    if original_query and original_query != query:
        st.caption(f"原始输入: {original_query}")

    if not results:
        st.info("没有找到匹配结果。")
        return

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

    st.dataframe(display_rows, use_container_width=True, hide_index=True)

    for index, item in enumerate(results[:5], start=1):
        with st.expander(f"#{index} {item['title']}"):
            st.caption(
                f"paper_id={item['paper_id']} | matched_field={item.get('matched_field', '')} | "
                f"match_type={item.get('match_type', '')}"
            )
            snippet = item.get("matched_snippet", "")
            if snippet:
                st.write(snippet)

    payload = {
        "dataset": DATASET_LABEL,
        "db_path": str(DEFAULT_DB_PATH),
        "original_query": original_query,
        "query": query,
        "mode": mode,
        "results": results,
    }
    filename = "papercompass-search-results.json"
    download_payload_button("下载结果 JSON", payload, filename)

    if show_raw_json:
        with st.expander("查看原始 JSON"):
            st.json(payload)


def render_query_variants(frame: Dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        render_string_list("Coarse Queries", frame.get("coarse_queries", []), "无 coarse query")
    with col2:
        render_string_list("Dense Queries", frame.get("dense_queries", []), "无 dense query")
    with col3:
        render_string_list("Exact Queries", frame.get("exact_queries", []), "无 exact query")


def render_intent_summary(frame: Dict[str, Any], title: str) -> None:
    st.subheader(title)
    answered_slots = frame.get("answered_slots", [])
    missing_slots = frame.get("missing_slots", [])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已回答槽位", len(answered_slots))
    with col2:
        st.metric("缺失槽位", len(missing_slots))
    with col3:
        st.metric("需要追问", "是" if frame.get("clarification_needed") else "否")

    if frame.get("clarification_needed") and frame.get("clarification_question"):
        st.warning(frame["clarification_question"])

    render_query_variants(frame)

    with st.expander("查看完整 IntentFrame JSON"):
        st.json(frame)


def render_gap_report(gap_report: Dict[str, Any]) -> None:
    st.subheader("Gap Report")
    render_string_list("缺失槽位", gap_report.get("query_gap", []), "无明显缺失槽位")
    render_string_list("证据缺口", gap_report.get("evidence_gap", []), "暂无明显证据缺口")
    render_string_list("已命中维度", gap_report.get("matched_dimensions", []), "暂无稳定命中维度")
    render_string_list("当前结果为什么偏宽", gap_report.get("why_current_results_are_broad", []), "结果已经较集中")
    render_string_list(
        "下一轮补充什么最有帮助",
        gap_report.get("what_next_answer_would_improve", []),
        "当前结果已可直接使用",
    )


def render_chain_result_card(rank: int, result: Dict[str, Any], evidence_pack: Optional[Dict[str, Any]]) -> None:
    evidence_pack = evidence_pack or {}
    title = result.get("title", "Untitled paper")
    with st.expander(f"#{rank} {title}", expanded=rank <= 3):
        st.caption(
            f"paper_id={result['paper_id']} | final={result['final_score']:.4f} | "
            f"base={result['base_score']:.4f} | intent={result['intent_score']:.4f}"
        )
        st.caption(
            f"paper_type={result.get('paper_type', '')} | retrieval_sources={', '.join(result.get('retrieval_sources', []))} | "
            f"explanation_parser={result.get('explanation_parser', '')}"
        )

        render_string_list("推荐理由", result.get("ranking_reasons", []), "暂无推荐理由")
        render_string_list("未满足约束", result.get("unmet_constraints", []), "暂无明显未满足约束")

        snippets = evidence_pack.get("matched_snippets", [])
        if snippets:
            st.markdown("**证据片段**")
            for snippet in snippets[:3]:
                field = snippet.get("field", "")
                text = snippet.get("snippet", "")
                st.write(f"- [{field}] {text[:240]}")

        matched_sections = evidence_pack.get("matched_sections", [])
        if matched_sections:
            st.caption("匹配章节: " + " | ".join(matched_sections))


def display_chain_results(payload: Dict[str, Any], show_raw_json: bool) -> None:
    final_frame = payload["final_intent_frame"]
    initial_frame = payload["initial_intent_frame"]
    gap_report = payload["intent_gap_report"]
    top_k_results = payload["top_k_results"]
    evidence_packs = payload.get("paper_evidence_packs", {})

    st.subheader("链式检索概览")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("候选池大小", payload.get("candidate_pool_size", 0))
    with col2:
        st.metric("Top-K 结果", len(top_k_results))
    with col3:
        st.metric("是否补充追问", "是" if payload.get("follow_up_reply") else "否")

    left, right = st.columns(2)
    with left:
        render_intent_summary(final_frame, "最终 IntentFrame")
    with right:
        render_gap_report(gap_report)

    if payload.get("follow_up_reply"):
        with st.expander("查看首轮 IntentFrame"):
            st.json(initial_frame)

    st.subheader("Top-K 结果")
    if not top_k_results:
        st.info("当前链路没有返回候选结果。")
    else:
        for rank, item in enumerate(top_k_results, start=1):
            render_chain_result_card(rank, item, evidence_packs.get(item["paper_id"]))

    download_payload_button("下载链路结果 JSON", payload, "papercompass-chain-results.json")

    if show_raw_json:
        with st.expander("查看原始 JSON"):
            st.json(payload)


def render_basic_search_tab(show_raw_json: bool) -> None:
    st.write("适合精确关键词、作者名、标题线索和基础布尔式需求。")
    with st.form("basic_search_form"):
        keyword = st.text_input(
            "输入 query",
            value="",
            help="支持自然语言短句、作者名、标题线索、方法名和数据集名。",
        )
        search_mode_label = st.radio(
            "检索模式",
            list(SEARCH_MODE_OPTIONS.keys()),
            horizontal=True,
        )
        top_k = st.slider("返回条数", min_value=5, max_value=20, value=10, step=1)
        show_scores = st.checkbox("显示分数与匹配类型", value=True)
        use_nl = st.checkbox(
            "先用 API 改写 query",
            value=False,
            help="将自然语言请求改写为更紧凑的检索关键词，再执行基础检索。",
        )
        submitted = st.form_submit_button("搜索论文", type="primary")

    if submitted:
        original_query = keyword.strip()
        if not original_query:
            st.warning("请输入 query。")
            return

        rewritten_query = original_query
        if use_nl:
            with st.spinner("正在改写 query ..."):
                rewritten_query = generate_keywords_via_model(original_query, OPENAI_API_KEY) or original_query

        with st.spinner("正在执行直接检索 ..."):
            results = run_search(rewritten_query, SEARCH_MODE_OPTIONS[search_mode_label], top_k)

        st.session_state["basic_search_payload"] = {
            "original_query": original_query,
            "query": rewritten_query,
            "mode": SEARCH_MODE_OPTIONS[search_mode_label],
            "show_scores": show_scores,
            "results": results,
        }

    payload = st.session_state.get("basic_search_payload")
    if payload:
        display_search_results(
            results=payload["results"],
            query=payload["query"],
            original_query=payload["original_query"],
            mode=payload["mode"],
            show_scores=payload["show_scores"],
            show_raw_json=show_raw_json,
        )
    else:
        st.info("填写参数后点击“搜索论文”。")


def render_chain_search_tab(show_raw_json: bool) -> None:
    st.write("适合自然语言需求、多条件约束、需要追问提示和推荐理由解释的场景。")
    with st.form("chain_search_form"):
        query = st.text_area(
            "输入自然语言需求",
            value="",
            height=110,
            help="例如：我想找最近两年关于 RAG hallucination mitigation 的综述，并解释为什么推荐。",
        )
        follow_up_reply = st.text_area(
            "补充说明（可选）",
            value="",
            height=80,
            help="如果首轮 query 还不够具体，可以一次性补充时间范围、方法约束、论文类型或解释偏好。",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider("Top-K", min_value=3, max_value=10, value=5, step=1)
        with col2:
            candidate_pool_size = st.slider("候选池大小", min_value=20, max_value=120, value=60, step=10)
        with col3:
            explain_limit = st.slider("解释条数", min_value=3, max_value=10, value=5, step=1)
        submitted = st.form_submit_button("运行意图链路", type="primary")

    if submitted:
        normalized_query = query.strip()
        normalized_follow_up = follow_up_reply.strip()
        if not normalized_query:
            st.warning("请输入自然语言 query。")
            return

        with st.spinner("正在解析意图、召回候选并重排 ..."):
            payload = run_chain_search(
                query=normalized_query,
                follow_up_reply=normalized_follow_up,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                explain_limit=max(top_k, explain_limit),
            )
        st.session_state["chain_search_payload"] = payload

    payload = st.session_state.get("chain_search_payload")
    if payload:
        display_chain_results(payload, show_raw_json=show_raw_json)
    else:
        st.info("填写需求后点击“运行意图链路”。")


def main() -> None:
    st.set_page_config(page_title="PaperCompass", layout="wide")
    st.title("PaperCompass")
    st.caption(f"当前检索数据集: {DATASET_LABEL}")

    if not project_database_exists(DEFAULT_DB_PATH):
        st.error(
            "项目索引库不存在。请先在 `tools/` 目录运行 `python papercompass.py build`，"
            "生成统一 SQLite 索引与语义层。"
        )
        return

    stats = load_search_stats()
    show_raw_json = render_sidebar(stats)
    render_project_overview(stats)

    direct_tab, chain_tab = st.tabs(["直接检索", "意图链路"])
    with direct_tab:
        render_basic_search_tab(show_raw_json)
    with chain_tab:
        render_chain_search_tab(show_raw_json)


if __name__ == "__main__":
    main()
