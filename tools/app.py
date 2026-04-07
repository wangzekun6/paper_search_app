from __future__ import annotations

from typing import Any, Dict, Iterable, List

import streamlit as st

from papercompass_core.config import APP_STATE_PATH, DATASET_LABEL, DATASET_RELATIVE_PATH, relative_to_project
from papercompass_core.llm import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL, test_openai_api
from papercompass_core.services import (
    get_default_db_path,
    get_saved_paper_ids,
    list_saved_papers,
    list_search_history,
    load_app_state,
    load_project_stats,
    load_standard_queries,
    project_database_exists,
    run_project_chain_session,
    save_paper,
    unsave_paper,
)


SEARCH_SCENE_LABELS = {
    "topic_exploration": "主题探索",
    "survey_lookup": "综述查找",
    "recent_progress": "近期进展",
    "specific_paper_lookup": "特定论文定位",
    "author_trace": "作者追踪",
    "method_constrained_search": "方法约束检索",
}

PAPER_TYPE_LABELS = {
    "survey": "综述",
    "benchmark": "基准/评测",
    "method": "方法论文",
    "empirical_study": "实证研究",
    "application_study": "应用研究",
    "theory": "理论研究",
    "analysis": "分析论文",
}

PREFERENCE_LABELS = {
    "yes": "是",
    "no": "否",
}

DIMENSION_LABELS = {
    "scene_match": "检索场景匹配",
    "topic_match": "主题匹配",
    "constraint_match": "技术约束匹配",
    "paper_type_match": "论文类型匹配",
    "time_preference_match": "时间偏好匹配",
    "survey_preference_match": "综述偏好匹配",
}

FIELD_LABELS = {
    "title": "标题",
    "authors": "作者",
    "abstract": "摘要",
    "section_titles": "章节标题",
    "section_snippet": "章节片段",
}

MATCH_TYPE_LABELS = {
    "title_hint": "标题线索命中",
    "author_match": "作者命中",
    "phrase_match": "短语命中",
    "fts": "全文检索召回",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def localize_identifier(value: Any, mapping: Dict[str, str]) -> str:
    text = clean_text(value)
    return mapping.get(text, text)


def localize_slot_value(value: Any, slot_key: str = "") -> str:
    text = clean_text(value)
    if not text:
        return ""
    if slot_key == "search_scene":
        return SEARCH_SCENE_LABELS.get(text, text)
    if slot_key == "document_attributes.paper_type":
        return PAPER_TYPE_LABELS.get(text, text)
    if slot_key.startswith("result_preferences."):
        return PREFERENCE_LABELS.get(text, text)
    return text


def localize_dimension_text(value: Any) -> str:
    text = clean_text(value)
    return DIMENSION_LABELS.get(text, text)


def localize_field_text(value: Any) -> str:
    text = clean_text(value)
    return FIELD_LABELS.get(text, text)


def localize_match_type_text(value: Any) -> str:
    text = clean_text(value)
    return MATCH_TYPE_LABELS.get(text, text)


def localize_display_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if text in DIMENSION_LABELS:
        return DIMENSION_LABELS[text]
    if text in MATCH_TYPE_LABELS:
        return MATCH_TYPE_LABELS[text]
    if text in FIELD_LABELS:
        return FIELD_LABELS[text]
    return text


def slot_display(slot: Dict[str, Any], slot_key: str = "") -> str:
    value = slot.get("value")
    if isinstance(value, list):
        normalized = [localize_slot_value(item, slot_key) for item in value if clean_text(item)]
        return "、".join(item for item in normalized if item) or "-"
    return localize_slot_value(value, slot_key) or "-"


def render_string_list(title: str, items: Iterable[str], empty_text: str = "-", formatter=None) -> None:
    normalized = []
    for item in items:
        text = clean_text(item)
        if not text:
            continue
        normalized.append(formatter(text) if formatter else text)
    st.markdown(f"**{title}**")
    if not normalized:
        st.caption(empty_text)
        return
    for item in normalized:
        st.write(f"- {item}")


def apply_demo_query(item: Dict[str, Any]) -> None:
    st.session_state["query_input"] = item["query"]
    st.session_state["follow_up_input"] = item.get("follow_up_reply", "")
    st.session_state["top_k_input"] = 5
    st.session_state["candidate_pool_input"] = 40
    st.session_state["explain_limit_input"] = 5


def render_sidebar(stats: Dict[str, int], standard_queries: List[Dict[str, Any]]) -> bool:
    with st.sidebar:
        st.subheader("系统状态")
        st.caption(DATASET_LABEL)
        st.code(DATASET_RELATIVE_PATH)
        st.code(relative_to_project(get_default_db_path()))
        st.caption(f"状态文件：{relative_to_project(APP_STATE_PATH)}")
        st.caption(
            " | ".join(
                [
                    f"论文={stats['papers']}",
                    f"章节={stats['sections']}",
                    f"FTS={stats['fts_rows']}",
                    f"语义卡={stats['semantic_cards']}",
                    f"历史={stats['intent_histories']}",
                    f"收藏={stats['saved_papers']}",
                ]
            )
        )

        st.subheader("LLM 运行状态")
        if OPENAI_API_KEY:
            st.success("已检测到 API Key")
        else:
            st.error("缺少 API Key，意图解析和 query-paper 匹配将无法正常工作。")
        st.caption(f"基础地址：{OPENAI_API_BASE}")
        st.caption(f"模型：{OPENAI_MODEL}")
        if st.button("测试 API", use_container_width=True):
            with st.spinner("正在测试 API..."):
                ok, message = test_openai_api(OPENAI_API_KEY)
            if ok:
                st.success(message)
            else:
                st.error(message)

        st.subheader("示例回放")
        for index, item in enumerate(standard_queries[:6], start=1):
            if st.button(f"示例 {index}", key=f"demo_{index}", use_container_width=True):
                apply_demo_query(item)

        return st.checkbox("显示原始 JSON", value=False)


def render_overview(stats: Dict[str, int]) -> None:
    columns = st.columns(6)
    metrics = [
        ("论文数", stats["papers"]),
        ("章节数", stats["sections"]),
        ("FTS 行数", stats["fts_rows"]),
        ("语义卡片", stats["semantic_cards"]),
        ("检索历史", stats["intent_histories"]),
        ("收藏论文", stats["saved_papers"]),
    ]
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


def render_intent_panel(frame: Dict[str, Any]) -> None:
    st.subheader("当前意图")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**检索场景**")
        st.write(slot_display(frame.get("search_scene", {}), "search_scene"))
        st.markdown("**研究主题**")
        st.write(
            " | ".join(
                [
                    slot_display(frame["research_topic"]["domain"]),
                    slot_display(frame["research_topic"]["task"]),
                    slot_display(frame["research_topic"]["problem"]),
                ]
            )
        )
        st.markdown("**主题关键词**")
        st.write(slot_display(frame["research_topic"]["keywords"]))
    with col2:
        st.markdown("**技术约束**")
        for label, slot in [
            ("方法", frame["technical_constraints"]["method"]),
            ("模型族", frame["technical_constraints"]["model_family"]),
            ("数据集", frame["technical_constraints"]["dataset"]),
            ("指标", frame["technical_constraints"]["metric"]),
            ("模态", frame["technical_constraints"]["modality"]),
        ]:
            st.write(f"{label}: {slot_display(slot)}")

        st.markdown("**文档属性**")
        for label, slot in [
            ("时间范围", frame["document_attributes"]["time_range"]),
            ("论文类型", frame["document_attributes"]["paper_type"]),
            ("作者", frame["document_attributes"]["author_name"]),
            ("标题线索", frame["document_attributes"]["title_hint"]),
        ]:
            slot_key = "document_attributes.paper_type" if label == "论文类型" else ""
            st.write(f"{label}: {slot_display(slot, slot_key)}")

        st.markdown("**结果偏好**")
        for label, slot in [
            ("偏好最新", frame["result_preferences"]["prefer_recent"]),
            ("偏好经典", frame["result_preferences"]["prefer_classic"]),
            ("偏好综述", frame["result_preferences"]["prefer_survey"]),
            ("偏好多样", frame["result_preferences"]["prefer_diverse"]),
            ("需要解释", frame["result_preferences"]["need_explainable_reason"]),
        ]:
            slot_key = {
                "偏好最新": "result_preferences.prefer_recent",
                "偏好经典": "result_preferences.prefer_classic",
                "偏好综述": "result_preferences.prefer_survey",
                "偏好多样": "result_preferences.prefer_diverse",
                "需要解释": "result_preferences.need_explainable_reason",
            }[label]
            st.write(f"{label}: {slot_display(slot, slot_key)}")

    if frame.get("clarification_needed") and frame.get("clarification_question"):
        st.warning(frame["clarification_question"])

    with st.expander("IntentFrame 原始 JSON"):
        st.json(frame)


def render_gap_panel(gap_report: Dict[str, Any], clarification_needed: bool) -> None:
    st.subheader("Gap 分析")
    render_string_list("缺失槽位", gap_report.get("query_gap", []), "当前没有缺失槽位")
    render_string_list("证据缺口", gap_report.get("evidence_gap", []), "当前没有明显证据缺口")
    render_string_list(
        "已命中维度",
        gap_report.get("matched_dimensions", []),
        "当前还没有稳定命中的维度",
        formatter=localize_dimension_text,
    )
    render_string_list(
        "当前结果为何较宽",
        gap_report.get("why_current_results_are_broad", []),
        "当前结果已经比较聚焦",
    )
    render_string_list(
        "继续补充什么最有帮助",
        gap_report.get("what_next_answer_would_improve", []),
        "当前结果已经具备可用性",
    )
    st.markdown("**追问回复入口**")
    if clarification_needed:
        st.caption("系统仍有未补齐信息，建议在这里回复后直接继续检索。")
    else:
        st.caption("如果你想进一步收敛结果，可继续补充偏好或约束。")
    st.text_area(
        "补充回复",
        key="gap_follow_up_input",
        height=80,
        placeholder="例如：只看最近两年，优先综述，并说明每篇论文为何匹配",
    )
    if st.button("按追问继续检索", key="run_gap_follow_up", use_container_width=True):
        reply = clean_text(st.session_state.get("gap_follow_up_input", ""))
        if not reply:
            st.warning("请先填写补充回复。")
        else:
            st.session_state["follow_up_input"] = reply
            run_query()


def toggle_saved_state(paper_id: str, saved_ids: set[str]) -> None:
    if paper_id in saved_ids:
        unsave_paper(paper_id)
    else:
        save_paper(paper_id)
    st.rerun()


def render_result_card(
    result: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    saved_ids: set[str],
    show_raw_json: bool,
) -> None:
    query_match = result.get("query_paper_match") or {}
    matched_dimensions = query_match.get("matched_dimensions", [])
    label = "取消收藏" if result["paper_id"] in saved_ids else "收藏论文"
    with st.expander(f"{result['title']}", expanded=False):
        header_left, header_right = st.columns([4, 1])
        with header_left:
            st.caption(
                f"{result.get('authors_raw', '')} | {result.get('year_month', '')} | "
                f"总分={result.get('final_score', 0):.3f} | 匹配分={query_match.get('match_score', 0):.3f}"
            )
        with header_right:
            if st.button(label, key=f"save_toggle_{result['paper_id']}", use_container_width=True):
                toggle_saved_state(result["paper_id"], saved_ids)

        render_string_list("排序理由", result.get("ranking_reasons", []), "暂无排序理由")
        render_string_list("已命中维度", matched_dimensions, "暂无命中维度", formatter=localize_dimension_text)
        render_string_list("未满足约束", result.get("unmet_constraints", []), "暂无未满足约束")

        st.markdown("**摘要**")
        st.write(result.get("abstract", ""))

        render_string_list("命中章节", evidence_pack.get("matched_sections", []), "暂无命中章节")
        render_string_list(
            "命中片段",
            [
                f"{localize_field_text(item.get('field', ''))}: {item.get('snippet', '')}"
                for item in evidence_pack.get("matched_snippets", [])
            ],
            "暂无命中片段",
        )

        st.markdown("**排序解释**")
        st.write(query_match.get("brief_reason", "暂无 query-paper 匹配解释"))

        retrieval_summary = []
        if result.get("matched_field"):
            retrieval_summary.append(f"命中字段：{localize_field_text(result.get('matched_field', ''))}")
        if result.get("exact_match_type"):
            retrieval_summary.append(f"命中类型：{localize_match_type_text(result.get('exact_match_type', ''))}")
        if retrieval_summary:
            st.caption(" | ".join(retrieval_summary))

        with st.expander("语义卡片"):
            st.json(evidence_pack.get("semantic_card", {}))

        with st.expander("Query-论文匹配"):
            st.json(query_match)

        if show_raw_json:
            with st.expander("结果原始 JSON"):
                st.json(result)


def render_management_area(standard_queries: List[Dict[str, Any]]) -> None:
    st.subheader("管理区")
    saved_tab, history_tab, demos_tab = st.tabs(["收藏论文", "历史记录", "标准示例"])

    with saved_tab:
        saved_items = list_saved_papers(limit=50)
        if not saved_items:
            st.info("当前还没有收藏论文。")
        else:
            for item in saved_items:
                st.write(f"- {item['title']} | {item['authors_raw']} | {item['year_month']}")

    with history_tab:
        history_items = list_search_history(limit=20)
        if not history_items:
            st.info("当前还没有检索历史。")
        else:
            for item in history_items:
                st.write(f"- {item['created_at']} | {item['query_text']}")

    with demos_tab:
        for index, item in enumerate(standard_queries, start=1):
            col_query, col_action = st.columns([5, 1])
            with col_query:
                st.write(f"{index}. {item['query']}")
                if item.get("follow_up_reply"):
                    st.caption(f"补充回复：{item['follow_up_reply']}")
            with col_action:
                if st.button("回放", key=f"replay_demo_{index}", use_container_width=True):
                    apply_demo_query(item)
                    run_query()


def run_query() -> None:
    query = clean_text(st.session_state.get("query_input", ""))
    follow_up = clean_text(st.session_state.get("follow_up_input", ""))
    top_k = int(st.session_state.get("top_k_input", 5))
    candidate_pool = int(st.session_state.get("candidate_pool_input", 40))
    explain_limit = int(st.session_state.get("explain_limit_input", 5))
    if not query:
        st.warning("请先输入自然语言检索问题。")
        return
    try:
        with st.spinner("正在执行意图解析、检索、query-paper 匹配与重排..."):
            payload = run_project_chain_session(
                query=query,
                follow_up_reply=follow_up or None,
                top_k=top_k,
                candidate_pool_size=candidate_pool,
                explain_limit=max(top_k, explain_limit),
            )
    except Exception as exc:
        st.session_state.pop("latest_payload", None)
        st.error(str(exc) or "系统执行失败，请检查数据库占用或 LLM 配置。")
        return
    st.session_state["latest_payload"] = payload


def main() -> None:
    st.set_page_config(page_title="PaperCompass", layout="wide")
    st.title("PaperCompass")
    st.caption(f"{DATASET_LABEL} | 统一意图驱动的论文检索系统")

    if not project_database_exists():
        st.error("项目数据库不存在，请先在 `tools/` 目录运行 `python papercompass.py build`。")
        return

    stats = load_project_stats()
    standard_queries = load_standard_queries()
    show_raw_json = render_sidebar(stats, standard_queries)
    render_overview(stats)

    app_state = load_app_state()
    if app_state:
        st.caption("已从 app_state.json 载入运行时状态")

    st.subheader("检索输入")
    st.caption(
        f"当前数据库：{relative_to_project(get_default_db_path())} | "
        f"论文数：{stats.get('papers', 0)} | "
        f"LLM：{'已配置 API Key' if OPENAI_API_KEY else '未配置 API Key'}"
    )
    query_col, config_col = st.columns([3, 2])
    with query_col:
        st.text_area(
            "自然语言查询",
            key="query_input",
            height=110,
            placeholder="例如：帮我找最近两年的 RAG 综述，并解释为什么推荐这些论文",
        )
        st.text_area(
            "可选补充回复",
            key="follow_up_input",
            height=80,
            placeholder="例如：最近两年，综述优先，方法不限",
        )
    with config_col:
        st.number_input("Top-K", min_value=3, max_value=10, value=5, step=1, key="top_k_input")
        st.number_input("候选池大小", min_value=20, max_value=120, value=40, step=10, key="candidate_pool_input")
        st.number_input("解释生成上限", min_value=3, max_value=10, value=5, step=1, key="explain_limit_input")
        st.button("开始检索", type="primary", use_container_width=True, on_click=run_query)

    payload = st.session_state.get("latest_payload")
    if not payload:
        st.info("运行一次查询后，可在这里查看完整系统链路。")
        render_management_area(standard_queries)
        return

    intent_col, gap_col = st.columns(2)
    with intent_col:
        render_intent_panel(payload["final_intent_frame"])
    with gap_col:
        render_gap_panel(
            payload["intent_gap_report"],
            clarification_needed=bool(payload.get("final_intent_frame", {}).get("clarification_needed")),
        )

    st.subheader("检索结果")
    st.caption(
        f"候选池={payload.get('candidate_pool_size', 0)} | "
        f"返回结果={len(payload.get('top_k_results', []))} | "
        f"历史记录 ID={payload.get('history_id')}"
    )
    saved_ids = set(get_saved_paper_ids())
    if not payload.get("top_k_results"):
        st.warning("当前查询没有返回可展示的排序结果。")
    else:
        for result in payload["top_k_results"]:
            evidence_pack = payload.get("paper_evidence_packs", {}).get(result["paper_id"], {})
            render_result_card(result, evidence_pack, saved_ids, show_raw_json)

    if show_raw_json:
        with st.expander("完整链路 JSON"):
            st.json(payload)

    render_management_area(standard_queries)


if __name__ == "__main__":
    main()
