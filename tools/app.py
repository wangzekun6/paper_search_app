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


LANGUAGE_OPTIONS = {
    "zh": "中文",
    "en": "English",
}

SEARCH_SCENE_LABELS = {
    "topic_exploration": {"zh": "主题探索", "en": "Topic Exploration"},
    "survey_lookup": {"zh": "综述查找", "en": "Survey Lookup"},
    "recent_progress": {"zh": "近期进展", "en": "Recent Progress"},
    "specific_paper_lookup": {"zh": "特定论文定位", "en": "Specific Paper Lookup"},
    "author_trace": {"zh": "作者追踪", "en": "Author Trace"},
    "method_constrained_search": {"zh": "方法约束检索", "en": "Method-Constrained Search"},
}

PAPER_TYPE_LABELS = {
    "survey": {"zh": "综述", "en": "Survey"},
    "benchmark": {"zh": "基准/评测", "en": "Benchmark / Evaluation"},
    "method": {"zh": "方法论文", "en": "Method Paper"},
    "empirical_study": {"zh": "实证研究", "en": "Empirical Study"},
    "application_study": {"zh": "应用研究", "en": "Application Study"},
    "theory": {"zh": "理论研究", "en": "Theory"},
    "analysis": {"zh": "分析论文", "en": "Analysis Paper"},
}

PREFERENCE_LABELS = {
    "yes": {"zh": "是", "en": "Yes"},
    "no": {"zh": "否", "en": "No"},
}

DIMENSION_LABELS = {
    "scene_match": {"zh": "检索场景匹配", "en": "Search Scene Match"},
    "topic_match": {"zh": "主题匹配", "en": "Topic Match"},
    "constraint_match": {"zh": "技术约束匹配", "en": "Constraint Match"},
    "paper_type_match": {"zh": "论文类型匹配", "en": "Paper Type Match"},
    "time_preference_match": {"zh": "时间偏好匹配", "en": "Time Preference Match"},
    "survey_preference_match": {"zh": "综述偏好匹配", "en": "Survey Preference Match"},
}

FIELD_LABELS = {
    "title": {"zh": "标题", "en": "Title"},
    "authors": {"zh": "作者", "en": "Authors"},
    "abstract": {"zh": "摘要", "en": "Abstract"},
    "section_titles": {"zh": "章节标题", "en": "Section Titles"},
    "section_snippet": {"zh": "章节片段", "en": "Section Snippet"},
}

MATCH_TYPE_LABELS = {
    "title_hint": {"zh": "标题线索命中", "en": "Title Hint Match"},
    "author_match": {"zh": "作者命中", "en": "Author Match"},
    "phrase_match": {"zh": "短语命中", "en": "Phrase Match"},
    "fts": {"zh": "全文检索召回", "en": "FTS Recall"},
}

UI_TEXT = {
    "page_caption": {"zh": "{dataset} | 基于用户意图驱动的论文检索系统", "en": "{dataset} | Unified intent-driven paper retrieval system"},
    "language_label": {"zh": "界面语言", "en": "Interface Language"},
    "system_status": {"zh": "系统状态", "en": "System Status"},
    "state_file": {"zh": "状态文件：{path}", "en": "State file: {path}"},
    "status_summary": {"zh": "论文={papers} | 章节={sections} | FTS={fts_rows} | 语义卡={semantic_cards} | 历史={intent_histories} | 收藏={saved_papers}", "en": "Papers={papers} | Sections={sections} | FTS={fts_rows} | Semantic Cards={semantic_cards} | History={intent_histories} | Saved={saved_papers}"},
    "llm_runtime": {"zh": "LLM 运行状态", "en": "LLM Runtime"},
    "api_key_detected": {"zh": "已检测到 API Key", "en": "API key detected"},
    "api_key_missing": {"zh": "缺少 API Key，意图解析和 query-paper 匹配将无法正常工作。", "en": "API key is missing. Intent parsing and query-paper matching will not work correctly."},
    "base_url": {"zh": "基础地址：{value}", "en": "Base URL: {value}"},
    "model_label": {"zh": "模型：{value}", "en": "Model: {value}"},
    "test_api": {"zh": "测试 API", "en": "Test API"},
    "testing_api": {"zh": "正在测试 API...", "en": "Testing API..."},
    "demo_replay": {"zh": "示例回放", "en": "Demo Replay"},
    "demo_button": {"zh": "示例 {index}", "en": "Demo {index}"},
    "show_raw_json": {"zh": "显示原始 JSON", "en": "Show Raw JSON"},
    "papers_metric": {"zh": "论文数", "en": "Papers"},
    "sections_metric": {"zh": "章节数", "en": "Sections"},
    "fts_metric": {"zh": "FTS 行数", "en": "FTS Rows"},
    "semantic_cards_metric": {"zh": "语义卡片", "en": "Semantic Cards"},
    "history_metric": {"zh": "检索历史", "en": "Search History"},
    "saved_metric": {"zh": "收藏论文", "en": "Saved Papers"},
    "current_intent": {"zh": "当前意图", "en": "Current Intent"},
    "search_scene": {"zh": "检索场景", "en": "Search Scene"},
    "research_topic": {"zh": "研究主题", "en": "Research Topic"},
    "topic_keywords": {"zh": "主题关键词", "en": "Topic Keywords"},
    "technical_constraints": {"zh": "技术约束", "en": "Technical Constraints"},
    "method": {"zh": "方法", "en": "Method"},
    "model_family": {"zh": "模型族", "en": "Model Family"},
    "dataset": {"zh": "数据集", "en": "Dataset"},
    "metric": {"zh": "指标", "en": "Metric"},
    "modality": {"zh": "模态", "en": "Modality"},
    "document_attributes": {"zh": "文档属性", "en": "Document Attributes"},
    "time_range": {"zh": "时间范围", "en": "Time Range"},
    "paper_type": {"zh": "论文类型", "en": "Paper Type"},
    "author_name": {"zh": "作者", "en": "Author"},
    "title_hint": {"zh": "标题线索", "en": "Title Hint"},
    "result_preferences": {"zh": "结果偏好", "en": "Result Preferences"},
    "prefer_recent": {"zh": "偏好最新", "en": "Prefer Recent"},
    "prefer_classic": {"zh": "偏好经典", "en": "Prefer Classic"},
    "prefer_survey": {"zh": "偏好综述", "en": "Prefer Survey"},
    "prefer_diverse": {"zh": "偏好多样", "en": "Prefer Diverse"},
    #"need_explainable_reason": {"zh": "需要解释", "en": "Need Explanation"},
    "intentframe_raw_json": {"zh": "IntentFrame 原始 JSON", "en": "IntentFrame Raw JSON"},
    "gap_analysis": {"zh": "Gap 分析", "en": "Gap Analysis"},
    "missing_slots": {"zh": "缺失槽位", "en": "Missing Slots"},
    "no_missing_slots": {"zh": "当前没有缺失槽位", "en": "No missing slots currently."},
    "evidence_gap": {"zh": "证据缺口", "en": "Evidence Gaps"},
    "no_evidence_gap": {"zh": "当前没有明显证据缺口", "en": "No obvious evidence gaps currently."},
    "matched_dimensions": {"zh": "已命中维度", "en": "Matched Dimensions"},
    "no_matched_dimensions": {"zh": "当前还没有稳定命中的维度", "en": "No stable matched dimensions yet."},
    "why_results_broad": {"zh": "当前结果为何较宽", "en": "Why Current Results Are Broad"},
    "results_focused": {"zh": "当前结果已经比较聚焦", "en": "Current results are already fairly focused."},
    "next_answer_helpful": {"zh": "继续补充什么最有帮助", "en": "What Further Input Would Help Most"},
    "results_usable": {"zh": "当前结果已经具备可用性", "en": "Current results are already usable."},
    "follow_up_entry": {"zh": "追问回复入口", "en": "Follow-up Reply"},
    "clarification_needed_caption": {"zh": "系统仍有未补齐信息，建议在这里回复后直接继续检索。", "en": "The system still needs missing information. Reply here to continue the search directly."},
    "clarification_optional_caption": {"zh": "如果你想进一步收敛结果，可继续补充偏好或约束。", "en": "Add more preferences or constraints here if you want narrower results."},
    "supplementary_reply": {"zh": "补充回复", "en": "Supplementary Reply"},
    "supplementary_reply_placeholder": {"zh": "例如：只看最近两年，优先综述，并说明每篇论文为何匹配", "en": "Example: Only papers from the last two years, surveys first, and explain why each paper matches."},
    "continue_search": {"zh": "按追问继续检索", "en": "Continue Search"},
    "fill_reply_warning": {"zh": "请先填写补充回复。", "en": "Enter a follow-up reply first."},
    "save_paper": {"zh": "收藏论文", "en": "Save Paper"},
    "unsave_paper": {"zh": "取消收藏", "en": "Unsave"},
    "result_summary": {"zh": "{authors} | {year_month} | 总分={final_score:.3f} | 匹配分={match_score:.3f}", "en": "{authors} | {year_month} | Final={final_score:.3f} | Match={match_score:.3f}"},
    "ranking_reasons": {"zh": "排序理由", "en": "Ranking Reasons"},
    "no_ranking_reasons": {"zh": "暂无排序理由", "en": "No ranking reasons available."},
    "unmet_constraints": {"zh": "未满足约束", "en": "Unmet Constraints"},
    "no_unmet_constraints": {"zh": "暂无未满足约束", "en": "No unmet constraints."},
    "abstract": {"zh": "摘要", "en": "Abstract"},
    "matched_sections": {"zh": "命中章节", "en": "Matched Sections"},
    "no_matched_sections": {"zh": "暂无命中章节", "en": "No matched sections."},
    "matched_snippets": {"zh": "命中片段", "en": "Matched Snippets"},
    "no_matched_snippets": {"zh": "暂无命中片段", "en": "No matched snippets."},
    "ranking_explanation": {"zh": "排序解释", "en": "Ranking Explanation"},
    "no_match_explanation": {"zh": "暂无 query-paper 匹配解释", "en": "No query-paper match explanation available."},
    "matched_field_label": {"zh": "命中字段：{value}", "en": "Matched Field: {value}"},
    "matched_type_label": {"zh": "命中类型：{value}", "en": "Matched Type: {value}"},
    "semantic_card": {"zh": "语义卡片", "en": "Semantic Card"},
    "query_paper_match": {"zh": "Query-论文匹配", "en": "Query-Paper Match"},
    "raw_result_json": {"zh": "结果原始 JSON", "en": "Raw Result JSON"},
    "management_area": {"zh": "管理区", "en": "Management"},
    "saved_papers_tab": {"zh": "收藏论文", "en": "Saved Papers"},
    "history_tab": {"zh": "历史记录", "en": "History"},
    "standard_demos_tab": {"zh": "标准示例", "en": "Standard Demos"},
    "no_saved_papers": {"zh": "当前还没有收藏论文。", "en": "No saved papers yet."},
    "no_history": {"zh": "当前还没有检索历史。", "en": "No search history yet."},
    "follow_up_reply_label": {"zh": "补充回复：{value}", "en": "Follow-up: {value}"},
    "replay": {"zh": "回放", "en": "Replay"},
    "enter_query_first": {"zh": "请先输入自然语言检索问题。", "en": "Enter a natural-language query first."},
    "running_pipeline": {"zh": "正在执行意图解析、检索、query-paper 匹配与重排...", "en": "Running intent parsing, retrieval, query-paper matching, and reranking..."},
    "run_failed": {"zh": "系统执行失败，请检查数据库占用或 LLM 配置。", "en": "Execution failed. Check database locks or LLM configuration."},
    "database_missing": {"zh": "项目数据库不存在，请先在 `tools/` 目录运行 `python papercompass.py build`。", "en": "Project database does not exist. Run `python papercompass.py build` in `tools/` first."},
    "database_empty": {"zh": "当前数据库文件存在，但论文数为 0。请先在 `tools/` 目录运行 `python papercompass.py build`。", "en": "The current database file exists but contains zero papers. Run `python papercompass.py build` in `tools/` first."},
    "current_database": {"zh": "当前数据库：{path}", "en": "Current database: {path}"},
    "state_loaded": {"zh": "已从 app_state.json 载入运行时状态", "en": "Runtime state loaded from app_state.json"},
    "search_input": {"zh": "检索输入", "en": "Search Input"},
    "search_status": {"zh": "当前数据库：{db_path} | 论文数：{papers} | LLM：{llm_status}", "en": "Database: {db_path} | Papers: {papers} | LLM: {llm_status}"},
    "api_key_configured": {"zh": "已配置 API Key", "en": "API key configured"},
    "api_key_unconfigured": {"zh": "未配置 API Key", "en": "API key not configured"},
    "natural_language_query": {"zh": "自然语言查询", "en": "Natural-Language Query"},
    "query_placeholder": {"zh": "例如：帮我找最近两年的 RAG 综述，并解释为什么推荐这些论文", "en": "Example: Find RAG survey papers from the last two years and explain why they are recommended."},
    "optional_follow_up": {"zh": "可选补充回复", "en": "Optional Follow-up"},
    "follow_up_placeholder": {"zh": "例如：最近两年，综述优先，方法不限", "en": "Example: Last two years, surveys first, no method restrictions."},
    "candidate_pool_size": {"zh": "候选池大小", "en": "Candidate Pool Size"},
    "explain_limit": {"zh": "解释生成上限", "en": "Explanation Limit"},
    "run_search": {"zh": "开始检索", "en": "Run Search"},
    "query_once_info": {"zh": "运行一次查询后，可在这里查看完整系统链路。", "en": "Run a query once to inspect the full pipeline here."},
    "search_results": {"zh": "检索结果", "en": "Search Results"},
    "results_caption": {"zh": "候选池={candidate_pool} | 返回结果={result_count} | 历史记录 ID={history_id}", "en": "Candidate Pool={candidate_pool} | Results={result_count} | History ID={history_id}"},
    "no_results": {"zh": "当前查询没有返回可展示的排序结果。", "en": "This query returned no ranked results to display."},
    "full_pipeline_json": {"zh": "完整链路 JSON", "en": "Full Pipeline JSON"},
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def current_locale() -> str:
    return st.session_state.get("ui_language", "zh")


def t(key: str, **kwargs: Any) -> str:
    entry = UI_TEXT.get(key, {})
    if isinstance(entry, dict):
        text = entry.get(current_locale()) or entry.get("zh") or key
    else:
        text = str(entry or key)
    return text.format(**kwargs) if kwargs else text


def translate_mapping_value(value: Any, mapping: Dict[str, Dict[str, str]]) -> str:
    text = clean_text(value)
    if not text:
        return ""
    label = mapping.get(text)
    if not label:
        return text
    return label.get(current_locale()) or label.get("zh") or text


def localize_slot_value(value: Any, slot_key: str = "") -> str:
    text = clean_text(value)
    if not text:
        return ""
    if slot_key == "search_scene":
        return translate_mapping_value(text, SEARCH_SCENE_LABELS)
    if slot_key == "document_attributes.paper_type":
        return translate_mapping_value(text, PAPER_TYPE_LABELS)
    if slot_key.startswith("result_preferences."):
        return translate_mapping_value(text, PREFERENCE_LABELS)
    return text


def localize_dimension_text(value: Any) -> str:
    return translate_mapping_value(value, DIMENSION_LABELS)


def localize_field_text(value: Any) -> str:
    return translate_mapping_value(value, FIELD_LABELS)


def localize_match_type_text(value: Any) -> str:
    return translate_mapping_value(value, MATCH_TYPE_LABELS)


def slot_display(slot: Dict[str, Any], slot_key: str = "") -> str:
    value = slot.get("value")
    if isinstance(value, list):
        normalized = [localize_slot_value(item, slot_key) for item in value if clean_text(item)]
        separator = "、" if current_locale() == "zh" else ", "
        return separator.join(item for item in normalized if item) or "-"
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
        current_value = st.session_state.get("ui_language", "zh")
        language_codes = list(LANGUAGE_OPTIONS.keys())
        default_index = language_codes.index(current_value) if current_value in language_codes else 0
        st.selectbox(
            t("language_label"),
            options=language_codes,
            index=default_index,
            key="ui_language",
            format_func=lambda code: LANGUAGE_OPTIONS.get(code, code),
        )

        st.subheader(t("system_status"))
        st.caption(DATASET_LABEL)
        st.code(DATASET_RELATIVE_PATH)
        st.code(relative_to_project(get_default_db_path()))
        st.caption(t("state_file", path=relative_to_project(APP_STATE_PATH)))
        st.caption(
            t(
                "status_summary",
                papers=stats["papers"],
                sections=stats["sections"],
                fts_rows=stats["fts_rows"],
                semantic_cards=stats["semantic_cards"],
                intent_histories=stats["intent_histories"],
                saved_papers=stats["saved_papers"],
            )
        )

        st.subheader(t("llm_runtime"))
        if OPENAI_API_KEY:
            st.success(t("api_key_detected"))
        else:
            st.error(t("api_key_missing"))
        st.caption(t("base_url", value=OPENAI_API_BASE))
        st.caption(t("model_label", value=OPENAI_MODEL))
        if st.button(t("test_api"), use_container_width=True):
            with st.spinner(t("testing_api")):
                ok, message = test_openai_api(OPENAI_API_KEY)
            if ok:
                st.success(message)
            else:
                st.error(message)

        st.subheader(t("demo_replay"))
        for index, item in enumerate(standard_queries[:6], start=1):
            if st.button(t("demo_button", index=index), key=f"demo_{index}", use_container_width=True):
                apply_demo_query(item)

        return st.checkbox(t("show_raw_json"), value=False)


def render_overview(stats: Dict[str, int]) -> None:
    columns = st.columns(6)
    metrics = [
        (t("papers_metric"), stats["papers"]),
        (t("sections_metric"), stats["sections"]),
        (t("fts_metric"), stats["fts_rows"]),
        (t("semantic_cards_metric"), stats["semantic_cards"]),
        (t("history_metric"), stats["intent_histories"]),
        (t("saved_metric"), stats["saved_papers"]),
    ]
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


def render_intent_panel(frame: Dict[str, Any]) -> None:
    st.subheader(t("current_intent"))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{t('search_scene')}**")
        st.write(slot_display(frame.get("search_scene", {}), "search_scene"))
        st.markdown(f"**{t('research_topic')}**")
        st.write(
            " | ".join(
                [
                    slot_display(frame["research_topic"]["domain"]),
                    slot_display(frame["research_topic"]["task"]),
                    slot_display(frame["research_topic"]["problem"]),
                ]
            )
        )
        st.markdown(f"**{t('topic_keywords')}**")
        st.write(slot_display(frame["research_topic"]["keywords"]))
    with col2:
        st.markdown(f"**{t('technical_constraints')}**")
        for label_key, slot in [
            ("method", frame["technical_constraints"]["method"]),
            ("model_family", frame["technical_constraints"]["model_family"]),
            ("dataset", frame["technical_constraints"]["dataset"]),
            ("metric", frame["technical_constraints"]["metric"]),
            ("modality", frame["technical_constraints"]["modality"]),
        ]:
            st.write(f"{t(label_key)}: {slot_display(slot)}")

        st.markdown(f"**{t('document_attributes')}**")
        for label_key, slot, slot_key in [
            ("time_range", frame["document_attributes"]["time_range"], ""),
            ("paper_type", frame["document_attributes"]["paper_type"], "document_attributes.paper_type"),
            ("author_name", frame["document_attributes"]["author_name"], ""),
            ("title_hint", frame["document_attributes"]["title_hint"], ""),
        ]:
            st.write(f"{t(label_key)}: {slot_display(slot, slot_key)}")

        st.markdown(f"**{t('result_preferences')}**")
        for label_key, slot, slot_key in [
            ("prefer_recent", frame["result_preferences"]["prefer_recent"], "result_preferences.prefer_recent"),
            ("prefer_classic", frame["result_preferences"]["prefer_classic"], "result_preferences.prefer_classic"),
            ("prefer_survey", frame["result_preferences"]["prefer_survey"], "result_preferences.prefer_survey"),
            ("prefer_diverse", frame["result_preferences"]["prefer_diverse"], "result_preferences.prefer_diverse"),
            (
                "need_explainable_reason",
                frame["result_preferences"]["need_explainable_reason"],
                "result_preferences.need_explainable_reason",
            ),
        ]:
            st.write(f"{t(label_key)}: {slot_display(slot, slot_key)}")

    if frame.get("clarification_needed") and frame.get("clarification_question"):
        st.warning(frame["clarification_question"])

    with st.expander(t("intentframe_raw_json")):
        st.json(frame)


def render_gap_panel(gap_report: Dict[str, Any], clarification_needed: bool) -> None:
    st.subheader(t("gap_analysis"))
    render_string_list(t("missing_slots"), gap_report.get("query_gap", []), t("no_missing_slots"))
    render_string_list(t("evidence_gap"), gap_report.get("evidence_gap", []), t("no_evidence_gap"))
    render_string_list(
        t("matched_dimensions"),
        gap_report.get("matched_dimensions", []),
        t("no_matched_dimensions"),
        formatter=localize_dimension_text,
    )
    render_string_list(
        t("why_results_broad"),
        gap_report.get("why_current_results_are_broad", []),
        t("results_focused"),
    )
    render_string_list(
        t("next_answer_helpful"),
        gap_report.get("what_next_answer_would_improve", []),
        t("results_usable"),
    )
    st.markdown(f"**{t('follow_up_entry')}**")
    if clarification_needed:
        st.caption(t("clarification_needed_caption"))
    else:
        st.caption(t("clarification_optional_caption"))
    st.text_area(
        t("supplementary_reply"),
        key="gap_follow_up_input",
        height=80,
        placeholder=t("supplementary_reply_placeholder"),
    )
    if st.button(t("continue_search"), key="run_gap_follow_up", use_container_width=True):
        reply = clean_text(st.session_state.get("gap_follow_up_input", ""))
        if not reply:
            st.warning(t("fill_reply_warning"))
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
    label = t("unsave_paper") if result["paper_id"] in saved_ids else t("save_paper")
    with st.expander(f"{result['title']}", expanded=False):
        header_left, header_right = st.columns([4, 1])
        with header_left:
            st.caption(
                t(
                    "result_summary",
                    authors=result.get("authors_raw", ""),
                    year_month=result.get("year_month", ""),
                    final_score=result.get("final_score", 0),
                    match_score=query_match.get("match_score", 0),
                )
            )
        with header_right:
            if st.button(label, key=f"save_toggle_{result['paper_id']}", use_container_width=True):
                toggle_saved_state(result["paper_id"], saved_ids)

        render_string_list(t("ranking_reasons"), result.get("ranking_reasons", []), t("no_ranking_reasons"))
        render_string_list(
            t("matched_dimensions"),
            matched_dimensions,
            t("no_matched_dimensions"),
            formatter=localize_dimension_text,
        )
        render_string_list(t("unmet_constraints"), result.get("unmet_constraints", []), t("no_unmet_constraints"))

        st.markdown(f"**{t('abstract')}**")
        st.write(result.get("abstract", ""))

        render_string_list(t("matched_sections"), evidence_pack.get("matched_sections", []), t("no_matched_sections"))
        render_string_list(
            t("matched_snippets"),
            [
                f"{localize_field_text(item.get('field', ''))}: {item.get('snippet', '')}"
                for item in evidence_pack.get("matched_snippets", [])
            ],
            t("no_matched_snippets"),
        )

        st.markdown(f"**{t('ranking_explanation')}**")
        st.write(query_match.get("brief_reason", t("no_match_explanation")))

        retrieval_summary = []
        if result.get("matched_field"):
            retrieval_summary.append(t("matched_field_label", value=localize_field_text(result.get("matched_field", ""))))
        if result.get("exact_match_type"):
            retrieval_summary.append(
                t("matched_type_label", value=localize_match_type_text(result.get("exact_match_type", "")))
            )
        if retrieval_summary:
            st.caption(" | ".join(retrieval_summary))

        with st.expander(t("semantic_card")):
            st.json(evidence_pack.get("semantic_card", {}))

        with st.expander(t("query_paper_match")):
            st.json(query_match)

        if show_raw_json:
            with st.expander(t("raw_result_json")):
                st.json(result)


def render_management_area(standard_queries: List[Dict[str, Any]]) -> None:
    st.subheader(t("management_area"))
    saved_tab, history_tab, demos_tab = st.tabs([t("saved_papers_tab"), t("history_tab"), t("standard_demos_tab")])

    with saved_tab:
        saved_items = list_saved_papers(limit=50)
        if not saved_items:
            st.info(t("no_saved_papers"))
        else:
            for item in saved_items:
                st.write(f"- {item['title']} | {item['authors_raw']} | {item['year_month']}")

    with history_tab:
        history_items = list_search_history(limit=20)
        if not history_items:
            st.info(t("no_history"))
        else:
            for item in history_items:
                st.write(f"- {item['created_at']} | {item['query_text']}")

    with demos_tab:
        for index, item in enumerate(standard_queries, start=1):
            col_query, col_action = st.columns([5, 1])
            with col_query:
                st.write(f"{index}. {item['query']}")
                if item.get("follow_up_reply"):
                    st.caption(t("follow_up_reply_label", value=item["follow_up_reply"]))
            with col_action:
                if st.button(t("replay"), key=f"replay_demo_{index}", use_container_width=True):
                    apply_demo_query(item)
                    run_query()


def run_query() -> None:
    query = clean_text(st.session_state.get("query_input", ""))
    follow_up = clean_text(st.session_state.get("follow_up_input", ""))
    top_k = int(st.session_state.get("top_k_input", 5))
    candidate_pool = int(st.session_state.get("candidate_pool_input", 40))
    explain_limit = int(st.session_state.get("explain_limit_input", 5))
    if not query:
        st.warning(t("enter_query_first"))
        return
    try:
        with st.spinner(t("running_pipeline")):
            payload = run_project_chain_session(
                query=query,
                follow_up_reply=follow_up or None,
                top_k=top_k,
                candidate_pool_size=candidate_pool,
                explain_limit=max(top_k, explain_limit),
            )
    except Exception as exc:
        st.session_state.pop("latest_payload", None)
        st.error(str(exc) or t("run_failed"))
        return
    st.session_state["latest_payload"] = payload


def main() -> None:
    st.session_state.setdefault("ui_language", "zh")
    st.set_page_config(page_title="PaperCompass", layout="wide")
    st.title("PaperCompass")
    st.caption(t("page_caption", dataset=DATASET_LABEL))

    if not project_database_exists():
        st.error(t("database_missing"))
        return

    stats = load_project_stats()
    if stats.get("papers", 0) <= 0:
        st.error(t("database_empty"))
        st.caption(t("current_database", path=relative_to_project(get_default_db_path())))
        return

    standard_queries = load_standard_queries()
    show_raw_json = render_sidebar(stats, standard_queries)
    render_overview(stats)

    app_state = load_app_state()
    if app_state:
        st.caption(t("state_loaded"))

    st.subheader(t("search_input"))
    st.caption(
        t(
            "search_status",
            db_path=relative_to_project(get_default_db_path()),
            papers=stats.get("papers", 0),
            llm_status=t("api_key_configured") if OPENAI_API_KEY else t("api_key_unconfigured"),
        )
    )
    query_col, config_col = st.columns([3, 2])
    with query_col:
        st.text_area(
            t("natural_language_query"),
            key="query_input",
            height=110,
            placeholder=t("query_placeholder"),
        )
        st.text_area(
            t("optional_follow_up"),
            key="follow_up_input",
            height=80,
            placeholder=t("follow_up_placeholder"),
        )
    with config_col:
        st.number_input("Top-K", min_value=3, max_value=10, value=5, step=1, key="top_k_input")
        st.number_input(t("candidate_pool_size"), min_value=20, max_value=120, value=40, step=10, key="candidate_pool_input")
        st.number_input(t("explain_limit"), min_value=3, max_value=10, value=5, step=1, key="explain_limit_input")
        st.button(t("run_search"), type="primary", use_container_width=True, on_click=run_query)

    payload = st.session_state.get("latest_payload")
    if not payload:
        st.info(t("query_once_info"))
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

    st.subheader(t("search_results"))
    st.caption(
        t(
            "results_caption",
            candidate_pool=payload.get("candidate_pool_size", 0),
            result_count=len(payload.get("top_k_results", [])),
            history_id=payload.get("history_id"),
        )
    )
    saved_ids = set(get_saved_paper_ids())
    if not payload.get("top_k_results"):
        st.warning(t("no_results"))
    else:
        for result in payload["top_k_results"]:
            evidence_pack = payload.get("paper_evidence_packs", {}).get(result["paper_id"], {})
            render_result_card(result, evidence_pack, saved_ids, show_raw_json)

    if show_raw_json:
        with st.expander(t("full_pipeline_json")):
            st.json(payload)

    render_management_area(standard_queries)


if __name__ == "__main__":
    main()
