"""
PaperCompass 的 Streamlit 可视化入口。

这个文件负责把检索主链路的输入、意图解析结果、Gap 分析、排序结果、
语义卡片以及收藏/历史管理整合成一个可交互页面。
"""

from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List

import streamlit as st
import streamlit.components.v1 as components

from papercompass_core.config import APP_STATE_PATH, get_active_dataset_info, relative_to_project
from papercompass_core.llm import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL, test_openai_api
from papercompass_core.services import (
    format_authors_for_display,
    get_default_db_path,
    get_saved_paper_ids,
    list_saved_papers,
    list_search_history,
    load_app_state,
    load_demo_queries,
    load_project_stats,
    project_database_exists,
    run_project_chain_session,
    save_paper,
    start_project_query_runtime_warmup,
    unsave_paper,
)


# 以下映射表负责把系统内部枚举值和字段名翻译成界面展示文案。
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

TIME_RANGE_LABELS = {
    "recent": {"zh": "最近", "en": "Recent"},
    "last 2 years": {"zh": "最近两年", "en": "Last 2 Years"},
    "last 3 years": {"zh": "最近三年", "en": "Last 3 Years"},
    "classic": {"zh": "经典时期", "en": "Classic Era"},
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

SLOT_PATH_LABELS = {
    "search_scene": {"zh": "检索场景", "en": "Search Scene"},
    "research_topic.domain": {"zh": "研究领域", "en": "Research Domain"},
    "research_topic.task": {"zh": "研究任务", "en": "Research Task"},
    "research_topic.problem": {"zh": "研究问题", "en": "Research Problem"},
    "research_topic.keywords": {"zh": "主题关键词", "en": "Topic Keywords"},
    "technical_constraints.method": {"zh": "方法约束", "en": "Method Constraint"},
    "technical_constraints.model_family": {"zh": "模型家族约束", "en": "Model Family Constraint"},
    "technical_constraints.dataset": {"zh": "数据集约束", "en": "Dataset Constraint"},
    "technical_constraints.metric": {"zh": "指标约束", "en": "Metric Constraint"},
    "technical_constraints.modality": {"zh": "模态约束", "en": "Modality Constraint"},
    "document_attributes.time_range": {"zh": "时间范围", "en": "Time Range"},
    "document_attributes.paper_type": {"zh": "论文类型", "en": "Paper Type"},
    "document_attributes.author_name": {"zh": "作者线索", "en": "Author Hint"},
    "document_attributes.title_hint": {"zh": "标题线索", "en": "Title Hint"},
    "result_preferences.prefer_recent": {"zh": "偏好最新", "en": "Prefer Recent"},
    "result_preferences.prefer_classic": {"zh": "偏好经典", "en": "Prefer Classic"},
    "result_preferences.prefer_survey": {"zh": "偏好综述", "en": "Prefer Survey"},
    "result_preferences.prefer_diverse": {"zh": "偏好多样", "en": "Prefer Diverse"},
    "result_preferences.need_explainable_reason": {"zh": "需要解释", "en": "Need Explanation"},
}

MATCH_TYPE_LABELS = {
    "title_hint": {"zh": "标题线索命中", "en": "Title Hint Match"},
    "author_match": {"zh": "作者命中", "en": "Author Match"},
    "phrase_match": {"zh": "短语命中", "en": "Phrase Match"},
    "fts": {"zh": "全文检索召回", "en": "FTS Recall"},
}

PIPELINE_STAGE_LABELS = {
    "intent_parse": {"zh": "LLM 意图解析", "en": "LLM Intent Parse"},
    "intent_follow_up_merge": {"zh": "追问合并", "en": "Follow-up Merge"},
    "retrieval_sparse": {"zh": "稀疏召回", "en": "Sparse Retrieval"},
    "retrieval_dense": {"zh": "稠密召回", "en": "Dense Retrieval"},
    "retrieval_exact": {"zh": "精确召回", "en": "Exact Retrieval"},
    "retrieval_fusion": {"zh": "候选融合", "en": "Candidate Fusion"},
    "candidate_rows_load": {"zh": "候选详情加载", "en": "Candidate Detail Load"},
    "semantic_card_backfill": {"zh": "语义卡补全", "en": "Semantic Card Backfill"},
    "query_paper_match": {"zh": "Query-Paper 匹配", "en": "Query-Paper Match"},
    "rerank_and_explain": {"zh": "重排与解释", "en": "Rerank and Explain"},
    "gap_report": {"zh": "Gap 分析", "en": "Gap Report"},
    "follow_up_suggestion": {"zh": "追问建议生成", "en": "Follow-up Suggestion"},
    "total": {"zh": "总耗时", "en": "Total Time"},
}

PIPELINE_STAGE_SEQUENCE = [
    "intent_parse",
    "intent_follow_up_merge",
    "retrieval_sparse",
    "retrieval_dense",
    "retrieval_exact",
    "retrieval_fusion",
    "candidate_rows_load",
    "semantic_card_backfill",
    "query_paper_match",
    "rerank_and_explain",
    "gap_report",
    "follow_up_suggestion",
    "total",
]

UI_TEXT = {
    "page_caption": {"zh": "{dataset} | 基于用户意图驱动的论文检索系统", "en": "{dataset} | Unified intent-driven paper retrieval system"},
    "workspace_settings": {"zh": "界面设置", "en": "View Settings"},
    "sidebar_hint": {"zh": "侧边栏仅保留界面设置；", "en": "The sidebar only keeps view settings. System status, LLM status, and demos now live on the main page."},
    "sidebar_demo_hint": {"zh": "点击左侧示例可直接回填到输入框", "en": "Click a sidebar demo to refill the inputs without running the search automatically."},
    "quick_navigation": {"zh": "步骤定位", "en": "Step Navigation"},
    "quick_navigation_hint": {"zh": "搜索完成后，可从这里一键跳到对应区域。", "en": "After a search completes, jump directly to the relevant section from here."},
    "quick_navigation_empty": {"zh": "完成一次检索后，这里会显示完整步骤定位。", "en": "Run one search to see the full section navigation here."},
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
    "run_process": {"zh": "运行过程", "en": "Process"},
    "run_process_details": {"zh": "查看完整运行过程", "en": "View Full Process"},
    "run_process_collapsed_hint": {"zh": "运行结束后，可在这里展开或收起阶段明细。", "en": "After the run finishes, you can expand or collapse the stage details here."},
    "process_live_title": {"zh": "链路执行中", "en": "Pipeline Running"},
    "process_done_title": {"zh": "本次链路已完成", "en": "Run Completed"},
    "process_current_stage": {"zh": "当前阶段：{value}", "en": "Current stage: {value}"},
    "process_last_stage": {"zh": "最近完成：{value}", "en": "Most recent stage: {value}"},
    "process_stage_progress": {"zh": "阶段进度：{completed}/{total}", "en": "Stage progress: {completed}/{total}"},
    "process_in_progress_kicker": {"zh": "处理中", "en": "Processing"},
    "process_live_summary": {
        "zh": "系统正在执行意图解析、候选召回、重排解释与结果整理，完成后会自动返回主工作台。",
        "en": "The system is parsing intent, recalling candidates, reranking, and organizing results. It will return to the main workspace automatically when finished.",
    },
    "process_completed_stages": {"zh": "已完成阶段", "en": "Completed Stages"},
    "process_remaining_stages": {"zh": "剩余阶段", "en": "Remaining Stages"},
    "process_current_progress_label": {"zh": "当前进度", "en": "Current Progress"},
    "process_live_tip": {
        "zh": "详细阶段记录会在检索完成后显示在下方运行过程区域。",
        "en": "Detailed stage logs will appear in the process panel below after the run finishes.",
    },
    "process_running_hint": {"zh": "进行中阶段会持续高亮闪动。", "en": "The active stage stays prominently animated while running."},
    "process_completed_hint": {"zh": "链路已跑完，可展开下方明细回看完整阶段。", "en": "The run is complete. Expand the details below to review all stages."},
    "process_preparing": {"zh": "准备执行检索链路", "en": "Preparing pipeline"},
    "stage_running": {"zh": "进行中", "en": "Running"},
    "stage_completed": {"zh": "已完成", "en": "Completed"},
    "stage_pending": {"zh": "待执行", "en": "Pending"},
    "model_workbench": {"zh": "模型工作台", "en": "Model Workbench"},
    "runtime_workspace_caption": {"zh": "这里集中展示链路阶段、模型使用、缓存命中和系统状态。", "en": "Pipeline stages, model usage, cache hits, and system status are centralized here."},
    "papers_metric": {"zh": "论文数", "en": "Papers"},
    "sections_metric": {"zh": "章节数", "en": "Sections"},
    "fts_metric": {"zh": "FTS 行数", "en": "FTS Rows"},
    "semantic_cards_metric": {"zh": "语义卡片", "en": "Semantic Cards"},
    "history_metric": {"zh": "检索历史", "en": "Search History"},
    "saved_metric": {"zh": "收藏论文", "en": "Saved Papers"},
    "current_intent": {"zh": "当前意图", "en": "Current Intent"},
    "system_understanding": {"zh": "系统理解了什么", "en": "What the System Understood"},
    "system_understanding_caption": {"zh": "先看系统抽取出的结构化意图，再决定要不要继续收窄。", "en": "Inspect the structured intent first, then decide whether to narrow the search further."},
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
    "need_explainable_reason": {"zh": "需要解释", "en": "Need Explanation"},
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
    "follow_up_workspace": {"zh": "还需要你补充什么", "en": "What Else You Should Add"},
    "follow_up_workspace_caption": {"zh": "优先补齐最影响排序收敛的信息，可一键填入系统建议。", "en": "Fill the information that most affects ranking convergence, and optionally insert the system suggestion with one click."},
    "follow_up_convergence": {"zh": "追问收敛摘要", "en": "Follow-up Convergence"},
    "follow_up_convergence_caption": {"zh": "这里显示追问后意图状态到底收紧了哪些地方。", "en": "This shows exactly how the intent state became more specific after the follow-up."},
    "slot_count_change": {"zh": "缺失槽位：{missing_before} -> {missing_after} | 已回答槽位：{answered_before} -> {answered_after}", "en": "Missing slots: {missing_before} -> {missing_after} | Answered slots: {answered_before} -> {answered_after}"},
    "newly_answered_slots": {"zh": "新确认的槽位", "en": "Newly Confirmed Slots"},
    "no_newly_answered_slots": {"zh": "这次追问没有新增已回答槽位。", "en": "This follow-up did not add new answered slots."},
    "query_variant_changes": {"zh": "Query 变体变化", "en": "Query Variant Changes"},
    "no_query_variant_changes": {"zh": "这次追问没有改写 query 变体。", "en": "This follow-up did not change the query variants."},
    "result_change_summary": {"zh": "结果变化摘要", "en": "Result Change Summary"},
    "result_change_caption": {"zh": "这里显示追问后 Top-K 是完全没变、只是重排了，还是出现了新候选。", "en": "This shows whether the Top-K stayed the same, only changed order, or introduced new candidates after the follow-up."},
    "result_overlap_summary": {"zh": "Top-K 重合度：{overlap_count}/{current_count} | 新增结果：{added_count} | 移出结果：{removed_count}", "en": "Top-K overlap: {overlap_count}/{current_count} | Added: {added_count} | Removed: {removed_count}"},
    "result_change_same_order": {"zh": "前后 Top-K 完全相同，当前追问主要改变了排序解释和 Gap 分析，没有带来新的候选。", "en": "The Top-K is exactly the same before and after. The follow-up mainly changed the ranking explanations and gap analysis, without introducing new candidates."},
    "result_change_reordered": {"zh": "前后 Top-K 候选集合相同，但顺序发生了变化。", "en": "The Top-K candidate set stayed the same, but the order changed."},
    "result_change_mixed": {"zh": "追问后 Top-K 出现了新候选，且有部分旧候选被替换。", "en": "The follow-up introduced new Top-K candidates and replaced some previous ones."},
    "result_change_only_added": {"zh": "追问后保留了已有候选，并额外引入了新的更匹配论文。", "en": "The follow-up kept the existing candidates and introduced additional better-matching papers."},
    "result_change_only_removed": {"zh": "追问后移除了部分不够匹配的旧候选，但没有引入新论文。", "en": "The follow-up removed some weaker previous candidates without adding new papers."},
    "new_results_added": {"zh": "新进入 Top-K 的论文", "en": "New Top-K Papers"},
    "no_new_results_added": {"zh": "这次追问没有引入新的 Top-K 论文。", "en": "This follow-up did not introduce new Top-K papers."},
    "results_removed": {"zh": "被挤出 Top-K 的论文", "en": "Papers Removed from Top-K"},
    "no_results_removed": {"zh": "这次追问没有移出原有 Top-K 论文。", "en": "This follow-up did not remove any previous Top-K papers."},
    "reordered_results": {"zh": "顺序发生变化的论文", "en": "Reordered Papers"},
    "no_reordered_results": {"zh": "这次追问没有改变共享候选的顺序。", "en": "This follow-up did not change the order of shared candidates."},
    "follow_up_entry": {"zh": "追问回复入口", "en": "Follow-up Reply"},
    "clarification_needed_caption": {"zh": "系统仍有未补齐信息，建议在这里回复后直接继续检索。", "en": "The system still needs missing information. Reply here to continue the search directly."},
    "clarification_optional_caption": {"zh": "如果你想进一步收敛结果，可继续补充偏好或约束。", "en": "Add more preferences or constraints here if you want narrower results."},
    "suggested_follow_up": {"zh": "建议直接补充这句话", "en": "Suggested Follow-up"},
    "fill_suggested_follow_up": {"zh": "一键填入建议追问", "en": "Fill Suggested Follow-up"},
    "gap_details": {"zh": "为什么系统还需要这些信息", "en": "Why the System Needs More Input"},
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
    "search_input_caption": {"zh": "先描述检索目标，再决定是否补充额外约束。", "en": "Describe the retrieval goal first, then decide whether to add extra constraints."},
    "query_placeholder": {"zh": "例如：帮我找最近两年的 RAG 综述，并解释为什么推荐这些论文", "en": "Example: Find RAG survey papers from the last two years and explain why they are recommended."},
    "optional_follow_up": {"zh": "可选补充回复", "en": "Optional Follow-up"},
    "follow_up_placeholder": {"zh": "例如：最近两年，综述优先，方法不限", "en": "Example: Last two years, surveys first, no method restrictions."},
    "search_config": {"zh": "检索配置", "en": "Search Configuration"},
    "candidate_pool_size": {"zh": "候选池大小", "en": "Candidate Pool Size"},
    "explain_limit": {"zh": "解释生成上限", "en": "Explanation Limit"},
    "run_search": {"zh": "开始检索", "en": "Run Search"},
    "query_once_info": {"zh": "运行一次查询后，可在这里查看完整系统链路。", "en": "Run a query once to inspect the full pipeline here."},
    "search_results": {"zh": "检索结果", "en": "Search Results"},
    "topk_recommendations": {"zh": "推荐结果 Top-K", "en": "Top-K Recommendations"},
    "topk_recommendations_caption": {"zh": "默认直接展示摘要、匹配理由和未满足约束，先看重点再决定是否深入。", "en": "Abstracts, match reasons, and unmet constraints are visible by default so you can scan the essentials first."},
    "paper_details": {"zh": "论文详情", "en": "Paper Details"},
    "paper_details_caption": {"zh": "这一节保留完整摘要、命中证据、语义卡和 query-paper match 细节。", "en": "This section keeps the full abstract, evidence, semantic cards, and query-paper match details."},
    "management_workspace": {"zh": "收藏 / 历史 / 示例", "en": "Saved / History / Demos"},
    "management_workspace_caption": {"zh": "收藏、历史和示例统一放到主线底部，避免打断当前检索。", "en": "Saved items, history, and demos live at the bottom so they do not interrupt the current retrieval flow."},
    "query_not_run_yet": {"zh": "运行一次查询后，这里会展示完整阶段事件和运行摘要。", "en": "Run a query once to see the full stage events and runtime summary here."},
    "abstract_preview": {"zh": "摘要预览", "en": "Abstract Preview"},
    "detail_expand_hint": {"zh": "完整论文详情见下一节。", "en": "See the next section for full paper details."},
    "results_caption": {"zh": "候选池={candidate_pool} | 返回结果={result_count} | 历史记录 ID={history_id}", "en": "Candidate Pool={candidate_pool} | Results={result_count} | History ID={history_id}"},
    "no_results": {"zh": "当前查询没有返回可展示的排序结果。", "en": "This query returned no ranked results to display."},
    "full_pipeline_json": {"zh": "完整链路 JSON", "en": "Full Pipeline JSON"},
    "runtime_summary": {"zh": "本次运行摘要", "en": "Run Summary"},
    "llm_usage_summary": {"zh": "是否经过 LLM：{value}", "en": "LLM used: {value}"},
    "models_used_summary": {"zh": "本次使用模型：{value}", "en": "Models used: {value}"},
    "cache_results_summary": {"zh": "来自缓存的结果：{value}", "en": "Cached results: {value}"},
    "no_cache_results": {"zh": "本次 Top-K 没有缓存结果", "en": "No cached Top-K results in this run."},
    "result_source_label": {"zh": "解释来源：{value}", "en": "Explanation source: {value}"},
    "result_model_label": {"zh": "解释模型：{value}", "en": "Explanation model: {value}"},
}

UI_TEXT.update(
    {
        "result_quality_delta": {
            "zh": "主意图满足 Top-K：{main_before} -> {main_after} | 平均匹配分：{match_before} -> {match_after} | 平均最终分：{final_before} -> {final_after}",
            "en": "Main-intent satisfied in Top-K: {main_before} -> {main_after} | Avg match score: {match_before} -> {match_after} | Avg final score: {final_before} -> {final_after}",
        },
        "result_convergence_improved": {
            "zh": "追问后结果更收敛：主意图满足 {main_before} -> {main_after}，平均匹配分 {match_before} -> {match_after}。",
            "en": "Results converged better after the follow-up: main-intent satisfied {main_before} -> {main_after}, average match score {match_before} -> {match_after}.",
        },
        "result_convergence_stable": {
            "zh": "追问后结果整体稳定：主意图满足 {main_before} -> {main_after}，平均匹配分 {match_before} -> {match_after}。",
            "en": "Results stayed broadly stable after the follow-up: main-intent satisfied {main_before} -> {main_after}, average match score {match_before} -> {match_after}.",
        },
        "result_convergence_weakened": {
            "zh": "追问后结果收敛减弱：主意图满足 {main_before} -> {main_after}，平均匹配分 {match_before} -> {match_after}。",
            "en": "Results converged less after the follow-up: main-intent satisfied {main_before} -> {main_after}, average match score {match_before} -> {match_after}.",
        },
    }
)

STEP_SECTION_ANCHORS = {
    1: "step-1-search-input",
    2: "step-2-runtime-workbench",
    3: "step-3-system-understanding",
    4: "step-4-follow-up",
    5: "step-5-topk-results",
    6: "step-6-paper-details",
    7: "step-7-management",
}
LIVE_PROCESS_SCROLL_ANCHOR = "live-process-top-animation"


# 统一清理字符串，避免界面上出现多余空白或 `None`。
def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def truncate_text(value: Any, limit: int = 320) -> str:
    text = clean_text(value)
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


# 当前界面语言统一从 session_state 中读取。
def current_locale() -> str:
    return st.session_state.get("ui_language", "zh")


# 国际化文案读取入口，所有界面文案都优先通过这里获取。
def t(key: str, **kwargs: Any) -> str:
    entry = UI_TEXT.get(key, {})
    if isinstance(entry, dict):
        text = entry.get(current_locale()) or entry.get("zh") or key
    else:
        text = str(entry or key)
    return text.format(**kwargs) if kwargs else text


def inject_runtime_process_styles() -> None:
    st.markdown(
        """
<style>
:root {
    --pc-bg: #f4f8ff;
    --pc-surface: rgba(255, 255, 255, 0.94);
    --pc-surface-soft: #f8fbff;
    --pc-border: #c9d8f1;
    --pc-border-strong: #9fb7dd;
    --pc-text: #17304f;
    --pc-muted: #5b6b80;
    --pc-blue: #2f5fa7;
    --pc-blue-soft: #eaf2ff;
    --pc-orange: #d9984d;
    --pc-orange-soft: #fff4e2;
    --pc-green: #1f8a5b;
    --pc-green-soft: #e9f8ef;
    --pc-shadow: 0 16px 42px rgba(25, 58, 112, 0.10);
}

.stApp {
    color: var(--pc-text);
    background:
        radial-gradient(circle at top left, rgba(47, 95, 167, 0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(217, 152, 77, 0.12), transparent 20%),
        linear-gradient(180deg, #f7faff 0%, #f1f6ff 100%);
}

.stApp,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stText"],
.stApp label,
.stApp p,
.stApp input,
.stApp textarea,
.stApp button,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
}

.stApp .material-symbols-rounded,
.stApp .material-symbols-outlined,
.stApp .material-icons,
.stApp [data-testid="stExpander"] summary span[aria-hidden="true"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.3rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(47, 95, 167, 0.14), transparent 22%),
        linear-gradient(180deg, #f7faff 0%, #eef4ff 100%);
    border-right: 1px solid rgba(201, 216, 241, 0.72);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 24px;
    border: 1px solid rgba(201, 216, 241, 0.90);
    background: var(--pc-surface);
    box-shadow: var(--pc-shadow);
    padding: 0.25rem 0.35rem;
}

div[data-testid="stMetric"] {
    border-radius: 20px;
    border: 1px solid rgba(201, 216, 241, 0.92);
    background: linear-gradient(180deg, #fbfdff 0%, #f4f8ff 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
    padding: 0.5rem 0.2rem;
}

div[data-testid="stMetricLabel"] p {
    color: var(--pc-muted);
    font-weight: 700;
}

div[data-testid="stMetricValue"] {
    color: var(--pc-text);
}

div[data-baseweb="textarea"] textarea,
div[data-baseweb="input"] input {
    border-radius: 18px !important;
    border: 1px solid rgba(201, 216, 241, 0.95) !important;
    background: #f9fbff !important;
    color: var(--pc-text) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75);
    padding-right: 3.2rem !important;
    line-height: 1.6 !important;
    min-height: 2.9rem;
}

div[data-baseweb="textarea"] textarea:focus,
div[data-baseweb="input"] input:focus {
    border-color: rgba(47, 95, 167, 0.70) !important;
    box-shadow: 0 0 0 1px rgba(47, 95, 167, 0.18) !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="popover"] input {
    border-radius: 16px !important;
    border-color: rgba(201, 216, 241, 0.95) !important;
    background: #f9fbff !important;
}

div[data-testid="stTextArea"] label p,
div[data-testid="stNumberInput"] label p,
div[data-testid="stSelectbox"] label p {
    font-weight: 700;
    color: var(--pc-text);
}

div[data-testid="stTextArea"] textarea {
    padding-top: 0.9rem !important;
    padding-bottom: 0.9rem !important;
}

[data-testid="InputInstructions"] {
    display: none !important;
}

.stButton > button {
    border-radius: 16px;
    border: 1px solid rgba(201, 216, 241, 0.92);
    background: linear-gradient(180deg, #ffffff 0%, #edf4ff 100%);
    color: var(--pc-text);
    min-height: 44px;
    font-weight: 700;
    box-shadow: 0 10px 24px rgba(47, 95, 167, 0.08);
}

.stButton > button:hover {
    border-color: rgba(47, 95, 167, 0.72);
    color: var(--pc-blue);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #3a6dbc 0%, #2f5fa7 100%);
    color: white;
    border-color: rgba(47, 95, 167, 0.95);
}

div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 14px;
    border: 1px solid transparent;
    padding: 0.48rem 0.95rem;
    font-weight: 700;
    color: var(--pc-muted);
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: var(--pc-blue-soft);
    color: var(--pc-blue);
    border-color: rgba(201, 216, 241, 0.95);
}

details[data-testid="stExpander"] {
    border-radius: 20px;
    border: 1px solid rgba(201, 216, 241, 0.92);
    background: rgba(255, 255, 255, 0.82);
    overflow: hidden;
}

details[data-testid="stExpander"] summary {
    background: linear-gradient(180deg, rgba(248, 251, 255, 0.98), rgba(239, 245, 255, 0.98));
}

.pc-hero {
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    gap: 22px;
    align-items: flex-start;
    border: 1px solid rgba(201, 216, 241, 0.92);
    border-radius: 28px;
    padding: 24px 26px 22px;
    margin-bottom: 1.1rem;
    background:
        radial-gradient(circle at top right, rgba(217, 152, 77, 0.18), transparent 26%),
        radial-gradient(circle at left center, rgba(47, 95, 167, 0.14), transparent 22%),
        linear-gradient(135deg, rgba(255,255,255,0.98), rgba(244, 248, 255, 0.98));
    box-shadow: 0 22px 48px rgba(25, 58, 112, 0.11);
}

.pc-hero-copy {
    flex: 1;
    min-width: 0;
}

.pc-hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 7px 12px;
    background: var(--pc-orange-soft);
    color: #8b5a19;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.02em;
}

.pc-hero-title {
    margin: 14px 0 8px;
    font-size: 2rem;
    line-height: 1.16;
    color: var(--pc-text);
    font-weight: 900;
}

.pc-hero-subtitle {
    max-width: 860px;
    font-size: 1rem;
    line-height: 1.72;
    color: var(--pc-muted);
}

.pc-hero-meta {
    width: min(360px, 34%);
    min-width: 260px;
    display: grid;
    gap: 10px;
}

.pc-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 0.35rem 0 0.45rem;
}

.pc-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 0.82rem;
    font-weight: 700;
    border: 1px solid rgba(201, 216, 241, 0.9);
    background: var(--pc-blue-soft);
    color: var(--pc-blue);
}

.pc-chip.soft {
    background: #f7faff;
    color: var(--pc-text);
}

.pc-chip.good {
    background: var(--pc-green-soft);
    color: var(--pc-green);
    border-color: rgba(129, 206, 166, 0.95);
}

.pc-panel-lead {
    margin-bottom: 0.9rem;
}

.pc-panel-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 5px 11px;
    background: var(--pc-blue-soft);
    color: var(--pc-blue);
    font-size: 0.76rem;
    font-weight: 800;
    margin-bottom: 0.55rem;
}

.pc-panel-title {
    font-size: 1.26rem;
    line-height: 1.3;
    font-weight: 800;
    color: var(--pc-text);
    margin-bottom: 0.28rem;
}

.pc-panel-caption {
    color: var(--pc-muted);
    font-size: 0.92rem;
    line-height: 1.65;
}

.pc-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin: 0.4rem 0 0.6rem;
}

.pc-kpi-card {
    border-radius: 18px;
    border: 1px solid rgba(201, 216, 241, 0.96);
    background: linear-gradient(180deg, #fbfdff 0%, #f3f8ff 100%);
    padding: 14px 16px;
}

.pc-kpi-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--pc-muted);
    margin-bottom: 6px;
}

.pc-kpi-value {
    font-size: 1.8rem;
    font-weight: 900;
    color: var(--pc-text);
}

.pc-kv-list {
    display: grid;
    gap: 10px;
    margin-top: 0.45rem;
}

.pc-kv-item {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 12px;
    border-radius: 16px;
    background: #f8fbff;
    border: 1px solid rgba(201, 216, 241, 0.88);
    min-width: 0;
}

.pc-kv-key {
    color: var(--pc-muted);
    font-size: 0.86rem;
    font-weight: 700;
    flex: 0 0 100%;
}

.pc-kv-value {
    color: var(--pc-text);
    font-size: 0.9rem;
    font-weight: 700;
    text-align: left;
    width: 100%;
    min-width: 0;
    overflow-wrap: anywhere;
    word-break: break-word;
}

.pc-list-title {
    font-size: 0.98rem;
    font-weight: 800;
    color: var(--pc-text);
    margin: 0.25rem 0 0.4rem;
}

.pc-list-block {
    border-radius: 18px;
    border: 1px solid rgba(201, 216, 241, 0.88);
    background: #f8fbff;
    padding: 10px 14px 8px;
    margin-bottom: 0.7rem;
}

.pc-list-block ul {
    margin: 0;
    padding-left: 1.1rem;
}

.pc-list-block li {
    color: var(--pc-text);
    margin: 0.22rem 0;
    line-height: 1.65;
}

.pc-empty-note {
    border-radius: 16px;
    background: #f8fbff;
    border: 1px dashed rgba(201, 216, 241, 0.95);
    color: var(--pc-muted);
    padding: 10px 12px;
    margin-bottom: 0.7rem;
}

.pc-section-heading {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin: 1.2rem 0 0.75rem;
}

.pc-section-step {
    flex: 0 0 auto;
    border-radius: 18px;
    background: linear-gradient(180deg, #2f5fa7 0%, #3b73c4 100%);
    color: white;
    font-weight: 900;
    font-size: 0.95rem;
    padding: 8px 12px;
    min-width: 46px;
    text-align: center;
    box-shadow: 0 10px 24px rgba(47, 95, 167, 0.18);
}

.pc-section-title {
    font-size: 1.28rem;
    font-weight: 900;
    color: var(--pc-text);
    line-height: 1.3;
}

.pc-section-caption {
    font-size: 0.92rem;
    color: var(--pc-muted);
    line-height: 1.6;
    margin-top: 4px;
}

.pc-note {
    color: var(--pc-muted);
    font-size: 0.9rem;
    line-height: 1.65;
}

.pc-result-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.35rem;
    color: var(--pc-blue);
    font-size: 0.82rem;
    font-weight: 800;
}

.pc-inline-meta {
    color: var(--pc-muted);
    font-size: 0.9rem;
    line-height: 1.65;
    margin: 0.15rem 0 0.55rem;
}

@media (max-width: 960px) {
    .pc-hero {
        flex-direction: column;
    }
    .pc-hero-meta {
        width: 100%;
        min-width: 0;
    }
}

.pc-process-shell {
    border: 1px solid rgba(38, 70, 83, 0.18);
    border-radius: 20px;
    padding: 16px 18px 14px;
    background:
        radial-gradient(circle at top right, rgba(233, 196, 106, 0.18), transparent 36%),
        linear-gradient(135deg, rgba(248, 249, 250, 0.98), rgba(241, 243, 245, 0.95));
    box-shadow: 0 18px 38px rgba(38, 70, 83, 0.10);
    margin-bottom: 8px;
}
.pc-process-shell.is-live {
    border-color: rgba(42, 157, 143, 0.35);
    box-shadow: 0 28px 70px rgba(42, 157, 143, 0.22);
}
.pc-process-live-overlay {
    position: fixed !important;
    inset: 0 !important;
    z-index: 2147483000 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 24px !important;
    pointer-events: auto !important;
}
.pc-process-live-backdrop {
    position: absolute !important;
    inset: 0 !important;
    background:
        radial-gradient(circle at center, rgba(47, 95, 167, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(217, 152, 77, 0.16), transparent 22%),
        rgba(17, 31, 51, 0.30);
    backdrop-filter: blur(14px) saturate(116%);
    animation: pc-live-backdrop-in 0.22s ease-out;
}
.pc-process-live-dock {
    position: relative !important;
    width: min(520px, 84vw) !important;
    max-height: min(82vh, 520px) !important;
    z-index: 1 !important;
    pointer-events: auto !important;
    animation: pc-live-dock-enter 0.24s ease-out, pc-live-dock-float 2.1s ease-in-out 0.24s infinite;
}
.pc-process-live-dock::before {
    content: "";
    position: absolute;
    inset: -30px -34px -34px;
    border-radius: 36px;
    background:
        radial-gradient(circle at center, rgba(42, 157, 143, 0.28), transparent 54%),
        radial-gradient(circle at top right, rgba(244, 162, 97, 0.26), transparent 40%);
    filter: blur(22px);
    z-index: 0;
}
.pc-process-live-dock .pc-process-shell {
    position: relative;
    z-index: 1;
    margin-bottom: 0;
    border-color: rgba(42, 157, 143, 0.30);
    background:
        radial-gradient(circle at top right, rgba(233, 196, 106, 0.20), transparent 34%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.97), rgba(246, 248, 249, 0.95));
    backdrop-filter: blur(18px);
    box-shadow: 0 34px 88px rgba(19, 42, 51, 0.24);
    padding: 24px 24px 20px;
}
.pc-process-live-compact {
    display: grid;
    gap: 16px;
}
.pc-process-live-compact-head {
    display: flex;
    align-items: center;
    gap: 14px;
}
.pc-process-live-compact-copy {
    min-width: 0;
    flex: 1;
}
.pc-process-live-compact-kicker {
    color: #2a9d8f;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.pc-process-live-compact-title {
    color: #17304f;
    font-size: 1.2rem;
    line-height: 1.35;
    font-weight: 900;
    margin-bottom: 5px;
}
.pc-process-live-compact-text {
    color: #415a63;
    font-size: 0.9rem;
    line-height: 1.65;
}
.pc-process-live-dock .pc-process-title {
    font-size: 1.12rem;
}
.pc-process-live-dock .pc-process-chip {
    background: rgba(255, 255, 255, 0.82);
}
.pc-process-live-statgrid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
}
.pc-process-live-stat {
    border-radius: 18px;
    border: 1px solid rgba(201, 216, 241, 0.88);
    background: rgba(255,255,255,0.82);
    padding: 12px 12px 10px;
}
.pc-process-live-stat-label {
    color: #5b6b80;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.pc-process-live-stat-value {
    color: #17304f;
    font-size: 1.34rem;
    font-weight: 900;
    line-height: 1;
}
.pc-process-live-tip {
    border-radius: 18px;
    border: 1px dashed rgba(159, 183, 221, 0.95);
    background: rgba(248, 251, 255, 0.94);
    color: #415a63;
    font-size: 0.84rem;
    line-height: 1.62;
    padding: 12px 13px;
}
.pc-process-live-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2px;
}
.pc-process-live-orb {
    position: relative;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: linear-gradient(180deg, #2a9d8f 0%, #1f8a7c 100%);
    box-shadow: 0 0 0 0 rgba(42, 157, 143, 0.36);
    animation: pc-live-orb-pulse 1.35s ease-in-out infinite;
    flex: 0 0 auto;
}
.pc-process-live-orb::before,
.pc-process-live-orb::after {
    content: "";
    position: absolute;
    inset: -6px;
    border-radius: 50%;
    border: 1px solid rgba(42, 157, 143, 0.24);
    animation: pc-live-orb-ring 1.6s ease-out infinite;
}
.pc-process-live-orb::after {
    animation-delay: 0.55s;
}
.pc-process-banner {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 12px;
}
.pc-process-banner-copy {
    flex: 1;
    min-width: 0;
}
.pc-process-kicker {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2a9d8f;
    margin-bottom: 4px;
}
.pc-process-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #132a33;
    margin-bottom: 3px;
}
.pc-process-subtitle {
    font-size: 0.92rem;
    color: #264653;
}
.pc-process-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-end;
}
.pc-process-chip {
    border-radius: 999px;
    padding: 7px 12px;
    background: rgba(19, 42, 51, 0.06);
    color: #18323a;
    font-size: 0.82rem;
    font-weight: 600;
}
.pc-process-progress-track {
    width: 100%;
    height: 12px;
    border-radius: 999px;
    background: rgba(19, 42, 51, 0.08);
    overflow: hidden;
    position: relative;
    margin-bottom: 14px;
}
.pc-process-progress-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #2a9d8f 0%, #e9c46a 55%, #f4a261 100%);
    position: relative;
}
.pc-process-shell.is-live .pc-process-progress-fill::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.72) 45%, transparent 100%);
    animation: pc-process-sweep 1.5s linear infinite;
}
.pc-stage-list {
    display: grid;
    gap: 10px;
}
.pc-stage-card {
    position: relative;
    border-radius: 18px;
    padding: 12px 14px;
    border: 1px solid rgba(38, 70, 83, 0.12);
    background: rgba(255, 255, 255, 0.76);
    overflow: hidden;
    margin-left: 8px;
}
.pc-stage-card::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 5px;
    background: rgba(38, 70, 83, 0.14);
}
.pc-stage-card-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 6px;
    position: relative;
    padding-left: 4px;
}
.pc-stage-card-top::before {
    content: "";
    position: absolute;
    left: -19px;
    top: 2px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #d3dfef;
    box-shadow: 0 0 0 4px rgba(255,255,255,0.94);
}
.pc-stage-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #1f3f49;
}
.pc-stage-status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: rgba(38, 70, 83, 0.28);
}
.pc-stage-order {
    font-size: 0.78rem;
    font-weight: 700;
    color: rgba(19, 42, 51, 0.46);
}
.pc-stage-label {
    font-size: 0.96rem;
    font-weight: 700;
    color: #132a33;
    margin-bottom: 5px;
}
.pc-stage-meta-line {
    font-size: 0.82rem;
    line-height: 1.5;
    color: #415a63;
}
.pc-stage-card.is-completed {
    border-color: rgba(42, 157, 143, 0.24);
    background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(233, 250, 246, 0.92));
}
.pc-stage-card.is-completed::before {
    background: linear-gradient(180deg, #2a9d8f 0%, #8bd3c7 100%);
}
.pc-stage-card.is-completed .pc-stage-card-top::before {
    background: #2a9d8f;
}
.pc-stage-card.is-completed .pc-stage-status-dot {
    background: #2a9d8f;
}
.pc-stage-card.is-running {
    border-color: rgba(244, 162, 97, 0.55);
    background:
        linear-gradient(135deg, rgba(255,255,255,0.98), rgba(255, 246, 233, 0.98));
    box-shadow: 0 0 0 0 rgba(244, 162, 97, 0.36);
    animation: pc-stage-pulse 1.35s ease-in-out infinite;
}
.pc-stage-card.is-running::before {
    background: linear-gradient(180deg, #f4a261 0%, #e76f51 100%);
}
.pc-stage-card.is-running .pc-stage-card-top::before {
    background: #f4a261;
}
.pc-stage-card.is-running::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,0.78) 40%, transparent 75%);
    animation: pc-stage-shine 1.6s linear infinite;
}
.pc-stage-card.is-running .pc-stage-order {
    color: rgba(231, 111, 81, 0.78);
}
.pc-stage-card.is-running .pc-stage-status-dot {
    background: #f4a261;
    box-shadow: 0 0 0 0 rgba(244, 162, 97, 0.45);
    animation: pc-dot-pulse 1.25s ease-in-out infinite;
}
@keyframes pc-live-backdrop-in {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
@keyframes pc-live-dock-enter {
    0% { opacity: 0; transform: translateY(12px) scale(0.985); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes pc-live-dock-float {
    0% { transform: translateY(0); }
    50% { transform: translateY(-4px); }
    100% { transform: translateY(0); }
}
@keyframes pc-live-orb-pulse {
    0% { box-shadow: 0 0 0 0 rgba(42, 157, 143, 0.34); }
    70% { box-shadow: 0 0 0 14px rgba(42, 157, 143, 0); }
    100% { box-shadow: 0 0 0 0 rgba(42, 157, 143, 0); }
}
@keyframes pc-live-orb-ring {
    0% { transform: scale(0.9); opacity: 0.65; }
    100% { transform: scale(1.35); opacity: 0; }
}
@keyframes pc-process-sweep {
    0% { transform: translateX(-120%); }
    100% { transform: translateX(140%); }
}
@keyframes pc-stage-shine {
    0% { transform: translateX(-120%); }
    100% { transform: translateX(140%); }
}
@keyframes pc-dot-pulse {
    0% { box-shadow: 0 0 0 0 rgba(244, 162, 97, 0.46); }
    70% { box-shadow: 0 0 0 12px rgba(244, 162, 97, 0); }
    100% { box-shadow: 0 0 0 0 rgba(244, 162, 97, 0); }
}
@keyframes pc-stage-pulse {
    0% { box-shadow: 0 0 0 0 rgba(244, 162, 97, 0.30); }
    70% { box-shadow: 0 0 0 12px rgba(244, 162, 97, 0); }
    100% { box-shadow: 0 0 0 0 rgba(244, 162, 97, 0); }
}
@media (max-width: 960px) {
    .pc-process-live-overlay {
        padding: 14px;
    }
    .pc-process-live-dock {
        width: 94vw;
        max-height: 72vh;
        animation: pc-live-dock-enter 0.24s ease-out;
    }
    .pc-process-live-dock .pc-process-shell {
        padding: 16px 16px 14px;
    }
    .pc-process-live-statgrid {
        grid-template-columns: 1fr;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


# 把内部枚举值翻译成当前语言下的展示标签。
def localize_stage_label(stage: str, fallback_label: str = "") -> str:
    text = clean_text(stage)
    label = PIPELINE_STAGE_LABELS.get(text)
    if label:
        return label.get(current_locale()) or label.get("zh") or fallback_label or text
    return clean_text(fallback_label) or text


# 把内部枚举值翻译成当前语言下的展示标签。
def translate_mapping_value(value: Any, mapping: Dict[str, Dict[str, str]]) -> str:
    text = clean_text(value)
    if not text:
        return ""
    label = mapping.get(text)
    if not label:
        return text
    return label.get(current_locale()) or label.get("zh") or text


# 针对不同槽位应用各自的本地化规则。
def localize_slot_value(value: Any, slot_key: str = "") -> str:
    text = clean_text(value)
    if not text:
        return ""
    if slot_key == "search_scene":
        return translate_mapping_value(text, SEARCH_SCENE_LABELS)
    if slot_key == "document_attributes.time_range":
        return translate_mapping_value(text, TIME_RANGE_LABELS)
    if slot_key == "document_attributes.paper_type":
        return translate_mapping_value(text, PAPER_TYPE_LABELS)
    if slot_key.startswith("result_preferences."):
        return translate_mapping_value(text, PREFERENCE_LABELS)
    return text


def localize_dimension_text(value: Any) -> str:
    return translate_mapping_value(value, DIMENSION_LABELS)


def localize_field_text(value: Any) -> str:
    return translate_mapping_value(value, FIELD_LABELS)


def localize_slot_path_text(value: Any) -> str:
    return translate_mapping_value(value, SLOT_PATH_LABELS)


def localize_match_type_text(value: Any) -> str:
    return translate_mapping_value(value, MATCH_TYPE_LABELS)


def slot_display(slot: Dict[str, Any], slot_key: str = "") -> str:
    value = slot.get("value")
    if isinstance(value, list):
        normalized = [localize_slot_value(item, slot_key) for item in value if clean_text(item)]
        separator = "、" if current_locale() == "zh" else ", "
        return separator.join(item for item in normalized if item) or "-"
    return localize_slot_value(value, slot_key) or "-"


# 把字符串列表统一渲染成界面块，减少重复展示逻辑。
def render_string_list(title: str, items: Iterable[str], empty_text: str = "-", formatter=None) -> None:
    normalized = []
    for item in items:
        text = clean_text(item)
        if not text:
            continue
        normalized.append(formatter(text) if formatter else text)
    st.markdown(f"<div class='pc-list-title'>{html.escape(title)}</div>", unsafe_allow_html=True)
    if not normalized:
        st.markdown(f"<div class='pc-empty-note'>{html.escape(empty_text)}</div>", unsafe_allow_html=True)
        return
    items_markup = "".join(f"<li>{html.escape(clean_text(item))}</li>" for item in normalized if clean_text(item))
    st.markdown(f"<div class='pc-list-block'><ul>{items_markup}</ul></div>", unsafe_allow_html=True)


def queue_query_state(
    *,
    query: str = "",
    follow_up: str = "",
    top_k: int = 5,
    candidate_pool: int = 120,
    explain_limit: int = 5,
    auto_run: bool = False,
) -> None:
    st.session_state["_pending_query_input"] = query
    st.session_state["_pending_follow_up_input"] = follow_up
    st.session_state["_pending_top_k_input"] = top_k
    st.session_state["_pending_candidate_pool_input"] = candidate_pool
    st.session_state["_pending_explain_limit_input"] = explain_limit
    st.session_state["_pending_auto_run_query"] = auto_run


def apply_pending_query_state() -> bool:
    pending_map = {
        "_pending_query_input": "query_input",
        "_pending_follow_up_input": "follow_up_input",
        "_pending_top_k_input": "top_k_input",
        "_pending_candidate_pool_input": "candidate_pool_input",
        "_pending_explain_limit_input": "explain_limit_input",
    }
    applied = False
    for pending_key, widget_key in pending_map.items():
        if pending_key in st.session_state:
            st.session_state[widget_key] = st.session_state.pop(pending_key)
            applied = True
    return applied


def queue_run_request(*, query_override: str = "", follow_up_override: str = "") -> None:
    st.session_state["_pending_run_query"] = True
    st.session_state["_pending_run_query_override"] = clean_text(query_override)
    st.session_state["_pending_run_follow_up_override"] = clean_text(follow_up_override)


def follow_up_suggestion_signature(payload: Dict[str, Any]) -> str:
    suggestion_payload = payload.get("follow_up_suggestion", {}) or {}
    parts = [
        clean_text(payload.get("query", "")),
        clean_text(payload.get("follow_up_reply", "")),
        clean_text(suggestion_payload.get("question", "")),
        clean_text(suggestion_payload.get("draft", "")),
        clean_text(suggestion_payload.get("generator", "")),
        clean_text(suggestion_payload.get("used_model", "")),
    ]
    return " | ".join(part for part in parts if part)


def sync_generated_follow_up_entry(payload: Dict[str, Any]) -> None:
    suggestion_payload = payload.get("follow_up_suggestion", {}) or {}
    suggested_reply = clean_text(suggestion_payload.get("draft", ""))
    signature = follow_up_suggestion_signature(payload)
    if not suggested_reply or not signature:
        return
    current_value = clean_text(st.session_state.get("gap_follow_up_input", ""))
    previous_auto_value = clean_text(st.session_state.get("_auto_gap_follow_up_value", ""))
    previous_signature = clean_text(st.session_state.get("_auto_gap_follow_up_signature", ""))
    should_apply = not current_value or current_value == previous_auto_value
    if signature != previous_signature:
        if should_apply:
            st.session_state["gap_follow_up_input"] = suggested_reply
            st.session_state["_auto_gap_follow_up_value"] = suggested_reply
        st.session_state["_auto_gap_follow_up_signature"] = signature
        return
    if should_apply and current_value != suggested_reply:
        st.session_state["gap_follow_up_input"] = suggested_reply
        st.session_state["_auto_gap_follow_up_value"] = suggested_reply


def build_follow_up_draft(frame: Dict[str, Any], gap_report: Dict[str, Any]) -> str:
    missing_slots = gap_report.get("query_gap", [])
    segments: List[str] = []

    def slot_value(group: str, key: str) -> str:
        slot_key = f"{group}.{key}"
        slot = frame.get(group, {}).get(key, {})
        value = slot.get("value")
        if isinstance(value, list):
            localized_items = [localize_slot_value(item, slot_key) for item in value if clean_text(item)]
            return "、".join(item for item in localized_items if clean_text(item))
        return localize_slot_value(value, slot_key)

    if any("研究领域" in item for item in missing_slots):
        segments.append(f"研究领域是{slot_value('research_topic', 'domain') or '大语言模型'}")
    if any("研究任务" in item for item in missing_slots):
        segments.append(f"研究任务是{slot_value('research_topic', 'task') or 'retrieval-augmented generation'}")
    if any("研究问题" in item for item in missing_slots):
        segments.append(f"研究问题是{slot_value('research_topic', 'problem') or '当前查询关注的问题'}")
    if any(any(token in item for token in ("方法", "模型家族", "数据集", "指标", "模态")) for item in missing_slots):
        segments.append("方法、模型家族、数据集、指标和模态不限")

    time_range = slot_value("document_attributes", "time_range")
    if time_range:
        segments.append(f"时间范围是{time_range}")
    elif any("时间范围" in item for item in missing_slots):
        segments.append("时间范围只看最近两年")

    paper_type = slot_value("document_attributes", "paper_type")
    if paper_type:
        segments.append(f"论文类型优先{paper_type}")
    elif any("论文类型" in item for item in missing_slots):
        segments.append("论文类型以综述为主")

    if slot_value("result_preferences", "need_explainable_reason") == "yes":
        segments.append("并解释每篇论文为何匹配")

    return "；".join(segment for segment in segments if clean_text(segment))


def format_stage_event_text(event: Dict[str, Any]) -> str:
    stage = clean_text(event.get("stage"))
    label = localize_stage_label(stage, clean_text(event.get("label") or stage))
    status = clean_text(event.get("status"))
    prefix = t("stage_running") if status == "running" else t("stage_completed")
    parts = [f"{prefix}：{label}"]
    if event.get("generator"):
        generator_label = {"llm": "LLM", "rule": "规则兜底"}.get(clean_text(event["generator"]), clean_text(event["generator"]))
        parts.append(f"生成={generator_label}")
    if event.get("used_model"):
        parts.append(f"模型={event['used_model']}")
    if event.get("result_count") is not None:
        parts.append(f"结果={event['result_count']}")
    if event.get("candidate_pool_size") is not None:
        parts.append(f"候选池={event['candidate_pool_size']}")
    if event.get("paper_count") is not None:
        parts.append(f"论文={event['paper_count']}")
    if event.get("cached_count") is not None:
        parts.append(f"缓存={event['cached_count']}")
    if event.get("llm_count") is not None:
        parts.append(f"LLM={event['llm_count']}")
    if event.get("duration") is not None and status == "completed":
        parts.append(f"{event['duration']:.4f}s")
    return " | ".join(parts)


def summarize_stage_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered_keys: List[str] = []
    stage_map: Dict[str, Dict[str, Any]] = {}
    for raw_event in events:
        event = dict(raw_event)
        stage = clean_text(event.get("stage"))
        label = localize_stage_label(stage, clean_text(event.get("label") or stage))
        if not label:
            continue
        key = stage or label
        if key not in stage_map:
            ordered_keys.append(key)
            stage_map[key] = {"stage": stage, "label": label}
        stage_map[key].update(event)
        stage_map[key]["stage"] = stage
        stage_map[key]["label"] = label
    return [stage_map[key] for key in ordered_keys]


def build_stage_event_meta(event: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    if event.get("generator"):
        generator_label = {"llm": "LLM", "rule": "规则兜底"}.get(clean_text(event["generator"]), clean_text(event["generator"]))
        parts.append(f"生成={generator_label}")
    if event.get("used_model"):
        parts.append(f"模型={clean_text(event['used_model'])}")
    if event.get("result_count") is not None:
        parts.append(f"结果={event['result_count']}")
    if event.get("candidate_pool_size") is not None:
        parts.append(f"候选池={event['candidate_pool_size']}")
    if event.get("paper_count") is not None:
        parts.append(f"论文={event['paper_count']}")
    if event.get("cached_count") is not None:
        parts.append(f"缓存={event['cached_count']}")
    if event.get("llm_count") is not None:
        parts.append(f"LLM={event['llm_count']}")
    if event.get("duration") is not None and clean_text(event.get("status")) == "completed":
        parts.append(f"{event['duration']:.4f}s")
    return parts


def build_stage_events_markup(events: Iterable[Dict[str, Any]], *, live: bool = False) -> str:
    normalized = summarize_stage_events(events)
    if not normalized:
        return ""

    stage_positions = {stage: index for index, stage in enumerate(PIPELINE_STAGE_SEQUENCE, start=1)}
    total_steps = max(len(PIPELINE_STAGE_SEQUENCE), len(normalized))
    running_event = next((event for event in normalized if clean_text(event.get("status")) == "running"), None)
    current_event = running_event or (normalized[-1] if normalized else None)
    current_stage_label = clean_text((current_event or {}).get("label", ""))
    current_stage_step = stage_positions.get(clean_text((current_event or {}).get("stage")), len(normalized))

    if running_event:
        progress_ratio = min(max((current_stage_step - 0.4) / max(total_steps, 1), 0.08), 0.96)
        title = t("process_live_title")
        subtitle = t("process_current_stage", value=current_stage_label)
        hint = t("process_running_hint")
    else:
        progress_ratio = 1.0
        title = t("process_done_title")
        subtitle = t("process_last_stage", value=current_stage_label or t("run_process"))
        hint = t("process_completed_hint")

    cards: List[str] = []
    for index, event in enumerate(normalized, start=1):
        status = clean_text(event.get("status"))
        status_label = {
            "running": t("stage_running"),
            "completed": t("stage_completed"),
        }.get(status, t("stage_pending"))
        card_classes = "pc-stage-card"
        if status == "running":
            card_classes += " is-running"
        elif status == "completed":
            card_classes += " is-completed"
        meta_text = " | ".join(build_stage_event_meta(event))
        cards.append(
            "".join(
                [
                    f"<div class='{card_classes}'>",
                    "<div class='pc-stage-card-top'>",
                    f"<div class='pc-stage-status'><span class='pc-stage-status-dot'></span>{html.escape(status_label)}</div>",
                    f"<div class='pc-stage-order'>{index:02d}</div>",
                    "</div>",
                    f"<div class='pc-stage-label'>{html.escape(clean_text(event.get('label')))}</div>",
                    f"<div class='pc-stage-meta-line'>{html.escape(meta_text or hint)}</div>",
                    "</div>",
                ]
            )
        )

    is_live_running = bool(live and running_event)
    shell_class = "pc-process-shell is-live" if is_live_running else "pc-process-shell"
    progress_width = f"{max(min(progress_ratio * 100, 100.0), 6.0):.1f}%"
    completed_count = sum(1 for event in normalized if clean_text(event.get("status")) == "completed")
    active_count = 1 if running_event else 0
    remaining_count = max(total_steps - completed_count - active_count, 0)
    live_body_markup = "".join(
        [
            "<div class='pc-process-live-compact'>",
            "<div class='pc-process-live-compact-head'>",
            "<div class='pc-process-live-orb'></div>",
            "<div class='pc-process-live-compact-copy'>",
            f"<div class='pc-process-live-compact-kicker'>{html.escape(t('process_in_progress_kicker'))}</div>",
            f"<div class='pc-process-live-compact-title'>{html.escape(current_stage_label or title)}</div>",
            f"<div class='pc-process-live-compact-text'>{html.escape(t('process_live_summary'))}</div>",
            "</div>",
            "</div>",
            "<div class='pc-process-live-statgrid'>",
            (
                "<div class='pc-process-live-stat'>"
                f"<div class='pc-process-live-stat-label'>{html.escape(t('process_completed_stages'))}</div>"
                f"<div class='pc-process-live-stat-value'>{completed_count}</div>"
                "</div>"
            ),
            (
                "<div class='pc-process-live-stat'>"
                f"<div class='pc-process-live-stat-label'>{html.escape(t('process_current_progress_label'))}</div>"
                f"<div class='pc-process-live-stat-value'>{current_stage_step}/{total_steps}</div>"
                "</div>"
            ),
            (
                "<div class='pc-process-live-stat'>"
                f"<div class='pc-process-live-stat-label'>{html.escape(t('process_remaining_stages'))}</div>"
                f"<div class='pc-process-live-stat-value'>{remaining_count}</div>"
                "</div>"
            ),
            "</div>",
            f"<div class='pc-process-live-tip'>{html.escape(t('process_live_tip'))}</div>",
            "</div>",
        ]
    ) if is_live_running else f"<div class='pc-stage-list'>{''.join(cards)}</div>"
    process_meta_markup = (
        ""
        if is_live_running
        else "".join(
            [
                "<div class='pc-process-meta'>",
                f"<div class='pc-process-chip'>{html.escape(t('process_stage_progress', completed=current_stage_step, total=total_steps))}</div>",
                f"<div class='pc-process-chip'>{html.escape(hint)}</div>",
                "</div>",
            ]
        )
    )
    shell_markup = "".join(
        [
            f"<div class='{shell_class}'>",
            "<div class='pc-process-banner'>",
            "<div class='pc-process-banner-copy'>",
            f"<div class='pc-process-kicker'>{html.escape(t('run_process'))}</div>",
            (
                "<div class='pc-process-live-header'>"
                "<div class='pc-process-live-orb'></div>"
                f"<div class='pc-process-title'>{html.escape(title)}</div>"
                "</div>"
                if is_live_running
                else f"<div class='pc-process-title'>{html.escape(title)}</div>"
            ),
            f"<div class='pc-process-subtitle'>{html.escape(subtitle)}</div>",
            "</div>",
            process_meta_markup,
            "</div>",
            "<div class='pc-process-progress-track'>",
            f"<div class='pc-process-progress-fill' style='width:{progress_width};'></div>",
            "</div>",
            live_body_markup,
            "</div>",
        ]
    )
    if is_live_running:
        return "".join(
            [
                "<div class='pc-process-live-overlay'>",
                "<div class='pc-process-live-backdrop'></div>",
                f"<div class='pc-process-live-dock'>{shell_markup}</div>",
                "</div>",
            ]
        )
    return shell_markup


def render_stage_events(events: Iterable[Dict[str, Any]], *, live: bool = False, show_header: bool = True) -> None:
    markup = build_stage_events_markup(events, live=live)
    if not markup:
        return
    if show_header:
        st.subheader(t("run_process"))
    st.markdown(markup, unsafe_allow_html=True)


def summarize_models_used(payload: Dict[str, Any]) -> List[str]:
    models: List[str] = []
    for key in ("initial_intent_model", "follow_up_intent_model"):
        value = clean_text(payload.get(key, ""))
        if value:
            models.append(value)
    suggestion = payload.get("follow_up_suggestion", {}) or {}
    suggestion_model = clean_text(suggestion.get("used_model", ""))
    if suggestion_model:
        models.append(suggestion_model)
    for result in payload.get("top_k_results", []):
        used_model = clean_text(result.get("used_model", ""))
        if used_model:
            models.append(used_model)
    deduped: List[str] = []
    seen = set()
    for item in models:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped


def describe_result_source(result: Dict[str, Any]) -> str:
    parser_name = clean_text(result.get("explanation_parser", ""))
    parser_labels = {
        "llm_query_paper_match_cache": "缓存命中",
        "llm_query_paper_match_batch": "LLM 批量生成",
        "llm_query_paper_match_single_fallback": "LLM 单条回退",
        "llm_query_paper_match": "LLM 单条生成",
        "pre_rank": "规则预排",
    }
    return parser_labels.get(parser_name, parser_name or "-")


def render_runtime_summary(payload: Dict[str, Any]) -> None:
    stage_events = payload.get("stage_events", [])
    top_results = payload.get("top_k_results", [])
    used_llm = any(
        clean_text(payload.get(key, "")) == "llm"
        for key in ("initial_intent_parser", "follow_up_intent_parser")
    )
    used_llm = used_llm or any(clean_text(result.get("used_model", "")) for result in top_results)
    used_llm = used_llm or clean_text((payload.get("follow_up_suggestion", {}) or {}).get("generator", "")) == "llm"
    models_used = summarize_models_used(payload)
    cached_titles = [clean_text(result.get("title", "")) for result in top_results if clean_text(result.get("explanation_parser", "")) == "llm_query_paper_match_cache"]

    st.subheader(t("runtime_summary"))
    st.caption(t("llm_usage_summary", value="是" if used_llm else "否"))
    if models_used:
        st.caption(t("models_used_summary", value="；".join(models_used)))
    query_match_event = next(
        (
            event
            for event in reversed(stage_events)
            if clean_text(event.get("stage")) == "query_paper_match" and clean_text(event.get("status")) == "completed"
        ),
        None,
    )
    if query_match_event:
        st.caption(
            "Query-Paper Match：缓存="
            + str(query_match_event.get("cached_count", 0))
            + " | LLM="
            + str(query_match_event.get("llm_count", 0))
        )
    if cached_titles:
        st.caption(t("cache_results_summary", value="；".join(cached_titles)))
    else:
        st.caption(t("no_cache_results"))


def render_section_header(step_number: int, title: str, caption: str = "", anchor_id: str = "") -> None:
    if anchor_id:
        st.markdown(f"<div id='{anchor_id}'></div>", unsafe_allow_html=True)
    markup = [
        "<div class='pc-section-heading'>",
        f"<div class='pc-section-step'>{step_number:02d}</div>",
        "<div>",
        f"<div class='pc-section-title'>{html.escape(title)}</div>",
    ]
    if caption:
        markup.append(f"<div class='pc-section-caption'>{html.escape(caption)}</div>")
    markup.extend(["</div>", "</div>"])
    st.markdown("".join(markup), unsafe_allow_html=True)


def render_anchor_autoscroll(anchor_id: str, *, behavior: str = "smooth", block: str = "start") -> None:
    target_id = clean_text(anchor_id)
    if not target_id:
        return
    components.html(
        f"""
<div style="height:0; overflow:hidden;"></div>
<script>
(function() {{
    const targetId = {target_id!r};
    const behavior = {behavior!r};
    const block = {block!r};
    const maxAttempts = 72;
    let attempts = 0;

    function scrollToAnchor() {{
        const parentWindow = window.parent;
        const parentDoc = parentWindow && parentWindow.document;
        if (!parentDoc) return false;
        const anchor = parentDoc.getElementById(targetId);
        if (!anchor) return false;
        anchor.scrollIntoView({{ behavior, block }});
        try {{
            parentWindow.location.hash = "#" + targetId;
        }} catch (error) {{
        }}
        return true;
    }}

    if (scrollToAnchor()) return;

    const timer = window.setInterval(function() {{
        attempts += 1;
        if (scrollToAnchor() || attempts >= maxAttempts) {{
            window.clearInterval(timer);
        }}
    }}, 120);
}})();
</script>
        """,
        height=0,
        width=0,
    )


# 将标准示例查询写回界面状态；可选择仅回填，或回填后自动运行。
def apply_demo_query(item: Dict[str, Any], *, auto_run: bool = True) -> None:
    queue_query_state(
        query=item["query"],
        follow_up=item.get("follow_up_reply", ""),
        top_k=5,
        candidate_pool=120,
        explain_limit=5,
        auto_run=auto_run,
    )


def build_step_navigation_items(has_payload: bool) -> List[tuple[str, str]]:
    items = [
        (t("search_input"), STEP_SECTION_ANCHORS[1]),
        (f"{t('run_process')} / {t('model_workbench')}", STEP_SECTION_ANCHORS[2]),
    ]
    if has_payload:
        items.extend(
            [
                (t("system_understanding"), STEP_SECTION_ANCHORS[3]),
                (t("follow_up_workspace"), STEP_SECTION_ANCHORS[4]),
                (t("topk_recommendations"), STEP_SECTION_ANCHORS[5]),
                (t("paper_details"), STEP_SECTION_ANCHORS[6]),
            ]
        )
    items.append((t("management_workspace"), STEP_SECTION_ANCHORS[7]))
    return items


def render_step_navigation(has_payload: bool) -> None:
    st.subheader(t("quick_navigation"))
    st.caption(t("quick_navigation_hint"))
    if not has_payload:
        st.caption(t("quick_navigation_empty"))
    links = "\n".join(
        [
            f"- <a href='#{anchor}' target='_self'>{label}</a>"
            for label, anchor in build_step_navigation_items(has_payload)
        ]
    )
    st.markdown(links, unsafe_allow_html=True)


# 侧边栏保留界面设置、步骤定位和轻量示例快捷入口，避免打断主页面主线。
def render_sidebar(demo_queries: List[Dict[str, Any]], has_payload: bool) -> bool:
    with st.sidebar:
        current_value = st.session_state.get("ui_language", "zh")
        language_codes = list(LANGUAGE_OPTIONS.keys())
        default_index = language_codes.index(current_value) if current_value in language_codes else 0
        st.subheader(t("workspace_settings"))
        st.selectbox(
            t("language_label"),
            options=language_codes,
            index=default_index,
            key="ui_language",
            format_func=lambda code: LANGUAGE_OPTIONS.get(code, code),
        )
        st.caption(t("sidebar_hint"))
        render_step_navigation(has_payload)
        st.subheader(t("demo_replay"))
        st.caption(t("sidebar_demo_hint"))
        for index, item in enumerate(demo_queries, start=1):
            if st.button(t("demo_button", index=index), key=f"sidebar_demo_{index}", use_container_width=True):
                apply_demo_query(item, auto_run=False)
                st.rerun()
        return st.checkbox(t("show_raw_json"), value=False)


def render_panel_lead(title: str, caption: str = "", kicker: str = "") -> None:
    markup = ["<div class='pc-panel-lead'>"]
    if kicker:
        markup.append(f"<div class='pc-panel-kicker'>{html.escape(kicker)}</div>")
    markup.append(f"<div class='pc-panel-title'>{html.escape(title)}</div>")
    if caption:
        markup.append(f"<div class='pc-panel-caption'>{html.escape(caption)}</div>")
    markup.append("</div>")
    st.markdown("".join(markup), unsafe_allow_html=True)


def render_chip_row(items: Iterable[str], *, tone: str = "soft") -> None:
    chips = [clean_text(item) for item in items if clean_text(item)]
    if not chips:
        return
    markup = "".join(
        f"<span class='pc-chip {html.escape(tone)}'>{html.escape(text)}</span>"
        for text in chips
    )
    st.markdown(f"<div class='pc-chip-row'>{markup}</div>", unsafe_allow_html=True)


def render_metric_grid(metrics: List[tuple[str, Any]]) -> None:
    cards = "".join(
        [
            "<div class='pc-kpi-card'>"
            f"<div class='pc-kpi-label'>{html.escape(clean_text(label))}</div>"
            f"<div class='pc-kpi-value'>{html.escape(clean_text(value))}</div>"
            "</div>"
            for label, value in metrics
        ]
    )
    st.markdown(f"<div class='pc-kpi-grid'>{cards}</div>", unsafe_allow_html=True)


def render_key_value_list(items: List[tuple[str, Any]]) -> None:
    rows = "".join(
        [
            "<div class='pc-kv-item'>"
            f"<div class='pc-kv-key'>{html.escape(clean_text(key))}</div>"
            f"<div class='pc-kv-value'>{html.escape(clean_text(value))}</div>"
            "</div>"
            for key, value in items
        ]
    )
    st.markdown(f"<div class='pc-kv-list'>{rows}</div>", unsafe_allow_html=True)


def render_hero_banner(stats: Dict[str, int]) -> None:
    dataset_info = get_active_dataset_info()
    hero = "".join(
        [
            "<div class='pc-hero'>",
            "<div class='pc-hero-copy'>",
            "<div class='pc-hero-kicker'>基于用户意图理解的学术论文检索与管理系统</div>",
            "<div class='pc-hero-title'>PaperCompass 检索工作台</div>",
            "<div class='pc-hero-subtitle'>",
            #"界面层改成更贴近论文插图和正式系统演示的样式，但检索主链路、意图解析、重排解释、收藏与历史逻辑保持不变。",
            "</div>",
            "</div>",
            "<div class='pc-hero-meta'>",
            f"<div class='pc-chip soft'>{html.escape(str(dataset_info.get('label', 'Dataset')))}</div>",
            f"<div class='pc-chip soft'>数据库：{html.escape(relative_to_project(get_default_db_path()))}</div>",
            f"<div class='pc-chip good'>论文 {stats.get('papers', 0)} 篇</div>",
            f"<div class='pc-chip soft'>语义卡 {stats.get('semantic_cards', 0)} 张</div>",
            f"<div class='pc-chip soft'>历史记录 {stats.get('intent_histories', 0)} 条</div>",
            "</div>",
            "</div>",
        ]
    )
    st.markdown(hero, unsafe_allow_html=True)


def render_search_workspace(stats: Dict[str, int], demo_queries: List[Dict[str, Any]]) -> None:
    left_col, right_col = st.columns([0.92, 2.48], gap="large")
    with left_col:
        with st.container(border=True):
            render_panel_lead("界面设置 / 系统状态", "这里集中展示运行库、模型状态和当前检索模式。", "状态总览")
            render_key_value_list(
                [
                    ("当前数据库", relative_to_project(get_default_db_path())),
                    ("论文数", stats.get("papers", 0)),
                    ("章节数", stats.get("sections", 0)),
                    ("FTS 行数", stats.get("fts_rows", 0)),
                    ("语义卡", stats.get("semantic_cards", 0)),
                    ("检索历史", stats.get("intent_histories", 0)),
                    ("收藏论文", stats.get("saved_papers", 0)),
                    ("LLM 状态", t("api_key_configured") if OPENAI_API_KEY else t("api_key_unconfigured")),
                ]
            )
            render_chip_row(
                [
                    "主链路检索",
                    f"Top-K = {int(st.session_state.get('top_k_input', 5))}",
                    f"候选池 = {int(st.session_state.get('candidate_pool_input', 120))}",
                ]
            )

    with right_col:
        with st.container(border=True):
            render_panel_lead(t("search_input"), t("search_input_caption"), "自然语言入口")
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
            config_col1, config_col2, config_col3, config_col4 = st.columns([1, 1, 1, 1.1], gap="medium")
            with config_col1:
                st.number_input("Top-K", min_value=3, max_value=10, value=5, step=1, key="top_k_input")
            with config_col2:
                st.number_input(
                    t("candidate_pool_size"),
                    min_value=20,
                    max_value=200,
                    value=120,
                    step=10,
                    key="candidate_pool_input",
                )
            with config_col3:
                st.number_input(t("explain_limit"), min_value=3, max_value=10, value=5, step=1, key="explain_limit_input")
            with config_col4:
                st.write("")
                if st.button(t("run_search"), type="primary", use_container_width=True, key="run_search_button"):
                    queue_run_request()
                    st.rerun()

        with st.container(border=True):
            render_panel_lead("标准示例与状态面板", "点击示例会回填到输入框，不会自动检索。", "快速回填")
            render_metric_grid(
                [
                    (t("papers_metric"), stats.get("papers", 0)),
                    (t("sections_metric"), stats.get("sections", 0)),
                    (t("semantic_cards_metric"), stats.get("semantic_cards", 0)),
                    (t("history_metric"), stats.get("intent_histories", 0)),
                ]
            )
            demo_cols = st.columns(2, gap="medium")
            for index, item in enumerate(demo_queries, start=1):
                with demo_cols[(index - 1) % 2]:
                    label = truncate_text(item.get("query", ""), 24)
                    if st.button(f"示例 {index} · {label}", key=f"main_demo_{index}", use_container_width=True):
                        apply_demo_query(item, auto_run=False)
                        st.rerun()


# 首页概览区域展示数据库和语义层的总体统计。
def render_overview(stats: Dict[str, int]) -> None:
    metrics = [
        (t("papers_metric"), stats["papers"]),
        (t("sections_metric"), stats["sections"]),
        (t("fts_metric"), stats["fts_rows"]),
        (t("semantic_cards_metric"), stats["semantic_cards"]),
        (t("history_metric"), stats["intent_histories"]),
        (t("saved_metric"), stats["saved_papers"]),
    ]
    render_metric_grid(metrics)


def render_system_status_panel(stats: Dict[str, int], state_loaded: bool) -> None:
    dataset_info = get_active_dataset_info()
    render_panel_lead(t("system_status"), str(dataset_info.get("label", "Dataset")), "运行状态")
    render_key_value_list(
        [
            ("数据集目录", dataset_info.get("display_path", "-")),
            ("当前数据库", relative_to_project(get_default_db_path())),
            ("状态文件", relative_to_project(APP_STATE_PATH)),
            (
                "统计摘要",
                t(
                    "status_summary",
                    papers=stats["papers"],
                    sections=stats["sections"],
                    fts_rows=stats["fts_rows"],
                    semantic_cards=stats["semantic_cards"],
                    intent_histories=stats["intent_histories"],
                    saved_papers=stats["saved_papers"],
                ),
            ),
        ]
    )
    if state_loaded:
        st.caption(t("state_loaded"))


def render_llm_runtime_panel() -> None:
    render_panel_lead(t("llm_runtime"), "这里仅展示模型连通性和当前接口配置。", "模型连接")
    if OPENAI_API_KEY:
        st.success(t("api_key_detected"))
    else:
        st.error(t("api_key_missing"))
    render_key_value_list(
        [
            ("Base URL", OPENAI_API_BASE),
            ("Model", OPENAI_MODEL),
        ]
    )
    if st.button(t("test_api"), key="test_api_workbench", use_container_width=True):
        with st.spinner(t("testing_api")):
            ok, message = test_openai_api(OPENAI_API_KEY)
        if ok:
            st.success(message)
        else:
            st.error(message)


def render_runtime_workspace(stats: Dict[str, int], payload: Dict[str, Any] | None, state_loaded: bool) -> None:
    with st.container(border=True):
        render_panel_lead(f"{t('run_process')} / {t('model_workbench')}", t("runtime_workspace_caption"), "链路工作区")
        process_tab, workbench_tab = st.tabs([t("run_process"), t("model_workbench")])
        with process_tab:
            if payload:
                stage_events = payload.get("stage_events", st.session_state.get("latest_stage_events", []))
                normalized = [event for event in stage_events if clean_text(event.get("label") or event.get("stage"))]
                if normalized:
                    st.caption(t("run_process_collapsed_hint"))
                    with st.expander(f"{t('run_process_details')}（{len(normalized)}）", expanded=False):
                        render_stage_events(normalized, live=False, show_header=False)
                else:
                    st.info(t("query_not_run_yet"))
            else:
                st.info(t("query_not_run_yet"))
        with workbench_tab:
            if payload:
                render_runtime_summary(payload)
            else:
                st.info(t("query_not_run_yet"))
            render_overview(stats)
            status_col, llm_col = st.columns([3, 2], gap="large")
            with status_col:
                with st.container(border=True):
                    render_system_status_panel(stats, state_loaded)
            with llm_col:
                with st.container(border=True):
                    render_llm_runtime_panel()


def collect_query_variant_changes(initial_frame: Dict[str, Any], final_frame: Dict[str, Any]) -> List[str]:
    changes: List[str] = []
    for key, label in [
        ("coarse_queries", "Coarse"),
        ("dense_queries", "Dense"),
        ("exact_queries", "Exact"),
    ]:
        before = [clean_text(item) for item in initial_frame.get(key, []) if clean_text(item)]
        after = [clean_text(item) for item in final_frame.get(key, []) if clean_text(item)]
        added = [item for item in after if item not in before]
        removed = [item for item in before if item not in after]
        if added:
            changes.append(f"{label} + {clean_text('；'.join(added[:2]))}")
        if removed:
            changes.append(f"{label} - {clean_text('；'.join(removed[:2]))}")
    return changes


def coerce_score(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize_result_quality(results: List[Dict[str, Any]], limit: int = 5) -> Dict[str, Any]:
    top_slice = list(results[:limit])
    if not top_slice:
        return {
            "slice_size": 0,
            "main_intent_count": 0,
            "avg_match_score": 0.0,
            "avg_final_score": 0.0,
        }

    main_intent_count = 0
    match_scores: List[float] = []
    final_scores: List[float] = []
    for item in top_slice:
        query_match = item.get("query_paper_match") or {}
        if query_match.get("main_intent_satisfied"):
            main_intent_count += 1
        match_scores.append(coerce_score(query_match.get("match_score", 0.0)))
        final_scores.append(coerce_score(item.get("final_score", 0.0)))
    return {
        "slice_size": len(top_slice),
        "main_intent_count": main_intent_count,
        "avg_match_score": round(sum(match_scores) / max(len(match_scores), 1), 3),
        "avg_final_score": round(sum(final_scores) / max(len(final_scores), 1), 3),
    }


def build_result_change_summary(previous_payload: Dict[str, Any], current_payload: Dict[str, Any]) -> Dict[str, Any] | None:
    previous_results = previous_payload.get("top_k_results") or []
    current_results = current_payload.get("top_k_results") or []
    if not previous_results or not current_results:
        return None

    previous_ids = [clean_text(item.get("paper_id", "")) for item in previous_results if clean_text(item.get("paper_id", ""))]
    current_ids = [clean_text(item.get("paper_id", "")) for item in current_results if clean_text(item.get("paper_id", ""))]
    if not previous_ids or not current_ids:
        return None

    previous_index = {paper_id: index for index, paper_id in enumerate(previous_ids)}
    current_index = {paper_id: index for index, paper_id in enumerate(current_ids)}
    previous_titles = {clean_text(item.get("paper_id", "")): clean_text(item.get("title", "")) for item in previous_results}
    current_titles = {clean_text(item.get("paper_id", "")): clean_text(item.get("title", "")) for item in current_results}

    overlap_ids = [paper_id for paper_id in current_ids if paper_id in previous_index]
    added_ids = [paper_id for paper_id in current_ids if paper_id not in previous_index]
    removed_ids = [paper_id for paper_id in previous_ids if paper_id not in current_index]
    reordered_items = []
    for paper_id in overlap_ids:
        before_rank = previous_index[paper_id] + 1
        after_rank = current_index[paper_id] + 1
        if before_rank != after_rank:
            reordered_items.append(f"{current_titles.get(paper_id, paper_id)} ({before_rank} -> {after_rank})")

    same_set = set(previous_ids) == set(current_ids)
    same_order = previous_ids == current_ids
    if same_set and same_order:
        headline_key = "result_change_same_order"
    elif same_set:
        headline_key = "result_change_reordered"
    elif added_ids and removed_ids:
        headline_key = "result_change_mixed"
    elif added_ids:
        headline_key = "result_change_only_added"
    else:
        headline_key = "result_change_only_removed"

    previous_quality = summarize_result_quality(previous_results)
    current_quality = summarize_result_quality(current_results)
    main_before = previous_quality.get("main_intent_count", 0)
    main_after = current_quality.get("main_intent_count", 0)
    match_before = previous_quality.get("avg_match_score", 0.0)
    match_after = current_quality.get("avg_match_score", 0.0)
    final_before = previous_quality.get("avg_final_score", 0.0)
    final_after = current_quality.get("avg_final_score", 0.0)
    match_delta = round(match_after - match_before, 3)
    final_delta = round(final_after - final_before, 3)
    main_delta = int(main_after - main_before)

    if main_delta >= 1 or match_delta >= 0.05 or final_delta >= 0.05:
        convergence_status = "improved"
        convergence_headline = t(
            "result_convergence_improved",
            main_before=main_before,
            main_after=main_after,
            match_before=match_before,
            match_after=match_after,
        )
    elif main_delta <= -1 or match_delta <= -0.05 or final_delta <= -0.05:
        convergence_status = "weakened"
        convergence_headline = t(
            "result_convergence_weakened",
            main_before=main_before,
            main_after=main_after,
            match_before=match_before,
            match_after=match_after,
        )
    else:
        convergence_status = "stable"
        convergence_headline = t(
            "result_convergence_stable",
            main_before=main_before,
            main_after=main_after,
            match_before=match_before,
            match_after=match_after,
        )

    return {
        "headline": t(headline_key),
        "overlap_count": len(overlap_ids),
        "current_count": len(current_ids),
        "added_titles": [current_titles.get(paper_id, paper_id) for paper_id in added_ids],
        "removed_titles": [previous_titles.get(paper_id, paper_id) for paper_id in removed_ids],
        "reordered_titles": reordered_items,
        "quality_before": previous_quality,
        "quality_after": current_quality,
        "convergence_status": convergence_status,
        "convergence_headline": convergence_headline,
    }


def render_follow_up_convergence(payload: Dict[str, Any]) -> None:
    if not clean_text(payload.get("follow_up_reply", "")):
        return
    initial_frame = payload.get("initial_intent_frame") or {}
    final_frame = payload.get("final_intent_frame") or {}
    if not initial_frame or not final_frame:
        return
    initial_missing = initial_frame.get("missing_slots", []) or []
    final_missing = final_frame.get("missing_slots", []) or []
    initial_answered = initial_frame.get("answered_slots", []) or []
    final_answered = final_frame.get("answered_slots", []) or []
    newly_answered = [item for item in final_answered if item not in initial_answered]
    query_changes = collect_query_variant_changes(initial_frame, final_frame)

    st.markdown(f"**{t('follow_up_convergence')}**")
    st.caption(t("follow_up_convergence_caption"))
    st.caption(
        t(
            "slot_count_change",
            missing_before=len(initial_missing),
            missing_after=len(final_missing),
            answered_before=len(initial_answered),
            answered_after=len(final_answered),
        )
    )
    result_summary = payload.get("result_change_summary") or {}
    convergence_headline = clean_text(result_summary.get("convergence_headline", ""))
    if convergence_headline:
        convergence_status = clean_text(result_summary.get("convergence_status", ""))
        if convergence_status == "improved":
            st.success(convergence_headline)
        elif convergence_status == "weakened":
            st.warning(convergence_headline)
        else:
            st.info(convergence_headline)
        quality_before = result_summary.get("quality_before") or {}
        quality_after = result_summary.get("quality_after") or {}
        st.caption(
            t(
                "result_quality_delta",
                main_before=quality_before.get("main_intent_count", 0),
                main_after=quality_after.get("main_intent_count", 0),
                match_before=quality_before.get("avg_match_score", 0.0),
                match_after=quality_after.get("avg_match_score", 0.0),
                final_before=quality_before.get("avg_final_score", 0.0),
                final_after=quality_after.get("avg_final_score", 0.0),
            )
        )
    render_string_list(
        t("newly_answered_slots"),
        newly_answered,
        t("no_newly_answered_slots"),
        formatter=localize_slot_path_text,
    )
    render_string_list(t("query_variant_changes"), query_changes, t("no_query_variant_changes"))


def render_result_change_summary(payload: Dict[str, Any]) -> None:
    summary = payload.get("result_change_summary")
    if not isinstance(summary, dict):
        return
    st.markdown(f"**{t('result_change_summary')}**")
    st.caption(t("result_change_caption"))
    st.info(summary.get("headline", ""))
    quality_before = summary.get("quality_before") or {}
    quality_after = summary.get("quality_after") or {}
    if quality_before or quality_after:
        st.caption(
            t(
                "result_quality_delta",
                main_before=quality_before.get("main_intent_count", 0),
                main_after=quality_after.get("main_intent_count", 0),
                match_before=quality_before.get("avg_match_score", 0.0),
                match_after=quality_after.get("avg_match_score", 0.0),
                final_before=quality_before.get("avg_final_score", 0.0),
                final_after=quality_after.get("avg_final_score", 0.0),
            )
        )
    st.caption(
        t(
            "result_overlap_summary",
            overlap_count=summary.get("overlap_count", 0),
            current_count=summary.get("current_count", 0),
            added_count=len(summary.get("added_titles", [])),
            removed_count=len(summary.get("removed_titles", [])),
        )
    )
    render_string_list(t("reordered_results"), summary.get("reordered_titles", []), t("no_reordered_results"))
    render_string_list(t("new_results_added"), summary.get("added_titles", []), t("no_new_results_added"))
    render_string_list(t("results_removed"), summary.get("removed_titles", []), t("no_results_removed"))


# 意图面板展示结构化槽位及其当前状态。
def resolve_follow_up_question(frame: Dict[str, Any], follow_up_suggestion: Dict[str, Any] | None = None) -> str:
    suggestion_payload = follow_up_suggestion or {}
    question = clean_text(suggestion_payload.get("question", ""))
    if question:
        return question
    return clean_text(frame.get("clarification_question", ""))


def describe_follow_up_source(suggestion_payload: Dict[str, Any]) -> str:
    generator = clean_text(suggestion_payload.get("generator", ""))
    used_fallback = bool(suggestion_payload.get("used_fallback"))
    if generator == "llm" and used_fallback:
        return "LLM 生成（含兜底修复）"
    return {"llm": "LLM 生成", "rule": "规则兜底"}.get(generator, "系统建议")


def render_intent_panel(frame: Dict[str, Any], follow_up_suggestion: Dict[str, Any] | None = None) -> None:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            render_panel_lead(t("current_intent"), "系统先把自然语言需求整理成结构化意图，再进入检索主链路。", "意图概览")
            render_key_value_list(
                [
                    (t("search_scene"), slot_display(frame.get("search_scene", {}), "search_scene")),
                    (
                        t("research_topic"),
                        " | ".join(
                            [
                                slot_display(frame["research_topic"]["domain"]),
                                slot_display(frame["research_topic"]["task"]),
                                slot_display(frame["research_topic"]["problem"]),
                            ]
                        ),
                    ),
                    (t("topic_keywords"), slot_display(frame["research_topic"]["keywords"])),
                ]
            )
    with col2:
        with st.container(border=True):
            render_panel_lead("约束与偏好", "这里汇总技术约束、文档属性和结果偏好。", "检索条件")
            render_key_value_list(
                [
                    (t("method"), slot_display(frame["technical_constraints"]["method"])),
                    (t("model_family"), slot_display(frame["technical_constraints"]["model_family"])),
                    (t("dataset"), slot_display(frame["technical_constraints"]["dataset"])),
                    (t("metric"), slot_display(frame["technical_constraints"]["metric"])),
                    (t("modality"), slot_display(frame["technical_constraints"]["modality"])),
                    (t("time_range"), slot_display(frame["document_attributes"]["time_range"], "document_attributes.time_range")),
                    (t("paper_type"), slot_display(frame["document_attributes"]["paper_type"], "document_attributes.paper_type")),
                    (t("author_name"), slot_display(frame["document_attributes"]["author_name"], "")),
                    (t("title_hint"), slot_display(frame["document_attributes"]["title_hint"], "")),
                    (
                        t("result_preferences"),
                        " | ".join(
                            [
                                f"{t('prefer_recent')}={slot_display(frame['result_preferences']['prefer_recent'], 'result_preferences.prefer_recent')}",
                                f"{t('prefer_classic')}={slot_display(frame['result_preferences']['prefer_classic'], 'result_preferences.prefer_classic')}",
                                f"{t('prefer_survey')}={slot_display(frame['result_preferences']['prefer_survey'], 'result_preferences.prefer_survey')}",
                                f"{t('prefer_diverse')}={slot_display(frame['result_preferences']['prefer_diverse'], 'result_preferences.prefer_diverse')}",
                                f"{t('need_explainable_reason')}={slot_display(frame['result_preferences']['need_explainable_reason'], 'result_preferences.need_explainable_reason')}",
                            ]
                        ),
                    ),
                ]
            )

    clarification_question = resolve_follow_up_question(frame, follow_up_suggestion)
    if frame.get("clarification_needed") and clarification_question:
        st.warning(clarification_question)

    with st.container(border=True):
        with st.expander(t("intentframe_raw_json")):
            st.json(frame)


def render_gap_panel(
    gap_report: Dict[str, Any],
    clarification_needed: bool,
    frame: Dict[str, Any],
    follow_up_suggestion: Dict[str, Any] | None = None,
) -> None:
    left_col, right_col = st.columns([1.02, 1.15], gap="large")
    with left_col:
        with st.container(border=True):
            render_panel_lead(t("gap_analysis"), "系统说明当前结果为什么还偏宽，以及下一步该补充什么。", "缺口诊断")
            render_string_list(
                t("next_answer_helpful"),
                gap_report.get("what_next_answer_would_improve", []),
                t("results_usable"),
            )
            with st.expander(t("gap_details"), expanded=clarification_needed):
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
    with right_col:
        with st.container(border=True):
            render_panel_lead(t("follow_up_workspace"), "优先补充最影响重排收敛的信息。", "追问补充")
            if clarification_needed:
                st.warning(t("clarification_needed_caption"))
            else:
                st.caption(t("clarification_optional_caption"))
            suggestion_payload = follow_up_suggestion or {}
            clarification_question = resolve_follow_up_question(frame, suggestion_payload)
            if clarification_question:
                st.info(clarification_question)
            suggested_reply = clean_text(suggestion_payload.get("draft", ""))
            if suggested_reply:
                st.success(f"{t('suggested_follow_up')}（{describe_follow_up_source(suggestion_payload)}）")
                st.write(suggested_reply)
                if clean_text(st.session_state.get("gap_follow_up_input", "")) == suggested_reply:
                    st.caption("这条追问语句已自动写入下方入口，可直接继续检索。")
                suggestion_meta = []
                rationale = clean_text(suggestion_payload.get("rationale", ""))
                if rationale:
                    suggestion_meta.append(rationale)
                used_model = clean_text(suggestion_payload.get("used_model", ""))
                if used_model:
                    suggestion_meta.append(f"模型：{used_model}")
                if suggestion_meta:
                    st.caption(" | ".join(suggestion_meta))
                if st.button(t("fill_suggested_follow_up"), key="fill_gap_follow_up", use_container_width=True):
                    st.session_state["gap_follow_up_input"] = suggested_reply
                    st.session_state["_auto_gap_follow_up_value"] = suggested_reply
            st.markdown(f"**{t('follow_up_entry')}**")
            st.text_area(
                t("supplementary_reply"),
                key="gap_follow_up_input",
                height=100,
                placeholder=suggested_reply or t("supplementary_reply_placeholder"),
            )
            if st.button(t("continue_search"), key="run_gap_follow_up", use_container_width=True):
                reply = clean_text(st.session_state.get("gap_follow_up_input", ""))
                if not reply:
                    st.warning(t("fill_reply_warning"))
                else:
                    queue_run_request(follow_up_override=reply)
                    st.rerun()


def toggle_saved_state(paper_id: str, saved_ids: set[str]) -> None:
    if paper_id in saved_ids:
        unsave_paper(paper_id)
    else:
        save_paper(paper_id)
    st.rerun()


# 结果摘要与详情共用同一套来源标签，避免前后信息不一致。
def build_result_metadata(result: Dict[str, Any]) -> List[str]:
    metadata: List[str] = []
    if result.get("matched_field"):
        metadata.append(t("matched_field_label", value=localize_field_text(result.get("matched_field", ""))))
    if result.get("exact_match_type"):
        metadata.append(t("matched_type_label", value=localize_match_type_text(result.get("exact_match_type", ""))))
    metadata.append(t("result_source_label", value=describe_result_source(result)))
    used_model = clean_text(result.get("used_model", ""))
    if used_model:
        metadata.append(t("result_model_label", value=used_model))
    return metadata


def build_explanation_caption(result: Dict[str, Any]) -> str:
    source = describe_result_source(result)
    used_model = clean_text(result.get("used_model", ""))
    if used_model:
        return f"LLM 排序解释来源：{source} | 模型：{used_model}"
    if source and source != "-":
        return f"排序解释来源：{source}"
    return ""


# Top-K 摘要默认直接展开，优先给出可快速扫描的重点信息。
def render_result_summary_card(
    rank: int,
    result: Dict[str, Any],
    saved_ids: set[str],
) -> None:
    query_match = result.get("query_paper_match") or {}
    matched_dimensions = query_match.get("matched_dimensions", [])
    label = t("unsave_paper") if result["paper_id"] in saved_ids else t("save_paper")
    with st.container(border=True):
        header_left, header_right = st.columns([6, 1], gap="medium")
        with header_left:
            paper_type = translate_mapping_value(result.get("paper_type", ""), PAPER_TYPE_LABELS) or "论文"
            st.markdown(
                f"<div class='pc-result-kicker'>结果 {rank} · {html.escape(paper_type)} · 最终得分 {result.get('final_score', 0):.3f}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"#### {result['title']}")
            st.markdown(
                f"<div class='pc-inline-meta'>{html.escape(format_authors_for_display(result.get('authors_raw', '')))} | {html.escape(clean_text(result.get('year_month', '')))} | 匹配分 {query_match.get('match_score', 0):.3f}</div>",
                unsafe_allow_html=True,
            )
        with header_right:
            if st.button(label, key=f"save_toggle_summary_{result['paper_id']}", use_container_width=True):
                toggle_saved_state(result["paper_id"], saved_ids)

        metadata = build_result_metadata(result)
        if metadata:
            render_chip_row(metadata, tone="soft")

        summary_chips = [
            f"主意图{'已满足' if query_match.get('main_intent_satisfied') else '未满足'}",
            f"时间编码 {clean_text(result.get('year_month', ''))}",
        ]
        render_chip_row(summary_chips, tone="good" if query_match.get("main_intent_satisfied") else "soft")

        st.markdown(f"**{t('ranking_explanation')}**")
        explanation_caption = build_explanation_caption(result)
        if explanation_caption:
            st.caption(explanation_caption)
        st.write(query_match.get("brief_reason", t("no_match_explanation")))
        render_string_list(t("ranking_reasons"), result.get("ranking_reasons", []), t("no_ranking_reasons"))
        render_string_list(
            t("matched_dimensions"),
            matched_dimensions,
            t("no_matched_dimensions"),
            formatter=localize_dimension_text,
        )
        render_string_list(t("unmet_constraints"), result.get("unmet_constraints", []), t("no_unmet_constraints"))
        abstract_preview = truncate_text(result.get("abstract", ""))
        if abstract_preview:
            st.markdown(f"**{t('abstract_preview')}**")
            st.write(abstract_preview)
        st.caption(t("detail_expand_hint"))


# 论文详情区保留完整摘要、命中证据和可调试的结构化输出。
def render_result_detail_card(
    rank: int,
    result: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    saved_ids: set[str],
    show_raw_json: bool,
) -> None:
    query_match = result.get("query_paper_match") or {}
    label = t("unsave_paper") if result["paper_id"] in saved_ids else t("save_paper")
    with st.expander(f"Top {rank}. {result['title']}", expanded=rank == 1):
        with st.container(border=True):
            header_left, header_right = st.columns([6, 1], gap="medium")
            with header_left:
                st.markdown(
                    f"<div class='pc-inline-meta'>{html.escape(format_authors_for_display(result.get('authors_raw', '')))} | {html.escape(clean_text(result.get('year_month', '')))} | 最终得分 {result.get('final_score', 0):.3f} | 匹配分 {query_match.get('match_score', 0):.3f}</div>",
                    unsafe_allow_html=True,
                )
            with header_right:
                if st.button(label, key=f"save_toggle_detail_{result['paper_id']}", use_container_width=True):
                    toggle_saved_state(result["paper_id"], saved_ids)

            metadata = build_result_metadata(result)
            if metadata:
                render_chip_row(metadata, tone="soft")

            body_left, body_right = st.columns([1.15, 0.95], gap="large")
            with body_left:
                render_string_list(t("ranking_reasons"), result.get("ranking_reasons", []), t("no_ranking_reasons"))
                render_string_list(
                    t("matched_dimensions"),
                    query_match.get("matched_dimensions", []),
                    t("no_matched_dimensions"),
                    formatter=localize_dimension_text,
                )
                render_string_list(t("unmet_constraints"), result.get("unmet_constraints", []), t("no_unmet_constraints"))
                st.markdown(f"**{t('abstract')}**")
                st.write(result.get("abstract", ""))
            with body_right:
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
                explanation_caption = build_explanation_caption(result)
                if explanation_caption:
                    st.caption(explanation_caption)
                st.write(query_match.get("brief_reason", t("no_match_explanation")))

            with st.expander(t("semantic_card")):
                st.json(evidence_pack.get("semantic_card", {}))

            with st.expander(t("query_paper_match")):
                st.json(query_match)

            if show_raw_json:
                with st.expander(t("raw_result_json")):
                    st.json(result)


# 管理区集中展示收藏、历史和标准演示。
def render_management_area(demo_queries: List[Dict[str, Any]], *, show_header: bool = True) -> None:
    if show_header:
        st.subheader(t("management_area"))
    saved_col, history_col, demos_col = st.columns([1.05, 1.2, 1.1], gap="large")

    with saved_col:
        with st.container(border=True):
            render_panel_lead(t("saved_papers_tab"), "收藏后的论文会沉淀在这里，便于后续写作和比对。", "管理区")
            saved_items = list_saved_papers(limit=50)
            if not saved_items:
                st.info(t("no_saved_papers"))
            else:
                render_string_list(
                    t("saved_papers_tab"),
                    [f"{item['title']} | {format_authors_for_display(item['authors_raw'])} | {item['year_month']}" for item in saved_items],
                    t("no_saved_papers"),
                )

    with history_col:
        with st.container(border=True):
            render_panel_lead(t("history_tab"), "按时间回看自然语言查询与追问记录。", "管理区")
            history_items = list_search_history(limit=20)
            if not history_items:
                st.info(t("no_history"))
            else:
                render_string_list(
                    t("history_tab"),
                    [f"{item['created_at']} | {item['query_text']}" for item in history_items],
                    t("no_history"),
                )

    with demos_col:
        with st.container(border=True):
            render_panel_lead(t("standard_demos_tab"), "用于课堂演示、论文截图和固定样例回放。", "管理区")
            for index, item in enumerate(demo_queries, start=1):
                query_label = truncate_text(item["query"], 46)
                if st.button(f"{index}. {query_label}", key=f"replay_demo_{index}", use_container_width=True):
                    apply_demo_query(item, auto_run=True)
                    st.rerun()
                if item.get("follow_up_reply"):
                    st.caption(t("follow_up_reply_label", value=item["follow_up_reply"]))


# 收集界面输入并执行一次完整主链路。
def run_query(*, query_override: str = "", follow_up_override: str = "") -> None:
    previous_payload = st.session_state.get("latest_payload")
    query = clean_text(query_override or st.session_state.get("query_input", ""))
    follow_up = clean_text(follow_up_override or st.session_state.get("follow_up_input", ""))
    top_k = int(st.session_state.get("top_k_input", 5))
    candidate_pool = int(st.session_state.get("candidate_pool_input", 120))
    explain_limit = int(st.session_state.get("explain_limit_input", 5))
    if not query:
        st.warning(t("enter_query_first"))
        return
    st.session_state["_pending_live_process_scroll"] = True
    scroll_placeholder = st.empty()
    progress_placeholder = st.empty()
    stage_events: List[Dict[str, Any]] = []

    with scroll_placeholder.container():
        render_anchor_autoscroll(LIVE_PROCESS_SCROLL_ANCHOR)

    def update_progress(event: Dict[str, Any]) -> None:
        stage_events.append(dict(event))
        st.session_state["latest_stage_events"] = list(stage_events)
        progress_placeholder.markdown(
            build_stage_events_markup(stage_events, live=True),
            unsafe_allow_html=True,
        )

    try:
        progress_placeholder.markdown(
            build_stage_events_markup(
                [
                    {
                        "stage": "bootstrap",
                        "label": t("process_preparing"),
                        "status": "running",
                    }
                ],
                live=True,
            ),
            unsafe_allow_html=True,
        )
        payload = run_project_chain_session(
            query=query,
            follow_up_reply=follow_up or None,
            top_k=top_k,
            candidate_pool_size=candidate_pool,
            explain_limit=max(top_k, explain_limit),
            stage_callback=update_progress,
        )
    except Exception as exc:
        scroll_placeholder.empty()
        progress_placeholder.empty()
        st.session_state.pop("latest_payload", None)
        st.error(str(exc) or t("run_failed"))
        return
    scroll_placeholder.empty()
    progress_placeholder.empty()
    st.session_state["latest_stage_events"] = payload.get("stage_events", stage_events)
    if (
        previous_payload
        and clean_text(previous_payload.get("query", "")) == query
        and clean_text(previous_payload.get("follow_up_reply", "")) != clean_text(payload.get("follow_up_reply", ""))
        and clean_text(payload.get("follow_up_reply", ""))
    ):
        payload["result_change_summary"] = build_result_change_summary(previous_payload, payload)
    st.session_state["latest_payload"] = payload
    sync_generated_follow_up_entry(payload)


# 页面总入口：负责状态检查、表单渲染和结果展示。
def main() -> None:
    dataset_info = get_active_dataset_info()
    st.session_state.setdefault("ui_language", "zh")
    st.set_page_config(page_title="PaperCompass", layout="wide")
    inject_runtime_process_styles()
    st.title("PaperCompass")
    st.caption(t("page_caption", dataset=str(dataset_info.get("label", "Dataset"))))
    apply_pending_query_state()

    if not project_database_exists():
        st.error(t("database_missing"))
        return
    start_project_query_runtime_warmup()

    stats = load_project_stats()
    if stats.get("papers", 0) <= 0:
        st.error(t("database_empty"))
        st.caption(t("current_database", path=relative_to_project(get_default_db_path())))
        return

    app_state = load_app_state()
    demo_queries = load_demo_queries()

    render_hero_banner(stats)
    render_section_header(1, t("search_input"), t("search_input_caption"), anchor_id=STEP_SECTION_ANCHORS[1])
    st.markdown(
        f"<div class='pc-note'>{html.escape(t('search_status', db_path=relative_to_project(get_default_db_path()), papers=stats.get('papers', 0), llm_status=t('api_key_configured') if OPENAI_API_KEY else t('api_key_unconfigured')))}</div>",
        unsafe_allow_html=True,
    )
    render_search_workspace(stats, demo_queries)
    render_section_header(
        2,
        f"{t('run_process')} / {t('model_workbench')}",
        t("runtime_workspace_caption"),
        anchor_id=STEP_SECTION_ANCHORS[2],
    )
    st.markdown(f"<div id='{LIVE_PROCESS_SCROLL_ANCHOR}'></div>", unsafe_allow_html=True)
    if st.session_state.pop("_pending_live_process_scroll", False):
        render_anchor_autoscroll(LIVE_PROCESS_SCROLL_ANCHOR)

    pending_run_query = st.session_state.pop("_pending_run_query", False)
    pending_query_override = st.session_state.pop("_pending_run_query_override", "")
    pending_follow_up_override = st.session_state.pop("_pending_run_follow_up_override", "")
    if st.session_state.pop("_pending_auto_run_query", False) or pending_run_query:
        run_query(query_override=pending_query_override, follow_up_override=pending_follow_up_override)

    payload = st.session_state.get("latest_payload")
    if payload:
        sync_generated_follow_up_entry(payload)
    show_raw_json = render_sidebar(demo_queries, bool(payload))
    render_runtime_workspace(stats, payload, bool(app_state))

    if not payload:
        render_section_header(
            7,
            t("management_workspace"),
            t("management_workspace_caption"),
            anchor_id=STEP_SECTION_ANCHORS[7],
        )
        render_management_area(demo_queries, show_header=False)
        return

    render_section_header(
        3,
        t("system_understanding"),
        t("system_understanding_caption"),
        anchor_id=STEP_SECTION_ANCHORS[3],
    )
    render_intent_panel(payload["final_intent_frame"], payload.get("follow_up_suggestion"))

    render_section_header(
        4,
        t("follow_up_workspace"),
        t("follow_up_workspace_caption"),
        anchor_id=STEP_SECTION_ANCHORS[4],
    )
    render_follow_up_convergence(payload)
    render_gap_panel(
        payload["intent_gap_report"],
        clarification_needed=bool(payload.get("final_intent_frame", {}).get("clarification_needed")),
        frame=payload["final_intent_frame"],
        follow_up_suggestion=payload.get("follow_up_suggestion"),
    )

    render_section_header(
        5,
        t("topk_recommendations"),
        t("topk_recommendations_caption"),
        anchor_id=STEP_SECTION_ANCHORS[5],
    )
    render_result_change_summary(payload)
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
        for rank, result in enumerate(payload["top_k_results"], start=1):
            render_result_summary_card(rank, result, saved_ids)
            if rank < len(payload["top_k_results"]):
                st.divider()

    render_section_header(
        6,
        t("paper_details"),
        t("paper_details_caption"),
        anchor_id=STEP_SECTION_ANCHORS[6],
    )
    if not payload.get("top_k_results"):
        st.info(t("no_results"))
    else:
        for rank, result in enumerate(payload["top_k_results"], start=1):
            evidence_pack = payload.get("paper_evidence_packs", {}).get(result["paper_id"], {})
            render_result_detail_card(rank, result, evidence_pack, saved_ids, show_raw_json)

    if show_raw_json:
        with st.expander(t("full_pipeline_json")):
            st.json(payload)

    render_section_header(
        7,
        t("management_workspace"),
        t("management_workspace_caption"),
        anchor_id=STEP_SECTION_ANCHORS[7],
    )
    render_management_area(demo_queries, show_header=False)


if __name__ == "__main__":
    main()
