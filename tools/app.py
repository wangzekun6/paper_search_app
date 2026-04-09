"""
PaperCompass 的 Streamlit 可视化入口。

这个文件负责把检索主链路的输入、意图解析结果、Gap 分析、排序结果、
语义卡片以及收藏/历史管理整合成一个可交互页面。
"""

from __future__ import annotations

import html
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
    "sidebar_hint": {"zh": "侧边栏仅保留界面设置；系统状态、LLM 状态和示例已移到主页面。", "en": "The sidebar only keeps view settings. System status, LLM status, and demos now live on the main page."},
    "sidebar_demo_hint": {"zh": "点击左侧示例可直接回填到输入框，不会自动检索。", "en": "Click a sidebar demo to refill the inputs without running the search automatically."},
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

STEP_SECTION_ANCHORS = {
    1: "step-1-search-input",
    2: "step-2-runtime-workbench",
    3: "step-3-system-understanding",
    4: "step-4-follow-up",
    5: "step-5-topk-results",
    6: "step-6-paper-details",
    7: "step-7-management",
}


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
    box-shadow: 0 22px 44px rgba(42, 157, 143, 0.18);
}
.pc-process-live-dock {
    position: fixed;
    left: 50%;
    top: 58%;
    transform: translate(-50%, -50%);
    width: min(820px, 74vw);
    z-index: 999999;
    pointer-events: none;
}
.pc-process-live-dock::before {
    content: "";
    position: absolute;
    inset: -24px -30px -28px;
    border-radius: 30px;
    background:
        radial-gradient(circle at center, rgba(42, 157, 143, 0.22), transparent 56%),
        radial-gradient(circle at top right, rgba(244, 162, 97, 0.20), transparent 42%);
    filter: blur(14px);
    z-index: 0;
}
.pc-process-live-dock .pc-process-shell {
    position: relative;
    z-index: 1;
    margin-bottom: 0;
    border-color: rgba(42, 157, 143, 0.30);
    background:
        radial-gradient(circle at top right, rgba(233, 196, 106, 0.20), transparent 34%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(246, 248, 249, 0.92));
    backdrop-filter: blur(16px);
    box-shadow: 0 30px 78px rgba(19, 42, 51, 0.22);
}
.pc-process-live-dock .pc-process-title {
    font-size: 1.08rem;
}
.pc-process-live-dock .pc-process-chip {
    background: rgba(255, 255, 255, 0.72);
}
.pc-process-live-dock .pc-stage-list {
    max-height: min(38vh, 360px);
    overflow: auto;
    padding-right: 4px;
}
.pc-process-live-dock .pc-stage-card.is-running {
    transform: scale(1.01);
}
.pc-process-live-dock .pc-stage-list::-webkit-scrollbar {
    width: 8px;
}
.pc-process-live-dock .pc-stage-list::-webkit-scrollbar-thumb {
    background: rgba(38, 70, 83, 0.18);
    border-radius: 999px;
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
    height: 11px;
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
.pc-stage-card.is-running::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,0.78) 40%, transparent 75%);
    animation: pc-stage-shine 1.6s linear infinite;
}
.pc-stage-card.is-running .pc-stage-status-dot {
    background: #f4a261;
    box-shadow: 0 0 0 0 rgba(244, 162, 97, 0.45);
    animation: pc-dot-pulse 1.25s ease-in-out infinite;
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
    .pc-process-live-dock {
        width: 92vw;
        top: 62%;
    }
    .pc-process-live-dock .pc-process-shell {
        padding: 14px 14px 12px;
    }
    .pc-process-live-dock .pc-stage-list {
        max-height: min(34vh, 300px);
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
    st.markdown(f"**{title}**")
    if not normalized:
        st.caption(empty_text)
        return
    for item in normalized:
        st.write(f"- {item}")


def queue_query_state(
    *,
    query: str = "",
    follow_up: str = "",
    top_k: int = 5,
    candidate_pool: int = 40,
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
    shell_markup = "".join(
        [
            f"<div class='{shell_class}'>",
            "<div class='pc-process-banner'>",
            "<div class='pc-process-banner-copy'>",
            f"<div class='pc-process-kicker'>{html.escape(t('run_process'))}</div>",
            f"<div class='pc-process-title'>{html.escape(title)}</div>",
            f"<div class='pc-process-subtitle'>{html.escape(subtitle)}</div>",
            "</div>",
            "<div class='pc-process-meta'>",
            f"<div class='pc-process-chip'>{html.escape(t('process_stage_progress', completed=current_stage_step, total=total_steps))}</div>",
            f"<div class='pc-process-chip'>{html.escape(hint)}</div>",
            "</div>",
            "</div>",
            "<div class='pc-process-progress-track'>",
            f"<div class='pc-process-progress-fill' style='width:{progress_width};'></div>",
            "</div>",
            "<div class='pc-stage-list'>",
            "".join(cards),
            "</div>",
            "</div>",
        ]
    )
    if is_live_running:
        return f"<div class='pc-process-live-dock'>{shell_markup}</div>"
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
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


# 将标准示例查询写回界面状态；可选择仅回填，或回填后自动运行。
def apply_demo_query(item: Dict[str, Any], *, auto_run: bool = True) -> None:
    queue_query_state(
        query=item["query"],
        follow_up=item.get("follow_up_reply", ""),
        top_k=5,
        candidate_pool=40,
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
def render_sidebar(standard_queries: List[Dict[str, Any]], has_payload: bool) -> bool:
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
        for index, item in enumerate(standard_queries[:6], start=1):
            if st.button(t("demo_button", index=index), key=f"sidebar_demo_{index}", use_container_width=True):
                apply_demo_query(item, auto_run=False)
                st.rerun()
        return st.checkbox(t("show_raw_json"), value=False)


# 首页概览区域展示数据库和语义层的总体统计。
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


def render_system_status_panel(stats: Dict[str, int], state_loaded: bool) -> None:
    st.markdown(f"**{t('system_status')}**")
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
    if state_loaded:
        st.caption(t("state_loaded"))


def render_llm_runtime_panel() -> None:
    st.markdown(f"**{t('llm_runtime')}**")
    if OPENAI_API_KEY:
        st.success(t("api_key_detected"))
    else:
        st.error(t("api_key_missing"))
    st.caption(t("base_url", value=OPENAI_API_BASE))
    st.caption(t("model_label", value=OPENAI_MODEL))
    if st.button(t("test_api"), key="test_api_workbench", use_container_width=True):
        with st.spinner(t("testing_api")):
            ok, message = test_openai_api(OPENAI_API_KEY)
        if ok:
            st.success(message)
        else:
            st.error(message)


def render_runtime_workspace(stats: Dict[str, int], payload: Dict[str, Any] | None, state_loaded: bool) -> None:
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
        status_col, llm_col = st.columns([3, 2])
        with status_col:
            render_system_status_panel(stats, state_loaded)
        with llm_col:
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

    return {
        "headline": t(headline_key),
        "overlap_count": len(overlap_ids),
        "current_count": len(current_ids),
        "added_titles": [current_titles.get(paper_id, paper_id) for paper_id in added_ids],
        "removed_titles": [previous_titles.get(paper_id, paper_id) for paper_id in removed_ids],
        "reordered_titles": reordered_items,
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
            ("time_range", frame["document_attributes"]["time_range"], "document_attributes.time_range"),
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


# Gap 面板解释当前结果为什么还宽，以及下一步补什么最有效。
def render_gap_panel(
    gap_report: Dict[str, Any],
    clarification_needed: bool,
    frame: Dict[str, Any],
    follow_up_suggestion: Dict[str, Any] | None = None,
) -> None:
    render_string_list(
        t("next_answer_helpful"),
        gap_report.get("what_next_answer_would_improve", []),
        t("results_usable"),
    )
    if clarification_needed:
        st.warning(t("clarification_needed_caption"))
    else:
        st.caption(t("clarification_optional_caption"))
    suggestion_payload = follow_up_suggestion or {}
    suggested_reply = clean_text(suggestion_payload.get("draft", "")) or build_follow_up_draft(frame, gap_report)
    if suggested_reply:
        generator = clean_text(suggestion_payload.get("generator", ""))
        source_label = {"llm": "LLM 生成", "rule": "规则兜底"}.get(generator, "系统建议")
        st.success(f"{t('suggested_follow_up')}（{source_label}）")
        st.write(suggested_reply)
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
    st.markdown(f"**{t('follow_up_entry')}**")
    st.text_area(
        t("supplementary_reply"),
        key="gap_follow_up_input",
        height=80,
        placeholder=suggested_reply or t("supplementary_reply_placeholder"),
    )
    if st.button(t("continue_search"), key="run_gap_follow_up", use_container_width=True):
        reply = clean_text(st.session_state.get("gap_follow_up_input", ""))
        if not reply:
            st.warning(t("fill_reply_warning"))
        else:
            run_query(follow_up_override=reply)
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


# 收藏动作会直接同步到数据库，并刷新页面状态。
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


# Top-K 摘要默认直接展开，优先给出可快速扫描的重点信息。
def render_result_summary_card(
    rank: int,
    result: Dict[str, Any],
    saved_ids: set[str],
) -> None:
    query_match = result.get("query_paper_match") or {}
    matched_dimensions = query_match.get("matched_dimensions", [])
    label = t("unsave_paper") if result["paper_id"] in saved_ids else t("save_paper")
    header_left, header_right = st.columns([6, 1])
    with header_left:
        st.markdown(f"#### Top {rank}. {result['title']}")
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
        if st.button(label, key=f"save_toggle_summary_{result['paper_id']}", use_container_width=True):
            toggle_saved_state(result["paper_id"], saved_ids)

    metadata = build_result_metadata(result)
    if metadata:
        st.caption(" | ".join(metadata))

    st.markdown(f"**{t('ranking_explanation')}**")
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
        header_left, header_right = st.columns([6, 1])
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
            if st.button(label, key=f"save_toggle_detail_{result['paper_id']}", use_container_width=True):
                toggle_saved_state(result["paper_id"], saved_ids)

        metadata = build_result_metadata(result)
        if metadata:
            st.caption(" | ".join(metadata))

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

        with st.expander(t("semantic_card")):
            st.json(evidence_pack.get("semantic_card", {}))

        with st.expander(t("query_paper_match")):
            st.json(query_match)

        if show_raw_json:
            with st.expander(t("raw_result_json")):
                st.json(result)


# 管理区集中展示收藏、历史和标准演示。
def render_management_area(standard_queries: List[Dict[str, Any]], *, show_header: bool = True) -> None:
    if show_header:
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
                    apply_demo_query(item, auto_run=True)
                    st.rerun()


# 收集界面输入并执行一次完整主链路。
def run_query(*, query_override: str = "", follow_up_override: str = "") -> None:
    previous_payload = st.session_state.get("latest_payload")
    query = clean_text(query_override or st.session_state.get("query_input", ""))
    follow_up = clean_text(follow_up_override or st.session_state.get("follow_up_input", ""))
    top_k = int(st.session_state.get("top_k_input", 5))
    candidate_pool = int(st.session_state.get("candidate_pool_input", 40))
    explain_limit = int(st.session_state.get("explain_limit_input", 5))
    if not query:
        st.warning(t("enter_query_first"))
        return
    progress_placeholder = st.empty()
    stage_events: List[Dict[str, Any]] = []

    def update_progress(event: Dict[str, Any]) -> None:
        stage_events.append(dict(event))
        st.session_state["latest_stage_events"] = list(stage_events)
        preview = stage_events[-5:]
        progress_placeholder.markdown(
            build_stage_events_markup(preview, live=True),
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
        progress_placeholder.empty()
        st.session_state.pop("latest_payload", None)
        st.error(str(exc) or t("run_failed"))
        return
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


# 页面总入口：负责状态检查、表单渲染和结果展示。
def main() -> None:
    st.session_state.setdefault("ui_language", "zh")
    st.set_page_config(page_title="PaperCompass", layout="wide")
    inject_runtime_process_styles()
    st.title("PaperCompass")
    st.caption(t("page_caption", dataset=DATASET_LABEL))
    apply_pending_query_state()

    if not project_database_exists():
        st.error(t("database_missing"))
        return

    stats = load_project_stats()
    if stats.get("papers", 0) <= 0:
        st.error(t("database_empty"))
        st.caption(t("current_database", path=relative_to_project(get_default_db_path())))
        return

    app_state = load_app_state()
    standard_queries = load_standard_queries()

    render_section_header(1, t("search_input"), t("search_input_caption"), anchor_id=STEP_SECTION_ANCHORS[1])
    st.caption(
        t(
            "search_status",
            db_path=relative_to_project(get_default_db_path()),
            papers=stats.get("papers", 0),
            llm_status=t("api_key_configured") if OPENAI_API_KEY else t("api_key_unconfigured"),
        )
    )
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
    with st.expander(t("search_config"), expanded=False):
        config_col1, config_col2, config_col3 = st.columns(3)
        with config_col1:
            st.number_input("Top-K", min_value=3, max_value=10, value=5, step=1, key="top_k_input")
        with config_col2:
            st.number_input(
                t("candidate_pool_size"),
                min_value=20,
                max_value=120,
                value=40,
                step=10,
                key="candidate_pool_input",
            )
        with config_col3:
            st.number_input(t("explain_limit"), min_value=3, max_value=10, value=5, step=1, key="explain_limit_input")
    st.button(t("run_search"), type="primary", use_container_width=True, on_click=run_query)

    if st.session_state.pop("_pending_auto_run_query", False):
        run_query()

    payload = st.session_state.get("latest_payload")
    show_raw_json = render_sidebar(standard_queries, bool(payload))
    render_section_header(
        2,
        f"{t('run_process')} / {t('model_workbench')}",
        t("runtime_workspace_caption"),
        anchor_id=STEP_SECTION_ANCHORS[2],
    )
    render_runtime_workspace(stats, payload, bool(app_state))

    if not payload:
        render_section_header(
            7,
            t("management_workspace"),
            t("management_workspace_caption"),
            anchor_id=STEP_SECTION_ANCHORS[7],
        )
        render_management_area(standard_queries, show_header=False)
        return

    render_section_header(
        3,
        t("system_understanding"),
        t("system_understanding_caption"),
        anchor_id=STEP_SECTION_ANCHORS[3],
    )
    render_intent_panel(payload["final_intent_frame"])

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
    render_management_area(standard_queries, show_header=False)


if __name__ == "__main__":
    main()
