# 核心链路演示说明

## 固定链路
`query -> IntentFrame -> 聚合追问 -> hybrid recall -> evidence pack -> query-paper match -> gap report -> top-K explanation`

## 关键输出文件
- 标准查询集：`system_outputs/demos/standard_queries.json`
- 演示结果：`system_outputs/demos/demo_runs.json`
- gap 报告：`system_outputs/eval/gap_reports.json`
- 排序评估：`system_outputs/eval/ranking_eval.json`
- 解释样例：`system_outputs/eval/explanation_samples.json`
- 回归报告：`system_outputs/eval/regression_report.json`

## 标准查询说明
1. 查询：`retrieval augmented generation survey`
   补充回复：recent two years, explain why each paper matches
   预期关键槽位：检索场景=survey_lookup；研究任务=retrieval-augmented generation；论文类型=survey
   预期追问方向：时间范围
   预期 top 结果类型：survey
2. 查询：`recent agent memory papers`
   补充回复：无
   预期关键槽位：检索场景=recent_progress；研究问题=memory mechanism
   预期追问方向：论文类型
   预期 top 结果类型：method
3. 查询：`papers by authors of MALT`
   补充回复：explain why they are related
   预期关键槽位：检索场景=author_trace
   预期追问方向：作者
   预期 top 结果类型：author_trace
4. 查询：`quality estimation with COMET`
   补充回复：prefer recent work
   预期关键槽位：检索场景=method_constrained_search；指标约束=COMET
   预期追问方向：数据集约束
   预期 top 结果类型：method
5. 查询：`long context survey papers`
   补充回复：无
   预期关键槽位：检索场景=survey_lookup；论文类型=survey
   预期追问方向：无需追问
   预期 top 结果类型：survey
6. 查询：`benchmark for large language models on reasoning`
   补充回复：无
   预期关键槽位：研究任务=reasoning；论文类型=benchmark
   预期追问方向：时间范围
   预期 top 结果类型：benchmark
7. 查询：`multimodal reasoning papers`
   补充回复：prefer diverse results
   预期关键槽位：研究领域=multimodal NLP；研究任务=reasoning
   预期追问方向：论文类型
   预期 top 结果类型：method
8. 查询：`Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey`
   补充回复：无
   预期关键槽位：检索场景=specific_paper_lookup
   预期追问方向：无需追问
   预期 top 结果类型：specific_paper_lookup
9. 查询：`early exit for quality estimation`
   补充回复：无
   预期关键槽位：方法约束=early exit；研究任务=quality estimation
   预期追问方向：时间范围
   预期 top 结果类型：method
10. 查询：`translation quality estimation explainable reason`
   补充回复：无
   预期关键槽位：研究任务=quality estimation；需要可解释理由=yes
   预期追问方向：数据集约束；时间范围
   预期 top 结果类型：method
