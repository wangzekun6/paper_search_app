# Core Chain Walkthrough

## Fixed Pipeline
`query -> IntentFrame -> follow-up merge -> hybrid recall -> evidence pack -> query-paper match -> gap report -> top-K explanation`

## Key Output Files
- Standard queries: `system_outputs/demos/standard_queries.json`
- Demo runs: `system_outputs/demos/demo_runs.json`
- Gap reports: `system_outputs/eval/gap_reports.json`
- Ranking eval: `system_outputs/eval/ranking_eval.json`
- Explanation samples: `system_outputs/eval/explanation_samples.json`
- Regression report: `system_outputs/eval/regression_report.json`

## Standard Query Notes
1. Query: `检索 RAG 综述论文`
   Follow-up: 时间范围 2023-2026；聚焦 RAG 幻觉缓解方向；论文类型以综述为主；模型家族、数据集、指标不限；仅文本模态；偏好多样结果否；并解释每篇论文为何匹配。
   Expected intent slots: 检索场景=survey_lookup; 研究任务=retrieval-augmented generation; 研究问题=hallucination mitigation; 论文类型=survey; 偏好多样结果=no; 需要可解释理由=yes
   Expected clarification focus: none
   Expected top result type: survey
2. Query: `最近的 agent memory 论文`
   Follow-up: 时间范围 2023-2026；关注 LLM Agent 长期记忆机制；论文类型方法/基准优先；模型家族、数据集和指标不限；仅文本模态；作者不限；标题线索不限；偏好综述否；偏好多样结果是；并解释命中理由。
   Expected intent slots: 检索场景=recent_progress; 研究问题=memory mechanism; 论文类型=method; 偏好综述=no; 偏好多样结果=yes
   Expected clarification focus: none
   Expected top result type: method
3. Query: `找 MALT 作者的论文`
   Follow-up: MALT 指 Mechanistic Ablation of Lossy Translation；优先该论文作者后续相关工作；时间范围 2023-2026；论文类型 method 与 analysis；偏好最新，不偏好经典，不要求综述；并解释它们之间的关联。
   Expected intent slots: 检索场景=author_trace; 时间范围=2023-2026; 论文类型=method
   Expected clarification focus: none
   Expected top result type: author_trace
4. Query: `用 COMET 做质量估计的论文`
   Follow-up: 时间范围 2023-2026；聚焦 COMET 在质量估计中的使用；数据集不限；作者不限；偏好最新，不偏好经典；偏好综述否；偏好多样结果是；需要可解释理由；并说明每篇与 COMET QE 的关系。
   Expected intent slots: 检索场景=method_constrained_search; 指标约束=COMET; 需要可解释理由=yes; 偏好综述=no
   Expected clarification focus: none
   Expected top result type: method
5. Query: `长上下文论文进展`
   Follow-up: 时间范围 2023-2026；主题聚焦长上下文建模；论文类型不限（可包含综述）；方法约束不限；模型家族、数据集、指标不限；仅文本模态；作者不限；偏好多样结果否；需要可解释理由否。
   Expected intent slots: 检索场景=recent_progress; 时间范围=2023-2026; 偏好多样结果=no; 需要可解释理由=no
   Expected clarification focus: none
   Expected top result type: method
6. Query: `大语言模型推理论文（benchmark 优先）`
   Follow-up: 时间范围 2023-2026；任务聚焦推理评测；benchmark 优先但不限；方法、模型家族、作者、标题线索不限；指标可包含 Pass@1/GSM8K/MATH；偏好多样结果否；需要可解释理由否。
   Expected intent slots: 研究任务=reasoning; 论文类型=benchmark; 时间范围=2023-2026; 偏好多样结果=no; 需要可解释理由=no
   Expected clarification focus: none
   Expected top result type: benchmark
7. Query: `multimodal reasoning papers`
   Follow-up: prefer diverse results
   Expected intent slots: 研究领域=multimodal NLP; 研究任务=reasoning
   Expected clarification focus: 论文类型
   Expected top result type: method
8. Query: `Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey`
   Follow-up: none
   Expected intent slots: 检索场景=specific_paper_lookup
   Expected clarification focus: none
   Expected top result type: specific_paper_lookup
9. Query: `early exit for quality estimation`
   Follow-up: none
   Expected intent slots: 方法约束=early exit; 研究任务=quality estimation
   Expected clarification focus: 时间范围
   Expected top result type: method
10. Query: `translation quality estimation explainable reason`
   Follow-up: none
   Expected intent slots: 研究任务=quality estimation; 需要可解释理由=yes
   Expected clarification focus: 数据集约束; 时间范围
   Expected top result type: method
