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
1. Query: `retrieval augmented generation survey`
   Follow-up: recent two years, explain why each paper matches
   Expected intent slots: Search scene=survey_lookup; Research task=retrieval-augmented generation; Paper type=survey
   Expected clarification focus: Time range
   Expected top result type: survey
2. Query: `recent agent memory papers`
   Follow-up: none
   Expected intent slots: Search scene=recent_progress; Research problem=memory mechanism
   Expected clarification focus: Paper type
   Expected top result type: method
3. Query: `papers by authors of MALT`
   Follow-up: explain why they are related
   Expected intent slots: Search scene=author_trace
   Expected clarification focus: Author
   Expected top result type: author_trace
4. Query: `quality estimation with COMET`
   Follow-up: prefer recent work
   Expected intent slots: Search scene=method_constrained_search; Metric constraint=COMET
   Expected clarification focus: Dataset constraint
   Expected top result type: method
5. Query: `long context survey papers`
   Follow-up: none
   Expected intent slots: Search scene=survey_lookup; Paper type=survey
   Expected clarification focus: none
   Expected top result type: survey
6. Query: `benchmark for large language models on reasoning`
   Follow-up: none
   Expected intent slots: Research task=reasoning; Paper type=benchmark
   Expected clarification focus: Time range
   Expected top result type: benchmark
7. Query: `multimodal reasoning papers`
   Follow-up: prefer diverse results
   Expected intent slots: Research domain=multimodal NLP; Research task=reasoning
   Expected clarification focus: Paper type
   Expected top result type: method
8. Query: `Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey`
   Follow-up: none
   Expected intent slots: Search scene=specific_paper_lookup
   Expected clarification focus: none
   Expected top result type: specific_paper_lookup
9. Query: `early exit for quality estimation`
   Follow-up: none
   Expected intent slots: Method constraint=early exit; Research task=quality estimation
   Expected clarification focus: Time range
   Expected top result type: method
10. Query: `translation quality estimation explainable reason`
   Follow-up: none
   Expected intent slots: Research task=quality estimation; Need explainable reason=yes
   Expected clarification focus: Dataset constraint; Time range
   Expected top result type: method
