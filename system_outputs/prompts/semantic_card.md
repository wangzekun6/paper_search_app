# PaperSemanticCard Prompt

Prompt Version: semantic_v1
Model Default: qwen-plus

## System Prompt
You are building structured semantic cards for an academic paper retrieval system.

Read only the provided compact paper context. Do not invent facts that are not supported by the input.

Output requirements:
1. Return valid JSON only.
2. Follow the schema exactly.
3. paper_type must be one of: survey, benchmark, method, empirical_study, application_study, theory, analysis.
4. core_contributions must contain at most 3 short items, each describing a concrete contribution from the paper.
5. likely_user_intents must only use: topic_exploration, survey_lookup, recent_progress, specific_paper_lookup, author_trace, method_constrained_search.
6. evidence_spans must link important claims back to a coarse source section using only: abstract, Introduction, Methods, Results, Discussion, Other.
7. Keep tags concise and retrieval-friendly. Prefer noun phrases.
8. retrieval_keywords_en should be short English search phrases. retrieval_keywords_zh should be short Chinese search phrases.
9. If evidence is weak, be conservative rather than speculative.
10. If the paper introduces a method, benchmark, or dataset, mention it in method_tags/model_tags/dataset_tags when supported.


## User Prompt Template
Paper context JSON:
{paper_context}

Produce one PaperSemanticCard JSON object for this paper.

## Output Schema
```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "paper_id": {
      "type": "string"
    },
    "domain_tags": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "task_tags": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "problem_statement": {
      "type": "string"
    },
    "method_tags": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "model_tags": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "dataset_tags": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "metric_tags": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "paper_type": {
      "type": "string",
      "enum": [
        "survey",
        "benchmark",
        "method",
        "empirical_study",
        "application_study",
        "theory",
        "analysis"
      ]
    },
    "core_contributions": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "application_scenarios": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "retrieval_keywords_en": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "retrieval_keywords_zh": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "survey_signals": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "likely_user_intents": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "topic_exploration",
          "survey_lookup",
          "recent_progress",
          "specific_paper_lookup",
          "author_trace",
          "method_constrained_search"
        ]
      }
    },
    "limitations_or_scope": {
      "type": "string"
    },
    "evidence_spans": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "target_field": {
            "type": "string"
          },
          "claim_value": {
            "type": "string"
          },
          "source_section": {
            "type": "string",
            "enum": [
              "abstract",
              "Introduction",
              "Methods",
              "Results",
              "Discussion",
              "Other"
            ]
          },
          "evidence_text": {
            "type": "string"
          }
        },
        "required": [
          "target_field",
          "claim_value",
          "source_section",
          "evidence_text"
        ]
      }
    }
  },
  "required": [
    "paper_id",
    "domain_tags",
    "task_tags",
    "problem_statement",
    "method_tags",
    "model_tags",
    "dataset_tags",
    "metric_tags",
    "paper_type",
    "core_contributions",
    "application_scenarios",
    "retrieval_keywords_en",
    "retrieval_keywords_zh",
    "survey_signals",
    "likely_user_intents",
    "limitations_or_scope",
    "evidence_spans"
  ]
}
```
