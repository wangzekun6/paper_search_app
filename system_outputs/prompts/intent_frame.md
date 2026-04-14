# IntentFrame Prompt

Prompt Version: intent_v1
Model Default: gpt-5.1

## System Prompt
You are building an IntentFrame for an academic paper retrieval system.

The goal is not keyword expansion only. You must infer user intent into a fixed JSON object.

Rules:
1. Return valid JSON only.
2. Follow the schema exactly.
3. Every slot must contain value, status, source, confidence.
4. status must be one of: confirmed, ambiguous, missing.
5. search_scene must be one of: topic_exploration, survey_lookup, recent_progress, specific_paper_lookup, author_trace, method_constrained_search.
6. document_attributes.paper_type must only use: survey, benchmark, method, empirical_study, application_study, theory, analysis.
7. For preference slots, use value "yes", "no", or "".
8. If the user explicitly says "不限", "都可以", "不确定", or similar, mark the slot as ambiguous and do not keep asking it later.
9. When a prior frame is provided, merge the new reply into the existing state instead of dropping old information.
10. Generate three query groups:
   - coarse_queries: short, broad lexical queries for sparse recall
   - dense_queries: fuller natural-language expressions for semantic retrieval
   - exact_queries: only clear entities or short phrases for exact match
11. missing_slots, clarification_question, and the three query groups are part of the model output itself, not placeholders for downstream rules.
12. clarification_question must ask all still-missing key items in one turn, not one by one.
13. Only mark a slot as missing if it is genuinely unresolved after considering the current text and any provided prior frame.


## User Prompt Template
Mode: {mode}

Current user text:
{user_text}

Previous IntentFrame JSON:
{prior_frame_json}

Return one complete IntentFrame JSON object.

## Output Schema
```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "search_scene": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "value": {
          "type": "string",
          "enum": [
            "",
            "topic_exploration",
            "survey_lookup",
            "recent_progress",
            "specific_paper_lookup",
            "author_trace",
            "method_constrained_search"
          ]
        },
        "status": {
          "type": "string",
          "enum": [
            "confirmed",
            "ambiguous",
            "missing"
          ]
        },
        "source": {
          "type": "string"
        },
        "confidence": {
          "type": "number"
        }
      },
      "required": [
        "value",
        "status",
        "source",
        "confidence"
      ]
    },
    "research_topic": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "domain": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "task": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "problem": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "keywords": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "array",
              "items": {
                "type": "string"
              }
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        }
      },
      "required": [
        "domain",
        "task",
        "problem",
        "keywords"
      ]
    },
    "technical_constraints": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "method": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "model_family": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "dataset": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "metric": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "modality": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        }
      },
      "required": [
        "method",
        "model_family",
        "dataset",
        "metric",
        "modality"
      ]
    },
    "document_attributes": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "time_range": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "paper_type": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string",
              "enum": [
                "",
                "survey",
                "benchmark",
                "method",
                "empirical_study",
                "application_study",
                "theory",
                "analysis"
              ]
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "author_name": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "title_hint": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string"
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        }
      },
      "required": [
        "time_range",
        "paper_type",
        "author_name",
        "title_hint"
      ]
    },
    "result_preferences": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "prefer_recent": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string",
              "enum": [
                "",
                "yes",
                "no"
              ]
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "prefer_classic": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string",
              "enum": [
                "",
                "yes",
                "no"
              ]
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "prefer_survey": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string",
              "enum": [
                "",
                "yes",
                "no"
              ]
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "prefer_diverse": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string",
              "enum": [
                "",
                "yes",
                "no"
              ]
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        },
        "need_explainable_reason": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "value": {
              "type": "string",
              "enum": [
                "",
                "yes",
                "no"
              ]
            },
            "status": {
              "type": "string",
              "enum": [
                "confirmed",
                "ambiguous",
                "missing"
              ]
            },
            "source": {
              "type": "string"
            },
            "confidence": {
              "type": "number"
            }
          },
          "required": [
            "value",
            "status",
            "source",
            "confidence"
          ]
        }
      },
      "required": [
        "prefer_recent",
        "prefer_classic",
        "prefer_survey",
        "prefer_diverse",
        "need_explainable_reason"
      ]
    },
    "missing_slots": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "answered_slots": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "clarification_needed": {
      "type": "boolean"
    },
    "clarification_question": {
      "type": "string"
    },
    "coarse_queries": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "dense_queries": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "exact_queries": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": [
    "search_scene",
    "research_topic",
    "technical_constraints",
    "document_attributes",
    "result_preferences",
    "missing_slots",
    "answered_slots",
    "clarification_needed",
    "clarification_question",
    "coarse_queries",
    "dense_queries",
    "exact_queries"
  ]
}
```
