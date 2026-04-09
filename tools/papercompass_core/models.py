"""
项目的核心数据契约定义。

这里用 dataclass 固定了几类核心对象：
1. 原始论文记录 RawPaperRecord
2. 检索用的论文索引记录 PaperIndexRecord
3. 章节级记录 PaperSectionRecord
4. 语义卡片 PaperSemanticCard

各条核心流水线都围绕这些对象传递数据，
所以这个文件本质上是在定义“项目内部的统一数据语言”。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


# 保存原始 JSON，便于追溯、调试和详情展示。
# 原始论文 JSON 的轻量封装，主要用于入库和追溯原始来源。
@dataclass
class RawPaperRecord:
    """对应一篇论文的原始 JSON 记录，尽量不做信息损失。"""

    paper_id: str
    source_path: str
    raw_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 保存扁平化后的章节信息，便于章节级证据检索和片段展示。
# 扁平化章节记录，供章节级证据展示和检索使用。
@dataclass
class PaperSectionRecord:
    """对应论文中的一个扁平化章节节点。"""

    paper_id: str
    section_order: int
    section_title: str
    section_type: str
    section_text: str
    section_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 保存主检索索引所需的论文级字段，是检索与语义模块的核心输入对象。
# 论文级索引记录，是检索和语义模块共享的主输入对象。
@dataclass
class PaperIndexRecord:
    """对应论文级索引视图，聚合了检索和模型最常用的核心字段。"""

    paper_id: str
    source_path: str
    year_month: str
    title: str
    authors_raw: str
    normalized_authors: List[str] = field(default_factory=list)
    abstract: str = ""
    section_titles: List[str] = field(default_factory=list)
    intro_text: str = ""
    methods_text: str = ""
    results_text: str = ""
    discussion_text: str = ""
    appendix_titles: List[str] = field(default_factory=list)
    fulltext_for_sparse: str = ""
    embedding_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_storage_row(self) -> Dict[str, Any]:
        row = asdict(self)
        # SQLite 表结构保持扁平，因此列表字段统一序列化为 JSON 字符串。
        # SQLite 表结构保持扁平，列表字段统一序列化成 JSON 字符串再入库。
        row["normalized_authors"] = json.dumps(self.normalized_authors, ensure_ascii=False)
        row["section_titles"] = json.dumps(self.section_titles, ensure_ascii=False)
        row["appendix_titles"] = json.dumps(self.appendix_titles, ensure_ascii=False)
        return row


# 保存语义模块生成的结构化语义标签，后续用于更高层的检索和展示。
# 结构化语义卡片，用于高层检索、解释和展示。
@dataclass
class PaperSemanticCard:
    """对应一篇论文的结构化语义卡片。"""

    paper_id: str
    domain_tags: List[str] = field(default_factory=list)
    task_tags: List[str] = field(default_factory=list)
    problem_statement: str = ""
    method_tags: List[str] = field(default_factory=list)
    model_tags: List[str] = field(default_factory=list)
    dataset_tags: List[str] = field(default_factory=list)
    metric_tags: List[str] = field(default_factory=list)
    paper_type: str = ""
    core_contributions: List[str] = field(default_factory=list)
    application_scenarios: List[str] = field(default_factory=list)
    retrieval_keywords_en: List[str] = field(default_factory=list)
    retrieval_keywords_zh: List[str] = field(default_factory=list)
    survey_signals: List[str] = field(default_factory=list)
    likely_user_intents: List[str] = field(default_factory=list)
    limitations_or_scope: str = ""
    evidence_spans: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
