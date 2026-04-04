-- Day 1 固定下来的 SQLite schema。
-- 这份 schema 定义了项目的基础数据层结构，包括：
-- 1. raw_papers：原始论文 JSON
-- 2. papers：论文级索引字段
-- 3. paper_sections：章节级扁平记录
-- 4. paper_semantic_cards：Day 3 生成的语义卡片缓存
-- 5. paper_search_fts：SQLite FTS 检索表

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS raw_papers (
    paper_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    year_month TEXT NOT NULL,
    title TEXT NOT NULL,
    authors_raw TEXT NOT NULL,
    normalized_authors TEXT NOT NULL,
    abstract TEXT NOT NULL,
    section_titles TEXT NOT NULL,
    intro_text TEXT NOT NULL,
    methods_text TEXT NOT NULL,
    results_text TEXT NOT NULL,
    discussion_text TEXT NOT NULL,
    appendix_titles TEXT NOT NULL,
    fulltext_for_sparse TEXT NOT NULL,
    embedding_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_year_month ON papers(year_month);
CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);

CREATE TABLE IF NOT EXISTS paper_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    section_order INTEGER NOT NULL,
    section_title TEXT NOT NULL,
    section_type TEXT NOT NULL CHECK (section_type IN ('intro', 'methods', 'results', 'discussion', 'other')),
    section_text TEXT NOT NULL,
    section_snippet TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_sections_paper_order
    ON paper_sections(paper_id, section_order);

CREATE INDEX IF NOT EXISTS idx_paper_sections_type
    ON paper_sections(section_type);

CREATE TABLE IF NOT EXISTS paper_semantic_cards (
    paper_id TEXT PRIMARY KEY,
    semantic_card_json TEXT NOT NULL,
    card_status TEXT NOT NULL DEFAULT 'pending',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    intent_frame_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS saved_papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS paper_search_fts USING fts5(
    paper_id UNINDEXED,
    title,
    abstract,
    section_titles,
    section_snippet
);
