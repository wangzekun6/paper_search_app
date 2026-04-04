# PaperCompass

PaperCompass 现在整理为一个面向本地 `arXiv 2025-02 cs.CL` 数据集的论文检索项目。

## 当前功能

- Day 2 全量 SQLite 入库与 FTS5 检索
- 支持 `title / abstract / section_titles / section_snippet` 的基础检索
- 支持标题 hint、作者粗匹配、方法名 / 数据集名短语级 exact match
- Day 3 PaperSemanticCard 语义卡片生成与缓存
- 提供 Streamlit Web 界面
- 提供 Day 2 命令行构建与检索入口
- 保留 Day 1 数据契约、SQLite schema 和样例产物

## 数据集位置

项目内置数据集目录：

```text
data/arxiv_202502_cs_cl/
```

该目录下存放逐篇论文 JSON 文件。

## 本地运行

### 1. 安装依赖

```bash
cd PaperCompass-main/tools
pip install -r requirements.txt
```

### 2. 启动 Web 界面

```bash
cd PaperCompass-main/tools
python day2_pipeline.py build
streamlit run app.py
```

### 3. 命令行检索

```bash
cd PaperCompass-main/tools
python day2_pipeline.py build
python day2_pipeline.py search "retrieval augmented generation" --mode hybrid --top-k 10
python day2_pipeline.py debug-queries
python day3_pipeline.py build --target-count 100
```

## 项目结构

```text
PaperCompass-main/
├── data/
│   └── arxiv_202502_cs_cl/     # 当前主数据集
├── day1_outputs/               # Day 1 产物与样例数据库
├── day2_outputs/               # Day 2 全量数据库与 query 调试产物
├── day3_outputs/               # Day 3 Prompt、语义卡片样例、质量检查与缓存策略
├── tools/
│   ├── app.py                  # Streamlit Day 2 检索界面
│   ├── day2_pipeline.py        # Day 2 全量入库、FTS5、exact match、query 调试
│   ├── day3_pipeline.py        # Day 3 语义卡片生成、缓存、质量检查
│   ├── openai_helpers.py       # OpenAI API 访问与查询改写共用模块
│   ├── extract.py              # Day 1/旧版 JSON 检索入口
│   ├── day1_pipeline.py        # Day 1 数据契约与样例生成
│   └── day1_schema.sql         # SQLite schema 草案
```

## 说明

- 旧的会议目录数据已经移除，不再作为默认读取源。
- 当前所有默认读取路径都已经切换到 `data/arxiv_202502_cs_cl`。
- Day 2 默认数据库输出为 `day2_outputs/day2_full.db`。
- Day 3 默认会把语义卡片写回 `day2_outputs/day2_full.db` 的 `paper_semantic_cards` 表，并在 `day3_outputs/` 输出 Prompt、质量检查和缓存策略文件。
