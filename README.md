# PaperCompass

PaperCompass 是一个面向本地 `arXiv 2025-02 cs.CL` 数据集的完整论文检索项目。
项目内部保留了 `day1 / day2 / day3` 文件作为实现演进痕迹，但对外应按一个统一系统来理解和扩展。

## 当前功能

- 统一项目索引构建：原始论文 JSON、论文索引字段、章节记录、FTS5 检索表共用一个 SQLite 库
- 统一检索能力：支持 `title / abstract / section_titles / section_snippet` 的基础检索
- 统一精确匹配能力：支持标题 hint、作者粗匹配、方法名 / 数据集名短语级 exact match
- 统一语义层：支持 `PaperSemanticCard` 生成、缓存、抽查和稳定性报告
- 统一意图层：支持自然语言 query -> `IntentFrame`、聚合追问、二轮状态合并、三路 query 生成
- 统一主链路：支持 `query -> 意图解析 -> 三路检索 -> gap 分析 -> 意图重排 -> 结果解释`
- 统一入口：提供 `papercompass.py` 项目级 CLI 与 Streamlit Web 界面
- 保留样例协议、SQLite schema 和调试产物，便于后续扩展

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
python papercompass.py build
streamlit run app.py
```

### 3. 命令行检索

```bash
cd PaperCompass-main/tools
python papercompass.py build
python papercompass.py status
python papercompass.py search "retrieval augmented generation" --mode hybrid --top-k 10
python papercompass.py cards --target-count 100
python papercompass.py intent "找最近两年关于RAG的综述，方法不限，最好解释为什么推荐"
python papercompass.py intent "最近两年，综述优先，方法不限" --history-id 1
python papercompass.py intent-build
python papercompass.py chain "我想看 RAG" --follow-up "最近两年，综述优先，最好解释为什么推荐"
python papercompass.py chain-build
```

## 项目结构

```text
PaperCompass-main/
├── data/
│   └── arxiv_202502_cs_cl/     # 当前主数据集
├── day1_outputs/               # Day 1 产物与样例数据库
├── day2_outputs/               # Day 2 全量数据库与 query 调试产物
├── day3_outputs/               # Day 3 Prompt、语义卡片样例、质量检查与缓存策略
├── day4_outputs/               # Day 4 IntentFrame Prompt、测试结果、合并样例
├── day5_outputs/               # Day 5 核心链路演示、gap 报告、重排结果、解释 Prompt
├── tools/
│   ├── app.py                  # Streamlit 项目界面
│   ├── papercompass.py         # 统一项目级 CLI 入口
│   ├── papercompass_services.py # 统一项目服务层
│   ├── papercompass_intent.py  # 统一意图理解层
│   ├── papercompass_chain.py   # 统一核心方法链路
│   ├── day2_pipeline.py        # 内部检索底座实现
│   ├── day3_pipeline.py        # 内部语义卡片实现
│   ├── openai_helpers.py       # OpenAI API 访问与查询改写共用模块
│   ├── extract.py              # 旧版 JSON 检索入口
│   ├── day1_pipeline.py        # 样例协议与验证产物生成
│   └── day1_schema.sql         # SQLite schema
```

## 说明

- 旧的会议目录数据已经移除，不再作为默认读取源。
- 当前所有默认读取路径都已经切换到 `data/arxiv_202502_cs_cl`。
- 默认项目数据库输出为 `day2_outputs/day2_full.db`。
- 语义卡片默认写回同一个 SQLite 库的 `paper_semantic_cards` 表，并在 `day3_outputs/` 输出 Prompt、质量检查和缓存策略文件。
- 后续新增功能应优先接入 `papercompass.py` 和 `papercompass_services.py`，避免直接把产品层耦合到某个 day 文件。
