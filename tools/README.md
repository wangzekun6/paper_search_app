# Tools

当前 `tools/` 目录服务于 PaperCompass 的统一项目实现，面向项目内置的本地 arXiv 数据集：

```text
../data/arxiv_202502_cs_cl/
```

If the extracted dataset folder is missing, the build entrypoints will
automatically restore it from the local tar archive first:

```text
../bundled_data/arxiv_202502_cs_cl.tar
```

Optional local full gzip archive:

```text
../bundled_data/arxiv_202502_cs_cl.tar.gz
```

If neither local archive exists, the tools will download the release asset from:

```text
https://github.com/wangzekun6/paper_search_app/releases/download/dataset-20260407/arxiv_202502_cs_cl.tar.gz.part01
https://github.com/wangzekun6/paper_search_app/releases/download/dataset-20260407/arxiv_202502_cs_cl.tar.gz.part02
...
https://github.com/wangzekun6/paper_search_app/releases/download/dataset-20260407/arxiv_202502_cs_cl.tar.gz.partNN
```

## 主要文件

- `papercompass.py`: 项目级 CLI 入口
- `papercompass_services.py`: 项目级服务层，统一封装建库、检索和语义卡片能力
- `papercompass_intent.py`: 项目级意图理解层，负责 IntentFrame、追问和状态合并
- `papercompass_chain.py`: 项目级核心方法链路，负责三路召回、gap、重排和解释
- `app.py`: Streamlit 项目界面
- `day2_pipeline.py`: 内部全量入库、FTS5、exact match、20 条 query 调试导出
- `day3_pipeline.py`: 内部语义卡片生成、缓存、质量检查、缓存策略导出
- `openai_helpers.py`: OpenAI API 访问与 query 改写共用模块
- `extract.py`: 旧版 JSON 命令行检索入口
- `day1_pipeline.py`: Day 1 数据结构验证、样例记录生成与 SQLite 样例库构建
- `day1_contracts.py`: Day 1 固定对象协议
- `day1_schema.sql`: Day 1 SQLite schema 草案

## Web 运行

```bash
cd tools
python papercompass.py build
streamlit run app.py
```

## 命令行运行

```bash
cd tools
python papercompass.py build
python papercompass.py status
python papercompass.py search "retrieval augmented generation" --mode hybrid --top-k 10
python papercompass.py cards --target-count 100
python papercompass.py cards --paper-id 2502.12701
python papercompass.py intent "找最近两年关于RAG的综述，方法不限，最好解释为什么推荐"
python papercompass.py intent "最近两年，综述优先，方法不限" --history-id 1
python papercompass.py intent-build
python papercompass.py chain "我想看 RAG" --follow-up "最近两年，综述优先，最好解释为什么推荐"
python papercompass.py chain-build
```

## 默认检索字段

- `title`
- `abstract`
- `section_titles`
- `section_snippet`

## 扩展约束

- 后续新增功能优先接入 `papercompass_services.py`，让前端和外部调用都复用统一接口。
- `day1 / day2 / day3` 文件保留为内部实现模块，不建议继续作为产品级入口向外扩散。
- Day 4 的意图理解也应优先通过 `papercompass_intent.py` 和 `papercompass_services.py` 接入，不要在前端里另写一套独立状态机。
- Day 5 的候选融合、gap、重排和解释也应优先通过 `papercompass_chain.py` 接入，不要把排序逻辑分散写回 `day2_pipeline.py` 或前端页面。
