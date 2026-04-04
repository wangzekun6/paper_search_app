# Tools

当前 `tools/` 目录服务于项目内置的本地 arXiv 数据集：

```text
../data/arxiv_202502_cs_cl/
```

## 主要文件

- `app.py`: Streamlit Day 2 检索界面
- `day2_pipeline.py`: Day 2 全量入库、FTS5、exact match、20 条 query 调试导出
- `day3_pipeline.py`: Day 3 语义卡片生成、缓存、质量检查、缓存策略导出
- `openai_helpers.py`: OpenAI API 访问与 query 改写共用模块
- `extract.py`: 旧版 JSON 命令行检索入口
- `day1_pipeline.py`: Day 1 数据结构验证、样例记录生成与 SQLite 样例库构建
- `day1_contracts.py`: Day 1 固定对象协议
- `day1_schema.sql`: Day 1 SQLite schema 草案

## Web 运行

```bash
cd tools
python day2_pipeline.py build
python day3_pipeline.py build --target-count 100
streamlit run app.py
```

## 命令行运行

```bash
cd tools
python day2_pipeline.py build
python day2_pipeline.py search "retrieval augmented generation" --mode hybrid --top-k 10
python day2_pipeline.py debug-queries
python day3_pipeline.py build --target-count 100
python day3_pipeline.py generate-paper 2502.12701
```

## Day 2 默认检索字段

- `title`
- `abstract`
- `section_titles`
- `section_snippet`
