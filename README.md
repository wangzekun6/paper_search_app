# PaperCompass（小白可用版）

PaperCompass 是一个“自然语言找论文”系统。  
你输入一句话，它会自动做意图解析、检索、重排，并返回可解释的推荐结果。

如果你是第一次接触这个项目，按下面“5 分钟快速开始”做就行。

## 你能用它做什么

- 用自然语言查论文（中英文都可以）。
- 自动补全检索意图（场景、主题、方法、时间范围、论文类型等）。
- 支持追问式检索（先查一轮，再补充需求继续查）。
- 支持收藏、历史记录、论文详情查看。
- 同时支持前端界面和命令行。

## 5 分钟快速开始

### 1) 准备环境

- Windows / macOS / Linux 均可。
- Python 建议 `3.10+`（已在 `3.11` 环境验证）。
- 需要能访问网络（首次可能下载默认数据集，且 LLM 调用需要联网）。

### 2) 安装依赖

```bash
cd tools
pip install -r requirements.txt
```

### 3) 首次构建数据库（必做）

```bash
python papercompass.py build
```

说明：

- 这一步会建立项目数据库。
- 如果 `data/arxiv_202502_cs_cl/` 不存在，系统会尝试从本地归档或在线 release 恢复默认数据集。

### 4) 启动前端

```bash
python -m streamlit run app.py
```

启动后在浏览器打开页面（通常是 `http://localhost:8501`）。

### 5) 直接搜索

在输入框里填一句话，例如：

```text
帮我找最近两年的 RAG 综述，并解释为什么推荐
```

如果系统提示“请补充信息”，在追问输入框补一句再点“继续检索”即可。

## 命令行最常用操作

如果你不想开前端，也可以直接用 CLI。
以下命令默认在 `tools/` 目录执行。

### 看系统状态

```bash
python papercompass.py status
```

### 普通检索（basic / exact / hybrid）

```bash
python papercompass.py search "retrieval augmented generation" --mode hybrid --top-k 10
```

### 完整主链路（推荐）

```bash
python papercompass.py chain "recent agent memory papers" --top-k 5 --candidate-pool-size 60 --explain-limit 5
```

带追问继续检索：

```bash
python papercompass.py chain "recent agent memory papers" --follow-up "recent two years, explain why each paper matches"
```

### 常用管理命令

```bash
python papercompass.py history --limit 20
python papercompass.py saved --limit 50
python papercompass.py save 2502.06872
python papercompass.py unsave 2502.06872
python papercompass.py paper 2502.06872
```

## 给新手的推荐使用顺序

1. `build` 一次（首次必做）。  
2. 启动 Streamlit 前端。  
3. 先查一轮，再根据提示补一句追问。  
4. 需要批量验证时，用 `chain-build` 生成 demo 与回归报告。  

```bash
python papercompass.py chain-build --top-k 3 --candidate-pool-size 30 --explain-limit 3
```

## 输出文件都在哪

所有运行产物都在：

```text
system_outputs/
```

重点看这几个目录：

- `system_outputs/runtime/`：运行数据库、`app_state.json`。
- `system_outputs/cache/`：语义卡、意图会话、query-paper 匹配缓存。
- `system_outputs/prompts/`：提示词文件。
- `system_outputs/eval/`：评估与回归报告。
- `system_outputs/demos/`：标准查询与演示产物。

## LLM 配置说明（可选）

当前项目默认使用 `tools/papercompass_core/llm.py` 里的配置。  
如果你要换 API 地址、Key 或默认模型，修改该文件中以下常量：

- `DEFAULT_OPENAI_API_BASE`
- `DEFAULT_OPENAI_API_KEY`
- `DEFAULT_OPENAI_MODEL_CANDIDATES`

改完后重启前端或重新执行命令行进程即可生效。

## 常见问题（小白高频）

### 1) `python` 或 `streamlit` 命令找不到

- 先确认 Python 已安装并加入 PATH。
- 执行 `pip install -r requirements.txt`。
- 用 `python -m streamlit run app.py` 启动，避免 PATH 问题。

### 2) 首次 `build` 比较慢

- 首次需要建库，且可能下载数据集，慢是正常的。
- 后续重复使用会快很多。

### 3) 搜索结果为空

- 先试英文关键词（例如 `multimodal reasoning survey`）。
- 补一条追问，明确时间范围、论文类型或方法约束。
- 用 `python papercompass.py status` 确认数据库里有论文数据。

### 4) 前端提示 LLM 不可用

- 检查 `llm.py` 中的 API 配置是否正确。
- 先跑一次：

```bash
python -c "from papercompass_core.llm import OPENAI_API_KEY,test_openai_api; print(test_openai_api(OPENAI_API_KEY))"
```

## 核心目录（开发者参考）

```text
tools/
├── papercompass.py          # CLI 入口
├── app.py                   # Streamlit 入口
└── papercompass_core/
    ├── services.py          # 统一服务层（前端/CLI都走这里）
    ├── intent.py            # 意图解析与追问合并
    ├── chain.py             # 核心主链路（检索+重排+解释）
    ├── retrieval.py         # sparse / exact / hybrid 检索
    ├── semantic.py          # 语义卡生成
    ├── llm.py               # LLM 调用封装
    ├── ingest.py            # 数据入库
    ├── models.py            # 数据结构定义
    └── config.py            # 路径与系统配置
```

## 一句话总结

先 `build`，再 `streamlit run`，输入自然语言直接查。  
不满意就补一句追问继续查。
