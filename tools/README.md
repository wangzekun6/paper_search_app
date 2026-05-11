# PaperSearch Tools

这个目录包含 PaperSearch 的可执行入口、前端界面以及核心业务模块。
如果根目录 `README.md` 面向“怎么运行系统”，那么这个文件主要回答“代码分别放在哪里、每个文件负责什么”。

## 目录结构

```text
tools/
├── README.md
├── requirements.txt
├── .env.example
├── papercompass.py
├── app.py
└── papercompass_core/
    ├── __init__.py
    ├── config.py
    ├── models.py
    ├── ingest.py
    ├── retrieval.py
    ├── semantic.py
    ├── semantic_backfill.py
    ├── llm.py
    ├── intent.py
    ├── chain.py
    └── services.py
```

## 顶层文件作用

- `README.md`：当前目录说明文件，主要解释 `tools/` 内部结构与模块职责。
- `requirements.txt`：`tools/` 运行依赖清单。
- `.env.example`：本地私有 LLM 配置示例文件，用来复制为 `.env` 后填写 API Key、Base URL、模型名。
- `papercompass.py`：统一命令行入口。
- `app.py`：Streamlit 前端入口。

## `papercompass.py` 负责什么

`papercompass.py` 是整个系统的 CLI 门面。
它负责解析命令并调用服务层，不直接承载核心业务逻辑。

当前主要子命令包括：

- `build`：构建数据库、恢复缓存、预热索引、启动语义卡后台补全
- `status`：查看系统状态
- `search`：基础检索
- `chain`：执行完整主链路
- `paper`：查看单篇论文详情
- `saved` / `save` / `unsave`：收藏管理
- `cards`：语义卡生成/刷新
- `semantic-backfill`：语义卡后台补全与状态查询
- `intent` / `intent-build`：意图分析与相关构建任务
- `chain-build`：批量生成演示与评估产物

## `app.py` 负责什么

`app.py` 是 Streamlit 前端页面入口，负责把服务层返回的数据组织成可交互界面。

当前前端主要区域包括：

- 搜索输入区
- Intent Frame / Gap Analysis 展示区
- 搜索结果区
- 搜藏管理区
- 历史记录区
- 标准 demo 回放区
- `Paper Explorer` 直接详情查看区

前端里的完整论文详情模板也在这里实现，包括：

- Overview
- Semantic Tags
- Match Evidence
- Structure
- Debug

## `papercompass_core/` 模块职责

### `config.py`

全局路径与系统配置集中定义在这里，包括：

- 数据集目录
- `system_outputs/` 路径
- 运行数据库路径
- 缓存目录路径
- prompt / eval / demos 路径
- 语义卡 backfill 状态文件与日志路径

这个模块是全系统的路径基座。

### `models.py`

定义核心数据结构，例如：

- 论文记录
- section 记录
- `IntentFrame`
- `PaperSemanticCard`

如果你要看“系统内部对象长什么样”，先看这里。

### `ingest.py`

负责原始数据集的解析、清洗、结构规范化和入库前准备。

主要工作包括：

- 论文 JSON 解析
- 作者字段清洗
- section 扁平化
- 检索用文本字段构建
- 记录写入 SQLite

### `retrieval.py`

负责基础检索能力，是召回层核心。

主要包括：

- SQLite / FTS 数据库构建
- `basic` 检索
- `exact` 检索
- `hybrid` 检索
- section 与 snippet 加载
- 检索分数、命中字段、匹配片段生成

### `semantic.py`

负责语义卡系统本体。

主要包括：

- 语义卡生成
- 语义卡缓存写入
- 语义卡缓存恢复
- 语义卡质量检查
- 语义卡样本导出

语义卡的持久化缓存目录是：

```text
system_outputs/cache/semantic_cards/
```

### `semantic_backfill.py`

负责后台语义卡补全任务。

它的作用不是前台即时生成，而是：

- 在构建后后台启动补全 worker
- 先恢复已有缓存
- 优先覆盖标准 / 演示 query 高频候选集
- 可继续补齐全量缺失语义卡
- 输出状态文件与日志文件

### `llm.py`

负责所有 OpenAI-compatible LLM 调用封装。

主要包括：

- 读取 `.env` / 环境变量 / Windows 环境变量
- 统一 API Key / Base URL / Model 配置
- 普通 chat completion
- 结构化 JSON 输出
- 最小 API 可用性测试
- retry 与代理兼容处理

这个文件现在不再依赖硬编码 API Key。

### `intent.py`

负责查询意图理解。

主要包括：

- 自然语言查询解析为 `IntentFrame`
- 追问回复与旧意图合并
- 槽位补全与澄清问题生成
- 意图评估与错误样本输出

### `chain.py`

负责系统的主链路编排，是“LLM-first 检索流程”的核心执行模块。

主要包括：

- coarse / dense / exact 多路召回
- 候选集收口
- query-paper match
- 重排
- 解释生成
- 标准 query 演示与回归评估
- 标准 query 候选集的语义卡预热

### `services.py`

这是前端和 CLI 共用的统一服务层。

它的意义是把底层复杂模块收敛成稳定接口，避免 `app.py` 或 `papercompass.py` 直接拼装所有细节。

当前典型能力包括：

- 项目构建
- 项目检索
- 论文详情加载
- 收藏管理
- 历史记录管理
- 标准 query 加载
- 语义卡缓存恢复
- 语义卡 backfill 启动与状态查询

## 模块调用关系

最常见的两条调用链如下。

### 前端链路

```text
app.py
  -> services.py
    -> intent.py / chain.py / retrieval.py / semantic.py / llm.py
```

### 命令行链路

```text
papercompass.py
  -> services.py
    -> retrieval.py / semantic.py / semantic_backfill.py / intent.py / chain.py
```

## 哪些文件最适合继续改

如果你要改某一类能力，可以优先看这些入口：

- 改前端展示：`app.py`
- 改命令行行为：`papercompass.py`
- 改论文详情、收藏、系统状态接口：`services.py`
- 改检索召回：`retrieval.py`
- 改主链路重排与解释：`chain.py`
- 改意图分析：`intent.py`
- 改语义卡生成或缓存：`semantic.py`
- 改后台语义卡补全：`semantic_backfill.py`
- 改 LLM 配置与请求：`llm.py`

## 相关输出目录

虽然这些目录不在 `tools/` 下，但这里的代码会持续读写它们：

```text
system_outputs/
```

常见内容包括：

- `runtime/`：当前运行数据库、app state、backfill 状态与日志
- `cache/`：语义卡、意图会话、query-paper match 缓存
- `prompts/`：各类提示词
- `eval/`：评估报告
- `demos/`：标准 query 与演示产物

## 一句话总结

`tools/` 是整个 PaperCompass 的执行层与业务核心：`papercompass.py` 管 CLI，`app.py` 管前端，`papercompass_core/` 管真正的检索、意图、语义卡、LLM 和服务逻辑。
