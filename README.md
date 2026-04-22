# PaperCompass：面向自然语言论文检索的 LLM-first 系统

## 1. 项目简介

PaperCompass 是一个面向学术论文检索场景的智能检索系统。系统以自然语言查询为输入，围绕“用户意图理解、候选召回、query-paper match、重排解释、论文详情展示”构建完整闭环，旨在提升传统关键词检索在意图表达、多轮追问与结果可解释性方面的能力。

与传统论文搜索方式相比，本项目不要求用户先将需求拆解为精确关键词，而是允许用户直接输入自然语言问题，例如“帮我找最近两年的 RAG 综述，并解释为什么推荐”。系统会先通过大语言模型识别查询意图与约束条件，再执行检索、排序与解释生成，最后以可交互的方式返回结果。

从工程实现角度看，PaperCompass 并非单一算法脚本，而是一套较完整的系统原型，覆盖了前端界面、命令行入口、LLM 接入、检索链路编排、语义卡缓存、后台补全、评估产物输出等多个层面，适合作为“基于大语言模型的论文检索系统”相关毕设项目进行展示与说明。

## 2. 研究背景与问题定义

在传统论文检索场景中，用户通常面临以下问题：

- 查询表达不稳定。用户往往更擅长描述需求，而不是一次性给出准确关键词。
- 隐含条件难以显式表达。例如“近两年”“优先综述”“带某种方法约束”等需求，在普通关键词检索中往往难以被完整捕捉。
- 候选结果缺少语义级判断。检索系统可能返回主题相关但并不真正满足当前需求的论文。
- 排序依据不透明。用户通常只能看到分数或列表，难以理解系统为什么推荐某篇论文。
- 大模型调用成本较高。如果所有语义理解都在查询时临时生成，会造成冷启动时间过长。

因此，本项目围绕“如何让论文检索系统既能理解自然语言意图，又能提供可解释、可复用、可演示的完整结果链路”展开设计与实现。

## 3. 项目目标

PaperCompass 的主要目标如下：

1. 支持用户使用自然语言直接检索论文，而不是被迫手动构造关键词。
2. 使所有用户输入默认经过 LLM 意图分析，确保系统主路径真正由语义理解驱动。
3. 在召回阶段之后，通过 query-paper match 判断候选论文与当前需求是否真正匹配。
4. 为最终排序结果提供可解释的依据，而不是只返回黑盒列表。
5. 将语义卡作为持久化资产保存，实现跨构建复用与后台补全，降低首次响应成本。

## 4. 系统定位

从功能定位看，PaperCompass 不是一个简单的论文搜索页面，而是一套 LLM-first 的论文检索系统。其主要特征包括：

- 输入侧：支持中文或英文自然语言查询。
- 理解侧：通过 LLM 将查询解析为结构化意图表示 `IntentFrame`。
- 检索侧：组合多种召回方式，构建候选论文集合。
- 判断侧：通过 query-paper match 对候选论文与用户需求之间的真实匹配程度进行评估。
- 排序侧：综合检索信号、意图匹配信号和解释结果完成最终重排。
- 展示侧：支持追问、收藏、历史记录与完整论文详情查看。

## 5. 核心能力

当前系统已经实现以下能力：

- 自然语言论文检索
- LLM-first 意图分析
- 追问式多轮缩小结果范围
- sparse / exact / hybrid / dense 协同召回
- query-paper match 与重排解释
- 收藏、历史记录、单篇论文详情查看
- 完整论文详情模板
- `paper_id` 直达详情页（Paper Explorer）
- 语义卡持久化缓存
- 构建后自动启动后台语义卡补全任务

## 6. 系统总体架构

系统主链路可以概括为：

```text
用户自然语言输入
  -> Intent Analysis
  -> Coarse / Dense / Exact Candidate Recall
  -> Candidate Pool Merge
  -> Query-Paper Match
  -> Rerank
  -> Explanation Generation
  -> Frontend Result + Paper Detail
```

从工程实现角度看，前端和命令行最终都会通过统一服务层调用核心模块：

```text
app.py / papercompass.py
  -> services.py
    -> intent.py
    -> retrieval.py
    -> chain.py
    -> semantic.py
    -> llm.py
```

各层职责如下：

- 前端层：负责交互、结果展示、追问输入、详情页查看。
- 服务层：负责为前端与 CLI 暴露统一、稳定的调用接口。
- 意图层：负责将自然语言请求转换为结构化意图。
- 检索层：负责数据库构建、FTS 检索、多路召回与候选组织。
- 匹配与排序层：负责评估候选论文与查询之间的匹配程度，并输出最终排序结果。
- 语义资产层：负责语义卡生成、缓存恢复、后台补全与质量检查。
- LLM 接入层：负责管理模型配置、结构化输出、重试与错误处理。

## 7. 关键设计说明

### 7.1 LLM-first 意图理解

本系统要求所有用户输入默认先经过 LLM 意图分析，而不是仅在个别环节调用模型。这样做的原因在于：如果核心意图理解仍由 heuristic 主导，那么系统的核心主张将无法成立。PaperCompass 将 LLM 置于正式主路径，用于识别以下信息：

- 搜索场景
- 研究主题
- 技术约束
- 文献属性要求
- 结果偏好
- 是否仍需进一步澄清

上述结构化结果被组织为一个统一的“IntentFrame”。IntentFrame 是系统的顶层意图表示，将搜索意图拆分为五个互补的部分：search_scene、research_topic、technical_constraints、document_attributes 和 result_preferences。每个槽位不仅包含提取到的值（value），还维护该值的状态（status）、来源（source）和置信度（confidence）。也就是说，系统不仅记录用户表达了什么，还记录这些信息是否清晰、是否直接来自用户的原始表述或由模型推断、以及该信息的可靠程度。

这种表示既保留了用户的原始表述，又为后续的检索、追问和重排序提供了可判断的信息质量与来源的依据，从而在多轮交互和结果解释中能够更可靠地驱动系统行为。为了适配不同检索目的，系统提供了六类搜索模式以便更明确地表达查询意图：topic_exploration（主题探索）、survey_lookup（综述检索）、recent_progress（最新进展检索）、specific_paper_lookup（具体论文检索）、author_trace（作者追踪）和 method_constrained_search（方法约束检索）。通过在查询理解阶段明确意图类型与槽位质量，系统能有效减少误检、提高重排解释性并支持更有针对性的追问策略。

### 7.2 多路召回与候选收口

系统不会只依赖单一检索方式，而是组合多路召回能力：

- coarse recall
- dense recall
- exact match
- hybrid recall

这样做的目的在于兼顾主题召回、精确约束命中与语义扩展能力。召回后的候选结果不会直接返回给用户，而是继续进入 query-paper match 与重排阶段进行收口。

### 7.3 Query-Paper Match

query-paper match 是本系统区别于普通检索页面的重要环节。系统不会在候选召回后直接输出论文，而是进一步判断：

- 该论文是否真正符合当前查询主题；
- 是否满足时间范围、论文类型、方法约束等附加条件；
- 为什么它应当排在前面。

因此，最终排序不仅依赖检索得分，也会综合意图匹配结果与解释信息。

### 7.4 完整论文详情模板

系统前端已实现完整论文详情模板，支持从以下三个入口查看：

1. 搜索结果中的 `Paper Details`
2. `Management -> Saved Papers`
3. `Management -> Paper Explorer`

完整详情包括：

- 基础信息：标题、作者、时间、论文类型、章节数、语义卡状态
- 摘要与核心结论：abstract、core contributions、problem statement、limitations
- 语义标签：domain / task / method / model / dataset / metric 等
- 匹配证据：query-paper match explanation、matched dimensions、matched snippets
- 正文结构：section titles、section snippets
- 调试信息：semantic card JSON、query-paper match JSON、原始字段

### 7.5 语义卡持久化缓存

为降低系统首次检索时的大模型调用开销，本项目将语义卡设计为可复用的持久化资产。生成后的语义卡不会仅保存在当前运行数据库中，而会写入磁盘缓存目录：

```text
system_outputs/cache/semantic_cards/
```

这样做带来的直接收益包括：

- 重建数据库时可以直接恢复已有语义卡；
- 构建后可通过后台任务继续补全缺失语义卡；
- 标准 query 高频命中的候选集可以优先覆盖；
- 能够降低冷启动阶段的大模型生成压力。

## 8. 仓库结构说明

项目根目录当前主要结构如下：

```text
PaperCompass-main/
├── README.md
├── tools/
├── system_outputs/
├── data/
├── bundled_data/
└── archive/
```

各目录含义如下：

- `README.md`：项目总说明文档
- `tools/`：实际可运行代码、前端和 CLI 入口、核心业务模块
- `system_outputs/`：运行数据库、缓存、日志、评估结果与 demo 产物
- `data/`：默认数据集解压目录
- `bundled_data/`：数据集压缩包或拆分归档
- `archive/`：历史参考资产或归档内容

若需要进一步查看 `tools/` 内部模块职责，请阅读：

- `tools/README.md`

## 9. 主要模块说明

### 9.1 `tools/papercompass.py`

统一命令行入口，负责：

- `build`
- `status`
- `search`
- `chain`
- `paper`
- `saved / save / unsave`
- `cards`
- `semantic-backfill`
- `intent / intent-build`
- `chain-build`

### 9.2 `tools/app.py`

Streamlit 前端入口，负责：

- 搜索输入区
- Intent Frame / Gap Analysis 展示
- 搜索结果卡片展示
- 收藏与历史记录管理
- 标准 demo 回放
- 完整论文详情模板
- Paper Explorer 直达详情区

### 9.3 `tools/papercompass_core/`

核心业务模块所在目录，主要包含：

- `config.py`：路径与系统配置
- `models.py`：数据结构定义
- `ingest.py`：原始数据清洗与入库
- `retrieval.py`：基础检索与召回能力
- `semantic.py`：语义卡生成、缓存与质量检查
- `semantic_backfill.py`：后台语义卡补全任务
- `llm.py`：大模型调用封装
- `intent.py`：意图分析与追问合并
- `chain.py`：主链路编排与评估
- `services.py`：前端与 CLI 共用服务层

## 10. 部署与运行步骤

以下步骤默认在项目根目录执行。

### 10.1 创建虚拟环境并安装依赖

```bash
cd tools
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 10.2 配置 LLM 私有参数

推荐复制示例文件并填写：

```bash
cp tools/.env.example tools/.env
```

Windows PowerShell：

```powershell
Copy-Item tools\.env.example tools\.env
```

最小配置如下：

```env
OPENAI_API_KEY=your_private_key
OPENAI_API_BASE=http://newapi.hjlyywp.com/v1
OPENAI_MODEL=gpt-5.2
```

系统读取配置的优先顺序为：

1. 当前终端环境变量
2. `tools/.env` 或 `tools/.env.local`
3. Windows 用户 / 系统环境变量

### 10.3 首次构建数据库

```bash
cd tools
python papercompass.py build
```

这一步会：

- 构建当前运行数据库
- 建立 FTS 检索资产
- 预热 dense 检索索引
- 恢复历史语义卡缓存
- 启动后台语义卡补全任务

### 10.4 启动前端

```bash
cd tools
python -m streamlit run app.py
```

默认地址通常为：

```text
http://localhost:8501
```

## 11. 前端演示建议

如果该项目用于毕业设计答辩，建议按以下顺序进行演示：

1. 展示搜索输入区，输入一条自然语言查询；
2. 展示 Intent Frame，说明系统如何理解用户需求；
3. 展示 Gap Analysis，说明系统如何引导用户通过追问补全条件；
4. 展示搜索结果及 query-paper match explanation；
5. 打开某篇论文的完整详情模板；
6. 进入 `Saved Papers` 或 `Paper Explorer`，展示独立论文详情查看能力；
7. 展示 `status` 或 `semantic-backfill --status`，说明缓存与后台补全机制。

按照这一顺序，能够更清楚地体现该系统并非单纯的论文搜索页面，而是一套包含语义理解、检索收口、解释生成和资产缓存机制的完整信息检索系统。

## 12. 常用命令

以下命令默认在 `tools/` 目录执行。

### 构建数据库

```bash
python papercompass.py build
```

常见构建变体：

```bash
python papercompass.py build --with-debug-queries
python papercompass.py build --with-semantic-cards
python papercompass.py build --semantic-backfill-mode all
```

### 查看系统状态

```bash
python papercompass.py status
```

### 普通检索

```bash
python papercompass.py search "retrieval augmented generation" --mode hybrid --top-k 10
```

### 完整主链路检索

```bash
python papercompass.py chain "recent agent memory papers" --top-k 5 --candidate-pool-size 60 --explain-limit 5
```

### 带追问继续检索

```bash
python papercompass.py chain "recent agent memory papers" --follow-up "recent two years, explain why each paper matches"
```

### 查看单篇论文详情

```bash
python papercompass.py paper 2502.06872
```

### 收藏管理

```bash
python papercompass.py saved --limit 20
python papercompass.py save 2502.06872
python papercompass.py unsave 2502.06872
```

### 语义卡与后台补全

```bash
python papercompass.py cards --paper-id 2502.06872
python papercompass.py cards --target-count 100
python papercompass.py semantic-backfill --status
python papercompass.py semantic-backfill --mode all
python papercompass.py semantic-backfill --restore-cache-only
```

### 意图与评估相关

```bash
python papercompass.py intent "找最近两年的 RAG 综述"
python papercompass.py chain-build --top-k 3 --candidate-pool-size 30 --explain-limit 3
```

## 13. 运行产物说明

所有运行产物默认位于：

```text
system_outputs/
```

重点目录如下：

- `system_outputs/runtime/`：运行数据库、`app_state.json`、backfill 状态与日志
- `system_outputs/cache/`：语义卡缓存、意图会话缓存、query-paper match 缓存
- `system_outputs/prompts/`：提示词文件
- `system_outputs/eval/`：评估与回归结果
- `system_outputs/demos/`：标准 query 与演示产物

后台补全状态文件与日志位于：

```text
system_outputs/runtime/semantic_backfill_state.json
system_outputs/runtime/semantic_backfill.log
```

## 14. 当前实现特点

### 14.1 LLM-first，而不是 heuristic-first

系统正式主路径要求所有用户输入先经过 LLM 意图分析。heuristic 可以存在于局部辅助逻辑中，但不再作为正式能力替代 LLM。

### 14.2 论文详情与检索上下文分离

论文完整详情可以独立查看，但 query-paper match explanation 只有在某次检索上下文中才是完整成立的。因此：

- 从搜索结果里打开详情，通常同时能看到匹配解释；
- 从 `Saved Papers` 或 `Paper Explorer` 打开的详情，不一定有当前查询上下文。

### 14.3 语义卡是持久化资产

语义卡会被写入磁盘缓存，而不是只存在于当前运行时数据库。这对于冷启动优化、候选覆盖率和跨构建复用都很关键。

## 15. 常见问题与排障

### 15.1 `python` 或 `streamlit` 命令找不到

请先确认：

- 已进入 `tools/`
- 已激活 `.venv`
- 已执行 `python -m pip install -r requirements.txt`

建议统一使用：

```bash
python -m streamlit run app.py
```

### 15.2 首次 `build` 很慢

这是正常现象。首次构建通常需要：

- 建库
- 索引预热
- 恢复或生成语义卡
- 可能恢复默认数据集

后续重复使用会快很多。

### 15.3 前端提示 LLM 不可用

优先检查 `tools/.env` 或环境变量：

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `OPENAI_MODEL`

可以先运行最小测试：

```bash
python -c "from papercompass_core.llm import OPENAI_API_KEY,test_openai_api; print(test_openai_api(OPENAI_API_KEY))"
```

### 15.4 为什么有些论文只有详情，没有匹配解释

直接从 `Saved Papers` 或 `Paper Explorer` 打开的详情，不一定带当前检索上下文。这时可以看到论文完整信息，但不会有当前 query 对应的匹配解释。

### 15.5 为什么结果为空或不理想

可以优先尝试：

- 改用更明确的英文关键词
- 在 follow-up 中补充时间范围、论文类型、方法约束
- 使用 `status` 确认数据库中确实存在论文数据
- 使用 `chain` 直接查看完整链路输出

## 16. 文档分工

- 当前 `README.md`：面向项目说明、部署、演示和答辩展示
- `tools/README.md`：面向代码结构、模块职责和开发者阅读

## 17. 总结

PaperCompass 以论文检索为应用场景，围绕“LLM 意图理解、多路召回、query-paper match、重排解释与完整论文详情展示”构建了一套较完整的智能检索系统。

对于毕设答辩场景，该项目既能展示算法与系统设计思路，也能展示工程落地能力，包括前端交互、命令行工具、缓存机制、后台任务以及评估产物生成。
