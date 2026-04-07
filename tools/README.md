# PaperCompass Tools

## 项目结构

```text
tools/
├── README.md
├── requirements.txt
├── papercompass.py
├── app.py
└── papercompass_core/
    ├── __init__.py
    ├── config.py
    ├── models.py
    ├── ingest.py
    ├── retrieval.py
    ├── semantic.py
    ├── llm.py
    ├── intent.py
    ├── chain.py
    └── services.py
```

## 各子文件作用

- `requirements.txt`：项目依赖清单。
- `papercompass.py`：统一命令行入口，负责解析子命令并调用服务层。
- `app.py`：Streamlit 前端入口，负责页面渲染与交互。

- `papercompass_core/config.py`：全局配置与路径管理（数据集、输出目录、缓存目录等）。
- `papercompass_core/models.py`：核心数据结构定义（论文记录、章节记录等）。
- `papercompass_core/ingest.py`：原始数据清洗与入库前处理逻辑。
- `papercompass_core/retrieval.py`：数据库检索能力（sparse / exact / hybrid）与相关工具。
- `papercompass_core/semantic.py`：语义卡（PaperSemanticCard）生成与质量检查相关逻辑。
- `papercompass_core/llm.py`：LLM 请求封装（模型调用、重试、结构化输出等）。
- `papercompass_core/intent.py`：自然语言意图解析、追问合并、IntentFrame 处理。
- `papercompass_core/chain.py`：核心主链路编排（召回、重排、query-paper 匹配、解释）。
- `papercompass_core/services.py`：统一服务层，对前端和 CLI 暴露稳定调用接口。
