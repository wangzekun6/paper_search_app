"""
项目级数据集配置。

这个文件只负责集中维护当前项目使用的数据目录、展示名称和相对路径，
避免在不同脚本里重复硬编码路径。
Day 1 / Day 2 / Day 3 的命令行工具以及 Streamlit 页面都会从这里读取配置。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_LABEL = "arXiv 2025-02 cs.CL"
DATASET_DIR = PROJECT_ROOT / "data" / "arxiv_202502_cs_cl"
DATASET_RELATIVE_PATH = DATASET_DIR.relative_to(PROJECT_ROOT).as_posix()
