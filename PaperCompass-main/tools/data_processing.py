import json
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 JSON 数据文件，支持多种顶层结构（列表、字典等）。
    """
    if not os.path.exists(file_path):
        logger.error(f"文件 {file_path} 不存在")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载数据文件: {file_path}, 数据条目数: {len(data) if isinstance(data, list) else 'N/A'}")

        # 如果顶层是列表，直接返回
        if isinstance(data, list):
            logger.info(f"文件 {file_path} 的顶层结构为列表，共 {len(data)} 条记录")
            return data

        # 如果顶层是字典，尝试提取常见字段
        elif isinstance(data, dict):
            logger.info(f"文件 {file_path} 的顶层结构为字典")
            # 优先提取 "categories" 字段
            if "categories" in data:
                logger.info("从字典中提取 'categories' 字段")
                return [{"category": key, "papers": value} for key, value in data["categories"].items()]
            # 如果存在其他字段（如 "data" 或 "records"），尝试提取
            elif "data" in data and isinstance(data["data"], list):
                logger.info("从字典中提取 'data' 字段")
                return data["data"]
            elif "records" in data and isinstance(data["records"], list):
                logger.info("从字典中提取 'records' 字段")
                return data["records"]
            else:
                logger.warning("未找到可提取的字段，返回空列表")
                return []

        # 如果顶层结构不支持，记录错误
        else:
            logger.error(f"文件 {file_path} 的顶层结构既不是列表也不是字典，无法处理")
            return []

    except Exception as e:
        logger.error(f"加载 JSON 文件失败: {e}")
        return []

def preprocess_data(data: List[Any]) -> List[Dict[str, Any]]:
    """
    对数据进行预处理，确保数据格式正确。
    """
    preprocessed_data = []
    for item in data:
        if isinstance(item, dict):  # 确保每个条目是字典
            preprocessed_data.append(item)
        else:
            logger.warning(f"跳过无效条目（非字典类型）: {item}")
    logger.info(f"数据预处理完成，共处理 {len(preprocessed_data)} 条记录")
    return preprocessed_data

def generate_data_quality_report(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    生成数据质量报告。
    """
    if not data:
        logger.warning("数据为空，无法生成数据质量报告")
        return {"total_records": 0, "missing_fields": {}}

    total_records = len(data)
    field_counts = {}
    for item in data:
        for key in item.keys():
            field_counts[key] = field_counts.get(key, 0) + 1

    missing_fields = {key: total_records - count for key, count in field_counts.items()}
    report = {
        "total_records": total_records,
        "missing_fields": missing_fields,
    }
    logger.info(f"数据质量报告: {report}")
    return report

def extract_features(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从数据中提取特征。
    """
    for item in data:
        if isinstance(item, dict):  # 再次检查，确保 item 是字典
            # 示例：提取标题长度作为特征
            item['title_length'] = len(item.get('title', ''))
            # 示例：提取关键词数量作为特征
            item['keyword_count'] = len(item.get('keywords', '').split(','))
        else:
            logger.warning(f"跳过无效条目（非字典类型）: {item}")
    logger.info(f"特征提取完成，共处理 {len(data)} 条记录")
    return data

def augment_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对数据进行增强（示例：生成伪造数据）。
    """
    augmented_data = []
    for item in data:
        if isinstance(item, dict):  # 确保 item 是字典
            augmented_item = item.copy()
            # 示例：对标题进行简单变换
            augmented_item['title'] = item.get('title', '') + " (Augmented)"
            augmented_data.append(augmented_item)
        else:
            logger.warning(f"跳过无效条目（非字典类型）: {item}")
    logger.info(f"数据增强完成，生成 {len(augmented_data)} 条增强数据")
    return data + augmented_data