"""
基于 Streamlit 的论文搜索和筛选 Web 界面。
该模块提供了一个用户友好的界面，用于搜索和分析学术论文。222222
"""

import streamlit as st
import json
import os
import glob
from typing import List, Dict, Any, Optional
import logging
from extract import load_data, filter_data, count_results, SEARCH_MODE_AND, SEARCH_MODE_OR, DEFAULT_FIELDS
from key_fields_loader import load_conference_key_fields, get_available_conferences, get_conference_years, load_conference_categories
import sqlite3
from hashlib import sha256
import traceback  # 新增：用于异常追踪
import requests   # 新增：用于调用百度文心千帆 HTTP 接口
from data_processing import load_json_data, preprocess_data, extract_features, augment_data
from model import PASAModel
from utils import setup_logger

# 初始化日志
logger = setup_logger()

# 项目目录（改为更稳健的绝对路径）
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if not os.path.isdir(PROJECT_DIR):
    logger.warning("PROJECT_DIR (%s) 不存在，使用当前工作目录作为后备。", PROJECT_DIR)
    PROJECT_DIR = os.path.abspath(".")

# 获取会议列表：更健壮的探测（仅把包含 JSON 的目录视为会议目录）
try:
    _entries = os.listdir(PROJECT_DIR)
except Exception as e:
    logger.error("列出 PROJECT_DIR 失败: %s", e)
    _entries = []

CONFERENCES = []
for name in _entries:
    p = os.path.join(PROJECT_DIR, name)
    if os.path.isdir(p) and name != 'tools' and not name.startswith('.'):
        try:
            # 仅当目录内存在 .json 文件时才认为是会议目录
            if any(f.lower().endswith('.json') for f in os.listdir(p)):
                CONFERENCES.append(name)
        except Exception:
            # 忽略无权限等问题
            continue

# 如果没有发现任何会议目录，做一次递归性扫描（作为最后的后备）
if not CONFERENCES:
    logger.info("未在 PROJECT_DIR 顶层发现会议目录，进行一次递归扫描以查找包含 JSON 的子目录...")
    found = set()
    for root, dirs, files in os.walk(PROJECT_DIR):
        for d in dirs:
            dd = os.path.join(root, d)
            try:
                if any(f.lower().endswith('.json') for f in os.listdir(dd)):
                    rel = os.path.relpath(dd, PROJECT_DIR).split(os.sep)[0]
                    found.add(rel)
            except Exception:
                continue
    CONFERENCES = sorted(list(found))
    logger.info("递归扫描发现会议：%s", CONFERENCES)

# 数据搜索模式
DATA_SEARCH_MODES = ["所有论文", "特定会议"]

# 新增：文心千帆 Key
BAIDU_QF_KEY = os.environ.get("BAIDU_QF_KEY", "bce-v3/ALTAK-WfFDFmuJ6ib2B0y18YhMq/17f6425aa0d49303cf3c88bae3730120ab2a9a3f")

# 使用 Streamlit 缓存装饰器来优化加载性能
# 禁用缓存以确保每次加载数据时重新读取文件
def load_conference_data(conference_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    加载会议数据（支持动态年份选择）。
    """
    # 找到会议数据所在目录
    conf_dir = os.path.join(PROJECT_DIR, conference_name)
    if not os.path.isdir(conf_dir):
        logger.warning("找不到 %s 的顶层目录 (%s). 将尝试在项目中递归搜索相关 JSON 文件。", conference_name, conf_dir)
        json_files = glob.glob(os.path.join(PROJECT_DIR, "**", f"*{conference_name}*.json"), recursive=True)
        if not json_files:
            logger.error("未能在项目中找到任何匹配 %s 的 JSON 文件。", conference_name)
            st.error(f"找不到 {conference_name} 的数据文件（递归搜索失败）。")
            return None
        latest_file = sorted(json_files)[-1]
        logger.info(f"加载会议数据: {conference_name}, 文件路径: {latest_file}")
        return load_json_data(latest_file)

    # 查找会议目录下的 JSON 文件
    json_files = glob.glob(os.path.join(conf_dir, "*.json"))
    if not json_files:
        st.error(f"未找到 {conference_name} 的 JSON 文件（在目录 {conf_dir} 内未发现）。")
        logger.error("目录 %s 中没有 JSON 文件。", conf_dir)
        return None

    latest_file = sorted(json_files)[-1]
    logger.info(f"加载会议数据: {conference_name}, 文件路径: {latest_file}")
    return load_json_data(latest_file)

def create_search_sidebar() -> Dict[str, Any]:
    """
    创建搜索配置的侧边栏。
    
    返回:
        Dict[str, Any]: 包含搜索参数的字典
    """
    with st.sidebar:
        st.subheader("选择论文来源")
        data_search_mode = st.radio(
            "数据源:",
            DATA_SEARCH_MODES,
            help="选择您希望如何搜索论文",
            horizontal=False,
            index=None,
            label_visibility="collapsed"
        )
        
        st.header("搜索配置")
        keyword = st.text_input(
            "输入关键词:", 
            value="",
            help="多个关键词可以用逗号或空格分隔（例如：'retrieval agent' 或 'retrieval,agent'）。留空将显示所有结果。"
        )
        
        search_mode = st.radio(
            "关键词搜索模式:",
            [SEARCH_MODE_OR, SEARCH_MODE_AND],
            help=f"{SEARCH_MODE_OR}: 查找包含任一关键词的论文。{SEARCH_MODE_AND}: 查找包含所有关键词的论文。",
            horizontal=True
        )
        
        fields_to_search = st.multiselect(
            "选择要搜索的字段（多选）:",
            options=DEFAULT_FIELDS,
            default=None
        )
        
        st.subheader("其它选项")
        show_all_fields = st.checkbox(
            "显示全部字段", 
            value=False,
            help="勾选此项将显示论文的所有字段。"
        )
        
        include_rejected = st.checkbox(
            "包含被拒绝/撤回的论文", 
            value=False,
            help="勾选此项可包含被拒绝或撤回的论文。"
        )
        
        # 新增：启用文心千帆自然语言解析选项与内置 Key
        use_nl = st.checkbox(
            "是否启用大模型查询解析（文心千帆）",
            value=False,
            help="启用后系统会调用文心千帆将您的自然语言查询解析为英文关键词，再使用这些关键词进行全文匹配检索。"
        )

        if use_nl:
            st.info("将使用大模型进行关键词提取，点击是否测试key已经配置成功。")
            if st.button("测试 文心千帆 Key"):
                key_to_test = os.environ.get("BAIDU_QF_KEY", BAIDU_QF_KEY)
                ok, msg = test_baidu_qf_api(key_to_test)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg + " 常见原因：Key 不正确、网络受限或接口需按实际文档调整。")

    return {
        "keyword": keyword,
        "search_mode": search_mode,
        "fields_to_search": fields_to_search,
        "data_search_mode": data_search_mode,
        "show_all_fields": show_all_fields,
        "include_rejected": include_rejected,
        "use_nl": use_nl,
        "baidu_key": os.environ.get("BAIDU_QF_KEY", BAIDU_QF_KEY),
    }


def load_data_source(data_search_mode: str) -> tuple:
    """
    根据选择的数据源模式加载数据。
    
    参数:
        data_search_mode (str): 数据源模式
        
    返回:
        tuple: (data, source, key_fields_filters)，其中 data 是加载的数据，
               source 是数据源描述，key_fields_filters 包含关键字段筛选条件
    """
    data = []
    source = ""
    key_fields_filters = {}
    conference_categories = {}

    if data_search_mode == DATA_SEARCH_MODES[0]:  # All Papers
        for conf in CONFERENCES:
            conf_data = load_conference_data(conf)
            if conf_data:
                conf_data = preprocess_data(conf_data)
                for paper_item in conf_data:
                    paper_item['source'] = conf
                data.extend(conf_data)
        source = "All Papers"
    elif data_search_mode == DATA_SEARCH_MODES[1]:  # Specific Conferences
        conferences = st.multiselect("选择会议:", CONFERENCES)
        for conf in conferences:
            conf_data = load_conference_data(conf)
            if conf_data:
                conf_data = preprocess_data(conf_data)
                for paper_item in conf_data:
                    paper_item['source'] = conf
                data.extend(conf_data)
        source = "+".join(conferences)

    return data, source, key_fields_filters


def display_search_results(data, source, search_params):
    """
    筛选数据并显示搜索结果。
    
    参数:
        data: 要搜索的数据
        source: 数据源的描述
        search_params: 包含搜索参数的字典
    """
    keyword = search_params["keyword"]
    search_mode = search_params["search_mode"]
    fields_to_search = search_params["fields_to_search"]
    include_rejected = search_params["include_rejected"]
    key_fields_filters = search_params.get("key_fields_filters", {})
    show_all_fields = search_params.get("show_all_fields", False)
    
    conference_categories_selection = st.session_state.get('conference_categories', {})
    
    data = st.session_state.get('data')
    if source == '':
        st.warning("请选择会议")
        return
    if not data:
        st.error("无法加载数据，请检查。")
        return
        
    keywords_list = [k.strip() for k in keyword.replace(',', ' ').split() if k.strip()]

    if fields_to_search == [] and keywords_list != []:
        st.warning("请选择要搜索的字段")
        return
    if fields_to_search != [] and keywords_list == []:
        st.warning("请输入关键词")
        return

    if keywords_list:
        if len(keywords_list) > 1:
            if search_mode == SEARCH_MODE_OR:
                st.info(f"搜索包含任一关键词的论文: {', '.join(keywords_list)}")
            else:
                st.info(f"搜索包含所有关键词的论文: {', '.join(keywords_list)}")
        else:
            st.info(f"搜索包含关键词的论文: {keywords_list[0]}")
    else:
        st.info("未输入关键词，将显示所有符合筛选条件的论文。")
    
    if key_fields_filters:
        filter_descriptions = []
        for field, conf_values_dict in key_fields_filters.items():
            field_specific_descriptions = []
            for conf_name, values in conf_values_dict.items():
                if values:
                    str_values = [str(val) for val in values]
                    field_specific_descriptions.append(f"{conf_name} {field.capitalize()}: {', '.join(str_values)}")
            if field_specific_descriptions:
                 filter_descriptions.append(" | ".join(field_specific_descriptions))

        if filter_descriptions:
            st.info(f"应用的关键字段筛选: {'; '.join(filter_descriptions)}")
            
    if conference_categories_selection:
        display_messages = []
        for conf_name, selected_cats in conference_categories_selection.items():
            if selected_cats:
                display_messages.append(f"{conf_name} 研究方向: {', '.join(selected_cats)}")
        if display_messages:
            st.info(f"研究方向分类筛选: {' | '.join(display_messages)}")

    with st.spinner('正在处理数据...'):
        if keywords_list:
            status_filtered, filtered = filter_data(data, keyword, fields_to_search, search_mode, include_rejected)
        else:
            if include_rejected:
                status_filtered = data
                filtered = data
            else:
                status_filtered = [item for item in data if item.get('status') not in ['Withdraw', 'Reject', 'Desk Reject']]
                filtered = status_filtered
        
        if key_fields_filters and filtered:
            data_search_mode = search_params.get("data_search_mode", "")
            
            if data_search_mode == DATA_SEARCH_MODES[1]: # Conference(s)
                filtered_papers_after_key_fields = []
                for paper in filtered:
                    paper_conf = paper.get('source')
                    include_paper = True
                    
                    for field, conf_values_map in key_fields_filters.items():
                        if paper_conf in conf_values_map and conf_values_map[paper_conf]:
                            field_value_in_paper = paper.get(field)
                            selected_values_for_conf_field = conf_values_map[paper_conf]
                            
                            if field == 'award' and isinstance(field_value_in_paper, bool):
                                str_selected_values = [str(val).lower() for val in selected_values_for_conf_field]
                                if str(field_value_in_paper).lower() not in str_selected_values:
                                    include_paper = False
                                    break
                            elif field_value_in_paper not in selected_values_for_conf_field:
                                include_paper = False
                                break
                    
                    if include_paper:
                        filtered_papers_after_key_fields.append(paper)
                filtered = filtered_papers_after_key_fields
        
        data_search_mode = search_params.get("data_search_mode", "")
        if data_search_mode == DATA_SEARCH_MODES[1]: # Conference(s)
            if conference_categories_selection and filtered:
                filtered_papers_after_category = []
                for paper in filtered:
                    paper_conf = paper.get('source')
                    
                    if paper_conf not in conference_categories_selection or not conference_categories_selection[paper_conf]:
                        filtered_papers_after_category.append(paper)
                        continue
                    
                    selected_categories_for_conf = conference_categories_selection[paper_conf]
                    
                    actual_conf_categories_data = load_conference_categories(paper_conf)

                    if not actual_conf_categories_data:
                        filtered_papers_after_category.append(paper)
                        continue
                    
                    paper_ids_in_selected_cats = set()
                    for cat_name in selected_categories_for_conf:
                        if cat_name in actual_conf_categories_data:
                            paper_ids_in_selected_cats.update(actual_conf_categories_data[cat_name])
                    
                    if paper.get('id') in paper_ids_in_selected_cats:
                        filtered_papers_after_category.append(paper)
                
                filtered = filtered_papers_after_category

        counts = count_results(data, status_filtered, filtered, keyword, fields_to_search, search_mode)

        st.subheader("搜索统计")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总论文数", len(data))
        with col2:
            st.metric("状态筛选后的论文" if include_rejected else "已接收的论文", counts['status_filtered_count'])
        with col3:
            st.metric("匹配结果", len(filtered))

        if filtered:
            st.subheader(f"找到 {len(filtered)} 篇匹配的论文")
            
            for paper in filtered:
                if 'source' not in paper:
                    paper['source'] = source
            
            if not show_all_fields:
                display_fields = ['title', 'status', 'track', 'abstract', 'site', 'keywords', 'primary_area', 'award', 'source', 'id']
                filtered_display = []
                for paper in filtered:
                    paper_display = {}
                    for field in display_fields:
                        if field in paper:
                            paper_display[field] = paper[field]
                    if 'id' not in paper_display and 'id' in paper :
                         paper_display['id'] = paper['id']
                    filtered_display.append(paper_display)
                
                st.info("当前只显示部分重要字段。如需查看全部字段，请在侧边栏勾选\"显示全部字段\"选项。")
                st.dataframe(filtered_display)
            else:
                st.dataframe(filtered)

            output_data = {
                "total_papers": len(data),
                "papers_after_status_filter": counts['status_filtered_count'],
                "matching_results": len(filtered),
                "filtered_papers": filtered
            }
            
            filename = f"filtered_results-{source.replace('+', '_')}"
            if keyword:
                filename += f"-{keyword.replace(' ', '_').replace(',', '_')}"
            
            st.download_button(
                label="下载结果 (JSON)",
                data=json.dumps(output_data, ensure_ascii=False, indent=2),
                file_name=f"{filename}.json",
                mime="application/json"
            )
        else:
            st.info("没有找到符合条件的论文。")


# 数据库初始化
DB_PATH = os.path.join(PROJECT_DIR, "user_data.db")

def init_db():
    """初始化数据库，创建用户表、搜索记录表和密码重置申请表"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                keyword TEXT,
                search_mode TEXT,
                fields_to_search TEXT,
                data_search_mode TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                status TEXT DEFAULT 'Pending',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.commit()

        # 添加默认管理员账户
        admin_username = "admin"
        admin_password = hash_password("admin123")  # 默认密码为 "admin123"
        try:
            cursor.execute("""
                INSERT INTO users (username, password, is_admin) 
                VALUES (?, ?, ?)
            """, (admin_username, admin_password, 1))
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # 如果管理员账户已存在，则忽略

# 用户认证
def hash_password(password: str) -> str:
    """对密码进行哈希处理"""
    return sha256(password.encode()).hexdigest()

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """验证用户登录信息"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, is_admin FROM users WHERE username = ? AND password = ?", 
                       (username, hash_password(password)))
        user = cursor.fetchone()
        if user:
            return {"id": user[0], "username": user[1], "is_admin": bool(user[2])}
    return None

def register_user(username: str, password: str) -> bool:
    """注册新用户"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                           (username, hash_password(password)))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# 用户界面
def user_authentication():
	"""改为在主区显示的认证页面，登录/注册成功后自动跳转到搜索页面"""
	# 如果已经在其它地方登录则直接返回（保持向后兼容）
	if st.session_state.get("user"):
		return

	st.subheader("用户认证")
	# 显示任何来自回调的提示
	if st.session_state.get("_auth_success"):
		st.success(st.session_state.get("_auth_success"))
		# 清除提示以避免重复显示
		st.session_state.pop("_auth_success", None)
	if st.session_state.get("_auth_error"):
		st.error(st.session_state.get("_auth_error"))
		st.session_state.pop("_auth_error", None)

	auth_mode = st.radio("选择操作", ["登录", "注册"], horizontal=True, key="auth_mode_main")
	# 使用受控输入，值保存在 st.session_state 对应 key 中，回调可以读取
	username = st.text_input("用户名", key="auth_username_main")
	password = st.text_input("密码", type="password", key="auth_password_main")
	col1, col2 = st.columns([3,1])
	with col1:
		# 使用 on_click 回调处理登录/注册，避免依赖 st.experimental_rerun 或复杂 return 流程
		st.button("提交", key="auth_submit_main", on_click=auth_submit_handler)

	# 忘记密码（在认证页面内提供申请）
	st.markdown("---")
	st.subheader("忘记密码？")
	reset_username = st.text_input("请输入要申请重置密码的用户名", key="reset_username_main")
	if st.button("申请密码重置", key="reset_req_main"):
		if not reset_username:
			st.error("请输入用户名")
		else:
			with sqlite3.connect(DB_PATH) as conn:
				cursor = conn.cursor()
				cursor.execute("SELECT id FROM users WHERE username = ?", (reset_username,))
				user_row = cursor.fetchone()
				if user_row:
					user_id = user_row[0]
					cursor.execute("""
						INSERT INTO password_reset_requests (user_id, username) 
						VALUES (?, ?)
					""", (user_id, reset_username))
					conn.commit()
					st.success("密码重置申请已提交，请联系管理员处理。")
				else:
					st.error("用户名不存在，请检查后重试")

def save_search_history(user_id: int, search_params: Dict[str, Any]):
    """保存用户的搜索记录（仅保存用户输入的原始查询）"""
    original_input = search_params.get("original_keyword", search_params.get("keyword", ""))
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO search_history (user_id, keyword, search_mode, fields_to_search, data_search_mode)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, original_input, search_params["search_mode"], 
              ",".join(search_params["fields_to_search"] or []), search_params["data_search_mode"]))
        conn.commit()

def view_search_history(user_id: int):
    """显示用户的搜索记录"""
    st.subheader("我的搜索记录")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT keyword, search_mode, fields_to_search, data_search_mode, timestamp
            FROM search_history WHERE user_id = ? ORDER BY timestamp DESC
        """, (user_id,))
        records = cursor.fetchall()
        if records:
            for record in records:
                st.write(f"关键词: {record[0]}, 模式: {record[1]}, 字段: {record[2]}, 数据源: {record[3]}, 时间: {record[4]}")
        else:
            st.info("暂无搜索记录")

def admin_manage_users():
    """管理员管理用户信息"""
    st.subheader("用户管理")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, is_admin FROM users")
        users = cursor.fetchall()
        
        for user in users:
            user_id, username, is_admin = user
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                new_username = st.text_input(f"用户名 (ID: {user_id})", value=username, key=f"username_{user_id}")
            with col2:
                new_password = st.text_input(f"新密码 (可选)", value="", key=f"password_{user_id}", type="password")
            with col3:
                new_is_admin = st.checkbox("管理员权限", value=bool(is_admin), key=f"is_admin_{user_id}")
            with col4:
                if st.button("保存", key=f"save_{user_id}"):
                    try:
                        # 更新用户名
                        if new_username != username:
                            cursor.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, user_id))
                        
                        # 更新密码（如果提供了新密码）
                        if new_password:
                            hashed_password = hash_password(new_password)
                            cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
                        
                        # 更新管理员权限
                        cursor.execute("UPDATE users SET is_admin = ? WHERE id = ?", (int(new_is_admin), user_id))
                        
                        conn.commit()
                        st.success(f"用户 {user_id} 的信息已更新")
                    except sqlite3.IntegrityError:
                        st.error(f"用户名 {new_username} 已存在，请选择其他用户名")

def request_password_reset():
    """用户申请密码重置"""
    st.sidebar.subheader("忘记密码？")
    username = st.sidebar.text_input("请输入您的用户名以申请密码重置")
    if st.sidebar.button("申请密码重置"):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if user:
                user_id = user[0]
                cursor.execute("""
                    INSERT INTO password_reset_requests (user_id, username) 
                    VALUES (?, ?)
                """, (user_id, username))
                conn.commit()
                st.sidebar.success("密码重置申请已提交，请联系管理员处理")
            else:
                st.sidebar.error("用户名不存在，请检查后重试")

def admin_manage_password_resets():
    """管理员处理密码重置申请"""
    st.subheader("密码重置申请管理")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, user_id, username, status, timestamp 
            FROM password_reset_requests 
            WHERE status = 'Pending'
        """)
        requests = cursor.fetchall()
        
        if not requests:
            st.info("暂无待处理的密码重置申请")
            return
        
        for req in requests:
            req_id, user_id, username, status, timestamp = req
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"用户名: {username} (申请时间: {timestamp})")
            with col2:
                new_password = st.text_input(f"新密码 (用户: {username})", key=f"new_password_{req_id}", type="password")
            with col3:
                if st.button("重置密码", key=f"reset_{req_id}"):
                    if new_password:
                        hashed_password = hash_password(new_password)
                        try:
                            # 更新用户密码
                            cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
                            # 更新申请状态
                            cursor.execute("UPDATE password_reset_requests SET status = 'Completed' WHERE id = ?", (req_id,))
                            conn.commit()
                            st.success(f"用户 {username} 的密码已成功重置")
                        except Exception as e:
                            st.error(f"重置密码时发生错误: {str(e)}")
                    else:
                        st.error("新密码不能为空")

def test_baidu_qf_api(key: str) -> (bool, str):
    """
    使用提供的文心千帆 Key 做一次最小测试调用，返回 (success, message)。
    注意：下面使用的是通用占位 endpoint；如需兼容正式平台请替换为官方推荐 endpoint 与请求体。
    """
    if not key:
        return False, "未提供文心千帆 Key。"
    try:
        # 使用最小 prompt 进行一次生成尝试
        resp = call_baidu_qf_generate("test", key)
        if resp:
            return True, "文心千帆 Key 可用（已返回响应）。"
        return False, "文心千帆未返回有效响应（请检查 Key 与网络）。"
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("测试文心千帆失败: %s\n%s", e, tb)
        return False, f"测试时发生异常: {e}"

def call_baidu_qf_generate(natural_query: str, key: str) -> Optional[str]:
    """
    调用文心千帆生成接口，将自然语言查询转为简短英文关键词（逗号分隔）。
    说明：
      - 这里使用一个通用占位 RPC endpoint，并把用户提供的 bce-v3/... Key 放到 Authorization 头中。

    返回：生成的文本（string）或 None（失败）。
    """
    if not key:
        return None
    try:
        gen_url = os.environ.get("BAIDU_QF_ENDPOINT", "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxin")
        headers = {
            "Content-Type": "application/json",
            "Authorization": key
        }
        prompt = (
            "将下面的用户搜索意图解析为一组简短的英文关键词或短语，使用逗号分隔，"
            "每个关键词/短语尽量短且能覆盖主要搜索意图，只输出关键词列表，不要说明文字。\n\n"
            f"用户查询: {natural_query}\n\n关键词:"
        )
        payload = {"prompt": prompt, "max_tokens": 150}
        gen_resp = requests.post(gen_url, json=payload, headers=headers, timeout=20)
        if gen_resp.status_code != 200:
            logger.warning("文心千帆生成接口返回非200: %s %s", gen_resp.status_code, gen_resp.text)
            return None
        jr = gen_resp.json()
        # 尝试多种返回字段提取文本
        for k in ("result", "output", "text", "choices", "data"):
            if k in jr:
                val = jr[k]
                if isinstance(val, list) and val:
                    candidate = val[0]
                    if isinstance(candidate, dict):
                        for sub in ("content", "text", "output"):
                            if sub in candidate:
                                return candidate[sub].strip()
                    if isinstance(candidate, str):
                        return candidate.strip()
                elif isinstance(val, str):
                    return val.strip()
        # 最后退回到完整 JSON 字符串（便于调试）
        return str(jr)
    except Exception as e:
        logger.error("调用文心千帆失败: %s\n%s", e, traceback.format_exc())
        return None

# 新增：将自然语言转换为关键词（优先调用文心千帆，失败回退本地分词）
def generate_keywords_via_model(natural_query: str, baidu_key: str = "") -> str:
    """
    使用文心千帆将自然语言查询转成英文关键词（逗号分隔）。
    若调用失败则回退到本地简单分词（小写、以逗号分隔）。
    """
    if not natural_query or natural_query.strip() == "":
        return ""
    key = baidu_key or os.environ.get("BAIDU_QF_KEY", BAIDU_QF_KEY)
    if key:
        try:
            result = call_baidu_qf_generate(natural_query, key)
            if result:
                return result.strip()
            else:
                logger.warning("文心千帆未返回有效文本，使用本地分词回退。")
        except Exception as e:
            logger.error("文心千帆处理异常: %s\n%s", e, traceback.format_exc())

    # 回退策略：本地分词（小写、逗号分隔）
    tokens = [t.strip().lower() for t in natural_query.replace(',', ' ').split() if t.strip()]
    return ", ".join(tokens)

def go_to_auth():
    """将页面状态切换到认证页（供按钮回调使用）"""
    st.session_state['page'] = 'auth'

def auth_submit_handler():
    """
    按钮回调：从 st.session_state 读取 auth_mode/auth_username_main/auth_password_main，
    执行登录或注册。成功时写入 st.session_state['user'] 并设 page='search'。
    通过回调触发的按钮点击会导致 Streamlit 重跑，从而实现页面跳转。
    """
    mode = st.session_state.get("auth_mode_main", "登录")
    username = st.session_state.get("auth_username_main", "").strip()
    password = st.session_state.get("auth_password_main", "")
    # 清理旧提示
    st.session_state.pop("_auth_success", None)
    st.session_state.pop("_auth_error", None)

    if not username or not password:
        st.session_state["_auth_error"] = "用户名和密码不能为空"
        return

    if mode == "登录":
        user = authenticate_user(username, password)
        if user:
            st.session_state["user"] = user
            st.session_state['page'] = 'search'
            st.session_state["_auth_success"] = f"欢迎回来，{user['username']}！"
        else:
            st.session_state["_auth_error"] = "用户名或密码错误"
    else:  # 注册
        ok = register_user(username, password)
        if ok:
            user = authenticate_user(username, password)
            if user:
                st.session_state["user"] = user
                # 修正：注册成功后也切换到搜索页面
                st.session_state['page'] = 'search'
                st.session_state["_auth_success"] = f"注册成功，已登录：{user['username']}"
            else:
                st.session_state["_auth_error"] = "注册成功但自动登录失败，请手动登录"
        else:
            st.session_state["_auth_error"] = "用户名已存在"

def go_to_page(page: str):
    """页面跳转回调"""
    st.session_state['page'] = page

def logout():
    """登出"""
    if 'user' in st.session_state:
        st.session_state.pop('user')
    st.session_state['page'] = 'search'

def train_and_evaluate_model():
    """
    训练和评估模型。
    """
    st.subheader("模型训练与评估")

    # 动态配置训练数据文件路径
    data_file = st.text_input("输入训练数据文件路径,如:key_infos/aaai/aaai2021.json", value="data/train.json")
    model_path = st.text_input("输入模型保存路径:", value="models/pasa_model")

    if st.button("加载数据"):
        # 加载数据
        data = load_json_data(data_file)
        logger.info(f"加载数据文件: {data_file}, 数据条目数: {len(data)}")
        if not data:
            st.error("加载数据失败，文件可能为空或格式不正确")
            return
        data = preprocess_data(data)
        data = extract_features(data)
        data = augment_data(data)
        st.info(f"数据预处理、特征提取和增强完成，共 {len(data)} 条记录")

        model = PASAModel(model_path)
        model.train(data)
        st.success("模型训练完成")

        metrics = model.evaluate(data)
        st.write("评估结果:", metrics)

def main():
    """主函数"""
    st.set_page_config(page_title="论文搜索工具", layout="wide")
    init_db()

    # 初始化页面状态（默认显示搜索页）
    if 'page' not in st.session_state:
        st.session_state['page'] = 'search'

    # 可选：获取当前用户（可能为 None）
    user = st.session_state.get("user")

    # 顶部布局：右上角显示 登录/注册 或 用户名 + 登出
    top_cols = st.columns([8, 1, 1])
    with top_cols[1]:
        if user:
            st.button(f"{user.get('username')}", key="user_label_btn")
        else:
            st.button("登录/注册", key="top_auth_button", on_click=go_to_auth)
    with top_cols[2]:
        if user:
            st.button("登出", key="logout_btn", on_click=logout)

    # 如果当前页面是认证页面，则显示认证界面并返回
    if st.session_state.get('page') == 'auth':
        user_authentication()
        return

    # 如果用户已登录，显示导航按钮（我的记录；管理员额外显示用户管理与密码重置管理）
    if user:
        nav_cols = st.columns([1,1,1])
        # 我的记录（所有登录用户可见）
        with nav_cols[0]:
            st.button("我的记录", key="nav_history", on_click=lambda: go_to_page('history'))
        # 管理用户（仅管理员）
        with nav_cols[1]:
            if user.get("is_admin"):
                st.button("用户管理", key="nav_manage_users", on_click=lambda: go_to_page('manage_users'))
        # 密码重置管理（仅管理员）
        with nav_cols[2]:
            if user.get("is_admin"):
                st.button("密码重置申请管理", key="nav_manage_resets", on_click=lambda: go_to_page('manage_resets'))

    # 页面路由：根据 page 渲染不同视图
    page = st.session_state.get('page', 'search')

    # 管理页面或记录页面显示时，提供返回搜索的按钮
    if page in ('manage_users', 'manage_resets', 'history'):
        if st.button("返回搜索", key="back_to_search"):
            st.session_state['page'] = 'search'
            return

    if page == 'manage_users':
        # 仅管理员可访问
        if not user or not user.get('is_admin'):
            st.error("只有管理员可以访问用户管理页面。")
            return
        admin_manage_users()
        return

    if page == 'manage_resets':
        # 仅管理员可访问
        if not user or not user.get('is_admin'):
            st.error("只有管理员可以访问密码重置申请管理页面。")
            return
        admin_manage_password_resets()
        return

    if page == 'history':
        # 仅登录用户可查看自己的记录
        if not user:
            st.error("请先登录以查看您的搜索记录。")
            return
        view_search_history(user['id'])
        return

    # 默认：搜索页面（所有用户可访问）
    # 如果到这里，page 应为 'search'
    # 搜索功能（所有用户均可访问）
    search_params = create_search_sidebar()
    if st.button("搜索论文"):
        # 保留用户原始输入
        original_query = search_params.get("keyword", "")
        search_params["original_keyword"] = original_query

        # 若启用自然语言解析（文心千帆），先调用生成关键词并扩展检索字段包含全文
        if search_params.get("use_nl"):
            baidu_key = search_params.get("baidu_key", "") or os.environ.get("BAIDU_QF_KEY", BAIDU_QF_KEY)
            with st.spinner("正在使用文心千帆解析自然语言查询..."):
                nl_keywords = generate_keywords_via_model(original_query, baidu_key)
            if nl_keywords:
                st.info(f"系统生成关键词: {nl_keywords}")
                # 将用于后端检索的 keyword 字段设为模型生成的英文关键词
                search_params["keyword"] = nl_keywords
                # 将检索字段扩大到包含全文（如果数据包含此字段）
                # 优先使用用户选择的 fields_to_search（若未选择则使用 DEFAULT_FIELDS）
                fields = search_params.get("fields_to_search") or DEFAULT_FIELDS.copy()
                if "full_text" not in fields:
                    fields = fields + ["full_text"]
                search_params["fields_to_search"] = fields
            else:
                st.warning("未能生成关键词，使用原始输入进行匹配")
                # 保证检索时包含全文
                fields = search_params.get("fields_to_search") or DEFAULT_FIELDS.copy()
                if "full_text" not in fields:
                    fields = fields + ["full_text"]
                search_params["fields_to_search"] = fields
        else:
            # 不使用自然语言解析，保持用户原始输入和选择的字段
            search_params["keyword"] = original_query

        data, source, key_fields_filters = load_data_source(search_params["data_search_mode"])
        search_params["key_fields_filters"] = key_fields_filters
        display_search_results(data, source, search_params)

        # 仅在用户已登录时保存搜索历史
        if user:
            save_search_history(user["id"], search_params)

    # 添加训练与评估按钮
    if user and user.get("is_admin"):
        if st.button("模型训练与评估", key="train_button"):
            st.session_state['page'] = 'train'

    if page == 'train':
        train_and_evaluate_model()

# 添加主入口，确保可直接运行
if __name__ == "__main__":
    main()