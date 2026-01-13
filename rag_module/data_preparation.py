import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGROY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGROY_LABELS = list(set(CATEGROY_MAPPING.values()))
    DIFFICULTY_LABELS = ['非常简单','简单', '中等', '困难', '非常困难']

    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.documents: List[Document] = []     #父文档 完整的文档内容
        self.chunks: List[Document] = []        #子文档 按标题分割的小块
        self.parent_child_map: Dict[str, str] = {} #子文档id到父文档id的映射
        