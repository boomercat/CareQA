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
        




    def load_documents(self) -> List[Document]:
        """加载所有文档"""

        logger.info(f"正在从{self.data_path}加载文档...")


        #直接读取markdown文件保持原格式
        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.glob('*.md'):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                data_root = Path(self.data_path).resolve()
                relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()

            parent_id = hashlib.md5(relative_path.encode()).hexdigest()

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "parent_id": parent_id,
                    "doc_type": "parent" #标记为父文档
                }
            )
            documents.append(doc)

        for doc in documents:
            self._enhance_metadata(doc)
        
        self.documents = documents
        logger.info(f"成功加载{len(documents)}个文档")
        return documents

    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据
        
        Args:
            doc: 需要增强元数据的文档
        """
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts
        
        doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break
        
        # 提取名称
        doc.metadata['dish_name'] = file_path.stem

        # 分析难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'