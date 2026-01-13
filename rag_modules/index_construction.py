import logging
from typing import List
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """索引构建模块 - 负责构建向量索引"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称
            index_save_path: 索引保存路径
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("嵌入模型初始化完成")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引

        Args:
            chunks: 文档分块列表

        Returns:
            FAISS向量索引
        """
        logger.info(f"正在构建向量索引，文档数量: {len(chunks)}")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info(f"向量索引构建完成，索引保存路径: {self.index_save_path}")
        return self.vectorstore
    
    def add_documents(self, new_chunks: List[Document]):
        """
        添加新文档到索引

        Args:
            new_chunks: 新文档分块列表
        """
        if not self.vectorstore:
            logger.error("索引未构建，请先调用 build_vector_index 方法")
            raise ValueError("索引未构建，请先调用 build_vector_index 方法")
        
        logger.info(f"正在添加新文档到索引，文档数量: {len(new_chunks)}")
        self.vectorstore.add_documents(new_chunks)
        logger.info(f"新文档添加完成")
        
    def load_index(self):
        """加载已保存的索引"""

        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.error(f"索引保存路径不存在: {self.index_save_path}")
            raise FileNotFoundError(f"索引保存路径不存在: {self.index_save_path}")

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"成功加载索引: {self.index_save_path}")
            return self.vectorstore
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        执行相似度搜索

        Args:
            query: 查询文本
            k: 返回的文档数量，默认5

        Returns:
            相似度最高的文档列表
        """
        if not self.vectorstore:
            logger.error("索引未加载，请先调用 load_index 方法")
            raise ValueError("索引未加载，请先调用 load_index 方法")
        
        logger.info(f"正在执行相似度搜索，查询: {query}，返回文档数量: {k}")
        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"相似度搜索完成，返回文档数量: {len(results)}")
        return results