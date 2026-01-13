import os
import sys
import time
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from config import RAGConfig, DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)
load_dotenv()

class RAGSystem:
    """RAG系统主类"""

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        if not Path(self.config.data_path).exists():
            logger.info(f"数据路径 {self.config.data_path} 不存在")
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

    def initialize_system(self):
        """初始化RAG系统模块"""
        # 开始初始化数据模块
        self.data_module = DataPreparationModule(self.config.data_path)

        #初始化索引构建模块``
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )
        
        #初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        print("✅ 系统初始化完成！")
