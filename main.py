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

    def build_knowledge_base(self):
        """构建知识库"""
        print("🚀 开始构建知识库...")

        # 1 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()
        
        if vectorstore is not None:
            print("✅ 已加载已保存的索引！")
            print("加载食谱文档")


        else:
            print("⚠️ 未找到已保存的索引，将构建新索引...")

            self.data_module.load_documents()

        print("✅ 知识库构建完成！")


    def run_interactivate(self):
        """ 运行交互式问答循环 """
        print("🚀 开始运行交互式问答循环...")
        self.initialize_system()


        #构建知识库
        slef.build_knowledge_base()




        while True:
            query = input("\n请输入您的问题（输入'退出'结束）: ")
            if query.lower() == '退出':
                print("✅ 会话结束！")
                break

            # 处理查询
            response = self.handle_query(query)
            print(f"🤖 回答: {response}")





def main():

    rag_system = RAGSystem()
    
    

    rag_system.run_interactive()




if __name__ == "__main__":
    main()
