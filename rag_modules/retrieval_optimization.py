import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()
    
    def setup_retrievers(self):
        """设置向量检索器"""
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retrieval_k}
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )

        logger.info(f"检索器已设置，返回前5个文档")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量，默认3
        
        Returns:
            混合检索到的文档列表
        """
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 使用RRF 重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        
        return reranked_docs[:top_k]
        
    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF重排文档
        
        Args:
            vector_docs: 向量检索到的文档列表
            bm25_docs: BM25检索到的文档列表
            k: RRF参数，默认60
        
        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        #计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            # RRF 公式 1 / (k+rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f'向量检索文档 {doc_id} 排名 {rank} 分数 {rrf_score:.4f}')
        
        # BM25检索结果的 RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = oc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            logger.debug(f'BM25检索文档 {doc_id} 排名 {rank} 分数 {rrf_score:.4f}')

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        #构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                #将RRF数据添加到文档元数据中
                doc.metadata['final_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f'最终排序 - 文档 {doc_id} 最终分数 {final_score:.4f}')

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")
        return reranked_docs