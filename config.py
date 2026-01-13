from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    data_path: str = "./data"
    index_path: str = "./index"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "Qwen/Qwen2.5"

    top_k: int = 3
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典创建配置实例"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

DEFAULT_CONFIG = RAGConfig()