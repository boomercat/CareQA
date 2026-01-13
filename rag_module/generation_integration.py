import os
import logging
from typing import List, Dict, Any


logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        logger.info(f"正在初始化LLM模型: {self.model_name}")

        self.llm = ChatOpenAI(
            model=os.getenv(self.model_name),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=os.getenv("OPENAI_API_BASE"),
        )
        logger.info("LLM初始化完成")
