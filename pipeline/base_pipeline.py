from abc import ABC, abstractmethod
from configs.llm_config import LLMConfig
from provider.llm_provider_registry import create_llm_instance
from utils.cost_manager import CostManager

class BasePipeline(ABC):
    def __init__(self) -> None:
        self.llm = create_llm_instance(LLMConfig.default())
        self.llm.cost_manager = CostManager()
    
    @abstractmethod
    def __call__(self, instruction: str):
        raise NotImplementedError("Subclasses must implement this method")