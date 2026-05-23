"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from .llm import VioletAgentsLLM
from .config import Config
from .message import Message
from collections import deque
class Agent(ABC):
    """Agent基类"""
    
    def __init__(self, 
                 name: str,
                 llm: VioletAgentsLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None):
        """
        初始化
        
        Args:
            name (str): 代理名称
            llm (VioletAgentsLLM): LLM模型
            system_prompt (Optional[str], optional): 系统提示. Defaults to None.
            config (Optional[Config], optional): 配置. Defaults to None.
        """
        self.name = name
        self.llm = llm or VioletAgentsLLM()
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history = deque[Message](maxlen=self.config.max_history_length if self.config.max_history_length > 0 else None)

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> Message:
        """运行Agent，处理输入并返回任务完成后的结果"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def get_history(self) -> deque[Message]:
        """
        获取消息历史记录的副本，返回一个新的deque对象，避免外部修改原始历史记录
        Returns:
            deque[Message]: 消息历史记录的副本
        """
        return self._history.copy()
    
    def clear_history(self):
        """清空消息历史记录"""
        self._history.clear()
