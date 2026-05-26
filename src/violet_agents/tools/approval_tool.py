"""
审批工具的调用
"""


from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Type
from ..core.llm import VioletAgentsLLM
from ..core.config import Config
from ..core.message import Message
from collections import deque
from .base import Tool, ToolParameters, ToolProperty


class ApprovalTool(ABC):
    """审批工具基类"""
    
    def __init__(self,
                 auto_approve_tools: Optional[List[Type[Tool]]] = None,
                 require_approval_tools: Optional[List[Type[Tool]]] = None,
                 auto_approve_if_no_rules: bool = False
                 ):
        """初始化审批工具
        假设一个tool在auto_approve_tools和require_approval_tools中都存在，则以require_approval_tools为准，即需要审批。
        假设一个tool既不在auto_approve_tools中，也不在require_approval_tools中，则根据auto_approve_if_no_rules的值来决定是否需要审批。

        
        Args:
            auto_approve_tools (Optional[List[Type[Tool]]], optional): 自动批准的工具列表. Defaults to None.
            require_approval_tools (Optional[List[Type[Tool]]], optional): 需要审批的工具列表. Defaults to None.
            auto_approve_if_no_rules (bool, optional): 当工具不在任何列表中时是否自动批准. Defaults to False.
        """
        self.auto_approve_tools = auto_approve_tools or []
        self.require_approval_tools = require_approval_tools or []
        self.auto_approve_if_no_rules = auto_approve_if_no_rules
        
    def approve(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        """
        根据工具类型和审批规则决定是否批准工具调用
        Args:
            tool (Tool): 被调用的工具实例
            parameters (Dict[str, Any]): 工具参数
            tool_call_id (str): 工具调用ID
        Returns:
            bool: 是否批准工具调用
        """
        if type(tool) in self.require_approval_tools:
            return self.do_approve(tool, parameters, tool_call_id)
        if type(tool) in self.auto_approve_tools:
            return True
        if self.auto_approve_if_no_rules == False:
            return self.do_approve(tool, parameters, tool_call_id)
        else:
            return True

    @abstractmethod
    def do_approve(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        """具体的审批逻辑，由子类实现
        
        Args:
            tool (Tool): 被调用的工具实例
            parameters (Dict[str, Any]): 工具参数
            tool_call_id (str): 工具调用ID
        """
        pass
