"""
工具拦截器 —— 在工具调用前进行拦截/审批。
同步版本用于 ToolRegistry.execute_tool，异步版本用于 ToolRegistry.aexecute_tool。
"""


from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Type
from .base import Tool


class ToolInterceptor(ABC):
    """工具拦截器基类 —— 在工具执行前进行拦截，决定是否放行。"""

    def __init__(self,
                 whitelist: Optional[List[Type[Tool]]] = None,
                 intercept_list: Optional[List[Type[Tool]]] = None,
                 auto_approve_if_no_rules: bool = False
                 ):
        """初始化工具拦截器
        假设一个tool在whitelist和intercept_list中都存在，则以intercept_list为准，即需要拦截。
        假设一个tool既不在whitelist中，也不在intercept_list中，则根据auto_approve_if_no_rules的值来决定是否需要拦截。


        Args:
            whitelist (Optional[List[Type[Tool]]], optional): 白名单工具列表，自动放行. Defaults to None.
            intercept_list (Optional[List[Type[Tool]]], optional): 拦截列表，需要拦截确认. Defaults to None.
            auto_approve_if_no_rules (bool, optional): 当工具不在任何列表中时是否自动放行. Defaults to False.
        """
        self.whitelist = whitelist or []
        self.intercept_list = intercept_list or []
        self.auto_approve_if_no_rules = auto_approve_if_no_rules

    def intercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        """
        同步拦截 —— 根据拦截规则决定是否放行工具调用。
        由 ToolRegistry.execute_tool 调用，调用 do_intercept()。
        """
        if type(tool) in self.intercept_list:
            return self.do_intercept(tool, parameters, tool_call_id)
        if type(tool) in self.whitelist:
            return True
        if self.auto_approve_if_no_rules == False:
            return self.do_intercept(tool, parameters, tool_call_id)
        else:
            return True

    async def aintercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        """
        异步拦截 —— 根据拦截规则决定是否放行工具调用。
        由 ToolRegistry.aexecute_tool 调用，调用 ado_intercept()。
        默认实现回退到同步版本，子类可覆写以提供真正的异步实现。
        """
        if type(tool) in self.intercept_list:
            return await self.ado_intercept(tool, parameters, tool_call_id)
        if type(tool) in self.whitelist:
            return True
        if self.auto_approve_if_no_rules == False:
            return await self.ado_intercept(tool, parameters, tool_call_id)
        else:
            return True

    @abstractmethod
    def do_intercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        """具体的同步拦截逻辑，由子类实现。"""

    async def ado_intercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        """
        具体的异步拦截逻辑。默认回退到同步版本 do_intercept()。
        如果子类的 do_intercept 包含阻塞 I/O（如 input()），应覆写此方法
        将阻塞操作放到线程池执行，避免阻塞事件循环。
        """
        return self.do_intercept(tool, parameters, tool_call_id)
