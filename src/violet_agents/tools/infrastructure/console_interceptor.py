
import asyncio
import logging
from typing import Dict, Any, Optional, List, Type
from ..interceptor import ToolInterceptor
from ..base import Tool
from ..builtin import TerminalTool

logger = logging.getLogger(__name__)


class ConsoleConfirmInterceptor(ToolInterceptor):
    """终端交互式拦截确认 —— 通过命令行输入 y/n 决定是否放行工具调用。"""

    def __init__(self,
                 whitelist: Optional[List[Type[Tool]]] = None,
                 intercept_list: Optional[List[Type[Tool]]] = None,
                 auto_approve_if_no_rules: bool = False,
                 max_attempts: int = 10):
        """初始化终端拦截确认器
        假设一个tool在whitelist和intercept_list中都存在，则以intercept_list为准，即需要拦截。
        假设一个tool既不在whitelist中，也不在intercept_list中，则根据auto_approve_if_no_rules的值来决定是否需要拦截。


        Args:
            whitelist (Optional[List[Type[Tool]]], optional): 自动批准的工具列表. Defaults to None.
            intercept_list (Optional[List[Type[Tool]]], optional): 需要审批的工具列表. Defaults to None.
            auto_approve_if_no_rules (bool, optional): 当工具不在任何列表中时是否自动批准. Defaults to False.
            max_attempts (int, optional): 最大尝试次数. Defaults to 10.
        """
        super().__init__(whitelist, intercept_list, auto_approve_if_no_rules)
        self.max_attempts = max_attempts

    def do_intercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        if isinstance(tool, TerminalTool):
            current_dir = tool.current_dir
            prompt = f"{current_dir}> {parameters.get('command', '')}"
        else:
            prompt = f"工具名称: {tool.name}\n工具描述: {tool.description}\n工具参数: {parameters}"
        max_attempts = self.max_attempts
        while max_attempts > 0:
            is_approved = input(f"{prompt}\n是否批准该工具调用？(y/n): ")
            if is_approved.lower() == "y":
                return True
            elif is_approved.lower() == "n":
                return False
            else:
                logger.warning("无效输入，请输入 'y' 或 'n'，剩余尝试次数: %s", max_attempts - 1)
                max_attempts -= 1
        logger.warning("超过最大尝试次数，默认拒绝该工具调用")
        return False

    async def ado_intercept(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
        """异步版拦截：将阻塞的 input() 放到线程池执行，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.do_intercept, tool, parameters, tool_call_id)
