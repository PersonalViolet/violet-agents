
from typing import Dict, Any, Optional, List, Type
from ..approval_tool import ApprovalTool
from ..base import Tool
from ..builtin import TerminalTool


class DefaultApprovalTool(ApprovalTool):
    """默认审批工具"""

    def __init__(self,
                 auto_approve_tools: Optional[List[Type[Tool]]] = None,
                 require_approval_tools: Optional[List[Type[Tool]]] = None,
                 auto_approve_if_no_rules: bool = False,                  
                 max_attempts: int = 10):
        """初始化审批工具
        假设一个tool在auto_approve_tools和require_approval_tools中都存在，则以require_approval_tools为准，即需要审批。
        假设一个tool既不在auto_approve_tools中，也不在require_approval_tools中，则根据auto_approve_if_no_rules的值来决定是否需要审批。

        
        Args:
            auto_approve_tools (Optional[List[Type[Tool]]], optional): 自动批准的工具列表. Defaults to None.
            require_approval_tools (Optional[List[Type[Tool]]], optional): 需要审批的工具列表. Defaults to None.
            auto_approve_if_no_rules (bool, optional): 当工具不在任何列表中时是否自动批准. Defaults to False.
            max_attempts (int, optional): 最大尝试次数. Defaults to 10.
        """
        super().__init__(auto_approve_tools, require_approval_tools, auto_approve_if_no_rules)
        self.max_attempts = max_attempts

    def do_approve(self, tool: Tool, parameters: Dict[str, Any], tool_call_id: str) -> bool:
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
                print(f"无效输入，请输入 'y' 或 'n'\n 剩余尝试次数: {max_attempts - 1}")
                max_attempts -= 1
        print("超过最大尝试次数，默认拒绝该工具调用")
        return False
