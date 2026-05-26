

from typing import Dict, Optional, Any, List, Union
from .base import Tool
from ..core.message import Message
import json
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall
from .approval_tool import ApprovalTool
class ToolRegistry:
    """
    工具注册表，用于管理和调用工具
    """
    def __init__(self, approval_tool: Optional[ApprovalTool] = None):
        self._tools: Dict[str, Tool] = {}
        self.approval_tool = approval_tool

    
    def register_tool(self, tool: Tool):
        """
        注册工具

        Args:
            tool (Tool): 要注册的工具实例
        """
        if tool.name in self._tools:
            print(f"工具 {tool.name} 已经注册，覆盖原有工具")
        self._tools[tool.name] = tool
        print(f"工具 {tool.name} 注册成功")

    def execute_tool(self, tool_call: Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]) -> Message:
        """
        执行工具

        Args:
            tool_call (Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]): 工具调用信息，llm返回的工具调用格式，包含工具名称、参数等信息

        Returns:
            Message: 工具执行结果封装的Message对象
        """
        if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
            tool_call = tool_call.model_dump()

        if tool_call.get("type") != "function":
            raise ValueError("工具调用类型目前仅支持 'function'")
        
        tool_call_id = tool_call.get("id", "")
        if not tool_call_id:
            raise ValueError("工具调用信息缺少 'id' 字段，无法唯一标识工具调用")
        
        function = tool_call.get("function", {})
        if not function:
            raise ValueError("工具调用信息缺少 'function' 字段")
        
        tool_name = function.get("name")
        arguments = function.get("arguments", "{}")
        try:
            parameters: Dict[str, Any] = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise ValueError(f"工具参数解析失败，确保参数是有效的JSON字符串: {e}")

        if tool_name not in self._tools:
            raise ValueError(f"工具 {tool_name} 未注册")
        tool = self._tools[tool_name]
        if not tool.validate_parameters(parameters):
            raise ValueError(f"工具 {tool_name} 参数验证失败，缺少必要参数")
        
        if self.approval_tool:
            approved = self.approval_tool.approve(tool, parameters, tool_call_id)
            if not approved:
                return Message(role="tool", content=f"❌ 工具调用未通过用户的审批，已被用户拒绝: {tool_name}", tool_call_id=tool_call_id)

        return tool.run(parameters, tool_call_id)

    def get_openai_tools(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        获取指定工具列表，转换为OpenAI API要求的格式

        Args:
            tool_names (List[str]): 工具名称列表

        Returns:
            List[Dict[str, Any]]: 工具列表
        """
        if tool_names is None:
            return [tool.to_openai_dict() for tool in self._tools.values()]
        return [self._tools[name].to_openai_dict() for name in tool_names if name in self._tools]
