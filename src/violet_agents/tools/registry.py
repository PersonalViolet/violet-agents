

from typing import Dict, Optional, Any, List, Union
from .base import Tool
from ..core.message import Message
import json
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall
from .interceptor import ToolInterceptor
class ToolRegistry:
    """
    工具注册表，用于管理和调用工具

    Attributes:
        _tools (Dict[str, Tool]): 已注册的工具字典，键为工具名称，值为Tool对象
        _defer_tools (Dict[str, Tool]): 延迟工具字典，专门用于存储那些需要等到Agent发现后才调用的工具
        interceptor (Optional[ToolInterceptor]): 可选的审批工具实例，如果提供了审批工具，在执行任何工具前都会先进行用户审批

    """
    def __init__(self, interceptor: Optional[ToolInterceptor] = None):
        self._tools: Dict[str, Tool] = {}
        self._defer_tools: Dict[str, Tool] = {} # 延迟工具字典，不会对外暴露，专门用于存储那些需要等到Agent发现后才调用的工具
        self.interceptor = interceptor

    
    def register_tool(self, 
                      tool: Tool,
                      is_defer: bool = False):
        """
        注册工具

        Args:
            tool (Tool): 要注册的工具实例
        """
        # 如果工具名已存在于目标字典中，打印覆盖提示
        if (is_defer and tool.name in self._defer_tools) or (not is_defer and tool.name in self._tools):
            print(f"工具 {tool.name} 已经注册，覆盖原有工具")
        
        # 如果工具名存在于另一个字典中，则删除它（实现覆盖行为）
        if tool.name in self._tools and is_defer:
            # 如果要注册为 defer 工具，但该名称已在普通工具中存在，则删除普通工具
            del self._tools[tool.name]
            print(f"工具 {tool.name} 从普通工具移动到延迟工具")
        elif tool.name in self._defer_tools and not is_defer:
            # 如果要注册为普通工具，但该名称已在 defer 工具中存在，则删除 defer 工具
            del self._defer_tools[tool.name]
            print(f"工具 {tool.name} 从延迟工具移动到普通工具")
        
        if is_defer:
            self._defer_tools[tool.name] = tool
            print(f"工具 {tool.name} 注册为延迟工具")
        else:
            self._tools[tool.name] = tool
            print(f"工具 {tool.name} 注册为普通工具")

    def register_tools(self, *tools: Tool, is_defer: bool = False) -> "ToolRegistry":
        """
        批量注册工具，支持链式调用

        Args:
            *tools (Tool): 要注册的工具实例列表
            is_defer (bool): 是否将这些工具注册为延迟工具，默认为 False
        """
        for tool in tools:
            self.register_tool(tool, is_defer=is_defer)
        return self

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取在_tools的Tool对象"""
        return self._tools.get(name)

    def get_defer_tools(self) -> Dict[str, Tool]:
        """获取所有延迟工具"""
        return self._defer_tools

    def get_all_tools(self) -> Dict[str, Tool]:
        """获取所有工具（包括普通工具和延迟工具）"""
        all_tools = self._tools.copy()
        all_tools.update(self._defer_tools)
        return all_tools

    def execute_tool(self, tool_call: Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]) -> Message:
        """
        执行_tools, _defer_tools中注册的工具

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

        if tool_name not in self._tools and tool_name not in self._defer_tools:
            raise ValueError(f"工具 {tool_name} 未注册")
        if tool_name in self._tools:
            tool = self._tools[tool_name]
        elif tool_name in self._defer_tools:
            tool = self._defer_tools[tool_name]
        else:
            raise ValueError(f"工具 {tool_name} 未找到")
        
        if not tool.validate_parameters(parameters):
            raise ValueError(f"工具 {tool_name} 参数验证失败，缺少必要参数")
        
        if self.interceptor:
            approved = self.interceptor.intercept(tool, parameters, tool_call_id)
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

    def reset_all_tools(self) -> None:
        """重置所有已注册工具到初始状态。"""
        for tool in list(self._tools.values()) + list(self._defer_tools.values()):
            tool.reset()
