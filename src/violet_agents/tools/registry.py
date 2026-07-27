

from typing import Dict, Optional, Any, List, Union
from .base import Tool
from ..core.message import Message
import json
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall
from .interceptor import ToolInterceptor
import concurrent.futures
import asyncio
import contextvars

class ToolRegistry:
    """
    工具注册表，用于管理和调用工具

    Attributes:
        _tools (Dict[str, Tool]): 已注册的工具字典，键为工具名称，值为Tool对象
        _defer_tools (Dict[str, Tool]): 延迟工具字典，专门用于存储那些需要等到Agent发现后才调用的工具
        interceptor (Optional[ToolInterceptor]): 可选的审批工具实例，如果提供了审批工具，在执行任何工具前都会先进行用户审批

    """
    def __init__(self, 
                 interceptor: Optional[ToolInterceptor] = None,
                 max_workers: int = 5):
        """
        初始化工具注册表

        Args:
            interceptor (Optional[ToolInterceptor], optional): 可选的审批工具实例. Defaults to None.
            max_workers (int, optional): 线程池最大工作线程数，用于异步执行工具. Defaults to 5.
        """
        self._tools: Dict[str, Tool] = {}
        self._defer_tools: Dict[str, Tool] = {} # 延迟工具字典，不会对外暴露，专门用于存储那些需要等到Agent发现后才调用的工具
        # 动态工具源：持有外部可变字典的引用，查询时实时反映外部变更
        self._dynamic_sources: List[Dict[str, Tool]] = []
        self._dynamic_defer_sources: List[Dict[str, Tool]] = []
        self.interceptor = interceptor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    
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

    def register_dynamic_tools(
        self,
        tools_source: Dict[str, Tool],
        is_defer: bool = False
    ) -> "ToolRegistry":
        """
        动态注册一批工具。传入的 tools_source 是一个可变字典，由外部代码创建和维护。
        注册表仅持有该字典的引用，查询工具时会实时反映外部对该字典的修改。

        适用场景：工具集由其他模块动态管理（如 MCP 协议发现的远程工具、插件系统等），
        注册表无需手动同步即可感知外部增删改。

        Args:
            tools_source: 工具字典，键为工具名称，值为 Tool 对象。
                          该字典由外部代码创建和维护，注册表持有其引用。
            is_defer: 是否注册到延迟工具区，默认注册到普通工具区。

        Returns:
            self，支持链式调用

        Example:
            >>> registry = ToolRegistry()
            >>> mcp_tools: Dict[str, Tool] = {}
            >>> registry.register_dynamic_tools(mcp_tools)
            >>> # 外部代码添加工具
            >>> mcp_tools["remote_search"] = RemoteSearchTool()
            >>> # 注册表自动感知
            >>> tool = registry.get_tool("remote_search")  # 返回 RemoteSearchTool
        """
        if is_defer:
            self._dynamic_defer_sources.append(tools_source)
        else:
            self._dynamic_sources.append(tools_source)
        return self

    def unregister_dynamic_tools(self, tools_source: Dict[str, Tool]) -> "ToolRegistry":
        """
        移除之前通过 register_dynamic_tools 注册的动态工具源。

        Args:
            tools_source: 之前注册的动态工具字典引用。

        Returns:
            self，支持链式调用

        Raises:
            ValueError: 如果 tools_source 未曾注册过。
        """
        if tools_source in self._dynamic_sources:
            self._dynamic_sources.remove(tools_source)
        elif tools_source in self._dynamic_defer_sources:
            self._dynamic_defer_sources.remove(tools_source)
        else:
            raise ValueError("指定的动态工具源未注册")
        return self

    def _resolve_tool(self, name: str) -> Optional[Tool]:
        """
        在所有工具源中按优先级查找工具：
        _tools > _dynamic_sources > _defer_tools > _dynamic_defer_sources

        显式注册的工具优先于动态源中的同名工具。
        """
        if name in self._tools:
            return self._tools[name]
        for source in self._dynamic_sources:
            if name in source:
                return source[name]
        if name in self._defer_tools:
            return self._defer_tools[name]
        for source in self._dynamic_defer_sources:
            if name in source:
                return source[name]
        return None

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        获取指定名称的普通工具对象（不包含延迟工具）。
        查找优先级：_tools > _dynamic_sources
        """
        if name in self._tools:
            return self._tools[name]
        for source in self._dynamic_sources:
            if name in source:
                return source[name]
        return None

    def get_defer_tools(self) -> Dict[str, Tool]:
        """
        获取所有延迟工具（包括动态延迟源）。
        显式注册的延迟工具优先于动态延迟源中的同名工具。
        """
        result: Dict[str, Tool] = {}
        for source in self._dynamic_defer_sources:
            result.update(source)
        result.update(self._defer_tools)
        return result

    def get_all_tools(self) -> Dict[str, Tool]:
        """
        获取所有工具（包括普通工具、延迟工具及所有动态源中的工具）。
        优先级：显式注册 > 动态源；普通工具 > 延迟工具。
        """
        all_tools: Dict[str, Tool] = {}
        # 先收集动态源（优先级低）
        for source in self._dynamic_sources:
            all_tools.update(source)
        for source in self._dynamic_defer_sources:
            all_tools.update(source)
        # 显式注册的工具覆写（优先级高）
        all_tools.update(self._tools)
        all_tools.update(self._defer_tools)
        return all_tools


    def _parse_and_validate_tool_call(
        self, tool_call: Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]
    ) -> tuple[Tool, Dict[str, Any], str]:
        """
        解析并验证工具调用信息，返回 (tool, parameters, tool_call_id)。

        供 execute_tool / aexecute_tool 共用，消除重复代码。
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

        tool = self._resolve_tool(tool_name)
        if tool is None:
            raise ValueError(f"工具 {tool_name} 未注册")

        if not tool.validate_parameters(parameters):
            raise ValueError(f"工具 {tool_name} 参数验证失败，缺少必要参数")

        return tool, parameters, tool_call_id

    def execute_tool(self, tool_call: Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]) -> Message:
        """
        执行_tools, _defer_tools中注册的工具

        Args:
            tool_call (Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]): 工具调用信息，llm返回的工具调用格式，包含工具名称、参数等信息

        Returns:
            Message: 工具执行结果封装的Message对象
        """
        tool, parameters, tool_call_id = self._parse_and_validate_tool_call(tool_call)

        if self.interceptor:
            approved = self.interceptor.intercept(tool, parameters, tool_call_id)
            if not approved:
                return Message(role="tool", content=f"❌ 工具调用未通过用户的审批，已被用户拒绝: {tool.name}", tool_call_id=tool_call_id)

        return tool.run(parameters, tool_call_id)

    def execute_tools(self, tool_calls: List[Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]]) -> List[Message]:
        """
        批量串行执行工具调用

        Args:
            tool_calls (List[Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]]): 工具调用信息列表

        Returns:
            List[Message]: 工具执行结果封装的Message对象列表
        """
        results = []
        for tool_call in tool_calls:
            result = self.execute_tool(tool_call)
            results.append(result)
        return results

    def execute_tools_concurrently(self, tool_calls: List[Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]]) -> List[Message]:
        """
        批量并发执行工具调用（同步），结果顺序与输入顺序一致。

        Args:
            tool_calls (List[Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]]): 工具调用信息列表

        Returns:
            List[Message]: 工具执行结果封装的Message对象列表，与输入顺序一一对应
        """
        # 空列表直接返回
        if not tool_calls:
            return []

        # 按索引提交任务，避免 as_completed 打乱顺序
        future_to_index: dict[concurrent.futures.Future, int] = {}
        for i, tool_call in enumerate(tool_calls):
            ctx = contextvars.copy_context()
            future = self.executor.submit(ctx.run, self.execute_tool, tool_call)
            future_to_index[future] = i

        # 预分配结果列表
        results: list[Optional[Message]] = [None] * len(tool_calls)
        exceptions: list[tuple[int, Exception]] = []

        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                exceptions.append((idx, e))

        if exceptions:
            # 如果有工具失败，抛出聚合异常并附带上下文信息
            failed_names = [
                f"tool_calls[{i}]: {type(e).__name__}: {e}"
                for i, e in exceptions
            ]
            raise RuntimeError(
                f"{len(exceptions)}/{len(tool_calls)} 个工具执行失败:\n" +
                "\n".join(failed_names)
            )

        return results  # type: ignore[return-value]

    async def aexecute_tool(self, tool_call: Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]) -> Message:
        """
        异步执行工具，使用异步拦截器（aintercept）和工具的异步执行方法（arun）。

        Args:
            tool_call (Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]): 工具调用信息

        Returns:
            Message: 工具执行结果封装的Message对象
        """
        tool, parameters, tool_call_id = self._parse_and_validate_tool_call(tool_call)

        if self.interceptor:
            approved = await self.interceptor.aintercept(tool, parameters, tool_call_id)
            if not approved:
                return Message(role="tool", content=f"❌ 工具调用未通过用户的审批，已被用户拒绝: {tool.name}", tool_call_id=tool_call_id)

        return await tool.arun(parameters, tool_call_id)

    async def aexecute_tools(self, tool_calls: List[Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]]) -> List[Message]:
        """
        批量异步并发执行工具调用，结果顺序与输入顺序一致。
        单个工具失败不会影响其他工具的执行，失败的工具会返回包含错误信息的 Message。

        Args:
            tool_calls (List[Union[Dict[str, Any], ChatCompletionMessageFunctionToolCall]]): 工具调用信息列表

        Returns:
            List[Message]: 工具执行结果封装的Message对象列表，与输入顺序一一对应
        """
        if not tool_calls:
            return []

        tasks = [self.aexecute_tool(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        safe_results: List[Message] = []
        for tool_call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                tc = tool_call if isinstance(tool_call, dict) else tool_call.model_dump()
                tool_call_id = tc.get("id", "")
                safe_results.append(
                    Message(role="tool", content=f"❌ 工具执行失败 ({type(result).__name__}): {result}", tool_call_id=tool_call_id)
                )
            else:
                safe_results.append(result)

        return safe_results
        

    def get_openai_tools(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        获取指定工具列表，转换为OpenAI API要求的格式。
        包含显式注册的普通工具和动态普通工具源中的工具，
        显式注册的工具优先于动态源中的同名工具。

        Args:
            tool_names (List[str]): 工具名称列表，为 None 时返回所有普通工具

        Returns:
            List[Dict[str, Any]]: 工具列表
        """
        if tool_names is None:
            # 构建合并视图：动态源 + 显式注册（显式覆写动态同名工具）
            merged: Dict[str, Tool] = {}
            for source in self._dynamic_sources:
                merged.update(source)
            merged.update(self._tools)
            return [tool.to_openai_dict() for tool in merged.values()]
        else:
            results = []
            for name in tool_names:
                tool = self._resolve_tool(name)
                if tool is not None:
                    results.append(tool.to_openai_dict())
            return results

    def reset_all_tools(self) -> None:
        """重置所有已注册工具（包括动态源中的工具）到初始状态。"""
        for tool in list(self._tools.values()) + list(self._defer_tools.values()):
            tool.reset()
        for source in self._dynamic_sources:
            for tool in source.values():
                tool.reset()
        for source in self._dynamic_defer_sources:
            for tool in source.values():
                tool.reset()
