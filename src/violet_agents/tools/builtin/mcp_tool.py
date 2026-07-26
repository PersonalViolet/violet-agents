from ..base import Tool, ToolParameters, ToolProperty
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING
import os
from fastmcp import FastMCP
import asyncio
from ...protocols import MCPClient
from ...core.message import Message

if TYPE_CHECKING:
    from ..registry import ToolRegistry

class MCPTool(Tool):
    """MCP (Model Context Protocol) 工具

    连接到 MCP 服务器并调用其提供的工具、资源和提示词。
    """

    def __init__(self, 
                 name: str = "mcp_tool",
                 description: str = None,
                 server_source: Union[str, List[str], FastMCP, Dict[str, Any]] = None,
                 server_args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None,
                 env_keys: Optional[List[str]] = None,
                 auto_expand: bool = False,
                 tool_time_out: int = 60):
        """
        初始化 MCP 工具

        Args:
            name: 工具名称（默认为"mcp"，建议为不同服务器指定不同名称）
            description: 工具描述（可选，默认为通用描述）
            server_source: 服务器源，支持多种格式：
                - FastMCP 实例: 内存传输（用于测试）
                - 字符串路径: Python 脚本路径（如 "server.py"）
                - HTTP URL: 远程服务器（如 "https://api.example.com/mcp"）
                - 命令列表: 完整命令（如 ["python", "server.py"]）
                - 配置字典: 传输配置，详情请参考 fastmcp 文档
            server_args: 服务器参数列表（可选）
            env: 环境变量字典（优先级最高，直接传递给MCP服务器）
            env_keys: 要从系统环境变量加载的key列表（优先级中等）
            tool_time_out: 工具调用超时时间（秒）

        环境变量优先级（从高到低）：
            1. 直接传递的env参数
            2. env_keys指定的环境变量

        示例：
            >>> # 方式1：直接传递环境变量（优先级最高）
            >>> github_tool = MCPTool(
            ...     name="github",
            ...     server_source=["npx", "-y", "@modelcontextprotocol/server-github"],
            ...     env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"}
            ... )
            >>>
            >>> # 方式2：从.env文件加载指定的环境变量
            >>> github_tool = MCPTool(
            ...     name="github",
            ...     server_source=["npx", "-y", "@modelcontextprotocol/server-github"],
            ...     env_keys=["GITHUB_PERSONAL_ACCESS_TOKEN"]
            ... )
        """
        self.server_source = server_source
        self.server_args = server_args
        self.env = self._prepare_env(env=env, env_keys=env_keys)
        self.auto_expand = auto_expand
        self.tool_time_out = tool_time_out
        self.client = MCPClient(server_source=self.server_source, server_args=self.server_args, env=self.env)
        self._expanded_tools_cache: Optional[Dict[str, Tool]] = None

        self._discover_tools()

        self.description = description or self._generate_description()


        super().__init__(name=name, description=description)

    def _prepare_env(self, env: Optional[Dict[str, str]] = None, env_keys: Optional[List[str]] = None) -> Dict[str, str]:
        """
        准备环境变量字典

        Args:
            env: 直接传递的环境变量字典（优先级最高）
            env_keys: 要从系统环境变量加载的key列表（优先级中等）
        
        Returns:
            result_env: 准备好的环境变量字典
        """
        result_env = {}
        # 1. env_keys指定的环境变量
        if env_keys:
            for key in env_keys:
                value = os.getenv(key)
                if value:
                    result_env[key] = value
                    print(f"从系统环境变量加载: {key}")
                else:
                    print(f"警告: 系统环境变量中未找到 {key}")

        # 2. 直接传递的env（优先级最高）
        if env:
            result_env.update(env)
            for key in env.keys():
                print(f"直接传递环境变量: {key}")

        return result_env

    def _discover_tools(self):
        """
        发现 MCP 服务器提供的工具
        """
        try:

            async def discover():
                async with MCPClient(server_source=self.server_source, server_args=self.server_args, env=self.env) as client:
                    tools = await client.list_tools()
                    return tools

            try:
                loop = asyncio.get_running_loop()
                # 已有运行的事件循环，在新线程中运行 discover
                import concurrent.futures
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(discover())
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    self._available_tools = future.result()
            except RuntimeError:
                # 没有运行的事件循环
                self._available_tools = asyncio.run(discover())
            
        except Exception as e:
            print(f"发现工具时出错: {e}")
            return []


    def _generate_description(self) -> str:
        """
        生成工具描述
        """
        servers = MCPClient(server_source=self.server_source, server_args=self.server_args, env=self.env).get_server_names()
        server_count = len(servers)
        server_list = ", ".join(sorted(servers))

        if not self._available_tools:
            return (
                f"=== MCP 服务器连接信息 ===\n"
                f"已连接 {server_count} 个服务器: {server_list}\n"
                f"未发现可用工具。"
            )

        if self.auto_expand:
            lines = [
                f"=== MCP 服务器连接信息 ===",
                f"已连接 {server_count} 个服务器: {server_list}",
                f"",
                f"可用工具 ({len(self._available_tools)} 个):",
                f"如果你想获取工具信息，只有在尝试完其它获取工具的途径而无果时才使用该工具提供的list_tools操作。"
            ]
            # for tool in self._available_tools:
            #     name = getattr(tool, 'name', str(tool))
            #     desc = getattr(tool, 'description', '')
            #     if desc:
            #         lines.append(f"  • {name}: {desc}")
            #     else:
            #         lines.append(f"  • {name}")
            return "\n".join(lines)
        else:
            # 如果不自动展开工具列表，则在提示词中显示MCP服务器连接情况以及每个工具如何调用
            tool_names = []
            for tool in self._available_tools:
                name = getattr(tool, 'name', str(tool))
                tool_names.append(name)

            lines = [
                f"=== MCP 服务器连接信息 ===",
                f"已连接 {server_count} 个服务器: {server_list}",
                f"",
                f"可用工具 ({len(self._available_tools)} 个):",
                f"  {', '.join(sorted(tool_names))}",
                f"",
                f"使用方式：",
                f"  1. 查看工具列表: action='list_tools'",
                f"  2. 调用指定工具: action='call_tool', tool_name='<工具名>', arguments={{...}}",
                f"  3. 查看资源列表: action='list_resources'",
                f"  4. 读取指定资源: action='read_resource', uri='<资源URI>'",
                f"  5. 查看提示词列表: action='list_prompts'",
                f"  6. 获取指定提示词: action='get_prompt', prompt_name='<提示词名>', prompt_arguments={{...}}",
            ]
            return "\n".join(lines)

    def get_expanded_tools(self) -> Dict[str, Tool]:
        """
        获取 MCP 服务器提供的工具列表。

        首次调用时构建并缓存，后续调用返回同一字典引用。
        这确保 register_dynamic_tools 持有的引用始终是最新的。
        """
        if not self.auto_expand:
            return {}
        if self._expanded_tools_cache is not None:
            return self._expanded_tools_cache
        from .mcp_wrapped_tool import MCPWrappedTool
        self._expanded_tools_cache = {}
        for tool_info in self._available_tools:
            wrapped_tool = MCPWrappedTool(
                mcp_tool=self,
                tool_info=tool_info
            )
            self._expanded_tools_cache[wrapped_tool.name] = wrapped_tool
        return self._expanded_tools_cache

    def register_to(
        self,
        registry: "ToolRegistry",
        self_defer: bool = False,
        expanded_defer: bool = True,
    ) -> "MCPTool":
        """
        将 MCPTool 及其展开的子工具注册到指定的 ToolRegistry。

        Args:
            registry: 目标工具注册表。
            self_defer: MCPTool 自身是否注册到延迟工具区。默认 False（直接暴露给 LLM，）。
            expanded_defer: 展开的子工具是否注册到延迟工具区。默认 True（不直接暴露给 LLM）。

        Returns:
            self，支持链式调用。

        Example:
            >>> config = {"github": {"command": "npx", "args": [...]}}
            >>> # MCPTool 自身直接暴露给 LLM，展开的子工具放到延迟工具区
            >>> MCPTool(server_source=config, auto_expand=True).register_to(registry)
            >>>
            >>> # 全部放延迟区
            >>> MCPTool(server_source=config, auto_expand=True).register_to(
            ...     registry, self_defer=True, expanded_defer=True)
        """
        registry.register_tool(self, is_defer=self_defer)
        expanded = self.get_expanded_tools()
        if expanded:
            registry.register_dynamic_tools(expanded, is_defer=expanded_defer)
        return self

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        action = parameters.get("action", "")
        if not action and "tool_name" in parameters:
            action = "call_tool"
            parameters["action"] = action

        if not action:
            return Message(role="tool", content="❌ 未提供操作类型 (action)", tool_call_id=tool_call_id)

        try:
            import asyncio
            async def run_mcp_operation():
                async with MCPClient(server_source=self.server_source, server_args=self.server_args, env=self.env) as client:
                    if action == "list_tools":
                        tools = await client.list_tools()
                        if not tools:
                            return Message(role="tool", content="❌ 未发现可用工具", tool_call_id=tool_call_id)
                        import json
                        tool_schemas = []
                        for tool in tools:
                            name = tool.get('name', 'Unknown')
                            desc = tool.get('description', '')
                            input_schema = tool.get('input_schema', {})
                            properties = input_schema.get('properties', {})
                            required = input_schema.get('required', [])
                            tool_schemas.append({
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "description": desc,
                                    "parameters": {
                                        "type": "object",
                                        "properties": properties,
                                        "required": required,
                                    }
                                }
                            })
                        result = f"找到 {len(tools)} 个工具 (OpenAI function calling 格式):\n"
                        result += json.dumps(tool_schemas, ensure_ascii=False, indent=2)
                        return Message(role="tool", content=result, tool_call_id=tool_call_id)
                    elif action == "call_tool":
                        tool_name = parameters.get("tool_name")
                        arguments = parameters.get("arguments", {})
                        if not tool_name:
                            return Message(role="tool", content="❌ 未提供工具名称 (tool_name)", tool_call_id=tool_call_id)
                        result = await client.call_tool(tool_name, arguments, timeout=self.tool_time_out)
                        return Message(role="tool", content=result, tool_call_id=tool_call_id)
                    elif action == "list_resources":
                        resources = await client.list_resources()
                        if not resources:
                            return Message(role="tool", content="❌ 未发现可用资源", tool_call_id=tool_call_id)
                        result = f"找到{len(resources)} 个资源:\n"
                        for resource in resources:
                            result += f"- {resource.get('uri', 'Unknown')}: {resource.get('name', '')}\n"
                        return Message(role="tool", content=result, tool_call_id=tool_call_id)
                    elif action == "read_resource":
                        uri = parameters.get("uri")
                        if not uri:
                            return Message(role="tool", content="❌ 未提供资源 URI (uri)", tool_call_id=tool_call_id)
                        resource_content = await client.read_resource(uri)
                        return Message(role="tool", content=resource_content, tool_call_id=tool_call_id)
                    elif action == "list_prompts":
                        prompts = await client.list_prompts()
                        if not prompts:
                            return Message(role="tool", content="❌ 未发现可用提示词", tool_call_id=tool_call_id)
                        result = f"找到{len(prompts)} 个提示词:\n"
                        for prompt in prompts:
                            result += f"- {prompt.get('name', 'Unknown')}: {prompt.get('description', '')}\n"
                        return Message(role="tool", content=result, tool_call_id=tool_call_id)
                    elif action == "get_prompt":
                        prompt_name = parameters.get("prompt_name")
                        prompt_arguments = parameters.get("prompt_arguments", {})
                        if not prompt_name:
                            return Message(role="tool", content="❌ 未提供提示词名称 (prompt_name)", tool_call_id=tool_call_id)
                        messages = await client.get_prompt(prompt_name, prompt_arguments)
                        result = f"提示词 '{prompt_name}':\n"
                        for msg in messages:
                            result += f"[{msg.get('role', 'unknown')}] {msg.get('content', '')}\n"
                        return Message(role="tool", content=result, tool_call_id=tool_call_id)
                    else:
                        return Message(role="tool", content=f"❌ 未知操作类型: {action}", tool_call_id=tool_call_id)

            try:
                # 检查是否已有运行中的事件循环
                try:
                    loop = asyncio.get_running_loop()
                    # 如果有运行中的循环，在新线程中运行新的事件循环
                    import concurrent.futures

                    def run_in_thread():
                        # 在新线程中创建新的事件循环
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(run_mcp_operation())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        return future.result()
                except RuntimeError:
                    # 没有运行中的循环，直接运行
                    try:
                        return asyncio.run(run_mcp_operation())
                    except asyncio.CancelledError:
                        return Message(role="tool", content="❌ 操作被取消，可能因为超时或断开连接", tool_call_id=tool_call_id)
            except Exception as e:
                msg = f"异步操作失败: {str(e)}"
                return Message(role="tool", content=msg, tool_call_id=tool_call_id)
        except Exception as e:
            return Message(role="tool", content=f"❌ 运行时错误: {e}", tool_call_id=tool_call_id)
    
    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "action": ToolProperty(
                    type="string",
                    description="操作类型: list_tools, call_tool, list_resources, read_resource, list_prompts, get_prompt"
                ),
                "tool_name": ToolProperty(
                    type="string",
                    description="工具名称（call_tool 操作需要）"
                ),
                "arguments": ToolProperty(
                    type="object",
                    description="工具参数（call_tool 操作需要）"
                ),
                "uri": ToolProperty(
                    type="string",
                    description="资源 URI（read_resource 操作需要）"
                ),
                "prompt_name": ToolProperty(
                    type="string",
                    description="提示词名称（get_prompt 操作需要）"
                ),
                "prompt_arguments": ToolProperty(
                    type="object",
                    description="提示词参数（get_prompt 操作可选）"
                ),
            },
            required=["action"]
        )

        