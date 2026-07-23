from typing import Dict, Any, List, Optional, Union, Literal

try:
    from fastmcp import Client, FastMCP
    from fastmcp.client.transports import PythonStdioTransport, SSETransport, StreamableHttpTransport
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

SUPPORTED_TRANSPORTS = Literal["stdio", "http"]  # 支持的传输类型列表

class MCPClient(Client):

    def __init__(self,
                 server_source: Union[str, List[str], FastMCP, Dict[str, Any]],
                 server_args: Optional[List[str]] = None,
                 env: Optional[Dict[str, Any]] = None,
                 **transport_kwargs):
        """
        初始化MCP 客户端

        Args:
            server_source: 服务器源，支持多种格式：
                - FastMCP 实例: 内存传输（用于测试）
                - 字符串路径: Python 脚本路径（如 "server.py"）
                - HTTP URL: 远程服务器（如 "https://api.example.com/mcp"）
                - 命令列表: 完整命令（如 ["python", "server.py"]）
                - 配置字典: 传输配置，详情请参考 fastmcp 文档
            server_args: 服务器参数列表（可选）
            env: 环境变量字典（传递给MCP服务器进程）
            **transport_kwargs: 传输特定的额外参数

            
        Raises:
            ImportError: 如果 fastmcp 库未安装
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError("fastmcp 库未安装，请使用 'pip install fastmcp' 安装。")
        self.server_args = server_args or []
        self.env = env or {}
        self.transport_kwargs = transport_kwargs
        self.server_source = self._prepare_server_source(server_source)
        self.client: Optional[Client] = None
 
    def _prepare_server_source(self, server_source: Union[str, List[str], FastMCP, Dict[str, Any]]):
        """
        准备服务器源，确保其格式正确。

        Args:
            server_source: 服务器源，支持多种格式

        Returns:
            处理后的服务器源
        """
        # 1. FastMCP 实例（内存传输）
        if isinstance(server_source, FastMCP):
            print("使用 FastMCP 实例（内存传输）进行通信。")
            return server_source
        
        # 2. HTTP URL - HTTP 传输（SSE已不推荐）
        if isinstance(server_source, str) and (server_source.startswith("http://") or server_source.startswith("https://")):
            transport_type = "http"
            print(f"使用 HTTP URL（{transport_type}）进行通信。")
            if transport_type == "sse":
                raise ValueError("SSE 传输已不推荐，请使用 HTTP 传输。")
            elif transport_type == "http":
                return StreamableHttpTransport(url=server_source, **self.transport_kwargs)

        # 3. Python 脚本路径 - STDIO 传输
        if isinstance(server_source, str) and server_source.endswith(".py"):
            transport_type = "stdio"
            print(f"使用 Python 脚本（{transport_type}）进行通信。")
            return PythonStdioTransport(script_path=server_source,
                                        args=self.server_args,
                                        env=self.env,
                                        **self.transport_kwargs)

        # 4. 命令列表 - STDIO 传输
        if isinstance(server_source, list) and len(server_source) > 0:
            print(f"使用命令列表进行通信。传输类型：STDIO")
            if server_source[0] == "python" and len(server_source) > 1 and server_source[1].endswith(".py"):
                # Python 脚本
                return PythonStdioTransport(
                    script_path=server_source[1],
                    args=server_source[2:] + self.server_args,
                    env=self.env if self.env else None,
                    **self.transport_kwargs
                )
            else:
                # 其他命令，使用通用 Stdio 传输
                from fastmcp.client.transports import StdioTransport
                return StdioTransport(
                    command=server_source[0],
                    args=server_source[1:] + self.server_args,
                    env=self.env if self.env else None,
                    **self.transport_kwargs
                )

        # 5. 配置字典 - 根据配置创建传输
        if isinstance(server_source, dict):
            print("使用配置字典，请确保格式符合 fastmcp 文档要求。")
            return server_source  # 假设字典已经是有效的传输配置

        raise ValueError("无法识别的服务器源类型。请提供 FastMCP 实例、HTTP URL、Python 脚本路径、命令列表或配置字典。")

        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        print("🔗 连接到 MCP 服务器...")
        self.client = Client(self.server_source)
        await self.client.__aenter__()
        print("✅ 连接成功！")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
            self.client = None
        print("🔌 连接已断开")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用的工具"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_tools()

        # 处理不同的返回格式
        if hasattr(result, 'tools'):
            tools = result.tools
        elif isinstance(result, list):
            tools = result
        else:
            tools = []

        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            }
            for tool in tools
        ]

    async def call_tool(self, 
                        tool_name: str, 
                        arguments: Dict[str, Any],
                        timeout: Optional[float] = None) -> Any:
        """调用 MCP 工具"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.call_tool(tool_name, arguments, timeout=timeout)

        # 解析结果 - FastMCP 返回 ToolResult 对象
        if hasattr(result, 'content') and result.content:
            if len(result.content) == 1:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'data'):
                    return content.data
            return [
                getattr(c, 'text', getattr(c, 'data', str(c)))
                for c in result.content
            ]
        return None

    async def list_resources(self) -> List[Dict[str, Any]]:
        """列出所有可用的资源"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_resources()
        resources = result if isinstance(result, list) else getattr(result, 'resources', [])
        return [
            {
                "uri": str(resource.uri),
                "name": resource.name or "",
                "description": resource.description or "",
                "mime_type": getattr(resource, 'mimeType', None)
            }
            for resource in resources
        ]

    async def read_resource(self, uri: str) -> Any:
        """读取资源内容"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.read_resource(uri)

        # 处理不同的返回格式（list 或带 .contents 的对象）
        contents = result if isinstance(result, list) else getattr(result, 'contents', [])
        if contents:
            if len(contents) == 1:
                content = contents[0]
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'blob'):
                    return content.blob
            return [
                getattr(c, 'text', getattr(c, 'blob', str(c)))
                for c in contents
            ]
        return None

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """列出所有可用的提示词模板"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_prompts()
        prompts = result if isinstance(result, list) else getattr(result, 'prompts', [])
        return [
            {
                "name": prompt.name,
                "description": prompt.description or "",
                "arguments": getattr(prompt, 'arguments', [])
            }
            for prompt in prompts
        ]

    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """获取提示词内容"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.get_prompt(prompt_name, arguments or {})

        # 解析提示词消息
        if hasattr(result, 'messages') and result.messages:
            return [
                {
                    "role": msg.role,
                    "content": getattr(msg.content, 'text', str(msg.content)) if hasattr(msg.content, 'text') else str(msg.content)
                }
                for msg in result.messages
            ]
        return []

    async def ping(self) -> bool:
        """测试服务器连接"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")
        
        try:
            await self.client.ping()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    import asyncio

    async def main():
        # 示例：使用 github 上的 MCP 服务器脚本进行测试
        server_source = "https://api.githubcopilot.com/mcp/"
        async with MCPClient(server_source=server_source, headers={"Authorization": "Bearer your_token"}) as client:
            tools = await client.list_tools()
            print("可用工具:", tools)

            resources = await client.list_resources()
            print("可用资源:", resources)

            prompts = await client.list_prompts()
            print("可用提示词模板:", prompts)

            is_alive = await client.ping()
            print("服务器连接状态:", "在线" if is_alive else "离线")

    asyncio.run(main())