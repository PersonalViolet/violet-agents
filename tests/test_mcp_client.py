"""
测试 MCPClient —— MCP 协议客户端。

覆盖：
- _prepare_server_source（各种 server_source 格式）
- 未连接状态下的 RuntimeError
- 上下文管理器（__aenter__ / __aexit__）
- 集成测试（FastMCP 内存传输）
- transport_kwargs 传递
"""
import asyncio

import pytest

from src.violet_agents.protocols.mcp.client import (
    MCPClient,
    FASTMCP_AVAILABLE,
)

if FASTMCP_AVAILABLE:
    from fastmcp import FastMCP


# ---------------------------------------------------------------------------
# 测试用 FastMCP 服务器
# ---------------------------------------------------------------------------

def _make_test_server() -> "FastMCP":
    """创建一个带有工具、资源和提示词的 FastMCP 测试服务器。"""
    mcp = FastMCP("TestServer")

    @mcp.tool()
    def echo(message: str) -> str:
        """回显输入消息。"""
        return f"echo: {message}"

    @mcp.tool()
    def add(a: float, b: float) -> str:
        """两数之和。"""
        return str(a + b)

    @mcp.resource("test://greeting")
    def greeting_resource() -> str:
        """问候资源。"""
        return "hello from resource"

    @mcp.prompt()
    def welcome(name: str = "World") -> str:
        """欢迎提示词。"""
        return f"Welcome, {name}!"

    return mcp


# ---------------------------------------------------------------------------
# _prepare_server_source 测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="fastmcp 未安装")
class TestPrepareServerSource:
    """测试 _prepare_server_source 对各种 server_source 格式的处理。"""

    def test_fastmcp_instance(self):
        """FastMCP 实例原样返回。"""
        server = _make_test_server()
        client = MCPClient(server)
        assert client.server_source is server

    def test_http_url(self):
        """HTTP URL → StreamableHttpTransport。"""
        from fastmcp.client.transports import StreamableHttpTransport

        client = MCPClient("https://api.example.com/mcp")
        assert isinstance(client.server_source, StreamableHttpTransport)

    def test_http_url_plain_http(self):
        """普通 HTTP URL 也识别。"""
        from fastmcp.client.transports import StreamableHttpTransport

        client = MCPClient("http://localhost:8080/mcp")
        assert isinstance(client.server_source, StreamableHttpTransport)

    def test_python_script_path(self, tmp_path):
        """.py 脚本路径 → PythonStdioTransport。"""
        from fastmcp.client.transports import PythonStdioTransport

        script = tmp_path / "server.py"
        script.write_text("# test server")
        client = MCPClient(str(script), server_args=["--port", "8080"])
        assert isinstance(client.server_source, PythonStdioTransport)

    def test_command_list_python_script(self, tmp_path):
        """命令列表 [python, *.py, ...] → PythonStdioTransport。"""
        from fastmcp.client.transports import PythonStdioTransport

        script = tmp_path / "server.py"
        script.write_text("# test server")
        client = MCPClient(["python", str(script), "--debug"])
        assert isinstance(client.server_source, PythonStdioTransport)

    def test_command_list_generic(self):
        """命令列表（非 Python）→ StdioTransport。"""
        from fastmcp.client.transports import StdioTransport

        client = MCPClient(["node", "server.js"])
        assert isinstance(client.server_source, StdioTransport)

    def test_config_dict(self):
        """配置字典原样返回。"""
        config = {"url": "https://example.com", "transport": "http"}
        client = MCPClient(config)
        assert client.server_source is config

    def test_invalid_source_int_raises(self):
        """无法识别的类型（int）→ ValueError。"""
        with pytest.raises(ValueError, match="无法识别"):
            MCPClient(42)

    def test_empty_command_list_raises(self):
        """空命令列表 → ValueError（不匹配任何已知格式）。"""
        with pytest.raises(ValueError, match="无法识别"):
            MCPClient([])


# ---------------------------------------------------------------------------
# fastmcp 不可用时的错误
# ---------------------------------------------------------------------------

class TestFastmcpNotAvailable:
    """fastmcp 不可用时抛 ImportError。"""

    def test_import_error_on_init(self, monkeypatch):
        """模拟 FASTMCP_AVAILABLE = False 时初始化报错。"""
        import src.violet_agents.protocols.mcp.client as client_module

        monkeypatch.setattr(client_module, "FASTMCP_AVAILABLE", False)
        with pytest.raises(ImportError, match="fastmcp"):
            client_module.MCPClient("server.py")


# ---------------------------------------------------------------------------
# 未连接状态 —— RuntimeError
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="fastmcp 未安装")
class TestClientNotConnected:
    """未使用上下文管理器时所有方法抛 RuntimeError。"""

    @pytest.fixture
    def client(self):
        return MCPClient(_make_test_server())

    def test_list_tools_raises(self, client):
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.list_tools())

    def test_call_tool_raises(self, client):
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.call_tool("echo", {"message": "x"}))

    def test_list_resources_raises(self, client):
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.list_resources())

    def test_read_resource_raises(self, client):
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.read_resource("test://x"))

    def test_list_prompts_raises(self, client):
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.list_prompts())

    def test_get_prompt_raises(self, client):
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.get_prompt("welcome"))

    def test_ping_raises(self, client):
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(client.ping())


# ---------------------------------------------------------------------------
# 上下文管理器 连接 / 断开
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="fastmcp 未安装")
class TestContextManager:
    """测试 __aenter__ / __aexit__ 生命周期。"""

    def test_connect_and_disconnect(self):
        async def _run():
            client = MCPClient(_make_test_server())
            assert client.client is None

            await client.__aenter__()
            assert client.client is not None

            await client.__aexit__(None, None, None)
            assert client.client is not None

        asyncio.run(_run())

    def test_disconnect_on_exception(self):
        """即使上下文因异常退出，client 也为None。"""
        async def _run():
            client = MCPClient(_make_test_server())
            await client.__aenter__()
            assert client.client is not None

            exc = RuntimeError("simulated")
            try:
                raise exc
            except RuntimeError:
                await client.__aexit__(RuntimeError, exc, None)

            assert client.client is not None

        asyncio.run(_run())

    def test_disconnect_when_client_is_none(self):
        """client 为 None 时 __aexit__ 也不报错。"""
        async def _run():
            client = MCPClient(_make_test_server())
            # 未连接直接退出
            await client.__aexit__(None, None, None)
            assert client.client is None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# FastMCP 内存传输集成测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="fastmcp 未安装")
class TestMCPClientConnected:
    """使用 FastMCP 内存传输进行完整集成测试。"""

    # -- helpers --

    @staticmethod
    async def _connect():
        client = MCPClient(_make_test_server())
        await client.__aenter__()
        return client

    @staticmethod
    async def _disconnect(client):
        await client.__aexit__(None, None, None)

    # ---- list_tools ----

    def test_list_tools_returns_all(self):
        async def _run():
            client = await self._connect()
            try:
                tools = await client.list_tools()
                names = {t["name"] for t in tools}
                assert "echo" in names
                assert "add" in names
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    def test_list_tools_structure(self):
        """验证返回的工具字典结构完整。"""
        async def _run():
            client = await self._connect()
            try:
                tools = await client.list_tools()
                assert len(tools) >= 2
                for t in tools:
                    assert "name" in t
                    assert "description" in t
                    assert "input_schema" in t
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    # ---- call_tool ----

    def test_call_tool_echo(self):
        async def _run():
            client = await self._connect()
            try:
                result = await client.call_tool("echo", {"message": "hello"})
                assert "hello" in str(result)
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    def test_call_tool_add(self):
        async def _run():
            client = await self._connect()
            try:
                result = await client.call_tool("add", {"a": 3, "b": 7})
                assert "10" in str(result)
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    # ---- list_resources ----

    def test_list_resources(self):
        async def _run():
            client = await self._connect()
            try:
                resources = await client.list_resources()
                assert len(resources) >= 1
                uris = {r["uri"] for r in resources}
                assert "test://greeting" in uris
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    def test_list_resources_structure(self):
        """验证返回的资源字典结构完整。"""
        async def _run():
            client = await self._connect()
            try:
                resources = await client.list_resources()
                assert len(resources) >= 1
                for r in resources:
                    assert "uri" in r
                    assert "name" in r
                    assert "description" in r
                    assert "mime_type" in r
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    # ---- read_resource ----

    def test_read_resource(self):
        async def _run():
            client = await self._connect()
            try:
                content = await client.read_resource("test://greeting")
                assert "hello" in str(content).lower()
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    # ---- list_prompts ----

    def test_list_prompts(self):
        async def _run():
            client = await self._connect()
            try:
                prompts = await client.list_prompts()
                names = {p["name"] for p in prompts}
                assert "welcome" in names
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    def test_list_prompts_structure(self):
        """验证返回的提示词字典结构完整。"""
        async def _run():
            client = await self._connect()
            try:
                prompts = await client.list_prompts()
                assert len(prompts) >= 1
                for p in prompts:
                    assert "name" in p
                    assert "description" in p
                    assert "arguments" in p
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    # ---- get_prompt ----

    def test_get_prompt_with_args(self):
        async def _run():
            client = await self._connect()
            try:
                messages = await client.get_prompt("welcome", {"name": "Claude"})
                assert len(messages) >= 1
                assert "role" in messages[0]
                assert "content" in messages[0]
                assert "Claude" in str(messages[0]["content"])
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    def test_get_prompt_default_args(self):
        async def _run():
            client = await self._connect()
            try:
                messages = await client.get_prompt("welcome")
                assert len(messages) >= 1
                assert "World" in str(messages[0]["content"])
            finally:
                await self._disconnect(client)

        asyncio.run(_run())

    # ---- ping ----

    def test_ping_returns_true(self):
        async def _run():
            client = await self._connect()
            try:
                alive = await client.ping()
                assert alive is True
            finally:
                await self._disconnect(client)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# transport_kwargs 传递
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="fastmcp 未安装")
class TestTransportKwargs:
    """验证 transport_kwargs 正确传递给底层传输。"""

    def test_http_transport_extra_kwargs(self):
        """HTTP 传输接收 headers、sse_read_timeout 等额外参数。"""
        from fastmcp.client.transports import StreamableHttpTransport

        client = MCPClient(
            "https://api.example.com/mcp",
            headers={"Authorization": "Bearer test"},
            sse_read_timeout=30.0,
        )
        assert isinstance(client.server_source, StreamableHttpTransport)

    def test_stdio_transport_env_passed(self, tmp_path):
        """STDIO 传输接收环境变量。"""
        from fastmcp.client.transports import PythonStdioTransport

        script = tmp_path / "server.py"
        script.write_text("# test server")
        client = MCPClient(
            str(script),
            server_args=["--verbose"],
            env={"DEBUG": "1"},
        )
        assert isinstance(client.server_source, PythonStdioTransport)

    def test_command_list_with_extra_args(self, tmp_path):
        """命令列表格式 + server_args 合并。"""
        from fastmcp.client.transports import PythonStdioTransport

        script = tmp_path / "server.py"
        script.write_text("# test server")
        client = MCPClient(
            ["python", str(script), "--host", "0.0.0.0"],
            server_args=["--port", "9090"],
            env={"LOG_LEVEL": "debug"},
        )
        assert isinstance(client.server_source, PythonStdioTransport)
